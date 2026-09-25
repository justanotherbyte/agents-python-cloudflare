from __future__ import annotations

import asyncio
import copy
import inspect
import json
import math
from collections.abc import Callable, Iterable
from contextlib import suppress
from contextvars import ContextVar
from typing import Any, cast

from js import Object, ReadableStream, TextEncoder  # ty: ignore[unresolved-import]
from pyodide.ffi import create_proxy, to_js
from workers import Response

from ..core.agent import Agent
from ..core.utils import (
    dumps_wire,
    error_message,
    gen_id,
    loads_dict_or_none,
    now_ms,
    url_path,
)
from ..lifecycle._job_driver import _is_memory_limit_reset, _is_platform_failure
from ..lifecycle.websockets import Connection, ConnectionContext
from ..sessions import Session, SessionChangeEvent, Sessions
from ..tasks import TaskStep, TaskStepConfig, TaskStepRetryOptions
from .agent_tools import ChildAgentToolRuns
from .continuation import AutoContinuationController, ContinuationRequest
from .folding import (
    TOOL_RESOLVED_STATES,
    MessageAccumulator,
    _find_last_part,
    _find_tool_part,
    apply_chunk_to_parts,  # noqa: F401 - re-exported from agents
)
from .messages import (
    _assistant_content_key,
    _legacy_created_at_ms,
    _sanitize_message,
    _transform_message,
    _valid_message,
)
from .normalize import _TEXT_ID, _normalize
from .pre_stream import PreStreamTurns
from .protocol import (
    ChatMessageType,
    chat_clear_frame,
    chat_messages_frame,
    chat_response,
    message_updated_frame,
    stream_resume_none_frame,
    stream_resuming_frame,
)
from .resumable_stream import ResumableStream, StreamStatus, StreamStorageUnavailable
from .recovery import (
    CHAT_RECOVERING_KEY,
    CHAT_RECOVERING_FLAG_TTL_MS,
    CHAT_RECOVERY_INCIDENT_TTL_MS,
    CHAT_RECOVERY_INCIDENT_KEY_PREFIX,
    CHAT_RECOVERY_PROGRESS_KEY,
    RecoveryPolicy,
    evaluate_incident,
    incident_id,
    incident_key,
)
from .turn_queue import TurnContext, TurnQueue
from .types import ChatOptions, ChatReplyT, ChunkT, MessageT

_PART_TYPES = ("text", "reasoning")

# A tool part that already carries an output. A second, differing result for the same
# call is a duplicate rather than a correction, so the first one stands.
# What each kind of inbound answer may act on. The result set deliberately includes the
# resolved states, so a duplicate is recognised as one and dropped quietly instead of
# being reported as an answer to a call that could not be found.
_TOOL_APPROVAL_FROM = ("input-available", "approval-requested")
_TOOL_RESULT_FROM = (
    "input-available",
    "approval-requested",
    "approval-responded",
) + TOOL_RESOLVED_STATES

# How long an answer waits for the message holding its tool call to show up. Bounded so
# an answer naming a call that never existed cannot hold a connection open indefinitely.
_TOOL_PART_ATTEMPTS = 10
_TOOL_PART_RETRY_S = 0.1

_CHAT_TURN_TASK = "__cf_internal_chat_turn"
_CHAT_RECOVERY_TASK = "__cf_internal_chat_recovery"
_CHAT_TASK_RUN_PREFIX = "chat_"
_CHAT_RECOVERY_VERSION = 1
_LEGACY_LIFT_WINDOW_ROWS = 25
_LEGACY_LIFT_WINDOW_BYTES = 4 * 1024 * 1024
_TOOL_INTERACTION_CONNECTION: ContextVar[Connection | None] = ContextVar(
    "chat_tool_interaction_connection",
    default=None,
)


class AIChatAgent(Agent):
    # Storage only: this caps rows kept in SQLite and does not affect what the model
    # is sent. None means unlimited.
    max_persisted_messages: int | None = None
    durable_chat_recovery = False
    chat_recovery_max_chunks = 10_000
    chat_recovery_max_bytes = 16 * 1024 * 1024
    chat_recovery_max_attempts = 10
    chat_recovery_no_progress_timeout_ms = 5 * 60 * 1000
    chat_recovery_max_work = 1_000
    chat_recovery_max_oom_retries = 3
    chat_recovery_retry_delay_seconds = 3
    chat_recovery_stable_timeout_ms = 10_000
    chat_recovery_acceptance_attempts = 3
    chat_provider_stall_timeout_ms: int | None = None
    chat_recovery_terminal_message = (
        "The assistant was interrupted and could not recover. Please try again."
    )
    hydration_byte_budget: int | float = 32 * 1024 * 1024

    def __init__(self, ctx, env):
        super().__init__(ctx, env)

        self._aborts: dict[str, asyncio.Event] = {}
        self._request_task_runs: dict[str, str] = {}
        self._live_chat_turns: dict[str, Callable[[], Any]] = {}
        self._turn_queue = TurnQueue()

        # Held out of the live broadcast until they ACK, so the buffer replay and the
        # live tail reach them in order with no gap and no duplicate.
        self._pending_resume_connections: set[str] = set()
        self._pending_resume_requests: dict[str, str] = {}
        self._pre_stream = PreStreamTurns()

        # The assistant message currently being built, or None between turns. A tool
        # result or approval can arrive while the turn is still streaming, and until the
        # turn ends this is the only place that message exists.
        self._streaming_message: MessageT | None = None
        self._interaction_lock = asyncio.Lock()
        self._pending_interactions = 0
        self._interactions_idle = asyncio.Event()
        self._interactions_idle.set()
        self._migration_lock = asyncio.Lock()
        self.messages: list[MessageT] = []
        self._messages_truncated = False
        self._legacy_migration_incomplete = False
        self._chat_startup_complete = False

        self.sessions = Sessions()
        self._lifecycle.use(self.sessions)
        self._session: Session = self.sessions.session()
        self.sessions.subscribe(self._on_session_change)

        self._resumable = ResumableStream(self.sql, ChatMessageType.USE_CHAT_RESPONSE)
        self._child_agent_tool_runs = ChildAgentToolRuns(
            self.sql,
            self._resumable,
            self._turn_queue,
            self.prepare_agent_tool_turn,
            self.run_agent_tool_turn,
            self.collect_agent_tool_result,
        )
        self.tasks._register_reserved_definition(
            _CHAT_TURN_TASK,
            self._chat_turn_task,
        )
        self.tasks._register_reserved_definition(
            _CHAT_RECOVERY_TASK,
            self._chat_recovery_task,
        )
        self._continuation = AutoContinuationController(
            generate_request_id=gen_id,
            is_stream_active=lambda: self._streaming_message is not None,
            has_pending_interaction=lambda: self._pending_interactions > 0,
            has_incomplete_tool_batch=self._has_incomplete_tool_batch,
            drain_interactions=self._drain_interactions,
            fire=self._fire_auto_continuation,
            spawn=self._spawn_continuation,
        )
        self._last_body: dict[str, Any] = {}
        self._last_client_tools: list[Any] | None = None
        self._last_progress_bump_at = 0
        self._terminal_request_ids: set[str] = set()
        self._continuation_request_ids: set[str] = set()

    # -- storage ---------------------------------------------------------

    async def _prepare_for_startup(self, _sql: object) -> None:
        await super()._prepare_for_startup(_sql)
        self._prepare_chat_storage()

    def _prepare_chat_storage(self) -> None:
        self._resumable.prepare()
        self._child_agent_tool_runs.prepare()
        self.sql("""
        CREATE TABLE IF NOT EXISTS cf_ai_chat_request_context (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """)
        self._restore_request_context()

    def _restore_request_context(self) -> None:
        for row in self.sql("SELECT key, value FROM cf_ai_chat_request_context"):
            try:
                value = json.loads(row["value"])
            except (TypeError, ValueError):
                continue
            if row["key"] == "lastBody" and isinstance(value, dict):
                self._last_body = value
            elif row["key"] == "lastClientTools" and isinstance(value, list):
                self._last_client_tools = value

    def _persist_request_context(self) -> None:
        self.sql(
            "INSERT OR REPLACE INTO cf_ai_chat_request_context (key, value) "
            "VALUES ('lastBody', ?)",
            dumps_wire(self._last_body),
        )
        if self._last_client_tools is None:
            self.sql(
                "DELETE FROM cf_ai_chat_request_context WHERE key = 'lastClientTools'"
            )
        else:
            self.sql(
                "INSERT OR REPLACE INTO cf_ai_chat_request_context (key, value) "
                "VALUES ('lastClientTools', ?)",
                dumps_wire(self._last_client_tools),
            )

    def _clear_request_context(self) -> None:
        self._last_body = {}
        self._last_client_tools = None
        self.sql("DELETE FROM cf_ai_chat_request_context")

    async def _lifecycle_host_start(self) -> None:
        await self._migrate_legacy_messages()
        await self._hydrate_messages()
        self._chat_startup_complete = True
        await super()._lifecycle_host_start()

    async def _migrate_legacy_messages(self) -> bool:
        present = self.sql(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name = 'cf_ai_chat_agent_messages'"
        )
        if not present:
            self._legacy_migration_incomplete = False
            return True

        try:
            order = self.sql(
                "SELECT id, LENGTH(CAST(message AS BLOB)) AS bytes, created_at "
                "FROM cf_ai_chat_agent_messages "
                "ORDER BY created_at ASC, rowid ASC"
            )
        except Exception as error:  # noqa: BLE001
            print(f"[AIChatAgent] failed to inspect legacy chat storage: {error}")
            self._legacy_migration_incomplete = True
            return False
        parent_id: str | None = None
        imported = 0
        skipped = 0
        index = 0
        stopped = False
        while index < len(order) and not stopped:
            window = []
            window_bytes = 0
            while index < len(order) and len(window) < _LEGACY_LIFT_WINDOW_ROWS:
                row = order[index]
                row_bytes = row.get("bytes")
                row_bytes = (
                    row_bytes if type(row_bytes) is int and row_bytes >= 0 else 0
                )
                if window and window_bytes + row_bytes > _LEGACY_LIFT_WINDOW_BYTES:
                    break
                window.append(row)
                window_bytes += row_bytes
                index += 1

            placeholders = ",".join("?" for _ in window)
            ids = [row["id"] for row in window]
            try:
                fetched = self.sql(
                    "SELECT id, message FROM cf_ai_chat_agent_messages "
                    f"WHERE id IN ({placeholders})",
                    *ids,
                )
            except Exception as error:  # noqa: BLE001
                print(f"[AIChatAgent] failed to read legacy chat storage: {error}")
                skipped += 1
                break
            bodies = {row["id"]: row["message"] for row in fetched}
            for row in window:
                body = bodies.get(row["id"])
                if not isinstance(body, str):
                    skipped += 1
                    stopped = True
                    break
                try:
                    message = _transform_message(json.loads(body))
                    if message is None:
                        raise ValueError("unsupported message structure")
                    created_at_ms = _legacy_created_at_ms(
                        row.get("created_at"), imported
                    )
                    await self._session.import_message(
                        message,
                        parent_id=parent_id,
                        created_at_ms=created_at_ms,
                    )
                    stored = await self._session.get_message(message["id"])
                    destination = self.sql(
                        "SELECT parent_id, created_at "
                        "FROM cf_agents_session_messages "
                        "WHERE session_id = '' AND id = ?",
                        message["id"],
                    )
                    if (
                        stored != message
                        or len(destination) != 1
                        or destination[0].get("parent_id") != parent_id
                        or destination[0].get("created_at") != created_at_ms
                    ):
                        raise ValueError("destination verification failed")
                except Exception as error:  # noqa: BLE001
                    print(
                        f"[AIChatAgent] failed to migrate message {row['id']}: {error}"
                    )
                    skipped += 1
                    stopped = True
                    break
                parent_id = message["id"]
                imported += 1

        if skipped == 0 and imported == len(order):
            try:
                self.sql("DROP TABLE cf_ai_chat_agent_messages")
            except Exception as error:  # noqa: BLE001
                print(f"[AIChatAgent] failed to finalize legacy chat lift: {error}")
                self._legacy_migration_incomplete = True
                return False
            self._legacy_migration_incomplete = False
            return True
        print(
            "[AIChatAgent] legacy lift imported "
            f"{imported} of {len(order)} rows ({skipped} skipped); source retained"
        )
        self._legacy_migration_incomplete = True
        return False

    async def _ensure_chat_storage_ready(self) -> None:
        await self._ensure_initialized()
        if not self._legacy_migration_incomplete:
            return
        async with self._migration_lock:
            if (
                self._legacy_migration_incomplete
                and await self._migrate_legacy_messages()
            ):
                await self._hydrate_messages()
        if self._legacy_migration_incomplete:
            raise RuntimeError("legacy chat migration incomplete")

    async def _hydrate_messages(self) -> None:
        budget = self.hydration_byte_budget
        if type(budget) in (int, float) and math.isfinite(budget) and budget > 0:
            recent = await self._session.get_recent_history(budget)
            stored = recent.messages
            self._messages_truncated = recent.truncated
        else:
            stored = await self._session.get_history()
            self._messages_truncated = False
        self.messages = [
            message
            for index, stored_message in enumerate(stored)
            if (message := _transform_message(stored_message, index)) is not None
        ]

    def _on_session_change(self, event: SessionChangeEvent) -> None:
        if event["sessionId"] != self._session.session_id:
            return
        if event["type"] == "append":
            if not event["inserted"]:
                return
            message = _transform_message(event["message"])
            if message is None:
                return
            for index, existing in enumerate(self.messages):
                if existing["id"] == message["id"]:
                    self.messages[index] = message
                    return
            self.messages.append(message)
        elif event["type"] == "update":
            message = _transform_message(event["message"])
            if message is None:
                return
            for index, existing in enumerate(self.messages):
                if existing["id"] == message["id"]:
                    self.messages[index] = message
                    return
        elif event["type"] == "delete":
            deleted = set(event["messageIds"])
            self.messages = [
                message for message in self.messages if message["id"] not in deleted
            ]
        elif event["type"] == "clear":
            self.messages = []
            self._messages_truncated = False

    async def _reconciliation_prior(
        self,
        incoming: list[MessageT],
    ) -> list[MessageT]:
        if not self._messages_truncated:
            return [
                _sanitize_message(message)
                for index, value in enumerate(self.messages)
                if (message := _transform_message(value, index)) is not None
            ]

        incoming_ids = {message["id"] for message in incoming}
        local = {
            message["id"]: _sanitize_message(message)
            for index, value in enumerate(self.messages)
            if (message := _transform_message(value, index)) is not None
        }
        prior = []
        for message_id in incoming_ids:
            message = local.get(message_id)
            if message is None:
                stored = await self._session.get_message(message_id)
                message = None if stored is None else _transform_message(stored)
                if message is not None:
                    message = _sanitize_message(message)
            if message is not None:
                prior.append(message)

        prior_ids = {message["id"] for message in prior}
        if all(
            message["role"] != "assistant" or message["id"] in prior_ids
            for message in incoming
        ):
            return prior

        incoming_tool_ids = {
            tool_call_id
            for message in incoming
            if message["id"] not in prior_ids
            if message["role"] == "assistant"
            for part in message["parts"]
            if isinstance((tool_call_id := part.get("toolCallId")), str)
        }
        incoming_content_needs: dict[str, int] = {}
        for message in incoming:
            if message["id"] in prior_ids:
                continue
            content_key = _assistant_content_key(message)
            if content_key is not None:
                incoming_content_needs[content_key] = (
                    incoming_content_needs.get(content_key, 0) + 1
                )

        if not incoming_tool_ids and not incoming_content_needs:
            return prior

        candidate_ids = set(prior_ids)
        async for batch in self._session.history_batches():
            for index, value in enumerate(batch):
                message = _transform_message(value, index)
                if message is None or message["id"] in candidate_ids:
                    continue
                message = _sanitize_message(message)
                tool_ids = {
                    tool_call_id
                    for part in message["parts"]
                    if isinstance((tool_call_id := part.get("toolCallId")), str)
                }
                matching_tool_ids = tool_ids & incoming_tool_ids
                content_key = _assistant_content_key(message)
                matches_content = (
                    content_key is not None
                    and incoming_content_needs.get(content_key, 0) > 0
                )
                if not matching_tool_ids and not matches_content:
                    continue
                prior.append(message)
                candidate_ids.add(message["id"])
                incoming_tool_ids.difference_update(matching_tool_ids)
                if matches_content and content_key is not None:
                    remaining = incoming_content_needs[content_key] - 1
                    if remaining:
                        incoming_content_needs[content_key] = remaining
                    else:
                        incoming_content_needs.pop(content_key)
                if not incoming_tool_ids and not incoming_content_needs:
                    return prior
        return prior

    @staticmethod
    def _reconcile_messages(
        incoming: list[MessageT],
        prior: list[MessageT],
    ) -> list[MessageT]:
        resolved_parts: dict[str, ChunkT] = {}
        tool_messages: dict[str, MessageT] = {}
        for message in prior:
            if message["role"] != "assistant":
                continue
            for part in message["parts"]:
                tool_call_id = part.get("toolCallId")
                if not isinstance(tool_call_id, str):
                    continue
                tool_messages.setdefault(tool_call_id, message)
                if part.get("state") in TOOL_RESOLVED_STATES:
                    resolved_parts[tool_call_id] = part

        merged_outputs = []
        for message in incoming:
            if message["role"] != "assistant":
                merged_outputs.append(message)
                continue
            changed = False
            parts = []
            for part in message["parts"]:
                tool_call_id = part.get("toolCallId")
                server_part = (
                    resolved_parts.get(tool_call_id)
                    if isinstance(tool_call_id, str)
                    else None
                )
                if server_part is None or part.get("state") not in (
                    "input-available",
                    "approval-requested",
                    "approval-responded",
                ):
                    parts.append(part)
                    continue
                updated = {**part, "state": server_part["state"]}
                result_key = {
                    "output-available": "output",
                    "output-error": "errorText",
                    "output-denied": "approval",
                }.get(server_part["state"])
                if result_key is not None and result_key in server_part:
                    updated[result_key] = server_part[result_key]
                parts.append(updated)
                changed = True
            merged_outputs.append({**message, "parts": parts} if changed else message)

        claimed: set[int] = set()
        exact_matches: set[int] = set()
        for incoming_index, message in enumerate(merged_outputs):
            for server_index, server_message in enumerate(prior):
                if server_index in claimed or server_message["id"] != message["id"]:
                    continue
                claimed.add(server_index)
                exact_matches.add(incoming_index)
                break

        reconciled = []
        for incoming_index, message in enumerate(merged_outputs):
            content_key = _assistant_content_key(message)
            if (
                incoming_index in exact_matches
                or message["role"] != "assistant"
                or content_key is None
            ):
                reconciled.append(message)
                continue
            match = next(
                (
                    (server_index, server_message)
                    for server_index, server_message in enumerate(prior)
                    if server_index not in claimed
                    and server_message["role"] == "assistant"
                    and _assistant_content_key(server_message) == content_key
                ),
                None,
            )
            if match is None:
                reconciled.append(message)
                continue
            server_index, server_message = match
            claimed.add(server_index)
            reconciled.append({**message, "id": server_message["id"]})

        resolved = []
        for message in reconciled:
            if message["role"] != "assistant":
                resolved.append(message)
                continue
            server_message = None
            for part in message["parts"]:
                tool_call_id = part.get("toolCallId")
                if isinstance(tool_call_id, str) and tool_call_id in tool_messages:
                    server_message = tool_messages[tool_call_id]
                    break
            if server_message is not None and server_message["id"] != message["id"]:
                message = {**message, "id": server_message["id"]}
            resolved.append(message)
        return resolved

    async def _persist_messages(
        self,
        messages: list[Any],
        exclude: Iterable[str] = (),
        *,
        delete_stale_rows: bool = False,
        reconcile_with_history: bool = True,
        broadcast: bool = True,
    ) -> list[MessageT]:
        await self._ensure_chat_storage_ready()
        transformed = [
            _sanitize_message(message)
            for index, value in enumerate(messages)
            if (message := _transform_message(value, index)) is not None
        ]
        prior = (
            await self._reconciliation_prior(transformed)
            if reconcile_with_history
            else [
                _sanitize_message(message)
                for index, value in enumerate(self.messages)
                if (message := _transform_message(value, index)) is not None
            ]
        )
        merged = (
            self._reconcile_messages(transformed, prior)
            if reconcile_with_history
            else transformed
        )
        prior_by_id = {message["id"]: message for message in prior}
        stale_candidates = (
            await self._session.get_history_row_stats()
            if delete_stale_rows
            and all(message["id"] in prior_by_id for message in merged)
            else []
        )
        for message in merged:
            if prior_by_id.get(message["id"]) == message:
                continue
            await self._session.upsert_message(message)

        if delete_stale_rows:
            server_ids = {row.id for row in stale_candidates}
            if all(message["id"] in server_ids for message in merged):
                keep_ids = {message["id"] for message in merged}
                await self._session.delete_messages(
                    [row.id for row in stale_candidates if row.id not in keep_ids]
                )

        cap = self.max_persisted_messages
        if cap is not None:
            stored = await self._session.get_history_row_stats()
            excess = len(stored) - cap
            if excess > 0:
                await self._session.delete_messages([row.id for row in stored[:excess]])

        if broadcast:
            self.broadcast_json(chat_messages_frame(merged), exclude=exclude)
        return merged

    def _transcript_with_message(
        self,
        message: MessageT,
        transcript: list[MessageT] | None = None,
    ) -> list[MessageT]:
        return [
            *(
                existing
                for existing in (self.messages if transcript is None else transcript)
                if existing["id"] != message["id"]
            ),
            message,
        ]

    async def _persist_finished_streaming_message(
        self,
        message: MessageT,
        transcript: list[MessageT] | None = None,
        *,
        broadcast: bool = True,
    ) -> None:
        async with self._interaction_lock:
            if self._streaming_message is message:
                self._streaming_message = None
            await self._persist_messages(
                self._transcript_with_message(message, transcript),
                reconcile_with_history=False,
                broadcast=broadcast,
            )

    async def _persist_streaming_snapshot(self, message: MessageT) -> None:
        # Written as soon as a tool asks for approval, because a turn otherwise persists
        # nothing until it ends: a reload while the prompt is on screen would rebuild
        # from storage, find the part still merely running, and offer no way to answer.
        #
        # Deliberately silent. The client drew the prompt from the live chunk already,
        # so broadcasting the transcript here would draw it a second time.
        #
        # Best-effort on purpose: this only buys durability across a reload, and the
        # authoritative write still happens when the turn ends. Letting a failure escape
        # would abort a turn that is otherwise streaming fine, and the same bad value
        # would be raised by that later write anyway.
        with suppress(Exception):
            await self._session.upsert_message(message)

    # -- tool interactions -----------------------------------------------

    async def _find_tool_message(self, tool_call_id: str) -> MessageT | None:
        # The in-flight message first: while a turn is streaming, that is the only place
        # its parts exist.
        streaming = self._streaming_message
        if (
            streaming is not None
            and _find_tool_part(streaming["parts"], tool_call_id) is not None
        ):
            return streaming

        # A client-side tool can answer faster than the turn that called it takes to
        # finish and persist, so an answer that arrives to nothing waits for the message
        # rather than being dropped. An approval prompt is written out the moment it is
        # raised, so this mostly covers plain client tools.
        for attempt in range(_TOOL_PART_ATTEMPTS):
            if attempt:
                await asyncio.sleep(_TOOL_PART_RETRY_S)

            for message in reversed(self.messages):
                parts = message.get("parts") or []
                if _find_tool_part(parts, tool_call_id) is not None:
                    return message

        return None

    async def _apply_to_tool_part(
        self,
        tool_call_id: str,
        match_states: tuple[str, ...],
        update: Callable[[ChunkT], bool],
    ) -> bool:
        await self._ensure_chat_storage_ready()
        async with self._interaction_lock:
            message = await self._find_tool_message(tool_call_id)
            if message is None:
                return False

            part = _find_tool_part(message["parts"], tool_call_id)
            if part is None or part.get("state") not in match_states:
                # Already answered. A client retrying and a provider replaying both land
                # here, and the first answer is the one that stands, so this is a quiet
                # no-op rather than an error.
                return False

            if message is self._streaming_message:
                # Settled in place: the turn writes this message out when it ends, so
                # there is nothing to persist here.
                if update(part):
                    self.broadcast_json(message_updated_frame(message))
                    return True
                return False

            return await self._commit_tool_part(message, part, update)

    async def _commit_tool_part(
        self,
        message: MessageT,
        part: ChunkT,
        update: Callable[[ChunkT], bool],
    ) -> bool:
        parts = message.get("parts") or []
        candidate = dict(part)
        if not update(candidate):
            return False

        updated = {**message, "parts": [candidate if p is part else p for p in parts]}

        # Serialized before anything is committed. A tool output arrives as whatever the
        # client put in it, which can include a value that cannot go back out on the
        # wire, and a failure there has to leave both the stored row and the in-memory
        # transcript exactly as they were.
        dumps_wire(updated)
        stored = await self._session.update_message(updated)
        if stored is None:
            return False
        persisted = _transform_message(stored)
        if persisted is not None:
            self.broadcast_json(message_updated_frame(persisted))
        return True

    async def _handle_tool_result(self, data: dict[str, Any]) -> None:
        tool_call_id = data.get("toolCallId")
        if not isinstance(tool_call_id, str):
            return

        # The client sets this to mark a failure it is reporting itself, most often a
        # tool it declined to run; any other value means the call succeeded.
        errored = data.get("state") == "output-error"
        error_text = data.get("errorText")
        output = data.get("output")
        duplicate_matched = False

        def update(part: ChunkT) -> bool:
            nonlocal duplicate_matched
            if part.get("state") in TOOL_RESOLVED_STATES:
                duplicate_matched = (
                    part.get("state") == "output-error"
                    and errored
                    and part.get("errorText")
                    == (error_text or "Tool execution denied by user")
                ) or (
                    part.get("state") == "output-available"
                    and not errored
                    and part.get("output") == output
                )
                return False

            if errored:
                part["state"] = "output-error"
                part["errorText"] = error_text or "Tool execution denied by user"
            else:
                part["state"] = "output-available"
                part["output"] = output
                part["preliminary"] = False
            return True

        await self._apply_tool_interaction(
            _TOOL_INTERACTION_CONNECTION.get(),
            data,
            lambda: self._apply_to_tool_part(
                tool_call_id,
                _TOOL_RESULT_FROM,
                update,
            ),
            duplicate_matched=lambda: duplicate_matched,
        )

    async def _handle_tool_approval(self, data: dict[str, Any]) -> None:
        tool_call_id = data.get("toolCallId")
        approved = data.get("approved")
        if not isinstance(tool_call_id, str) or not isinstance(approved, bool):
            return

        def update(part: ChunkT) -> bool:
            existing = part.get("approval")
            if not isinstance(existing, dict):
                existing = {}

            approval_id = existing.get("id")
            if not isinstance(approval_id, str):
                # The decision frame carries no id of its own, so a part that never
                # recorded one — a hand-seeded transcript, or a provider that omitted it
                # — falls back to the call the approval belongs to. Something has to be
                # there, or the approval cannot be described back to the model.
                approval_id = tool_call_id

            part["state"] = "approval-responded" if approved else "output-denied"
            part["approval"] = {**existing, "id": approval_id, "approved": approved}
            return True

        await self._apply_tool_interaction(
            _TOOL_INTERACTION_CONNECTION.get(),
            data,
            lambda: self._apply_to_tool_part(
                tool_call_id,
                _TOOL_APPROVAL_FROM,
                update,
            ),
        )

    async def _apply_tool_interaction(
        self,
        connection: Connection | None,
        data: dict[str, Any],
        apply: Callable[[], Any],
        duplicate_matched: Callable[[], bool] | None = None,
    ) -> None:
        self._pending_interactions += 1
        self._interactions_idle.clear()
        try:
            applied = await apply()
        finally:
            self._pending_interactions -= 1
            if self._pending_interactions == 0:
                self._interactions_idle.set()
        if not applied and not (duplicate_matched is not None and duplicate_matched()):
            return
        if data.get("autoContinue") is True and connection is not None:
            body = data.get("body")
            client_tools = data.get("clientTools")
            continuation_body = (
                copy.deepcopy(body) if isinstance(body, dict) else dict(self._last_body)
            )
            continuation_tools = (
                copy.deepcopy(client_tools)
                if isinstance(client_tools, list)
                else copy.deepcopy(self._last_client_tools)
            )
            self._last_body = continuation_body
            self._last_client_tools = continuation_tools
            self._persist_request_context()
            self._continuation.schedule(
                connection,
                continuation_body,
                continuation_tools,
            )
        else:
            self._continuation.rearm()

    async def _drain_interactions(self) -> None:
        while self._pending_interactions:
            await self._interactions_idle.wait()

    def _has_incomplete_tool_batch(self) -> bool:
        assistant = next(
            (
                message
                for message in reversed(self.messages)
                if message["role"] == "assistant"
            ),
            None,
        )
        if assistant is None:
            return False
        states = {
            part.get("state")
            for part in assistant.get("parts", [])
            if isinstance(part.get("toolCallId"), str)
        }
        return bool(
            states & set(TOOL_RESOLVED_STATES + ("approval-responded",))
        ) and bool(states & {"input-available", "approval-requested"})

    def _spawn_continuation(
        self,
        factory: Callable[[], Any],
    ) -> None:
        self._lifecycle._retain_work(factory)

    # -- hooks -----------------------------------------------------------

    async def on_chat_message(self, options: ChatOptions) -> ChatReplyT:
        raise NotImplementedError(
            "received a chat message, override on_chat_message and return text "
            "or an (async) iterable of chunks to send to the client"
        )

    def format_agent_tool_input(self, input: Any, run_id: str) -> MessageT:
        text = input if isinstance(input, str) else dumps_wire(input)
        return {
            "id": f"agent-tool-input-{run_id}",
            "role": "user",
            "parts": [{"type": "text", "text": text}],
        }

    def get_agent_tool_output(
        self,
        run_id: str,
        input: Any,
        messages: list[MessageT],
    ) -> Any:
        return self._latest_assistant_text(messages)

    def get_agent_tool_summary(
        self,
        run_id: str,
        input: Any,
        output: Any,
        messages: list[MessageT],
    ) -> str:
        if isinstance(output, str):
            return output
        if output is None:
            return self._latest_assistant_text(messages) or ""
        return dumps_wire(output)

    async def prepare_agent_tool_turn(
        self,
        input: Any,
        run_id: str,
    ) -> frozenset[str]:
        assistant_ids = {
            message["id"]
            for message in self.messages
            if message.get("role") == "assistant" and isinstance(message.get("id"), str)
        }
        await self._persist_messages(
            self._transcript_with_message(self.format_agent_tool_input(input, run_id)),
            reconcile_with_history=False,
            broadcast=not self._messages_truncated,
        )
        return frozenset(assistant_ids)

    async def run_agent_tool_turn(
        self,
        run_id: str,
        input: Any,
        request_id: str,
        abort: asyncio.Event,
        context: TurnContext,
    ) -> str | None:
        message_id = f"msg-{gen_id()}"
        body = {"agentToolInput": input}
        recovery = self._turn_recovery_payload(
            request_id,
            message_id,
            body,
            None,
            continuation=False,
        )
        result: str | None = None

        async def run() -> None:
            nonlocal result
            result = await self._run_turn(
                None,
                request_id,
                "submit-message",
                body,
                abort,
                context,
                message_id=message_id,
                transcript=list(self.messages),
                broadcast_transcript=not self._messages_truncated,
            )

        await self._run_chat_turn_task(request_id, recovery, run)
        return result

    async def collect_agent_tool_result(
        self,
        run_id: str,
        input: Any,
        *,
        previous_assistant_ids: frozenset[str] | None = None,
        message_id: str | None = None,
    ) -> tuple[Any, str]:
        await self._ensure_chat_storage_ready()
        messages = []
        if message_id is not None:
            stored = await self._session.get_message(message_id)
            message = None if stored is None else _transform_message(stored)
            if message is not None and message.get("role") == "assistant":
                messages.append(message)

        for message in self.messages:
            if message.get("role") != "assistant":
                continue
            current_id = message.get("id")
            if (
                message_id is not None
                or not isinstance(current_id, str)
                or current_id in (previous_assistant_ids or ())
            ):
                continue
            messages.append(message)

        output = self.get_agent_tool_output(run_id, input, messages)
        if inspect.isawaitable(output):
            output = await output
        summary = self.get_agent_tool_summary(run_id, input, output, messages)
        if inspect.isawaitable(summary):
            summary = await summary
        return output, summary if isinstance(summary, str) else str(summary)

    def _stream_messages_as_json(self):
        encoder = TextEncoder.new()
        batches = self._session.history_batches()
        opened = False
        first = True
        proxies: list[Any] = []

        def release_proxies() -> None:
            while proxies:
                destroy = getattr(proxies.pop(), "destroy", None)
                if destroy is not None:
                    with suppress(Exception):
                        destroy()

        async def pull(controller) -> None:
            nonlocal opened, first
            try:
                if not opened:
                    opened = True
                    controller.enqueue(encoder.encode("["))
                    return
                batch = await anext(batches)
                chunk = []
                for index, stored in enumerate(batch):
                    message = _transform_message(stored, index)
                    if message is None:
                        continue
                    chunk.append(("" if first else ",") + dumps_wire(message))
                    first = False
                if chunk:
                    controller.enqueue(encoder.encode("".join(chunk)))
            except StopAsyncIteration:
                try:
                    controller.enqueue(encoder.encode("]"))
                    controller.close()
                finally:
                    release_proxies()
            except asyncio.CancelledError:
                try:
                    close = getattr(batches, "aclose", None)
                    if close is not None:
                        await close()
                finally:
                    release_proxies()
                raise
            except Exception as error:  # noqa: BLE001
                try:
                    controller.error(error_message(error))
                finally:
                    try:
                        close = getattr(batches, "aclose", None)
                        if close is not None:
                            await close()
                    finally:
                        release_proxies()

        async def cancel(_reason=None) -> None:
            try:
                close = getattr(batches, "aclose", None)
                if close is not None:
                    await close()
            finally:
                release_proxies()

        try:
            pull_proxy = create_proxy(pull)
            proxies.append(pull_proxy)
            cancel_proxy = create_proxy(cancel)
            proxies.append(cancel_proxy)
            source = to_js(
                {"pull": pull_proxy, "cancel": cancel_proxy},
                dict_converter=Object.fromEntries,
            )
            return ReadableStream.new(source)
        except Exception:
            release_proxies()
            raise

    async def _dispatch_request(self, request) -> Response:
        if url_path(request.url).endswith("/get-messages"):
            await self._ensure_chat_storage_ready()
            return Response(
                self._stream_messages_as_json(),
                headers={"Content-Type": "application/json"},
            )

        return await super()._dispatch_request(request)

    # -- dispatch --------------------------------------------------------

    async def _dispatch_connect(
        self,
        connection: Connection,
        ctx: ConnectionContext,
    ) -> None:
        await super()._dispatch_connect(connection, ctx)
        if not self.should_send_protocol_messages(connection, ctx):
            return
        if self._resumable.has_active_stream() or self._pre_stream.latest_request_id:
            return
        recovering = await self.ctx.storage.get(CHAT_RECOVERING_KEY)
        if not isinstance(recovering, dict):
            return
        at = recovering.get("at")
        request_id = recovering.get("requestId")
        if (
            type(at) is not int
            or not isinstance(request_id, str)
            or now_ms() - at >= CHAT_RECOVERING_FLAG_TTL_MS
        ):
            await self.ctx.storage.delete(CHAT_RECOVERING_KEY)
            return
        connection.send_if_open(
            {
                "type": ChatMessageType.CHAT_RECOVERING,
                "recovering": True,
                "id": request_id,
            }
        )

    async def _dispatch_message(self, connection: Connection, message: str) -> None:
        # not JSON, not an object, or a binary frame — hand it straight to the user
        data = loads_dict_or_none(message)
        if data is None:
            await super()._dispatch_message(connection, message)
            return

        _type = data.get("type")

        if _type == ChatMessageType.USE_CHAT_REQUEST:
            await self._handle_use_chat_request(connection, data)
        elif _type == ChatMessageType.CHAT_CLEAR:
            await self._handle_chat_clear(connection)
        elif _type == ChatMessageType.CHAT_REQUEST_CANCEL:
            await self._handle_cancel(data)
        elif _type == ChatMessageType.CHAT_MESSAGES:
            messages = data.get("messages")
            if not isinstance(messages, list):
                raise TypeError("chat messages must be a list")
            await self._persist_messages(
                messages,
                exclude=(connection.id,),
            )
        elif _type == ChatMessageType.STREAM_RESUME_REQUEST:
            self._handle_resume_request(connection, data)
        elif _type == ChatMessageType.STREAM_RESUME_ACK:
            self._handle_resume_ack(connection, data)
        elif _type == ChatMessageType.TOOL_RESULT:
            # Awaited rather than left to run on its own, so answers commit in the order
            # they arrive and the read-modify-write inside stays uninterrupted.
            token = _TOOL_INTERACTION_CONNECTION.set(connection)
            try:
                await self._handle_tool_result(data)
            finally:
                _TOOL_INTERACTION_CONNECTION.reset(token)
        elif _type == ChatMessageType.TOOL_APPROVAL:
            token = _TOOL_INTERACTION_CONNECTION.set(connection)
            try:
                await self._handle_tool_approval(data)
            finally:
                _TOOL_INTERACTION_CONNECTION.reset(token)
        else:
            await super()._dispatch_message(connection, message)

    async def _dispatch_close(
        self,
        connection: Connection,
        code: int,
        reason: str,
        was_clean: bool,
    ) -> None:
        # Dropped so the pending-resume set does not accumulate ghosts of connections
        # that closed before they ACKed.
        self._pending_resume_connections.discard(connection.id)
        self._pending_resume_requests.pop(connection.id, None)
        self._pre_stream.release(connection.id)
        self._continuation.release_connection(connection.id)
        await super()._dispatch_close(connection, code, reason, was_clean)

    async def _handle_chat_clear(self, connection: Connection) -> None:
        await self._ensure_chat_storage_ready()
        self._turn_queue.reset()
        active_request_ids = tuple(self._aborts)
        for abort in self._aborts.values():
            abort.set()
        for request_id in active_request_ids:
            self._broadcast_terminal(request_id)
        self._child_agent_tool_runs.abort_active()
        self._continuation.reset()
        self._continuation_request_ids.clear()

        task_rows = self.sql(
            "SELECT run_id FROM cf_agents_task_runs WHERE definition IN (?, ?) "
            "AND state NOT IN ('completed', 'failed', 'cancelled')",
            _CHAT_TURN_TASK,
            _CHAT_RECOVERY_TASK,
        )
        for row in task_rows:
            await self.tasks.cancel(row["run_id"], "chat cleared")

        await self._session.clear_messages()
        incident_keys = await self.ctx.storage.list(
            {"prefix": CHAT_RECOVERY_INCIDENT_KEY_PREFIX}
        )
        recovery_keys = [*incident_keys, CHAT_RECOVERING_KEY]
        await self.ctx.storage.delete(recovery_keys)
        self._clear_request_context()
        self._pre_stream.release_awaiting()
        self._resumable.clear_all()
        self._pre_stream.reset()
        self._pending_resume_connections.clear()
        self._pending_resume_requests.clear()
        self.broadcast_json(chat_clear_frame(), exclude=(connection.id,))
        self._spawn_reschedule()

    async def _handle_cancel(self, data: dict[str, Any]) -> None:
        request_id = data.get("id")
        if not isinstance(request_id, str):
            return

        abort = self._aborts.get(request_id)
        if abort is not None:
            abort.set()
            self._broadcast_terminal(request_id)
            self._resumable.cancel_request(request_id)
        run_id = self._request_task_runs.get(request_id)
        if run_id is not None:
            await self.tasks.cancel(run_id, "chat request cancelled")

    def _handle_resume_request(
        self,
        connection: Connection,
        data: dict[str, Any],
    ) -> None:
        # The client needs a definite answer here or it waits out its probe timeout.
        # Synchronous on purpose: with no await it runs to completion between two awaits
        # of a live turn, so the offer cannot interleave with a chunk broadcast.
        probe_id = data.get("probeId")
        if self._resumable.has_active_stream():
            if (
                self._continuation.active_request_id
                == self._resumable.active_request_id
                and self._continuation.active_connection_id not in (None, connection.id)
                and self._continuation.active_connection_id in self._connections
            ):
                self._send_resume_none(
                    connection,
                    probe_id,
                    reason="continuation-owned",
                )
            else:
                self._notify_stream_resuming(connection, probe_id)
        elif (
            self._continuation.pending is not None
            and self._continuation.pending.connection_id == connection.id
        ):
            self._continuation.awaiting_connections[connection.id] = (
                connection,
                probe_id,
            )
            connection.send_if_open(
                {
                    "type": ChatMessageType.STREAM_PENDING,
                    "id": self._continuation.pending.request_id,
                    **({"probeId": probe_id} if probe_id is not None else {}),
                }
            )
        elif (
            self._continuation.pending is not None
            and self._continuation.pending.connection_id in self._connections
        ):
            self._send_resume_none(
                connection,
                probe_id,
                reason="continuation-owned",
            )
        elif (terminal := self._resumable.latest_terminal_error()) is not None:
            self._notify_stream_resuming(
                connection,
                probe_id,
                request_id=terminal.request_id,
            )
        elif self._pre_stream.park(connection, probe_id):
            return
        else:
            self._send_resume_none(connection, probe_id)

    def _handle_resume_ack(
        self,
        connection: Connection,
        data: dict[str, Any],
    ) -> None:
        request_id = data.get("id")
        if not isinstance(request_id, str):
            return

        # Done before the replay, so chunks stored after this point arrive live and the
        # replay covers everything up to now — no gap, no duplicate.
        self._pending_resume_connections.discard(connection.id)
        self._pending_resume_requests.pop(connection.id, None)

        active = self._resumable.active_request_id
        if active == request_id:
            self._resumable.replay_active_chunks(connection, request_id)
            self._spawn_reschedule()
        elif (
            terminal := self._resumable.latest_terminal_error()
        ) is not None and terminal.request_id == request_id:
            self._resumable.replay_error_chunks(connection, request_id)
            connection.send_if_open(
                self._terminal_frame(
                    request_id,
                    terminal.body,
                    continuation=terminal.is_continuation,
                )
            )
        elif not self._resumable.replay_completed_chunks(connection, request_id):
            # A different stream may now be active, but this ACK is for its own id: if
            # that turn already completed its buffer is still retained and gets replayed
            # above, so reaching here means neither active nor retained. Settle the
            # client with a bare terminal so it does not hang waiting out its probe.
            connection.send_if_open(
                chat_response(
                    ChatMessageType.USE_CHAT_RESPONSE,
                    request_id,
                    done=True,
                    replay=True,
                    continuation=self._is_continuation_request(request_id),
                )
            )

    def _notify_stream_resuming(
        self,
        connection: Connection,
        probe_id: Any = None,
        *,
        request_id: str | None = None,
    ) -> None:
        request_id = request_id or self._resumable.active_request_id
        if request_id is None:
            return

        # Only exclude it from the broadcast if the offer actually went out; a closed
        # socket never ACKs and should not linger pending.
        if connection.send_if_open(stream_resuming_frame(request_id, probe_id)):
            self._pending_resume_connections.add(connection.id)
            self._pending_resume_requests[connection.id] = request_id

    def _send_resume_none(
        self,
        connection: Connection,
        probe_id: Any = None,
        *,
        reason: str = "idle",
    ) -> None:
        connection.send_if_open(stream_resume_none_frame(probe_id, reason=reason))

    # -- the turn --------------------------------------------------------

    async def _handle_use_chat_request(
        self,
        connection: Connection,
        data: dict[str, Any],
    ) -> None:
        init = data.get("init")
        if not isinstance(init, dict):
            return

        if str(init.get("method") or "").upper() != "POST":
            return

        request_id = data.get("id")
        if not isinstance(request_id, str):
            # Without an id there is nothing to address a reply on, so the frame is
            # dropped.
            return

        # Everything past here has an id and must answer: the client resolves only on a
        # terminal frame, so a silent escape leaves its promise pending forever.
        if request_id in self._aborts:
            self._send_terminal(
                connection,
                request_id,
                error="Request already active",
                dedupe=False,
            )
            return

        self._terminal_request_ids.discard(request_id)
        self._pre_stream.begin(request_id)
        terminal = self._resumable.latest_terminal_error()
        if terminal is not None:
            for connection_id, pending_request_id in tuple(
                self._pending_resume_requests.items()
            ):
                if pending_request_id == terminal.request_id:
                    self._pending_resume_connections.discard(connection_id)
                    self._pending_resume_requests.pop(connection_id, None)
        self._resumable.clear_terminal_error()
        abort = asyncio.Event()
        self._aborts[request_id] = abort
        try:

            async def run(context: TurnContext) -> None:
                if abort.is_set():
                    self._send_terminal(connection, request_id)
                    return
                await self._start_turn(
                    connection,
                    request_id,
                    init,
                    abort,
                    context,
                )

            result = await self._turn_queue.enqueue(request_id, run)
            if result.status == "stale":
                self._send_terminal(connection, request_id)
        except Exception as exc:
            if _is_platform_failure(exc):
                raise
            # Terminal first, because the client's promise depends on one arriving and a
            # re-raising hook must not stop it. Then on_error, for observability.
            self._send_terminal(
                connection,
                request_id,
                error=error_message(exc),
            )
            self._resumable.record_terminal_error(
                request_id,
                error_message(exc),
                continuation=self._is_continuation_request(request_id),
            )
            await self._report_error(exc, connection)
        finally:
            self._aborts.pop(request_id, None)
            self._terminal_request_ids.discard(request_id)
            if self._pre_stream.settle(request_id):
                terminal = self._resumable.latest_terminal_error()
                if terminal is None:
                    self._pre_stream.release_awaiting()
                else:
                    self._pre_stream.flush_on_stream_start(
                        lambda pending: self._notify_stream_resuming(
                            pending,
                            request_id=terminal.request_id,
                        )
                    )

    async def _start_turn(
        self,
        connection: Connection,
        request_id: str,
        init: dict[str, Any],
        abort: asyncio.Event,
        context: TurnContext,
    ) -> None:
        payload = json.loads(init.get("body") or "{}")
        if not isinstance(payload, dict):
            raise TypeError("request body must be a JSON object")

        incoming = payload.pop("messages", None)
        if not isinstance(incoming, list):
            raise TypeError("chat request messages must be a list")
        transformed = []
        for index, value in enumerate(incoming):
            message = _transform_message(value, index)
            if message is None:
                raise TypeError(f"invalid chat message at index {index}")
            transformed.append(message)

        trigger = payload.pop("trigger", None)
        if trigger not in ("regenerate-message", "submit-message"):
            trigger = "submit-message"
        client_tools = payload.pop("clientTools", None)
        if not isinstance(client_tools, list):
            client_tools = None
        self._last_client_tools = client_tools
        self._last_body = dict(payload)
        self._persist_request_context()

        transcript = await self._persist_messages(
            transformed,
            exclude=(connection.id,),
            delete_stale_rows=True,
        )

        message_id = f"msg-{gen_id()}"
        latest_user_id = next(
            (
                message.get("id")
                for message in reversed(transformed)
                if message.get("role") == "user"
            ),
            None,
        )
        recovery = {
            "version": _CHAT_RECOVERY_VERSION,
            "requestId": request_id,
            "messageId": message_id,
            "trigger": trigger,
            "body": payload,
            "clientTools": client_tools,
            "continuation": False,
            "recoveryRootRequestId": request_id,
            "latestUserMessageId": latest_user_id,
            "startedAt": now_ms(),
        }

        await self._run_chat_turn_task(
            request_id,
            recovery,
            lambda: self._run_turn(
                connection,
                request_id,
                trigger,
                payload,
                abort,
                context,
                message_id=message_id,
                transcript=transcript,
                client_tools=client_tools,
            ),
        )

    async def _run_chat_turn_task(
        self,
        request_id: str,
        recovery: dict[str, Any],
        run: Callable[[], Any],
    ) -> None:
        nonce = gen_id()
        run_id = f"{_CHAT_TASK_RUN_PREFIX}{nonce}"
        self._live_chat_turns[nonce] = run
        self._request_task_runs[request_id] = run_id
        try:
            receipt = await self.tasks._run_reserved(
                _CHAT_TURN_TASK,
                {**recovery, "nonce": nonce},
                run_id=run_id,
                metadata={"requestId": request_id},
                retain=False,
            )
            await self.tasks._execution_task(receipt.run_id)
        finally:
            self._live_chat_turns.pop(nonce, None)
            if self._request_task_runs.get(request_id) == run_id:
                self._request_task_runs.pop(request_id, None)

    async def _chat_turn_task(self, input: Any, step: TaskStep) -> None:
        if not isinstance(input, dict):
            raise TypeError("chat turn Task input must be an object")
        nonce = input.get("nonce")
        if not isinstance(nonce, str):
            raise TypeError("chat turn Task input requires a nonce")

        async def model_turn(_attempt) -> None:
            live = self._live_chat_turns.get(nonce)
            if live is not None:
                result = live()
                if inspect.isawaitable(result):
                    await result
                return
            if self.durable_chat_recovery:
                await self._recover_interrupted_chat_turn(input)

        await step.do(
            "model-turn",
            TaskStepConfig(
                retries=TaskStepRetryOptions(limit=1),
                timeout="1 day",
            ),
            model_turn,
        )

    async def _chat_recovery_task(self, input: Any, step: TaskStep) -> None:
        if not isinstance(input, dict) or not isinstance(input.get("data"), dict):
            raise TypeError("chat recovery Task input must be an object")
        delay = input.get("delaySeconds", 0)
        if not isinstance(delay, (int, float)) or isinstance(delay, bool) or delay < 0:
            raise TypeError("chat recovery Task delay must be non-negative")
        if delay:
            await step.sleep("backoff", delay * 1000)

        async def continuation(_attempt) -> None:
            await self._dispatch_recovery_handoff(input["data"])

        await step.do(
            "continuation",
            TaskStepConfig(
                retries=TaskStepRetryOptions(limit=1),
                timeout="15 minutes",
            ),
            continuation,
        )

    async def _fire_auto_continuation(
        self,
        pending: ContinuationRequest,
    ) -> None:
        request_id = pending.request_id
        self._continuation_request_ids.add(request_id)
        abort = asyncio.Event()
        self._aborts[request_id] = abort
        self._pre_stream.begin(request_id)
        try:
            assistant = next(
                (
                    message
                    for message in reversed(self.messages)
                    if message["role"] == "assistant"
                ),
                None,
            )
            if assistant is None:
                self._continuation.reset()
                return

            async def run(context: TurnContext) -> None:
                recovery = self._turn_recovery_payload(
                    request_id,
                    assistant["id"],
                    pending.body,
                    pending.client_tools,
                    continuation=True,
                )
                await self._run_chat_turn_task(
                    request_id,
                    recovery,
                    lambda: self._run_turn(
                        pending.connection,
                        request_id,
                        "submit-message",
                        pending.body,
                        abort,
                        context,
                        message_id=assistant["id"],
                        transcript=list(self.messages),
                        continuation=True,
                        client_tools=pending.client_tools,
                    ),
                )

            result = await self._turn_queue.enqueue(request_id, run)
            if result.status == "stale":
                self._continuation.reset()
        except Exception as exc:
            if _is_platform_failure(exc):
                raise
            self._continuation.reset()
            await self._report_error(exc, pending.connection)
        finally:
            self._aborts.pop(request_id, None)
            self._terminal_request_ids.discard(request_id)
            if self._pre_stream.settle(request_id):
                self._pre_stream.release_awaiting()

    def _turn_recovery_payload(
        self,
        request_id: str,
        message_id: str,
        body: dict[str, Any],
        client_tools: list[Any] | None,
        *,
        continuation: bool,
        recovery_root_request_id: str | None = None,
    ) -> dict[str, Any]:
        latest_user_id = next(
            (
                message["id"]
                for message in reversed(self.messages)
                if message["role"] == "user"
            ),
            None,
        )
        return {
            "version": _CHAT_RECOVERY_VERSION,
            "requestId": request_id,
            "messageId": message_id,
            "trigger": "submit-message",
            "body": body,
            "clientTools": client_tools,
            "continuation": continuation,
            "recoveryRootRequestId": recovery_root_request_id or request_id,
            "latestUserMessageId": latest_user_id,
            "startedAt": now_ms(),
        }

    async def _run_turn(
        self,
        connection: Connection | None,
        request_id: str,
        trigger: str,
        body: dict[str, Any],
        abort: asyncio.Event,
        context: TurnContext,
        *,
        message_id: str | None = None,
        transcript: list[MessageT] | None = None,
        broadcast_transcript: bool = True,
        continuation: bool = False,
        client_tools: list[Any] | None = None,
    ) -> str | None:
        message_id = message_id or f"msg-{gen_id()}"
        options = ChatOptions(
            request_id,
            trigger,
            body,
            abort,
            continuation=continuation,
            client_tools=client_tools,
        )
        prior_message = (
            next(
                (
                    message
                    for message in reversed(transcript or self.messages)
                    if message.get("role") == "assistant"
                    and message.get("id") == message_id
                ),
                None,
            )
            if continuation
            else None
        )
        parts: list[ChunkT] = copy.deepcopy(
            prior_message.get("parts", []) if prior_message is not None else []
        )
        # part type -> chunk id, for blocks opened but not yet closed
        open_blocks: dict[str, str] = {}

        # Pre-stream tracking covers queue admission; registration takes over before
        # the first chunk so a resume probe never observes accepted work as idle.
        self._resumable.clear_terminal_error()
        try:
            stream_id = self._resumable.start(
                request_id,
                message_id,
                continuation=continuation,
            )
        except Exception as exc:  # noqa: BLE001
            error = error_message(exc)
            self._resumable.record_terminal_error(
                request_id,
                error,
                continuation=continuation,
            )
            if connection is not None:
                self._send_terminal(
                    connection,
                    request_id,
                    error=error,
                    continuation=continuation,
                )
            await self._report_error(exc, connection)
            return error
        for connection_id, pending_request_id in tuple(
            self._pending_resume_requests.items()
        ):
            if pending_request_id != request_id:
                self._pending_resume_connections.discard(connection_id)
                self._pending_resume_requests.pop(connection_id, None)
        self._pre_stream.settle(request_id)
        self._pre_stream.flush_on_stream_start(
            lambda pending: self._notify_stream_resuming(
                pending,
                request_id=request_id,
            )
        )
        if self._continuation.pending is not None:
            self._continuation.activate(request_id)
            for pending, _probe_id in tuple(
                self._continuation.awaiting_connections.values()
            ):
                self._notify_stream_resuming(pending, request_id=request_id)
            self._continuation.awaiting_connections.clear()
        self._spawn_reschedule()

        # Shares the parts list rather than copying it, so settling a part in place is
        # visible through both.
        message: MessageT = {
            **(
                {
                    key: copy.deepcopy(value)
                    for key, value in prior_message.items()
                    if key not in ("id", "role", "parts")
                }
                if prior_message is not None
                else {}
            ),
            "id": message_id,
            "role": "assistant",
            "parts": parts,
        }
        accumulator = MessageAccumulator(parts, message)

        # The final write detaches this under the interaction lock, so an arriving tool
        # result either mutates this object first or waits and updates its stored row.
        self._streaming_message = message
        try:
            try:
                self._emit(
                    stream_id,
                    request_id,
                    {
                        "type": "start",
                        "messageId": message_id,
                    },
                )

                reply = self.on_chat_message(options)
                if inspect.isawaitable(reply):
                    try:
                        if self.chat_provider_stall_timeout_ms is None:
                            reply = await reply
                        else:
                            reply = await asyncio.wait_for(
                                reply,
                                self.chat_provider_stall_timeout_ms / 1000,
                            )
                    except TimeoutError as exc:
                        raise TimeoutError("Chat provider stream stalled") from exc

                if not self._turn_queue.is_current(context):
                    self._resumable.mark_error(stream_id)
                    if connection is not None:
                        self._send_terminal(connection, request_id)
                    return None

                chunks = _normalize(reply).__aiter__()
                while True:
                    try:
                        if self.chat_provider_stall_timeout_ms is None:
                            chunk = await anext(chunks)
                        else:
                            chunk = await asyncio.wait_for(
                                anext(chunks),
                                self.chat_provider_stall_timeout_ms / 1000,
                            )
                    except StopAsyncIteration:
                        break
                    except TimeoutError as exc:
                        raise TimeoutError("Chat provider stream stalled") from exc
                    if not self._turn_queue.is_current(context):
                        self._resumable.mark_error(stream_id)
                        if connection is not None:
                            self._send_terminal(connection, request_id)
                        return None
                    if accumulator.should_suppress(chunk):
                        continue

                    effect = accumulator.apply(chunk)
                    if chunk.get("type") == "start" and message["id"] != message_id:
                        message_id = message["id"]
                        self._resumable.update_message_id(stream_id, message_id)
                    if effect.terminal_error is not None:
                        self._resumable.mark_error(stream_id)
                        self._resumable.record_terminal_error(
                            request_id,
                            effect.terminal_error,
                            continuation=continuation,
                        )
                        self._broadcast_terminal(
                            request_id,
                            effect.terminal_error,
                            continuation=continuation,
                        )
                        return effect.terminal_error

                    self._track_open_block(open_blocks, chunk)
                    self._emit(stream_id, request_id, chunk)
                    await self._maybe_bump_recovery_progress(chunk)

                    if effect.persist_now:
                        await self._persist_streaming_snapshot(message)

                    if abort.is_set():
                        break

                if not self._turn_queue.is_current(context):
                    self._resumable.mark_error(stream_id)
                    if connection is not None:
                        self._send_terminal(connection, request_id)
                    return None

                # A no-op when the stream closed its own blocks. An abort breaks out
                # before they close, and a provider can end a turn without them, so both
                # would otherwise leave the client with a part stuck in "streaming".
                self._close_open_blocks(stream_id, request_id, parts, open_blocks)

                self._emit(stream_id, request_id, {"type": "finish"})

                if parts:
                    await self._persist_finished_streaming_message(
                        message,
                        transcript,
                        broadcast=broadcast_transcript,
                    )
            except Exception as exc:
                if _is_platform_failure(exc):
                    raise
                if not self._turn_queue.is_current(context):
                    self._resumable.mark_error(stream_id)
                    if connection is not None:
                        self._send_terminal(connection, request_id)
                    return None
                # Terminal first so the promise settles, then notify. Marking errored
                # keeps the stream from being offered for resume.
                self._resumable.mark_error(stream_id)
                self._resumable.record_terminal_error(
                    request_id,
                    error_message(exc),
                    continuation=continuation,
                )
                self._broadcast_terminal(
                    request_id,
                    error=error_message(exc),
                    continuation=continuation,
                )
                await self._report_error(exc, connection)
                return error_message(exc)

            self._resumable.complete(stream_id)
            self._broadcast_terminal(request_id, continuation=continuation)
            return None
        finally:
            self._streaming_message = None
            if continuation:
                self._continuation.finish(request_id)
            else:
                self._continuation.rearm()
            self._spawn_reschedule()

    async def on_chat_recovery(self, _context: dict[str, Any]) -> dict[str, Any]:
        """Customize whether an interrupted partial is persisted and continued."""
        return {}

    async def on_chat_recovery_exhausted(self, _context: dict[str, Any]) -> None:
        """Observe a recovery incident after its bounded budget is exhausted."""

    async def _recover_interrupted_chat_turn(self, recovery: dict[str, Any]) -> None:
        request_id = recovery.get("requestId")
        message_id = recovery.get("messageId")
        if not isinstance(request_id, str) or not isinstance(message_id, str):
            raise TypeError("interrupted chat turn has invalid recovery identity")
        stream = self._resumable.latest_for_request(request_id)
        current_message_id = stream.message_id if stream is not None else message_id
        latest_user_id = recovery.get("latestUserMessageId")
        leaf = self.messages[-1] if self.messages else None
        if (
            not isinstance(latest_user_id, str)
            or leaf is None
            or leaf.get("id") not in (latest_user_id, current_message_id)
        ):
            stale = stream
            if stale is not None and stale.status == StreamStatus.STREAMING:
                self._resumable.mark_orphaned(stale.stream_id)
            return
        snapshot = self._resumable.recovery_snapshot(
            request_id,
            current_message_id,
            max_chunks=self.chat_recovery_max_chunks,
            max_bytes=self.chat_recovery_max_bytes,
        )
        if snapshot is not None and snapshot.stream.status != StreamStatus.STREAMING:
            return
        continuation = bool(recovery.get("continuation"))
        stream_id = snapshot.stream.stream_id if snapshot is not None else ""
        message_id = (
            snapshot.stream.message_id or message_id
            if snapshot is not None
            else message_id
        )
        existing = next(
            (message for message in self.messages if message.get("id") == message_id),
            None,
        )
        parts: list[ChunkT] = copy.deepcopy(
            existing.get("parts", []) if continuation and existing is not None else []
        )
        message: MessageT = {
            **(
                {
                    key: copy.deepcopy(value)
                    for key, value in existing.items()
                    if key not in ("id", "role", "parts")
                }
                if existing is not None
                else {}
            ),
            "id": message_id,
            "role": "assistant",
            "parts": parts,
        }
        accumulator = MessageAccumulator(parts, message)
        partial_parts_start = len(parts)
        for body in snapshot.bodies if snapshot is not None else ():
            chunk = loads_dict_or_none(body)
            if chunk is None or accumulator.should_suppress(chunk):
                continue
            effect = accumulator.apply(chunk)
            if effect.terminal_error is not None:
                if stream_id:
                    self._resumable.mark_error(stream_id)
                self._resumable.record_terminal_error(
                    request_id,
                    effect.terminal_error,
                    continuation=continuation,
                )
                return
        for part in parts:
            if part.get("state") == "streaming":
                part["state"] = "done"

        if snapshot is not None and snapshot.limit_exceeded:
            if len(parts) > partial_parts_start:
                await self._persist_recovered_partial(message)
            await self._exhaust_recovery(
                recovery,
                "reconstruction_limits_exceeded",
                stream_id=snapshot.stream.stream_id,
            )
            return

        recovery_kind = (
            "continue" if continuation or len(parts) > partial_parts_start else "retry"
        )
        incident, exhausted = await self._begin_recovery_incident(
            recovery,
            recovery_kind,
        )
        if exhausted:
            if len(parts) > partial_parts_start:
                await self._persist_recovered_partial(message)
            await self._exhaust_recovery(
                recovery,
                str(incident.get("reason") or "max_attempts_exceeded"),
                incident=incident,
                stream_id=stream_id,
            )
            return

        context = {
            "incidentId": incident["incidentId"],
            "recoveryRootRequestId": recovery.get("recoveryRootRequestId")
            or request_id,
            "attempt": incident["attempt"],
            "maxAttempts": incident["maxAttempts"],
            "recoveryKind": recovery_kind,
            "streamId": stream_id,
            "requestId": request_id,
            "partialParts": copy.deepcopy(parts[partial_parts_start:]),
            "messages": copy.deepcopy(self.messages),
            "createdAt": recovery.get("startedAt"),
        }
        try:
            options = self.on_chat_recovery(context)
            if inspect.isawaitable(options):
                options = await options
            if not isinstance(options, dict):
                options = {}
            settled = any(
                part.get("state") in TOOL_RESOLVED_STATES
                for part in parts[partial_parts_start:]
            )
            if len(parts) > partial_parts_start and (
                options.get("persist") is not False or settled
            ):
                await self._persist_recovered_partial(message)
            if stream_id:
                self._resumable.complete(stream_id)
            self._broadcast_terminal(request_id, continuation=continuation)
            if self._awaiting_client_interaction():
                await self._update_recovery_incident(
                    incident,
                    "skipped",
                    "awaiting_client_interaction",
                )
                return
            if options.get("continue") is False:
                await self._update_recovery_incident(
                    incident,
                    "skipped",
                    "continue_disabled",
                )
                return
            target = (
                message_id
                if recovery_kind == "continue"
                and any(message.get("id") == message_id for message in self.messages)
                else None
            )
            data = {
                "incidentId": incident["incidentId"],
                "kind": "continue" if target is not None else "retry",
                "originalRequestId": recovery.get("recoveryRootRequestId")
                or request_id,
                "latestUserMessageId": recovery.get("latestUserMessageId"),
                "body": recovery.get("body")
                if isinstance(recovery.get("body"), dict)
                else copy.deepcopy(self._last_body),
                "clientTools": recovery.get("clientTools")
                if isinstance(recovery.get("clientTools"), list)
                else copy.deepcopy(self._last_client_tools),
                **({"targetAssistantId": target} if target is not None else {}),
            }
            await self._update_recovery_incident(incident, "scheduled")
            await self._set_recovering(True, data["originalRequestId"])
            try:
                await self._enqueue_recovery(data, initial=True)
            except BaseException as exc:
                if not _is_platform_failure(exc):
                    raise
                await self._exhaust_recovery(
                    recovery,
                    "acceptance_failed",
                    incident=incident,
                    stream_id=stream_id,
                )
        except BaseException as exc:
            if isinstance(exc, asyncio.CancelledError) or _is_platform_failure(exc):
                raise
            await self._update_recovery_incident(
                incident,
                "failed",
                error_message(exc),
            )
            raise

    async def _persist_recovered_partial(self, message: MessageT) -> None:
        await self._persist_finished_streaming_message(
            message,
            broadcast=not self._messages_truncated,
        )
        if self._messages_truncated:
            self.broadcast_json(message_updated_frame(message))

    async def _begin_recovery_incident(
        self,
        recovery: dict[str, Any],
        recovery_kind: str,
    ) -> tuple[dict[str, Any], bool]:
        now = now_ms()
        stored = await self.ctx.storage.list(
            {"prefix": CHAT_RECOVERY_INCIDENT_KEY_PREFIX}
        )
        stale = [
            key
            for key, value in stored.items()
            if not isinstance(value, dict)
            or now - int(value.get("lastAttemptAt") or value.get("firstSeenAt") or 0)
            > CHAT_RECOVERY_INCIDENT_TTL_MS
        ]
        if stale:
            await self.ctx.storage.delete(stale)
        request_id = str(recovery["requestId"])
        root_id = recovery.get("recoveryRootRequestId")
        user_id = recovery.get("latestUserMessageId")
        value = incident_id(
            request_id,
            root_id if isinstance(root_id, str) else None,
            user_id if isinstance(user_id, str) else None,
        )
        key = incident_key(value)
        existing = await self.ctx.storage.get(key)
        if isinstance(existing, dict) and existing.get("status") == "exhausted":
            return existing, True
        progress = await self.ctx.storage.get(CHAT_RECOVERY_PROGRESS_KEY)
        decision = evaluate_incident(
            request_id=request_id,
            recovery_root_request_id=root_id if isinstance(root_id, str) else None,
            latest_user_message_id=user_id if isinstance(user_id, str) else None,
            recovery_kind="continue" if recovery_kind == "continue" else "retry",
            existing=existing if isinstance(existing, dict) else None,
            progress=progress if type(progress) is int else 0,
            awaiting_client_interaction=self._awaiting_client_interaction(),
            now=now,
            policy=RecoveryPolicy(
                max_attempts=self.chat_recovery_max_attempts,
                no_progress_timeout_ms=self.chat_recovery_no_progress_timeout_ms,
                max_work=self.chat_recovery_max_work,
                max_oom_retries=self.chat_recovery_max_oom_retries,
            ),
        )
        await self.ctx.storage.put(key, decision.incident)
        return decision.incident, decision.exhausted

    async def _update_recovery_incident(
        self,
        incident: dict[str, Any],
        status: str,
        reason: str | None = None,
    ) -> None:
        key = incident_key(str(incident["incidentId"]))
        if status == "completed":
            await self.ctx.storage.delete(key)
        else:
            updated = {**incident, "status": status, "lastAttemptAt": now_ms()}
            if reason is not None:
                updated["reason"] = reason
            await self.ctx.storage.put(key, updated)
        if status in ("completed", "skipped", "failed", "exhausted"):
            await self._set_recovering(False, None)

    async def _set_recovering(self, active: bool, request_id: str | None) -> None:
        existing = await self.ctx.storage.get(CHAT_RECOVERING_KEY)
        if active:
            if isinstance(existing, dict):
                return
            await self.ctx.storage.put(
                CHAT_RECOVERING_KEY,
                {"requestId": request_id, "at": now_ms()},
            )
        else:
            if existing is None:
                return
            await self.ctx.storage.delete(CHAT_RECOVERING_KEY)
            if request_id is None and isinstance(existing, dict):
                value = existing.get("requestId")
                request_id = value if isinstance(value, str) else None
        frame: dict[str, Any] = {
            "type": ChatMessageType.CHAT_RECOVERING,
            "recovering": active,
        }
        if request_id is not None:
            frame["id"] = request_id
        self.broadcast_json(frame)

    def _awaiting_client_interaction(self) -> bool:
        return any(
            part.get("state") in ("input-available", "approval-requested")
            for message in reversed(self.messages)
            if message.get("role") == "assistant"
            for part in message.get("parts", [])
        )

    async def _enqueue_recovery(
        self,
        data: dict[str, Any],
        *,
        initial: bool,
        delay_seconds: float = 0,
        run_id: str | None = None,
    ) -> None:
        callback = str(data.get("kind") or "retry")
        acceptance_attempts = max(1, self.chat_recovery_acceptance_attempts)
        for attempt in range(acceptance_attempts):
            try:
                await self.tasks._run_reserved(
                    _CHAT_RECOVERY_TASK,
                    {"data": data, "delaySeconds": delay_seconds},
                    run_id=run_id,
                    idempotency_key=(
                        f"chat-recovery:{data['incidentId']}" if initial else None
                    ),
                    metadata={
                        "callback": callback,
                        "incidentId": str(data["incidentId"]),
                        "recoveredRequestId": str(data["originalRequestId"]),
                    },
                    retain=False,
                )
                return
            except BaseException as exc:
                if not _is_platform_failure(exc) or attempt + 1 >= acceptance_attempts:
                    raise

    async def _redefer_recovery(self, data: dict[str, Any], reason: str) -> None:
        sequence = int(data.get("redeferSequence") or 0) + 1
        incident_value = data.get("incidentId")
        if not isinstance(incident_value, str):
            return
        key = incident_key(incident_value)
        stored_incident = await self.ctx.storage.get(key)
        incident = (
            stored_incident
            if isinstance(stored_incident, dict)
            else {
                "incidentId": incident_value,
                "maxAttempts": self.chat_recovery_max_attempts,
            }
        )
        if sequence > int(
            incident.get("maxAttempts") or self.chat_recovery_max_attempts
        ):
            await self._exhaust_recovery(
                {"recoveryRootRequestId": data.get("originalRequestId")},
                "max_attempts_exceeded",
                incident=incident,
            )
            return
        successor = {**data, "redeferSequence": sequence}
        try:
            await self._enqueue_recovery(
                successor,
                initial=False,
                delay_seconds=self.chat_recovery_retry_delay_seconds,
                run_id=(f"chat-recovery-redefer:{incident_value}:{reason}:{sequence}"),
            )
        except BaseException as exc:
            if not _is_platform_failure(exc):
                raise
            await self._exhaust_recovery(
                {"recoveryRootRequestId": data.get("originalRequestId")},
                "acceptance_failed",
                incident=incident,
            )

    async def _dispatch_recovery_handoff(self, data: dict[str, Any]) -> None:
        started = asyncio.Event()

        async def detached() -> None:
            try:
                await self._execute_recovery_attempt(data, started)
            except BaseException as exc:
                if _is_memory_limit_reset(exc):
                    await self._handle_recovery_oom(data)
                    return
                if _is_platform_failure(exc):
                    await self._redefer_recovery(data, "platform")
                    return
                raise

        turn = asyncio.create_task(detached())
        started_wait = asyncio.create_task(started.wait())
        done, _ = await asyncio.wait(
            {turn, started_wait},
            return_when=asyncio.FIRST_COMPLETED,
            timeout=self.chat_recovery_stable_timeout_ms / 1000,
        )
        if not done:
            turn.cancel()
            started_wait.cancel()
            await asyncio.gather(turn, started_wait, return_exceptions=True)
            await self._reschedule_stable_timeout(data)
            return
        if turn in done:
            started_wait.cancel()
            await asyncio.gather(started_wait, return_exceptions=True)
            await turn
            return
        started_wait.cancel()
        await asyncio.gather(started_wait, return_exceptions=True)
        if not self._lifecycle.track_alarm_work(turn):
            await turn

    async def _reschedule_stable_timeout(self, data: dict[str, Any]) -> None:
        incident_value = data.get("incidentId")
        if not isinstance(incident_value, str):
            return
        key = incident_key(incident_value)
        incident = await self.ctx.storage.get(key)
        if not isinstance(incident, dict):
            return
        attempt = int(incident.get("attempt") or 0)
        current_time = now_ms()
        progress = await self.ctx.storage.get(CHAT_RECOVERY_PROGRESS_KEY)
        progress_value = progress if type(progress) is int else 0
        work = progress_value - int(incident.get("workBaseline") or 0)
        reason = None
        if attempt >= int(
            incident.get("maxAttempts") or self.chat_recovery_max_attempts
        ):
            reason = "max_attempts_exceeded"
        elif (
            current_time - int(incident.get("lastProgressAt") or current_time)
            > self.chat_recovery_no_progress_timeout_ms
        ):
            reason = "no_progress_timeout"
        elif work > self.chat_recovery_max_work:
            reason = "work_budget_exceeded"
        if reason is not None:
            await self._exhaust_recovery(
                {"recoveryRootRequestId": data.get("originalRequestId")},
                reason,
                incident=incident,
            )
            return
        updated = {
            **incident,
            "attempt": attempt + 1,
            "status": "scheduled",
            "reason": "stable_timeout_retry",
            "lastAttemptAt": current_time,
        }
        await self.ctx.storage.put(key, updated)
        await self._redefer_recovery(data, "stable-timeout")

    async def _handle_recovery_oom(self, data: dict[str, Any]) -> None:
        incident_value = data.get("incidentId")
        if not isinstance(incident_value, str):
            return
        key = incident_key(incident_value)
        incident = await self.ctx.storage.get(key)
        if not isinstance(incident, dict):
            return
        oom_attempts = int(incident.get("oomAttempts") or 0) + 1
        updated = {
            **incident,
            "oomAttempts": oom_attempts,
            "lastAttemptAt": now_ms(),
            "reason": "out_of_memory"
            if oom_attempts > self.chat_recovery_max_oom_retries
            else "oom_retry",
        }
        await self.ctx.storage.put(key, updated)
        if oom_attempts > self.chat_recovery_max_oom_retries:
            await self._exhaust_recovery(
                {"recoveryRootRequestId": data.get("originalRequestId")},
                "out_of_memory",
                incident=updated,
            )
            return
        await self._redefer_recovery(data, "out-of-memory")

    async def _execute_recovery_attempt(
        self,
        data: dict[str, Any],
        started: asyncio.Event,
    ) -> None:
        incident_value = data.get("incidentId")
        if not isinstance(incident_value, str):
            return
        key = incident_key(incident_value)
        incident = await self.ctx.storage.get(key)
        if not isinstance(incident, dict) or incident.get("status") in (
            "completed",
            "exhausted",
        ):
            return
        if self._awaiting_client_interaction():
            await self._update_recovery_incident(
                incident,
                "skipped",
                "awaiting_client_interaction",
            )
            return

        kind = data.get("kind")
        target_id = data.get("targetAssistantId")
        latest_user_id = data.get("latestUserMessageId")
        leaf = self.messages[-1] if self.messages else None
        if kind == "continue":
            if (
                not isinstance(target_id, str)
                or leaf is None
                or leaf.get("role") != "assistant"
                or leaf.get("id") != target_id
            ):
                await self._update_recovery_incident(
                    incident,
                    "skipped",
                    "conversation_changed",
                )
                return
            message_id = target_id
        else:
            if (
                leaf is None
                or leaf.get("role") != "user"
                or isinstance(latest_user_id, str)
                and leaf.get("id") != latest_user_id
            ):
                await self._update_recovery_incident(
                    incident,
                    "skipped",
                    "conversation_changed",
                )
                return
            message_id = f"msg-{gen_id()}"

        request_id = gen_id()
        if kind == "continue":
            self._continuation_request_ids.add(request_id)
        abort = asyncio.Event()
        self._aborts[request_id] = abort
        body_value = data.get("body")
        body = (
            cast(dict[str, Any], copy.deepcopy(body_value))
            if isinstance(body_value, dict)
            else dict(self._last_body)
        )
        client_tools = data.get("clientTools")
        if not isinstance(client_tools, list):
            client_tools = copy.deepcopy(self._last_client_tools)
        else:
            client_tools = copy.deepcopy(client_tools)
        self._last_body = body
        self._last_client_tools = client_tools
        self._persist_request_context()
        recovery = self._turn_recovery_payload(
            request_id,
            message_id,
            body,
            client_tools,
            continuation=kind == "continue",
            recovery_root_request_id=str(data.get("originalRequestId") or request_id),
        )
        turn_error: str | None = None
        try:

            async def run(context: TurnContext) -> None:
                nonlocal turn_error

                async def provider_turn() -> None:
                    nonlocal turn_error
                    started.set()
                    turn_error = await self._run_turn(
                        None,
                        request_id,
                        "submit-message",
                        body,
                        abort,
                        context,
                        message_id=message_id,
                        transcript=list(self.messages),
                        broadcast_transcript=not self._messages_truncated,
                        continuation=kind == "continue",
                        client_tools=client_tools,
                    )

                await self._run_chat_turn_task(request_id, recovery, provider_turn)

            result = await self._turn_queue.enqueue(request_id, run)
            if result.status == "stale":
                await self._update_recovery_incident(
                    incident,
                    "skipped",
                    "conversation_changed",
                )
            elif turn_error is None:
                await self._update_recovery_incident(incident, "completed")
            else:
                await self._update_recovery_incident(incident, "failed", turn_error)
        finally:
            self._aborts.pop(request_id, None)
            self._terminal_request_ids.discard(request_id)

    async def _exhaust_recovery(
        self,
        recovery: dict[str, Any],
        reason: str,
        *,
        incident: dict[str, Any] | None = None,
        stream_id: str = "",
    ) -> None:
        if incident is not None and incident.get("exhaustionReported") is True:
            return
        request_id = str(
            recovery.get("recoveryRootRequestId") or recovery.get("requestId") or ""
        )
        if stream_id:
            self._resumable.mark_error(stream_id)
        error = (
            "Recovered chat stream exceeded transcript reconstruction limits"
            if reason == "reconstruction_limits_exceeded"
            else self.chat_recovery_terminal_message
        )
        continuation = self._is_continuation_request(request_id)
        self._resumable.record_terminal_error(
            request_id,
            error,
            continuation=continuation,
        )
        self._broadcast_terminal(
            request_id,
            error,
            continuation=continuation,
        )
        context = {
            "incidentId": (incident or {}).get("incidentId"),
            "requestId": request_id,
            "reason": reason,
            "partialMessages": copy.deepcopy(self.messages),
        }
        if incident is not None:
            incident = {**incident, "exhaustionReported": True}
            await self._update_recovery_incident(incident, "exhausted", reason)
        result = self.on_chat_recovery_exhausted(context)
        if inspect.isawaitable(result):
            await result
        self._spawn_reschedule()

    async def _maybe_bump_recovery_progress(self, chunk: ChunkT) -> None:
        chunk_type = chunk.get("type")
        milestone = chunk_type in (
            "text-end",
            "reasoning-end",
            "tool-input-available",
            "tool-output-available",
            "tool-output-error",
            "tool-output-denied",
        )
        delta = chunk_type in ("text-delta", "reasoning-delta", "tool-input-delta")
        now = now_ms()
        if not milestone and (not delta or now - self._last_progress_bump_at < 5_000):
            return
        current = await self.ctx.storage.get(CHAT_RECOVERY_PROGRESS_KEY)
        await self.ctx.storage.put(
            CHAT_RECOVERY_PROGRESS_KEY,
            (current if type(current) is int else 0) + 1,
        )
        self._last_progress_bump_at = now

    def _collect_alarm_deadline(self, now: int, current: int | None) -> int | None:
        deadline = super()._collect_alarm_deadline(now, current)
        stream_deadline = self._resumable.next_cleanup_deadline()
        if deadline is None:
            return stream_deadline
        if stream_deadline is None:
            return deadline
        return min(deadline, stream_deadline)

    async def _alarm_housekeeping(self) -> None:
        await super()._alarm_housekeeping()
        self._resumable.cleanup(now_ms())

    @staticmethod
    def _track_open_block(open_blocks: dict[str, str], chunk: ChunkT) -> None:
        chunk_type = chunk.get("type")
        if not isinstance(chunk_type, str):
            return

        for name in _PART_TYPES:
            if chunk_type == f"{name}-start":
                open_blocks[name] = str(chunk.get("id", _TEXT_ID))
            elif chunk_type == f"{name}-end":
                open_blocks.pop(name, None)

    def _close_open_blocks(
        self,
        stream_id: str,
        request_id: str,
        parts: list[ChunkT],
        open_blocks: dict[str, str],
    ) -> None:
        for name, chunk_id in list(open_blocks.items()):
            part = _find_last_part(parts, name)
            if part is not None:
                part["state"] = "done"
            self._emit(
                stream_id,
                request_id,
                {"type": f"{name}-end", "id": chunk_id},
            )

        open_blocks.clear()

    def _emit(self, stream_id: str, request_id: str, chunk: ChunkT) -> None:
        # Buffer before broadcast, or a reconnecting client misses a chunk that was sent
        # but not yet stored. body is one JSON-encoded chunk and the stored copy is that
        # same string, so replay is byte-identical. Connections mid-resume are excluded
        # because the ACK replay catches them up.
        body = dumps_wire(chunk)
        if not self._resumable.store_chunk(stream_id, body):
            raise StreamStorageUnavailable("stream chunk could not be persisted")
        self._child_agent_tool_runs.capture_chunk(request_id, body)
        self.broadcast_json(
            chat_response(
                ChatMessageType.USE_CHAT_RESPONSE,
                request_id,
                body,
                done=False,
                continuation=self._resumable.active_is_continuation,
            ),
            exclude=self._pending_resume_connections,
        )

    def _is_continuation_request(self, request_id: str) -> bool:
        if request_id in self._continuation_request_ids:
            return True
        stream = self._resumable.latest_for_request(request_id)
        return stream is not None and stream.is_continuation

    def _terminal_frame(
        self,
        request_id: str,
        error: str | None,
        *,
        continuation: bool | None = None,
    ) -> dict[str, Any]:
        # The failure message rides in body and error is a bare flag — see AGENTS.md.
        frame = chat_response(
            ChatMessageType.USE_CHAT_RESPONSE,
            request_id,
            "" if error is None else error,
            done=True,
            continuation=(
                self._is_continuation_request(request_id)
                if continuation is None
                else continuation
            ),
        )

        if error is not None:
            frame["error"] = True

        return frame

    def _broadcast_terminal(
        self,
        request_id: str,
        error: str | None = None,
        *,
        continuation: bool | None = None,
        dedupe: bool = True,
    ) -> bool:
        # Not buffered: replay synthesizes its own terminal, so storing this would
        # append a second one on resume.
        if dedupe and request_id in self._terminal_request_ids:
            return False
        if dedupe:
            self._terminal_request_ids.add(request_id)
        if error is not None:
            self._child_agent_tool_runs.capture_error(request_id, error)
        self.broadcast_json(
            self._terminal_frame(request_id, error, continuation=continuation),
            exclude=self._pending_resume_connections,
        )
        return True

    def _send_terminal(
        self,
        connection: Connection,
        request_id: str,
        error: str | None = None,
        *,
        continuation: bool | None = None,
        dedupe: bool = True,
    ) -> bool:
        # Unicast, for a failure before the stream is registered: no stream exists to
        # buffer against and only the requester is waiting.
        if dedupe and request_id in self._terminal_request_ids:
            return False
        if dedupe:
            self._terminal_request_ids.add(request_id)
        return connection.send_if_open(
            self._terminal_frame(request_id, error, continuation=continuation)
        )

    # -- child agent-tool adapter ---------------------------------------

    async def _cf_start_agent_tool_run(self, input_json: str, run_id: str) -> str:
        await self._ensure_initialized()
        inspection = await self._child_agent_tool_runs.start(input_json, run_id)
        return dumps_wire(inspection.to_wire())

    async def _cf_cancel_agent_tool_run(
        self, run_id: str, reason: str | None = None
    ) -> None:
        await self._ensure_initialized()
        await self._child_agent_tool_runs.cancel(run_id, reason)

    async def _cf_inspect_agent_tool_run(self, run_id: str) -> str | None:
        await self._ensure_initialized()
        inspection = await self._child_agent_tool_runs.inspect(run_id)
        return None if inspection is None else dumps_wire(inspection.to_wire())

    async def _cf_get_agent_tool_chunks(
        self,
        run_id: str,
        after_sequence: int = -1,
        limit: int = 100,
    ) -> str:
        await self._ensure_initialized()
        chunks = self._child_agent_tool_runs.chunks(run_id, after_sequence, limit)
        return dumps_wire([chunk.to_wire() for chunk in chunks])

    @staticmethod
    def _latest_assistant_text(messages: list[MessageT]) -> str | None:
        for message in reversed(messages):
            if message.get("role") != "assistant":
                continue
            text = "".join(
                str(part.get("text") or "")
                for part in message.get("parts", [])
                if part.get("type") == "text"
            )
            if text:
                return text
        return None
