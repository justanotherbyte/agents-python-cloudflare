from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from .agent_tool_protocol import (
    CHILD_AGENT_TOOL_CHUNK_PAGE_SIZE,
    ChildAgentToolStub,
    agent_tool_aborted_event,
    agent_tool_chunk_event,
    agent_tool_error_event,
    agent_tool_event_frame,
    agent_tool_finished_event,
    agent_tool_interrupted_event,
    agent_tool_started_event,
)
from .utils import MISSING, dumps_wire, error_message, gen_id, now_ms

AgentToolStatus = Literal["completed", "error", "aborted", "interrupted"]
_SqlFn = Callable[..., list[dict[str, Any]]]

_HARD_TERMINAL = frozenset({"completed", "error", "aborted"})
_REPLAY_TERMINAL = _HARD_TERMINAL | {"interrupted"}
_RUN_SELECT = (
    "SELECT run_id, parent_tool_call_id, agent_type, input_preview, status, "
    "summary, output_json, error_message, interrupted_reason, "
    "child_still_running, display_metadata, display_order, chunks_mirrored, "
    "started_at, completed_at FROM cf_agent_tool_runs"
)


@dataclass(frozen=True)
class AgentToolResult:
    run_id: str
    agent_type: str
    status: AgentToolStatus
    output: Any = None
    summary: str | None = None
    error: str | None = None
    reason: str | None = None
    child_still_running: bool | None = None


def _parse_json(value: Any) -> Any:
    if value is None:
        return None
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return value


def _parse_run_json(value: Any) -> Any:
    if value is None:
        return MISSING
    return _parse_json(value)


def _default_input_preview(input: Any) -> Any:
    if isinstance(input, str):
        return input[:500]
    if input is None:
        return None
    try:
        preview = dumps_wire(input)
    except Exception:  # noqa: BLE001
        preview = str(input)
    return preview if len(preview) <= 500 else f"{preview[:497]}..."


def _result_from_row(row: dict[str, Any]) -> AgentToolResult:
    child = row["child_still_running"]
    return AgentToolResult(
        run_id=row["run_id"],
        agent_type=row["agent_type"],
        status=row["status"],
        output=_parse_json(row["output_json"]),
        summary=row["summary"],
        error=row["error_message"],
        reason=row["interrupted_reason"],
        child_still_running=None if child is None else child != 0,
    )


def _result_from_inspection(
    inspection: dict[str, Any], agent_type: str
) -> AgentToolResult:
    status = inspection.get("status")
    if status not in _HARD_TERMINAL:
        raise ValueError(f"child agent-tool run is not terminal: {status!r}")
    return AgentToolResult(
        run_id=inspection["runId"],
        agent_type=agent_type,
        status=status,
        output=inspection.get("output"),
        summary=inspection.get("summary"),
        error=inspection.get("error"),
    )


def _terminal_event(result: AgentToolResult) -> dict[str, Any]:
    if result.status == "completed":
        return agent_tool_finished_event(result.run_id, result.summary or "")
    if result.status == "aborted":
        return agent_tool_aborted_event(result.run_id, result.error)
    if result.status == "interrupted":
        return agent_tool_interrupted_event(
            result.run_id,
            result.error or "Agent tool run was interrupted",
            reason=result.reason,
            child_still_running=result.child_still_running,
        )
    return agent_tool_error_event(
        result.run_id, result.error or "Agent tool run failed"
    )


class AgentToolConnection(Protocol):
    def send_json(self, data: dict[str, Any]) -> None: ...


class AgentToolRuns:
    """Awaited sub-agent orchestration and reconnect-safe event replay."""

    def __init__(
        self,
        *,
        sql: _SqlFn,
        publish: Callable[[dict[str, Any]], None],
        resolve_sub_agent: Callable[[str, str], Awaitable[ChildAgentToolStub]],
        max_concurrent: Callable[[], int],  # TODO: does this need to be a callable?
        recovery_grace_ms: Callable[[], int],  # TODO: does this need to be a callable?
    ) -> None:
        self._sql = sql
        self._publish = publish
        self._resolve_sub_agent = resolve_sub_agent
        self._max_concurrent = max_concurrent
        self._recovery_grace_ms = recovery_grace_ms
        self._active_agent_tool_ids: set[str] = set()
        self._agent_tool_waiters: dict[str, asyncio.Event] = {}

    # -- schema ---------------------------------------------------------

    def prepare(self) -> None:
        self._sql("""
        CREATE TABLE IF NOT EXISTS cf_agent_tool_chunks (
            run_id TEXT NOT NULL,
            sequence INTEGER NOT NULL,
            body TEXT NOT NULL,
            PRIMARY KEY (run_id, sequence)
        )
        """)
        self._sql("""
        CREATE TABLE IF NOT EXISTS cf_agent_tool_terminals (
            run_id TEXT PRIMARY KEY,
            sequence INTEGER NOT NULL
        )
        """)
        columns = {
            row["name"] for row in self._sql("PRAGMA table_info(cf_agent_tool_runs)")
        }
        if "chunks_mirrored" not in columns:
            self._sql(
                "ALTER TABLE cf_agent_tool_runs ADD COLUMN chunks_mirrored "
                "INTEGER NOT NULL DEFAULT 0"
            )

    # -- reconnect replay ----------------------------------------------

    def replay(self, connection: AgentToolConnection) -> None:
        try:
            rows = self._sql(_RUN_SELECT + " ORDER BY started_at ASC")
        except Exception:  # noqa: BLE001
            return

        cutoff = now_ms() - self._recovery_grace_ms()
        for row in rows:
            try:
                if (
                    row["status"] not in _REPLAY_TERMINAL
                    and row["started_at"] <= cutoff
                    and row["run_id"] not in self._active_agent_tool_ids
                ):
                    self._interrupt_stale_agent_tool_run(row)
                self._replay_one_agent_tool_run(connection, row)
            except Exception:  # noqa: BLE001, S112
                # One damaged run or a partial migration must not fail the handshake.
                continue

    def _interrupt_stale_agent_tool_run(self, row: dict[str, Any]) -> None:
        message = "Agent tool run exceeded its recovery grace period"
        self._sql(
            "UPDATE cf_agent_tool_runs SET status = 'interrupted', "
            "error_message = ?, interrupted_reason = 'recovery-deadline', "
            "child_still_running = NULL, completed_at = ? WHERE run_id = ? "
            "AND status NOT IN ('completed', 'error', 'aborted')",
            message,
            now_ms(),
            row["run_id"],
        )
        row["status"] = "interrupted"
        row["error_message"] = message
        row["interrupted_reason"] = "recovery-deadline"
        row["child_still_running"] = None

    def _replay_one_agent_tool_run(
        self, connection: AgentToolConnection, row: dict[str, Any]
    ) -> None:
        parent = row["parent_tool_call_id"]
        started = agent_tool_started_event(
            row["run_id"],
            row["agent_type"],
            row["display_order"],
            input_preview=_parse_run_json(row["input_preview"]),
            display=_parse_run_json(row["display_metadata"]),
        )
        connection.send_json(
            agent_tool_event_frame(started, 0, parent_tool_call_id=parent, replay=True)
        )

        sequence = 1
        try:
            chunks = self._sql(
                "SELECT sequence, body FROM cf_agent_tool_chunks "
                "WHERE run_id = ? ORDER BY sequence ASC",
                row["run_id"],
            )
        except Exception:  # noqa: BLE001
            chunks = []

        for chunk in chunks:
            sequence = chunk["sequence"]
            connection.send_json(
                agent_tool_event_frame(
                    agent_tool_chunk_event(row["run_id"], chunk["body"]),
                    sequence,
                    parent_tool_call_id=parent,
                    replay=True,
                )
            )
            sequence += 1

        if row["status"] in _REPLAY_TERMINAL:
            sequence = self._replay_terminal_sequence(row["run_id"], sequence)
            connection.send_json(
                agent_tool_event_frame(
                    _terminal_event(_result_from_row(row)),
                    sequence,
                    parent_tool_call_id=parent,
                    replay=True,
                )
            )

    # -- awaited execution ---------------------------------------------

    async def run_agent_tool(
        self,
        cls: Any,
        *,
        input: Any,
        run_id: str | None = None,
        parent_tool_call_id: str | None = None,
        display_order: int = 0,
        input_preview: Any = MISSING,
        display: Any = MISSING,
        abort: asyncio.Event | None = None,
    ) -> AgentToolResult:
        """Run a chat-capable child agent and await its terminal result.

        The child is named by ``run_id``, making retries idempotent. This first
        implementation intentionally collects the child stream after its active RPC
        returns; live concurrent tailing is enabled only after deployed runtime proof.
        """
        resolved_id = gen_id() if run_id is None else run_id
        if not resolved_id.strip():
            raise ValueError("run_id must not be blank")
        if not isinstance(cls, type):
            raise TypeError("cls must be an agent class")

        while (active := self._agent_tool_waiters.get(resolved_id)) is not None:
            await active.wait()
            row = self._read_agent_tool_run(resolved_id)
            if row is None:
                raise RuntimeError(f"agent tool run {resolved_id!r} disappeared")
            if row["status"] in _HARD_TERMINAL and not (
                row["status"] == "completed" and row["chunks_mirrored"] == 0
            ):
                return _result_from_row(row)

        existing = self._read_agent_tool_run(resolved_id)
        if existing is not None:
            if existing["status"] in _HARD_TERMINAL and not (
                existing["status"] == "completed" and existing["chunks_mirrored"] == 0
            ):
                return _result_from_row(existing)
            if (
                existing["status"] not in _HARD_TERMINAL
                and abort is not None
                and abort.is_set()
            ):
                result = AgentToolResult(
                    resolved_id, existing["agent_type"], "aborted", error="cancelled"
                )
                return self._finish_agent_tool_run(
                    result,
                    existing["parent_tool_call_id"],
                    self._next_parent_sequence(resolved_id),
                )
            input_json = (
                None if existing["status"] in _HARD_TERMINAL else dumps_wire(input)
            )
            waiter = asyncio.Event()
            self._agent_tool_waiters[resolved_id] = waiter
            self._active_agent_tool_ids.add(resolved_id)
            try:
                return await self._resume_agent_tool_run(existing, input_json, abort)
            except asyncio.CancelledError:
                self._finish_cancelled_agent_tool_run(existing)
                raise
            finally:
                self._active_agent_tool_ids.discard(resolved_id)
                waiter.set()
                self._agent_tool_waiters.pop(resolved_id, None)

        agent_type = cls.__name__
        started_at = now_ms()
        preview = (
            _default_input_preview(input) if input_preview is MISSING else input_preview
        )
        preview_json = None if preview is MISSING else dumps_wire(preview)
        display_json = None if display is MISSING else dumps_wire(display)
        input_json = dumps_wire(input)

        self._sweep_stale_agent_tool_runs()
        max_concurrent = self._max_concurrent()
        if self._active_agent_tool_run_count() >= max_concurrent:
            result = AgentToolResult(
                resolved_id,
                agent_type,
                "error",
                error=(f"max_concurrent_agent_tools ({max_concurrent}) exceeded"),
            )
            self._insert_agent_tool_run(
                result,
                parent_tool_call_id,
                preview_json,
                display_json,
                display_order,
                started_at,
            )
            self._broadcast_started(
                result, parent_tool_call_id, preview, display_order, display
            )
            self._broadcast_result(result, parent_tool_call_id, 1)
            return result

        self._sql(
            "INSERT INTO cf_agent_tool_runs (run_id, parent_tool_call_id, "
            "agent_type, input_preview, status, display_metadata, display_order, "
            "started_at) VALUES (?, ?, ?, ?, 'starting', ?, ?, ?)",
            resolved_id,
            parent_tool_call_id,
            agent_type,
            preview_json,
            display_json,
            display_order,
            started_at,
        )
        initial = AgentToolResult(resolved_id, agent_type, "interrupted")
        self._broadcast_started(
            initial, parent_tool_call_id, preview, display_order, display
        )

        waiter = asyncio.Event()
        self._agent_tool_waiters[resolved_id] = waiter
        self._active_agent_tool_ids.add(resolved_id)
        child = None
        try:
            if abort is not None and abort.is_set():
                result = AgentToolResult(
                    resolved_id, agent_type, "aborted", error="cancelled"
                )
                return self._finish_agent_tool_run(result, parent_tool_call_id, 1)

            self._sql(
                "UPDATE cf_agent_tool_runs SET status = 'running' WHERE run_id = ?",
                resolved_id,
            )
            child = await self._resolve_sub_agent(
                agent_type,
                resolved_id,
            )
            inspection_json = await child._cf_start_agent_tool_run(
                input_json, resolved_id
            )
            sequence = await self._mirror_child_chunks(
                child, resolved_id, parent_tool_call_id
            )

            inspection = json.loads(inspection_json)
            result = _result_from_inspection(inspection, agent_type)
            if abort is not None and abort.is_set() and result.status == "completed":
                result = AgentToolResult(
                    resolved_id, agent_type, "aborted", error="cancelled"
                )
            return self._finish_agent_tool_run(result, parent_tool_call_id, sequence)
        except Exception as exc:  # noqa: BLE001
            if child is not None:
                try:
                    recovered = await self._recover_child_terminal(
                        child,
                        resolved_id,
                        agent_type,
                        parent_tool_call_id,
                        abort,
                    )
                except asyncio.CancelledError:
                    self._finish_cancelled_agent_tool_run(
                        {
                            "run_id": resolved_id,
                            "agent_type": agent_type,
                            "parent_tool_call_id": parent_tool_call_id,
                        },
                        child_still_running=True,
                    )
                    raise
                if recovered is not None:
                    return recovered
                if abort is not None and abort.is_set():
                    result = AgentToolResult(
                        resolved_id, agent_type, "aborted", error="cancelled"
                    )
                    return self._finish_agent_tool_run(
                        result,
                        parent_tool_call_id,
                        self._next_parent_sequence(resolved_id),
                    )
                result = AgentToolResult(
                    resolved_id,
                    agent_type,
                    "interrupted",
                    error=error_message(exc),
                    reason="child-rpc-failed",
                )
                sequence = self._next_parent_sequence(resolved_id)
                return self._finish_agent_tool_run(
                    result, parent_tool_call_id, sequence
                )
            result = AgentToolResult(
                resolved_id, agent_type, "error", error=error_message(exc)
            )
            sequence = self._next_parent_sequence(resolved_id)
            return self._finish_agent_tool_run(result, parent_tool_call_id, sequence)
        except asyncio.CancelledError:
            self._finish_cancelled_agent_tool_run(
                {
                    "run_id": resolved_id,
                    "agent_type": agent_type,
                    "parent_tool_call_id": parent_tool_call_id,
                },
                child_still_running=child is not None,
            )
            raise
        finally:
            self._active_agent_tool_ids.discard(resolved_id)
            waiter.set()
            self._agent_tool_waiters.pop(resolved_id, None)

    async def _resume_agent_tool_run(
        self,
        row: dict[str, Any],
        input_json: str | None,
        abort: asyncio.Event | None,
    ) -> AgentToolResult:
        try:
            child = await self._resolve_sub_agent(
                row["agent_type"],
                row["run_id"],
            )
            inspection_json = await child._cf_inspect_agent_tool_run(row["run_id"])
            inspection = (
                None if inspection_json is None else json.loads(inspection_json)
            )
        except Exception:  # noqa: BLE001
            if row["status"] in _HARD_TERMINAL:
                return _result_from_row(row)
            if abort is not None and abort.is_set():
                return self._finish_agent_tool_run(
                    AgentToolResult(
                        row["run_id"],
                        row["agent_type"],
                        "aborted",
                        error="cancelled",
                    ),
                    row["parent_tool_call_id"],
                    self._next_parent_sequence(row["run_id"]),
                )
            result = AgentToolResult(
                row["run_id"],
                row["agent_type"],
                "interrupted",
                error="Child inspection failed; retry the same run_id",
                reason="inspect-failed",
            )
            return self._finish_agent_tool_run(
                result,
                row["parent_tool_call_id"],
                self._next_parent_sequence(row["run_id"]),
            )

        if inspection is not None and inspection.get("status") in _HARD_TERMINAL:
            try:
                sequence = await self._mirror_child_chunks(
                    child, row["run_id"], row["parent_tool_call_id"]
                )
            except Exception as exc:  # noqa: BLE001
                if row["status"] in _HARD_TERMINAL:
                    return _result_from_row(row)
                if abort is not None and abort.is_set():
                    return self._finish_agent_tool_run(
                        AgentToolResult(
                            row["run_id"],
                            row["agent_type"],
                            "aborted",
                            error="cancelled",
                        ),
                        row["parent_tool_call_id"],
                        self._next_parent_sequence(row["run_id"]),
                    )
                return self._finish_agent_tool_run(
                    AgentToolResult(
                        row["run_id"],
                        row["agent_type"],
                        "interrupted",
                        error=error_message(exc),
                        reason="child-rpc-failed",
                    ),
                    row["parent_tool_call_id"],
                    self._next_parent_sequence(row["run_id"]),
                )
            result = _result_from_inspection(inspection, row["agent_type"])
            if row["status"] not in _HARD_TERMINAL:
                result = self._apply_agent_tool_abort(result, abort)
            return self._finish_agent_tool_run(
                result, row["parent_tool_call_id"], sequence
            )

        if row["status"] in _HARD_TERMINAL:
            return _result_from_row(row)

        if inspection is None:
            try:
                if input_json is None:
                    raise RuntimeError("agent tool input is unavailable for restart")
                inspection_json = await child._cf_start_agent_tool_run(
                    input_json, row["run_id"]
                )
                sequence = await self._mirror_child_chunks(
                    child, row["run_id"], row["parent_tool_call_id"]
                )
                inspection = json.loads(inspection_json)
                result = _result_from_inspection(inspection, row["agent_type"])
                result = self._apply_agent_tool_abort(result, abort)
                return self._finish_agent_tool_run(
                    result, row["parent_tool_call_id"], sequence
                )
            except Exception as exc:  # noqa: BLE001
                recovered = await self._recover_child_terminal(
                    child,
                    row["run_id"],
                    row["agent_type"],
                    row["parent_tool_call_id"],
                    abort,
                )
                if recovered is not None:
                    return recovered
                if abort is not None and abort.is_set():
                    return self._finish_agent_tool_run(
                        AgentToolResult(
                            row["run_id"],
                            row["agent_type"],
                            "aborted",
                            error="cancelled",
                        ),
                        row["parent_tool_call_id"],
                        self._next_parent_sequence(row["run_id"]),
                    )
                return self._finish_agent_tool_run(
                    AgentToolResult(
                        row["run_id"],
                        row["agent_type"],
                        "interrupted",
                        error=error_message(exc),
                        reason="child-rpc-failed",
                    ),
                    row["parent_tool_call_id"],
                    self._next_parent_sequence(row["run_id"]),
                )

        if abort is not None and abort.is_set():
            return self._finish_agent_tool_run(
                AgentToolResult(
                    row["run_id"], row["agent_type"], "aborted", error="cancelled"
                ),
                row["parent_tool_call_id"],
                self._next_parent_sequence(row["run_id"]),
            )
        result = AgentToolResult(
            row["run_id"],
            row["agent_type"],
            "interrupted",
            error="Child is still running but live tailing is unavailable",
            reason="not-tailable",
            child_still_running=True,
        )
        return self._finish_agent_tool_run(
            result,
            row["parent_tool_call_id"],
            self._next_parent_sequence(row["run_id"]),
        )

    async def _recover_child_terminal(
        self,
        child: Any,
        run_id: str,
        agent_type: str,
        parent_tool_call_id: str | None,
        abort: asyncio.Event | None,
    ) -> AgentToolResult | None:
        try:
            inspection_json = await child._cf_inspect_agent_tool_run(run_id)
            if inspection_json is None:
                return None
            inspection = json.loads(inspection_json)
            if inspection.get("status") not in _HARD_TERMINAL:
                return None
            sequence = await self._mirror_child_chunks(
                child, run_id, parent_tool_call_id
            )
            result = _result_from_inspection(inspection, agent_type)
            result = self._apply_agent_tool_abort(result, abort)
            return self._finish_agent_tool_run(result, parent_tool_call_id, sequence)
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _apply_agent_tool_abort(
        result: AgentToolResult, abort: asyncio.Event | None
    ) -> AgentToolResult:
        if abort is not None and abort.is_set() and result.status == "completed":
            return AgentToolResult(
                result.run_id, result.agent_type, "aborted", error="cancelled"
            )
        return result

    def _finish_cancelled_agent_tool_run(
        self, row: dict[str, Any], *, child_still_running: bool | None = None
    ) -> None:
        current = self._read_agent_tool_run(row["run_id"])
        if current is None or current["status"] in _HARD_TERMINAL:
            return
        self._finish_agent_tool_run(
            AgentToolResult(
                row["run_id"],
                row["agent_type"],
                "interrupted",
                error="Agent tool owner task was cancelled",
                reason="parent-task-cancelled",
                child_still_running=child_still_running,
            ),
            row["parent_tool_call_id"],
            self._next_parent_sequence(row["run_id"]),
        )

    async def _mirror_child_chunks(
        self, child: Any, run_id: str, parent_tool_call_id: str | None
    ) -> int:
        existing = self._sql(
            "SELECT COUNT(*) AS count, COALESCE(MAX(sequence), 0) AS sequence "
            "FROM cf_agent_tool_chunks WHERE run_id = ?",
            run_id,
        )[0]
        terminal_rows = self._sql(
            "SELECT sequence FROM cf_agent_tool_terminals WHERE run_id = ?",
            run_id,
        )
        high_water = existing["sequence"]
        if terminal_rows:
            high_water = max(high_water, terminal_rows[0]["sequence"])
        child_sequences = set()
        sequence = high_water + 1
        child_cursor = -1
        child_position = 0
        while True:
            chunks_json = await child._cf_get_agent_tool_chunks(
                run_id,
                child_cursor,
                CHILD_AGENT_TOOL_CHUNK_PAGE_SIZE,
            )
            chunks = json.loads(chunks_json)
            if not chunks:
                break
            for chunk in chunks:
                child_sequence = int(chunk["sequence"])
                child_cursor = max(child_cursor, child_sequence)
                if child_sequence in child_sequences:
                    continue
                child_sequences.add(child_sequence)
                if child_position < existing["count"]:
                    child_position += 1
                    continue
                body = chunk["body"]
                inserted = self._sql(
                    "INSERT OR IGNORE INTO cf_agent_tool_chunks "
                    "(run_id, sequence, body) VALUES (?, ?, ?) RETURNING sequence",
                    run_id,
                    sequence,
                    body,
                )
                if inserted:
                    self._publish(
                        agent_tool_event_frame(
                            agent_tool_chunk_event(run_id, body),
                            sequence,
                            parent_tool_call_id=parent_tool_call_id,
                        )
                    )
                sequence += 1
                child_position += 1
            if len(chunks) < CHILD_AGENT_TOOL_CHUNK_PAGE_SIZE:
                break
        self._sql(
            "UPDATE cf_agent_tool_runs SET chunks_mirrored = 1 WHERE run_id = ?",
            run_id,
        )
        return sequence

    def _finish_agent_tool_run(
        self,
        result: AgentToolResult,
        parent_tool_call_id: str | None,
        sequence: int,
    ) -> AgentToolResult:
        output_json = (
            dumps_wire(result.output) if result.status == "completed" else None
        )
        self._sql(
            "UPDATE cf_agent_tool_runs SET status = ?, summary = ?, "
            "output_json = ?, error_message = ?, interrupted_reason = ?, "
            "child_still_running = ?, completed_at = ? WHERE run_id = ? "
            "AND status NOT IN ('completed', 'error', 'aborted')",
            result.status,
            result.summary,
            output_json,
            result.error,
            result.reason,
            None
            if result.child_still_running is None
            else int(result.child_still_running),
            now_ms(),
            result.run_id,
        )
        if result.status == "completed":
            self._sql(
                "UPDATE cf_agent_tool_runs SET summary = COALESCE(summary, ?), "
                "output_json = COALESCE(output_json, ?), "
                "completed_at = COALESCE(completed_at, ?) WHERE run_id = ? "
                "AND status = 'completed'",
                result.summary,
                output_json,
                now_ms(),
                result.run_id,
            )
        sequence = self._record_terminal_sequence(result.run_id, sequence)
        self._broadcast_result(result, parent_tool_call_id, sequence)
        return result

    def _broadcast_started(
        self,
        result: AgentToolResult,
        parent_tool_call_id: str | None,
        input_preview: Any,
        display_order: int,
        display: Any,
    ) -> None:
        self._publish(
            agent_tool_event_frame(
                agent_tool_started_event(
                    result.run_id,
                    result.agent_type,
                    display_order,
                    input_preview=input_preview,
                    display=display,
                ),
                0,
                parent_tool_call_id=parent_tool_call_id,
            )
        )

    def _broadcast_result(
        self,
        result: AgentToolResult,
        parent_tool_call_id: str | None,
        sequence: int,
    ) -> None:
        self._publish(
            agent_tool_event_frame(
                _terminal_event(result),
                sequence,
                parent_tool_call_id=parent_tool_call_id,
            )
        )

    def _insert_agent_tool_run(
        self,
        result: AgentToolResult,
        parent_tool_call_id: str | None,
        preview_json: str | None,
        display_json: str | None,
        display_order: int,
        started_at: int,
    ) -> None:
        self._sql(
            "INSERT INTO cf_agent_tool_runs (run_id, parent_tool_call_id, "
            "agent_type, input_preview, status, error_message, display_metadata, "
            "display_order, started_at, completed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            result.run_id,
            parent_tool_call_id,
            result.agent_type,
            preview_json,
            result.status,
            result.error,
            display_json,
            display_order,
            started_at,
            now_ms(),
        )

    def _read_agent_tool_run(self, run_id: str) -> dict[str, Any] | None:
        rows = self._sql(
            _RUN_SELECT + " WHERE run_id = ?",
            run_id,
        )
        return rows[0] if rows else None

    def _active_agent_tool_run_count(self) -> int:
        rows = self._sql(
            "SELECT COUNT(*) AS count FROM cf_agent_tool_runs "
            "WHERE status NOT IN ('completed', 'error', 'aborted', 'interrupted')"
        )
        return rows[0]["count"]

    def _sweep_stale_agent_tool_runs(self) -> None:
        cutoff = now_ms() - self._recovery_grace_ms()
        rows = self._sql(
            _RUN_SELECT
            + " WHERE status IN ('starting', 'running') AND started_at <= ?",
            cutoff,
        )
        for row in rows:
            if row["run_id"] not in self._active_agent_tool_ids:
                self._interrupt_stale_agent_tool_run(row)

    def _next_parent_sequence(self, run_id: str) -> int:
        rows = self._sql(
            "SELECT COALESCE(MAX(sequence), 0) + 1 AS sequence "
            "FROM cf_agent_tool_chunks WHERE run_id = ?",
            run_id,
        )
        return rows[0]["sequence"]

    def _record_terminal_sequence(self, run_id: str, proposed: int) -> int:
        rows = self._sql(
            "SELECT sequence FROM cf_agent_tool_terminals WHERE run_id = ?",
            run_id,
        )
        sequence = proposed if not rows else max(proposed, rows[0]["sequence"] + 1)
        self._sql(
            "INSERT INTO cf_agent_tool_terminals (run_id, sequence) VALUES (?, ?) "
            "ON CONFLICT (run_id) DO UPDATE SET sequence = excluded.sequence",
            run_id,
            sequence,
        )
        return sequence

    def _replay_terminal_sequence(self, run_id: str, proposed: int) -> int:
        rows = self._sql(
            "SELECT sequence FROM cf_agent_tool_terminals WHERE run_id = ?",
            run_id,
        )
        if rows:
            sequence = max(proposed, rows[0]["sequence"])
            if sequence != rows[0]["sequence"]:
                self._sql(
                    "UPDATE cf_agent_tool_terminals SET sequence = ? WHERE run_id = ?",
                    sequence,
                    run_id,
                )
            return sequence
        self._sql(
            "INSERT INTO cf_agent_tool_terminals (run_id, sequence) VALUES (?, ?)",
            run_id,
            proposed,
        )
        return proposed
