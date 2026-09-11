from __future__ import annotations

import asyncio
import inspect
import logging
import math
import re
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Iterator,
    Mapping,
    Sequence,
)
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal, NotRequired, TypedDict, cast

from ._session_compaction import (
    _COMPACTION_PREFIX,
    _StoredCompaction,
    _build_summary_prompt,
    _overlay_message,
    _plan_overlays,
    _prepare_compaction_input,
)
from ._session_json import (
    clone_session_json,
    dumps_session_json,
    session_message_from_value,
)
from ._session_store import _SessionPathRow, _SessionStore
from .core.schema import parse_schema_version
from .core.utils import MISSING
from .lifecycle import LifecycleCapability


_LOGGER = logging.getLogger(__name__)


class SessionMessagePart(TypedDict):
    type: str
    text: NotRequired[str]
    reasoning: NotRequired[str]
    toolCallId: NotRequired[str]
    toolName: NotRequired[str]
    input: NotRequired[object]
    output: NotRequired[object]
    state: NotRequired[str]
    result: NotRequired[object]
    mediaType: NotRequired[str]
    url: NotRequired[str]
    filename: NotRequired[str]


class SessionMessage(TypedDict):
    id: str
    role: str
    parts: list[SessionMessagePart]
    metadata: NotRequired[object]
    createdAt: NotRequired[str | datetime]


class _SessionAppendEvent(TypedDict):
    type: Literal["append"]
    sessionId: str
    message: SessionMessage
    parentId: NotRequired[str | None]
    inserted: bool


class _SessionUpdateEvent(TypedDict):
    type: Literal["update"]
    sessionId: str
    message: SessionMessage


class _SessionDeleteEvent(TypedDict):
    type: Literal["delete"]
    sessionId: str
    messageIds: list[str]


class _SessionClearEvent(TypedDict):
    type: Literal["clear"]
    sessionId: str


class _SessionCompactEvent(TypedDict):
    type: Literal["compact"]
    sessionId: str


class _SessionErrorPayload(TypedDict):
    sessionId: str
    event: str
    error: str


class _SessionCompactionErrorPayload(TypedDict):
    sessionId: str
    error: str


class _SessionCompactedPayload(TypedDict):
    sessionId: str
    compactionId: str


class _SessionMigrationIncompletePayload(TypedDict):
    table: str
    source: int
    copied: int


class _SessionMessageEventPayload(TypedDict):
    sessionId: str
    messageId: str


class _SessionAppendPayload(_SessionMessageEventPayload):
    tokenEstimate: int


class _SessionDeletePayload(TypedDict):
    sessionId: str
    count: int


class _SessionClearPayload(TypedDict):
    sessionId: str


type SessionChangeEvent = (
    _SessionAppendEvent
    | _SessionUpdateEvent
    | _SessionDeleteEvent
    | _SessionClearEvent
    | _SessionCompactEvent
)
type SessionChangeListener = Callable[[SessionChangeEvent], object | Awaitable[object]]


@dataclass(frozen=True)
class AppendResult:
    """Report whether an append inserted and return the stored message."""

    inserted: bool
    message: SessionMessage


@dataclass(frozen=True)
class SessionRowStat:
    """Describe one active-path row without hydrating its content."""

    id: str
    role: str
    bytes: int
    token_estimate: int


@dataclass(frozen=True)
class RecentHistoryResult:
    """Return a newest-first-budgeted path in root-to-leaf order."""

    messages: list[SessionMessage]
    truncated: bool
    total_content_bytes: int


@dataclass(frozen=True, kw_only=True)
class SearchOptions:
    """Limit full-text search results for one session."""

    limit: int | float = 20


@dataclass(frozen=True)
class SearchResult:
    """Return the indexed fields for one matching message."""

    id: str
    role: str
    content: str


@dataclass(frozen=True)
class CompactResult:
    """Describe the raw message range replaced by a summary overlay."""

    from_message_id: str
    to_message_id: str
    summary: str


type CompactionFunction = Callable[
    [list[SessionMessage]],
    CompactResult | None | Awaitable[CompactResult | None],
]


@dataclass(frozen=True, kw_only=True)
class CompactOptions:
    """Configure the reference middle-summary compaction function."""

    summarize: Callable[[str], str | Awaitable[str]]
    keep_recent_tokens: int | float = 20_000

    def __post_init__(self) -> None:
        if not callable(self.summarize):
            raise TypeError("summarize must be callable")
        _finite_number(self.keep_recent_tokens, "keep_recent_tokens")


@dataclass(frozen=True)
class StoredCompaction:
    """Describe one durable compaction overlay."""

    id: str
    summary: str
    from_message_id: str
    to_message_id: str
    created_at: str


@dataclass(frozen=True, kw_only=True)
class SessionsOptions:
    """Configure metadata keys removed from client-source writes."""

    reserved_metadata_keys: Sequence[str] = ()

    def __post_init__(self) -> None:
        keys = tuple(self.reserved_metadata_keys)
        if any(type(key) is not str for key in keys):
            raise TypeError("reserved metadata keys must be strings")
        object.__setattr__(self, "reserved_metadata_keys", keys)


@dataclass(frozen=True, kw_only=True)
class WriteOptions:
    """Select whether a write comes from a trusted server or a client."""

    source: Literal["client", "server"] = "server"


@dataclass(frozen=True, kw_only=True)
class AppendOptions:
    """Configure write trust and omitted, root, or explicit parent behavior."""

    source: Literal["client", "server"] = "server"
    parent_id: str | None | object = MISSING


@dataclass(frozen=True, kw_only=True)
class HistoryReadOptions:
    """Select a history leaf and an optional cooperative abort signal."""

    leaf_id: str | None = None
    signal: object | None = None


@dataclass(frozen=True, kw_only=True)
class HistoryBatchReadOptions:
    """Bound streamed history batches by message count and serialized bytes."""

    leaf_id: str | None = None
    signal: object | None = None
    batch_size: int | float = 50
    max_batch_bytes: int | float = 4 * 1024 * 1024


class SessionStorageError(RuntimeError):
    """Raised when persisted Sessions storage cannot support the public API."""


class Sessions(LifecycleCapability):
    """Own durable session trees; startup storage failures remain retryable."""

    capability_id = "sessions"

    def __init__(self, options: SessionsOptions | None = None) -> None:
        self._options = SessionsOptions() if options is None else options
        self._handles: dict[str, Session] = {}
        self._listeners: dict[SessionChangeListener, None] = {}
        self._store: _SessionStore | None = None
        self._startup_error: SessionStorageError | None = None

    async def on_start(self) -> None:
        version = parse_schema_version(
            await self.lifecycle.storage.get("cf_agents:sessions_schema_version")
        )
        if version > 1:
            return
        store = self._session_store()
        if not store.schema_is_compatible():
            self._startup_error = SessionStorageError(
                "persisted Sessions tables do not match schema version 1"
            )
            await self.lifecycle.events.emit(
                "session:error",
                _SessionErrorPayload(
                    sessionId="",
                    event="startup",
                    error=str(self._startup_error),
                ),
            )
            return
        store.prepare()
        if version >= 1:
            return
        incomplete = store.migrate_legacy()
        for report in incomplete:
            await self.lifecycle.events.emit(
                "session:migration:incomplete",
                _SessionMigrationIncompletePayload(
                    table=report.table,
                    source=report.source,
                    copied=report.copied,
                ),
            )
        if not incomplete:
            await self.lifecycle.storage.put("cf_agents:sessions_schema_version", 1)

    def session(self, session_id: str = "") -> Session:
        """Return the activation-cached handle for a session ID."""
        if type(session_id) is not str:
            raise TypeError("session_id must be a string")
        handle = self._handles.get(session_id)
        if handle is None:
            handle = Session(session_id, self)
            self._handles[session_id] = handle
        return handle

    def subscribe(self, listener: SessionChangeListener) -> Callable[[], None]:
        """Subscribe once to ordered post-commit changes and return an unsubscriber."""
        self._listeners[listener] = None

        def unsubscribe() -> None:
            self._listeners.pop(listener, None)

        return unsubscribe

    async def _ready(self) -> None:
        await self.lifecycle.ready()
        if self._startup_error is not None:
            raise self._startup_error

    def _session_store(self) -> _SessionStore:
        if self._store is None:
            self._store = _SessionStore(
                self.lifecycle.sql,
                self.lifecycle.storage.transaction_sync,
            )
        return self._store

    async def _emit(self, event_type: str, payload: object) -> None:
        await self.lifecycle.events.emit(event_type, payload)

    async def _notify(self, event: SessionChangeEvent) -> None:
        session_id = event["sessionId"]
        event_type = event["type"]
        for listener in tuple(self._listeners):
            try:
                delivered = cast(SessionChangeEvent, clone_session_json(event))
                result = listener(delivered)
                if inspect.isawaitable(result):
                    await result
            except Exception as error:
                await self._emit(
                    "session:error",
                    _SessionErrorPayload(
                        sessionId=session_id,
                        event=event_type,
                        error=str(error) or type(error).__name__,
                    ),
                )


class Session:
    """Read and mutate one tree; invalid input and unavailable storage raise."""

    def __init__(self, session_id: str, sessions: Sessions) -> None:
        self.session_id = session_id
        self._sessions = sessions
        self._compaction_fn: CompactionFunction | None = None
        self._token_threshold: int | float | None = None
        self._compaction_lock = asyncio.Lock()
        self._message_generation = 0

    def on_compaction(self, fn: CompactionFunction) -> Session:
        """Register this activation's compaction callback and return the handle."""
        if not callable(fn):
            raise TypeError("compaction function must be callable")
        self._compaction_fn = fn
        return self

    def compact_after(self, token_threshold: int | float) -> Session:
        """Compact after inserted appends whose active estimate exceeds a threshold."""
        self._token_threshold = _finite_number(token_threshold, "token_threshold")
        return self

    async def history(
        self,
        options: HistoryReadOptions | None = None,
    ) -> AsyncIterator[SessionMessage]:
        """Stream one root-to-leaf path, skipping malformed stored messages."""
        await self._sessions._ready()
        options = HistoryReadOptions() if options is None else options
        _raise_if_aborted(options.signal)
        path = self._store.path(self.session_id, options.leaf_id)
        if not path:
            return
        messages = self._history_values(path)
        while True:
            _raise_if_aborted(options.signal)
            try:
                message = next(messages)
            except StopIteration:
                return
            yield cast(SessionMessage, message)

    async def history_batches(
        self,
        options: HistoryBatchReadOptions | None = None,
    ) -> AsyncIterator[list[SessionMessage]]:
        """Stream non-empty batches bounded by count and serialized byte size."""
        options = HistoryBatchReadOptions() if options is None else options
        batch_size = _positive_floor(options.batch_size, "batch_size")
        max_batch_bytes = _positive_floor(
            options.max_batch_bytes,
            "max_batch_bytes",
        )
        read_options = HistoryReadOptions(
            leaf_id=options.leaf_id,
            signal=options.signal,
        )
        batch: list[SessionMessage] = []
        batch_bytes = 0
        async for message in self.history(read_options):
            message_bytes = len(
                dumps_session_json(_normalize_json(message)).encode("utf-8")
            )
            if batch and (
                len(batch) >= batch_size
                or batch_bytes + message_bytes > max_batch_bytes
            ):
                yield batch
                batch = []
                batch_bytes = 0
            batch.append(message)
            batch_bytes += message_bytes
            if len(batch) >= batch_size or batch_bytes >= max_batch_bytes:
                yield batch
                batch = []
                batch_bytes = 0
        if batch:
            yield batch

    async def get_history(
        self,
        options: HistoryReadOptions | None = None,
    ) -> list[SessionMessage]:
        """Materialize one selected history path; prefer streaming for large paths."""
        return [message async for message in self.history(options)]

    async def get_recent_history(
        self,
        max_content_bytes: int | float,
        *,
        leaf_id: str | None = None,
    ) -> RecentHistoryResult:
        """Return the longest recent suffix fitting the byte budget."""
        await self._sessions._ready()
        budget = _finite_number(max_content_bytes, "max_content_bytes")
        rows = self._store.path(self.session_id, leaf_id)
        if not rows:
            return RecentHistoryResult([], False, 0)
        total = sum(row.bytes for row in rows)
        start = len(rows) - 1
        used = rows[start].bytes
        while start > 0 and used + rows[start - 1].bytes <= budget:
            start -= 1
            used += rows[start].bytes
        messages = [
            cast(SessionMessage, message)
            for message in self._history_values(rows[start:])
        ]
        capped = len(rows) > 10_000 and rows[0].parent_id is not None
        return RecentHistoryResult(messages, start > 0 or capped, total)

    async def get_history_row_stats(
        self,
        leaf_id: str | None = None,
    ) -> list[SessionRowStat]:
        """Return content-free size and token statistics for one history path."""
        await self._sessions._ready()
        return [
            SessionRowStat(row.id, row.role, row.bytes, row.token_estimate)
            for row in self._store.path(self.session_id, leaf_id)
        ]

    async def get_message(self, message_id: str) -> SessionMessage | None:
        """Return one stored message, or None when absent or malformed."""
        await self._sessions._ready()
        return _as_session_message(self._store.get(self.session_id, message_id))

    async def search(
        self,
        query: str,
        options: SearchOptions | None = None,
    ) -> list[SearchResult]:
        """Find messages whose text parts contain the literal phrase."""
        await self._sessions._ready()
        if type(query) is not str:
            raise TypeError("query must be a string")
        options = SearchOptions() if options is None else options
        return [
            SearchResult(
                id=cast(str, row["id"]),
                role=cast(str, row["role"]),
                content=cast(str, row["content"]),
            )
            for row in self._store.search(self.session_id, query, options.limit)
        ]

    async def add_compaction(
        self,
        summary: str,
        from_message_id: str,
        to_message_id: str,
    ) -> StoredCompaction:
        """Store a non-destructive summary overlay and notify after commit."""
        await self._sessions._ready()
        for name, value in (
            ("summary", summary),
            ("from_message_id", from_message_id),
            ("to_message_id", to_message_id),
        ):
            if type(value) is not str:
                raise TypeError(f"{name} must be a string")
        async with self._compaction_lock:
            compaction = self._store.add_compaction(
                self.session_id,
                summary,
                from_message_id,
                to_message_id,
            )
        await self._announce_compaction(compaction)
        return _public_compaction(compaction)

    async def _announce_compaction(self, compaction: _StoredCompaction) -> None:
        await self._sessions._emit(
            "session:compacted",
            _SessionCompactedPayload(
                sessionId=self.session_id,
                compactionId=compaction.id,
            ),
        )
        await self._sessions._notify(
            _SessionCompactEvent(type="compact", sessionId=self.session_id)
        )

    async def get_compactions(self) -> list[StoredCompaction]:
        """Return durable overlays in insertion order."""
        await self._sessions._ready()
        return [
            _public_compaction(compaction)
            for compaction in self._store.compactions(self.session_id)
        ]

    async def compact(self, leaf_id: str | None = None) -> CompactResult | None:
        """Run the registered callback and store its valid summary overlay."""
        await self._sessions._ready()
        if leaf_id is not None and type(leaf_id) is not str:
            raise TypeError("leaf_id must be a string or None")
        async with self._compaction_lock:
            outcome = await self._run_compaction(leaf_id)
        return await self._finish_compaction(outcome)

    async def _finish_compaction(
        self,
        outcome: tuple[
            CompactResult | None,
            _StoredCompaction | None,
            Exception | None,
        ],
    ) -> CompactResult | None:
        effective, compaction, error = outcome
        if error is not None:
            await self._emit_compaction_error(error)
            return None
        if compaction is not None:
            await self._announce_compaction(compaction)
        return effective

    async def _run_compaction(
        self,
        leaf_id: str | None,
    ) -> tuple[CompactResult | None, _StoredCompaction | None, Exception | None]:
        fn = self._compaction_fn
        if fn is None:
            raise RuntimeError(
                "No compaction function registered. Call on_compaction() first."
            )
        generation = self._message_generation
        history = await self.get_history(HistoryReadOptions(leaf_id=leaf_id))
        try:
            result = fn(history)
            if inspect.isawaitable(result):
                result = await result
            if result is not None and not isinstance(result, CompactResult):
                raise TypeError("compaction function must return CompactResult or None")
            if result is not None and any(
                type(value) is not str
                for value in (
                    result.from_message_id,
                    result.to_message_id,
                    result.summary,
                )
            ):
                raise TypeError("CompactResult fields must be strings")
        except Exception as error:
            return None, None, error
        if result is None:
            return None, None, None
        if self._message_generation != generation:
            return None, None, None
        history_ids = {message["id"] for message in history}
        existing = [
            compaction
            for compaction in self._store.compaction_ranges(self.session_id)
            if f"{_COMPACTION_PREFIX}{compaction.id}" in history_ids
            or (
                compaction.from_message_id in history_ids
                and compaction.to_message_id in history_ids
            )
        ]
        from_message_id = (
            existing[0].from_message_id if existing else result.from_message_id
        )
        raw_path = self._store.path(self.session_id, leaf_id)
        indexes = {row.id: index for index, row in enumerate(raw_path)}
        start = indexes.get(from_message_id)
        end = indexes.get(result.to_message_id)
        if start is None or end is None or end < start:
            return None, None, None
        effective = CompactResult(
            from_message_id=from_message_id,
            to_message_id=result.to_message_id,
            summary=result.summary,
        )
        compaction = self._store.add_compaction(
            self.session_id,
            effective.summary,
            effective.from_message_id,
            effective.to_message_id,
        )
        return effective, compaction, None

    async def get_latest_leaf(self) -> SessionMessage | None:
        """Return the maximum-sequence message for this activation's session."""
        await self._sessions._ready()
        message_id = self._store.latest_leaf_id(self.session_id)
        return None if message_id is None else await self.get_message(message_id)

    async def get_branches(self, message_id: str) -> list[SessionMessage]:
        """Return direct child branches in ascending sequence order."""
        await self._sessions._ready()
        return [
            cast(SessionMessage, message)
            for message in self._store.branches(self.session_id, message_id)
        ]

    async def append_message(
        self,
        message: Mapping[str, object],
        options: AppendOptions | None = None,
    ) -> AppendResult:
        """Append idempotently; invalid JSON or unavailable storage raises."""
        await self._sessions._ready()
        options = AppendOptions() if options is None else options
        if (
            options.parent_id is not MISSING
            and options.parent_id is not None
            and type(options.parent_id) is not str
        ):
            raise TypeError("parent_id must be a string, None, or omitted")
        prepared = _prepare_message(
            message,
            source=options.source,
            reserved_keys=self._sessions._options.reserved_metadata_keys,
        )
        token_estimate = _estimate_message_tokens(prepared)
        inserted, stored = self._store.append(
            self.session_id,
            cast(dict[str, object], prepared),
            options.parent_id,
            MISSING,
            token_estimate,
        )
        if inserted:
            self._message_generation += 1
        stored_message = _clone_message(stored)
        if inserted:
            await self._sessions._emit(
                "session:message:appended",
                _SessionAppendPayload(
                    sessionId=self.session_id,
                    messageId=stored_message["id"],
                    tokenEstimate=token_estimate,
                ),
            )
        event = _SessionAppendEvent(
            type="append",
            sessionId=self.session_id,
            message=stored_message,
            inserted=inserted,
        )
        if options.parent_id is not MISSING:
            event["parentId"] = cast(str | None, options.parent_id)
        await self._sessions._notify(event)
        if inserted:
            await self._maybe_auto_compact()
        return AppendResult(inserted, _clone_message(stored_message))

    async def update_message(
        self,
        message: Mapping[str, object],
        options: WriteOptions | None = None,
    ) -> SessionMessage | None:
        """Update without moving the row; invalid input or storage raises."""
        await self._sessions._ready()
        options = WriteOptions() if options is None else options
        prepared = _prepare_message(
            message,
            source=options.source,
            reserved_keys=self._sessions._options.reserved_metadata_keys,
        )
        outcome = self._store.update(
            self.session_id,
            cast(dict[str, object], prepared),
            _estimate_message_tokens(prepared),
        )
        if outcome == "missing":
            return None
        if outcome == "updated":
            self._message_generation += 1
            await self._sessions._emit(
                "session:message:updated",
                _SessionMessageEventPayload(
                    sessionId=self.session_id,
                    messageId=prepared["id"],
                ),
            )
            await self._sessions._notify(
                _SessionUpdateEvent(
                    type="update",
                    sessionId=self.session_id,
                    message=prepared,
                )
            )
        return _clone_message(prepared)

    async def upsert_message(
        self,
        message: Mapping[str, object],
        options: AppendOptions | None = None,
    ) -> AppendResult:
        """Append an absent ID or update an existing row without moving it."""
        await self._sessions._ready()
        options = AppendOptions() if options is None else options
        message_id = message.get("id")
        if type(message_id) is str and self._store.exists(self.session_id, message_id):
            stored = await self.update_message(
                message,
                WriteOptions(source=options.source),
            )
            if stored is None:
                raise SessionStorageError("stored Session message disappeared")
            return AppendResult(False, stored)
        return await self.append_message(message, options)

    async def import_message(
        self,
        message: Mapping[str, object],
        *,
        parent_id: str | None,
        created_at_ms: int,
    ) -> None:
        """Import a historical row verbatim with no telemetry or change event."""
        await self._sessions._ready()
        if type(parent_id) is not str and parent_id is not None:
            raise TypeError("parent_id must be a string or None")
        if type(created_at_ms) is not int or created_at_ms < 0:
            raise ValueError("created_at_ms must be a non-negative integer")
        prepared = _prepare_message(message, sanitize=False)
        inserted = self._store.import_message(
            self.session_id,
            cast(dict[str, object], prepared),
            parent_id,
            created_at_ms,
            _estimate_message_tokens(prepared),
        )
        if inserted:
            self._message_generation += 1

    async def delete_messages(self, message_ids: Sequence[str]) -> None:
        """Delete requested IDs atomically and splice surviving children upward."""
        await self._sessions._ready()
        ids = tuple(message_ids)
        if any(type(message_id) is not str for message_id in ids):
            raise TypeError("message IDs must be strings")
        count = self._store.delete_messages(self.session_id, ids)
        if count:
            self._message_generation += 1
            await self._sessions._emit(
                "session:messages:deleted",
                _SessionDeletePayload(sessionId=self.session_id, count=count),
            )
        await self._sessions._notify(
            _SessionDeleteEvent(
                type="delete",
                sessionId=self.session_id,
                messageIds=list(ids),
            )
        )

    async def clear_messages(self) -> None:
        """Delete all rows for this session and reset its activation-local tail."""
        await self._sessions._ready()
        self._store.clear(self.session_id)
        self._message_generation += 1
        await self._sessions._emit(
            "session:cleared",
            _SessionClearPayload(sessionId=self.session_id),
        )
        await self._sessions._notify(
            _SessionClearEvent(type="clear", sessionId=self.session_id)
        )

    def _history_values(
        self,
        path: Sequence[_SessionPathRow],
    ) -> Iterator[dict[str, object]]:
        spans = _plan_overlays(
            [row.id for row in path],
            self._store.compaction_ranges(self.session_id),
        )
        spans_by_start = {span.start_index: span for span in spans}
        index = 0
        while index < len(path):
            span = spans_by_start.get(index)
            if span is not None:
                compaction = self._store.compaction(
                    self.session_id,
                    span.compaction.id,
                )
                if compaction is None:
                    yield from self._store.hydrate(
                        self.session_id,
                        path[index : span.end_index + 1],
                    )
                else:
                    yield _overlay_message(compaction)
                index = span.end_index + 1
                continue
            run_end = index + 1
            while run_end < len(path) and run_end not in spans_by_start:
                run_end += 1
            yield from self._store.hydrate(self.session_id, path[index:run_end])
            index = run_end

    def _active_token_estimate(self) -> int:
        path = self._store.path(self.session_id, None)
        estimate = sum(row.token_estimate for row in path)
        for span in _plan_overlays(
            [row.id for row in path],
            self._store.compaction_ranges(self.session_id),
        ):
            compaction = self._store.compaction(self.session_id, span.compaction.id)
            if compaction is None:
                continue
            estimate -= sum(
                path[index].token_estimate
                for index in range(span.start_index, span.end_index + 1)
            )
            estimate += _estimate_unknown_tokens(compaction.summary)
        return max(0, math.ceil(estimate))

    async def _maybe_auto_compact(self) -> None:
        threshold = self._token_threshold
        if threshold is None or self._compaction_fn is None:
            return
        try:
            async with self._compaction_lock:
                if self._active_token_estimate() <= threshold:
                    return
                outcome = await self._run_compaction(None)
            await self._finish_compaction(outcome)
        except Exception as error:
            _LOGGER.warning("Sessions auto-compaction failed: %s", error)
            await self._emit_compaction_error(error)

    async def _emit_compaction_error(self, error: Exception) -> None:
        await self._sessions._emit(
            "session:error",
            _SessionCompactionErrorPayload(
                sessionId=self.session_id,
                error=str(error),
            ),
        )

    @property
    def _store(self) -> _SessionStore:
        return self._sessions._session_store()


def create_compact_function(options: CompactOptions) -> CompactionFunction:
    """Create the reference head/middle/tail summarization policy."""

    async def compact(messages: list[SessionMessage]) -> CompactResult | None:
        prepared = _prepare_compaction_input(
            messages,
            options.keep_recent_tokens,
            _estimate_compaction_tokens,
        )
        if prepared is None:
            return None
        summary = options.summarize(_build_summary_prompt(prepared))
        if inspect.isawaitable(summary):
            summary = await summary
        if not summary.strip():
            return None
        return CompactResult(
            from_message_id=cast(str, prepared.messages[0]["id"]),
            to_message_id=cast(str, prepared.messages[-1]["id"]),
            summary=summary,
        )

    return compact


def _public_compaction(compaction: _StoredCompaction) -> StoredCompaction:
    created_at = (
        datetime.fromtimestamp(compaction.created_at_ms / 1000, UTC)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )
    return StoredCompaction(
        id=compaction.id,
        summary=compaction.summary,
        from_message_id=compaction.from_message_id,
        to_message_id=compaction.to_message_id,
        created_at=created_at,
    )


def _prepare_message(
    message: Mapping[str, object],
    *,
    source: Literal["client", "server"] = "server",
    reserved_keys: Sequence[str] = (),
    sanitize: bool = True,
) -> SessionMessage:
    if source not in ("client", "server"):
        raise ValueError("source must be 'client' or 'server'")
    normalized = _normalize_json(dict(message))
    if not isinstance(normalized, dict):
        raise TypeError("Session message must be an object")
    message_id = normalized.get("id")
    role = normalized.get("role")
    parts = normalized.get("parts")
    if type(message_id) is not str or not message_id:
        raise ValueError("Session message id must be a non-empty string")
    if type(role) is not str or not role:
        raise ValueError("Session message role must be a non-empty string")
    if not isinstance(parts, list):
        raise TypeError("Session message parts must be a list")
    if any(
        not isinstance(part, dict) or type(part.get("type")) is not str
        for part in parts
    ):
        raise TypeError("Session message parts must be objects with string types")
    if sanitize:
        normalized["parts"] = _sanitize_parts(cast(list[dict[str, object]], parts))
        if source == "client":
            _strip_reserved_metadata(normalized, reserved_keys)
    try:
        dumps_session_json(normalized)
    except (TypeError, ValueError) as error:
        raise ValueError("Session message must contain strict JSON values") from error
    return cast(SessionMessage, normalized)


def _normalize_json(value: object) -> object:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError("Session datetimes must include a timezone")
        return (
            value.astimezone(UTC)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z")
        )
    if isinstance(value, dict):
        return {key: _normalize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_json(item) for item in value]
    return value


def _sanitize_parts(parts: list[dict[str, object]]) -> list[dict[str, object]]:
    sanitized_parts = []
    for original in parts:
        part = dict(original)
        for key in ("providerMetadata", "callProviderMetadata"):
            metadata = part.get(key)
            if not isinstance(metadata, dict) or "openai" not in metadata:
                continue
            openai = metadata.get("openai")
            if not isinstance(openai, dict):
                continue
            remaining_openai = {
                name: value
                for name, value in openai.items()
                if name not in ("itemId", "reasoningEncryptedContent")
            }
            remaining_metadata = {
                name: value for name, value in metadata.items() if name != "openai"
            }
            if remaining_openai:
                remaining_metadata["openai"] = remaining_openai
            if remaining_metadata:
                part[key] = remaining_metadata
            else:
                part.pop(key, None)
        if part["type"] == "reasoning":
            text = part.get("text")
            metadata = part.get("providerMetadata")
            if (not isinstance(text, str) or not text.strip()) and not (
                isinstance(metadata, dict) and metadata
            ):
                continue
        sanitized_parts.append(part)
    return sanitized_parts


def _strip_reserved_metadata(
    message: dict[str, object],
    reserved_keys: Sequence[str],
) -> None:
    metadata = message.get("metadata")
    if not isinstance(metadata, dict):
        return
    filtered = {
        key: value for key, value in metadata.items() if key not in reserved_keys
    }
    if filtered:
        message["metadata"] = filtered
    else:
        message.pop("metadata", None)


def _estimate_message_tokens(message: SessionMessage) -> int:
    tokens = _estimate_compaction_tokens(message)
    for part in message["parts"]:
        if part["type"] != "file":
            continue
        url = part.get("url")
        if not isinstance(url, str) or not url.startswith("data:"):
            continue
        media_type = part.get("mediaType", "application/octet-stream")
        if not isinstance(media_type, str):
            media_type = "application/octet-stream"
        if media_type.startswith("image/"):
            tokens += 1_600
        else:
            tokens += min(math.ceil(_estimated_data_url_bytes(url) / 4), 20_000)
    return tokens


def _estimate_compaction_tokens(message: Mapping[str, object]) -> int:
    tokens = 4
    parts = message.get("parts")
    if not isinstance(parts, list):
        return tokens
    for part in parts:
        if not isinstance(part, Mapping):
            continue
        part_type = part.get("type")
        if not isinstance(part_type, str):
            continue
        if part_type in ("text", "reasoning"):
            value = part.get("text")
            if value is None:
                value = part.get("reasoning")
            tokens += _estimate_unknown_tokens(value)
        elif part_type.startswith("tool-") or part_type == "dynamic-tool":
            tokens += _estimate_unknown_tokens(part.get("input"))
            value = part.get("output")
            if value is None:
                value = part.get("result")
            tokens += _estimate_unknown_tokens(value)
        elif "text" in part:
            tokens += _estimate_unknown_tokens(part["text"])
        elif "result" in part:
            tokens += _estimate_unknown_tokens(part["result"])
    return tokens


def _estimate_unknown_tokens(value: object) -> int:
    if value is None:
        return 0
    text = value if isinstance(value, str) else dumps_session_json(value)
    if not text:
        return 0
    char_estimate = _utf16_length(text) / 4
    word_estimate = len(re.findall(r"\S+", text)) * 1.3
    return math.ceil(max(char_estimate, word_estimate))


def _utf16_length(value: str) -> int:
    return len(value.encode("utf-16-le", errors="surrogatepass")) // 2


def _estimated_data_url_bytes(url: str) -> int:
    comma = url.find(",")
    if not url.startswith("data:") or comma < 0:
        return 0
    header = url[5:comma]
    payload = url[comma + 1 :]
    length = _utf16_length(payload)
    return math.floor(length * 3 / 4) if header.endswith(";base64") else length


def _as_session_message(value: object) -> SessionMessage | None:
    parsed = session_message_from_value(value)
    return None if parsed is None else cast(SessionMessage, parsed)


def _clone_message(value: object) -> SessionMessage:
    cloned = _as_session_message(clone_session_json(value))
    if cloned is None:
        raise SessionStorageError("stored Session message is malformed")
    return cloned


def _positive_floor(value: int | float, name: str) -> int:
    number = _finite_number(value, name)
    return max(1, math.floor(number))


def _finite_number(value: int | float, name: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _raise_if_aborted(signal: object | None) -> None:
    if signal is None or not bool(getattr(signal, "aborted", False)):
        return
    reason = getattr(signal, "reason", None)
    if isinstance(reason, BaseException):
        raise reason
    raise RuntimeError("History read aborted")


__all__ = [
    "AppendOptions",
    "AppendResult",
    "CompactOptions",
    "CompactResult",
    "CompactionFunction",
    "HistoryBatchReadOptions",
    "HistoryReadOptions",
    "RecentHistoryResult",
    "SearchOptions",
    "SearchResult",
    "Session",
    "SessionChangeEvent",
    "SessionChangeListener",
    "SessionMessage",
    "SessionMessagePart",
    "SessionRowStat",
    "SessionStorageError",
    "Sessions",
    "SessionsOptions",
    "StoredCompaction",
    "WriteOptions",
    "create_compact_function",
]
