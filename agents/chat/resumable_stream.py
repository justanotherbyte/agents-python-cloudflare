from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

from .protocol import chat_response
from ..core.utils import gen_id, now_ms

SqlFn = Callable[..., list[dict[str, Any]]]


class StreamStatus(StrEnum):
    STREAMING = "streaming"
    COMPLETED = "completed"
    ERROR = "error"


class _ReplayResult(StrEnum):
    SENT = "sent"
    CLOSED = "closed"
    READ_FAILED = "read-failed"


@dataclass(frozen=True)
class StreamRecord:
    stream_id: str
    request_id: str
    status: StreamStatus
    message_id: str | None


@dataclass(frozen=True)
class RecoverySnapshot:
    stream: StreamRecord
    bodies: tuple[str, ...]
    limit_exceeded: bool = False


@dataclass(frozen=True)
class TerminalRecord:
    request_id: str
    body: str


class StreamStorageUnavailable(RuntimeError):
    pass


class _Sendable(Protocol):
    def send_if_open(self, data: dict[str, Any]) -> bool: ...


_CHUNK_MAX_BYTES = 1_800_000
_CHUNK_BUFFER_SIZE = 10
_SEGMENT_MAX_BYTES = 512_000
_REPLAY_PAGE_SIZE = 100
_COMPLETED_RETENTION_MS = 10 * 60 * 1000
_ERROR_RETENTION_MS = 10 * 60 * 1000
_ABANDONED_RETENTION_MS = 60 * 60 * 1000

_DELETE_ABANDONED_CHUNKS = (
    "DELETE FROM cf_ai_chat_stream_chunks WHERE stream_id IN ("
    "SELECT m.id FROM cf_ai_chat_stream_metadata m WHERE m.status = ? "
    "AND COALESCE((SELECT MAX(c.created_at) FROM cf_ai_chat_stream_chunks c "
    "WHERE c.stream_id = m.id), m.created_at) < ? AND m.id != ?)"
)
_DELETE_ABANDONED_METADATA = (
    "DELETE FROM cf_ai_chat_stream_metadata WHERE id IN ("
    "SELECT m.id FROM cf_ai_chat_stream_metadata m WHERE m.status = ? "
    "AND COALESCE((SELECT MAX(c.created_at) FROM cf_ai_chat_stream_chunks c "
    "WHERE c.stream_id = m.id), m.created_at) < ? AND m.id != ?)"
)


def _utf8_len(text: str) -> int:
    return len(text.encode("utf-8"))


def _unpack_body(body: str) -> tuple[str, ...]:
    try:
        packed = json.loads(body)
    except (TypeError, ValueError):
        packed = None
    if isinstance(packed, list) and all(isinstance(item, str) for item in packed):
        return tuple(packed)
    return (body,)


def _pack_bodies(bodies: list[str]) -> str:
    if len(bodies) == 1:
        return bodies[0]
    return json.dumps(bodies, separators=(",", ":"))


class ResumableStream:
    """Persist serialized chunks before delivery and replay them after reconnect."""

    def __init__(self, sql: SqlFn, response_type: str):
        self._sql = sql
        self._response_type = response_type
        self._active_stream_id: str | None = None
        self._active_request_id: str | None = None
        self._chunk_index = 0
        self._is_live = False
        self._chunk_buffer: list[tuple[str, str]] = []
        self._chunk_buffer_row_id: str | None = None

    def prepare(self) -> None:
        """Prepare storage and restore the active stream during startup."""
        self._ensure_tables()
        self._clear_active()
        self._restore()

    def _try_sql(self, query: str, *params: Any) -> list[dict[str, Any]] | None:
        try:
            return self._sql(query, *params)
        except Exception:  # noqa: BLE001
            return None

    def _strict_sql(self, query: str, *params: Any) -> list[dict[str, Any]]:
        try:
            return self._sql(query, *params)
        except Exception as exc:
            raise StreamStorageUnavailable("stream storage read failed") from exc

    def _ensure_tables(self) -> None:
        self._strict_sql("""
        CREATE TABLE IF NOT EXISTS cf_ai_chat_stream_chunks (
            id TEXT PRIMARY KEY,
            stream_id TEXT NOT NULL,
            body TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            created_at INTEGER NOT NULL
        )
        """)
        self._strict_sql("""
        CREATE TABLE IF NOT EXISTS cf_ai_chat_stream_metadata (
            id TEXT PRIMARY KEY,
            request_id TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            completed_at INTEGER,
            message_id TEXT,
            is_continuation INTEGER
        )
        """)
        self._strict_sql("""
        CREATE TABLE IF NOT EXISTS cf_ai_chat_terminal (
            slot INTEGER PRIMARY KEY CHECK (slot = 1),
            request_id TEXT NOT NULL,
            body TEXT NOT NULL,
            created_at INTEGER NOT NULL
        )
        """)
        self._reconcile_metadata_columns()
        self._strict_sql(
            "CREATE INDEX IF NOT EXISTS idx_stream_chunks_stream_id "
            "ON cf_ai_chat_stream_chunks(stream_id, chunk_index)"
        )

    def _reconcile_metadata_columns(self) -> None:
        rows = self._strict_sql("PRAGMA table_info(cf_ai_chat_stream_metadata)")
        columns = {row["name"] for row in rows}
        additions = (
            (
                "message_id",
                "ALTER TABLE cf_ai_chat_stream_metadata ADD COLUMN message_id TEXT",
            ),
            (
                "is_continuation",
                (
                    "ALTER TABLE cf_ai_chat_stream_metadata "
                    "ADD COLUMN is_continuation INTEGER"
                ),
            ),
        )
        for name, query in additions:
            if name not in columns:
                self._strict_sql(query)

    @property
    def active_request_id(self) -> str | None:
        return self._active_request_id

    def has_active_stream(self) -> bool:
        return self._active_stream_id is not None

    def start(self, request_id: str, message_id: str | None = None) -> str:
        self.flush_buffer()
        stream_id = gen_id()
        self._active_stream_id = stream_id
        self._active_request_id = request_id
        self._chunk_index = 0
        self._is_live = True
        wrote = self._try_sql(
            "INSERT INTO cf_ai_chat_stream_metadata "
            "(id, request_id, status, created_at, message_id, is_continuation) "
            "VALUES (?, ?, 'streaming', ?, ?, ?)",
            stream_id,
            request_id,
            now_ms(),
            message_id,
            0,
        )
        if wrote is None:
            self._clear_active()
            raise StreamStorageUnavailable("stream metadata could not be persisted")
        return stream_id

    def store_chunk(self, stream_id: str, body: str) -> bool:
        body_bytes = _utf8_len(body)
        if body_bytes > _CHUNK_MAX_BYTES:
            return False
        if self._chunk_buffer and self._chunk_buffer[0][0] != stream_id:
            self.flush_buffer()
        buffered = [chunk_body for _, chunk_body in self._chunk_buffer]
        candidate = buffered + [body]
        packed = _pack_bodies(candidate)
        if self._chunk_buffer and _utf8_len(packed) > _SEGMENT_MAX_BYTES:
            self.flush_buffer()
        self._chunk_buffer.append((stream_id, body))
        if not self._persist_buffer():
            return False
        if len(self._chunk_buffer) >= _CHUNK_BUFFER_SIZE:
            self.flush_buffer()
        return True

    def update_message_id(self, stream_id: str, message_id: str) -> None:
        rows = self._strict_sql(
            "UPDATE cf_ai_chat_stream_metadata SET message_id = ? "
            "WHERE id = ? AND status = 'streaming' RETURNING id",
            message_id,
            stream_id,
        )
        if not rows:
            raise StreamStorageUnavailable("stream message ID could not be persisted")

    def flush_buffer(self) -> None:
        """Seal the current durable segment so the next chunk starts a new row."""
        if not self._chunk_buffer:
            return
        self._chunk_buffer = []
        self._chunk_buffer_row_id = None

    def _persist_buffer(self) -> bool:
        stream_id = self._chunk_buffer[0][0]
        bodies = [body for _, body in self._chunk_buffer]
        segment_body = _pack_bodies(bodies)
        if self._chunk_buffer_row_id is not None:
            updated = self._try_sql(
                "UPDATE cf_ai_chat_stream_chunks SET body = ?, created_at = ? "
                "WHERE id = ? RETURNING id",
                segment_body,
                now_ms(),
                self._chunk_buffer_row_id,
            )
            if not updated:
                self.flush_buffer()
                return False
            return True

        row_id = gen_id()
        wrote = self._try_sql(
            "INSERT INTO cf_ai_chat_stream_chunks "
            "(id, stream_id, body, chunk_index, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            row_id,
            stream_id,
            segment_body,
            self._chunk_index,
            now_ms(),
        )
        if wrote is not None:
            self._chunk_buffer_row_id = row_id
            self._chunk_index += 1
            return True
        else:
            self.flush_buffer()
            return False

    def complete(self, stream_id: str) -> None:
        self._finish(stream_id, StreamStatus.COMPLETED)

    def mark_error(self, stream_id: str) -> None:
        self._finish(stream_id, StreamStatus.ERROR)

    def _finish(self, stream_id: str, status: StreamStatus) -> None:
        self.flush_buffer()
        self._try_sql(
            "UPDATE cf_ai_chat_stream_metadata "
            "SET status = ?, completed_at = ? WHERE id = ?",
            status,
            now_ms(),
            stream_id,
        )
        if self._active_stream_id == stream_id:
            self._clear_active()

    def record_terminal_error(self, request_id: str, body: str) -> None:
        self._try_sql(
            "INSERT INTO cf_ai_chat_terminal (slot, request_id, body, created_at) "
            "VALUES (1, ?, ?, ?) ON CONFLICT (slot) DO UPDATE SET "
            "request_id = excluded.request_id, body = excluded.body, "
            "created_at = excluded.created_at",
            request_id,
            body,
            now_ms(),
        )

    def latest_terminal_error(self) -> TerminalRecord | None:
        rows = self._try_sql(
            "SELECT request_id, body FROM cf_ai_chat_terminal WHERE slot = 1"
        )
        if not rows:
            return None
        return TerminalRecord(rows[0]["request_id"], rows[0]["body"])

    def clear_terminal_error(self) -> None:
        self._try_sql("DELETE FROM cf_ai_chat_terminal")

    def replay_active_chunks(self, connection: _Sendable, request_id: str) -> None:
        stream_id = self._active_stream_id
        if stream_id is None:
            return
        replay = self._replay_rows(connection, stream_id, request_id)
        if replay == _ReplayResult.READ_FAILED:
            if not self._is_live:
                self.mark_error(stream_id)
            self._send_replay_failure(connection, request_id)
            return
        if replay == _ReplayResult.CLOSED:
            return
        if self._is_live:
            connection.send_if_open(
                chat_response(
                    self._response_type,
                    request_id,
                    done=False,
                    replay=True,
                    replay_complete=True,
                )
            )
            return
        connection.send_if_open(
            chat_response(self._response_type, request_id, done=True, replay=True)
        )
        self.complete(stream_id)

    def replay_completed_chunks(self, connection: _Sendable, request_id: str) -> bool:
        stream_id = self._latest_stream_id_for_status(
            request_id, StreamStatus.COMPLETED
        )
        if stream_id is None:
            return False
        replay = self._replay_rows(connection, stream_id, request_id)
        if replay == _ReplayResult.READ_FAILED:
            self._send_replay_failure(connection, request_id)
            return True
        if replay == _ReplayResult.CLOSED:
            return True
        connection.send_if_open(
            chat_response(self._response_type, request_id, done=True, replay=True)
        )
        return True

    def replay_error_chunks(self, connection: _Sendable, request_id: str) -> bool:
        stream_id = self._latest_stream_id_for_status(request_id, StreamStatus.ERROR)
        if stream_id is None:
            return False
        self._replay_rows(connection, stream_id, request_id)
        return True

    def _replay_rows(
        self,
        connection: _Sendable,
        stream_id: str,
        request_id: str,
    ) -> _ReplayResult:
        cursor = (-1, 0)
        while True:
            rows = self._chunk_page(stream_id, cursor, strict=False)
            if rows is None:
                return _ReplayResult.READ_FAILED
            for row in rows:
                for body in _unpack_body(row["body"]):
                    if not connection.send_if_open(
                        chat_response(
                            self._response_type,
                            request_id,
                            body,
                            done=False,
                            replay=True,
                        )
                    ):
                        return _ReplayResult.CLOSED
            if len(rows) < _REPLAY_PAGE_SIZE:
                return _ReplayResult.SENT
            cursor = (int(rows[-1]["chunk_index"]), int(rows[-1]["row_id"]))

    def _send_replay_failure(self, connection: _Sendable, request_id: str) -> None:
        frame = chat_response(
            self._response_type,
            request_id,
            "Unable to replay the retained chat stream",
            done=True,
        )
        frame["error"] = True
        connection.send_if_open(frame)

    def _chunk_page(
        self,
        stream_id: str,
        cursor: tuple[int, int],
        *,
        strict: bool,
    ) -> list[dict[str, Any]] | None:
        query = (
            "SELECT rowid AS row_id, chunk_index, body "
            "FROM cf_ai_chat_stream_chunks WHERE stream_id = ? "
            "AND (chunk_index > ? OR (chunk_index = ? AND rowid > ?)) "
            "ORDER BY chunk_index, rowid LIMIT ?"
        )
        params = (stream_id, cursor[0], cursor[0], cursor[1], _REPLAY_PAGE_SIZE)
        if strict:
            return self._strict_sql(query, *params)
        return self._try_sql(query, *params)

    def _strict_bodies(self, stream_id: str) -> Iterator[str]:
        cursor = (-1, 0)
        while True:
            rows = self._chunk_page(stream_id, cursor, strict=True) or []
            for row in rows:
                yield from _unpack_body(row["body"])
            if len(rows) < _REPLAY_PAGE_SIZE:
                return
            cursor = (int(rows[-1]["chunk_index"]), int(rows[-1]["row_id"]))

    def latest_for_request(self, request_id: str) -> StreamRecord | None:
        rows = self._strict_sql(
            "SELECT id, request_id, status, message_id FROM "
            "cf_ai_chat_stream_metadata WHERE request_id = ? "
            "ORDER BY created_at DESC, rowid DESC LIMIT 1",
            request_id,
        )
        return self._record(rows[0]) if rows else None

    def recovery_snapshot(
        self,
        request_id: str,
        message_id: str | None = None,
        *,
        max_chunks: int | None = None,
        max_bytes: int | None = None,
    ) -> RecoverySnapshot | None:
        where = "request_id = ?"
        params = [request_id]
        if message_id is not None:
            where += " AND message_id = ?"
            params.append(message_id)
        rows = self._strict_sql(
            "SELECT id, request_id, status, message_id FROM "
            f"cf_ai_chat_stream_metadata WHERE {where} "
            "ORDER BY created_at DESC, rowid DESC LIMIT 1",
            *params,
        )
        if not rows:
            return None
        stream = self._record(rows[0])
        bodies: list[str] = []
        total_bytes = 0
        for body in self._strict_bodies(stream.stream_id):
            bodies.append(body)
            total_bytes += _utf8_len(body)
            if max_chunks is not None and len(bodies) > max_chunks:
                return RecoverySnapshot(stream, (), limit_exceeded=True)
            if max_bytes is not None and total_bytes > max_bytes:
                return RecoverySnapshot(stream, (), limit_exceeded=True)
        return RecoverySnapshot(stream, tuple(bodies))

    def completed_bodies_for_request(
        self,
        request_id: str,
        *,
        after_sequence: int = -1,
        limit: int | None = None,
    ) -> tuple[str, ...]:
        rows = self._strict_sql(
            "SELECT id FROM cf_ai_chat_stream_metadata "
            "WHERE request_id = ? AND status = 'completed' "
            "ORDER BY created_at DESC, rowid DESC LIMIT 1",
            request_id,
        )
        if not rows:
            return ()
        bodies: list[str] = []
        for sequence, body in enumerate(self._strict_bodies(rows[0]["id"])):
            if sequence <= after_sequence:
                continue
            bodies.append(body)
            if limit is not None and len(bodies) >= limit:
                break
        return tuple(bodies)

    def mark_orphaned(self, stream_id: str) -> bool:
        self.flush_buffer()
        rows = self._strict_sql(
            "UPDATE cf_ai_chat_stream_metadata SET status = 'error', completed_at = ? "
            "WHERE id = ? AND status = 'streaming' RETURNING id",
            now_ms(),
            stream_id,
        )
        return bool(rows)

    @staticmethod
    def _record(row: dict[str, Any]) -> StreamRecord:
        try:
            status = StreamStatus(row["status"])
        except ValueError as exc:
            raise ValueError(f"unknown stream status: {row['status']!r}") from exc
        return StreamRecord(
            stream_id=row["id"],
            request_id=row["request_id"],
            status=status,
            message_id=row["message_id"],
        )

    def _latest_stream_id_for_status(
        self,
        request_id: str,
        status: StreamStatus,
    ) -> str | None:
        rows = self._try_sql(
            "SELECT id FROM cf_ai_chat_stream_metadata "
            "WHERE request_id = ? AND status = ? "
            "ORDER BY created_at DESC, rowid DESC LIMIT 1",
            request_id,
            status,
        )
        if not rows:
            return None
        return rows[0]["id"]

    def _restore(self) -> None:
        rows = self._strict_sql(
            "SELECT id, request_id FROM cf_ai_chat_stream_metadata "
            "WHERE status = 'streaming' "
            "ORDER BY created_at DESC, rowid DESC LIMIT 1"
        )
        if not rows:
            return
        self._active_stream_id = rows[0]["id"]
        self._active_request_id = rows[0]["request_id"]
        self._is_live = False
        max_rows = self._strict_sql(
            "SELECT MAX(chunk_index) AS max_index "
            "FROM cf_ai_chat_stream_chunks WHERE stream_id = ?",
            self._active_stream_id,
        )
        max_index = (
            None
            if not max_rows or max_rows[0]["max_index"] is None
            else int(max_rows[0]["max_index"])
        )
        self._chunk_index = max_index + 1 if max_index is not None else 0

    def _max_chunk_index(self, stream_id: str) -> int | None:
        rows = self._try_sql(
            "SELECT MAX(chunk_index) AS max_index "
            "FROM cf_ai_chat_stream_chunks WHERE stream_id = ?",
            stream_id,
        )
        if not rows or rows[0]["max_index"] is None:
            return None
        return int(rows[0]["max_index"])

    def clear_all(self) -> None:
        self._chunk_buffer = []
        self._chunk_buffer_row_id = None
        self._try_sql("DELETE FROM cf_ai_chat_stream_chunks")
        self._try_sql("DELETE FROM cf_ai_chat_stream_metadata")
        self.clear_terminal_error()
        self._clear_active()

    def _clear_active(self) -> None:
        self._active_stream_id = None
        self._active_request_id = None
        self._chunk_index = 0
        self._is_live = False

    def next_cleanup_deadline(self) -> int | None:
        rows = self._try_sql(
            "SELECT MIN(deadline) AS deadline FROM ("
            "SELECT completed_at + CASE status WHEN 'error' THEN ? ELSE ? END "
            "AS deadline FROM cf_ai_chat_stream_metadata "
            "WHERE status IN ('completed', 'error') AND completed_at IS NOT NULL "
            "UNION ALL "
            "SELECT COALESCE((SELECT MAX(c.created_at) "
            "FROM cf_ai_chat_stream_chunks c WHERE c.stream_id = m.id), "
            "m.created_at) + ? AS deadline FROM cf_ai_chat_stream_metadata m "
            "WHERE m.status = 'streaming')",
            _ERROR_RETENTION_MS,
            _COMPLETED_RETENTION_MS,
            _ABANDONED_RETENTION_MS,
        )
        if not rows or rows[0]["deadline"] is None:
            return None
        return int(rows[0]["deadline"])

    def cleanup(self, now: int) -> None:
        completed_cutoff = now - _COMPLETED_RETENTION_MS
        error_cutoff = now - _ERROR_RETENTION_MS
        abandoned_cutoff = now - _ABANDONED_RETENTION_MS
        live_stream_id = self._active_stream_id if self._is_live else ""
        self._try_sql(
            "DELETE FROM cf_ai_chat_stream_chunks WHERE stream_id IN ("
            "SELECT id FROM cf_ai_chat_stream_metadata WHERE "
            "(status = 'completed' AND completed_at < ?) OR "
            "(status = 'error' AND completed_at < ?))",
            completed_cutoff,
            error_cutoff,
        )
        self._try_sql(
            "DELETE FROM cf_ai_chat_stream_metadata WHERE "
            "(status = 'completed' AND completed_at < ?) OR "
            "(status = 'error' AND completed_at < ?)",
            completed_cutoff,
            error_cutoff,
        )
        self._try_sql(
            _DELETE_ABANDONED_CHUNKS,
            StreamStatus.STREAMING,
            abandoned_cutoff,
            live_stream_id,
        )
        self._try_sql(
            _DELETE_ABANDONED_METADATA,
            StreamStatus.STREAMING,
            abandoned_cutoff,
            live_stream_id,
        )
        if self._active_stream_id is not None:
            rows = self._try_sql(
                "SELECT id FROM cf_ai_chat_stream_metadata WHERE id = ?",
                self._active_stream_id,
            )
            if rows == []:
                self._clear_active()
