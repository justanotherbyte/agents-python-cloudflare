from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Literal, cast
from uuid import uuid4

from ._session_attachments import (
    _PendingAttachment,
    _StoredAttachment,
    _attachment_hashes,
    _extract_attachments,
    _resolve_attachments,
)
from ._session_json import dumps_session_json, parse_session_message
from ._session_compaction import _StoredCompaction
from .core.utils import now_ms
from .lifecycle import LifecycleSql


_MAX_PATH_DEPTH = 10_000
_MAX_INLINE_ROW_BYTES = 1536 * 1024
_ATTACHMENT_CHUNK_BYTES = _MAX_INLINE_ROW_BYTES
_HYDRATION_MAX_ROWS = 50
_HYDRATION_MAX_BYTES = 4 * 1024 * 1024
_FTS_BACKFILL_ROWS = 50
_COMPACTION_PAGE_ROWS = 50
_MIN_DATETIME_MS = -62_135_596_800_000
_MAX_DATETIME_MS = 253_402_300_799_999

_TABLE_COLUMNS = {
    "cf_agents_session_messages": (
        "session_id",
        "id",
        "seq",
        "parent_id",
        "type",
        "role",
        "content",
        "content_chunks",
        "token_estimate",
        "created_at",
    ),
    "cf_agents_session_message_chunks": ("session_id", "id", "idx", "content"),
    "cf_agents_session_compactions": (
        "session_id",
        "id",
        "seq",
        "summary",
        "from_message_id",
        "to_message_id",
        "created_at",
    ),
    "cf_agents_session_config": ("session_id", "key", "value"),
    "cf_agents_session_attachment_meta": (
        "hash",
        "bytes",
        "media_type",
        "chunks",
    ),
    "cf_agents_session_attachment_chunks": ("hash", "idx", "data"),
    "cf_agents_session_attachment_refs": ("session_id", "message_id", "hash"),
}


@dataclass(frozen=True)
class _SessionPathRow:
    id: str
    parent_id: str | None
    role: str
    bytes: int
    token_estimate: int


@dataclass(frozen=True)
class _SessionTail:
    leaf_id: str | None
    next_seq: int


@dataclass(frozen=True)
class _LegacyMigrationIncomplete:
    table: str
    source: int
    copied: int


type _UpdateOutcome = Literal["missing", "unchanged", "updated"]


class _SessionStore:
    def __init__(
        self,
        sql: LifecycleSql,
        transaction: Callable[[Callable[[], object]], object],
    ) -> None:
        self._sql = sql
        self._transaction = transaction
        self._tails: dict[str, _SessionTail] = {}
        self._fts_active: bool | None = None

    def schema_is_compatible(self) -> bool:
        for table, expected in _TABLE_COLUMNS.items():
            columns = self._sql.execute(f"PRAGMA table_info({table})")
            if columns and tuple(row["name"] for row in columns) != expected:
                return False
        return True

    def prepare(self) -> None:
        self._sql.execute("""
        CREATE TABLE IF NOT EXISTS cf_agents_session_messages (
          session_id TEXT NOT NULL,
          id TEXT NOT NULL,
          seq INTEGER NOT NULL,
          parent_id TEXT,
          type TEXT NOT NULL DEFAULT 'message',
          role TEXT NOT NULL,
          content TEXT NOT NULL,
          content_chunks INTEGER NOT NULL DEFAULT 0,
          token_estimate INTEGER NOT NULL DEFAULT 0,
          created_at INTEGER NOT NULL,
          PRIMARY KEY (session_id, id)
        ) WITHOUT ROWID
        """)
        self._sql.execute("""
        CREATE TABLE IF NOT EXISTS cf_agents_session_message_chunks (
          session_id TEXT NOT NULL,
          id TEXT NOT NULL,
          idx INTEGER NOT NULL,
          content TEXT NOT NULL,
          PRIMARY KEY (session_id, id, idx)
        ) WITHOUT ROWID
        """)
        self._sql.execute("""
        CREATE TABLE IF NOT EXISTS cf_agents_session_compactions (
          session_id TEXT NOT NULL,
          id TEXT NOT NULL,
          seq INTEGER NOT NULL,
          summary TEXT NOT NULL,
          from_message_id TEXT NOT NULL,
          to_message_id TEXT NOT NULL,
          created_at INTEGER NOT NULL,
          PRIMARY KEY (session_id, id)
        ) WITHOUT ROWID
        """)
        self._sql.execute("""
        CREATE TABLE IF NOT EXISTS cf_agents_session_config (
          session_id TEXT NOT NULL,
          key TEXT NOT NULL,
          value TEXT NOT NULL,
          PRIMARY KEY (session_id, key)
        ) WITHOUT ROWID
        """)
        self._sql.execute("""
        CREATE TABLE IF NOT EXISTS cf_agents_session_attachment_meta (
          hash TEXT PRIMARY KEY,
          bytes INTEGER NOT NULL,
          media_type TEXT NOT NULL,
          chunks INTEGER NOT NULL
        ) WITHOUT ROWID
        """)
        self._sql.execute("""
        CREATE TABLE IF NOT EXISTS cf_agents_session_attachment_chunks (
          hash TEXT NOT NULL,
          idx INTEGER NOT NULL,
          data BLOB NOT NULL,
          PRIMARY KEY (hash, idx)
        ) WITHOUT ROWID
        """)
        self._sql.execute("""
        CREATE TABLE IF NOT EXISTS cf_agents_session_attachment_refs (
          session_id TEXT NOT NULL,
          message_id TEXT NOT NULL,
          hash TEXT NOT NULL,
          PRIMARY KEY (session_id, message_id, hash)
        ) WITHOUT ROWID
        """)
        self._fts_active = self._detect_fts()

    def migrate_legacy(self) -> tuple[_LegacyMigrationIncomplete, ...]:
        incomplete = []
        for migrate in (
            self._migrate_legacy_messages,
            self._migrate_legacy_compactions,
        ):
            report = migrate()
            if report is not None:
                incomplete.append(report)

        for table in ("assistant_sessions", "assistant_fts"):
            if not self._table_exists(table):
                continue
            try:
                self._sql.execute(f"DROP TABLE {table}")
            except Exception:
                incomplete.append(
                    _LegacyMigrationIncomplete(
                        table,
                        self._safe_row_count(table),
                        0,
                    )
                )

        self._tails.clear()
        return tuple(incomplete)

    def _migrate_legacy_messages(self) -> _LegacyMigrationIncomplete | None:
        return self._migrate_legacy_source(
            source="assistant_messages",
            destination="cf_agents_session_messages",
            payload="content",
            columns=("id", "session_id", "parent_id", "role", "content", "created_at"),
            insert="""
            INSERT OR IGNORE INTO cf_agents_session_messages
              (session_id, id, seq, parent_id, role, content,
               token_estimate, created_at)
            SELECT session_id, id,
              ROW_NUMBER() OVER (
                PARTITION BY session_id ORDER BY created_at ASC, rowid ASC
              ),
              parent_id, role, content,
              CAST(LENGTH(CAST(content AS BLOB)) / 4 AS INTEGER),
              COALESCE(CAST(strftime('%s', created_at) AS INTEGER), 0) * 1000
            FROM assistant_messages
            """,
            reset_fts=True,
        )

    def _migrate_legacy_compactions(self) -> _LegacyMigrationIncomplete | None:
        return self._migrate_legacy_source(
            source="assistant_compactions",
            destination="cf_agents_session_compactions",
            payload="summary",
            columns=(
                "id",
                "session_id",
                "summary",
                "from_message_id",
                "to_message_id",
                "created_at",
            ),
            insert="""
            INSERT OR IGNORE INTO cf_agents_session_compactions
              (session_id, id, seq, summary, from_message_id,
               to_message_id, created_at)
            SELECT session_id, id,
              ROW_NUMBER() OVER (
                PARTITION BY session_id ORDER BY created_at ASC, rowid ASC
              ),
              summary, from_message_id, to_message_id,
              COALESCE(CAST(strftime('%s', created_at) AS INTEGER), 0) * 1000
            FROM assistant_compactions
            """,
        )

    def _migrate_legacy_source(
        self,
        *,
        source: str,
        destination: str,
        payload: str,
        columns: Sequence[str],
        insert: str,
        reset_fts: bool = False,
    ) -> _LegacyMigrationIncomplete | None:
        if not self._table_exists(source):
            return None
        try:
            source_count = self._row_count(source)
        except Exception:
            return _LegacyMigrationIncomplete(source, 0, 0)
        try:
            compatible = self._has_columns(source, columns)
        except Exception:
            compatible = False
        if not compatible:
            copied = self._safe_copied_count(source, destination, payload)
            return _LegacyMigrationIncomplete(source, source_count, copied)
        try:
            self._sql.execute(insert)
            if reset_fts and self._fts_is_active():
                self._sql.execute("DROP TABLE cf_agents_session_fts")
                self._fts_active = False
            copied = self._copied_count(source, destination, payload)
            if copied != source_count:
                return _LegacyMigrationIncomplete(source, source_count, copied)
            self._sql.execute(f"DROP TABLE {source}")
            return None
        except Exception:
            copied = self._safe_copied_count(source, destination, payload)
            return _LegacyMigrationIncomplete(source, source_count, copied)

    def _table_exists(self, table: str) -> bool:
        return bool(
            self._sql.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
                table,
            )
        )

    def _has_columns(self, table: str, required: Sequence[str]) -> bool:
        columns = {
            row["name"]
            for row in self._sql.execute(f"PRAGMA table_info({table})")
            if isinstance(row.get("name"), str)
        }
        return set(required) <= columns

    def _row_count(self, table: str) -> int:
        rows = self._sql.execute(f"SELECT COUNT(*) AS count FROM {table}")
        return cast(int, rows[0]["count"])

    def _safe_row_count(self, table: str) -> int:
        try:
            return self._row_count(table)
        except Exception:
            return 0

    def _copied_count(self, source: str, destination: str, payload: str) -> int:
        rows = self._sql.execute(
            f"SELECT COUNT(*) AS count FROM {source} AS legacy "
            f"JOIN {destination} AS lifted "
            "ON lifted.session_id = legacy.session_id "
            "AND lifted.id = legacy.id "
            f"AND lifted.{payload} = legacy.{payload}"
        )
        return cast(int, rows[0]["count"])

    def _safe_copied_count(
        self,
        source: str,
        destination: str,
        payload: str,
    ) -> int:
        try:
            return self._copied_count(source, destination, payload)
        except Exception:
            return 0

    def get(self, session_id: str, message_id: str) -> dict[str, object] | None:
        rows = self._sql.execute(
            "SELECT id, content, content_chunks FROM cf_agents_session_messages "
            "WHERE session_id = ? AND id = ?",
            session_id,
            message_id,
        )
        if not rows:
            return None
        content = self._reassemble(session_id, rows[0])
        message = None if content is None else parse_session_message(content)
        if message is None:
            return None
        return self._resolve_messages(session_id, [(message_id, message)])[0]

    def exists(self, session_id: str, message_id: str) -> bool:
        return bool(
            self._sql.execute(
                "SELECT id FROM cf_agents_session_messages "
                "WHERE session_id = ? AND id = ?",
                session_id,
                message_id,
            )
        )

    def latest_leaf_id(self, session_id: str) -> str | None:
        return self._tail(session_id).leaf_id

    def branches(
        self,
        session_id: str,
        message_id: str,
    ) -> list[dict[str, object]]:
        rows = self._sql.execute(
            "SELECT id, content, content_chunks FROM cf_agents_session_messages "
            "WHERE session_id = ? AND parent_id = ? ORDER BY seq ASC",
            session_id,
            message_id,
        )
        messages = []
        contents = self._reassemble_many(session_id, rows)
        for row in rows:
            content = contents.get(cast(str, row["id"]))
            if content is None:
                continue
            message = parse_session_message(content)
            if message is not None:
                messages.append((cast(str, row["id"]), message))
        return self._resolve_messages(session_id, messages)

    def search(
        self,
        session_id: str,
        query: str,
        limit: int | float,
    ) -> list[dict[str, object]]:
        self._ensure_fts()
        if not query:
            return []
        sanitized = _sqlite_text(query).replace('"', '""')
        phrase = f'"{sanitized}"'
        return self._sql.execute(
            "SELECT f.id, f.role, f.content FROM cf_agents_session_fts AS f "
            "JOIN cf_agents_session_messages AS message "
            "ON message.session_id = f.session_id AND message.id = f.id "
            "WHERE cf_agents_session_fts MATCH ? AND f.session_id = ? "
            "ORDER BY rank LIMIT ?",
            phrase,
            session_id,
            limit,
        )

    def path(
        self,
        session_id: str,
        leaf_id: str | None,
    ) -> list[_SessionPathRow]:
        leaf = self._resolve_leaf_id(session_id, leaf_id)
        if leaf is None:
            return []
        rows = self._sql.execute(
            f"""
            WITH RECURSIVE path(id, parent_id, depth) AS (
              SELECT id, parent_id, 0 FROM cf_agents_session_messages
              WHERE session_id = ? AND id = ?
              UNION ALL
              SELECT message.id, message.parent_id, path.depth + 1
              FROM cf_agents_session_messages AS message
              JOIN path ON message.id = path.parent_id
              WHERE message.session_id = ? AND path.depth < {_MAX_PATH_DEPTH}
            )
            SELECT message.id, message.parent_id, message.role,
                   LENGTH(CAST(message.content AS BLOB)) + COALESCE((
                     SELECT SUM(LENGTH(CAST(chunk.content AS BLOB)))
                     FROM cf_agents_session_message_chunks AS chunk
                     WHERE chunk.session_id = message.session_id
                       AND chunk.id = message.id
                   ), 0) + COALESCE((
                     SELECT SUM(((meta.bytes + 2) / 3) * 4)
                     FROM cf_agents_session_attachment_refs AS ref
                     JOIN cf_agents_session_attachment_meta AS meta
                       ON meta.hash = ref.hash
                     WHERE ref.session_id = message.session_id
                       AND ref.message_id = message.id
                   ), 0) AS bytes,
                   message.token_estimate
            FROM path
            JOIN cf_agents_session_messages AS message
              ON message.session_id = ? AND message.id = path.id
            ORDER BY path.depth DESC
            """,
            session_id,
            leaf,
            session_id,
            session_id,
        )
        return [
            _SessionPathRow(
                id=cast(str, row["id"]),
                parent_id=cast(str | None, row["parent_id"]),
                role=cast(str, row["role"]),
                bytes=cast(int, row["bytes"]),
                token_estimate=cast(int, row["token_estimate"]),
            )
            for row in rows
        ]

    def hydrate(
        self,
        session_id: str,
        path: Sequence[_SessionPathRow],
    ) -> Iterator[dict[str, object]]:
        for window in _hydration_windows(path):
            encoded_ids = dumps_session_json([row.id for row in window])
            rows = self._sql.execute(
                "SELECT id, content, content_chunks "
                "FROM cf_agents_session_messages "
                "WHERE session_id = ? "
                "AND id IN (SELECT value FROM json_each(?))",
                session_id,
                encoded_ids,
            )
            contents = self._reassemble_many(session_id, rows)
            messages = []
            for path_row in window:
                content = contents.get(path_row.id)
                if content is None:
                    continue
                message = parse_session_message(content)
                if message is not None:
                    messages.append((path_row.id, message))
            yield from self._resolve_messages(session_id, messages)

    def append(
        self,
        session_id: str,
        message: dict[str, object],
        parent_id: str | None | object,
        omitted_parent: object,
        token_estimate: int,
    ) -> tuple[bool, dict[str, object]]:
        message_id = cast(str, message["id"])
        existing = self.get(session_id, message_id)
        if existing is not None:
            return False, existing
        tail = self._tail(session_id)
        if parent_id is omitted_parent:
            parent = tail.leaf_id
        elif isinstance(parent_id, str) and self.exists(session_id, parent_id):
            parent = parent_id
        else:
            parent = None
        stored_message, attachments = _extract_attachments(message)
        content = dumps_session_json(stored_message)
        slices = _split_content(content)
        seq = tail.next_seq
        fts_active = self._fts_is_active()

        def write() -> None:
            self._put_attachments(attachments)
            self._sql.execute(
                "INSERT INTO cf_agents_session_messages "
                "(session_id, id, seq, parent_id, role, content, "
                "content_chunks, token_estimate, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                session_id,
                message_id,
                seq,
                parent,
                message["role"],
                slices[0],
                len(slices) - 1,
                token_estimate,
                now_ms(),
            )
            self._write_continuations(session_id, message_id, slices)
            self._add_attachment_refs(
                session_id,
                message_id,
                [attachment.hash for attachment in attachments],
            )
            if fts_active:
                self._replace_fts(session_id, stored_message)

        self._transaction(write)
        self._tails[session_id] = _SessionTail(message_id, seq + 1)
        return True, message

    def update(
        self,
        session_id: str,
        message: dict[str, object],
        token_estimate: int,
    ) -> _UpdateOutcome:
        message_id = cast(str, message["id"])
        rows = self._sql.execute(
            "SELECT id, content, content_chunks FROM cf_agents_session_messages "
            "WHERE session_id = ? AND id = ?",
            session_id,
            message_id,
        )
        if not rows:
            return "missing"
        stored_message, attachments = _extract_attachments(message)
        content = dumps_session_json(stored_message)
        if self._reassemble(session_id, rows[0]) == content:
            return "unchanged"
        slices = _split_content(content)
        fts_active = self._fts_is_active()

        def write() -> None:
            self._put_attachments(attachments)
            self._sql.execute(
                "UPDATE cf_agents_session_messages "
                "SET role = ?, content = ?, content_chunks = ?, token_estimate = ? "
                "WHERE session_id = ? AND id = ?",
                message["role"],
                slices[0],
                len(slices) - 1,
                token_estimate,
                session_id,
                message_id,
            )
            self._sql.execute(
                "DELETE FROM cf_agents_session_message_chunks "
                "WHERE session_id = ? AND id = ? "
                "AND (typeof(idx) != 'integer' OR idx < 1 OR idx > ?)",
                session_id,
                message_id,
                len(slices) - 1,
            )
            self._write_continuations(session_id, message_id, slices, update=True)
            self._replace_attachment_refs(
                session_id,
                message_id,
                [attachment.hash for attachment in attachments],
            )
            if fts_active:
                self._replace_fts(session_id, stored_message)

        self._transaction(write)
        return "updated"

    def import_message(
        self,
        session_id: str,
        message: dict[str, object],
        parent_id: str | None,
        created_at_ms: int,
        token_estimate: int,
    ) -> bool:
        message_id = cast(str, message["id"])
        if self.exists(session_id, message_id):
            return False
        tail = self._tail(session_id)
        stored_message, attachments = _extract_attachments(message)
        slices = _split_content(dumps_session_json(stored_message))
        inserted = False
        fts_active = self._fts_is_active()

        def write() -> None:
            nonlocal inserted
            self._sql.execute(
                "INSERT OR IGNORE INTO cf_agents_session_messages "
                "(session_id, id, seq, parent_id, role, content, content_chunks, "
                "token_estimate, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                session_id,
                message_id,
                tail.next_seq,
                parent_id,
                message["role"],
                slices[0],
                len(slices) - 1,
                token_estimate,
                created_at_ms,
            )
            inserted = _changes(self._sql) > 0
            if inserted:
                self._put_attachments(attachments)
                self._write_continuations(session_id, message_id, slices)
                self._add_attachment_refs(
                    session_id,
                    message_id,
                    [attachment.hash for attachment in attachments],
                )
                if fts_active:
                    self._replace_fts(session_id, stored_message)

        self._transaction(write)
        if inserted:
            self._tails[session_id] = _SessionTail(message_id, tail.next_seq + 1)
        return inserted

    def add_compaction(
        self,
        session_id: str,
        summary: str,
        from_message_id: str,
        to_message_id: str,
    ) -> _StoredCompaction:
        rows = self._sql.execute(
            "SELECT COALESCE(MAX(seq), 0) + 1 AS seq "
            "FROM cf_agents_session_compactions WHERE session_id = ?",
            session_id,
        )
        seq = cast(int, rows[0]["seq"])
        compaction = _StoredCompaction(
            id=str(uuid4()),
            summary=summary,
            from_message_id=from_message_id,
            to_message_id=to_message_id,
            created_at_ms=now_ms(),
        )
        self._sql.execute(
            "INSERT INTO cf_agents_session_compactions "
            "(session_id, id, seq, summary, from_message_id, to_message_id, "
            "created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            session_id,
            compaction.id,
            seq,
            compaction.summary,
            compaction.from_message_id,
            compaction.to_message_id,
            compaction.created_at_ms,
        )
        return compaction

    def compactions(self, session_id: str) -> list[_StoredCompaction]:
        rows = self._sql.execute(
            "SELECT id, summary, from_message_id, to_message_id, created_at "
            "FROM cf_agents_session_compactions WHERE session_id = ? "
            "AND typeof(seq) = 'integer' "
            "ORDER BY seq ASC",
            session_id,
        )
        return [
            compaction
            for row in rows
            if (compaction := _stored_compaction(row)) is not None
        ]

    def compaction(
        self,
        session_id: str,
        compaction_id: str,
    ) -> _StoredCompaction | None:
        rows = self._sql.execute(
            "SELECT id, summary, from_message_id, to_message_id, created_at "
            "FROM cf_agents_session_compactions "
            "WHERE session_id = ? AND id = ?",
            session_id,
            compaction_id,
        )
        return None if not rows else _stored_compaction(rows[0])

    def compaction_ranges(self, session_id: str) -> Iterator[_StoredCompaction]:
        cursor: tuple[int, str] | None = None
        while True:
            if cursor is None:
                rows = self._sql.execute(
                    "SELECT id, seq, from_message_id, to_message_id "
                    "FROM cf_agents_session_compactions "
                    "WHERE session_id = ? AND typeof(seq) = 'integer' "
                    "AND typeof(id) = 'text' "
                    "AND typeof(summary) = 'text' "
                    "AND typeof(from_message_id) = 'text' "
                    "AND typeof(to_message_id) = 'text' "
                    "AND typeof(created_at) = 'integer' "
                    "AND created_at BETWEEN ? AND ? "
                    "ORDER BY seq, id LIMIT ?",
                    session_id,
                    _MIN_DATETIME_MS,
                    _MAX_DATETIME_MS,
                    _COMPACTION_PAGE_ROWS,
                )
            else:
                seq, compaction_id = cursor
                rows = self._sql.execute(
                    "SELECT id, seq, from_message_id, to_message_id "
                    "FROM cf_agents_session_compactions "
                    "WHERE session_id = ? AND typeof(seq) = 'integer' "
                    "AND typeof(id) = 'text' "
                    "AND typeof(summary) = 'text' "
                    "AND typeof(from_message_id) = 'text' "
                    "AND typeof(to_message_id) = 'text' "
                    "AND typeof(created_at) = 'integer' "
                    "AND created_at BETWEEN ? AND ? "
                    "AND (seq > ? OR (seq = ? AND id > ?)) "
                    "ORDER BY seq, id LIMIT ?",
                    session_id,
                    _MIN_DATETIME_MS,
                    _MAX_DATETIME_MS,
                    seq,
                    seq,
                    compaction_id,
                    _COMPACTION_PAGE_ROWS,
                )
            if not rows:
                return
            for row in rows:
                yield _StoredCompaction(
                    id=cast(str, row["id"]),
                    summary="",
                    from_message_id=cast(str, row["from_message_id"]),
                    to_message_id=cast(str, row["to_message_id"]),
                    created_at_ms=0,
                )
            last = rows[-1]
            seq = last.get("seq")
            compaction_id = last.get("id")
            if type(seq) is not int or not isinstance(compaction_id, str):
                return
            cursor = seq, compaction_id

    def delete_messages(self, session_id: str, message_ids: Sequence[str]) -> int:
        unique_ids = tuple(dict.fromkeys(message_ids))
        if not unique_ids:
            return 0
        encoded_ids = dumps_session_json(list(unique_ids))
        fts_active = self._fts_is_active()

        def write() -> None:
            self._release_attachment_messages(session_id, unique_ids)
            self._sql.execute(
                """
                WITH RECURSIVE
                deleted(id) AS (SELECT value FROM json_each(?)),
                rewire(child_id, ancestor_id, depth) AS (
                  SELECT child.id, child.parent_id, 0
                  FROM cf_agents_session_messages AS child
                  JOIN deleted ON deleted.id = child.parent_id
                  WHERE child.session_id = ?
                    AND child.id NOT IN (SELECT id FROM deleted)
                  UNION ALL
                  SELECT rewire.child_id, parent.parent_id, rewire.depth + 1
                  FROM rewire
                  JOIN cf_agents_session_messages AS parent
                    ON parent.id = rewire.ancestor_id
                  JOIN deleted ON deleted.id = parent.id
                  WHERE parent.session_id = ? AND rewire.depth < 10000
                ),
                nearest(child_id, ancestor_id) AS (
                  SELECT child_id, ancestor_id FROM rewire
                  WHERE ancestor_id IS NULL
                     OR ancestor_id NOT IN (SELECT id FROM deleted)
                )
                UPDATE cf_agents_session_messages
                SET parent_id = (
                  SELECT nearest.ancestor_id FROM nearest
                  WHERE nearest.child_id = cf_agents_session_messages.id
                )
                WHERE session_id = ?
                  AND id IN (SELECT child_id FROM nearest)
                """,
                encoded_ids,
                session_id,
                session_id,
                session_id,
            )
            self._sql.execute(
                "DELETE FROM cf_agents_session_messages WHERE session_id = ? "
                "AND id IN (SELECT value FROM json_each(?))",
                session_id,
                encoded_ids,
            )
            self._sql.execute(
                "DELETE FROM cf_agents_session_message_chunks WHERE session_id = ? "
                "AND id IN (SELECT value FROM json_each(?))",
                session_id,
                encoded_ids,
            )
            if fts_active:
                self._sql.execute(
                    "DELETE FROM cf_agents_session_fts WHERE session_id = ? "
                    "AND id IN (SELECT value FROM json_each(?))",
                    session_id,
                    encoded_ids,
                )

        self._transaction(write)
        self._tails.pop(session_id, None)
        return len(unique_ids)

    def clear(self, session_id: str) -> None:
        fts_active = self._fts_is_active()

        def write() -> None:
            self._release_attachment_session(session_id)
            for table in (
                "cf_agents_session_messages",
                "cf_agents_session_message_chunks",
                "cf_agents_session_compactions",
            ):
                self._sql.execute(
                    f"DELETE FROM {table} WHERE session_id = ?", session_id
                )
            if fts_active:
                self._sql.execute(
                    "DELETE FROM cf_agents_session_fts WHERE session_id = ?",
                    session_id,
                )

        self._transaction(write)
        self._tails[session_id] = _SessionTail(None, 1)

    def _detect_fts(self) -> bool:
        return bool(
            self._sql.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                "AND name = 'cf_agents_session_fts' LIMIT 1"
            )
        )

    def _fts_is_active(self) -> bool:
        if self._fts_active is None:
            self._fts_active = self._detect_fts()
        return self._fts_active

    def _ensure_fts(self) -> None:
        if self._fts_is_active():
            return

        def activate() -> None:
            self._sql.execute("""
            CREATE VIRTUAL TABLE cf_agents_session_fts
            USING fts5(
              id UNINDEXED,
              session_id UNINDEXED,
              role UNINDEXED,
              content,
              tokenize='porter unicode61'
            )
            """)
            self._backfill_fts()

        self._transaction(activate)
        self._fts_active = True

    def _backfill_fts(self) -> None:
        cursor: tuple[str, str] | None = None
        while True:
            if cursor is None:
                rows = self._sql.execute(
                    "SELECT session_id, id FROM cf_agents_session_messages "
                    "WHERE typeof(session_id) = 'text' AND typeof(id) = 'text' "
                    "ORDER BY session_id, id LIMIT ?",
                    _FTS_BACKFILL_ROWS,
                )
            else:
                session_id, message_id = cursor
                rows = self._sql.execute(
                    "SELECT session_id, id FROM cf_agents_session_messages "
                    "WHERE typeof(session_id) = 'text' AND typeof(id) = 'text' "
                    "AND (session_id > ? OR (session_id = ? AND id > ?)) "
                    "ORDER BY session_id, id LIMIT ?",
                    session_id,
                    session_id,
                    message_id,
                    _FTS_BACKFILL_ROWS,
                )
            if not rows:
                return
            for row in rows:
                session_id = row.get("session_id")
                message_id = row.get("id")
                if not isinstance(session_id, str) or not isinstance(message_id, str):
                    continue
                message_rows = self._sql.execute(
                    "SELECT id, role, content, content_chunks "
                    "FROM cf_agents_session_messages "
                    "WHERE session_id = ? AND id = ?",
                    session_id,
                    message_id,
                )
                if not message_rows:
                    continue
                role = message_rows[0].get("role")
                if not isinstance(role, str):
                    continue
                content = self._reassemble(session_id, message_rows[0])
                message = None if content is None else parse_session_message(content)
                if message is not None:
                    self._insert_fts_row(
                        message_id,
                        session_id,
                        role,
                        _search_content(message, skip_empty=True),
                    )
            last = rows[-1]
            last_session_id = last.get("session_id")
            last_message_id = last.get("id")
            if not isinstance(last_session_id, str) or not isinstance(
                last_message_id, str
            ):
                return
            cursor = last_session_id, last_message_id

    def _replace_fts(
        self,
        session_id: str,
        message: dict[str, object],
    ) -> None:
        message_id = cast(str, message["id"])
        content = _sqlite_text(_search_content(message))
        rows = self._sql.execute(
            "SELECT content FROM cf_agents_session_fts WHERE session_id = ? AND id = ?",
            session_id,
            message_id,
        )
        if rows and rows[0].get("content") == content:
            return
        if rows:
            self._sql.execute(
                "DELETE FROM cf_agents_session_fts WHERE session_id = ? AND id = ?",
                session_id,
                message_id,
            )
        if content:
            self._insert_fts(session_id, message, content)

    def _insert_fts(
        self,
        session_id: str,
        message: dict[str, object],
        content: str | None = None,
    ) -> None:
        indexed = _search_content(message) if content is None else content
        if not indexed:
            return
        self._insert_fts_row(
            cast(str, message["id"]),
            session_id,
            cast(str, message["role"]),
            indexed,
        )

    def _insert_fts_row(
        self,
        message_id: str,
        session_id: str,
        role: str,
        content: str,
    ) -> None:
        content = _sqlite_text(content)
        if not content:
            return
        self._sql.execute(
            "INSERT INTO cf_agents_session_fts (id, session_id, role, content) "
            "VALUES (?, ?, ?, ?)",
            message_id,
            session_id,
            role,
            content,
        )

    def _tail(self, session_id: str) -> _SessionTail:
        cached = self._tails.get(session_id)
        if cached is not None:
            return cached
        rows = self._sql.execute(
            "SELECT id, seq FROM cf_agents_session_messages "
            "WHERE session_id = ? ORDER BY seq DESC LIMIT 1",
            session_id,
        )
        tail = (
            _SessionTail(cast(str, rows[0]["id"]), cast(int, rows[0]["seq"]) + 1)
            if rows
            else _SessionTail(None, 1)
        )
        self._tails[session_id] = tail
        return tail

    def _resolve_leaf_id(self, session_id: str, leaf_id: str | None) -> str | None:
        if leaf_id:
            return leaf_id if self.exists(session_id, leaf_id) else None
        return self.latest_leaf_id(session_id)

    def _put_attachments(
        self,
        attachments: Sequence[_PendingAttachment],
    ) -> None:
        for attachment in attachments:
            existing = self._sql.execute(
                "SELECT hash FROM cf_agents_session_attachment_meta WHERE hash = ?",
                attachment.hash,
            )
            if existing:
                continue
            orphaned = self._sql.execute(
                "SELECT hash FROM cf_agents_session_attachment_chunks "
                "WHERE hash = ? LIMIT 1",
                attachment.hash,
            )
            if orphaned:
                self._sql.execute(
                    "DELETE FROM cf_agents_session_attachment_chunks WHERE hash = ?",
                    attachment.hash,
                )
            chunks = 0
            for start in range(0, len(attachment.data), _ATTACHMENT_CHUNK_BYTES):
                self._sql.execute(
                    "INSERT INTO cf_agents_session_attachment_chunks "
                    "(hash, idx, data) VALUES (?, ?, ?)",
                    attachment.hash,
                    chunks,
                    attachment.data[start : start + _ATTACHMENT_CHUNK_BYTES],
                )
                chunks += 1
            self._sql.execute(
                "INSERT INTO cf_agents_session_attachment_meta "
                "(hash, bytes, media_type, chunks) VALUES (?, ?, ?, ?)",
                attachment.hash,
                len(attachment.data),
                attachment.media_type,
                chunks,
            )

    def _add_attachment_refs(
        self,
        session_id: str,
        message_id: str,
        hashes: Sequence[str],
    ) -> None:
        for digest in hashes:
            self._sql.execute(
                "INSERT OR IGNORE INTO cf_agents_session_attachment_refs "
                "(session_id, message_id, hash) VALUES (?, ?, ?)",
                session_id,
                message_id,
                digest,
            )

    def _replace_attachment_refs(
        self,
        session_id: str,
        message_id: str,
        hashes: Sequence[str],
    ) -> None:
        rows = self._sql.execute(
            "SELECT hash FROM cf_agents_session_attachment_refs "
            "WHERE session_id = ? AND message_id = ?",
            session_id,
            message_id,
        )
        held = {
            cast(str, row["hash"]) for row in rows if isinstance(row.get("hash"), str)
        }
        wanted = set(hashes)
        dropped = held - wanted
        for digest in dropped:
            self._sql.execute(
                "DELETE FROM cf_agents_session_attachment_refs "
                "WHERE session_id = ? AND message_id = ? AND hash = ?",
                session_id,
                message_id,
                digest,
            )
        self._add_attachment_refs(
            session_id,
            message_id,
            [digest for digest in hashes if digest not in held],
        )
        self._collect_attachments(dropped)

    def _release_attachment_messages(
        self,
        session_id: str,
        message_ids: Sequence[str],
    ) -> None:
        encoded_ids = dumps_session_json(list(message_ids))
        rows = self._sql.execute(
            "SELECT DISTINCT hash FROM cf_agents_session_attachment_refs "
            "WHERE session_id = ? "
            "AND message_id IN (SELECT value FROM json_each(?))",
            session_id,
            encoded_ids,
        )
        if not rows:
            return
        hashes = [row["hash"] for row in rows if isinstance(row.get("hash"), str)]
        self._sql.execute(
            "DELETE FROM cf_agents_session_attachment_refs "
            "WHERE session_id = ? "
            "AND message_id IN (SELECT value FROM json_each(?))",
            session_id,
            encoded_ids,
        )
        self._collect_attachments(hashes)

    def _release_attachment_session(self, session_id: str) -> None:
        rows = self._sql.execute(
            "SELECT DISTINCT hash FROM cf_agents_session_attachment_refs "
            "WHERE session_id = ?",
            session_id,
        )
        if not rows:
            return
        hashes = [row["hash"] for row in rows if isinstance(row.get("hash"), str)]
        self._sql.execute(
            "DELETE FROM cf_agents_session_attachment_refs WHERE session_id = ?",
            session_id,
        )
        self._collect_attachments(hashes)

    def _collect_attachments(self, hashes: Iterable[str]) -> None:
        for digest in hashes:
            referenced = self._sql.execute(
                "SELECT hash FROM cf_agents_session_attachment_refs "
                "WHERE hash = ? LIMIT 1",
                digest,
            )
            if referenced:
                continue
            self._sql.execute(
                "DELETE FROM cf_agents_session_attachment_chunks WHERE hash = ?",
                digest,
            )
            self._sql.execute(
                "DELETE FROM cf_agents_session_attachment_meta WHERE hash = ?",
                digest,
            )

    def _resolve_messages(
        self,
        session_id: str,
        messages: Sequence[tuple[str, dict[str, object]]],
    ) -> list[dict[str, object]]:
        requested = {
            message_id: _attachment_hashes(message) for message_id, message in messages
        }
        pointer_ids = [message_id for message_id, hashes in requested.items() if hashes]
        if not pointer_ids:
            return [message for _, message in messages]
        rows = self._sql.execute(
            "SELECT message_id, hash FROM cf_agents_session_attachment_refs "
            "WHERE session_id = ? "
            "AND message_id IN (SELECT value FROM json_each(?))",
            session_id,
            dumps_session_json(pointer_ids),
        )
        allowed: dict[str, set[str]] = {message_id: set() for message_id in pointer_ids}
        for row in rows:
            message_id = row.get("message_id")
            digest = row.get("hash")
            if (
                isinstance(message_id, str)
                and message_id in allowed
                and isinstance(digest, str)
            ):
                allowed[message_id].add(digest)
        cache: dict[str, _StoredAttachment | None] = {}
        resolved = []
        for message_id, message in messages:

            def load(digest: str) -> _StoredAttachment | None:
                if digest not in allowed.get(message_id, set()):
                    return None
                if digest not in cache:
                    cache[digest] = self._get_attachment(digest)
                return cache[digest]

            resolved.append(_resolve_attachments(message, load))
        return resolved

    def _get_attachment(self, digest: str) -> _StoredAttachment | None:
        rows = self._sql.execute(
            "SELECT bytes, media_type, chunks "
            "FROM cf_agents_session_attachment_meta WHERE hash = ?",
            digest,
        )
        if not rows:
            return None
        row = rows[0]
        byte_count = row.get("bytes")
        media_type = row.get("media_type")
        chunk_count = row.get("chunks")
        if (
            type(byte_count) is not int
            or byte_count < 0
            or not isinstance(media_type, str)
            or type(chunk_count) is not int
            or chunk_count < 0
        ):
            return None
        expected_chunks = (
            byte_count + _ATTACHMENT_CHUNK_BYTES - 1
        ) // _ATTACHMENT_CHUNK_BYTES
        if chunk_count != expected_chunks:
            return None
        chunks = self._sql.execute(
            "SELECT idx, data FROM cf_agents_session_attachment_chunks "
            "WHERE hash = ? ORDER BY idx",
            digest,
        )
        if len(chunks) != chunk_count:
            return None
        parts = []
        for expected_index, chunk in enumerate(chunks):
            data = chunk.get("data")
            if type(chunk.get("idx")) is not int or chunk["idx"] != expected_index:
                return None
            if not isinstance(data, (bytes, bytearray, memoryview)):
                return None
            part = bytes(data)
            expected_bytes = min(
                _ATTACHMENT_CHUNK_BYTES,
                byte_count - expected_index * _ATTACHMENT_CHUNK_BYTES,
            )
            if len(part) != expected_bytes:
                return None
            parts.append(part)
        data = b"".join(parts)
        if hashlib.sha256(data).hexdigest() != digest:
            return None
        return _StoredAttachment(media_type, data)

    def _write_continuations(
        self,
        session_id: str,
        message_id: str,
        slices: Sequence[str],
        *,
        update: bool = False,
    ) -> None:
        statement = (
            "INSERT INTO cf_agents_session_message_chunks "
            "(session_id, id, idx, content) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(session_id, id, idx) DO UPDATE SET content = excluded.content"
            if update
            else "INSERT INTO cf_agents_session_message_chunks "
            "(session_id, id, idx, content) VALUES (?, ?, ?, ?)"
        )
        for index, content in enumerate(slices[1:], start=1):
            self._sql.execute(
                statement,
                session_id,
                message_id,
                index,
                content,
            )

    def _reassemble(
        self,
        session_id: str,
        row: dict[str, object],
    ) -> str | None:
        message_id = row.get("id")
        if not isinstance(message_id, str):
            return None
        return self._reassemble_many(session_id, [row]).get(message_id)

    def _reassemble_many(
        self,
        session_id: str,
        rows: Sequence[dict[str, object]],
    ) -> dict[str, str]:
        contents: dict[str, str] = {}
        expected_chunks: dict[str, int] = {}
        for row in rows:
            message_id = row.get("id")
            content = row.get("content")
            chunk_count = row.get("content_chunks")
            if (
                not isinstance(message_id, str)
                or not isinstance(content, str)
                or type(chunk_count) is not int
                or chunk_count < 0
            ):
                continue
            contents[message_id] = content
            if chunk_count:
                expected_chunks[message_id] = chunk_count
        chunked_ids = list(expected_chunks)
        if not chunked_ids:
            return contents
        encoded_ids = dumps_session_json(chunked_ids)
        continuations = self._sql.execute(
            "SELECT id, idx, content "
            "FROM cf_agents_session_message_chunks "
            "WHERE session_id = ? "
            "AND id IN (SELECT value FROM json_each(?)) "
            "ORDER BY id, idx",
            session_id,
            encoded_ids,
        )
        fragments: dict[str, list[tuple[int, str]]] = {
            message_id: [] for message_id in chunked_ids
        }
        invalid_ids: set[str] = set()
        for row in continuations:
            message_id = row.get("id")
            index = row.get("idx")
            content = row.get("content")
            if not isinstance(message_id, str) or message_id not in fragments:
                continue
            if type(index) is not int or not isinstance(content, str):
                invalid_ids.add(message_id)
                continue
            fragments[message_id].append((index, content))
        for message_id, expected in expected_chunks.items():
            pieces = fragments[message_id]
            valid_indexes = len(pieces) == expected and all(
                index == expected_index
                for expected_index, (index, _) in enumerate(pieces, start=1)
            )
            if message_id in invalid_ids or not valid_indexes:
                contents.pop(message_id, None)
                continue
            contents[message_id] += "".join(content for _, content in pieces)
        return contents


def _split_content(content: str) -> list[str]:
    if not content:
        return [""]
    if content.isascii():
        return [
            content[start : start + _MAX_INLINE_ROW_BYTES]
            for start in range(0, len(content), _MAX_INLINE_ROW_BYTES)
        ]
    slices = []
    start = 0
    slice_bytes = 0
    for index, character in enumerate(content):
        code = ord(character)
        if code < 0x80:
            width = 1
        elif code < 0x800:
            width = 2
        elif code < 0x10000:
            width = 3
        else:
            width = 4
        if slice_bytes and slice_bytes + width > _MAX_INLINE_ROW_BYTES:
            slices.append(content[start:index])
            start = index
            slice_bytes = 0
        slice_bytes += width
    slices.append(content[start:])
    return slices


def _search_content(
    message: dict[str, object],
    *,
    skip_empty: bool = False,
) -> str:
    parts = message.get("parts")
    if not isinstance(parts, list):
        return ""
    return " ".join(
        part["text"]
        for part in parts
        if isinstance(part, dict)
        and part.get("type") == "text"
        and isinstance(part.get("text"), str)
        and (not skip_empty or bool(part["text"]))
    )


def _stored_compaction(row: dict[str, object]) -> _StoredCompaction | None:
    compaction_id = row.get("id")
    summary = row.get("summary")
    from_message_id = row.get("from_message_id")
    to_message_id = row.get("to_message_id")
    created_at = row.get("created_at")
    if (
        not isinstance(compaction_id, str)
        or not isinstance(summary, str)
        or not isinstance(from_message_id, str)
        or not isinstance(to_message_id, str)
        or type(created_at) is not int
        or not _MIN_DATETIME_MS <= created_at <= _MAX_DATETIME_MS
    ):
        return None
    return _StoredCompaction(
        id=compaction_id,
        summary=summary,
        from_message_id=from_message_id,
        to_message_id=to_message_id,
        created_at_ms=created_at,
    )


def _sqlite_text(value: str) -> str:
    # SQLite text bindings reject lone surrogates; UTF-8 conversion replaces them.
    normalized = []
    index = 0
    while index < len(value):
        code = ord(value[index])
        if 0xD800 <= code <= 0xDBFF and index + 1 < len(value):
            low = ord(value[index + 1])
            if 0xDC00 <= low <= 0xDFFF:
                normalized.append(chr(0x10000 + ((code - 0xD800) << 10) + low - 0xDC00))
                index += 2
                continue
        normalized.append("\ufffd" if 0xD800 <= code <= 0xDFFF else value[index])
        index += 1
    return "".join(normalized)


def _hydration_windows(
    rows: Sequence[_SessionPathRow],
) -> Iterator[Sequence[_SessionPathRow]]:
    start = 0
    window_bytes = 0
    for index, row in enumerate(rows):
        window_size = index - start
        if window_size and (
            window_size >= _HYDRATION_MAX_ROWS
            or window_bytes + row.bytes > _HYDRATION_MAX_BYTES
        ):
            yield rows[start:index]
            start = index
            window_bytes = 0
        window_bytes += row.bytes
    if start < len(rows):
        yield rows[start:]


def _changes(sql: LifecycleSql) -> int:
    rows = sql.execute("SELECT changes() AS count")
    return cast(int, rows[0]["count"])
