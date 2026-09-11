from __future__ import annotations

import json

import fakes
import pytest

from agents.lifecycle import Lifecycle
from agents.sessions import HistoryReadOptions, Sessions, StoredCompaction


def _message(message_id: str, text: str, role: str) -> str:
    return json.dumps(
        {
            "id": message_id,
            "role": role,
            "parts": [{"type": "text", "text": text}],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _legacy_schema(ctx: fakes.FakeCtx) -> None:
    for statement in (
        """
        CREATE TABLE assistant_messages (
          id TEXT PRIMARY KEY,
          session_id TEXT NOT NULL DEFAULT '',
          parent_id TEXT,
          role TEXT NOT NULL,
          content TEXT NOT NULL,
          created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE assistant_compactions (
          id TEXT PRIMARY KEY,
          session_id TEXT NOT NULL DEFAULT '',
          summary TEXT NOT NULL,
          from_message_id TEXT NOT NULL,
          to_message_id TEXT NOT NULL,
          created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE assistant_config (
          session_id TEXT NOT NULL,
          key TEXT NOT NULL,
          value TEXT NOT NULL,
          PRIMARY KEY (session_id, key)
        )
        """,
        "CREATE TABLE assistant_sessions (id TEXT PRIMARY KEY, name TEXT)",
        """
        CREATE VIRTUAL TABLE assistant_fts USING fts5(
          id UNINDEXED,
          session_id UNINDEXED,
          role UNINDEXED,
          content,
          tokenize='porter unicode61'
        )
        """,
    ):
        ctx.storage.sql.exec(statement)


def _legacy_rows(ctx: fakes.FakeCtx) -> dict[str, str]:
    contents = {
        "m1": _message("m1", "root question", "user"),
        "m2": _message("m2", "first answer", "assistant"),
        "m3": _message("m3", "second answer snowman \u2603", "assistant"),
    }
    for row in (
        ("m1", "", None, "user", contents["m1"], "2026-01-02 03:04:05"),
        ("m2", "", "m1", "assistant", contents["m2"], "2026-01-02 03:04:06"),
        ("m3", "", "m1", "assistant", contents["m3"], "2026-01-02 03:04:06"),
    ):
        ctx.storage.sql.exec(
            "INSERT INTO assistant_messages "
            "(id, session_id, parent_id, role, content, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            *row,
        )
    ctx.storage.sql.exec(
        "INSERT INTO assistant_compactions "
        "(id, session_id, summary, from_message_id, to_message_id, created_at) "
        "VALUES ('c1', '', 'first branch summary', 'm1', 'm2', "
        "'2026-01-02 03:05:00')"
    )
    ctx.storage.sql.exec(
        "INSERT INTO assistant_config VALUES ('', 'prompt', 'frozen prompt')"
    )
    ctx.storage.sql.exec("INSERT INTO assistant_sessions VALUES ('chat', 'Chat')")
    ctx.storage.sql.exec(
        "INSERT INTO assistant_fts VALUES ('m1', '', 'user', 'root question')"
    )
    return contents


async def _start(
    ctx: fakes.FakeCtx,
    *,
    events: list[object] | None = None,
) -> Sessions:
    sessions = Sessions()
    lifecycle = Lifecycle(
        ctx,
        host=object(),
        event_listeners=() if events is None else (events.append,),
    )
    lifecycle.use(sessions)
    await lifecycle.start()
    return sessions


def _table_names(ctx: fakes.FakeCtx) -> set[str]:
    return {
        row["name"]
        for row in ctx.storage.sql.exec(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).toArray()
    }


@pytest.mark.asyncio
async def test_legacy_lift_preserves_payload_order_estimates_and_compactions():
    ctx = fakes.FakeCtx()
    _legacy_schema(ctx)
    contents = _legacy_rows(ctx)

    sessions = await _start(ctx)

    assert [message["id"] for message in await sessions.session().get_history()] == [
        "m1",
        "m3",
    ]
    assert await sessions.session().get_history() == [
        json.loads(contents["m1"]),
        json.loads(contents["m3"]),
    ]
    compacted = await sessions.session().get_history(HistoryReadOptions(leaf_id="m2"))
    assert len(compacted) == 1
    assert compacted[0]["id"] == "compaction_c1"
    assert compacted[0]["role"] == "assistant"
    assert compacted[0]["parts"] == [{"type": "text", "text": "first branch summary"}]
    assert await sessions.session().get_compactions() == [
        StoredCompaction(
            id="c1",
            summary="first branch summary",
            from_message_id="m1",
            to_message_id="m2",
            created_at="2026-01-02T03:05:00.000Z",
        )
    ]
    rows = ctx.storage.sql.exec(
        "SELECT id, seq, parent_id, type, role, content, content_chunks, "
        "token_estimate, created_at FROM cf_agents_session_messages ORDER BY seq"
    ).toArray()
    assert rows == [
        {
            "id": "m1",
            "seq": 1,
            "parent_id": None,
            "type": "message",
            "role": "user",
            "content": contents["m1"],
            "content_chunks": 0,
            "token_estimate": len(contents["m1"].encode()) // 4,
            "created_at": 1_767_323_045_000,
        },
        {
            "id": "m2",
            "seq": 2,
            "parent_id": "m1",
            "type": "message",
            "role": "assistant",
            "content": contents["m2"],
            "content_chunks": 0,
            "token_estimate": len(contents["m2"].encode()) // 4,
            "created_at": 1_767_323_046_000,
        },
        {
            "id": "m3",
            "seq": 3,
            "parent_id": "m1",
            "type": "message",
            "role": "assistant",
            "content": contents["m3"],
            "content_chunks": 0,
            "token_estimate": len(contents["m3"].encode()) // 4,
            "created_at": 1_767_323_046_000,
        },
    ]
    assert await ctx.storage.get("cf_agents:sessions_schema_version") == 1
    tables = _table_names(ctx)
    assert "assistant_messages" not in tables
    assert "assistant_compactions" not in tables
    assert "assistant_sessions" not in tables
    assert "assistant_fts" not in tables
    assert "assistant_config" in tables

    restarted = await _start(ctx)
    assert [message["id"] for message in await restarted.session().get_history()] == [
        "m1",
        "m3",
    ]


@pytest.mark.asyncio
async def test_collision_retains_only_incomplete_source_and_retries_partial_copy():
    ctx = fakes.FakeCtx()
    await _start(ctx)
    await ctx.storage.put("cf_agents:sessions_schema_version", 0)
    _legacy_schema(ctx)
    contents = _legacy_rows(ctx)
    ctx.storage.sql.exec(
        "INSERT INTO cf_agents_session_messages "
        "(session_id, id, seq, parent_id, role, content, created_at) "
        "VALUES ('', 'm2', 20, NULL, 'user', ?, 1)",
        _message("m2", "squatter", "user"),
    )
    events = []

    await _start(ctx, events=events)

    assert "assistant_messages" in _table_names(ctx)
    assert "assistant_compactions" not in _table_names(ctx)
    assert "assistant_sessions" not in _table_names(ctx)
    assert "assistant_fts" not in _table_names(ctx)
    assert await ctx.storage.get("cf_agents:sessions_schema_version") == 0
    assert [
        event.payload
        for event in events
        if getattr(event, "type", None) == "session:migration:incomplete"
    ] == [{"table": "assistant_messages", "source": 3, "copied": 2}]
    assert ctx.storage.sql.exec(
        "SELECT id FROM cf_agents_session_messages ORDER BY id"
    ).toArray() == [{"id": "m1"}, {"id": "m2"}, {"id": "m3"}]

    ctx.storage.sql.exec(
        "DELETE FROM cf_agents_session_messages WHERE session_id = '' AND id = 'm2'"
    )
    retried = await _start(ctx)

    assert "assistant_messages" not in _table_names(ctx)
    assert await ctx.storage.get("cf_agents:sessions_schema_version") == 1
    assert await retried.session().get_message("m2") == json.loads(contents["m2"])


@pytest.mark.asyncio
async def test_malformed_source_schema_is_reported_and_other_sources_still_lift():
    ctx = fakes.FakeCtx()
    ctx.storage.sql.exec("CREATE TABLE assistant_messages (id TEXT)")
    ctx.storage.sql.exec("INSERT INTO assistant_messages VALUES ('broken')")
    ctx.storage.sql.exec(
        """
        CREATE TABLE assistant_compactions (
          id TEXT PRIMARY KEY,
          session_id TEXT NOT NULL DEFAULT '',
          summary TEXT NOT NULL,
          from_message_id TEXT NOT NULL,
          to_message_id TEXT NOT NULL,
          created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    ctx.storage.sql.exec(
        "INSERT INTO assistant_compactions VALUES "
        "('c1', '', 'summary', 'm1', 'm2', '2026-01-02 03:05:00')"
    )
    events = []

    sessions = await _start(ctx, events=events)

    assert "assistant_messages" in _table_names(ctx)
    assert "assistant_compactions" not in _table_names(ctx)
    assert await ctx.storage.get("cf_agents:sessions_schema_version") is None
    assert [
        event.payload
        for event in events
        if getattr(event, "type", None) == "session:migration:incomplete"
    ] == [{"table": "assistant_messages", "source": 1, "copied": 0}]
    assert [item.id for item in await sessions.session().get_compactions()] == ["c1"]
    await sessions.session().append_message(
        json.loads(_message("new", "usable", "user"))
    )
    assert await sessions.session().get_message("new") is not None


@pytest.mark.asyncio
async def test_interrupted_verification_keeps_copies_and_retries(monkeypatch):
    ctx = fakes.FakeCtx()
    _legacy_schema(ctx)
    _legacy_rows(ctx)
    execute = ctx.storage.sql.exec

    def fail_verification(query, *params):
        normalized = " ".join(query.split())
        if normalized.startswith(
            "SELECT COUNT(*) AS count FROM assistant_messages AS legacy"
        ):
            raise RuntimeError("verification interrupted")
        return execute(query, *params)

    monkeypatch.setattr(ctx.storage.sql, "exec", fail_verification)
    events = []
    await _start(ctx, events=events)

    assert "assistant_messages" in _table_names(ctx)
    assert ctx.storage.sql.exec(
        "SELECT id FROM cf_agents_session_messages ORDER BY id"
    ).toArray() == [{"id": "m1"}, {"id": "m2"}, {"id": "m3"}]
    assert await ctx.storage.get("cf_agents:sessions_schema_version") is None
    assert any(
        getattr(event, "type", None) == "session:migration:incomplete"
        for event in events
    )

    monkeypatch.undo()
    sessions = await _start(ctx)
    assert "assistant_messages" not in _table_names(ctx)
    assert [message["id"] for message in await sessions.session().get_history()] == [
        "m1",
        "m3",
    ]


@pytest.mark.asyncio
async def test_active_fts_backfills_only_missing_migrated_rows():
    ctx = fakes.FakeCtx()
    current = await _start(ctx)
    await current.session().append_message(
        json.loads(_message("current", "current searchable", "user"))
    )
    assert await current.session().search("current searchable")
    await ctx.storage.put("cf_agents:sessions_schema_version", 0)
    _legacy_schema(ctx)
    _legacy_rows(ctx)

    migrated = await _start(ctx)

    assert "cf_agents_session_fts" not in _table_names(ctx)
    assert [
        result.id for result in await migrated.session().search("root question")
    ] == ["m1"]
    assert [
        result.id for result in await migrated.session().search("current searchable")
    ] == ["current"]
    assert ctx.storage.sql.exec(
        "SELECT id, COUNT(*) AS count FROM cf_agents_session_fts "
        "GROUP BY id ORDER BY id"
    ).toArray() == [
        {"id": "current", "count": 1},
        {"id": "m1", "count": 1},
        {"id": "m2", "count": 1},
        {"id": "m3", "count": 1},
    ]


@pytest.mark.asyncio
async def test_malformed_json_is_copied_verbatim_and_invalid_date_becomes_epoch():
    ctx = fakes.FakeCtx()
    _legacy_schema(ctx)
    ctx.storage.sql.exec(
        "INSERT INTO assistant_messages "
        "(id, session_id, parent_id, role, content, created_at) "
        "VALUES ('bad', '', NULL, 'user', 'not json', 'not a date')"
    )

    sessions = await _start(ctx)

    assert ctx.storage.sql.exec(
        "SELECT content, token_estimate, created_at "
        "FROM cf_agents_session_messages WHERE id = 'bad'"
    ).toArray() == [{"content": "not json", "token_estimate": 2, "created_at": 0}]
    assert await sessions.session().get_history() == []
    assert "assistant_messages" not in _table_names(ctx)


@pytest.mark.asyncio
@pytest.mark.parametrize("marker", ["malformed", -1])
async def test_malformed_and_negative_markers_retry_empty_legacy_lifts(marker):
    ctx = fakes.FakeCtx()
    _legacy_schema(ctx)
    await ctx.storage.put("cf_agents:sessions_schema_version", marker)

    await _start(ctx)

    assert "assistant_messages" not in _table_names(ctx)
    assert "assistant_compactions" not in _table_names(ctx)
    assert await ctx.storage.get("cf_agents:sessions_schema_version") == 1


@pytest.mark.asyncio
async def test_future_and_completed_markers_do_not_revisit_legacy_sources():
    for marker in (1, 2):
        ctx = fakes.FakeCtx()
        await _start(ctx)
        await ctx.storage.put("cf_agents:sessions_schema_version", marker)
        _legacy_schema(ctx)
        contents = _legacy_rows(ctx)

        await _start(ctx)

        assert "assistant_messages" in _table_names(ctx)
        assert (
            ctx.storage.sql.exec("SELECT id FROM cf_agents_session_messages").toArray()
            == []
        )
        assert ctx.storage.sql.exec(
            "SELECT content FROM assistant_messages WHERE id = 'm1'"
        ).toArray() == [{"content": contents["m1"]}]
        assert await ctx.storage.get("cf_agents:sessions_schema_version") == marker
