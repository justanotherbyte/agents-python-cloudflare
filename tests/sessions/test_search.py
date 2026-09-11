from __future__ import annotations

import base64

import fakes
import pytest

import agents.sessions as sessions_module
from agents.lifecycle import Lifecycle
from agents.sessions import SearchOptions, SearchResult, Sessions


async def _started(ctx: fakes.FakeCtx | None = None):
    ctx = fakes.FakeCtx() if ctx is None else ctx
    sessions = Sessions()
    lifecycle = Lifecycle(ctx, host=object())
    lifecycle.use(sessions)
    await lifecycle.start()
    return sessions, ctx


def _text(message_id: str, content: str, *, role: str = "user"):
    return {
        "id": message_id,
        "role": role,
        "parts": [{"type": "text", "text": content}],
    }


def _fts_tables(ctx):
    return [
        row["name"]
        for row in ctx.storage.sql.exec(
            "SELECT name FROM sqlite_master "
            "WHERE name LIKE 'cf_agents_session_fts%' ORDER BY name"
        ).toArray()
    ]


@pytest.mark.asyncio
async def test_search_surface_is_exported_and_first_search_backfills_every_session():
    assert "SearchOptions" in sessions_module.__all__
    assert "SearchResult" in sessions_module.__all__
    sessions, ctx = await _started()
    first = sessions.session("first")
    second = sessions.session("second")
    await first.append_message(_text("quick", "the quick brown fox"))
    await second.append_message(_text("lazy", "the lazy dog"))
    await first.append_message(_text("large", "x" * (1536 * 1024) + " needle"))
    for index in range(55):
        await second.append_message(_text(f"page-{index}", f"page token {index}"))

    assert _fts_tables(ctx) == []
    assert await first.search("quick brown") == [
        SearchResult(id="quick", role="user", content="the quick brown fox")
    ]
    assert await first.search("needle") == [
        SearchResult(
            id="large",
            role="user",
            content="x" * (1536 * 1024) + " needle",
        )
    ]
    assert await first.search("lazy dog") == []
    assert [result.id for result in await second.search("lazy dog")] == ["lazy"]
    assert [result.id for result in await second.search("page token 54")] == ["page-54"]
    assert _fts_tables(ctx) == [
        "cf_agents_session_fts",
        "cf_agents_session_fts_config",
        "cf_agents_session_fts_content",
        "cf_agents_session_fts_data",
        "cf_agents_session_fts_docsize",
        "cf_agents_session_fts_idx",
    ]
    ddl = ctx.storage.sql.exec(
        "SELECT sql FROM sqlite_master WHERE name = 'cf_agents_session_fts'"
    ).toArray()[0]["sql"]
    assert " ".join(ddl.split()) == (
        "CREATE VIRTUAL TABLE cf_agents_session_fts USING fts5( "
        "id UNINDEXED, session_id UNINDEXED, role UNINDEXED, content, "
        "tokenize='porter unicode61' )"
    )


@pytest.mark.asyncio
async def test_search_treats_queries_as_literal_phrases_and_applies_limits():
    sessions, _ = await _started()
    session = sessions.session()
    for message in (
        _text("phrase", "the quick brown fox"),
        _text("separate", "quick agile brown fox"),
        _text("quote", 'say "hello" now'),
        _text("operator", "OR is a word"),
    ):
        await session.append_message(message)
    for index in range(25):
        await session.append_message(_text(f"common-{index}", "common phrase"))

    assert [result.id for result in await session.search("quick brown")] == ["phrase"]
    assert [result.id for result in await session.search('say "hello"')] == ["quote"]
    assert [result.id for result in await session.search("OR")] == ["operator"]
    assert await session.search("") == []
    assert len(await session.search("common phrase")) == 20
    assert len(await session.search("fox", SearchOptions(limit=1))) == 1
    assert await session.search("fox", SearchOptions(limit=0)) == []


@pytest.mark.asyncio
async def test_active_search_index_tracks_every_message_mutation():
    sessions, ctx = await _started()
    session = sessions.session()
    assert await session.search("activate") == []

    await session.append_message(_text("append", "quick brown"))
    await session.import_message(
        _text("import", "historical record"),
        parent_id=None,
        created_at_ms=1,
    )
    await session.upsert_message(_text("upsert", "newly inserted"))
    assert [result.id for result in await session.search("quick brown")] == ["append"]
    assert [result.id for result in await session.search("historical")] == ["import"]
    assert [result.id for result in await session.search("newly inserted")] == [
        "upsert"
    ]

    await session.update_message(_text("append", "alert cat", role="assistant"))
    assert await session.search("quick brown") == []
    assert await session.search("alert cat") == [
        SearchResult(id="append", role="assistant", content="alert cat")
    ]
    await session.update_message(
        {
            "id": "import",
            "role": "user",
            "parts": [{"type": "reasoning", "text": "hidden"}],
        }
    )
    assert await session.search("historical") == []

    await session.delete_messages(["append"])
    assert await session.search("alert cat") == []
    await session.clear_messages()
    assert ctx.storage.sql.exec("SELECT id FROM cf_agents_session_fts").toArray() == []


@pytest.mark.asyncio
async def test_search_indexes_text_but_not_attachment_payloads_or_malformed_rows():
    sessions, ctx = await _started()
    image = base64.b64encode(b"secret image payload").decode("ascii")
    await sessions.session().append_message(
        {
            "id": "media",
            "role": "user",
            "parts": [
                {"type": "text", "text": "visible caption"},
                {
                    "type": "file",
                    "mediaType": "image/png",
                    "url": f"data:image/png;base64,{image}",
                },
            ],
        }
    )
    await sessions.session().append_message(_text("surrogate", "bad \ud800 value"))
    await sessions.session().append_message(
        {
            "id": "empty-segments",
            "role": "user",
            "parts": [
                {"type": "text", "text": ""},
                {"type": "text", "text": "kept"},
                {"type": "text", "text": ""},
            ],
        }
    )
    await sessions.session().append_message(_text("empty", ""))
    ctx.storage.sql.exec(
        "INSERT INTO cf_agents_session_messages "
        "(session_id, id, seq, parent_id, role, content, created_at) "
        "VALUES ('', 'malformed', 3, 'media', 'user', 'not json', 3)"
    )
    ctx.storage.sql.exec(
        "INSERT INTO cf_agents_session_messages "
        "(session_id, id, seq, parent_id, role, content, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        "",
        "physical",
        4,
        "malformed",
        "assistant",
        '{"id":"spoofed","role":"user","parts":'
        '[{"type":"text","text":"physical truth"}]}',
        4,
    )

    assert [result.id for result in await sessions.session().search("visible")] == [
        "media"
    ]
    assert await sessions.session().search("secret image") == []
    assert await sessions.session().search("physical truth") == [
        SearchResult(id="physical", role="assistant", content="physical truth")
    ]
    assert await sessions.session().search("bad \ud800 value") == [
        SearchResult(id="surrogate", role="user", content="bad \ufffd value")
    ]
    assert await sessions.session().search("kept") == [
        SearchResult(id="empty-segments", role="user", content="kept")
    ]
    assert (
        ctx.storage.sql.exec(
            "SELECT id FROM cf_agents_session_fts WHERE id = 'empty'"
        ).toArray()
        == []
    )
    assert ctx.storage.sql.exec(
        "SELECT id, content FROM cf_agents_session_fts WHERE id = 'media'"
    ).toArray() == [{"id": "media", "content": "visible caption"}]

    ctx.storage.sql.exec(
        "INSERT INTO cf_agents_session_fts (id, session_id, role, content) "
        "VALUES ('orphan', '', 'user', 'orphan term')"
    )
    assert await sessions.session().search("orphan term") == []

    await sessions.session().append_message(_text("pair", "face \ud83d\ude00"))
    assert await sessions.session().search("face \U0001f600") == [
        SearchResult(id="pair", role="user", content="face \U0001f600")
    ]


@pytest.mark.asyncio
async def test_writes_before_search_and_unchanged_index_text_have_no_fts_writes(
    monkeypatch,
):
    sessions, ctx = await _started()
    session = sessions.session()
    execute = ctx.storage.sql.exec
    statements = []

    def record(query, *params):
        statements.append(query)
        return execute(query, *params)

    monkeypatch.setattr(ctx.storage.sql, "exec", record)
    await session.append_message(_text("message", "same \ud800 indexed text"))
    assert not [query for query in statements if "cf_agents_session_fts" in query]

    monkeypatch.undo()
    await session.search("same \ud800 indexed text")
    statements.clear()
    monkeypatch.setattr(ctx.storage.sql, "exec", record)
    await session.update_message(
        {
            **_text("message", "same \ud800 indexed text"),
            "metadata": {"changed": True},
        }
    )
    writes = [
        query
        for query in statements
        if "cf_agents_session_fts" in query
        and query.lstrip().startswith(("INSERT", "UPDATE", "DELETE"))
    ]
    assert writes == []


@pytest.mark.asyncio
async def test_existing_fts_is_detected_after_activation_restart():
    sessions, ctx = await _started()
    await sessions.session().append_message(_text("before", "before restart"))
    assert [result.id for result in await sessions.session().search("before")] == [
        "before"
    ]

    restarted, _ = await _started(ctx)
    await restarted.session().append_message(_text("after", "after restart"))

    assert [result.id for result in await restarted.session().search("after")] == [
        "after"
    ]


@pytest.mark.asyncio
async def test_fts_creation_and_active_mutations_are_atomic_and_retryable(monkeypatch):
    sessions, ctx = await _started()
    session = sessions.session()
    await session.append_message(_text("backfill", "retry this"))
    execute = ctx.storage.sql.exec

    def fail_insert(query, *params):
        if "INSERT INTO cf_agents_session_fts" in query:
            raise RuntimeError("fts insert failed")
        return execute(query, *params)

    monkeypatch.setattr(ctx.storage.sql, "exec", fail_insert)
    with pytest.raises(RuntimeError, match="fts insert failed"):
        await session.search("retry")
    assert _fts_tables(ctx) == []

    monkeypatch.undo()
    assert [result.id for result in await session.search("retry")] == ["backfill"]
    monkeypatch.setattr(ctx.storage.sql, "exec", fail_insert)
    with pytest.raises(RuntimeError, match="fts insert failed"):
        await session.append_message(_text("failed", "new searchable row"))
    assert (
        execute(
            "SELECT id FROM cf_agents_session_messages WHERE id = 'failed'"
        ).toArray()
        == []
    )

    monkeypatch.undo()
    monkeypatch.setattr(ctx.storage.sql, "exec", fail_insert)
    with pytest.raises(RuntimeError, match="fts insert failed"):
        await session.update_message(_text("backfill", "replacement text"))
    monkeypatch.undo()
    assert (await session.get_message("backfill"))["parts"][0]["text"] == "retry this"
    assert [result.id for result in await session.search("retry this")] == ["backfill"]

    monkeypatch.setattr(ctx.storage.sql, "exec", fail_insert)
    with pytest.raises(RuntimeError, match="fts insert failed"):
        await session.import_message(
            _text("import-failed", "historical failure"),
            parent_id=None,
            created_at_ms=1,
        )
    monkeypatch.undo()
    assert await session.get_message("import-failed") is None

    def fail_delete(query, *params):
        if "DELETE FROM cf_agents_session_fts" in query:
            raise RuntimeError("fts delete failed")
        return execute(query, *params)

    monkeypatch.setattr(ctx.storage.sql, "exec", fail_delete)
    with pytest.raises(RuntimeError, match="fts delete failed"):
        await session.delete_messages(["backfill"])
    monkeypatch.undo()
    assert await session.get_message("backfill") is not None

    monkeypatch.setattr(ctx.storage.sql, "exec", fail_delete)
    with pytest.raises(RuntimeError, match="fts delete failed"):
        await session.clear_messages()
    monkeypatch.undo()
    assert await session.get_message("backfill") is not None
