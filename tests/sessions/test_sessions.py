from __future__ import annotations

from datetime import UTC, datetime

import fakes
import pytest

import agents.sessions as sessions_module
from agents.lifecycle import Lifecycle, LifecycleEvent
from agents.sessions import (
    AppendOptions,
    HistoryBatchReadOptions,
    HistoryReadOptions,
    Session,
    Sessions,
    SessionsOptions,
    WriteOptions,
)


def _message(message_id: str, text: str, *, role: str = "user"):
    return {
        "id": message_id,
        "role": role,
        "parts": [{"type": "text", "text": text}],
    }


async def _started(*, options: SessionsOptions | None = None, events=()):
    ctx = fakes.FakeCtx()
    sessions = Sessions(options)
    lifecycle = Lifecycle(ctx, host=object(), event_listeners=events)
    lifecycle.use(sessions)
    await lifecycle.start()
    return sessions, lifecycle, ctx


def test_sessions_module_exports_public_surface():
    assert set(sessions_module.__all__) == {
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
    }


@pytest.mark.asyncio
async def test_constructor_is_infallible_handles_are_cached_and_schema_is_exact():
    ctx = fakes.FakeCtx()
    sessions = Sessions()
    default = sessions.session()

    assert isinstance(default, Session)
    assert default is sessions.session("")
    assert sessions.session("other") is sessions.session("other")
    assert (
        ctx.storage.sql.exec(
            "SELECT name FROM sqlite_master WHERE name LIKE 'cf_agents_session_%'"
        ).toArray()
        == []
    )

    lifecycle = Lifecycle(ctx, host=object())
    lifecycle.use(sessions)
    await lifecycle.start()

    tables = ctx.storage.sql.exec(
        "SELECT name FROM sqlite_master WHERE name LIKE 'cf_agents_session_%' "
        "ORDER BY name"
    ).toArray()
    assert tables == [
        {"name": "cf_agents_session_attachment_chunks"},
        {"name": "cf_agents_session_attachment_meta"},
        {"name": "cf_agents_session_attachment_refs"},
        {"name": "cf_agents_session_compactions"},
        {"name": "cf_agents_session_config"},
        {"name": "cf_agents_session_message_chunks"},
        {"name": "cf_agents_session_messages"},
    ]
    assert await ctx.storage.get("cf_agents:sessions_schema_version") == 1
    assert (
        ctx.storage.sql.exec(
            "SELECT name FROM sqlite_master WHERE name = 'cf_agents_session_fts'"
        ).toArray()
        == []
    )

    columns = ctx.storage.sql.exec(
        "PRAGMA table_info(cf_agents_session_messages)"
    ).toArray()
    columns = [
        {
            "name": row["name"],
            "type": row["type"],
            "notnull": row["notnull"],
            "dflt_value": row["dflt_value"],
            "pk": row["pk"],
        }
        for row in columns
    ]
    assert columns == [
        {
            "name": "session_id",
            "type": "TEXT",
            "notnull": 1,
            "dflt_value": None,
            "pk": 1,
        },
        {"name": "id", "type": "TEXT", "notnull": 1, "dflt_value": None, "pk": 2},
        {"name": "seq", "type": "INTEGER", "notnull": 1, "dflt_value": None, "pk": 0},
        {
            "name": "parent_id",
            "type": "TEXT",
            "notnull": 0,
            "dflt_value": None,
            "pk": 0,
        },
        {
            "name": "type",
            "type": "TEXT",
            "notnull": 1,
            "dflt_value": "'message'",
            "pk": 0,
        },
        {"name": "role", "type": "TEXT", "notnull": 1, "dflt_value": None, "pk": 0},
        {"name": "content", "type": "TEXT", "notnull": 1, "dflt_value": None, "pk": 0},
        {
            "name": "content_chunks",
            "type": "INTEGER",
            "notnull": 1,
            "dflt_value": "0",
            "pk": 0,
        },
        {
            "name": "token_estimate",
            "type": "INTEGER",
            "notnull": 1,
            "dflt_value": "0",
            "pk": 0,
        },
        {
            "name": "created_at",
            "type": "INTEGER",
            "notnull": 1,
            "dflt_value": None,
            "pk": 0,
        },
    ]


@pytest.mark.asyncio
async def test_legacy_sources_leave_schema_marker_unstamped_until_the_lift():
    ctx = fakes.FakeCtx()
    ctx.storage.sql.exec("CREATE TABLE assistant_messages (id TEXT)")
    sessions = Sessions()
    lifecycle = Lifecycle(ctx, host=object())
    lifecycle.use(sessions)

    await lifecycle.start()

    assert await ctx.storage.get("cf_agents:sessions_schema_version") is None
    assert ctx.storage.sql.exec(
        "SELECT name FROM sqlite_master WHERE name = 'cf_agents_session_messages'"
    ).toArray() == [{"name": "cf_agents_session_messages"}]


@pytest.mark.asyncio
async def test_future_schema_marker_is_not_mutated_or_downgraded():
    ctx = fakes.FakeCtx()
    current = Sessions()
    current_lifecycle = Lifecycle(ctx, host=object())
    current_lifecycle.use(current)
    await current_lifecycle.start()
    await ctx.storage.put("cf_agents:sessions_schema_version", 2)
    schema_before = ctx.storage.sql.exec(
        "SELECT name, sql FROM sqlite_master "
        "WHERE name LIKE 'cf_agents_session_%' ORDER BY name"
    ).toArray()
    sessions = Sessions()
    lifecycle = Lifecycle(ctx, host=object())
    lifecycle.use(sessions)

    await lifecycle.start()

    assert (
        ctx.storage.sql.exec(
            "SELECT name, sql FROM sqlite_master "
            "WHERE name LIKE 'cf_agents_session_%' ORDER BY name"
        ).toArray()
        == schema_before
    )
    assert await ctx.storage.get("cf_agents:sessions_schema_version") == 2
    assert await sessions.session().get_history() == []


def test_constructor_does_not_invoke_option_truthiness():
    class HostileOptions(SessionsOptions):
        def __bool__(self):
            raise AssertionError("options truthiness evaluated")

    sessions = Sessions(HostileOptions())

    assert sessions.session() is sessions.session()


@pytest.mark.asyncio
async def test_message_tree_reads_branch_from_the_active_maximum_sequence():
    sessions, _, ctx = await _started()
    session = sessions.session()

    await session.append_message(
        _message("root", "root"), AppendOptions(parent_id=None)
    )
    await session.append_message(_message("first", "first"))
    await session.append_message(
        _message("branch", "branch"), AppendOptions(parent_id="root")
    )
    await session.append_message(_message("leaf", "leaf"))

    assert await session.get_latest_leaf() == _message("leaf", "leaf")
    assert await session.get_branches("root") == [
        _message("first", "first"),
        _message("branch", "branch"),
    ]
    assert [message["id"] async for message in session.history()] == [
        "root",
        "branch",
        "leaf",
    ]
    first_history = await session.get_history(HistoryReadOptions(leaf_id="first"))
    assert [message["id"] for message in first_history] == [
        "root",
        "first",
    ]
    assert await session.get_history(HistoryReadOptions(leaf_id="missing")) == []

    rows = ctx.storage.sql.exec(
        "SELECT id, seq, parent_id FROM cf_agents_session_messages ORDER BY seq"
    ).toArray()
    assert rows == [
        {"id": "root", "seq": 1, "parent_id": None},
        {"id": "first", "seq": 2, "parent_id": "root"},
        {"id": "branch", "seq": 3, "parent_id": "root"},
        {"id": "leaf", "seq": 4, "parent_id": "branch"},
    ]


@pytest.mark.asyncio
async def test_idempotent_writes_sanitization_and_event_order_are_exact():
    lifecycle_events: list[LifecycleEvent] = []
    sessions, _, ctx = await _started(
        options=SessionsOptions(reserved_metadata_keys=("private",)),
        events=(lifecycle_events.append,),
    )
    session = sessions.session("conversation")
    changes = []
    session_changes_after_commit = []

    async def listener(event):
        session_changes_after_commit.append(
            ctx.storage.sql.exec(
                "SELECT COUNT(*) AS count FROM cf_agents_session_messages"
            ).toArray()[0]["count"]
        )
        changes.append(event)

    sessions.subscribe(listener)
    message = {
        "id": "message",
        "role": "assistant",
        "parts": [
            {
                "type": "text",
                "text": "stored",
                "providerMetadata": {
                    "openai": {
                        "itemId": "ephemeral",
                        "reasoningEncryptedContent": "secret",
                        "stable": True,
                    }
                },
            },
            {
                "type": "reasoning",
                "text": " ",
                "providerMetadata": {"openai": {"reasoningEncryptedContent": "secret"}},
            },
        ],
        "metadata": {"private": True, "visible": "yes"},
    }

    inserted = await session.append_message(
        message,
        AppendOptions(source="client", parent_id=None),
    )
    duplicate = await session.append_message(
        _message("message", "different"),
        AppendOptions(parent_id="absent"),
    )

    assert inserted.inserted is True
    assert inserted.message == {
        "id": "message",
        "role": "assistant",
        "parts": [
            {
                "type": "text",
                "text": "stored",
                "providerMetadata": {"openai": {"stable": True}},
            }
        ],
        "metadata": {"visible": "yes"},
    }
    assert duplicate.inserted is False
    assert duplicate.message == inserted.message
    assert session_changes_after_commit == [1, 1]
    assert changes == [
        {
            "type": "append",
            "sessionId": "conversation",
            "message": inserted.message,
            "parentId": None,
            "inserted": True,
        },
        {
            "type": "append",
            "sessionId": "conversation",
            "message": inserted.message,
            "parentId": "absent",
            "inserted": False,
        },
    ]
    assert [event.type for event in lifecycle_events] == ["session:message:appended"]


@pytest.mark.asyncio
async def test_listener_mutation_cannot_change_later_delivery_or_return_values():
    lifecycle_events: list[LifecycleEvent] = []
    sessions, _, _ = await _started(events=(lifecycle_events.append,))
    observed = []

    async def mutating(event):
        event["message"]["parts"].clear()
        event.clear()
        raise RuntimeError("mutated")

    async def later(event):
        observed.append(event)

    sessions.subscribe(mutating)
    sessions.subscribe(later)

    result = await sessions.session().append_message(_message("id", "stable"))

    assert result.message == _message("id", "stable")
    assert observed == [
        {
            "type": "append",
            "sessionId": "",
            "message": _message("id", "stable"),
            "inserted": True,
        }
    ]
    assert [event.type for event in lifecycle_events] == [
        "session:message:appended",
        "session:error",
    ]


@pytest.mark.asyncio
async def test_updates_upserts_imports_and_listener_failures_preserve_commits():
    lifecycle_events: list[LifecycleEvent] = []
    sessions, _, ctx = await _started(
        options=SessionsOptions(reserved_metadata_keys=("private",)),
        events=(lifecycle_events.append,),
    )
    session = sessions.session()
    changes = []

    async def failing(event):
        if event["type"] == "update":
            raise RuntimeError("listener failed")
        changes.append(event)

    async def later(event):
        changes.append(("later", event["type"]))

    sessions.subscribe(failing)
    sessions.subscribe(later)
    original = _message("existing", "old")
    await session.append_message(original, AppendOptions(parent_id=None))

    assert await session.update_message(original) == original
    updated = _message("existing", "new", role="assistant")
    assert (
        await session.update_message(updated, WriteOptions(source="server")) == updated
    )
    assert await session.get_message("existing") == updated
    assert await session.update_message(_message("missing", "none")) is None

    existing = await session.upsert_message(_message("existing", "newest"))
    added = await session.upsert_message(_message("added", "added"))
    assert existing.inserted is False
    assert existing.message == _message("existing", "newest")
    assert added.inserted is True

    imported = {
        **_message("imported", "raw"),
        "metadata": {"private": True},
    }
    await session.import_message(
        imported,
        parent_id="existing",
        created_at_ms=1_700_000_000_000,
    )
    assert await session.get_message("imported") == imported
    assert ctx.storage.sql.exec(
        "SELECT parent_id, created_at FROM cf_agents_session_messages "
        "WHERE id = 'imported'"
    ).toArray() == [{"parent_id": "existing", "created_at": 1_700_000_000_000}]
    assert ("later", "update") in changes
    assert [event.type for event in lifecycle_events].count("session:error") == 2
    assert [event.type for event in lifecycle_events].count(
        "session:message:updated"
    ) == 2


@pytest.mark.asyncio
async def test_history_batches_recent_stats_splice_delete_and_clear():
    sessions, _, _ = await _started()
    session = sessions.session()
    for message_id in ("root", "middle", "deleted", "leaf"):
        await session.append_message(_message(message_id, message_id))

    batches = [
        batch
        async for batch in session.history_batches(
            HistoryBatchReadOptions(batch_size=2, max_batch_bytes=10_000)
        )
    ]
    assert [[message["id"] for message in batch] for batch in batches] == [
        ["root", "middle"],
        ["deleted", "leaf"],
    ]

    stats = await session.get_history_row_stats()
    assert [stat.id for stat in stats] == ["root", "middle", "deleted", "leaf"]
    assert all(stat.bytes > 0 and stat.token_estimate > 0 for stat in stats)
    recent = await session.get_recent_history(stats[-1].bytes)
    assert [message["id"] for message in recent.messages] == ["leaf"]
    assert recent.truncated is True
    assert recent.total_content_bytes == sum(stat.bytes for stat in stats)

    await session.delete_messages(["middle", "deleted", "middle"])
    assert [message["id"] for message in await session.get_history()] == [
        "root",
        "leaf",
    ]
    assert [message["id"] for message in await session.get_branches("root")] == ["leaf"]

    await session.clear_messages()
    assert await session.get_history() == []
    assert await session.get_latest_leaf() is None
    after_clear = await session.append_message(_message("fresh", "fresh"))
    assert after_clear.inserted


@pytest.mark.asyncio
async def test_history_batch_byte_limit_yields_one_oversized_message():
    sessions, _, _ = await _started()
    session = sessions.session()
    await session.append_message(_message("large", "x" * 100))
    await session.append_message(_message("small", "small"))

    batches = [
        batch
        async for batch in session.history_batches(
            HistoryBatchReadOptions(batch_size=50, max_batch_bytes=1)
        )
    ]

    assert [[message["id"] for message in batch] for batch in batches] == [
        ["large"],
        ["small"],
    ]


@pytest.mark.asyncio
async def test_message_json_is_strict_and_datetime_is_stored_as_iso_text():
    sessions, _, _ = await _started()
    session = sessions.session()

    with pytest.raises(ValueError, match="JSON"):
        await session.append_message(
            {**_message("bad", "bad"), "metadata": {"value": float("nan")}}
        )

    created_at = datetime(2026, 9, 10, 12, 30, tzinfo=UTC)
    stored = await session.append_message(
        {**_message("dated", "dated"), "createdAt": created_at}
    )
    assert stored.message["createdAt"] == "2026-09-10T12:30:00.000Z"


@pytest.mark.asyncio
async def test_unicode_and_integral_numbers_use_javascript_json_bytes():
    sessions, _, ctx = await _started()
    session = sessions.session()
    message = {
        **_message("unicode", "café"),
        "metadata": {
            "number": 1.0,
            "exact": 2**53,
            "keys": {"2": "b", "1": "a", "x": "c"},
        },
    }

    await session.append_message(message)

    content = ctx.storage.sql.exec(
        "SELECT content FROM cf_agents_session_messages WHERE id = 'unicode'"
    ).toArray()[0]["content"]
    assert content == (
        '{"id":"unicode","role":"user","parts":'
        '[{"type":"text","text":"café"}],"metadata":{"number":1,'
        '"exact":9007199254740992,"keys":{"1":"a","2":"b","x":"c"}}}'
    )


@pytest.mark.asyncio
async def test_malformed_persisted_parts_are_skipped_consistently():
    sessions, _, ctx = await _started()
    ctx.storage.sql.exec(
        "INSERT INTO cf_agents_session_messages "
        "(session_id, id, seq, parent_id, role, content, created_at) "
        "VALUES ('', 'bad', 1, NULL, 'user', ?, 1)",
        '{"id":"bad","role":"user","parts":[1]}',
    )

    assert await sessions.session().get_message("bad") is None
    assert await sessions.session().get_history() == []


@pytest.mark.asyncio
async def test_malformed_continuations_skip_only_their_messages():
    sessions, _, ctx = await _started()
    session = sessions.session()
    await session.append_message(_message("good", "good"))
    prefix = '{"id":"bad-count","role":"user","parts":[{"type":"text","text":"'
    suffix = 'bad"}]}'
    ctx.storage.sql.exec(
        "INSERT INTO cf_agents_session_messages "
        "(session_id, id, seq, parent_id, role, content, content_chunks, created_at) "
        "VALUES ('', 'bad-count', 2, 'good', 'user', ?, 9223372036854775807, 2)",
        prefix,
    )
    ctx.storage.sql.exec(
        "INSERT INTO cf_agents_session_message_chunks "
        "(session_id, id, idx, content) VALUES ('', 'bad-count', 1, ?)",
        suffix,
    )
    blob_prefix = prefix.replace("bad-count", "bad-blob")
    ctx.storage.sql.exec(
        "INSERT INTO cf_agents_session_messages "
        "(session_id, id, seq, parent_id, role, content, content_chunks, created_at) "
        "VALUES ('', 'bad-blob', 3, 'bad-count', 'user', ?, 1, 3)",
        blob_prefix,
    )
    ctx.storage.sql.exec(
        "INSERT INTO cf_agents_session_message_chunks "
        "(session_id, id, idx, content) VALUES ('', 'bad-blob', 1, ?)",
        suffix.encode(),
    )
    ctx.storage.sql.exec(
        "INSERT INTO cf_agents_session_messages "
        "(session_id, id, seq, parent_id, role, content, created_at) "
        "VALUES ('', 'leaf', 4, 'bad-blob', 'user', ?, 4)",
        '{"id":"leaf","role":"user","parts":[{"type":"text","text":"leaf"}]}',
    )
    ctx.storage.sql.exec(
        "INSERT INTO cf_agents_session_messages "
        "(session_id, id, seq, parent_id, role, content, created_at) "
        "VALUES ('', 'sibling', 5, 'good', 'user', ?, 5)",
        '{"id":"sibling","role":"user","parts":[{"type":"text","text":"sibling"}]}',
    )

    assert await session.get_message("bad-count") is None
    assert await session.get_message("bad-blob") is None
    history = await session.get_history(HistoryReadOptions(leaf_id="leaf"))
    assert [message["id"] for message in history] == [
        "good",
        "leaf",
    ]
    assert [message["id"] for message in await session.get_branches("good")] == [
        "sibling"
    ]


@pytest.mark.asyncio
async def test_recent_history_reports_the_recursive_depth_cap():
    sessions, _, ctx = await _started()
    ctx.storage.sql.exec("""
        WITH RECURSIVE numbers(value) AS (
          SELECT 0
          UNION ALL SELECT value + 1 FROM numbers WHERE value < 10001
        )
        INSERT INTO cf_agents_session_messages
          (session_id, id, seq, parent_id, role, content, created_at)
        SELECT '', 'm' || value, value + 1,
          CASE WHEN value = 0 THEN NULL ELSE 'm' || (value - 1) END,
          'user',
          json_object(
            'id', 'm' || value,
            'role', 'user',
            'parts', json_array(json_object('type', 'text', 'text', value))
          ),
          value
        FROM numbers
    """)

    recent = await sessions.session().get_recent_history(0)

    assert [message["id"] for message in recent.messages] == ["m10001"]
    assert recent.truncated is True


@pytest.mark.asyncio
async def test_oversized_text_and_emoji_round_trip_through_continuations():
    sessions, _, ctx = await _started()
    session = sessions.session()
    text_body = "z" * (2 * 1024 * 1024)
    emoji_body = "a" + "🙂" * 500_000

    await session.append_message(_message("text", text_body))
    await session.append_message(_message("emoji", emoji_body))

    rows = ctx.storage.sql.exec(
        "SELECT id, content_chunks, LENGTH(CAST(content AS BLOB)) AS bytes "
        "FROM cf_agents_session_messages ORDER BY seq"
    ).toArray()
    assert rows[0] == {
        "id": "text",
        "content_chunks": 1,
        "bytes": 1536 * 1024,
    }
    assert rows[1]["id"] == "emoji"
    assert rows[1]["content_chunks"] == 1
    assert rows[1]["bytes"] < 1536 * 1024
    continuations = ctx.storage.sql.exec(
        "SELECT id, idx, LENGTH(CAST(content AS BLOB)) AS bytes "
        "FROM cf_agents_session_message_chunks ORDER BY id, idx"
    ).toArray()
    assert [row["idx"] for row in continuations] == [1, 1]
    assert all(0 < row["bytes"] <= 1536 * 1024 for row in continuations)
    assert (await session.get_message("text"))["parts"][0]["text"] == text_body
    assert (await session.get_message("emoji"))["parts"][0]["text"] == emoji_body


@pytest.mark.asyncio
async def test_updates_compare_full_content_and_delete_surplus_continuations():
    sessions, _, ctx = await _started()
    session = sessions.session()
    five_mb = "s" * 5_000_000
    two_mb = "s" * 2_000_000
    await session.append_message(_message("shrink", five_mb))
    assert ctx.storage.sql.exec(
        "SELECT content_chunks FROM cf_agents_session_messages WHERE id = 'shrink'"
    ).toArray() == [{"content_chunks": 3}]
    assert ctx.storage.sql.exec(
        "SELECT idx FROM cf_agents_session_message_chunks "
        "WHERE id = 'shrink' ORDER BY idx"
    ).toArray() == [{"idx": 1}, {"idx": 2}, {"idx": 3}]

    await session.update_message(_message("shrink", two_mb))
    assert ctx.storage.sql.exec(
        "SELECT idx FROM cf_agents_session_message_chunks "
        "WHERE id = 'shrink' ORDER BY idx"
    ).toArray() == [{"idx": 1}]

    changes = []
    sessions.subscribe(changes.append)
    changes_before = ctx.storage.sql.exec("SELECT total_changes() AS count").toArray()[
        0
    ]["count"]
    await session.update_message(_message("shrink", two_mb))
    assert changes == []
    assert ctx.storage.sql.exec("SELECT total_changes() AS count").toArray() == [
        {"count": changes_before}
    ]
    await session.update_message(_message("shrink", f"{two_mb}!"))
    assert [event["type"] for event in changes] == ["update"]

    await session.update_message(_message("shrink", "tiny"))
    assert ctx.storage.sql.exec(
        "SELECT content_chunks FROM cf_agents_session_messages WHERE id = 'shrink'"
    ).toArray() == [{"content_chunks": 0}]
    assert (
        ctx.storage.sql.exec(
            "SELECT idx FROM cf_agents_session_message_chunks WHERE id = 'shrink'"
        ).toArray()
        == []
    )

    ctx.storage.sql.exec(
        "INSERT INTO cf_agents_session_message_chunks "
        "(session_id, id, idx, content) VALUES ('', 'shrink', 0, 'corrupt')"
    )
    await session.update_message(_message("shrink", "repaired"))
    assert (
        ctx.storage.sql.exec(
            "SELECT idx FROM cf_agents_session_message_chunks WHERE id = 'shrink'"
        ).toArray()
        == []
    )
    assert await session.get_message("shrink") == _message("shrink", "repaired")


@pytest.mark.asyncio
async def test_continuation_write_failure_rolls_back_the_message(monkeypatch):
    sessions, _, ctx = await _started()
    execute = ctx.storage.sql.exec

    def fail_continuation(query, *params):
        if "INSERT INTO cf_agents_session_message_chunks" in query:
            raise RuntimeError("continuation failed")
        return execute(query, *params)

    monkeypatch.setattr(ctx.storage.sql, "exec", fail_continuation)

    with pytest.raises(RuntimeError, match="continuation failed"):
        await sessions.session().append_message(
            _message("rollback", "x" * (2 * 1024 * 1024))
        )

    assert (
        execute(
            "SELECT id FROM cf_agents_session_messages WHERE id = 'rollback'"
        ).toArray()
        == []
    )

    monkeypatch.undo()
    await sessions.session().append_message(
        _message("existing", "before" * (512 * 1024))
    )
    monkeypatch.setattr(ctx.storage.sql, "exec", fail_continuation)

    with pytest.raises(RuntimeError, match="continuation failed"):
        await sessions.session().update_message(
            _message("existing", "after" * (512 * 1024))
        )
    assert (await sessions.session().get_message("existing"))["parts"][0][
        "text"
    ] == "before" * (512 * 1024)

    with pytest.raises(RuntimeError, match="continuation failed"):
        await sessions.session().import_message(
            _message("import-rollback", "x" * (2 * 1024 * 1024)),
            parent_id=None,
            created_at_ms=1,
        )
    assert (
        execute(
            "SELECT id FROM cf_agents_session_messages WHERE id = 'import-rollback'"
        ).toArray()
        == []
    )


@pytest.mark.asyncio
async def test_import_delete_and_clear_own_continuation_rows():
    sessions, _, ctx = await _started()
    session = sessions.session()
    body = "i" * (2 * 1024 * 1024)

    await session.import_message(
        _message("imported-big", body),
        parent_id=None,
        created_at_ms=1_000,
    )
    assert await session.get_message("imported-big") == _message("imported-big", body)
    assert ctx.storage.sql.exec(
        "SELECT idx FROM cf_agents_session_message_chunks WHERE id = 'imported-big'"
    ).toArray() == [{"idx": 1}]

    await session.delete_messages(["imported-big"])
    assert (
        ctx.storage.sql.exec(
            "SELECT idx FROM cf_agents_session_message_chunks WHERE id = 'imported-big'"
        ).toArray()
        == []
    )

    await session.append_message(_message("first-big", body))
    await session.append_message(_message("second-big", body))
    await session.clear_messages()
    assert ctx.storage.sql.exec(
        "SELECT COUNT(*) AS count FROM cf_agents_session_message_chunks"
    ).toArray() == [{"count": 0}]


@pytest.mark.asyncio
async def test_continuation_bytes_bound_recent_history():
    sessions, _, _ = await _started()
    session = sessions.session()
    await session.append_message(_message("root", "z" * 200))
    await session.append_message(_message("big", "z" * (2 * 1024 * 1024)))
    await session.append_message(_message("leaf", "z" * 200))

    root, big, leaf = await session.get_history_row_stats()
    assert big.bytes > 1536 * 1024
    generous = await session.get_recent_history(big.bytes + leaf.bytes + 64)
    assert [message["id"] for message in generous.messages] == ["big", "leaf"]
    tight = await session.get_recent_history(1536 * 1024 + leaf.bytes)
    assert [message["id"] for message in tight.messages] == ["leaf"]
    assert tight.total_content_bytes == root.bytes + big.bytes + leaf.bytes


@pytest.mark.asyncio
async def test_history_hydrates_in_fixed_windows_with_one_continuation_query():
    sessions, _, ctx = await _started()
    session = sessions.session()
    for index in range(49):
        await session.append_message(_message(f"small-{index}", "small"))
    await session.append_message(_message("big", "b" * (2 * 1024 * 1024)))
    await session.append_message(_message("last", "last"))

    queries = []
    execute = ctx.storage.sql.exec

    def record(query, *params):
        queries.append(query)
        return execute(query, *params)

    ctx.storage.sql.exec = record
    history = await session.get_history()

    assert len(history) == 51
    hydration_queries = [
        query
        for query in queries
        if "SELECT id, content, content_chunks" in query and "parent_id" not in query
    ]
    continuation_queries = [
        query
        for query in queries
        if "SELECT id, idx, content" in query
        and "FROM cf_agents_session_message_chunks" in query
    ]
    assert len(hydration_queries) == 2
    assert len(continuation_queries) == 1


@pytest.mark.asyncio
async def test_history_hydration_windows_are_also_bounded_by_stored_bytes():
    sessions, _, ctx = await _started()
    session = sessions.session()
    body = "b" * (2 * 1024 * 1024)
    for index in range(3):
        await session.append_message(_message(f"big-{index}", body))

    queries = []
    execute = ctx.storage.sql.exec

    def record(query, *params):
        queries.append(query)
        return execute(query, *params)

    ctx.storage.sql.exec = record
    assert len(await session.get_history()) == 3

    hydration_queries = [
        query
        for query in queries
        if "SELECT id, content, content_chunks" in query and "parent_id" not in query
    ]
    assert len(hydration_queries) == 3


@pytest.mark.asyncio
async def test_session_json_normalizes_surrogate_pairs_and_escapes_lone_surrogates():
    sessions, _, ctx = await _started()
    session = sessions.session()

    await session.append_message(_message("pair", "\ud83d\ude42"))
    await session.append_message(_message("lone", "\ud800"))

    rows = ctx.storage.sql.exec(
        "SELECT id, content FROM cf_agents_session_messages ORDER BY seq"
    ).toArray()
    assert "🙂" in rows[0]["content"]
    assert "\\ud800" in rows[1]["content"]
    assert (await session.get_message("pair"))["parts"][0]["text"] == "🙂"
    assert (await session.get_message("lone"))["parts"][0]["text"] == "\ud800"


@pytest.mark.asyncio
async def test_pre_aborted_history_does_not_hydrate_content():
    sessions, _, ctx = await _started()
    session = sessions.session()
    await session.append_message(_message("message", "message"))
    queries = []
    execute = ctx.storage.sql.exec

    def record(query, *params):
        queries.append(query)
        return execute(query, *params)

    class Signal:
        aborted = True
        reason = None

    ctx.storage.sql.exec = record
    with pytest.raises(RuntimeError, match="History read aborted"):
        await anext(session.history(HistoryReadOptions(signal=Signal())))
    assert all("SELECT id, content, content_chunks" not in query for query in queries)
