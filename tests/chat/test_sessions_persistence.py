from __future__ import annotations

import asyncio
import json

import fakes
import pytest
from workers import Request

import agents.chat.agent as chat_module
from agents import AIChatAgent


def _message(message_id: str, text: str | None = None) -> dict:
    return {
        "id": message_id,
        "role": "user",
        "parts": [{"type": "text", "text": text or message_id}],
    }


def _tool_message(
    message_id: str,
    *,
    state: str,
    result: dict | None = None,
) -> dict:
    part = {
        "type": "tool-search",
        "toolCallId": "call-1",
        "state": state,
        "input": {"query": "weather"},
    }
    part.update(result or {})
    return {"id": message_id, "role": "assistant", "parts": [part]}


def _legacy_table(conn) -> None:
    conn.execute(
        "CREATE TABLE cf_ai_chat_agent_messages ("
        "id TEXT PRIMARY KEY, message TEXT NOT NULL, "
        "created_at DATETIME DEFAULT CURRENT_TIMESTAMP)"
    )


def _table_names(agent: AIChatAgent) -> set[str]:
    return {
        row["name"]
        for row in agent.sql("SELECT name FROM sqlite_master WHERE type = 'table'")
    }


@pytest.mark.asyncio
async def test_legacy_chat_messages_lift_to_one_session_chain_and_drop_source():
    conn = fakes.new_sqlite()
    _legacy_table(conn)
    conn.execute(
        "INSERT INTO cf_ai_chat_agent_messages (id, message, created_at) "
        "VALUES (?, ?, ?)",
        (
            "legacy-v4",
            json.dumps({"id": "legacy-v4", "role": "user", "content": "old"}),
            "2026-01-01 00:00:00",
        ),
    )
    conn.execute(
        "INSERT INTO cf_ai_chat_agent_messages (id, message, created_at) "
        "VALUES (?, ?, ?)",
        (
            "legacy-v5",
            json.dumps(_message("legacy-v5", "new")),
            "2026-01-01 00:00:01",
        ),
    )

    agent = fakes.build_chat_agent(conn=conn)
    assert agent.messages == []

    await agent._ensure_initialized()

    assert "cf_ai_chat_agent_messages" not in _table_names(agent)
    assert agent.messages == [
        _message("legacy-v4", "old"),
        _message("legacy-v5", "new"),
    ]
    assert agent.sql(
        "SELECT id, parent_id FROM cf_agents_session_messages "
        "WHERE session_id = '' ORDER BY seq"
    ) == [
        {"id": "legacy-v4", "parent_id": None},
        {"id": "legacy-v5", "parent_id": "legacy-v4"},
    ]


@pytest.mark.asyncio
async def test_incomplete_legacy_lift_retains_source_and_retries_on_next_wake():
    conn = fakes.new_sqlite()
    _legacy_table(conn)
    conn.execute(
        "INSERT INTO cf_ai_chat_agent_messages (id, message, created_at) "
        "VALUES ('first', ?, '2026-01-01 00:00:00')",
        (json.dumps(_message("first")),),
    )
    conn.execute(
        "INSERT INTO cf_ai_chat_agent_messages (id, message, created_at) "
        "VALUES ('second', 'not-json', '2026-01-01 00:00:01')"
    )

    first = fakes.build_chat_agent(conn=conn)
    await first._ensure_initialized()

    assert "cf_ai_chat_agent_messages" in _table_names(first)
    assert first._legacy_migration_incomplete is True
    assert first.sql(
        "SELECT id FROM cf_agents_session_messages WHERE session_id = '' ORDER BY seq"
    ) == [{"id": "first"}]

    conn.execute(
        "UPDATE cf_ai_chat_agent_messages SET message = ? WHERE id = 'second'",
        (json.dumps(_message("second")),),
    )
    second = fakes.build_chat_agent(conn=conn)
    await second._ensure_initialized()

    assert "cf_ai_chat_agent_messages" not in _table_names(second)
    assert [message["id"] for message in second.messages] == ["first", "second"]


@pytest.mark.asyncio
async def test_legacy_lift_stops_at_a_bad_predecessor_then_retries_one_chain():
    conn = fakes.new_sqlite()
    _legacy_table(conn)
    conn.execute(
        "INSERT INTO cf_ai_chat_agent_messages (id, message, created_at) "
        "VALUES ('first', 'not-json', '2026-01-01 00:00:00')"
    )
    conn.execute(
        "INSERT INTO cf_ai_chat_agent_messages (id, message, created_at) "
        "VALUES ('second', ?, '2026-01-01 00:00:01')",
        (json.dumps(_message("second")),),
    )

    first = fakes.build_chat_agent(conn=conn)
    await first._ensure_initialized()

    assert "cf_ai_chat_agent_messages" in _table_names(first)
    assert first._legacy_migration_incomplete is True
    assert (
        first.sql("SELECT id FROM cf_agents_session_messages WHERE session_id = ''")
        == []
    )

    conn.execute(
        "UPDATE cf_ai_chat_agent_messages SET message = ? WHERE id = 'first'",
        (json.dumps(_message("first")),),
    )
    second = fakes.build_chat_agent(conn=conn)
    await second._ensure_initialized()

    assert "cf_ai_chat_agent_messages" not in _table_names(second)
    assert second.sql(
        "SELECT id, parent_id FROM cf_agents_session_messages "
        "WHERE session_id = '' ORDER BY seq"
    ) == [
        {"id": "first", "parent_id": None},
        {"id": "second", "parent_id": "first"},
    ]


@pytest.mark.asyncio
async def test_legacy_lift_retains_source_on_conflicting_destination_message():
    conn = fakes.new_sqlite()
    existing = fakes.build_chat_agent(conn=conn)
    await existing._persist_messages([_message("same-id", "destination")])
    _legacy_table(conn)
    conn.execute(
        "INSERT INTO cf_ai_chat_agent_messages (id, message, created_at) "
        "VALUES ('same-id', ?, '2026-01-01 00:00:00')",
        (json.dumps(_message("same-id", "source")),),
    )

    migrated = fakes.build_chat_agent(conn=conn)
    await migrated._ensure_initialized()

    assert "cf_ai_chat_agent_messages" in _table_names(migrated)
    assert migrated._legacy_migration_incomplete is True
    assert migrated.sql(
        "SELECT id FROM cf_agents_session_messages WHERE session_id = '' ORDER BY seq"
    ) == [{"id": "same-id"}]


@pytest.mark.asyncio
async def test_legacy_lift_rejects_a_modern_message_with_an_unsupported_role():
    conn = fakes.new_sqlite()
    _legacy_table(conn)
    invalid = _message("invalid-role")
    invalid["role"] = "data"
    conn.execute(
        "INSERT INTO cf_ai_chat_agent_messages (id, message, created_at) "
        "VALUES ('invalid-role', ?, '2026-01-01 00:00:00')",
        (json.dumps(invalid),),
    )

    agent = fakes.build_chat_agent(conn=conn)
    await agent._ensure_initialized()

    assert "cf_ai_chat_agent_messages" in _table_names(agent)
    assert agent._legacy_migration_incomplete is True
    assert (
        agent.sql("SELECT id FROM cf_agents_session_messages WHERE session_id = ''")
        == []
    )


@pytest.mark.asyncio
async def test_malformed_legacy_schema_fences_only_chat_storage():
    agent = fakes.build_chat_agent()
    agent.sql(
        "CREATE TABLE cf_ai_chat_agent_messages ("
        "id TEXT PRIMARY KEY, message TEXT NOT NULL)"
    )
    agent.sql(
        "INSERT INTO cf_ai_chat_agent_messages (id, message) VALUES ('legacy', ?)",
        json.dumps(_message("legacy")),
    )

    await agent._ensure_initialized()

    assert agent._lifecycle._ready is True
    assert agent._legacy_migration_incomplete is True
    with pytest.raises(RuntimeError, match="legacy chat migration incomplete"):
        await agent._persist_messages([_message("blocked")])


@pytest.mark.asyncio
async def test_incomplete_legacy_lift_blocks_intervening_session_writes():
    conn = fakes.new_sqlite()
    _legacy_table(conn)
    conn.execute(
        "INSERT INTO cf_ai_chat_agent_messages (id, message, created_at) "
        "VALUES ('first', ?, '2026-01-01 00:00:00')",
        (json.dumps(_message("first")),),
    )
    conn.execute(
        "INSERT INTO cf_ai_chat_agent_messages (id, message, created_at) "
        "VALUES ('second', 'not-json', '2026-01-01 00:00:01')"
    )
    agent = fakes.build_chat_agent(conn=conn)

    with pytest.raises(RuntimeError, match="legacy chat migration incomplete"):
        await agent._persist_messages([_message("intervening")])

    assert agent.sql(
        "SELECT id FROM cf_agents_session_messages WHERE session_id = '' ORDER BY seq"
    ) == [{"id": "first"}]

    conn.execute(
        "UPDATE cf_ai_chat_agent_messages SET message = ? WHERE id = 'second'",
        (json.dumps(_message("second")),),
    )
    retried = fakes.build_chat_agent(conn=conn)
    await retried._ensure_initialized()

    assert "cf_ai_chat_agent_messages" not in _table_names(retried)
    assert [message["id"] for message in retried.messages] == ["first", "second"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        "{}",
        '{"messages":null}',
        '{"messages":{}}',
        '{"messages":[{"id":"existing","role":"user","parts":[{}]}]}',
    ],
)
async def test_malformed_chat_request_cannot_delete_stored_history(body):
    agent = fakes.build_chat_agent()
    original = _message("existing")
    await agent._persist_messages([original])
    connection = fakes.FakeConnection()
    agent._connections[connection.id] = connection

    await agent._handle_use_chat_request(
        connection,
        {"id": "request", "init": {"method": "POST", "body": body}},
    )

    assert await agent._session.get_history() == [original]


@pytest.mark.asyncio
async def test_bounded_hydration_keeps_full_history_available_as_streamed_json():
    class TinyHydrationChat(AIChatAgent):
        hydration_byte_budget = 1

    first = fakes.build_chat_agent(cls=TinyHydrationChat)
    messages = [_message(f"message-{index}") for index in range(3)]
    await first._persist_messages(messages)

    second = fakes.build_chat_agent(cls=TinyHydrationChat, conn=first.ctx.conn)
    await second._ensure_initialized()

    assert [message["id"] for message in second.messages] == ["message-2"]
    response = await second._dispatch_request(
        Request("https://example.com/agents/chat/name/get-messages")
    )
    body = await response.body.read_all()
    assert response.headers["content-type"] == "application/json"
    assert json.loads(body) == messages


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_streamed_history_releases_callback_proxies(monkeypatch, cancel):
    destroyed = []

    class RetainedProxy:
        def __init__(self, callback):
            self.callback = callback

        def __call__(self, *args):
            return self.callback(*args)

        def destroy(self):
            destroyed.append(self)

    proxies = []

    def retain(callback):
        proxy = RetainedProxy(callback)
        proxies.append(proxy)
        return proxy

    monkeypatch.setattr(chat_module, "create_proxy", retain)
    agent = fakes.build_chat_agent()
    await agent._persist_messages([_message("first")])
    response = await agent._dispatch_request(
        Request("https://example.com/agents/chat/name/get-messages")
    )

    if cancel:
        await response.body.cancel()
    else:
        await response.body.read_all()

    assert len(proxies) == 2
    assert destroyed == list(reversed(proxies))


@pytest.mark.asyncio
async def test_streamed_history_releases_first_proxy_when_second_creation_fails(
    monkeypatch,
):
    destroyed = []

    class RetainedProxy:
        def __init__(self, callback):
            self.callback = callback

        def destroy(self):
            destroyed.append(self)

    first = RetainedProxy(lambda: None)
    calls = 0

    def retain(callback):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("proxy creation failed")
        first.callback = callback
        return first

    monkeypatch.setattr(chat_module, "create_proxy", retain)
    agent = fakes.build_chat_agent()

    with pytest.raises(RuntimeError, match="proxy creation failed"):
        agent._stream_messages_as_json()

    assert destroyed == [first]


@pytest.mark.asyncio
async def test_streamed_history_releases_proxies_when_pull_is_cancelled(monkeypatch):
    started = asyncio.Event()
    blocked = asyncio.Event()
    destroyed = []

    class RetainedProxy:
        def __init__(self, callback):
            self.callback = callback

        def __call__(self, *args):
            return self.callback(*args)

        def destroy(self):
            destroyed.append(self)

    proxies = []

    def retain(callback):
        proxy = RetainedProxy(callback)
        proxies.append(proxy)
        return proxy

    async def blocked_history():
        started.set()
        await blocked.wait()
        yield []

    monkeypatch.setattr(chat_module, "create_proxy", retain)
    agent = fakes.build_chat_agent()
    await agent._ensure_initialized()
    monkeypatch.setattr(agent._session, "history_batches", blocked_history)
    response = await agent._dispatch_request(
        Request("https://example.com/agents/chat/name/get-messages")
    )
    reading = asyncio.create_task(response.body.read_all())
    await asyncio.wait_for(started.wait(), timeout=1)

    reading.cancel()
    with pytest.raises(asyncio.CancelledError):
        await reading

    assert destroyed == list(reversed(proxies))


@pytest.mark.asyncio
async def test_retention_counts_stored_rows_outside_the_hydrated_window():
    class RetainedChat(AIChatAgent):
        hydration_byte_budget = 1
        max_persisted_messages = 3

    first = fakes.build_chat_agent(cls=RetainedChat)
    await first._persist_messages([_message(f"message-{index}") for index in range(3)])

    second = fakes.build_chat_agent(cls=RetainedChat, conn=first.ctx.conn)
    await second._ensure_initialized()
    assert [message["id"] for message in second.messages] == ["message-2"]

    await second._persist_messages([_message("message-3")])

    history = await second._session.get_history()
    assert [message["id"] for message in history] == [
        "message-1",
        "message-2",
        "message-3",
    ]


@pytest.mark.asyncio
async def test_regeneration_deletes_only_a_strict_server_subset():
    agent = fakes.build_chat_agent()
    messages = [_message(f"message-{index}") for index in range(3)]
    await agent._persist_messages(messages)

    await agent._persist_messages(
        [messages[0], messages[2]],
        delete_stale_rows=True,
    )

    history = await agent._session.get_history()
    assert [message["id"] for message in history] == ["message-0", "message-2"]


@pytest.mark.asyncio
async def test_regeneration_reconciles_against_history_outside_hydrated_window():
    class TinyHydrationChat(AIChatAgent):
        hydration_byte_budget = 1

    first = fakes.build_chat_agent(cls=TinyHydrationChat)
    messages = [_message(f"message-{index}") for index in range(3)]
    await first._persist_messages(messages)

    second = fakes.build_chat_agent(cls=TinyHydrationChat, conn=first.ctx.conn)
    await second._ensure_initialized()
    assert [message["id"] for message in second.messages] == ["message-2"]

    await second._persist_messages(
        [messages[0], messages[2]],
        delete_stale_rows=True,
    )

    assert await second._session.get_history() == [messages[0], messages[2]]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "settled_result",
    [
        {"state": "output-available", "output": "settled"},
        {"state": "output-error", "errorText": "failed"},
        {
            "state": "output-denied",
            "approval": {"id": "approval-1", "approved": False},
        },
    ],
)
async def test_reconciliation_preserves_settled_tool_output_outside_hydrated_window(
    settled_result,
):
    class TinyHydrationChat(AIChatAgent):
        hydration_byte_budget = 1

    first = fakes.build_chat_agent(cls=TinyHydrationChat)
    settled = _tool_message(
        "server-assistant",
        state=settled_result["state"],
        result=settled_result,
    )
    await first._persist_messages([settled, _message("latest")])

    second = fakes.build_chat_agent(cls=TinyHydrationChat, conn=first.ctx.conn)
    await second._ensure_initialized()
    assert [message["id"] for message in second.messages] == ["latest"]

    await second._persist_messages(
        [_tool_message("client-assistant", state="input-available")]
    )

    history = await second._session.get_history()
    assert [message["id"] for message in history] == ["server-assistant", "latest"]
    for key, value in settled_result.items():
        assert history[0]["parts"][0][key] == value


@pytest.mark.asyncio
async def test_reconciliation_restores_an_assistant_id_outside_hydrated_window():
    class TinyHydrationChat(AIChatAgent):
        hydration_byte_budget = 1

    first = fakes.build_chat_agent(cls=TinyHydrationChat)
    server = {
        "id": "server-assistant",
        "role": "assistant",
        "parts": [{"type": "text", "text": "same reply"}],
    }
    await first._persist_messages([server, _message("latest")])

    second = fakes.build_chat_agent(cls=TinyHydrationChat, conn=first.ctx.conn)
    await second._ensure_initialized()
    await second._persist_messages(
        [
            {
                **server,
                "id": "client-assistant",
            }
        ]
    )

    history = await second._session.get_history()
    assert [message["id"] for message in history] == ["server-assistant", "latest"]


@pytest.mark.asyncio
async def test_exact_reconciliation_does_not_scan_full_history(monkeypatch):
    class TinyHydrationChat(AIChatAgent):
        hydration_byte_budget = 1

    first = fakes.build_chat_agent(cls=TinyHydrationChat)
    messages = [_message(f"message-{index}") for index in range(100)]
    await first._persist_messages(messages)
    agent = fakes.build_chat_agent(cls=TinyHydrationChat, conn=first.ctx.conn)
    await agent._ensure_initialized()
    assert agent._messages_truncated is True

    async def fail_if_scanned():
        raise AssertionError("exact IDs should not scan history")
        yield []

    monkeypatch.setattr(agent._session, "history_batches", fail_if_scanned)

    assert await agent._persist_messages([messages[-1]]) == [messages[-1]]


@pytest.mark.asyncio
async def test_reconciliation_retains_only_needed_duplicate_content_candidates():
    class TinyHydrationChat(AIChatAgent):
        hydration_byte_budget = 1

    first = fakes.build_chat_agent(cls=TinyHydrationChat)
    duplicates = [
        {
            "id": f"assistant-{index}",
            "role": "assistant",
            "parts": [{"type": "text", "text": "same reply"}],
        }
        for index in range(50)
    ]
    await first._persist_messages([*duplicates, _message("latest")])
    second = fakes.build_chat_agent(cls=TinyHydrationChat, conn=first.ctx.conn)
    await second._ensure_initialized()
    incoming = [{**duplicates[0], "id": "client-assistant"}]

    prior = await second._reconciliation_prior(incoming)

    assert len(prior) == 1
    assert prior[0]["id"] == "assistant-0"


@pytest.mark.asyncio
async def test_agent_tool_result_reads_its_message_outside_hydrated_window():
    class TinyHydrationChat(AIChatAgent):
        hydration_byte_budget = 1

    first = fakes.build_chat_agent(cls=TinyHydrationChat)
    assistant = {
        "id": "tool-assistant",
        "role": "assistant",
        "parts": [{"type": "text", "text": "durable output"}],
    }
    await first._persist_messages([assistant, _message("latest")])
    second = fakes.build_chat_agent(cls=TinyHydrationChat, conn=first.ctx.conn)
    await second._ensure_initialized()

    output, summary = await second.collect_agent_tool_result(
        "run",
        {},
        message_id="tool-assistant",
    )

    assert output == "durable output"
    assert summary == "durable output"


@pytest.mark.asyncio
async def test_reconciliation_sanitizes_ephemeral_metadata_before_matching_ids():
    agent = fakes.build_chat_agent()
    server = {
        "id": "server-assistant",
        "role": "assistant",
        "parts": [
            {
                "type": "text",
                "text": "same reply",
                "providerMetadata": {"openai": {"itemId": "server-item"}},
            }
        ],
    }
    await agent._persist_messages([server])

    client = {
        **server,
        "id": "client-assistant",
        "parts": [
            {
                **server["parts"][0],
                "providerMetadata": {"openai": {"itemId": "client-item"}},
            }
        ],
    }
    await agent._persist_messages([client])

    history = await agent._session.get_history()
    assert [message["id"] for message in history] == ["server-assistant"]


@pytest.mark.asyncio
async def test_session_change_feed_keeps_the_activation_cache_coherent():
    agent = fakes.build_chat_agent()
    await agent._ensure_initialized()

    await agent._session.append_message(_message("first"))
    assert agent.messages == [_message("first")]

    await agent._session.update_message(_message("first", "updated"))
    assert agent.messages == [_message("first", "updated")]

    await agent._session.append_message(_message("second"))
    await agent._session.delete_messages(["first"])
    assert agent.messages == [_message("second")]

    await agent._session.clear_messages()
    assert agent.messages == []
