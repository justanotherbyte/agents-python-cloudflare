from __future__ import annotations

import asyncio
import json

import fakes
import pytest
from workers import Request

from agents import AIChatAgent, ChatMessageType
from agents.chat.recovery import incident_key
from agents.lifecycle.websockets import ConnectionContext


class RecoveringChat(AIChatAgent):
    durable_chat_recovery = True
    provider_calls = 0

    async def on_chat_message(self, options):
        type(self).provider_calls += 1

        async def chunks():
            yield "partial"
            raise asyncio.CancelledError()

        return chunks()


class EmptyRecoveringChat(AIChatAgent):
    durable_chat_recovery = True

    async def on_chat_message(self, options):
        raise asyncio.CancelledError()


class BoundedRecoveringChat(RecoveringChat):
    chat_recovery_max_chunks = 3


class TinyHydrationRecoveringChat(RecoveringChat):
    hydration_byte_budget = 1


class CompletingChat(AIChatAgent):
    durable_chat_recovery = True

    def on_chat_message(self, _options):
        return "completed reply"


class ExhaustionChat(CompletingChat):
    def __init__(self, ctx, env):
        super().__init__(ctx, env)
        self.exhaustions = 0

    def on_chat_recovery_exhausted(self, _context):
        self.exhaustions += 1


class AdoptingRecoveringChat(AIChatAgent):
    durable_chat_recovery = True

    async def on_chat_message(self, _options):
        yield {"type": "start", "messageId": "provider-message"}
        yield "durable output"
        raise asyncio.CancelledError()


class MixedRecoveryChat(RecoveringChat):
    async def on_fiber_recovered(self, _context):
        self.user_recoveries += 1
        raise RuntimeError("retry user recovery")


class ToolRecoveringChat(AIChatAgent):
    durable_chat_recovery = True

    async def on_chat_message(self, options):
        yield {
            "type": "tool-input-available",
            "toolCallId": "call",
            "toolName": "search",
            "input": {"query": "one"},
        }
        yield {
            "type": "tool-output-available",
            "toolCallId": "call",
            "output": "result",
        }
        raise asyncio.CancelledError()


def _body():
    return (
        '{"messages":[{"id":"user-1","role":"user",'
        '"parts":[{"type":"text","text":"hi"}]}]}'
    )


async def _interrupt(agent):
    connection = fakes.FakeConnection()
    agent._connections[connection.id] = connection
    data = {"id": "req-1", "init": {"method": "POST", "body": _body()}}
    with pytest.raises(asyncio.CancelledError):
        await agent._handle_use_chat_request(connection, data)


async def _wake_interrupted_turn(agent):
    await agent._ensure_initialized()
    await agent.alarm()


def _append_recovery_chunk(agent, body: dict) -> None:
    stream = agent._resumable.latest_for_request("req-1")
    assert stream is not None
    next_index = agent.sql(
        "SELECT COALESCE(MAX(chunk_index), -1) + 1 AS next_index "
        "FROM cf_ai_chat_stream_chunks WHERE stream_id = ?",
        stream.stream_id,
    )[0]["next_index"]
    agent.sql(
        "INSERT INTO cf_ai_chat_stream_chunks "
        "(id, stream_id, body, chunk_index, created_at) VALUES (?, ?, ?, ?, ?)",
        f"extra-{next_index}",
        stream.stream_id,
        json.dumps(body),
        next_index,
        next_index,
    )


@pytest.mark.asyncio
async def test_task_reconstructs_partial_turn_then_schedules_fresh_continuation():
    RecoveringChat.provider_calls = 0
    first = fakes.build_chat_agent(cls=RecoveringChat)
    await _interrupt(first)
    assert RecoveringChat.provider_calls == 1

    second = fakes.build_chat_agent(cls=RecoveringChat, conn=first.ctx.conn)
    await _wake_interrupted_turn(second)

    assert RecoveringChat.provider_calls == 1
    assistant = next(m for m in second.messages if m.get("role") == "assistant")
    assert assistant["parts"] == [{"type": "text", "text": "partial", "state": "done"}]
    assert second._resumable.has_active_stream() is False
    assert second.sql("SELECT definition, state FROM cf_agents_task_runs") == [
        {"definition": "__cf_internal_chat_recovery", "state": "pending"}
    ]

    await second.alarm()

    assert RecoveringChat.provider_calls == 2


@pytest.mark.asyncio
async def test_agent_tool_turn_runs_through_the_reserved_chat_task():
    agent = fakes.build_chat_agent(cls=CompletingChat)
    previous = await agent.prepare_agent_tool_turn("question", "tool-run")
    abort = asyncio.Event()
    result = None

    async def run(context):
        nonlocal result
        result = await agent.run_agent_tool_turn(
            "tool-run",
            "question",
            "tool-request",
            abort,
            context,
        )

    await agent._turn_queue.enqueue("tool-request", run)
    output, summary = await agent.collect_agent_tool_result(
        "tool-run",
        "question",
        previous_assistant_ids=previous,
    )

    assert result is None
    assert output == "completed reply"
    assert summary == "completed reply"
    assert agent.sql("SELECT run_id FROM cf_agents_task_runs") == []


@pytest.mark.asyncio
async def test_final_persistence_precedes_stream_completion(monkeypatch):
    first = fakes.build_chat_agent(cls=CompletingChat)
    connection = fakes.FakeConnection()
    first._connections[connection.id] = connection

    async def interrupt_persistence(*_args, **_kwargs):
        raise asyncio.CancelledError()

    monkeypatch.setattr(
        first,
        "_persist_finished_streaming_message",
        interrupt_persistence,
    )
    with pytest.raises(asyncio.CancelledError):
        await first._handle_use_chat_request(
            connection,
            {
                "id": "request",
                "init": {
                    "method": "POST",
                    "body": json.dumps(
                        {
                            "messages": [
                                {
                                    "id": "user",
                                    "role": "user",
                                    "parts": [{"type": "text", "text": "hi"}],
                                }
                            ]
                        }
                    ),
                },
            },
        )

    assert first.sql(
        "SELECT status FROM cf_ai_chat_stream_metadata WHERE request_id = 'request'"
    ) == [{"status": "streaming"}]

    second = fakes.build_chat_agent(cls=CompletingChat, conn=first.ctx.conn)
    await _wake_interrupted_turn(second)

    assistant = next(
        message for message in second.messages if message["role"] == "assistant"
    )
    assert assistant["parts"] == [
        {"type": "text", "text": "completed reply", "state": "done"}
    ]


@pytest.mark.asyncio
async def test_retryable_final_persistence_failure_remains_recoverable(monkeypatch):
    class RetryableStorageError(RuntimeError):
        retryable = True

    first = fakes.build_chat_agent(cls=CompletingChat)
    connection = fakes.FakeConnection()
    first._connections[connection.id] = connection

    async def fail_persistence(*_args, **_kwargs):
        raise RetryableStorageError("Durable Object storage reset")

    monkeypatch.setattr(
        first,
        "_persist_finished_streaming_message",
        fail_persistence,
    )
    with pytest.raises(RetryableStorageError):
        await first._handle_use_chat_request(
            connection,
            {"id": "request", "init": {"method": "POST", "body": _body()}},
        )

    assert first.sql(
        "SELECT status FROM cf_ai_chat_stream_metadata WHERE request_id = 'request'"
    ) == [{"status": "streaming"}]
    assert first.sql("SELECT definition FROM cf_agents_task_runs") == [
        {"definition": "__cf_internal_chat_turn"}
    ]

    second = fakes.build_chat_agent(cls=CompletingChat, conn=first.ctx.conn)
    await _wake_interrupted_turn(second)

    assistant = next(
        message for message in second.messages if message["role"] == "assistant"
    )
    assert assistant["parts"][0]["text"] == "completed reply"


@pytest.mark.asyncio
async def test_provider_adopted_message_id_survives_recovery():
    first = fakes.build_chat_agent(cls=AdoptingRecoveringChat)
    await _interrupt(first)

    stream = first._resumable.latest_for_request("req-1")
    assert stream is not None
    assert stream.message_id == "provider-message"

    second = fakes.build_chat_agent(cls=AdoptingRecoveringChat, conn=first.ctx.conn)
    await _wake_interrupted_turn(second)

    assistant = next(
        message for message in second.messages if message["role"] == "assistant"
    )
    assert assistant["id"] == "provider-message"
    output, _summary = await second.collect_agent_tool_result(
        "run",
        {},
        message_id=stream.message_id,
    )
    assert output == "durable output"


@pytest.mark.asyncio
async def test_unconfigured_facet_does_not_bypass_reserved_chat_task():
    agent = fakes.build_chat_agent(
        lambda _options: "completed reply",
        name="cf-agents:v2:chat:digest",
    )
    connection = fakes.FakeConnection()
    agent._connections[connection.id] = connection

    await agent._handle_use_chat_request(
        connection,
        {"id": "request", "init": {"method": "POST", "body": _body()}},
    )

    assert len(agent.sql("SELECT run_id FROM cf_agents_task_runs")) == 1
    assert agent._resumable.has_active_stream() is False
    assert [message["role"] for message in agent.messages] == ["user"]
    assert connection.sent[-1]["error"] is True


@pytest.mark.asyncio
async def test_legacy_lift_that_changes_the_leaf_invalidates_chat_recovery():
    first = fakes.build_chat_agent(cls=RecoveringChat)
    await _interrupt(first)
    first.sql("DELETE FROM cf_agents_session_messages")
    first.sql(
        "CREATE TABLE cf_ai_chat_agent_messages ("
        "id TEXT PRIMARY KEY, message TEXT NOT NULL, "
        "created_at DATETIME DEFAULT CURRENT_TIMESTAMP)"
    )
    first.sql(
        "INSERT INTO cf_ai_chat_agent_messages (id, message, created_at) "
        "VALUES ('user-1', ?, '2026-01-01 00:00:00')",
        json.dumps(
            {
                "id": "user-1",
                "role": "user",
                "parts": [{"type": "text", "text": "hi"}],
            }
        ),
    )
    first.sql(
        "INSERT INTO cf_ai_chat_agent_messages (id, message, created_at) "
        "VALUES ('user-2', 'not-json', '2026-01-01 00:00:01')"
    )

    second = fakes.build_chat_agent(cls=RecoveringChat, conn=first.ctx.conn)
    await second._ensure_initialized()

    assert second._lifecycle._ready is True
    assert second._legacy_migration_incomplete is True
    assert [message["role"] for message in second.messages] == ["user"]
    assert second.sql("SELECT definition FROM cf_agents_task_runs") == [
        {"definition": "__cf_internal_chat_turn"}
    ]

    second.sql(
        "UPDATE cf_ai_chat_agent_messages SET message = ? WHERE id = 'user-2'",
        json.dumps(
            {
                "id": "user-2",
                "role": "user",
                "parts": [{"type": "text", "text": "fixed"}],
            }
        ),
    )
    await second._ensure_chat_storage_ready()
    await second.alarm()

    assert (
        second.sql(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name = 'cf_ai_chat_agent_messages'"
        )
        == []
    )
    assert [message["role"] for message in second.messages] == [
        "user",
        "user",
    ]


@pytest.mark.asyncio
async def test_truncated_startup_recovery_does_not_broadcast_a_partial_transcript():
    first = fakes.build_chat_agent(cls=TinyHydrationRecoveringChat)
    await first._persist_messages(
        [
            {
                "id": f"message-{index}",
                "role": "user",
                "parts": [{"type": "text", "text": str(index)}],
            }
            for index in range(3)
        ]
    )
    await _interrupt(first)

    second = fakes.build_chat_agent(
        cls=TinyHydrationRecoveringChat,
        conn=first.ctx.conn,
    )
    connection = fakes.FakeConnection()
    second._connections[connection.id] = connection
    await _wake_interrupted_turn(second)

    assert any(message["role"] == "assistant" for message in second.messages)
    assert [
        frame
        for frame in connection.frames
        if frame.get("type") == "cf_agent_chat_messages"
    ] == []
    assert any(
        frame.get("type") == "cf_agent_message_updated" for frame in connection.frames
    )
    assert any(
        frame.get("id") == "req-1" and frame.get("done") is True
        for frame in connection.frames
    )


@pytest.mark.asyncio
async def test_task_recovery_does_not_retry_unrelated_fibers():
    first = fakes.build_chat_agent(cls=MixedRecoveryChat)
    await first._ensure_initialized()
    await _interrupt(first)
    first.sql(
        "INSERT INTO cf_agents_runs (id, name, snapshot, created_at) "
        "VALUES ('user-run', 'user-fiber', NULL, 9999999999999)"
    )

    second = fakes.build_chat_agent(cls=MixedRecoveryChat, conn=first.ctx.conn)
    second.user_recoveries = 0
    await _wake_interrupted_turn(second)

    assert second.user_recoveries == 1
    assert second.sql("SELECT id FROM cf_agents_runs WHERE id = 'user-run'") == [
        {"id": "user-run"}
    ]
    assert any(message["role"] == "assistant" for message in second.messages)


@pytest.mark.asyncio
async def test_task_recovery_is_idempotent():
    first = fakes.build_chat_agent(cls=RecoveringChat)
    await _interrupt(first)
    second = fakes.build_chat_agent(cls=RecoveringChat, conn=first.ctx.conn)

    await _wake_interrupted_turn(second)
    await second.alarm()

    assistants = [m for m in second.messages if m.get("role") == "assistant"]
    assert len(assistants) == 1


@pytest.mark.asyncio
async def test_zero_part_recovery_does_not_persist_phantom_assistant():
    first = fakes.build_chat_agent(cls=EmptyRecoveringChat)
    await _interrupt(first)
    second = fakes.build_chat_agent(cls=EmptyRecoveringChat, conn=first.ctx.conn)

    await _wake_interrupted_turn(second)

    assert [m for m in second.messages if m.get("role") == "assistant"] == []


@pytest.mark.asyncio
async def test_chat_clear_removes_interrupted_recovery_rows():
    first = fakes.build_chat_agent(cls=RecoveringChat)
    await _interrupt(first)
    second = fakes.build_chat_agent(cls=RecoveringChat, conn=first.ctx.conn)
    connection = fakes.FakeConnection()

    await second._handle_chat_clear(connection)

    assert second.sql("SELECT run_id FROM cf_agents_task_runs") == []


@pytest.mark.asyncio
async def test_recovery_overflow_persists_the_bounded_partial_prefix():
    first = fakes.build_chat_agent(cls=BoundedRecoveringChat)
    await _interrupt(first)
    _append_recovery_chunk(
        first,
        {"type": "message-metadata", "messageMetadata": {"extra": True}},
    )
    second = fakes.build_chat_agent(cls=BoundedRecoveringChat, conn=first.ctx.conn)

    await _wake_interrupted_turn(second)

    assistant = next(m for m in second.messages if m.get("role") == "assistant")
    assert assistant["parts"] == [{"type": "text", "text": "partial", "state": "done"}]
    terminal = second._resumable.latest_terminal_error()
    assert terminal is not None
    assert terminal.request_id == "req-1"
    assert "reconstruction limits" in terminal.body
    stream = second._resumable.latest_for_request("req-1")
    assert stream is not None
    assert stream.status == "error"


@pytest.mark.asyncio
async def test_recovery_uses_message_metadata_from_stored_chunks():
    first = fakes.build_chat_agent(cls=RecoveringChat)
    await _interrupt(first)
    _append_recovery_chunk(
        first,
        {"type": "message-metadata", "messageMetadata": {"model": "test"}},
    )
    second = fakes.build_chat_agent(cls=RecoveringChat, conn=first.ctx.conn)

    await _wake_interrupted_turn(second)

    assistant = next(m for m in second.messages if m.get("role") == "assistant")
    assert assistant["metadata"] == {"model": "test"}


@pytest.mark.asyncio
async def test_recovery_honors_replay_suppression_for_settled_tools():
    first = fakes.build_chat_agent(cls=ToolRecoveringChat)
    await _interrupt(first)
    _append_recovery_chunk(
        first,
        {
            "type": "tool-input-start",
            "toolCallId": "call",
            "toolName": "search",
        },
    )
    second = fakes.build_chat_agent(cls=ToolRecoveringChat, conn=first.ctx.conn)

    await _wake_interrupted_turn(second)

    assistant = next(m for m in second.messages if m.get("role") == "assistant")
    assert assistant["parts"] == [
        {
            "type": "tool-search",
            "toolCallId": "call",
            "toolName": "search",
            "state": "output-available",
            "input": {"query": "one"},
            "output": "result",
        }
    ]


@pytest.mark.asyncio
async def test_recovery_terminalizes_a_stored_error_chunk():
    first = fakes.build_chat_agent(cls=RecoveringChat)
    await _interrupt(first)
    _append_recovery_chunk(first, {"type": "error", "errorText": "stored failure"})
    second = fakes.build_chat_agent(cls=RecoveringChat, conn=first.ctx.conn)

    await _wake_interrupted_turn(second)

    assert [m for m in second.messages if m.get("role") == "assistant"] == []
    terminal = second._resumable.latest_terminal_error()
    assert terminal is not None
    assert terminal.body == "stored failure"


@pytest.mark.asyncio
async def test_recovery_exhaustion_hook_runs_once_per_incident():
    agent = fakes.build_chat_agent(cls=ExhaustionChat)
    incident = {
        "incidentId": "root:user",
        "requestId": "request",
        "recoveryRootRequestId": "root",
        "attempt": 11,
        "maxAttempts": 10,
        "status": "exhausted",
        "firstSeenAt": 1,
        "lastAttemptAt": 1,
        "lastProgressAt": 1,
        "progress": 0,
        "workBaseline": 0,
    }

    await agent._exhaust_recovery(
        {"recoveryRootRequestId": "root"},
        "max_attempts_exceeded",
        incident=incident,
    )
    stored = await agent.ctx.storage.get(incident_key("root:user"))
    await agent._exhaust_recovery(
        {"recoveryRootRequestId": "root"},
        "max_attempts_exceeded",
        incident=stored,
    )

    assert agent.exhaustions == 1
    assert stored["exhaustionReported"] is True


@pytest.mark.asyncio
async def test_post_handoff_platform_failure_enqueues_one_replacement(monkeypatch):
    class PlatformReset(RuntimeError):
        retryable = True

    agent = fakes.build_chat_agent(cls=CompletingChat)
    data = {
        "incidentId": "root:user",
        "kind": "retry",
        "originalRequestId": "root",
        "latestUserMessageId": "user",
        "body": {},
        "clientTools": None,
    }

    async def fail_after_handoff(_data, started):
        started.set()
        await asyncio.sleep(0)
        raise PlatformReset("Durable Object storage reset")

    monkeypatch.setattr(agent, "_execute_recovery_attempt", fail_after_handoff)

    await agent._dispatch_recovery_handoff(data)
    await agent._dispatch_recovery_handoff(data)

    rows = agent.sql(
        "SELECT run_id, definition, input FROM cf_agents_task_runs "
        "WHERE definition = '__cf_internal_chat_recovery'"
    )
    assert len(rows) == 1
    assert rows[0]["run_id"] == "chat-recovery-redefer:root:user:platform:1"
    assert json.loads(rows[0]["input"])["data"]["redeferSequence"] == 1
    assert json.loads(rows[0]["input"])["delaySeconds"] == 3


@pytest.mark.asyncio
async def test_recovery_acceptance_retries_are_bounded(monkeypatch):
    class PlatformReset(RuntimeError):
        retryable = True

    agent = fakes.build_chat_agent(cls=CompletingChat)
    attempts = 0

    async def accept(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        if attempts < agent.chat_recovery_acceptance_attempts:
            raise PlatformReset("Durable Object storage reset")

    monkeypatch.setattr(agent.tasks, "_run_reserved", accept)

    await agent._enqueue_recovery(
        {
            "incidentId": "incident",
            "kind": "retry",
            "originalRequestId": "request",
        },
        initial=False,
    )

    assert attempts == agent.chat_recovery_acceptance_attempts


@pytest.mark.asyncio
async def test_empty_pre_stream_continuation_stays_a_continuation():
    agent = fakes.build_chat_agent(cls=CompletingChat)
    await agent._persist_messages(
        [
            {
                "id": "user",
                "role": "user",
                "parts": [{"type": "text", "text": "question"}],
            },
            {
                "id": "assistant",
                "role": "assistant",
                "parts": [{"type": "text", "text": "partial"}],
            },
        ]
    )

    await agent._recover_interrupted_chat_turn(
        {
            "requestId": "request",
            "messageId": "assistant",
            "latestUserMessageId": "user",
            "continuation": True,
            "body": {},
            "clientTools": None,
        }
    )

    rows = agent.sql(
        "SELECT input FROM cf_agents_task_runs "
        "WHERE definition = '__cf_internal_chat_recovery'"
    )
    data = json.loads(rows[0]["input"])["data"]
    assert data["kind"] == "continue"
    assert data["targetAssistantId"] == "assistant"


@pytest.mark.asyncio
async def test_request_context_restores_and_chat_clear_removes_it():
    first = fakes.build_chat_agent(cls=CompletingChat)
    first._last_body = {"temperature": 0.25}
    first._last_client_tools = [{"name": "search"}]
    first._persist_request_context()

    second = fakes.build_chat_agent(cls=CompletingChat, conn=first.ctx.conn)
    await second._ensure_initialized()

    assert second._last_body == {"temperature": 0.25}
    assert second._last_client_tools == [{"name": "search"}]

    await second._handle_chat_clear(fakes.FakeConnection())
    third = fakes.build_chat_agent(cls=CompletingChat, conn=first.ctx.conn)
    await third._ensure_initialized()

    assert third._last_body == {}
    assert third._last_client_tools is None


@pytest.mark.asyncio
async def test_connect_replays_current_recovering_state():
    agent = fakes.build_chat_agent(cls=CompletingChat)
    await agent._set_recovering(True, "request")
    connection = fakes.FakeConnection()

    await agent._dispatch_connect(connection, ConnectionContext(Request()))

    assert {
        "type": ChatMessageType.CHAT_RECOVERING,
        "recovering": True,
        "id": "request",
    } in connection.frames
