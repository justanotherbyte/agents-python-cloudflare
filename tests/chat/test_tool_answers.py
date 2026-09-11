from __future__ import annotations

import asyncio
import copy
import json
import types
from typing import Any, cast

import fakes
import pytest

import agents.chat.agent as chat_module
from agents import ChatMessageType
from agents.lifecycle.websockets import Connection


def _message(
    *,
    message_id: str = "assistant-1",
    call_id: str = "call-1",
    state: str = "input-available",
    approval: dict[str, Any] | None = None,
    sibling: bool = False,
) -> dict[str, Any]:
    part: dict[str, Any] = {
        "type": "tool-search",
        "toolCallId": call_id,
        "toolName": "search",
        "state": state,
        "input": {"query": "weather"},
    }
    if approval is not None:
        part["approval"] = approval
    parts = [part]
    if sibling:
        parts.append({"type": "text", "text": "unchanged sibling"})
    return {
        "id": message_id,
        "role": "assistant",
        "parts": parts,
    }


def _frame(frame_type: ChatMessageType, **detail: Any) -> str:
    return json.dumps({"type": frame_type, **detail})


def _turn_request() -> dict[str, Any]:
    return {
        "id": "request-1",
        "init": {
            "method": "POST",
            "body": json.dumps(
                {
                    "messages": [
                        {
                            "id": "user-1",
                            "role": "user",
                            "parts": [{"type": "text", "text": "hi"}],
                        }
                    ]
                }
            ),
        },
    }


def _updates(connection: fakes.FakeConnection) -> list[dict[str, Any]]:
    return [
        frame
        for frame in connection.frames
        if frame.get("type") == ChatMessageType.MESSAGE_UPDATED
    ]


def _tool_part(message: dict[str, Any], call_id: str = "call-1") -> dict[str, Any]:
    return next(part for part in message["parts"] if part.get("toolCallId") == call_id)


def _connect(agent, *connections: fakes.FakeConnection) -> None:
    for connection in connections:
        agent._connections[connection.id] = cast(Connection, connection)


async def _dispatch(
    agent,
    connection: fakes.FakeConnection,
    frame_type: ChatMessageType,
    **detail: Any,
) -> None:
    await agent._dispatch_message(connection, _frame(frame_type, **detail))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("frame_type", "detail", "expected"),
    [
        (
            ChatMessageType.TOOL_RESULT,
            {"toolCallId": "call-1", "output": {"temperature": 72}},
            {
                "state": "output-available",
                "output": {"temperature": 72},
                "preliminary": False,
            },
        ),
        (
            ChatMessageType.TOOL_RESULT,
            {"toolCallId": "call-1", "state": "output-error"},
            {
                "state": "output-error",
                "errorText": "Tool execution denied by user",
            },
        ),
        (
            ChatMessageType.TOOL_RESULT,
            {
                "toolCallId": "call-1",
                "state": "output-error",
                "errorText": "client failed",
            },
            {"state": "output-error", "errorText": "client failed"},
        ),
        (
            ChatMessageType.TOOL_APPROVAL,
            {"toolCallId": "call-1", "approved": True},
            {
                "state": "approval-responded",
                "approval": {
                    "id": "approval-1",
                    "descriptor": "Search the web",
                    "approved": True,
                },
            },
        ),
        (
            ChatMessageType.TOOL_APPROVAL,
            {"toolCallId": "call-1", "approved": False},
            {
                "state": "output-denied",
                "approval": {
                    "id": "approval-1",
                    "descriptor": "Search the web",
                    "approved": False,
                },
            },
        ),
    ],
)
async def test_persisted_tool_answers_broadcast_whole_message_and_restore(
    frame_type,
    detail,
    expected,
):
    agent = fakes.build_chat_agent()
    message = _message(
        approval={"id": "approval-1", "descriptor": "Search the web"},
        sibling=True,
    )
    expected_message = copy.deepcopy(message)
    _tool_part(expected_message).update(expected)
    await agent._persist_messages([message])
    sender = fakes.FakeConnection(id="sender")
    observer = fakes.FakeConnection(id="observer")
    _connect(agent, sender, observer)

    await _dispatch(agent, sender, frame_type, **detail)

    expected_frame = {
        "type": ChatMessageType.MESSAGE_UPDATED,
        "message": expected_message,
    }
    assert agent.messages == [expected_message]
    assert _updates(sender) == [expected_frame]
    assert _updates(observer) == [expected_frame]
    assert set(_updates(sender)[0]) == {"type", "message"}

    restored = fakes.build_chat_agent(conn=agent.ctx.conn)
    await restored._ensure_initialized()
    assert restored.messages == [expected_message]


@pytest.mark.asyncio
async def test_approval_then_result_advances_once_per_stage_and_retries_are_quiet():
    agent = fakes.build_chat_agent()
    await agent._persist_messages([_message(state="approval-requested", approval={})])
    connection = fakes.FakeConnection()
    _connect(agent, connection)

    await _dispatch(
        agent,
        connection,
        ChatMessageType.TOOL_APPROVAL,
        toolCallId="call-1",
        approved=True,
    )
    await _dispatch(
        agent,
        connection,
        ChatMessageType.TOOL_RESULT,
        toolCallId="call-1",
        output="first result",
    )
    await _dispatch(
        agent,
        connection,
        ChatMessageType.TOOL_RESULT,
        toolCallId="call-1",
        output="first result",
    )
    await _dispatch(
        agent,
        connection,
        ChatMessageType.TOOL_RESULT,
        toolCallId="call-1",
        state="output-error",
        errorText="late conflict",
    )
    await _dispatch(
        agent,
        connection,
        ChatMessageType.TOOL_APPROVAL,
        toolCallId="call-1",
        approved=False,
    )

    part = agent.messages[0]["parts"][0]
    assert part["state"] == "output-available"
    assert part["output"] == "first result"
    assert part["approval"] == {"id": "call-1", "approved": True}
    assert len(_updates(connection)) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("approved", [True, False])
async def test_approval_retries_are_first_write_wins(approved):
    agent = fakes.build_chat_agent()
    await agent._persist_messages([_message(state="approval-requested", approval={})])
    connection = fakes.FakeConnection()
    _connect(agent, connection)

    for answer in (approved, approved, not approved):
        await _dispatch(
            agent,
            connection,
            ChatMessageType.TOOL_APPROVAL,
            toolCallId="call-1",
            approved=answer,
        )

    part = _tool_part(agent.messages[0])
    expected_state = "approval-responded" if approved else "output-denied"
    assert part["state"] == expected_state
    assert part["approval"] == {"id": "call-1", "approved": approved}
    assert len(_updates(connection)) == 1


@pytest.mark.asyncio
async def test_concurrent_results_are_first_write_wins():
    entries: list[str] = []

    class RecordingChat(chat_module.AIChatAgent):
        async def _handle_tool_result(self, data):
            entries.append(data["output"])
            await super()._handle_tool_result(data)

    agent = fakes.build_chat_agent(cls=RecordingChat)
    await agent._persist_messages([_message()])
    connection = fakes.FakeConnection()
    _connect(agent, connection)

    await asyncio.gather(
        _dispatch(
            agent,
            connection,
            ChatMessageType.TOOL_RESULT,
            toolCallId="call-1",
            output="first",
        ),
        _dispatch(
            agent,
            connection,
            ChatMessageType.TOOL_RESULT,
            toolCallId="call-1",
            output="second",
        ),
    )

    assert agent.messages[0]["parts"][0]["output"] == entries[0]
    assert len(_updates(connection)) == 1
    restored = fakes.build_chat_agent(conn=agent.ctx.conn)
    await restored._ensure_initialized()
    assert restored.messages[0]["parts"][0]["output"] == entries[0]


@pytest.mark.asyncio
async def test_concurrent_rejected_approval_and_result_are_first_write_wins():
    entries: list[str] = []

    class RecordingChat(chat_module.AIChatAgent):
        async def _handle_tool_approval(self, data):
            entries.append("approval")
            await super()._handle_tool_approval(data)

        async def _handle_tool_result(self, data):
            entries.append("result")
            await super()._handle_tool_result(data)

    agent = fakes.build_chat_agent(cls=RecordingChat)
    await agent._persist_messages([_message(state="approval-requested", approval={})])
    connection = fakes.FakeConnection()
    _connect(agent, connection)

    await asyncio.gather(
        _dispatch(
            agent,
            connection,
            ChatMessageType.TOOL_APPROVAL,
            toolCallId="call-1",
            approved=False,
        ),
        _dispatch(
            agent,
            connection,
            ChatMessageType.TOOL_RESULT,
            toolCallId="call-1",
            output="result",
        ),
    )

    expected_state = "output-denied" if entries[0] == "approval" else "output-available"
    assert _tool_part(agent.messages[0])["state"] == expected_state
    assert len(_updates(connection)) == 1
    restored = fakes.build_chat_agent(conn=agent.ctx.conn)
    await restored._ensure_initialized()
    assert restored.messages == agent.messages


@pytest.mark.asyncio
async def test_inflight_message_takes_priority_over_persisted_duplicate():
    started = asyncio.Event()
    release = asyncio.Event()

    async def reply(_options):
        yield {
            "type": "tool-input-available",
            "toolCallId": "call-1",
            "toolName": "search",
            "input": {"query": "new"},
        }
        started.set()
        await release.wait()

    agent = fakes.build_chat_agent(reply)
    old = _message(message_id="old", approval={"id": "old-approval"})
    await agent._persist_messages([old])
    connection = fakes.FakeConnection(id="submitter")
    _connect(agent, connection)
    turn = asyncio.create_task(
        agent._handle_use_chat_request(
            cast(Connection, connection),
            _turn_request(),
        )
    )
    try:
        await asyncio.wait_for(started.wait(), timeout=1)
        await _dispatch(
            agent,
            connection,
            ChatMessageType.TOOL_RESULT,
            toolCallId="call-1",
            output="new result",
        )
    finally:
        release.set()
        await asyncio.wait_for(turn, timeout=1)

    persisted_old = next(
        message for message in agent.messages if message["id"] == "old"
    )
    current = next(
        message
        for message in agent.messages
        if message.get("role") == "assistant" and message["id"] != "old"
    )
    assert _tool_part(persisted_old)["state"] == "input-available"
    assert _tool_part(current)["state"] == "output-available"
    assert _tool_part(current)["output"] == "new result"


@pytest.mark.asyncio
async def test_tool_result_racing_final_persistence_survives_restart(monkeypatch):
    release = asyncio.Event()
    persist_started = asyncio.Event()

    async def reply(_options):
        yield {
            "type": "tool-input-available",
            "toolCallId": "call-1",
            "toolName": "search",
            "input": {"query": "weather"},
        }
        await release.wait()

    agent = fakes.build_chat_agent(reply)
    await agent._ensure_initialized()
    original_upsert = agent._session.upsert_message
    continue_persist = asyncio.Event()

    async def paused_upsert(message, options=None):
        if message["role"] != "assistant":
            return await original_upsert(message, options)
        persist_started.set()
        await continue_persist.wait()
        return await original_upsert(message, options)

    monkeypatch.setattr(agent._session, "upsert_message", paused_upsert)
    connection = fakes.FakeConnection(id="submitter")
    _connect(agent, connection)
    turn = asyncio.create_task(
        agent._handle_use_chat_request(
            cast(Connection, connection),
            _turn_request(),
        )
    )
    try:
        while agent._streaming_message is None:
            await asyncio.sleep(0)
        release.set()
        await asyncio.wait_for(persist_started.wait(), timeout=1)
        result = asyncio.create_task(
            _dispatch(
                agent,
                connection,
                ChatMessageType.TOOL_RESULT,
                toolCallId="call-1",
                output="settled while persisting",
            )
        )
        await asyncio.sleep(0)
        continue_persist.set()
        await asyncio.wait_for(asyncio.gather(turn, result), timeout=1)
    finally:
        release.set()
        continue_persist.set()
        if not turn.done():
            turn.cancel()
        await asyncio.gather(turn, return_exceptions=True)

    restored = fakes.build_chat_agent(conn=agent.ctx.conn)
    await restored._ensure_initialized()
    assistant = next(
        message for message in restored.messages if message["role"] == "assistant"
    )
    assert _tool_part(assistant)["output"] == "settled while persisting"


@pytest.mark.asyncio
async def test_final_persistence_uses_cache_and_broadcasts_the_full_transcript(
    monkeypatch,
):
    agent = None

    async def fail_history():
        raise AssertionError("final persistence materialized durable history")

    async def reply(_options):
        monkeypatch.setattr(agent._session, "get_history", fail_history)
        yield "hello"

    agent = fakes.build_chat_agent(reply)
    connection = fakes.FakeConnection(id="submitter")
    _connect(agent, connection)

    await agent._handle_use_chat_request(
        cast(Connection, connection),
        _turn_request(),
    )

    transcripts = [
        frame
        for frame in connection.frames
        if frame.get("type") == ChatMessageType.CHAT_MESSAGES
    ]
    assert [
        [message["role"] for message in frame["messages"]] for frame in transcripts
    ] == [["user", "assistant"]]
    restored = fakes.build_chat_agent(conn=agent.ctx.conn)
    await restored._ensure_initialized()
    assert [message["role"] for message in restored.messages] == ["user", "assistant"]


@pytest.mark.asyncio
async def test_approval_prompt_snapshot_is_durable_and_silent_before_turn_ends():
    prompt_ready = asyncio.Event()
    release_prompt = asyncio.Event()
    snapshotted = asyncio.Event()
    release = asyncio.Event()

    async def reply(_options):
        yield {
            "type": "tool-input-available",
            "toolCallId": "call-1",
            "toolName": "search",
            "input": {"query": "weather"},
        }
        prompt_ready.set()
        await release_prompt.wait()
        yield {
            "type": "tool-approval-request",
            "toolCallId": "call-1",
            "approvalId": "approval-1",
            "approvalDescriptor": "Search the web",
        }
        snapshotted.set()
        await release.wait()

    agent = fakes.build_chat_agent(reply)
    connection = fakes.FakeConnection(id="submitter")
    observer = fakes.FakeConnection(id="observer")
    _connect(agent, connection, observer)
    turn = asyncio.create_task(
        agent._handle_use_chat_request(
            cast(Connection, connection),
            _turn_request(),
        )
    )
    try:
        await asyncio.wait_for(prompt_ready.wait(), timeout=1)
        frame_types = {
            ChatMessageType.CHAT_MESSAGES,
            ChatMessageType.MESSAGE_UPDATED,
        }
        baseline_frames = {
            connected.id: [
                frame for frame in connected.frames if frame.get("type") in frame_types
            ]
            for connected in (connection, observer)
        }
        release_prompt.set()
        await asyncio.wait_for(snapshotted.wait(), timeout=1)
        restored = fakes.build_chat_agent(conn=agent.ctx.conn)
        await restored._ensure_initialized()
        part = next(
            _tool_part(message)
            for message in restored.messages
            if any(part.get("toolCallId") == "call-1" for part in message["parts"])
        )
        assert part["state"] == "approval-requested"
        assert part["approval"] == {
            "id": "approval-1",
            "descriptor": "Search the web",
        }
        for connected in (connection, observer):
            current_frames = [
                frame for frame in connected.frames if frame.get("type") in frame_types
            ]
            assert current_frames == baseline_frames[connected.id]
    finally:
        release_prompt.set()
        release.set()
        await asyncio.wait_for(turn, timeout=1)


@pytest.mark.asyncio
async def test_tool_result_waits_for_a_message_that_is_about_to_persist(monkeypatch):
    agent = fakes.build_chat_agent()
    connection = fakes.FakeConnection()
    _connect(agent, connection)
    sleep_gate = fakes.AsyncGate()

    async def controlled_sleep(_delay: float) -> None:
        await sleep_gate.block()

    monkeypatch.setattr(
        chat_module,
        "asyncio",
        types.SimpleNamespace(sleep=controlled_sleep),
    )
    result = asyncio.create_task(
        _dispatch(
            agent,
            connection,
            ChatMessageType.TOOL_RESULT,
            toolCallId="call-1",
            output="arrived early",
        )
    )
    try:
        await asyncio.wait_for(sleep_gate.wait_until_blocked(), timeout=1)
        await agent._persist_messages([_message()])
        connection.sent.clear()
        sleep_gate.release()
        await asyncio.wait_for(result, timeout=1)
    finally:
        sleep_gate.release()
        if not result.done():
            result.cancel()
        await asyncio.gather(result, return_exceptions=True)

    assert agent.messages[0]["parts"][0]["output"] == "arrived early"
    assert len(_updates(connection)) == 1


@pytest.mark.asyncio
async def test_invalid_tool_answer_frames_are_consumed_without_mutation(
    monkeypatch,
):
    monkeypatch.setattr(chat_module, "_TOOL_PART_ATTEMPTS", 1)

    class RecordingChat(chat_module.AIChatAgent):
        def __init__(self, ctx, env):
            super().__init__(ctx, env)
            self.user_messages: list[str] = []

        async def on_message(self, connection, message):
            self.user_messages.append(message)

    agent = cast(RecordingChat, fakes.build_chat_agent(cls=RecordingChat))
    original = _message()
    await agent._persist_messages([copy.deepcopy(original)])
    connection = fakes.FakeConnection()
    _connect(agent, connection)
    invalid = [
        {"type": ChatMessageType.TOOL_RESULT},
        {"type": ChatMessageType.TOOL_RESULT, "toolCallId": None},
        {"type": ChatMessageType.TOOL_RESULT, "toolCallId": 1},
        {"type": ChatMessageType.TOOL_RESULT, "toolCallId": True},
        {"type": ChatMessageType.TOOL_RESULT, "toolCallId": ""},
        {"type": ChatMessageType.TOOL_RESULT, "toolCallId": "unknown"},
        {
            "type": ChatMessageType.TOOL_APPROVAL,
            "toolCallId": "unknown",
            "approved": True,
        },
        {
            "type": ChatMessageType.TOOL_APPROVAL,
            "toolCallId": "call-1",
            "approved": "yes",
        },
    ]

    for frame in invalid:
        await agent._dispatch_message(cast(Connection, connection), json.dumps(frame))

    assert agent.messages == [original]
    assert _updates(connection) == []
    assert agent.user_messages == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "frame_type",
    [ChatMessageType.TOOL_RESULT, ChatMessageType.TOOL_APPROVAL],
)
async def test_auto_continue_is_currently_ignored_pending_stage_five(
    frame_type, wait_until
):
    outcomes = []
    for auto_continue in (None, False, True):
        provider_calls = 0

        def provider(_options):
            nonlocal provider_calls
            provider_calls += 1
            return "unexpected continuation"

        agent = fakes.build_chat_agent(provider)
        message = _message(
            state=(
                "approval-requested"
                if frame_type == ChatMessageType.TOOL_APPROVAL
                else "input-available"
            ),
            approval={} if frame_type == ChatMessageType.TOOL_APPROVAL else None,
        )
        await agent._persist_messages([message])
        connection = fakes.FakeConnection()
        _connect(agent, connection)
        detail = {"toolCallId": "call-1"}
        if frame_type == ChatMessageType.TOOL_APPROVAL:
            detail["approved"] = True
        else:
            detail["output"] = "result"
        if auto_continue is not None:
            detail["autoContinue"] = auto_continue

        await _dispatch(agent, connection, frame_type, **detail)
        await asyncio.sleep(0)
        outcomes.append(
            (
                copy.deepcopy(agent.messages),
                copy.deepcopy(connection.frames),
                provider_calls,
            )
        )

    assert outcomes[0] == outcomes[1] == outcomes[2]
    messages, frames, provider_calls = outcomes[0]
    assert len(messages) == 1
    assert len(frames) == 1
    assert frames[0]["type"] == ChatMessageType.MESSAGE_UPDATED
    assert provider_calls == 0
    assert wait_until.records == []
