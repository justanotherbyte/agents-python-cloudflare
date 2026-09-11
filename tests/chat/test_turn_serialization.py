from __future__ import annotations

import asyncio
import json

import fakes
import pytest

from agents import AIChatAgent


def _request(request_id: str, text: str) -> dict:
    body = {
        "messages": [
            {
                "id": f"message-{request_id}",
                "role": "user",
                "parts": [{"type": "text", "text": text}],
            }
        ]
    }
    return {
        "id": request_id,
        "init": {"method": "POST", "body": json.dumps(body)},
    }


def _assistant_texts(agent: AIChatAgent) -> list[str]:
    return [
        part["text"]
        for message in agent.messages
        if message.get("role") == "assistant"
        for part in message.get("parts", [])
        if part.get("type") == "text"
    ]


@pytest.mark.asyncio
async def test_addressed_turn_blocks_headless_agent_tool_turn():
    class SerialAgent(AIChatAgent):
        addressed_started = asyncio.Event()
        release_addressed = asyncio.Event()
        headless_started = asyncio.Event()

        async def on_chat_message(self, options):
            if "agentToolInput" in options.body:
                type(self).headless_started.set()
                return "headless"
            type(self).addressed_started.set()
            await type(self).release_addressed.wait()
            return "addressed"

    agent = fakes.build_chat_agent(cls=SerialAgent)
    connection = fakes.FakeConnection()
    agent._connections[connection.id] = connection

    addressed = asyncio.create_task(
        agent._handle_use_chat_request(connection, _request("addressed", "one"))
    )
    await SerialAgent.addressed_started.wait()
    addressed_message = agent._streaming_message
    headless = asyncio.create_task(agent._cf_start_agent_tool_run('"two"', "headless"))
    await asyncio.sleep(0)

    assert not SerialAgent.headless_started.is_set()
    assert addressed_message is not None
    assert agent._streaming_message is addressed_message

    SerialAgent.release_addressed.set()
    _, inspection_json = await asyncio.gather(addressed, headless)
    inspection = json.loads(inspection_json)

    assert inspection["status"] == "completed"
    assert inspection["output"] == "headless"
    assert _assistant_texts(agent) == ["addressed", "headless"]
    assert [message["role"] for message in agent.messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    streams = agent.sql(
        "SELECT id, request_id FROM cf_ai_chat_stream_metadata "
        "WHERE request_id IN ('addressed', 'agent-tool-headless') "
        "ORDER BY request_id"
    )
    assert [row["request_id"] for row in streams] == [
        "addressed",
        "agent-tool-headless",
    ]
    assert len({row["id"] for row in streams}) == 2
    for stream in streams:
        indexes = [
            row["chunk_index"]
            for row in agent.sql(
                "SELECT chunk_index FROM cf_ai_chat_stream_chunks "
                "WHERE stream_id = ? ORDER BY chunk_index",
                stream["id"],
            )
        ]
        assert indexes == list(range(len(indexes)))


@pytest.mark.asyncio
async def test_headless_turn_blocks_addressed_turn():
    class SerialAgent(AIChatAgent):
        headless_started = asyncio.Event()
        release_headless = asyncio.Event()
        addressed_started = asyncio.Event()

        async def on_chat_message(self, options):
            if "agentToolInput" in options.body:
                type(self).headless_started.set()
                await type(self).release_headless.wait()
                return "headless"
            type(self).addressed_started.set()
            return "addressed"

    agent = fakes.build_chat_agent(cls=SerialAgent)
    connection = fakes.FakeConnection()
    agent._connections[connection.id] = connection

    headless = asyncio.create_task(agent._cf_start_agent_tool_run('"one"', "headless"))
    await SerialAgent.headless_started.wait()
    addressed = asyncio.create_task(
        agent._handle_use_chat_request(connection, _request("addressed", "two"))
    )
    await asyncio.sleep(0)

    assert not SerialAgent.addressed_started.is_set()

    SerialAgent.release_headless.set()
    await asyncio.gather(headless, addressed)

    assert SerialAgent.addressed_started.is_set()
    assert agent._streaming_message is None


@pytest.mark.asyncio
async def test_clear_aborts_queued_headless_turn_without_running_provider():
    class SerialAgent(AIChatAgent):
        addressed_started = asyncio.Event()
        release_addressed = asyncio.Event()
        headless_calls = 0

        async def on_chat_message(self, options):
            if "agentToolInput" in options.body:
                type(self).headless_calls += 1
                return "headless"
            type(self).addressed_started.set()
            await type(self).release_addressed.wait()
            return "addressed"

    agent = fakes.build_chat_agent(cls=SerialAgent)
    await agent._ensure_initialized()
    connection = fakes.FakeConnection()
    agent._connections[connection.id] = connection

    addressed = asyncio.create_task(
        agent._handle_use_chat_request(connection, _request("addressed", "one"))
    )
    await SerialAgent.addressed_started.wait()
    headless = asyncio.create_task(agent._cf_start_agent_tool_run('"two"', "headless"))
    await asyncio.sleep(0)

    await agent._handle_chat_clear(connection)
    SerialAgent.release_addressed.set()
    _, inspection_json = await asyncio.gather(addressed, headless)
    inspection = json.loads(inspection_json)

    assert SerialAgent.headless_calls == 0
    assert inspection["status"] == "aborted"
    assert agent.messages == []


@pytest.mark.asyncio
async def test_clear_aborts_queued_addressed_turn_without_running_provider():
    class SerialAgent(AIChatAgent):
        headless_started = asyncio.Event()
        release_headless = asyncio.Event()
        addressed_calls = 0

        async def on_chat_message(self, options):
            if "agentToolInput" in options.body:
                type(self).headless_started.set()
                await type(self).release_headless.wait()
                return "headless"
            type(self).addressed_calls += 1
            return "addressed"

    agent = fakes.build_chat_agent(cls=SerialAgent)
    connection = fakes.FakeConnection()
    agent._connections[connection.id] = connection

    headless = asyncio.create_task(agent._cf_start_agent_tool_run('"one"', "headless"))
    await SerialAgent.headless_started.wait()
    addressed = asyncio.create_task(
        agent._handle_use_chat_request(connection, _request("addressed", "two"))
    )
    await asyncio.sleep(0)

    await agent._handle_chat_clear(connection)
    SerialAgent.release_headless.set()
    await asyncio.gather(headless, addressed)

    assert SerialAgent.addressed_calls == 0
    assert agent.messages == []
    assert connection.frames[-1]["done"] is True
