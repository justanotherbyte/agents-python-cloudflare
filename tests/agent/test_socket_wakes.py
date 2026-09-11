from __future__ import annotations

from typing import Any, TypeVar, cast

import fakes
import pytest

from agents import Agent
from agents.core.error import HookError

AgentT = TypeVar("AgentT", bound=Agent)


def _attachment(connection_id: str = "connection-1") -> dict[str, Any]:
    return {"id": connection_id, "state": {}, "tags": []}


def _agent(
    cls: type[AgentT],
    *,
    name: str = "test-agent",
    websockets: list[Any] | None = None,
) -> AgentT:
    return cast(AgentT, fakes.build_agent(cls=cls, name=name, websockets=websockets))


class RecordingAgent(Agent):
    def __init__(self, ctx, env):
        super().__init__(ctx, env)
        self.errors: list[tuple[BaseException, str | None]] = []

    async def on_error(self, error, connection=None):
        connection_id = None if connection is None else connection.id
        self.errors.append((error, connection_id))
        raise error


@pytest.mark.asyncio
async def test_message_hook_error_is_reported_once_and_swallowed_at_wake_boundary():
    original = ValueError("message failed")

    class FailingMessage(RecordingAgent):
        async def on_message(self, connection, message):
            raise original

    socket: Any = fakes.FakeSocket(_attachment())
    agent = _agent(FailingMessage, websockets=[socket])

    await agent.webSocketMessage(socket, "hello")

    assert len(agent.errors) == 1
    error, connection_id = agent.errors[0]
    assert isinstance(error, HookError)
    assert error.hook == "on_message"
    assert error.original is original
    assert connection_id == "connection-1"


@pytest.mark.asyncio
async def test_close_hook_error_is_reported_once_and_connection_is_removed():
    original = ValueError("close failed")

    class FailingClose(RecordingAgent):
        async def on_close(self, connection, code, reason, was_clean):
            raise original

    socket: Any = fakes.FakeSocket(_attachment())
    agent = _agent(FailingClose, websockets=[socket])

    socket.closed = True
    await agent.webSocketClose(socket, 1001, "away", True)

    assert len(agent.errors) == 1
    error, connection_id = agent.errors[0]
    assert isinstance(error, HookError)
    assert error.hook == "on_close"
    assert error.original is original
    assert connection_id == "connection-1"
    assert agent.get_connections() == []


@pytest.mark.asyncio
async def test_raw_socket_error_keeps_identity_and_is_swallowed_after_reporting():
    socket: Any = fakes.FakeSocket(_attachment())
    agent = _agent(RecordingAgent, websockets=[socket])
    original = RuntimeError("socket failed")

    await agent.webSocketError(socket, original)

    assert agent.errors == [(original, "connection-1")]


@pytest.mark.asyncio
@pytest.mark.parametrize("event", ["message", "close", "error"])
async def test_startup_failure_on_socket_wake_is_not_reported_twice(event: str):
    original = RuntimeError("startup failed")

    class FailingStart(RecordingAgent):
        def __init__(self, ctx, env):
            super().__init__(ctx, env)
            self.messages: list[str] = []

        async def on_start(self):
            raise original

        async def on_message(self, connection, message):
            self.messages.append(message)

    socket: Any = fakes.FakeSocket(_attachment())
    agent = _agent(FailingStart, websockets=[socket])

    if event == "message":
        await agent.webSocketMessage(socket, "not delivered")
    elif event == "close":
        await agent.webSocketClose(socket, 1001, "not delivered", False)
    else:
        await agent.webSocketError(socket, RuntimeError("must not be reported"))

    assert len(agent.errors) == 1
    error, connection_id = agent.errors[0]
    assert isinstance(error, HookError)
    assert error.hook == "on_start"
    assert error.original is original
    assert connection_id is None
    assert agent.messages == []
    if event == "close":
        assert socket.closed is True
        assert agent.get_connections() == []


@pytest.mark.asyncio
async def test_socket_routing_failure_is_reported_once_and_swallowed():
    original = RuntimeError("routing failed")

    class FailingRoute(RecordingAgent):
        async def _forward_websocket_relay(self, payload):
            raise original

    socket: Any = fakes.FakeSocket(
        {
            **_attachment(),
            "target": {
                "url": "https://example.com/agents/agent/root/sub/child/leaf",
                "headers": {},
            },
        }
    )
    agent = _agent(FailingRoute, websockets=[socket])

    await agent.webSocketMessage(socket, "hello")

    assert agent.errors == [(original, "connection-1")]


@pytest.mark.asyncio
@pytest.mark.parametrize("attachment", [None, {"id": 42}])
async def test_message_and_close_ignore_sockets_without_a_string_id(attachment):
    class HookRecorder(RecordingAgent):
        def __init__(self, ctx, env):
            super().__init__(ctx, env)
            self.messages: list[str] = []
            self.closes: list[tuple[int, str, bool]] = []

        async def on_message(self, connection, message):
            self.messages.append(message)

        async def on_close(self, connection, code, reason, was_clean):
            self.closes.append((code, reason, was_clean))

    socket: Any = fakes.FakeSocket(attachment)
    agent = _agent(HookRecorder)

    await agent.webSocketMessage(socket, "ignored")
    await agent.webSocketClose(socket, 1000, "ignored", True)

    assert agent.messages == []
    assert agent.closes == []
    assert agent.errors == []
