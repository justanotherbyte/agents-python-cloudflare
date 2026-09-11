from __future__ import annotations

import asyncio
import json
import types

import fakes
import pytest
from workers import Request

import agents.lifecycle.websockets as websockets_module
from agents import Agent, rpc_callable
from agents.core.protocol import MessageType, mcp_servers_frame, state_frame


def _request(query: str = "") -> Request:
    return Request(
        f"https://example.com/agents/policy-agent/name{query}",
        headers={"connection": "upgrade", "upgrade": "websocket"},
    )


def _install_pair(monkeypatch, socket: object) -> None:
    pair = types.SimpleNamespace(object_values=lambda: (object(), socket))
    monkeypatch.setattr(websockets_module.WebSocketPair, "new", lambda: pair)


def _json_frames(socket: fakes.FakeSocket) -> list[dict[str, object]]:
    frames = []
    for frame in socket.sent:
        try:
            value = json.loads(frame)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            frames.append(value)
    return frames


class PolicyAgent(Agent):
    def __init__(self, ctx, env):
        super().__init__(ctx, env)
        self.policy_order: list[tuple[str, bool]] = []
        self._agent_tool_runs.replay = lambda connection: connection.send("replay")

    def should_connection_be_readonly(self, connection, ctx):
        readonly = "readonly=true" in ctx.request.url
        self.policy_order.append(("readonly", readonly))
        return readonly

    def should_send_protocol_messages(self, connection, ctx):
        self.policy_order.append(("protocol", self.is_connection_readonly(connection)))
        return "protocol=false" not in ctx.request.url

    async def on_connect(self, connection, ctx):
        connection.send("user")

    @rpc_callable()
    def read_state(self):
        return self.state

    @rpc_callable()
    def mutate_state(self):
        self.set_state({"count": 1})
        return self.state


@pytest.mark.asyncio
async def test_handshake_resolves_policy_before_protocol_and_user_hooks(monkeypatch):
    agent = fakes.build_agent(cls=PolicyAgent)
    socket = fakes.FakeSocket()
    _install_pair(monkeypatch, socket)

    await agent.fetch(_request("?_pk=policy&readonly=true"))

    assert [frame["type"] for frame in _json_frames(socket)[:3]] == [
        MessageType.CF_AGENT_IDENTITY,
        MessageType.CF_AGENT_STATE,
        MessageType.CF_AGENT_MCP_SERVERS,
    ]
    assert socket.sent[-2:] == ["replay", "user"]
    assert agent.policy_order == [("readonly", True), ("protocol", True)]

    connection = agent.get_connection("policy")
    assert connection is not None
    assert connection.state is None
    assert agent.is_connection_readonly(connection) is True
    assert agent.is_connection_protocol_enabled(connection) is True
    assert socket.deserializeAttachment().to_py()["__user"] == {"_cf_readonly": True}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query",
    ["?_pk=lazy", "?_pk=lazy&readonly=true"],
)
async def test_lazy_state_initialization_sends_connecting_socket_once(
    monkeypatch,
    query,
):
    class LazyStateAgent(PolicyAgent):
        def __init__(self, ctx, env):
            super().__init__(ctx, env)
            self.state_reads = 0

        @property
        def state(self):
            self.state_reads += 1
            if self.state_reads == 1:
                self.set_state({"lazy": True})
            return super().state

    agent = fakes.build_agent(cls=LazyStateAgent)
    socket = fakes.FakeSocket()
    _install_pair(monkeypatch, socket)

    await agent.fetch(_request(query))

    state_frames = [
        frame
        for frame in _json_frames(socket)
        if frame.get("type") == MessageType.CF_AGENT_STATE
    ]
    assert state_frames == [state_frame({"lazy": True})]


@pytest.mark.asyncio
async def test_suppression_skips_only_handshake_protocol_frames(monkeypatch):
    agent = fakes.build_agent(cls=PolicyAgent)
    socket = fakes.FakeSocket()
    _install_pair(monkeypatch, socket)

    await agent.fetch(_request("?_pk=suppressed&readonly=true&protocol=false"))

    assert socket.sent == ["replay", "user"]
    connection = agent.get_connection("suppressed")
    assert connection is not None
    assert connection.state is None
    assert agent.is_connection_readonly(connection) is True
    assert agent.is_connection_protocol_enabled(connection) is False

    await agent.webSocketMessage(
        socket,
        json.dumps({"type": "rpc", "id": "read", "method": "read_state", "args": []}),
    )

    assert _json_frames(socket)[-1] == {
        "type": "rpc",
        "id": "read",
        "success": True,
        "done": True,
        "result": {},
    }


@pytest.mark.asyncio
async def test_readonly_blocks_client_state_and_mutating_rpc_only(monkeypatch):
    agent = fakes.build_agent(cls=PolicyAgent)
    socket = fakes.FakeSocket()
    _install_pair(monkeypatch, socket)
    await agent.fetch(_request("?_pk=readonly&readonly=true"))
    socket.sent.clear()

    await agent.webSocketMessage(
        socket,
        json.dumps({"type": MessageType.CF_AGENT_STATE, "state": {"count": 9}}),
    )
    await agent.webSocketMessage(
        socket,
        json.dumps(
            {"type": "rpc", "id": "write", "method": "mutate_state", "args": []}
        ),
    )
    await agent.webSocketMessage(
        socket,
        json.dumps({"type": "rpc", "id": "read", "method": "read_state", "args": []}),
    )

    assert _json_frames(socket) == [
        {
            "type": MessageType.CF_AGENT_STATE_ERROR,
            "error": "Connection is readonly",
        },
        {
            "type": "rpc",
            "id": "write",
            "success": False,
            "error": "Connection is readonly",
        },
        {
            "type": "rpc",
            "id": "read",
            "success": True,
            "done": True,
            "result": {},
        },
    ]
    assert agent.state == {}


@pytest.mark.asyncio
async def test_connection_flags_are_hidden_preserved_and_rehydrated(monkeypatch):
    agent = fakes.build_agent(cls=PolicyAgent)
    socket = fakes.FakeSocket()
    _install_pair(monkeypatch, socket)
    await agent.fetch(_request("?_pk=policy&readonly=true&protocol=false"))
    connection = agent.get_connection("policy")
    assert connection is not None

    connection.set_state({"device": "sensor-1"})
    connection.set_state(lambda state: {**state, "zone": "a"})

    assert connection.state == {"device": "sensor-1", "zone": "a"}
    assert agent.is_connection_readonly(connection) is True
    assert agent.is_connection_protocol_enabled(connection) is False
    raw = socket.deserializeAttachment().to_py()["__user"]
    assert raw == {
        "device": "sensor-1",
        "zone": "a",
        "_cf_readonly": True,
        "_cf_no_protocol": True,
    }

    agent.set_connection_readonly(connection, False)
    reincarnated = fakes.build_agent(
        cls=PolicyAgent,
        websockets=[socket],
        conn=agent.ctx.conn,
    )
    restored = reincarnated.get_connection("policy")
    assert restored is not None
    assert restored.state == {"device": "sensor-1", "zone": "a"}
    assert reincarnated.is_connection_readonly(restored) is False
    assert reincarnated.is_connection_protocol_enabled(restored) is False


def test_internal_flags_reject_non_object_state_without_losing_data():
    agent = fakes.build_agent(cls=PolicyAgent)
    socket = fakes.FakeSocket()
    connection = agent._websockets.wrap(
        socket,
        agent._websockets.new_attachment(
            "https://example.com/agents/policy-agent/name",
            "policy",
        ),
    )
    connection.set_state(["preserved"])

    with pytest.raises(TypeError, match="must be an object or null"):
        agent.set_connection_readonly(connection)
    assert connection.state == ["preserved"]

    connection.set_state(None)
    agent.set_connection_readonly(connection)
    with pytest.raises(TypeError, match="must be an object or null"):
        connection.set_state(["rejected"])

    assert connection.state is None
    assert agent.is_connection_readonly(connection) is True


@pytest.mark.asyncio
async def test_callback_context_is_isolated_between_concurrent_connections(monkeypatch):
    class ConcurrentPolicyAgent(PolicyAgent):
        def __init__(self, ctx, env):
            super().__init__(ctx, env)
            self.started = 0
            self.all_started = asyncio.Event()
            self.release = asyncio.Event()
            self.other_agent = None
            self.cross_agent_writes = []

        @rpc_callable()
        async def gated_mutation(self, label):
            self.started += 1
            if self.started == 2:
                self.all_started.set()
            await self.release.wait()
            self.other_agent.set_state({"cross_agent": label})
            self.cross_agent_writes.append(label)
            self.set_state({"writer": label})
            return label

    agent = fakes.build_agent(cls=ConcurrentPolicyAgent)
    agent.other_agent = fakes.build_agent(cls=PolicyAgent)

    async def connect(query: str):
        socket = fakes.FakeSocket()
        _install_pair(monkeypatch, socket)
        await agent.fetch(_request(query))
        socket.sent.clear()
        return socket

    readonly = await connect("?_pk=readonly&readonly=true")
    writable = await connect("?_pk=writable")
    readonly_turn = asyncio.create_task(
        agent.webSocketMessage(
            readonly,
            json.dumps(
                {
                    "type": "rpc",
                    "id": "readonly",
                    "method": "gated_mutation",
                    "args": ["readonly"],
                }
            ),
        )
    )
    writable_turn = asyncio.create_task(
        agent.webSocketMessage(
            writable,
            json.dumps(
                {
                    "type": "rpc",
                    "id": "writable",
                    "method": "gated_mutation",
                    "args": ["writable"],
                }
            ),
        )
    )
    await agent.all_started.wait()
    agent.release.set()
    await asyncio.gather(readonly_turn, writable_turn)

    readonly_response = next(
        frame for frame in _json_frames(readonly) if frame.get("id") == "readonly"
    )
    writable_response = next(
        frame for frame in _json_frames(writable) if frame.get("id") == "writable"
    )
    assert readonly_response == {
        "type": "rpc",
        "id": "readonly",
        "success": False,
        "error": "Connection is readonly",
    }
    assert writable_response == {
        "type": "rpc",
        "id": "writable",
        "success": True,
        "done": True,
        "result": "writable",
    }
    assert sorted(agent.cross_agent_writes) == ["readonly", "writable"]

    agent.set_state({"outside": True})
    assert agent.state == {"outside": True}


@pytest.mark.asyncio
async def test_protocol_broadcast_filters_by_physical_connection(monkeypatch):
    agent = fakes.build_agent(cls=PolicyAgent)

    async def connect(query: str):
        socket = fakes.FakeSocket()
        _install_pair(monkeypatch, socket)
        await agent.fetch(_request(query))
        connection = agent._websockets._connection_for_socket(socket)
        assert connection is not None
        socket.sent.clear()
        return socket, connection

    source_socket, source = await connect("?_pk=shared")
    twin_socket, _ = await connect("?_pk=shared")
    suppressed_socket, _ = await connect("?_pk=suppressed&protocol=false")
    readonly_socket, _ = await connect("?_pk=readonly&readonly=true")

    agent._handle_state_update(source, {"state": {"count": 1}})

    assert source_socket.sent == []
    assert _json_frames(twin_socket) == [state_frame({"count": 1})]
    assert suppressed_socket.sent == []
    assert _json_frames(readonly_socket) == [state_frame({"count": 1})]

    for socket in (source_socket, twin_socket, suppressed_socket, readonly_socket):
        socket.sent.clear()
    agent._broadcast_mcp_servers()

    expected = [mcp_servers_frame(agent.get_mcp_servers())]
    assert _json_frames(source_socket) == expected
    assert _json_frames(twin_socket) == expected
    assert suppressed_socket.sent == []
    assert _json_frames(readonly_socket) == expected
