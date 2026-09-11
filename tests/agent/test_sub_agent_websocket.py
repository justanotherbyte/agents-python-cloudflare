from __future__ import annotations

import json
import types

import fakes
import pytest
from workers import Request, Response

import agents.lifecycle.websockets as websockets_module
from agents import AIChatAgent, Agent
from agents.core.error import HookError
from agents.core.subagent_relay import BufferedRelayConnection, RelayLimitError


class ChildAgent(Agent):
    async def on_message(self, connection, message):
        connection.send(f"child:{message}")


class GrandChild(Agent):
    async def on_message(self, connection, message):
        connection.send(f"grandchild:{message}")


class ChildChat(AIChatAgent):
    async def on_chat_message(self, options):
        return "from child"


class DirectStub:
    def __init__(self, agent):
        self.agent = agent

    async def _cf_ws_event(self, payload):
        return await self.agent._cf_ws_event(payload)


def _request(path):
    return Request(
        f"https://example.com{path}",
        headers={"connection": "upgrade", "upgrade": "websocket"},
    )


def _install_pair(monkeypatch, socket):
    client = object()
    pair = types.SimpleNamespace(object_values=lambda: (client, socket))
    monkeypatch.setattr(websockets_module.WebSocketPair, "new", lambda: pair)
    return client


def _wire_root(root, child):
    root.ctx.exports = {"ChildAgent": object()}

    async def resolve(class_name, name):
        assert (class_name, name) == ("ChildAgent", "leaf")
        return DirectStub(child)

    root._resolve_sub_agent = resolve


@pytest.mark.asyncio
async def test_root_owns_socket_and_relays_child_lifecycle(monkeypatch):
    root = fakes.build_agent()
    child = fakes.build_agent(
        ChildAgent,
        name="cf-agents:v2:leaf:0123456789abcdef",
    )
    _wire_root(root, child)
    socket = fakes.FakeSocket()
    client = _install_pair(monkeypatch, socket)

    response = await root.fetch(_request("/agents/agent/root/sub/child-agent/leaf"))

    assert response.status == 101
    assert response.web_socket is client
    assert root.get_connections() == []
    frames = [json.loads(frame) for frame in socket.sent]
    assert frames[0]["type"] == "cf_agent_identity"
    assert frames[0]["name"] == "leaf"
    assert child._connections == {}

    await root.webSocketMessage(socket, "hello")
    assert socket.sent[-1] == "child:hello"
    assert child._connections == {}

    await root.webSocketClose(socket, 1000, "done", True)
    assert root._connections == {}
    assert child._connections == {}


@pytest.mark.asyncio
async def test_child_connection_tags_are_canonical_and_persist_on_root(monkeypatch):
    class TaggedChild(Agent):
        def __init__(self, ctx, env):
            super().__init__(ctx, env)
            self.connected_tags = None

        def get_connection_tags(self, connection, ctx):
            return ["room:child", connection.id, "room:child"]

        async def on_connect(self, connection, ctx):
            self.connected_tags = connection.tags

    root = fakes.build_agent()
    child = fakes.build_agent(
        TaggedChild,
        name="cf-agents:v2:leaf:0123456789abcdef",
    )
    _wire_root(root, child)
    socket = fakes.FakeSocket()
    _install_pair(monkeypatch, socket)

    await root.fetch(
        _request("/agents/agent/root/sub/child-agent/leaf?_pk=child-connection")
    )

    assert child.connected_tags == [
        "child-connection",
        "room:child",
        "room:child",
    ]
    attachment = socket.deserializeAttachment().to_py()
    assert attachment["__pk"]["tags"] == child.connected_tags
    assert attachment["target"]["url"].endswith(
        "/agents/agent/root/sub/child-agent/leaf?_pk=child-connection"
    )


@pytest.mark.asyncio
async def test_child_connection_policy_survives_root_rehydration(monkeypatch):
    class PolicyChild(Agent):
        def should_connection_be_readonly(self, connection, ctx):
            return True

        def should_send_protocol_messages(self, connection, ctx):
            return False

        async def on_message(self, connection, message):
            if message == "clear":
                self.set_connection_readonly(connection, False)
            elif message == "mutate":
                self.set_state({"mutated": True})
            connection.send_json(
                {
                    "readonly": self.is_connection_readonly(connection),
                    "protocol": self.is_connection_protocol_enabled(connection),
                    "state": connection.state,
                }
            )

    root = fakes.build_agent()
    child = fakes.build_agent(
        PolicyChild,
        name="cf-agents:v2:leaf:0123456789abcdef",
    )
    _wire_root(root, child)
    socket = fakes.FakeSocket()
    _install_pair(monkeypatch, socket)
    await root.fetch(_request("/agents/agent/root/sub/child-agent/leaf"))

    assert socket.sent == []
    assert socket.deserializeAttachment().to_py()["__user"] == {
        "_cf_readonly": True,
        "_cf_no_protocol": True,
    }

    reincarnated = fakes.build_agent(websockets=[socket], conn=root.ctx.conn)
    _wire_root(reincarnated, child)
    await reincarnated.webSocketMessage(socket, "status")

    assert json.loads(socket.sent[-1]) == {
        "readonly": True,
        "protocol": False,
        "state": None,
    }

    await reincarnated.webSocketMessage(socket, "clear")
    assert socket.deserializeAttachment().to_py()["__user"] == {"_cf_no_protocol": True}

    restored_root = fakes.build_agent(websockets=[socket], conn=root.ctx.conn)
    _wire_root(restored_root, child)
    await restored_root.webSocketMessage(socket, "mutate")

    assert child.state == {"mutated": True}
    assert json.loads(socket.sent[-1]) == {
        "readonly": False,
        "protocol": False,
        "state": None,
    }


@pytest.mark.asyncio
async def test_relay_target_survives_root_rehydration(monkeypatch):
    first = fakes.build_agent()
    child = fakes.build_agent(
        ChildAgent,
        name="cf-agents:v2:leaf:0123456789abcdef",
    )
    _wire_root(first, child)
    socket = fakes.FakeSocket()
    _install_pair(monkeypatch, socket)
    await first.fetch(_request("/agents/agent/root/sub/child-agent/leaf"))

    reincarnated = fakes.build_agent(websockets=[socket], conn=first.ctx.conn)
    _wire_root(reincarnated, child)
    await reincarnated.webSocketMessage(socket, "again")

    assert socket.sent[-1] == "child:again"
    assert reincarnated.get_connections() == []


@pytest.mark.asyncio
async def test_gate_rejects_before_socket_acceptance(monkeypatch):
    class Rejecting(Agent):
        async def on_before_sub_agent(self, request, child):
            return Response("forbidden", status=403)

    root = fakes.build_agent(Rejecting)
    root.ctx.exports = {"ChildAgent": object()}
    socket = fakes.FakeSocket()
    _install_pair(monkeypatch, socket)

    response = await root.fetch(_request("/agents/rejecting/root/sub/child-agent/leaf"))

    assert response.status == 403
    assert root.ctx.accepted_websockets == []


@pytest.mark.asyncio
async def test_nested_relay_runs_each_gate_once(monkeypatch):
    gates = []

    class GatedChild(ChildAgent):
        async def on_before_sub_agent(self, request, child):
            gates.append((child["class_name"], child["name"]))

    root = fakes.build_agent()
    child = fakes.build_agent(
        GatedChild,
        name="cf-agents:v2:leaf:0123456789abcdef",
    )
    grandchild = fakes.build_agent(
        GrandChild,
        name="cf-agents:v2:deep:0123456789abcdef",
    )
    root.ctx.exports = {"ChildAgent": object()}
    child.ctx.exports = {"GrandChild": object()}

    async def root_resolve(class_name, name):
        return DirectStub(child)

    async def child_resolve(class_name, name):
        assert (class_name, name) == ("GrandChild", "deep")
        return DirectStub(grandchild)

    root._resolve_sub_agent = root_resolve
    child._resolve_sub_agent = child_resolve
    socket = fakes.FakeSocket()
    _install_pair(monkeypatch, socket)

    await root.fetch(
        _request("/agents/agent/root/sub/child-agent/leaf/sub/grand-child/deep")
    )
    await root.webSocketMessage(socket, "hello")

    assert gates == [("GrandChild", "deep")]
    assert socket.sent[-1] == "grandchild:hello"


@pytest.mark.asyncio
async def test_relay_failure_closes_the_socket(monkeypatch):
    class FailingChild(Agent):
        async def on_connect(self, connection, ctx):
            raise RuntimeError("nope")

        async def on_error(self, error, connection=None):
            pass

    root = fakes.build_agent()
    child = fakes.build_agent(
        FailingChild,
        name="cf-agents:v2:leaf:0123456789abcdef",
    )
    _wire_root(root, child)
    socket = fakes.FakeSocket()
    _install_pair(monkeypatch, socket)

    with pytest.raises(HookError):
        await root.fetch(_request("/agents/agent/root/sub/child-agent/leaf"))

    assert socket.closed is True


@pytest.mark.asyncio
@pytest.mark.parametrize("event", ["message", "close", "error"])
async def test_relay_socket_failures_report_only_to_child(monkeypatch, event):
    class RecordingRoot(Agent):
        def __init__(self, ctx, env):
            super().__init__(ctx, env)
            self.errors = []

        async def on_error(self, error, connection=None):
            self.errors.append(error)

    class FailingChild(Agent):
        def __init__(self, ctx, env):
            super().__init__(ctx, env)
            self.errors = []
            self.connected_id = None

        async def on_connect(self, connection, ctx):
            self.connected_id = connection.id

        async def on_message(self, connection, message):
            if event == "message":
                raise RuntimeError("facet message failed")

        async def on_close(self, connection, code, reason, was_clean):
            if event == "close":
                raise RuntimeError("facet close failed")

        async def on_error(self, error, connection=None):
            self.errors.append((error, None if connection is None else connection.id))

    root = fakes.build_agent(RecordingRoot)
    child = fakes.build_agent(
        FailingChild,
        name="cf-agents:v2:leaf:0123456789abcdef",
    )
    _wire_root(root, child)
    socket = fakes.FakeSocket()
    _install_pair(monkeypatch, socket)
    await root.fetch(_request("/agents/agent/root/sub/child-agent/leaf"))

    if event == "message":
        await root.webSocketMessage(socket, "hello")
    elif event == "close":
        await root.webSocketClose(socket, 1001, "away", False)
    else:
        await root.webSocketError(socket, RuntimeError("facet socket failed"))

    assert root.errors == []
    assert len(child.errors) == 1
    error, connection_id = child.errors[0]
    if event == "error":
        assert not isinstance(error, HookError)
        assert str(error) == "facet socket failed"
    else:
        assert isinstance(error, HookError)
        assert error.hook == f"on_{event}"
        assert str(error.original) == f"facet {event} failed"
    assert connection_id == child.connected_id
    if event in {"message", "error"}:
        assert socket.closed is False


@pytest.mark.asyncio
async def test_chat_turn_is_buffered_and_delivered_from_child(monkeypatch):
    root = fakes.build_agent()
    child = fakes.build_chat_agent(
        cls=ChildChat,
        name="cf-agents:v2:chat:0123456789abcdef",
    )
    root.ctx.exports = {"ChildChat": object()}

    async def resolve(class_name, name):
        assert (class_name, name) == ("ChildChat", "chat")
        return DirectStub(child)

    root._resolve_sub_agent = resolve
    socket = fakes.FakeSocket()
    _install_pair(monkeypatch, socket)
    await root.fetch(_request("/agents/agent/root/sub/child-chat/chat"))

    request = {
        "type": "cf_agent_use_chat_request",
        "id": "req-1",
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
    await root.webSocketMessage(socket, json.dumps(request))

    frames = [json.loads(frame) for frame in socket.sent if frame.startswith("{")]
    responses = [
        frame for frame in frames if frame.get("type") == "cf_agent_use_chat_response"
    ]
    assert responses[-1]["done"] is True
    assistant = next(m for m in child.messages if m.get("role") == "assistant")
    assert assistant["parts"][0]["text"] == "from child"


def test_buffered_relay_enforces_frame_limit():
    connection = BufferedRelayConnection(
        "c1", state={}, tags=[], max_frames=1, max_bytes=1_000
    )
    connection.send("one")

    with pytest.raises(RelayLimitError, match="frame limit"):
        connection.send("two")
