from __future__ import annotations

import asyncio
import types
from typing import Any

import fakes
import pytest
from workers import Request

import agents.lifecycle._runtime as lifecycle_module
import agents.lifecycle.websockets as websockets_module
from agents import Agent
from agents.lifecycle import Lifecycle, get_current_lifecycle_context
from agents.lifecycle.websockets import Connection, WebSockets


def _request(query: str = "") -> Request:
    return Request(
        f"https://example.com/agents/example/name{query}",
        headers={"connection": "upgrade", "upgrade": "websocket"},
    )


def _install_agent_pair(monkeypatch, socket: object) -> object:
    client = object()
    pair = types.SimpleNamespace(object_values=lambda: (client, socket))
    monkeypatch.setattr(websockets_module.WebSocketPair, "new", lambda: pair)
    return client


def _install_capability_pair(monkeypatch, socket: object) -> object:
    client = object()
    pair = types.SimpleNamespace(object_values=lambda: (client, socket))
    monkeypatch.setattr(websockets_module.WebSocketPair, "new", lambda: pair)
    return client


@pytest.mark.asyncio
async def test_plain_lifecycle_capability_owns_upgrade_and_wakes(monkeypatch):
    events: list[tuple[Any, ...]] = []
    scopes: list[tuple[object, object | None, object | None]] = []

    host = object()

    def record_scope() -> None:
        current = get_current_lifecycle_context()
        assert current is not None
        scopes.append((current.host, current.request, current.connection))

    async def on_connect(connection, context):
        record_scope()
        events.append(("connect", connection.id, context.request.url))

    async def on_message(connection, message):
        record_scope()
        events.append(("message", connection.id, message))

    async def on_close(connection, code, reason, was_clean):
        record_scope()
        events.append(("close", connection.id, code, reason, was_clean))

    ctx = fakes.FakeCtx()
    sockets = WebSockets(
        on_connect=on_connect,
        on_message=on_message,
        on_close=on_close,
        get_connection_tags=lambda connection, context: ["room:a"],
    )
    lifecycle = Lifecycle(ctx, host=host)
    lifecycle.use(sockets, fallback=True)
    socket = fakes.FakeSocket()
    client = _install_capability_pair(monkeypatch, socket)

    response = await lifecycle.websocket_upgrade(_request("?_pk=public-id"))

    assert response.status == 101
    assert response.web_socket is client
    assert ctx.getWebSockets("public-id") == [socket]
    assert ctx.getWebSockets("room:a") == [socket]
    assert sockets.get_connection("public-id") is not None

    assert await lifecycle.websocket_message(socket, "hello") is True
    assert await lifecycle.websocket_close(socket, 1000, "done", True) is True
    assert events == [
        ("connect", "public-id", _request("?_pk=public-id").url),
        ("message", "public-id", "hello"),
        ("close", "public-id", 1000, "done", True),
    ]
    assert [
        (seen_host, request is not None, connection.id)
        for seen_host, request, connection in scopes
    ] == [
        (host, True, "public-id"),
        (host, False, "public-id"),
        (host, False, "public-id"),
    ]
    assert sockets.get_connection("public-id") is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tags", "message"),
    [
        ([""], "must not be empty"),
        (["x" * 257], "must not exceed 256"),
        (["😀" * 129], "must not exceed 256"),
        ([str(index) for index in range(10)], "only have 10 tags"),
        (["valid", 42], "must be a string"),
    ],
)
async def test_invalid_tags_reject_before_native_accept(
    monkeypatch,
    tags,
    message,
):
    class InvalidTags(Agent):
        def get_connection_tags(self, connection, ctx):
            return tags

    server = fakes.build_agent(cls=InvalidTags)
    socket = fakes.FakeSocket()
    _install_agent_pair(monkeypatch, socket)

    with pytest.raises(ValueError, match=message):
        await server.fetch(_request())

    assert server.ctx.accepted_websockets == []
    assert server.get_connections() == []
    assert socket.closed is True


@pytest.mark.asyncio
async def test_tag_length_uses_javascript_utf16_units(monkeypatch):
    class TaggedServer(Agent):
        def get_connection_tags(self, connection, ctx):
            return ["😀" * 128]

    server = fakes.build_agent(cls=TaggedServer)
    socket = fakes.FakeSocket()
    _install_agent_pair(monkeypatch, socket)

    await server.fetch(_request())

    [connection] = server.get_connections()
    assert connection.tags == [connection.id, "😀" * 128]


@pytest.mark.asyncio
async def test_accept_uses_query_id_and_serializes_after_native_accept(monkeypatch):
    class RecordingContext(fakes.FakeCtx):
        def __init__(self):
            super().__init__()
            self.attachment_at_accept = object()
            self.tags_at_accept: tuple[str, ...] = ()

        def acceptWebSocket(self, websocket, tags=()):
            self.attachment_at_accept = websocket.deserializeAttachment()
            self.tags_at_accept = tuple(tags)
            super().acceptWebSocket(websocket, tags)

    class TaggedServer(Agent):
        def get_connection_tags(self, connection, ctx):
            assert self.get_connection(connection.id) is connection
            return ["room:a", connection.id, "room:a"]

    ctx = RecordingContext()
    server = TaggedServer(ctx, types.SimpleNamespace())
    socket = fakes.FakeSocket()
    _install_agent_pair(monkeypatch, socket)

    await server.fetch(_request("?_pk=chosen"))

    assert ctx.attachment_at_accept is None
    assert ctx.tags_at_accept == ("chosen", "room:a", "room:a")
    [connection] = server.get_connections()
    assert connection.id == "chosen"
    assert connection.tags == ["chosen", "room:a", "room:a"]


@pytest.mark.asyncio
async def test_accept_converts_native_tags_for_the_runtime(monkeypatch):
    converted = object()
    seen: list[object] = []

    class RecordingContext(fakes.FakeCtx):
        def acceptWebSocket(self, websocket, tags=()):
            seen.append(tags)

    monkeypatch.setattr(lifecycle_module, "to_js", lambda value: converted)
    ctx = RecordingContext()
    server = Agent(ctx, types.SimpleNamespace())
    socket = fakes.FakeSocket()
    _install_agent_pair(monkeypatch, socket)

    await server.fetch(_request())

    assert seen == [converted]


def test_constructor_does_not_evaluate_hook_descriptors():
    class DescriptorServer(Agent):
        @property
        def on_error(self):
            raise RuntimeError("on_error descriptor evaluated")

        @property
        def get_connection_tags(self):
            raise RuntimeError("tag descriptor evaluated")

        @property
        def _hide_relay_connections(self):
            raise RuntimeError("setting descriptor evaluated")

    DescriptorServer(fakes.FakeCtx(), types.SimpleNamespace())


def test_malformed_relay_target_is_not_adopted_as_a_root_connection():
    socket = fakes.FakeSocket(
        {
            "__pk": {"id": "connection-1", "tags": ["connection-1"]},
            "__user": None,
            "target": {"url": "https://example.com/sub", "headers": ["bad"]},
        }
    )
    sockets = WebSockets()

    assert sockets.adopt(socket) is None
    assert sockets.get_connections() == ()


def test_agent_rejects_relay_target_without_a_facet_route():
    socket = fakes.FakeSocket(
        {
            "__pk": {"id": "connection-1", "tags": ["connection-1"]},
            "__user": None,
            "target": {"url": "https://example.com/agents/agent/root", "headers": {}},
        }
    )
    agent = fakes.build_agent(websockets=[socket])

    assert agent.get_connections() == []
    assert agent._connections == {}


def test_tag_lookup_uses_native_index_before_full_hydration():
    calls: list[str | None] = []
    legacy = fakes.FakeSocket(
        {
            "__pk": {
                "id": "legacy",
                "tags": ["legacy", "room:a"],
            },
            "__user": None,
        }
    )
    ctx = fakes.FakeCtx(websockets=[legacy])
    socket = fakes.FakeSocket(
        {
            "__pk": {
                "id": "connection-1",
                "tags": ["connection-1", "room:a"],
            },
            "__user": None,
        }
    )
    ctx.acceptWebSocket(socket, ["connection-1", "room:a"])

    def source(tag):
        calls.append(tag)
        return ctx.getWebSockets(tag)

    sockets = WebSockets(socket_source=source)

    assert [connection.id for connection in sockets.get_connections("room:a")] == [
        "legacy",
        "connection-1",
    ]
    assert calls == ["room:a", None]


@pytest.mark.asyncio
async def test_direct_facet_upgrade_does_not_accept_a_physical_socket():
    facet = fakes.build_agent(
        cls=Agent,
        name="cf-agents:v2:leaf:0123456789abcdef",
    )

    response = await facet.fetch(_request())

    assert response.status == 404
    assert facet.ctx.accepted_websockets == []


@pytest.mark.asyncio
async def test_serialization_failure_closes_and_discards_accepted_socket(monkeypatch):
    class SerializationFailure(fakes.FakeSocket):
        def serializeAttachment(self, value):
            raise RuntimeError("cannot serialize")

    server = fakes.build_agent()
    socket = SerializationFailure()
    _install_agent_pair(monkeypatch, socket)

    with pytest.raises(RuntimeError, match="cannot serialize"):
        await server.fetch(_request())

    assert server.ctx.accepted_websockets == [socket]
    assert server.get_connections() == []
    assert socket.closed is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("code", "expected_close_calls"),
    [(1000, [(1000, "done")]), (1005, []), (1006, []), (1015, [])],
)
async def test_close_wake_reciprocates_only_legal_codes(code, expected_close_calls):
    class RecordingSocket(fakes.FakeSocket):
        def __init__(self):
            super().__init__(
                {
                    "__pk": {"id": "connection-1", "tags": ["connection-1"]},
                    "__user": None,
                }
            )
            self.close_calls: list[tuple[int, str]] = []

        def close(self, code=1000, reason=""):
            self.close_calls.append((code, reason))
            self.closed = True

    socket = RecordingSocket()
    seen: list[Connection] = []
    sockets = WebSockets(on_close=lambda connection, *args: seen.append(connection))

    assert await sockets.handle_close(socket, code, "done", True) is True
    assert [connection.id for connection in seen] == ["connection-1"]
    assert socket.close_calls == expected_close_calls


@pytest.mark.asyncio
async def test_close_reciprocation_is_best_effort():
    class ClosingFailure(fakes.FakeSocket):
        def close(self, code=1000, reason=""):
            raise RuntimeError("already closed")

    socket = ClosingFailure(
        {
            "__pk": {"id": "connection-1", "tags": ["connection-1"]},
            "__user": None,
        }
    )
    sockets = WebSockets(on_close=lambda *args: None)

    assert await sockets.handle_close(socket, 1000, "done", True) is True
    assert sockets.get_connections() == ()


@pytest.mark.asyncio
async def test_dispose_hydrates_and_closes_cold_managed_sockets():
    class RecordingSocket(fakes.FakeSocket):
        def __init__(self):
            super().__init__(
                {
                    "__pk": {"id": "connection-1", "tags": ["connection-1"]},
                    "__user": None,
                }
            )
            self.close_calls: list[tuple[int, str]] = []

        def close(self, code=1000, reason=""):
            self.close_calls.append((code, reason))
            self.closed = True

    socket = RecordingSocket()
    ctx = fakes.FakeCtx(websockets=[socket])
    sockets = WebSockets()
    lifecycle = Lifecycle(ctx, host=object())
    lifecycle.use(sockets)

    await lifecycle.dispose()

    assert socket.close_calls == [(1001, "Durable Object destroyed")]
    assert sockets._connections == {}


@pytest.mark.asyncio
async def test_relay_turns_are_serialized_per_physical_socket():
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    forwarded: list[tuple[str, object]] = []

    async def forward(payload):
        forwarded.append((payload["message"], payload["state"]))
        if payload["message"] == "first":
            first_started.set()
            await release_first.wait()
            return '[{"type":"state","data":{"step":1}}]'
        return "[]"

    socket = fakes.FakeSocket(
        {
            "__pk": {"id": "connection-1", "tags": ["connection-1"]},
            "__user": None,
            "target": {
                "url": "https://example.com/agents/agent/root/sub/child/leaf",
                "headers": {},
            },
        }
    )
    sockets = WebSockets(relay_forward=forward)

    first = asyncio.create_task(sockets.handle_message(socket, "first"))
    await first_started.wait()
    second = asyncio.create_task(sockets.handle_message(socket, "second"))
    await asyncio.sleep(0)

    assert forwarded == [("first", None)]
    release_first.set()
    await asyncio.gather(first, second)
    assert forwarded == [("first", None), ("second", {"step": 1})]


@pytest.mark.asyncio
async def test_unowned_wakes_are_declined_without_calling_handlers():
    events: list[str] = []
    sockets = WebSockets(
        on_message=lambda *args: events.append("message"),
        on_close=lambda *args: events.append("close"),
        on_error=lambda *args: events.append("error"),
    )
    socket = fakes.FakeSocket({"foreign": True})

    assert await sockets.handle_message(socket, "ignored") is False
    assert await sockets.handle_close(socket, 1000, "ignored", True) is False
    assert await sockets.handle_error(socket, RuntimeError("ignored")) is False
    assert events == []
