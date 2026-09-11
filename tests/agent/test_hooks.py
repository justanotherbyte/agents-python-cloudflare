"""Agent hook dispatch and startup behavior.

_dispatch_hook is the one place a hook failure is caught — it wraps the raise
in a HookError, reports it exactly once, and only re-raises when propagate is
true; an already-wrapped HookError passes through unreported so it is not
double-counted. Startup runs once and remains retryable after failure.
"""

from __future__ import annotations

import types

import fakes
import pytest
from workers import Request

import agents.lifecycle.websockets as websockets_module
from agents import Agent
from agents.core.error import HookError


class RecordingServer(Agent):
    def __init__(self, ctx, env):
        super().__init__(ctx, env)
        self.errors: list[BaseException] = []

    async def on_error(self, error, connection=None):
        self.errors.append(error)


def _recording() -> RecordingServer:
    return fakes.build_agent(cls=RecordingServer)


# ── _dispatch_hook ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_dispatch_hook_wraps_and_reraises_on_propagate():
    server = _recording()

    async def boom():
        raise ValueError("nope")

    with pytest.raises(HookError) as excinfo:
        await server._dispatch_hook("myhook", boom, propagate=True)

    err = excinfo.value
    assert isinstance(err.original, ValueError)
    assert server.errors == [err]
    assert server.errors[0].original is err.original


@pytest.mark.asyncio
async def test_dispatch_hook_swallows_on_no_propagate():
    server = _recording()

    async def boom():
        raise ValueError("nope")

    result = await server._dispatch_hook("myhook", boom, propagate=False)

    assert result is None
    assert len(server.errors) == 1
    assert isinstance(server.errors[0], HookError)
    assert isinstance(server.errors[0].original, ValueError)


@pytest.mark.asyncio
async def test_dispatch_hook_passes_prewrapped_hookerror_untouched():
    server = _recording()
    already = HookError("x", ValueError())

    async def boom():
        raise already

    with pytest.raises(HookError) as excinfo:
        await server._dispatch_hook("myhook", boom, propagate=True)

    assert excinfo.value is already
    assert server.errors == []


@pytest.mark.asyncio
async def test_dispatch_hook_invokes_successful_sync_hook():
    server = _recording()

    def answer(value):
        return value + 1

    assert await server._dispatch_hook("answer", answer, 4, propagate=True) == 5


@pytest.mark.asyncio
async def test_dispatch_hook_wraps_sync_raise():
    server = _recording()

    def boom():
        raise ValueError("sync nope")

    with pytest.raises(HookError) as excinfo:
        await server._dispatch_hook("myhook", boom, propagate=True)

    assert isinstance(excinfo.value.original, ValueError)
    assert server.errors == [excinfo.value]


@pytest.mark.asyncio
async def test_message_and_error_hooks_may_be_sync():
    class SyncServer(Agent):
        def __init__(self, ctx, env):
            super().__init__(ctx, env)
            self.messages = []
            self.errors = []

        def on_message(self, connection, message):
            self.messages.append((connection, message))

        def on_error(self, error, connection=None):
            self.errors.append((error, connection))

    server = fakes.build_agent(cls=SyncServer)
    connection = fakes.FakeConnection()
    error = ValueError("reported")

    await server._dispatch_message(connection, "hello")
    await server._report_error(error, connection)

    assert server.messages == [(connection, "hello")]
    assert server.errors == [(error, connection)]


# ── _ensure_initialized ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ensure_initialized_runs_on_start_once():
    class CountingServer(Agent):
        def __init__(self, ctx, env):
            super().__init__(ctx, env)
            self.start_calls = 0

        async def on_start(self):
            self.start_calls += 1

    server = fakes.build_agent(cls=CountingServer)

    await server._ensure_initialized()
    await server._ensure_initialized()

    assert server.start_calls == 1
    assert server._lifecycle._ready is True


@pytest.mark.asyncio
async def test_ensure_initialized_resets_status_on_failure_then_retries():
    class FlakyServer(Agent):
        def __init__(self, ctx, env):
            super().__init__(ctx, env)
            self.start_calls = 0
            self.succeeded = False

        async def on_error(self, error, connection=None):
            pass

        def on_start(self):
            self.start_calls += 1
            if self.start_calls == 1:
                raise ValueError("first attempt fails")
            self.succeeded = True

    server = fakes.build_agent(cls=FlakyServer)

    with pytest.raises(HookError):
        await server._ensure_initialized()
    assert server._lifecycle._ready is False

    await server._ensure_initialized()
    assert server.succeeded is True
    assert server.start_calls == 2
    assert server._lifecycle._ready is True


# ── WebSocket setup ──────────────────────────────────────────────────────────


def _install_socket_pair(monkeypatch, server):
    client = object()
    pair = types.SimpleNamespace(object_values=lambda: (client, server))
    monkeypatch.setattr(websockets_module.WebSocketPair, "new", lambda: pair)
    return client


def _websocket_request():
    return Request(
        "https://example.com/agents/test",
        headers={"connection": "upgrade", "upgrade": "websocket"},
    )


@pytest.mark.asyncio
async def test_failed_connect_closes_socket_and_removes_connection(monkeypatch):
    class FailingConnect(RecordingServer):
        async def on_connect(self, connection, ctx):
            raise ValueError("connect failed")

    server = fakes.build_agent(cls=FailingConnect)
    socket = fakes.FakeSocket()
    _install_socket_pair(monkeypatch, socket)

    with pytest.raises(HookError) as excinfo:
        await server.fetch(_websocket_request())

    assert str(excinfo.value.original) == "connect failed"
    assert socket.closed is True
    assert server._connections == {}
    assert server.errors == [excinfo.value]


@pytest.mark.asyncio
async def test_failed_tag_hook_closes_socket_without_registering(monkeypatch):
    class FailingTags(RecordingServer):
        def get_connection_tags(self, connection, ctx):
            raise ValueError("tags failed")

    server = fakes.build_agent(cls=FailingTags)
    socket = fakes.FakeSocket()
    _install_socket_pair(monkeypatch, socket)

    with pytest.raises(HookError) as excinfo:
        await server.fetch(_websocket_request())

    assert str(excinfo.value.original) == "tags failed"
    assert socket.closed is True
    assert server._connections == {}
    assert server.get_connections() == []
    assert server.errors == [excinfo.value]


@pytest.mark.asyncio
async def test_tag_hook_lookup_reuses_the_provisional_connection(monkeypatch):
    class ReentrantTags(Agent):
        def __init__(self, ctx, env):
            super().__init__(ctx, env)
            self.seen = None

        def get_connection_tags(self, connection, ctx):
            self.seen = self.get_connection(connection.id)
            return ["room:a"]

    server = fakes.build_agent(cls=ReentrantTags)
    socket = fakes.FakeSocket()
    _install_socket_pair(monkeypatch, socket)

    await server.fetch(_websocket_request())

    [connection] = server.get_connections()
    assert server.seen is connection
    assert connection.tags == [connection.id, "room:a"]
