from __future__ import annotations

import asyncio

import fakes
import pytest
from workers import Request, Response

from agents.lifecycle import (
    CapabilityRequestContext,
    CapabilityWebSocketMessageContext,
    CurrentLifecycleContext,
    Lifecycle,
    LifecycleCapability,
    LifecycleHostContextScope,
    get_current_lifecycle_context,
)


def current_context() -> CurrentLifecycleContext:
    context = get_current_lifecycle_context()
    assert context is not None
    return context


class ContextCapability(LifecycleCapability):
    capability_id = "context"

    def __init__(
        self,
        seen: list[tuple[str, CurrentLifecycleContext | None]],
    ):
        self.seen = seen

    async def on_start(self) -> None:
        self.seen.append(("capability:start", get_current_lifecycle_context()))

    async def on_request(self, context: CapabilityRequestContext) -> Response | None:
        self.seen.append(("capability:request", get_current_lifecycle_context()))
        return None

    async def on_websocket_message(
        self,
        context: CapabilityWebSocketMessageContext,
    ) -> bool:
        self.seen.append(("capability:socket", get_current_lifecycle_context()))
        return False


class ContextHost:
    def __init__(self):
        self.seen: list[tuple[str, CurrentLifecycleContext | None]] = []
        self.lifecycle = Lifecycle(
            fakes.FakeCtx(),
            host=self,
            on_start=self.on_start,
            on_request=self.on_request,
            on_websocket_message=self.on_websocket_message,
        )
        self.capability = ContextCapability(self.seen)
        self.lifecycle.use(self.capability)

    async def on_start(self) -> None:
        self.seen.append(("host:start", get_current_lifecycle_context()))

    async def on_request(self, _request: Request) -> Response:
        self.seen.append(("host:request", get_current_lifecycle_context()))
        return Response("host")

    async def on_websocket_message(self, _context: object) -> bool:
        self.seen.append(("host:socket", get_current_lifecycle_context()))
        return False


@pytest.mark.asyncio
async def test_capability_hooks_are_isolated_and_host_hooks_receive_semantic_context():
    host = ContextHost()
    request = Request("https://example.com/")
    websocket = object()

    assert get_current_lifecycle_context() is None
    await host.lifecycle.fetch(request)
    await host.lifecycle.websocket_message(websocket, "message")
    assert get_current_lifecycle_context() is None

    seen = dict(host.seen)
    assert seen["capability:start"] is None
    assert seen["capability:request"] is None
    assert seen["capability:socket"] is None
    host_start = seen["host:start"]
    host_request = seen["host:request"]
    host_socket = seen["host:socket"]
    assert host_start is not None
    assert host_request is not None
    assert host_socket is not None
    assert host_start.host is host
    assert host_start.request is None
    assert host_request.host is host
    assert host_request.request is request
    assert host_socket.host is host
    assert host_socket.connection is websocket


@pytest.mark.asyncio
async def test_capability_explicitly_enters_host_context_across_awaits():
    host = ContextHost()
    request = Request("https://example.com/")
    connection = object()
    seen = []

    async def callback() -> str:
        seen.append(current_context())
        await asyncio.sleep(0)
        seen.append(current_context())
        return "ok"

    result = await host.capability.lifecycle.run_in_host_context(
        callback,
        scope=LifecycleHostContextScope(
            request=request,
            connection=connection,
        ),
    )

    assert result == "ok"
    assert [context.host for context in seen] == [host, host]
    assert [context.request for context in seen] == [request, request]
    assert [context.connection for context in seen] == [connection, connection]
    assert get_current_lifecycle_context() is None


@pytest.mark.asyncio
async def test_capability_start_clears_an_inherited_host_context():
    host = ContextHost()

    await host.capability.lifecycle.run_in_host_context(host.lifecycle.start)

    seen = dict(host.seen)
    assert seen["capability:start"] is None
    host_start = seen["host:start"]
    assert host_start is not None
    assert host_start.host is host
    assert get_current_lifecycle_context() is None


@pytest.mark.asyncio
async def test_nested_and_concurrent_host_contexts_restore_their_callers():
    first = ContextHost()
    second = ContextHost()
    nested = []

    async def second_callback() -> None:
        nested.append(current_context().host)

    async def first_callback() -> None:
        nested.append(current_context().host)
        await second.capability.lifecycle.run_in_host_context(second_callback)
        nested.append(current_context().host)

    await first.capability.lifecycle.run_in_host_context(first_callback)
    assert nested == [first, second, first]

    async def observe(host: ContextHost) -> list[object]:
        values = [current_context().host]
        await asyncio.sleep(0)
        values.append(current_context().host)
        return values

    first_values, second_values = await asyncio.gather(
        first.capability.lifecycle.run_in_host_context(lambda: observe(first)),
        second.capability.lifecycle.run_in_host_context(lambda: observe(second)),
    )

    assert first_values == [first, first]
    assert second_values == [second, second]
    assert get_current_lifecycle_context() is None


@pytest.mark.asyncio
async def test_host_context_is_restored_after_callback_failure():
    host = ContextHost()

    async def fail() -> None:
        assert current_context().host is host
        raise RuntimeError("callback failed")

    with pytest.raises(RuntimeError, match="callback failed"):
        await host.capability.lifecycle.run_in_host_context(fail)

    assert get_current_lifecycle_context() is None


@pytest.mark.asyncio
async def test_host_context_is_restored_after_callback_cancellation():
    host = ContextHost()
    gate = fakes.AsyncGate()

    async def block() -> None:
        assert current_context().host is host
        await gate.block()

    task = asyncio.create_task(host.capability.lifecycle.run_in_host_context(block))
    await gate.wait_until_blocked()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert get_current_lifecycle_context() is None
