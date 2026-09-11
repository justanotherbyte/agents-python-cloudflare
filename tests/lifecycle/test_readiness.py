from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

import fakes
import pytest
from workers import DurableObject, Request, Response

from agents.lifecycle import Lifecycle, LifecycleCapability


class FirstCapability(LifecycleCapability):
    capability_id = "first"

    def __init__(self, events: list[str], gate: fakes.AsyncGate):
        self.events = events
        self.gate = gate
        self.starts = 0

    async def on_start(self) -> None:
        self.starts += 1
        self.events.append("first")
        if self.starts == 1:
            await self.gate.block()


class MiddleCapability(LifecycleCapability):
    capability_id = "middle"

    def __init__(
        self,
        events: list[str],
        failure: BaseException,
        *,
        fail_once: bool,
    ):
        self.events = events
        self.failure = failure
        self.fail_once = fail_once

    async def on_start(self) -> None:
        self.events.append("middle")
        if self.fail_once:
            self.fail_once = False
            raise self.failure


class FallbackCapability(LifecycleCapability):
    capability_id = "fallback"

    def __init__(self, events: list[str]):
        self.events = events

    async def on_start(self) -> None:
        self.events.append("fallback")


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_owner", ["middle", "host"])
async def test_concurrent_startup_shares_failure_then_retries_complete_sequence(
    failure_owner: str,
):
    events: list[str] = []
    gate = fakes.AsyncGate()
    failure = RuntimeError(f"{failure_owner} failed")
    host_starts = 0

    async def host_start() -> None:
        nonlocal host_starts
        host_starts += 1
        events.append("host")
        if failure_owner == "host" and host_starts == 1:
            raise failure

    lifecycle = Lifecycle(fakes.FakeCtx(), host=object(), on_start=host_start)
    lifecycle.use(FirstCapability(events, gate))
    lifecycle.use(
        MiddleCapability(
            events,
            failure,
            fail_once=failure_owner == "middle",
        )
    )
    lifecycle.use(FallbackCapability(events), fallback=True)

    leader = asyncio.create_task(lifecycle.start())
    await gate.wait_until_blocked()
    follower = asyncio.create_task(lifecycle.start())
    await asyncio.sleep(0)

    with pytest.raises(RuntimeError, match="registration is closed"):
        lifecycle.use(FallbackCapability(events))

    gate.release()
    first_result, second_result = await asyncio.gather(
        leader,
        follower,
        return_exceptions=True,
    )

    assert first_result is failure
    assert second_result is failure

    await lifecycle.start()
    completed_events = list(events)
    await lifecycle.start()

    first_attempt = ["first", "middle"]
    if failure_owner == "host":
        first_attempt.extend(["fallback", "host"])
    assert events == first_attempt + ["first", "middle", "fallback", "host"]
    assert events == completed_events


@pytest.mark.asyncio
async def test_cancelling_one_waiter_does_not_cancel_shared_startup():
    events: list[str] = []
    gate = fakes.AsyncGate()
    capability = FirstCapability(events, gate)
    lifecycle = Lifecycle(fakes.FakeCtx(), host=object())
    lifecycle.use(capability)

    leader = asyncio.create_task(lifecycle.start())
    await gate.wait_until_blocked()
    waiter = asyncio.create_task(lifecycle.start())
    await asyncio.sleep(0)
    waiter.cancel()

    with pytest.raises(asyncio.CancelledError):
        await waiter

    gate.release()
    await leader
    await lifecycle.start()

    assert capability.starts == 1
    assert events == ["first"]


@pytest.mark.asyncio
async def test_cancelling_startup_owner_fails_shared_attempt_then_allows_retry():
    events: list[str] = []
    gate = fakes.AsyncGate()
    capability = FirstCapability(events, gate)
    lifecycle = Lifecycle(fakes.FakeCtx(), host=object())
    lifecycle.use(capability)

    leader = asyncio.create_task(lifecycle.start())
    await gate.wait_until_blocked()
    waiter = asyncio.create_task(lifecycle.start())
    await asyncio.sleep(0)
    leader.cancel()

    leader_result, waiter_result = await asyncio.gather(
        leader,
        waiter,
        return_exceptions=True,
    )

    assert type(leader_result) is asyncio.CancelledError
    assert type(waiter_result) is asyncio.CancelledError

    gate.release()
    await lifecycle.start()

    assert capability.starts == 2
    assert events == ["first", "first"]


class ValueCapability(LifecycleCapability):
    capability_id = "value"

    def __init__(
        self,
        events: list[str],
        gate: fakes.AsyncGate,
        set_value: Callable[[str], None],
    ):
        self.events = events
        self.gate = gate
        self.set_value = set_value
        self.starts = 0

    async def on_start(self) -> None:
        self.starts += 1
        self.events.append("capability:start")
        await self.gate.block()
        self.set_value("prepared")
        self.events.append("capability:end")


class PlainHost(DurableObject):
    def __init__(self, ctx: Any, env: Any, gate: fakes.AsyncGate):
        super().__init__(ctx, env)
        self.events: list[str] = []
        self.value: str | None = None
        self.lifecycle = Lifecycle(
            ctx,
            host=self,
            on_start=self._on_start,
            on_request=self._on_request,
        )
        self.capability = ValueCapability(self.events, gate, self._set_value)
        self.lifecycle.use(self.capability)

    def _set_value(self, value: str) -> None:
        self.value = value

    async def _on_start(self) -> None:
        assert self.value == "prepared"
        self.events.append("host:start")

    async def _on_request(self, _context: object) -> Response:
        self.events.append("request")
        return Response(self.value)

    async def fetch(self, request: Request) -> Response:
        return await self.lifecycle.fetch(request)

    async def prepared_value(self) -> str | None:
        await self.lifecycle.start()
        self.events.append("rpc")
        return self.value


@pytest.mark.asyncio
async def test_plain_durable_object_request_and_native_rpc_share_readiness():
    gate = fakes.AsyncGate()
    host = PlainHost(fakes.FakeCtx(), object(), gate)

    assert PlainHost.__bases__ == (DurableObject,)
    assert host.events == []
    assert host.capability.starts == 0

    request = asyncio.create_task(host.fetch(Request("https://example.com/")))
    await gate.wait_until_blocked()
    rpc = asyncio.create_task(host.prepared_value())
    await asyncio.sleep(0)
    gate.release()

    response, value = await asyncio.gather(request, rpc)

    assert response.body == "prepared"
    assert value == "prepared"
    assert host.events[:3] == ["capability:start", "capability:end", "host:start"]
    assert sorted(host.events[3:]) == ["request", "rpc"]
    assert host.capability.starts == 1


class StartupDataCapability(LifecycleCapability):
    capability_id = "startup-data"

    def __init__(
        self,
        events: list[str],
        read_startup_data: Callable[[], Awaitable[list[str] | None]],
    ):
        self.events = events
        self.read_startup_data = read_startup_data

    async def on_start(self) -> None:
        assert await self.read_startup_data() == ["root", "child"]
        self.events.append("capability:start")


class PlainBootstrapHost(DurableObject):
    def __init__(self, ctx: Any, env: Any):
        super().__init__(ctx, env)
        self.events: list[str] = []
        self.lifecycle = Lifecycle(ctx, host=self, on_start=self._on_start)
        self.lifecycle.use(StartupDataCapability(self.events, self._read_startup_data))

    async def _read_startup_data(self) -> list[str] | None:
        return await self.ctx.storage.get("startup_data")

    async def _on_start(self) -> None:
        assert await self._read_startup_data() == ["root", "child"]
        self.events.append("host:start")

    async def bootstrap(self, startup_data: list[str]) -> None:
        self.events.append("prepare")
        await self.ctx.storage.put("startup_data", startup_data)
        self.events.append("persist")
        await self.lifecycle.start()


@pytest.mark.asyncio
async def test_plain_host_can_persist_data_before_lifecycle_startup():
    host = PlainBootstrapHost(fakes.FakeCtx(), object())

    await host.bootstrap(["root", "child"])

    assert host.events == ["prepare", "persist", "capability:start", "host:start"]
