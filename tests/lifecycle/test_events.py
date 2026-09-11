from __future__ import annotations

import asyncio

import fakes
import pytest

from agents.lifecycle import (
    Lifecycle,
    LifecycleCapability,
    LifecycleEvent,
    get_current_lifecycle_context,
)


class EventCapability(LifecycleCapability):
    capability_id = "events"


@pytest.mark.asyncio
async def test_events_use_stable_capability_source_and_listener_order(monkeypatch):
    delivered: list[tuple[str, LifecycleEvent]] = []
    capability = EventCapability()

    async def first(event: LifecycleEvent) -> None:
        delivered.append(("first", event))

    async def second(event: LifecycleEvent) -> None:
        delivered.append(("second", event))

    lifecycle = Lifecycle(
        fakes.FakeCtx(),
        host=object(),
        event_listeners=(first, second),
    )
    lifecycle.use(capability)
    monkeypatch.setattr(EventCapability, "capability_id", "changed")
    await lifecycle.start()

    await capability.lifecycle.events.emit("updated", {"value": 1})

    event = LifecycleEvent("events", "updated", {"value": 1})
    assert delivered == [("first", event), ("second", event)]


@pytest.mark.asyncio
async def test_failed_start_discards_buffered_events_and_retry_flushes_after_host():
    timeline: list[str] = []
    delivered: list[LifecycleEvent] = []

    class StartupEventCapability(LifecycleCapability):
        capability_id = "startup-events"

        def __init__(self):
            self.attempt = 0

        async def on_start(self) -> None:
            self.attempt += 1
            timeline.append(f"capability:{self.attempt}")
            await self.lifecycle.events.emit("attempt", self.attempt)
            if self.attempt == 1:
                raise RuntimeError("retry")

    async def host_start() -> None:
        timeline.append("host")

    async def listener(event: LifecycleEvent) -> None:
        timeline.append(f"listener:{event.payload}")
        delivered.append(event)

    lifecycle = Lifecycle(
        fakes.FakeCtx(),
        host=object(),
        on_start=host_start,
        event_listeners=(listener,),
    )
    lifecycle.use(StartupEventCapability())

    with pytest.raises(RuntimeError, match="retry"):
        await lifecycle.start()
    await lifecycle.start()

    assert timeline == ["capability:1", "capability:2", "host", "listener:2"]
    assert delivered == [LifecycleEvent("startup-events", "attempt", 2)]


@pytest.mark.asyncio
async def test_listener_failure_is_reported_without_reversing_storage():
    delivered: list[str] = []
    errors: list[BaseException] = []
    contexts = []

    async def failing(event: LifecycleEvent) -> None:
        contexts.append(get_current_lifecycle_context())
        delivered.append(f"failing:{event.type}")
        raise RuntimeError("listener failed")

    async def later(event: LifecycleEvent) -> None:
        contexts.append(get_current_lifecycle_context())
        delivered.append(f"later:{event.type}")

    async def report(error: BaseException) -> None:
        errors.append(error)

    lifecycle = Lifecycle(
        fakes.FakeCtx(),
        host=object(),
        event_listeners=(failing, later),
        on_error=report,
    )
    capability = EventCapability()
    lifecycle.use(capability)
    await lifecycle.start()

    await capability.lifecycle.storage.put("committed", True)
    await capability.lifecycle.events.emit("committed", None)

    assert await capability.lifecycle.storage.get("committed") is True
    assert delivered == ["failing:committed", "later:committed"]
    assert len(errors) == 1
    assert str(errors[0]) == "listener failed"
    assert contexts == [None, None]


@pytest.mark.asyncio
async def test_concurrent_events_are_serialized_in_emission_order():
    gate = fakes.AsyncGate()
    delivered: list[str] = []

    async def listener(event: LifecycleEvent) -> None:
        delivered.append(event.type)
        if event.type == "first":
            await gate.block()

    lifecycle = Lifecycle(
        fakes.FakeCtx(),
        host=object(),
        event_listeners=(listener,),
    )
    capability = EventCapability()
    lifecycle.use(capability)
    await lifecycle.start()

    first = asyncio.create_task(capability.lifecycle.events.emit("first", None))
    await gate.wait_until_blocked()
    second = asyncio.create_task(capability.lifecycle.events.emit("second", None))
    await asyncio.sleep(0)
    gate.release()
    await asyncio.gather(first, second)

    assert delivered == ["first", "second"]


@pytest.mark.asyncio
async def test_startup_events_are_queued_before_live_events():
    gate = fakes.AsyncGate()
    delivered: list[str] = []

    class StartupEventsCapability(LifecycleCapability):
        capability_id = "startup"

        async def on_start(self) -> None:
            await self.lifecycle.events.emit("startup:first", None)
            await self.lifecycle.events.emit("startup:second", None)

    async def listener(event: LifecycleEvent) -> None:
        delivered.append(event.type)
        if event.type == "startup:first":
            await gate.block()

    lifecycle = Lifecycle(
        fakes.FakeCtx(),
        host=object(),
        event_listeners=(listener,),
    )
    capability = StartupEventsCapability()
    lifecycle.use(capability)

    startup = asyncio.create_task(lifecycle.start())
    await gate.wait_until_blocked()
    readiness = asyncio.create_task(lifecycle.start())
    live = asyncio.create_task(capability.lifecycle.events.emit("live", None))
    await asyncio.sleep(0)

    assert not readiness.done()

    gate.release()
    await asyncio.gather(startup, readiness, live)

    assert delivered == ["startup:first", "startup:second", "live"]


@pytest.mark.asyncio
async def test_cancelling_event_delivery_does_not_get_swallowed_or_stall_queue():
    gate = fakes.AsyncGate()
    delivered: list[str] = []

    async def listener(event: LifecycleEvent) -> None:
        delivered.append(event.type)
        if event.type == "first":
            await gate.block()

    lifecycle = Lifecycle(
        fakes.FakeCtx(),
        host=object(),
        event_listeners=(listener,),
    )
    capability = EventCapability()
    lifecycle.use(capability)
    await lifecycle.start()

    first = asyncio.create_task(capability.lifecycle.events.emit("first", None))
    await gate.wait_until_blocked()
    first.cancel()

    with pytest.raises(asyncio.CancelledError):
        await first

    await capability.lifecycle.events.emit("second", None)

    assert delivered == ["first", "second"]
