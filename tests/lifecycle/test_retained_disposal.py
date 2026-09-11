from __future__ import annotations

import asyncio
from typing import Any, cast

import fakes
import pytest
from workers import Request, Response

import agents.lifecycle.jobs as lifecycle_jobs_module
from agents.lifecycle import Lifecycle, LifecycleCapability, LifecycleRouteAddress


class RetainedCapability(LifecycleCapability):
    capability_id = "retained"


def build_retained_lifecycle(retain_work=None):
    lifecycle = Lifecycle(
        fakes.FakeCtx(),
        host=object(),
        retain_work=retain_work,
    )
    capability = RetainedCapability()
    lifecycle.use(capability)
    return lifecycle, capability


def test_disabled_retained_work_rejects_before_calling_factory():
    _lifecycle, capability = build_retained_lifecycle()
    called = []

    def factory():
        called.append(True)
        raise AssertionError("factory called")

    assert capability.lifecycle.retained_work.available is False
    with pytest.raises(RuntimeError, match="not available"):
        capability.lifecycle.retained_work.retain(factory)
    assert called == []


@pytest.mark.asyncio
async def test_enabled_retained_work_submits_once_and_runs_without_host_context():
    recorder = fakes.WaitUntilRecorder(owner="retained", context="rpc")
    lifecycle, capability = build_retained_lifecycle(recorder)
    contexts = []

    async def work() -> str:
        from agents.lifecycle import get_current_lifecycle_context

        contexts.append(get_current_lifecycle_context())
        await capability.lifecycle.storage.put("retained", "complete")
        return "done"

    factory_contexts = []

    def factory():
        from agents.lifecycle import get_current_lifecycle_context

        factory_contexts.append(get_current_lifecycle_context())
        return work()

    async def schedule() -> None:
        capability.lifecycle.retained_work.retain(factory)

    await capability.lifecycle.run_in_host_context(schedule)

    assert capability.lifecycle.retained_work.available is True
    assert len(recorder.coros) == 1
    assert await recorder.drain_next() == "done"
    assert await capability.lifecycle.storage.get("retained") == "complete"
    assert contexts == [None]
    assert factory_contexts == [None]
    assert recorder.records[0]["status"] == "completed"
    await lifecycle.start()


def test_retained_work_rejection_does_not_call_factory():
    factory_calls = []

    async def work() -> None:
        pass

    def reject(_awaitable) -> None:
        raise RuntimeError("submission failed")

    _lifecycle, capability = build_retained_lifecycle(reject)

    def factory():
        factory_calls.append(True)
        return work()

    with pytest.raises(RuntimeError, match="submission failed"):
        capability.lifecycle.retained_work.retain(factory)
    assert factory_calls == []


def test_cancelling_pending_retained_work_does_not_call_factory():
    recorder = fakes.WaitUntilRecorder()
    _lifecycle, capability = build_retained_lifecycle(recorder)
    factory_calls = []

    async def work() -> None:
        pass

    def factory():
        factory_calls.append(True)
        return work()

    capability.lifecycle.retained_work.retain(factory)
    recorder.cancel_all()

    assert factory_calls == []


class DisposeCapability(LifecycleCapability):
    capability_id = "dispose"

    def __init__(
        self,
        name: str,
        events: list[str],
        *,
        failure: BaseException | None = None,
        gate: fakes.AsyncGate | None = None,
    ):
        self.name = name
        self.events = events
        self.failure = failure
        self.gate = gate

    async def on_dispose(self) -> None:
        self.events.append(self.name)
        await self.lifecycle.storage.put(f"disposed:{self.name}", True)
        if self.gate is not None:
            await self.gate.block()
        if self.failure is not None:
            raise self.failure


def dispose_capability(
    capability_id: str,
    events: list[str],
    **options: Any,
) -> DisposeCapability:
    capability_type = type(
        f"{capability_id.title()}Capability",
        (DisposeCapability,),
        {"capability_id": capability_id},
    )
    return capability_type(capability_id, events, **options)


@pytest.mark.asyncio
async def test_disposal_uses_reverse_installation_order_and_continues_after_error():
    events: list[str] = []
    errors: list[BaseException] = []
    failure = RuntimeError("dispose failed")

    async def report(error: BaseException) -> None:
        errors.append(error)

    ctx = fakes.FakeCtx()
    lifecycle = Lifecycle(ctx, host=object(), on_error=report)
    fallback_one = dispose_capability("fallback-one", events)
    normal_one = dispose_capability("normal-one", events, failure=failure)
    fallback_two = dispose_capability("fallback-two", events)
    normal_two = dispose_capability("normal-two", events)
    lifecycle.use(fallback_one, fallback=True)
    lifecycle.use(normal_one)
    lifecycle.use(fallback_two, fallback=True)
    lifecycle.use(normal_two)

    await lifecycle.dispose()
    await lifecycle.dispose()

    assert events == ["normal-two", "fallback-two", "normal-one", "fallback-one"]
    assert errors == [failure]
    for capability in (fallback_one, normal_one, fallback_two, normal_two):
        assert await ctx.storage.get(f"disposed:{capability.name}") is True


@pytest.mark.asyncio
async def test_concurrent_disposal_shares_one_attempt():
    events: list[str] = []
    gate = fakes.AsyncGate()
    lifecycle = Lifecycle(fakes.FakeCtx(), host=object())
    capability = dispose_capability("gated", events, gate=gate)
    lifecycle.use(capability)

    first = asyncio.create_task(lifecycle.dispose())
    await gate.wait_until_blocked()
    second = asyncio.create_task(lifecycle.dispose())
    await asyncio.sleep(0)
    gate.release()
    await asyncio.gather(first, second)

    assert events == ["gated"]


@pytest.mark.asyncio
async def test_invalid_registration_disposal_publishes_failure_to_later_callers():
    lifecycle = Lifecycle(fakes.FakeCtx(), host=object())
    lifecycle.use(cast(Any, object()))

    for _ in range(2):
        with pytest.raises(ValueError, match="must inherit LifecycleCapability"):
            await lifecycle.dispose()


@pytest.mark.asyncio
async def test_pre_disposal_job_preparation_failure_is_retryable(monkeypatch):
    lifecycle = Lifecycle(fakes.FakeCtx(), host=object())
    attempts = 0
    original_prepare = lifecycle_jobs_module._LifecycleJobQueue.prepare

    def fail_once(queue):
        nonlocal attempts
        if queue is lifecycle._job_queue:
            attempts += 1
            if attempts == 1:
                raise RuntimeError("job preparation failed")
        return original_prepare(queue)

    monkeypatch.setattr(
        lifecycle_jobs_module._LifecycleJobQueue,
        "prepare",
        fail_once,
    )

    with pytest.raises(RuntimeError, match="job preparation failed"):
        await lifecycle.dispose()

    assert lifecycle._dispose_attempt is None
    assert lifecycle._disposing is False
    assert lifecycle._dispose_owner is None

    await lifecycle.dispose()

    assert attempts == 2
    assert lifecycle._disposed is True


@pytest.mark.asyncio
async def test_disposal_is_terminal_for_dispatch_events_routes_and_retained_work():
    address = LifecycleRouteAddress("root", "root-data")

    async def request(_request: Request) -> Response:
        return Response("host")

    lifecycle = Lifecycle(
        fakes.FakeCtx(),
        host=object(),
        on_request=request,
        route_address=address,
        retain_work=fakes.WaitUntilRecorder(),
    )
    capability = RetainedCapability()
    lifecycle.use(capability)
    await lifecycle.dispose()
    factory_calls = []

    def factory():
        factory_calls.append(True)

        async def work() -> None:
            pass

        return work()

    with pytest.raises(RuntimeError, match="disposed"):
        await lifecycle.start()
    with pytest.raises(RuntimeError, match="disposed"):
        await lifecycle.fetch(Request("https://example.com/"))
    with pytest.raises(RuntimeError, match="disposed"):
        await capability.lifecycle.events.emit("late", None)
    with pytest.raises(RuntimeError, match="disposed"):
        await capability.lifecycle.routes.to_root(None)
    with pytest.raises(RuntimeError, match="disposed"):
        capability.lifecycle.retained_work.retain(factory)
    with pytest.raises(RuntimeError, match="disposed"):
        await capability.lifecycle.storage.get("late")
    with pytest.raises(RuntimeError, match="disposed"):
        capability.lifecycle.sql.execute("SELECT 1")
    with pytest.raises(RuntimeError, match="disposed"):
        capability.lifecycle.sockets.get()
    assert capability.lifecycle.retained_work.available is False
    assert factory_calls == []


@pytest.mark.asyncio
async def test_disposal_waits_for_active_dispatch_before_running_hooks():
    gate = fakes.AsyncGate()
    events: list[str] = []

    async def request(_request: Request) -> Response:
        events.append("request:start")
        await gate.block()
        events.append("request:end")
        return Response("host")

    lifecycle = Lifecycle(
        fakes.FakeCtx(),
        host=object(),
        on_request=request,
    )
    capability = dispose_capability("disposed", events)
    lifecycle.use(capability)

    dispatch = asyncio.create_task(lifecycle.fetch(Request("https://example.com/")))
    await gate.wait_until_blocked()
    disposal = asyncio.create_task(lifecycle.dispose())
    await asyncio.sleep(0)

    assert events == ["request:start"]
    with pytest.raises(RuntimeError, match="disposed"):
        await lifecycle.start()

    gate.release()
    await asyncio.gather(dispatch, disposal)

    assert events == ["request:start", "request:end", "disposed"]


@pytest.mark.asyncio
async def test_in_flight_dispatch_keeps_storage_access_during_disposal():
    gate = fakes.AsyncGate()
    events: list[str] = []
    errors: list[str] = []
    capability = dispose_capability("disposed", events)

    async def request(_request: Request) -> Response:
        await gate.block()
        await capability.lifecycle.storage.put("in-flight", True)
        with pytest.raises(RuntimeError, match="active operation") as error:
            await lifecycle.dispose()
        errors.append(str(error.value))
        events.append("request")
        return Response("host")

    ctx = fakes.FakeCtx()
    lifecycle = Lifecycle(ctx, host=object(), on_request=request)
    lifecycle.use(capability)

    dispatch = asyncio.create_task(lifecycle.fetch(Request("https://example.com/")))
    await gate.wait_until_blocked()
    disposal = asyncio.create_task(lifecycle.dispose())
    await asyncio.sleep(0)
    gate.release()
    await asyncio.gather(dispatch, disposal)

    assert await ctx.storage.get("in-flight") is True
    assert events == ["request", "disposed"]
    assert errors == ["Lifecycle cannot be disposed during an active operation"]


@pytest.mark.asyncio
async def test_disposal_hook_can_clean_up_storage_but_cannot_admit_new_work():
    errors: list[str] = []
    recorder = fakes.WaitUntilRecorder()

    class CleanupCapability(LifecycleCapability):
        capability_id = "cleanup"

        async def on_dispose(self) -> None:
            await self.lifecycle.storage.put("cleanup", True)
            actions = (
                self.lifecycle.ready,
                lambda: self.lifecycle.events.emit("late", None),
                lambda: self.lifecycle.routes.to_root(None),
            )
            for action in actions:
                with pytest.raises(RuntimeError, match="disposed") as error:
                    await action()
                errors.append(str(error.value))

            def work_factory():
                raise AssertionError("factory called")

            with pytest.raises(RuntimeError, match="disposed") as error:
                self.lifecycle.retained_work.retain(work_factory)
            errors.append(str(error.value))

            with pytest.raises(RuntimeError, match="active operation") as error:
                await lifecycle.dispose()
            errors.append(str(error.value))

    ctx = fakes.FakeCtx()
    lifecycle = Lifecycle(
        ctx,
        host=object(),
        retain_work=recorder,
    )
    lifecycle.use(CleanupCapability())

    await lifecycle.dispose()

    assert await ctx.storage.get("cleanup") is True
    assert errors == ["Lifecycle is disposed"] * 4 + [
        "Lifecycle cannot be disposed during an active operation"
    ]
    assert recorder.coros == ()


@pytest.mark.asyncio
async def test_cancelling_disposal_wait_finishes_cleanup_before_propagating():
    gate = fakes.AsyncGate()
    events: list[str] = []

    async def request(_request: Request) -> Response:
        await gate.block()
        return Response("host")

    lifecycle = Lifecycle(fakes.FakeCtx(), host=object(), on_request=request)
    lifecycle.use(dispose_capability("disposed", events))
    dispatch = asyncio.create_task(lifecycle.fetch(Request("https://example.com/")))
    await gate.wait_until_blocked()
    disposal = asyncio.create_task(lifecycle.dispose())
    await asyncio.sleep(0)
    disposal.cancel()
    await asyncio.sleep(0)
    disposal.cancel()
    await asyncio.sleep(0)

    assert not disposal.done()

    gate.release()
    await dispatch
    with pytest.raises(asyncio.CancelledError):
        await disposal
    await lifecycle.dispose()

    assert events == ["disposed"]


@pytest.mark.asyncio
async def test_cancelling_disposal_hook_continues_cleanup_then_propagates():
    gate = fakes.AsyncGate()
    events: list[str] = []

    class GatedCleanupCapability(LifecycleCapability):
        capability_id = "gated"

        async def on_dispose(self) -> None:
            events.append("gated:start")
            await gate.block()
            events.append("gated:end")

    lifecycle = Lifecycle(fakes.FakeCtx(), host=object())
    lifecycle.use(dispose_capability("later", events))
    lifecycle.use(GatedCleanupCapability())

    disposal = asyncio.create_task(lifecycle.dispose())
    await gate.wait_until_blocked()
    disposal.cancel()
    await asyncio.sleep(0)

    assert not disposal.done()

    gate.release()

    with pytest.raises(asyncio.CancelledError):
        await disposal
    await lifecycle.dispose()

    assert events == ["gated:start", "gated:end", "later"]
