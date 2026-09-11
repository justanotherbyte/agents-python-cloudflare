from __future__ import annotations

import asyncio

import fakes
import pytest

import agents.lifecycle._runtime as lifecycle_module
from agents.lifecycle import (
    Lifecycle,
    LifecycleCapability,
    LifecycleJobContext,
    LifecycleJobPushOptions,
    LifecycleJobReschedule,
    LifecycleMemoryLimitContext,
    LifecycleRouteAddress,
    get_current_lifecycle_context,
)


@pytest.mark.asyncio
async def test_rearm_combines_host_deadline_with_job_queue_under_one_owner(
    monkeypatch,
):
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)
    host_deadline = 1_500
    ctx = fakes.FakeCtx()
    lifecycle = Lifecycle(
        ctx,
        host=object(),
        alarm_deadline=lambda _now: host_deadline,
    )

    await lifecycle.jobs.push(
        LifecycleJobPushOptions(id="later", fn="work", time=2_000)
    )

    assert ctx.storage.alarm_time_ms == 1_500


@pytest.mark.asyncio
async def test_alarm_drives_jobs_in_order_then_host_and_applies_outcomes(monkeypatch):
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)
    events: list[tuple[str, str, int]] = []

    class JobsCapability(LifecycleCapability):
        capability_id = "jobs"

        async def on_job(self, context: LifecycleJobContext):
            assert get_current_lifecycle_context() is None
            events.append(("capability", context.job.fn, context.attempt))
            if context.job.fn == "reschedule":
                return LifecycleJobReschedule(reschedule_at=5_000)
            if context.job.fn == "yield":
                return "yield"
            return None

    async def host_job(context: LifecycleJobContext):
        current = get_current_lifecycle_context()
        assert current is not None
        assert current.host is host
        events.append(("host-job", context.job.fn, context.attempt))

    async def host_alarm():
        current = get_current_lifecycle_context()
        assert current is not None
        assert current.host is host
        events.append(("host-alarm", "alarm", 0))

    host = object()
    ctx = fakes.FakeCtx()
    lifecycle = Lifecycle(
        ctx,
        host=host,
        on_job=host_job,
        on_alarm=host_alarm,
    )
    capability = JobsCapability()
    lifecycle.use(capability)
    await lifecycle.start()

    await capability.lifecycle.jobs.push(
        LifecycleJobPushOptions(id="complete", fn="complete", time=100)
    )
    await capability.lifecycle.jobs.push(
        LifecycleJobPushOptions(id="reschedule", fn="reschedule", time=200)
    )
    await lifecycle.jobs.push(LifecycleJobPushOptions(id="host", fn="host", time=300))
    await capability.lifecycle.jobs.push(
        LifecycleJobPushOptions(id="yield", fn="yield", time=400)
    )

    await lifecycle.alarm()

    assert events == [
        ("capability", "complete", 1),
        ("capability", "reschedule", 1),
        ("host-job", "host", 1),
        ("capability", "yield", 1),
        ("host-alarm", "alarm", 0),
    ]
    assert await capability.lifecycle.jobs.get("complete") is None
    assert await lifecycle.jobs.get("host") is None
    rescheduled = await capability.lifecycle.jobs.get("reschedule")
    assert rescheduled is not None
    assert rescheduled.time == 5_000
    yielded = await capability.lifecycle.jobs.get("yield")
    assert yielded is not None
    assert yielded.time == 400
    assert ctx.storage.sql.exec(
        "SELECT running FROM cf_agents_jobs WHERE id = 'yield'"
    ).toArray() == [{"running": 0}]
    assert ctx.storage.alarm_time_ms == 1_001


@pytest.mark.asyncio
async def test_alarm_arms_deadman_before_dispatch(monkeypatch):
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)
    ctx = fakes.FakeCtx()

    async def on_job(_context: LifecycleJobContext) -> None:
        assert ctx.storage.alarm_time_ms == 31_000

    lifecycle = Lifecycle(ctx, host=object(), on_job=on_job)
    await lifecycle.jobs.push(LifecycleJobPushOptions(id="job", fn="work", time=100))

    await lifecycle.alarm()

    assert ctx.storage.alarm_history_ms[-2:] == (("set", 31_000), ("delete", None))


@pytest.mark.asyncio
async def test_alarm_refetches_each_due_job_before_claim(monkeypatch):
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)
    driven: list[str] = []

    class RetimingCapability(LifecycleCapability):
        capability_id = "retimer"

        async def on_job(self, context: LifecycleJobContext):
            driven.append(context.job.id)
            if context.job.id == "first":
                assert await self.lifecycle.jobs.reschedule("second", 5_000)

    ctx = fakes.FakeCtx()
    lifecycle = Lifecycle(ctx, host=object())
    capability = RetimingCapability()
    lifecycle.use(capability)
    await lifecycle.start()
    await capability.lifecycle.jobs.push(
        LifecycleJobPushOptions(id="first", fn="work", time=100)
    )
    await capability.lifecycle.jobs.push(
        LifecycleJobPushOptions(id="second", fn="work", time=200)
    )

    await lifecycle.alarm()

    assert driven == ["first"]
    second = await capability.lifecycle.jobs.get("second")
    assert second is not None
    assert second.time == 5_000
    assert ctx.storage.alarm_time_ms == 5_000


@pytest.mark.asyncio
async def test_callback_push_fences_its_older_completion_outcome(monkeypatch):
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)
    ctx = fakes.FakeCtx()
    lifecycle: Lifecycle

    async def replace_job(context: LifecycleJobContext) -> None:
        await lifecycle.jobs.push(
            LifecycleJobPushOptions(
                id=context.job.id,
                fn="new-intent",
                time=5_000,
            )
        )

    lifecycle = Lifecycle(ctx, host=object(), on_job=replace_job)
    await lifecycle.jobs.push(
        LifecycleJobPushOptions(id="same-id", fn="old-intent", time=100)
    )

    await lifecycle.alarm()

    replacement = await lifecycle.jobs.get("same-id")
    assert replacement is not None
    assert replacement.fn == "new-intent"
    assert replacement.time == 5_000
    assert ctx.storage.alarm_time_ms == 5_000


@pytest.mark.asyncio
async def test_active_callback_can_push_while_disposal_waits(monkeypatch):
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)
    gate = fakes.AsyncGate()
    ctx = fakes.FakeCtx()
    lifecycle: Lifecycle

    async def push_before_finishing(_context: LifecycleJobContext) -> None:
        await gate.block()
        await lifecycle.jobs.push(
            LifecycleJobPushOptions(id="nested", fn="later", time=5_000)
        )

    lifecycle = Lifecycle(ctx, host=object(), on_job=push_before_finishing)
    await lifecycle.jobs.push(
        LifecycleJobPushOptions(id="current", fn="work", time=100)
    )
    alarm = asyncio.create_task(lifecycle.alarm())
    await gate.wait_until_blocked()

    disposal = asyncio.create_task(lifecycle.dispose())
    await asyncio.sleep(0)

    assert not disposal.done()
    gate.release()
    await alarm
    await disposal
    assert ctx.storage.sql.exec("SELECT id, time FROM cf_agents_jobs").toArray() == [
        {"id": "nested", "time": 5_000}
    ]


@pytest.mark.asyncio
async def test_alarm_failure_rearms_without_masking_primary_error(monkeypatch):
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)
    primary = RuntimeError("host alarm failed")
    rearm_failure = RuntimeError("rearm failed")
    reported: list[BaseException] = []
    ctx = fakes.FakeCtx()

    async def host_alarm():
        raise primary

    async def report(error: BaseException):
        reported.append(error)

    lifecycle = Lifecycle(
        ctx,
        host=object(),
        on_alarm=host_alarm,
        on_error=report,
    )
    await lifecycle.start()

    async def fail_delete_alarm():
        raise rearm_failure

    monkeypatch.setattr(ctx.storage, "deleteAlarm", fail_delete_alarm)

    with pytest.raises(RuntimeError, match="host alarm failed") as raised:
        await lifecycle.alarm()

    assert raised.value is primary
    assert reported == [rearm_failure]


@pytest.mark.asyncio
async def test_startup_failure_keeps_due_job_armed_for_retry(monkeypatch):
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)
    ctx = fakes.FakeCtx()
    seed = Lifecycle(ctx, host=object())
    await seed.jobs.push(LifecycleJobPushOptions(id="due", fn="work", time=100))
    failure = RuntimeError("startup failed")

    async def fail_start():
        raise failure

    lifecycle = Lifecycle(ctx, host=object(), on_start=fail_start)

    with pytest.raises(RuntimeError, match="startup failed") as raised:
        await lifecycle.alarm()

    assert raised.value is failure
    assert ctx.storage.alarm_time_ms == 1_001
    assert ctx.storage.sql.exec(
        "SELECT id, running FROM cf_agents_jobs WHERE id = 'due'"
    ).toArray() == [{"id": "due", "running": 0}]


@pytest.mark.asyncio
async def test_application_failure_retries_then_uses_error_outcome(monkeypatch):
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)
    events: list[object] = []
    failure = RuntimeError("job failed")

    class FailingCapability(LifecycleCapability):
        capability_id = "failing"

        async def on_job(self, context: LifecycleJobContext):
            events.append(("attempt", context.attempt))
            raise failure

        async def on_job_error(
            self,
            context: LifecycleJobContext,
            error: BaseException,
        ) -> LifecycleJobReschedule:
            events.append(("error", context.attempt, error))
            return LifecycleJobReschedule(reschedule_at=5_000)

    async def host_alarm():
        events.append("host-alarm")

    ctx = fakes.FakeCtx()
    lifecycle = Lifecycle(ctx, host=object(), on_alarm=host_alarm)
    capability = FailingCapability()
    lifecycle.use(capability)
    await capability.lifecycle.jobs.push(
        LifecycleJobPushOptions(
            id="failed",
            fn="work",
            time=100,
            retry={"maxAttempts": 2, "baseDelayMs": 1, "maxDelayMs": 1},
        )
    )

    await lifecycle.alarm()

    assert events == [
        ("attempt", 1),
        ("attempt", 2),
        ("error", 2, failure),
        "host-alarm",
    ]
    assert ctx.storage.sql.exec(
        "SELECT id, time, running, execution_started_at FROM cf_agents_jobs"
    ).toArray() == [
        {
            "id": "failed",
            "time": 5_000,
            "running": 0,
            "execution_started_at": None,
        }
    ]
    assert ctx.storage.alarm_time_ms == 5_000


@pytest.mark.asyncio
async def test_platform_failure_preserves_job_and_skips_error_and_host(monkeypatch):
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)
    attempts: list[int] = []
    error_hook_calls = 0
    host_alarm_calls = 0

    class PlatformFailure(RuntimeError):
        retryable = True
        overloaded = False

    failure = PlatformFailure("storage unavailable")

    class FailingCapability(LifecycleCapability):
        capability_id = "failing"

        async def on_job(self, context: LifecycleJobContext) -> None:
            attempts.append(context.attempt)
            raise failure

        async def on_job_error(
            self,
            _context: LifecycleJobContext,
            _error: BaseException,
        ) -> None:
            nonlocal error_hook_calls
            error_hook_calls += 1

    async def host_alarm() -> None:
        nonlocal host_alarm_calls
        host_alarm_calls += 1

    ctx = fakes.FakeCtx()
    lifecycle = Lifecycle(ctx, host=object(), on_alarm=host_alarm)
    capability = FailingCapability()
    lifecycle.use(capability)
    await capability.lifecycle.jobs.push(
        LifecycleJobPushOptions(
            id="failed",
            fn="work",
            time=100,
            retry={"maxAttempts": 2, "baseDelayMs": 1, "maxDelayMs": 1},
        )
    )

    with pytest.raises(PlatformFailure) as raised:
        await lifecycle.alarm()

    assert raised.value is failure
    assert attempts == [1, 2]
    assert error_hook_calls == 0
    assert host_alarm_calls == 0
    assert ctx.storage.sql.exec(
        "SELECT id, running, execution_started_at FROM cf_agents_jobs"
    ).toArray() == [{"id": "failed", "running": 0, "execution_started_at": None}]
    assert ctx.storage.alarm_time_ms == 1_001


@pytest.mark.asyncio
async def test_platform_failure_from_job_error_preserves_job(monkeypatch):
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)

    class PlatformFailure(RuntimeError):
        retryable = True
        overloaded = False

    failure = PlatformFailure("storage unavailable")

    class FailingCapability(LifecycleCapability):
        capability_id = "failing"

        async def on_job(self, _context: LifecycleJobContext) -> None:
            raise ValueError("application failure")

        async def on_job_error(
            self,
            _context: LifecycleJobContext,
            _error: BaseException,
        ) -> None:
            raise failure

    ctx = fakes.FakeCtx()
    lifecycle = Lifecycle(ctx, host=object())
    capability = FailingCapability()
    lifecycle.use(capability)
    await capability.lifecycle.jobs.push(
        LifecycleJobPushOptions(
            id="failed",
            fn="work",
            time=100,
            retry={"maxAttempts": 1},
        )
    )

    with pytest.raises(PlatformFailure) as raised:
        await lifecycle.alarm()

    assert raised.value is failure
    assert ctx.storage.sql.exec(
        "SELECT id, running, execution_started_at FROM cf_agents_jobs"
    ).toArray() == [{"id": "failed", "running": 0, "execution_started_at": None}]
    assert ctx.storage.alarm_time_ms == 1_001


@pytest.mark.asyncio
async def test_memory_limit_from_job_error_reaches_breaker(monkeypatch):
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)

    class FailingCapability(LifecycleCapability):
        capability_id = "failing"

        async def on_job(self, _context: LifecycleJobContext) -> None:
            raise ValueError("application failure")

        async def on_job_error(
            self,
            _context: LifecycleJobContext,
            _error: BaseException,
        ) -> None:
            raise RuntimeError("isolate exceeded its memory limit")

    ctx = fakes.FakeCtx()
    resets: list[str] = []
    lifecycle = Lifecycle(ctx, host=object(), reset_alarm=resets.append)
    capability = FailingCapability()
    lifecycle.use(capability)
    await capability.lifecycle.jobs.push(
        LifecycleJobPushOptions(
            id="failed",
            fn="work",
            time=100,
            retry={"maxAttempts": 1},
        )
    )

    await lifecycle.alarm()

    assert resets == ["alarm memory-limit strike 1/3"]
    assert ctx.storage.sql.exec(
        "SELECT id, time, running, execution_started_at FROM cf_agents_jobs"
    ).toArray() == [
        {
            "id": "failed",
            "time": 31_000,
            "running": 0,
            "execution_started_at": None,
        }
    ]


@pytest.mark.asyncio
async def test_singleflight_skips_live_claim_then_reclaims_hung_row(monkeypatch):
    current_time = 1_000
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: current_time)
    attempts: list[int] = []

    async def on_job(context: LifecycleJobContext) -> None:
        attempts.append(context.attempt)

    ctx = fakes.FakeCtx()
    lifecycle = Lifecycle(ctx, host=object(), on_job=on_job)
    await lifecycle.jobs.push(
        LifecycleJobPushOptions(
            id="singleflight",
            fn="work",
            time=100,
            singleflight=True,
            hung_timeout_seconds=2,
        )
    )
    ctx.storage.sql.exec(
        "UPDATE cf_agents_jobs SET running = 1, execution_started_at = 900 "
        "WHERE id = 'singleflight'"
    )

    await lifecycle.alarm()

    assert attempts == []
    assert ctx.storage.alarm_time_ms == 2_900

    current_time = 3_000
    await lifecycle.alarm()

    assert attempts == [1]
    assert await lifecycle.jobs.get("singleflight") is None


@pytest.mark.asyncio
async def test_stale_singleflight_completion_cannot_delete_reclaimed_run(monkeypatch):
    current_time = 1_000
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: current_time)
    first_gate = fakes.AsyncGate()
    second_gate = fakes.AsyncGate()
    calls = 0

    async def on_job(_context: LifecycleJobContext) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            await first_gate.block()
        else:
            await second_gate.block()

    ctx = fakes.FakeCtx()
    lifecycle = Lifecycle(ctx, host=object(), on_job=on_job)
    await lifecycle.jobs.push(
        LifecycleJobPushOptions(
            id="singleflight",
            fn="work",
            time=100,
            singleflight=True,
            hung_timeout_seconds=2,
        )
    )

    stale_alarm = asyncio.create_task(lifecycle.alarm())
    await first_gate.wait_until_blocked()
    current_time = 4_000
    reclaim_alarm = asyncio.create_task(lifecycle.alarm())
    await second_gate.wait_until_blocked()

    first_gate.release()
    await stale_alarm

    assert ctx.storage.sql.exec(
        "SELECT running, execution_started_at FROM cf_agents_jobs "
        "WHERE id = 'singleflight'"
    ).toArray() == [{"running": 1, "execution_started_at": 4_000}]

    second_gate.release()
    await reclaim_alarm
    assert await lifecycle.jobs.get("singleflight") is None


@pytest.mark.asyncio
async def test_alarm_dispatches_only_exclusive_jobs_while_one_is_pending(monkeypatch):
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)
    dispatched: list[str] = []

    async def on_job(context: LifecycleJobContext) -> None:
        dispatched.append(context.job.id)

    ctx = fakes.FakeCtx()
    lifecycle = Lifecycle(ctx, host=object(), on_job=on_job)
    await lifecycle.jobs.push(
        LifecycleJobPushOptions(id="ordinary", fn="work", time=100)
    )
    await lifecycle.jobs.push(
        LifecycleJobPushOptions(
            id="exclusive",
            fn="work",
            time=500,
            exclusive=True,
        )
    )

    await lifecycle.alarm()

    assert dispatched == ["exclusive"]
    assert await lifecycle.jobs.get("exclusive") is None
    assert await lifecycle.jobs.get("ordinary") is not None
    assert ctx.storage.alarm_time_ms == 1_001


@pytest.mark.asyncio
async def test_exclusive_push_stops_later_ordinary_claims_in_same_alarm(monkeypatch):
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)
    dispatched: list[str] = []
    lifecycle: Lifecycle

    async def on_job(context: LifecycleJobContext) -> None:
        dispatched.append(context.job.id)
        if context.job.id == "first":
            await lifecycle.jobs.push(
                LifecycleJobPushOptions(
                    id="exclusive",
                    fn="work",
                    time=50,
                    exclusive=True,
                )
            )

    ctx = fakes.FakeCtx()
    lifecycle = Lifecycle(ctx, host=object(), on_job=on_job)
    await lifecycle.jobs.push(LifecycleJobPushOptions(id="first", fn="work", time=100))
    await lifecycle.jobs.push(LifecycleJobPushOptions(id="second", fn="work", time=200))

    await lifecycle.alarm()

    assert dispatched == ["first"]
    assert ctx.storage.sql.exec(
        "SELECT id FROM cf_agents_jobs ORDER BY id"
    ).toArray() == [{"id": "exclusive"}, {"id": "second"}]

    await lifecycle.alarm()

    assert dispatched == ["first", "exclusive"]
    assert await lifecycle.jobs.get("second") is not None


@pytest.mark.asyncio
async def test_corrupt_future_exclusive_job_does_not_starve_valid_work(monkeypatch):
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)
    dispatched: list[str] = []

    async def on_job(context: LifecycleJobContext) -> None:
        dispatched.append(context.job.id)

    ctx = fakes.FakeCtx()
    lifecycle = Lifecycle(ctx, host=object(), on_job=on_job)
    await lifecycle.jobs.push(
        LifecycleJobPushOptions(
            id="corrupt-exclusive",
            fn="work",
            time=5_000,
            exclusive=True,
        )
    )
    ctx.storage.sql.exec(
        "UPDATE cf_agents_jobs SET retry_options = '{' WHERE id = 'corrupt-exclusive'"
    )
    await lifecycle.jobs.push(
        LifecycleJobPushOptions(id="ordinary", fn="work", time=100)
    )

    assert ctx.storage.alarm_time_ms == 1_001
    await lifecycle.alarm()

    assert dispatched == ["ordinary"]
    assert ctx.storage.alarm_time_ms == 5_000


@pytest.mark.asyncio
async def test_older_alarm_outcome_cannot_delete_newer_claim(monkeypatch):
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)
    first_gate = fakes.AsyncGate()
    second_gate = fakes.AsyncGate()
    calls: list[str] = []

    async def on_job(context: LifecycleJobContext) -> None:
        calls.append(context.job.fn)
        if context.job.fn == "older":
            await first_gate.block()
        else:
            await second_gate.block()

    ctx = fakes.FakeCtx()
    lifecycle = Lifecycle(ctx, host=object(), on_job=on_job)
    await lifecycle.jobs.push(LifecycleJobPushOptions(id="same", fn="older", time=100))

    older_alarm = asyncio.create_task(lifecycle.alarm())
    await first_gate.wait_until_blocked()
    await lifecycle.jobs.push(LifecycleJobPushOptions(id="same", fn="newer", time=100))
    newer_alarm = asyncio.create_task(lifecycle.alarm())
    await second_gate.wait_until_blocked()

    first_gate.release()
    await older_alarm

    claimed = ctx.storage.sql.exec(
        "SELECT fn, running FROM cf_agents_jobs WHERE id = 'same'"
    ).toArray()
    assert claimed == [{"fn": "newer", "running": 1}]

    second_gate.release()
    await newer_alarm

    assert calls == ["older", "newer"]
    assert await lifecycle.jobs.get("same") is None


@pytest.mark.asyncio
async def test_startup_job_push_is_rearmed_after_startup(monkeypatch):
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)

    class StartupCapability(LifecycleCapability):
        capability_id = "startup"

        async def on_start(self) -> None:
            await self.lifecycle.jobs.push(
                LifecycleJobPushOptions(id="startup", fn="work", time=4_000)
            )

    ctx = fakes.FakeCtx()
    lifecycle = Lifecycle(ctx, host=object())
    lifecycle.use(StartupCapability())

    await lifecycle.start()

    assert ctx.storage.alarm_time_ms == 4_000


@pytest.mark.asyncio
async def test_failed_startup_rearms_a_job_pushed_during_startup(monkeypatch):
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)
    ctx = fakes.FakeCtx()
    lifecycle: Lifecycle

    async def fail_after_push() -> None:
        await lifecycle.jobs.push(
            LifecycleJobPushOptions(id="startup", fn="work", time=4_000)
        )
        raise RuntimeError("startup failed")

    lifecycle = Lifecycle(ctx, host=object(), on_start=fail_after_push)

    with pytest.raises(RuntimeError, match="startup failed"):
        await lifecycle.start()

    assert ctx.storage.alarm_time_ms == 4_000
    assert ctx.storage.sql.exec("SELECT id, running FROM cf_agents_jobs").toArray() == [
        {"id": "startup", "running": 0}
    ]


@pytest.mark.asyncio
async def test_facet_jobs_are_rejected_without_owner_keyed_root_routing(monkeypatch):
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)
    facet_ctx = fakes.FakeCtx()

    facet = Lifecycle(
        facet_ctx,
        host=object(),
        route_address=LifecycleRouteAddress("root/facet", "facet-data"),
        root_route_address=LifecycleRouteAddress("root", "root-data"),
    )

    with pytest.raises(RuntimeError, match="owner-keyed root routing"):
        await facet.jobs.push(
            LifecycleJobPushOptions(id="facet-job", fn="work", time=4_000)
        )
    with pytest.raises(RuntimeError, match="owner-keyed root routing"):
        await facet.alarm()

    assert facet_ctx.storage.alarm_time_ms is None
    assert facet_ctx.storage.alarm_history_ms == ()
    assert (
        facet_ctx.storage.sql.exec(
            "SELECT name FROM sqlite_master WHERE name = 'cf_agents_jobs'"
        ).toArray()
        == []
    )


@pytest.mark.asyncio
async def test_corrupt_due_job_is_reported_and_does_not_block_later_jobs(
    monkeypatch,
):
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)
    reported: list[BaseException] = []
    driven: list[str] = []

    async def on_job(context: LifecycleJobContext) -> None:
        driven.append(context.job.id)

    async def on_error(error: BaseException) -> None:
        reported.append(error)

    ctx = fakes.FakeCtx()
    lifecycle = Lifecycle(ctx, host=object(), on_job=on_job, on_error=on_error)
    await lifecycle.start()
    ctx.storage.sql.exec(
        "INSERT INTO cf_agents_jobs "
        "(id, capability, fn, time, payload) "
        "VALUES ('corrupt', 'host', 'work', 100, '{\"value\":NaN}')"
    )
    await lifecycle.jobs.push(LifecycleJobPushOptions(id="valid", fn="work", time=200))

    await lifecycle.alarm()

    assert driven == ["valid"]
    assert len(reported) == 1
    assert isinstance(reported[0], ValueError)
    assert ctx.storage.sql.exec("SELECT id FROM cf_agents_jobs").toArray() == []


@pytest.mark.asyncio
async def test_cleanup_failure_does_not_mask_job_failure(monkeypatch):
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)

    class PlatformFailure(RuntimeError):
        retryable = True
        overloaded = False

    primary = PlatformFailure("job failed")
    cleanup = RuntimeError("cleanup failed")
    reported: list[BaseException] = []

    async def on_job(_context: LifecycleJobContext) -> None:
        raise primary

    async def on_error(error: BaseException) -> None:
        reported.append(error)

    ctx = fakes.FakeCtx()
    lifecycle = Lifecycle(ctx, host=object(), on_job=on_job, on_error=on_error)
    await lifecycle.jobs.push(
        LifecycleJobPushOptions(
            id="job",
            fn="work",
            time=100,
            retry={"maxAttempts": 1},
        )
    )

    def fail_cleanup(_queue: object, _row: object) -> None:
        raise cleanup

    monkeypatch.setattr(
        type(lifecycle._job_queue),
        "clear_intent",
        fail_cleanup,
    )

    with pytest.raises(PlatformFailure, match="job failed") as raised:
        await lifecycle.alarm()

    assert raised.value is primary
    assert reported == [cleanup]


@pytest.mark.asyncio
async def test_rearm_failure_does_not_replace_cancellation(monkeypatch):
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)
    rearm_failure = RuntimeError("rearm failed")
    reported: list[BaseException] = []
    gate = fakes.AsyncGate()
    ctx = fakes.FakeCtx()

    async def on_error(error: BaseException) -> None:
        reported.append(error)

    async def fail_rearm(_deadline: int) -> None:
        await gate.block()
        raise rearm_failure

    lifecycle = Lifecycle(ctx, host=object(), on_error=on_error)
    await lifecycle.start()
    monkeypatch.setattr(ctx.storage, "setAlarm", fail_rearm)
    push = asyncio.create_task(
        lifecycle.jobs.push(
            LifecycleJobPushOptions(id="durable", fn="work", time=4_000)
        )
    )
    await gate.wait_until_blocked()

    push.cancel()
    gate.release()

    with pytest.raises(asyncio.CancelledError):
        await push
    assert reported == [rearm_failure]


@pytest.mark.asyncio
async def test_cancellation_during_failure_rearm_replaces_the_primary_error(
    monkeypatch,
):
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)
    gate = fakes.AsyncGate()
    ctx = fakes.FakeCtx()

    async def fail_alarm() -> None:
        raise RuntimeError("alarm failed")

    async def blocked_delete() -> None:
        await gate.block()

    lifecycle = Lifecycle(ctx, host=object(), on_alarm=fail_alarm)
    await lifecycle.start()
    monkeypatch.setattr(ctx.storage, "deleteAlarm", blocked_delete)
    alarm = asyncio.create_task(lifecycle.alarm())
    await gate.wait_until_blocked()

    alarm.cancel()
    await asyncio.sleep(0)

    assert not alarm.done()
    gate.release()
    with pytest.raises(asyncio.CancelledError):
        await alarm


@pytest.mark.asyncio
async def test_cancellation_during_secondary_error_reporting_is_preserved(
    monkeypatch,
):
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)
    gate = fakes.AsyncGate()
    ctx = fakes.FakeCtx()

    async def fail_alarm() -> None:
        raise RuntimeError("alarm failed")

    async def block_reporting(_error: BaseException) -> None:
        await gate.block()

    async def fail_rearm() -> None:
        raise RuntimeError("rearm failed")

    lifecycle = Lifecycle(
        ctx,
        host=object(),
        on_alarm=fail_alarm,
        on_error=block_reporting,
    )
    await lifecycle.start()
    monkeypatch.setattr(ctx.storage, "deleteAlarm", fail_rearm)
    alarm = asyncio.create_task(lifecycle.alarm())
    await gate.wait_until_blocked()

    alarm.cancel()

    with pytest.raises(asyncio.CancelledError):
        await alarm


@pytest.mark.asyncio
async def test_jobs_survive_eviction_and_drive_without_client_contact(monkeypatch):
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)
    runtime = fakes.FakeDurableObjectRuntime()
    driven: list[str] = []

    class DurableCapability(LifecycleCapability):
        capability_id = "durable"

        async def on_job(self, context: LifecycleJobContext):
            driven.append(context.job.id)

    first = Lifecycle(runtime.new_context(), host=object())
    first_capability = DurableCapability()
    first.use(first_capability)
    await first_capability.lifecycle.jobs.push(
        LifecycleJobPushOptions(id="durable", fn="work", time=100)
    )
    reincarnated = Lifecycle(runtime.evict(), host=object())
    reincarnated.use(DurableCapability())

    await reincarnated.alarm()

    assert driven == ["durable"]
    assert runtime._state.conn.execute("SELECT id FROM cf_agents_jobs").fetchall() == []


@pytest.mark.asyncio
async def test_disposal_rejects_later_job_and_alarm_operations(monkeypatch):
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)
    ctx = fakes.FakeCtx()
    lifecycle = Lifecycle(ctx, host=object())
    await lifecycle.start()
    await lifecycle.jobs.push(
        LifecycleJobPushOptions(id="future", fn="work", time=5_000)
    )
    history = tuple(ctx.storage.alarm_history_ms)

    await lifecycle.dispose()

    with pytest.raises(RuntimeError, match="disposed"):
        await lifecycle.jobs.push(
            LifecycleJobPushOptions(id="late", fn="work", time=2_000)
        )
    with pytest.raises(RuntimeError, match="disposed"):
        await lifecycle.rearm_alarm()
    with pytest.raises(RuntimeError, match="disposed"):
        await lifecycle.alarm()
    assert tuple(ctx.storage.alarm_history_ms) == history


@pytest.mark.asyncio
async def test_disposal_hook_cannot_leave_an_unarmed_job(monkeypatch):
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)

    class DisposalCapability(LifecycleCapability):
        capability_id = "disposal"

        async def on_dispose(self) -> None:
            with pytest.raises(RuntimeError, match="disposed"):
                await self.lifecycle.jobs.push(
                    LifecycleJobPushOptions(id="orphan", fn="work", time=5_000)
                )

    ctx = fakes.FakeCtx()
    lifecycle = Lifecycle(ctx, host=object())
    lifecycle.use(DisposalCapability())
    await lifecycle.start()

    await lifecycle.dispose()

    assert ctx.storage.sql.exec("SELECT id FROM cf_agents_jobs").toArray() == []
    assert ctx.storage.alarm_time_ms is None


@pytest.mark.asyncio
async def test_disposal_hook_can_only_cancel_owned_job_and_rearms_remaining_work(
    monkeypatch,
):
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)

    class DisposalCapability(LifecycleCapability):
        capability_id = "disposal"
        cancelled = False

        async def on_dispose(self) -> None:
            self.cancelled = await self.lifecycle.jobs.cancel("owned")
            with pytest.raises(RuntimeError, match="disposed"):
                await self.lifecycle.jobs.get("owned")
            with pytest.raises(RuntimeError, match="disposed"):
                await self.lifecycle.jobs.list()
            with pytest.raises(RuntimeError, match="disposed"):
                await self.lifecycle.jobs.reschedule("owned", 3_000)
            with pytest.raises(RuntimeError, match="disposed"):
                await self.lifecycle.jobs.push(
                    LifecycleJobPushOptions(id="late", fn="work", time=3_000)
                )

    ctx = fakes.FakeCtx()
    lifecycle = Lifecycle(ctx, host=object())
    capability = DisposalCapability()
    lifecycle.use(capability)
    await capability.lifecycle.jobs.push(
        LifecycleJobPushOptions(id="owned", fn="cleanup", time=2_000)
    )
    await lifecycle.jobs.push(
        LifecycleJobPushOptions(id="unrelated", fn="work", time=5_000)
    )

    await lifecycle.dispose()

    assert capability.cancelled is True
    assert ctx.storage.sql.exec(
        "SELECT id, capability FROM cf_agents_jobs"
    ).toArray() == [{"id": "unrelated", "capability": "host"}]
    assert ctx.storage.alarm_time_ms == 5_000


@pytest.mark.asyncio
async def test_cancel_missing_job_clears_stale_physical_alarm(monkeypatch):
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)
    ctx = fakes.FakeCtx()
    lifecycle = Lifecycle(ctx, host=object())
    await lifecycle.start()
    await ctx.storage.setAlarm(5_000)

    assert await lifecycle.jobs.cancel("missing") is False

    assert ctx.storage.alarm_time_ms is None
    assert ctx.storage.alarm_history_ms[-2:] == (
        ("set", 5_000),
        ("delete", None),
    )


@pytest.mark.asyncio
async def test_memory_limit_breaker_backs_off_then_seals_recovery_pack(monkeypatch):
    current_time = 1_000
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: current_time)
    ctx = fakes.FakeCtx()
    policies: list[tuple[str, LifecycleMemoryLimitContext]] = []
    resets: list[str] = []
    attempts: list[int] = []

    class RecoveryCapability(LifecycleCapability):
        capability_id = "recovery"

        async def on_job(self, context: LifecycleJobContext) -> None:
            attempts.append(context.attempt)
            raise RuntimeError(
                "Durable Object's isolate exceeded its memory limit and was reset."
            )

        async def on_memory_limit(
            self,
            context: LifecycleMemoryLimitContext,
        ) -> None:
            policies.append(("capability", context))

    async def host_policy(context: LifecycleMemoryLimitContext) -> None:
        policies.append(("host", context))

    def record_reset(reason: str) -> None:
        resets.append(reason)

    first = Lifecycle(
        ctx,
        host=object(),
        on_alarm_memory_limit=host_policy,
        max_alarm_memory_limit_strikes=2,
        reset_alarm=record_reset,
    )
    capability = RecoveryCapability()
    first.use(capability)
    await capability.lifecycle.jobs.push(
        LifecycleJobPushOptions(
            id="executing",
            fn="recover",
            time=100,
            retry={"maxAttempts": 2, "baseDelayMs": 1, "maxDelayMs": 1},
            recovery_loop=True,
        )
    )
    await capability.lifecycle.jobs.push(
        LifecycleJobPushOptions(
            id="sibling",
            fn="recover",
            time=200,
            retry={"maxAttempts": 2, "baseDelayMs": 1, "maxDelayMs": 1},
            recovery_loop=True,
        )
    )
    await capability.lifecycle.jobs.push(
        LifecycleJobPushOptions(
            id="corrupt",
            fn="recover",
            time=50_000,
            recovery_loop=True,
        )
    )
    ctx.storage.sql.exec("UPDATE cf_agents_jobs SET payload = '{' WHERE id = 'corrupt'")
    await first.jobs.push(
        LifecycleJobPushOptions(id="unrelated", fn="later", time=100_000)
    )

    await first.alarm()

    assert await ctx.storage.get("cf_agents:oom_alarm_strikes") == 1
    assert ctx.storage.sql.exec(
        "SELECT id, time FROM cf_agents_jobs ORDER BY id"
    ).toArray() == [
        {"id": "corrupt", "time": 50_000},
        {"id": "executing", "time": 31_000},
        {"id": "sibling", "time": 31_000},
        {"id": "unrelated", "time": 100_000},
    ]
    assert [name for name, _context in policies] == ["capability", "host"]
    assert not policies[0][1].sealed
    assert policies[0][1].next_time == 31_000
    assert policies[0][1].executing is not None
    assert policies[0][1].executing.id == "executing"
    assert policies[0][1].purged_recovery_loop_jobs == ()
    assert ctx.storage.sync_calls == 1
    assert len(resets) == 1
    assert attempts == [1, 2]

    current_time = 32_000
    policies.clear()
    second = Lifecycle(
        ctx,
        host=object(),
        on_alarm_memory_limit=host_policy,
        max_alarm_memory_limit_strikes=2,
        reset_alarm=record_reset,
    )
    second.use(RecoveryCapability())

    await second.alarm()

    assert await ctx.storage.get("cf_agents:oom_alarm_strikes") is None
    assert ctx.storage.sql.exec("SELECT id, time FROM cf_agents_jobs").toArray() == [
        {"id": "unrelated", "time": 100_000}
    ]
    assert [name for name, _context in policies] == ["capability", "host"]
    sealed = policies[0][1]
    assert sealed.sealed
    assert sealed.next_time is None
    assert {job.id for job in sealed.purged_recovery_loop_jobs} == {
        "executing",
        "sibling",
    }
    assert ctx.storage.alarm_time_ms == 100_000
    assert ctx.storage.sync_calls == 2
    assert len(resets) == 2
    assert attempts == [1, 2, 1, 2]


@pytest.mark.asyncio
async def test_memory_limit_strikes_continue_when_reset_callback_returns(monkeypatch):
    current_time = 1_000
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: current_time)
    resets: list[str] = []

    async def on_job(_context: LifecycleJobContext) -> None:
        raise RuntimeError("isolate exceeded its memory limit")

    ctx = fakes.FakeCtx()
    lifecycle = Lifecycle(
        ctx,
        host=object(),
        on_job=on_job,
        reset_alarm=resets.append,
    )
    await lifecycle.jobs.push(
        LifecycleJobPushOptions(
            id="memory",
            fn="work",
            time=100,
            retry={"maxAttempts": 1},
        )
    )

    await lifecycle.alarm()
    current_time = 32_000
    await lifecycle.alarm()

    assert await ctx.storage.get("cf_agents:oom_alarm_strikes") == 2
    assert ctx.storage.sql.exec(
        "SELECT time FROM cf_agents_jobs WHERE id = 'memory'"
    ).toArray() == [{"time": 92_000}]
    assert resets == [
        "alarm memory-limit strike 1/3",
        "alarm memory-limit strike 2/3",
    ]


@pytest.mark.asyncio
async def test_tracked_handoff_keeps_memory_attribution_and_deduplicates_strike(
    monkeypatch,
):
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)
    ctx = fakes.FakeCtx()
    await ctx.storage.put("cf_agents:oom_alarm_strikes", 1)
    gate = fakes.AsyncGate()
    recorder = fakes.WaitUntilRecorder()
    tracked: list[bool] = []
    resets: list[str] = []
    lifecycle: Lifecycle

    async def fail_after_handoff() -> None:
        await gate.block()
        raise RuntimeError("isolate exceeded its memory limit")

    async def on_job(_context: LifecycleJobContext) -> str:
        work = fail_after_handoff()
        tracked.append(lifecycle.track_alarm_work(work))
        tracked.append(lifecycle.track_alarm_work(work))
        return "yield"

    lifecycle = Lifecycle(
        ctx,
        host=object(),
        on_job=on_job,
        retain_work=recorder,
        reset_alarm=resets.append,
    )
    outside = asyncio.get_running_loop().create_future()
    outside.set_result(None)
    assert not lifecycle.track_alarm_work(outside)
    await lifecycle.jobs.push(
        LifecycleJobPushOptions(
            id="tracked",
            fn="work",
            time=100,
            recovery_loop=True,
        )
    )

    await lifecycle.alarm()

    assert tracked == [True, True]
    assert await ctx.storage.get("cf_agents:oom_alarm_strikes") == 1
    gate.release()
    await recorder.drain_next()

    assert len(resets) == 1
    assert await ctx.storage.get("cf_agents:oom_alarm_strikes") == 2
    row = ctx.storage.sql.exec(
        "SELECT time, running FROM cf_agents_jobs WHERE id = 'tracked'"
    ).toArray()[0]
    assert row == {"time": 61_000, "running": 0}


@pytest.mark.asyncio
async def test_alarm_tracking_rejects_handoff_without_retained_work(monkeypatch):
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)
    accepted: list[bool] = []
    lifecycle: Lifecycle

    async def work() -> None:
        pass

    async def on_job(_context: LifecycleJobContext) -> None:
        awaitable = work()
        accepted.append(lifecycle.track_alarm_work(awaitable))
        awaitable.close()

    lifecycle = Lifecycle(fakes.FakeCtx(), host=object(), on_job=on_job)
    await lifecycle.jobs.push(
        LifecycleJobPushOptions(id="tracked", fn="work", time=100)
    )

    await lifecycle.alarm()

    assert accepted == [False]


@pytest.mark.asyncio
async def test_clean_tracked_handoffs_clear_strikes_only_after_all_settle(monkeypatch):
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)
    ctx = fakes.FakeCtx()
    await ctx.storage.put("cf_agents:oom_alarm_strikes", 1)
    recorder = fakes.WaitUntilRecorder()
    first = fakes.AsyncGate()
    second = fakes.AsyncGate()
    lifecycle: Lifecycle

    async def wait_for(gate: fakes.AsyncGate) -> None:
        await gate.block()

    async def on_job(_context: LifecycleJobContext) -> None:
        assert lifecycle.track_alarm_work(wait_for(first))
        assert lifecycle.track_alarm_work(wait_for(second))

    lifecycle = Lifecycle(
        ctx,
        host=object(),
        on_job=on_job,
        retain_work=recorder,
    )
    await lifecycle.jobs.push(
        LifecycleJobPushOptions(id="tracked", fn="work", time=100)
    )

    await lifecycle.alarm()
    assert await ctx.storage.get("cf_agents:oom_alarm_strikes") == 1

    first.release()
    await recorder.drain_next()
    assert await ctx.storage.get("cf_agents:oom_alarm_strikes") == 1

    second.release()
    await recorder.drain_next()
    assert await ctx.storage.get("cf_agents:oom_alarm_strikes") is None


@pytest.mark.asyncio
async def test_early_tracked_failure_is_reported_and_does_not_clear_strikes(
    monkeypatch,
):
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)
    ctx = fakes.FakeCtx()
    await ctx.storage.put("cf_agents:oom_alarm_strikes", 1)
    reported = asyncio.Event()
    errors: list[BaseException] = []
    retained: list[asyncio.Task[object]] = []
    lifecycle: Lifecycle

    def retain(awaitable):
        retained.append(asyncio.create_task(awaitable))

    async def fail() -> None:
        raise ValueError("tracked failure")

    async def on_error(error: BaseException) -> None:
        errors.append(error)
        reported.set()

    async def on_job(_context: LifecycleJobContext) -> None:
        assert lifecycle.track_alarm_work(fail())
        await reported.wait()

    lifecycle = Lifecycle(
        ctx,
        host=object(),
        on_job=on_job,
        on_error=on_error,
        retain_work=retain,
    )
    await lifecycle.jobs.push(
        LifecycleJobPushOptions(id="tracked", fn="work", time=100)
    )

    await lifecycle.alarm()
    await asyncio.gather(*retained)

    assert [str(error) for error in errors] == ["tracked failure"]
    assert await ctx.storage.get("cf_agents:oom_alarm_strikes") == 1


@pytest.mark.asyncio
async def test_tracking_a_settled_awaitable_again_is_a_no_op(monkeypatch):
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)
    retained: list[asyncio.Task[object]] = []
    settled = asyncio.Event()
    lifecycle: Lifecycle

    def retain(awaitable):
        async def observe() -> object:
            result = await awaitable
            settled.set()
            return result

        retained.append(asyncio.create_task(observe()))

    async def on_job(_context: LifecycleJobContext) -> None:
        completed = asyncio.get_running_loop().create_future()
        completed.set_result(None)
        assert lifecycle.track_alarm_work(completed)
        await settled.wait()
        assert lifecycle.track_alarm_work(completed)

    lifecycle = Lifecycle(
        fakes.FakeCtx(),
        host=object(),
        on_job=on_job,
        retain_work=retain,
    )
    await lifecycle.jobs.push(
        LifecycleJobPushOptions(id="tracked", fn="work", time=100)
    )

    await lifecycle.alarm()
    await asyncio.gather(*retained)

    assert len(retained) == 1


@pytest.mark.asyncio
async def test_overlapping_alarms_track_one_canonical_awaitable_once(monkeypatch):
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)
    recorder = fakes.WaitUntilRecorder()
    first_gate = fakes.AsyncGate()
    second_gate = fakes.AsyncGate()
    canonical = asyncio.get_running_loop().create_future()
    tracked: list[bool] = []
    calls = 0
    lifecycle: Lifecycle

    async def on_job(_context: LifecycleJobContext) -> None:
        nonlocal calls
        calls += 1
        tracked.append(lifecycle.track_alarm_work(canonical))
        if calls == 1:
            await first_gate.block()
        else:
            await second_gate.block()

    lifecycle = Lifecycle(
        fakes.FakeCtx(),
        host=object(),
        on_job=on_job,
        retain_work=recorder,
    )
    await lifecycle.jobs.push(LifecycleJobPushOptions(id="shared", fn="work", time=100))

    first_alarm = asyncio.create_task(lifecycle.alarm())
    await first_gate.wait_until_blocked()
    second_alarm = asyncio.create_task(lifecycle.alarm())
    await second_gate.wait_until_blocked()

    assert tracked == [True, True]
    assert len(recorder.coros) == 1
    first_gate.release()
    second_gate.release()
    await asyncio.gather(first_alarm, second_alarm)
    canonical.set_result(None)
    await recorder.drain()


@pytest.mark.asyncio
@pytest.mark.parametrize("strike_limit", [1, 3])
async def test_breaker_does_not_overwrite_newer_job_intent(
    monkeypatch,
    strike_limit,
):
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)
    ctx = fakes.FakeCtx()
    lifecycle: Lifecycle

    async def on_job(_context: LifecycleJobContext) -> None:
        await lifecycle.jobs.push(
            LifecycleJobPushOptions(
                id="replace-me",
                fn="older",
                time=100,
                retry={"maxAttempts": 1},
            )
        )
        raise RuntimeError("isolate exceeded its memory limit")

    lifecycle = Lifecycle(
        ctx,
        host=object(),
        on_job=on_job,
        max_alarm_memory_limit_strikes=strike_limit,
        reset_alarm=lambda _reason: None,
    )
    await lifecycle.jobs.push(
        LifecycleJobPushOptions(
            id="replace-me",
            fn="older",
            time=100,
            retry={"maxAttempts": 1},
        )
    )

    await lifecycle.alarm()

    replacement = await lifecycle.jobs.get("replace-me")
    assert replacement is not None
    assert replacement.fn == "older"
    assert replacement.time == 100
    assert not replacement.recovery_loop


@pytest.mark.asyncio
@pytest.mark.parametrize("strike_limit", [1, 3])
async def test_tracked_breaker_does_not_overwrite_later_replacement(
    monkeypatch,
    strike_limit,
):
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)
    ctx = fakes.FakeCtx()
    recorder = fakes.WaitUntilRecorder()
    gate = fakes.AsyncGate()
    lifecycle: Lifecycle

    async def fail_later() -> None:
        await gate.block()
        raise RuntimeError("isolate exceeded its memory limit")

    async def on_job(_context: LifecycleJobContext) -> str:
        assert lifecycle.track_alarm_work(fail_later())
        return "yield"

    lifecycle = Lifecycle(
        ctx,
        host=object(),
        on_job=on_job,
        retain_work=recorder,
        max_alarm_memory_limit_strikes=strike_limit,
        reset_alarm=lambda _reason: None,
    )
    await lifecycle.jobs.push(
        LifecycleJobPushOptions(
            id="replace-me",
            fn="older",
            time=100,
        )
    )
    await lifecycle.alarm()
    await lifecycle.jobs.push(
        LifecycleJobPushOptions(
            id="replace-me",
            fn="older",
            time=100,
        )
    )

    gate.release()
    await recorder.drain_next()

    replacement = await lifecycle.jobs.get("replace-me")
    assert replacement is not None
    assert replacement.fn == "older"
    assert replacement.time == 100
    assert not replacement.recovery_loop


@pytest.mark.asyncio
async def test_tracked_handoff_uses_executing_version_when_callback_pushes(
    monkeypatch,
):
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)
    recorder = fakes.WaitUntilRecorder()
    gate = fakes.AsyncGate()
    lifecycle: Lifecycle

    async def fail_later() -> None:
        await gate.block()
        raise RuntimeError("isolate exceeded its memory limit")

    async def on_job(_context: LifecycleJobContext) -> None:
        await lifecycle.jobs.push(
            LifecycleJobPushOptions(id="replace-me", fn="same", time=100)
        )
        assert lifecycle.track_alarm_work(fail_later())

    lifecycle = Lifecycle(
        fakes.FakeCtx(),
        host=object(),
        on_job=on_job,
        retain_work=recorder,
        max_alarm_memory_limit_strikes=1,
        reset_alarm=lambda _reason: None,
    )
    await lifecycle.jobs.push(
        LifecycleJobPushOptions(id="replace-me", fn="same", time=100)
    )

    await lifecycle.alarm()
    gate.release()
    await recorder.drain()

    replacement = await lifecycle.jobs.get("replace-me")
    assert replacement is not None
    assert replacement.fn == "same"
    assert replacement.time == 100


@pytest.mark.asyncio
async def test_tracked_memory_failure_cannot_overwrite_newer_active_claim(monkeypatch):
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)
    recorder = fakes.WaitUntilRecorder()
    handoff_gate = fakes.AsyncGate()
    newer_claim = fakes.AsyncGate()
    calls = 0
    lifecycle: Lifecycle

    async def fail_later() -> None:
        await handoff_gate.block()
        raise RuntimeError("isolate exceeded its memory limit")

    async def on_job(_context: LifecycleJobContext) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            assert lifecycle.track_alarm_work(fail_later())
            return "yield"
        await newer_claim.block()
        return "yield"

    ctx = fakes.FakeCtx()
    lifecycle = Lifecycle(
        ctx,
        host=object(),
        on_job=on_job,
        retain_work=recorder,
        max_alarm_memory_limit_strikes=1,
        reset_alarm=lambda _reason: None,
    )
    await lifecycle.jobs.push(LifecycleJobPushOptions(id="shared", fn="work", time=100))
    await lifecycle.alarm()

    active_alarm = asyncio.create_task(lifecycle.alarm())
    await newer_claim.wait_until_blocked()
    handoff_gate.release()
    await recorder.drain()

    active = ctx.storage.sql.exec(
        "SELECT running, execution_started_at FROM cf_agents_jobs WHERE id = 'shared'"
    ).toArray()
    assert active == [{"running": 1, "execution_started_at": 1_001}]

    newer_claim.release()
    await active_alarm


@pytest.mark.asyncio
async def test_memory_limit_default_reset_disables_alarm_retry(monkeypatch):
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)

    async def on_job(_context: LifecycleJobContext) -> None:
        raise RuntimeError("isolate exceeded its memory limit")

    raw_ctx = fakes.FakeCtx()
    wrapper_abort_calls: list[str] = []

    class WrappedContext:
        _ctx = raw_ctx
        storage = raw_ctx.storage

        async def blockConcurrencyWhile(self, callback):
            return await callback()

        def abort(self, reason: str) -> None:
            wrapper_abort_calls.append(reason)

    ctx = WrappedContext()
    lifecycle = Lifecycle(ctx, host=object(), on_job=on_job)
    await lifecycle.jobs.push(
        LifecycleJobPushOptions(
            id="memory",
            fn="work",
            time=100,
            retry={"maxAttempts": 1},
        )
    )

    await lifecycle.alarm()
    await asyncio.sleep(0)

    assert wrapper_abort_calls == []
    assert len(raw_ctx.abort_calls) == 1
    reason, options = raw_ctx.abort_calls[0]
    assert reason == "alarm memory-limit strike 1/3"
    assert options == {"retryAlarm": False}


@pytest.mark.asyncio
async def test_cancellation_while_clearing_strikes_rearms_and_propagates(monkeypatch):
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)
    ctx = fakes.FakeCtx()
    await ctx.storage.put("cf_agents:oom_alarm_strikes", 1)
    gate = fakes.AsyncGate()
    original_get = ctx.storage.get

    async def blocked_get(key: str):
        if key == "cf_agents:oom_alarm_strikes":
            await gate.block()
        return await original_get(key)

    monkeypatch.setattr(ctx.storage, "get", blocked_get)
    lifecycle = Lifecycle(ctx, host=object())
    alarm = asyncio.create_task(lifecycle.alarm())
    await gate.wait_until_blocked()
    alarm.cancel()

    with pytest.raises(asyncio.CancelledError):
        await alarm

    assert ctx.storage.alarm_time_ms is None


@pytest.mark.asyncio
async def test_cancellation_during_memory_policy_finishes_breaker_then_propagates(
    monkeypatch,
):
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)
    policy_gate = fakes.AsyncGate()
    resets: list[str] = []

    async def on_job(_context: LifecycleJobContext) -> None:
        raise RuntimeError("isolate exceeded its memory limit")

    async def on_memory_limit(_context: LifecycleMemoryLimitContext) -> None:
        await policy_gate.block()

    ctx = fakes.FakeCtx()
    lifecycle = Lifecycle(
        ctx,
        host=object(),
        on_job=on_job,
        on_alarm_memory_limit=on_memory_limit,
        reset_alarm=resets.append,
    )
    await lifecycle.jobs.push(
        LifecycleJobPushOptions(
            id="memory",
            fn="work",
            time=100,
            retry={"maxAttempts": 1},
        )
    )

    alarm = asyncio.create_task(lifecycle.alarm())
    await policy_gate.wait_until_blocked()
    alarm.cancel()

    with pytest.raises(asyncio.CancelledError):
        await alarm

    assert resets == ["alarm memory-limit strike 1/3"]
    assert await ctx.storage.get("cf_agents:oom_alarm_strikes") == 1
    assert ctx.storage.sql.exec(
        "SELECT time, running FROM cf_agents_jobs WHERE id = 'memory'"
    ).toArray() == [{"time": 31_000, "running": 0}]
    assert ctx.storage.alarm_time_ms == 31_000


@pytest.mark.asyncio
async def test_cancellation_during_strike_write_waits_for_persistence(monkeypatch):
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)
    strike_read = fakes.AsyncGate()
    original_get = fakes.FakeStorage.get
    resets: list[str] = []

    async def blocked_get(storage, key):
        if key == "cf_agents:oom_alarm_strikes":
            await strike_read.block()
        return await original_get(storage, key)

    monkeypatch.setattr(fakes.FakeStorage, "get", blocked_get)

    async def on_job(_context: LifecycleJobContext) -> None:
        raise RuntimeError("isolate exceeded its memory limit")

    ctx = fakes.FakeCtx()
    lifecycle = Lifecycle(
        ctx,
        host=object(),
        on_job=on_job,
        reset_alarm=resets.append,
    )
    await lifecycle.jobs.push(
        LifecycleJobPushOptions(
            id="memory",
            fn="work",
            time=100,
            retry={"maxAttempts": 1},
        )
    )

    alarm = asyncio.create_task(lifecycle.alarm())
    await strike_read.wait_until_blocked()
    alarm.cancel()
    await asyncio.sleep(0)
    assert not alarm.done()
    strike_read.release()

    with pytest.raises(asyncio.CancelledError):
        await alarm

    assert await ctx.storage.get("cf_agents:oom_alarm_strikes") == 1
    assert resets == ["alarm memory-limit strike 1/3"]


@pytest.mark.asyncio
async def test_clean_alarm_cannot_erase_overlapping_memory_strike(monkeypatch):
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)
    clean_read = fakes.AsyncGate()
    original_get = fakes.FakeStorage.get
    blocked = False

    async def blocked_once(storage, key):
        nonlocal blocked
        if key == "cf_agents:oom_alarm_strikes" and not blocked:
            blocked = True
            await clean_read.block()
        return await original_get(storage, key)

    monkeypatch.setattr(fakes.FakeStorage, "get", blocked_once)

    async def on_job(_context: LifecycleJobContext) -> None:
        raise RuntimeError("isolate exceeded its memory limit")

    ctx = fakes.FakeCtx()
    await ctx.storage.put("cf_agents:oom_alarm_strikes", 1)
    lifecycle = Lifecycle(
        ctx,
        host=object(),
        on_job=on_job,
        reset_alarm=lambda _reason: None,
    )
    clean_alarm = asyncio.create_task(lifecycle.alarm())
    await clean_read.wait_until_blocked()
    await lifecycle.jobs.push(
        LifecycleJobPushOptions(
            id="memory",
            fn="work",
            time=100,
            retry={"maxAttempts": 1},
        )
    )
    memory_alarm = asyncio.create_task(lifecycle.alarm())
    await asyncio.sleep(0)
    clean_read.release()

    await asyncio.gather(clean_alarm, memory_alarm)

    assert await ctx.storage.get("cf_agents:oom_alarm_strikes") == 1


@pytest.mark.asyncio
async def test_stale_clean_alarm_cannot_clear_newer_memory_generation(monkeypatch):
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)
    host_gate = fakes.AsyncGate()
    host_calls = 0

    async def on_alarm() -> None:
        nonlocal host_calls
        host_calls += 1
        await host_gate.block()

    async def on_job(_context: LifecycleJobContext) -> None:
        raise RuntimeError("isolate exceeded its memory limit")

    ctx = fakes.FakeCtx()
    lifecycle = Lifecycle(
        ctx,
        host=object(),
        on_job=on_job,
        on_alarm=on_alarm,
        reset_alarm=lambda _reason: None,
    )
    stale_clean_alarm = asyncio.create_task(lifecycle.alarm())
    await host_gate.wait_until_blocked()
    await lifecycle.jobs.push(
        LifecycleJobPushOptions(
            id="memory",
            fn="work",
            time=100,
            retry={"maxAttempts": 1},
        )
    )

    await lifecycle.alarm()
    assert await ctx.storage.get("cf_agents:oom_alarm_strikes") == 1
    host_gate.release()
    await stale_clean_alarm

    assert host_calls == 1
    assert await ctx.storage.get("cf_agents:oom_alarm_strikes") == 1


@pytest.mark.asyncio
async def test_cancellation_waiting_for_strike_lock_still_finishes_breaker(monkeypatch):
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)
    first_strike = fakes.AsyncGate()
    second_dispatched = asyncio.Event()
    original_get = fakes.FakeStorage.get
    blocked = False
    resets: list[str] = []
    capability_policies: list[bool] = []
    host_policies: list[bool] = []

    class PolicyCapability(LifecycleCapability):
        capability_id = "policy"

        async def on_memory_limit(
            self,
            context: LifecycleMemoryLimitContext,
        ) -> None:
            capability_policies.append(context.sealed)

    async def blocked_once(storage, key):
        nonlocal blocked
        if key == "cf_agents:oom_alarm_strikes" and not blocked:
            blocked = True
            await first_strike.block()
        return await original_get(storage, key)

    monkeypatch.setattr(fakes.FakeStorage, "get", blocked_once)

    async def on_job(context: LifecycleJobContext) -> None:
        if context.job.id == "second":
            second_dispatched.set()
        raise RuntimeError("isolate exceeded its memory limit")

    async def host_policy(context: LifecycleMemoryLimitContext) -> None:
        host_policies.append(context.sealed)

    ctx = fakes.FakeCtx()
    lifecycle = Lifecycle(
        ctx,
        host=object(),
        on_job=on_job,
        on_alarm_memory_limit=host_policy,
        max_alarm_memory_limit_strikes=2,
        reset_alarm=resets.append,
    )
    lifecycle.use(PolicyCapability())
    await lifecycle.jobs.push(
        LifecycleJobPushOptions(
            id="first",
            fn="work",
            time=100,
            retry={"maxAttempts": 1},
            singleflight=True,
        )
    )
    await lifecycle.jobs.push(
        LifecycleJobPushOptions(
            id="second",
            fn="work",
            time=200,
            retry={"maxAttempts": 1},
        )
    )

    first_alarm = asyncio.create_task(lifecycle.alarm())
    await first_strike.wait_until_blocked()
    second_alarm = asyncio.create_task(lifecycle.alarm())
    await second_dispatched.wait()
    await asyncio.sleep(0)
    second_alarm.cancel()
    await asyncio.sleep(0)
    assert not second_alarm.done()
    first_strike.release()

    await first_alarm
    with pytest.raises(asyncio.CancelledError):
        await second_alarm

    assert await ctx.storage.get("cf_agents:oom_alarm_strikes") is None
    assert ctx.storage.sql.exec(
        "SELECT id, time, running FROM cf_agents_jobs ORDER BY id"
    ).toArray() == [{"id": "first", "time": 31_000, "running": 0}]
    assert resets == [
        "alarm memory-limit strike 1/2",
        "alarm memory-limit strike 2/2 (sealed)",
    ]
    assert capability_policies == [False, True]
    assert host_policies == [False, True]


@pytest.mark.asyncio
async def test_cancellation_during_memory_policy_reporting_propagates(monkeypatch):
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)
    report_gate = fakes.AsyncGate()
    resets: list[str] = []
    reported: list[str] = []

    class PolicyCapability(LifecycleCapability):
        capability_id = "policy"

        async def on_memory_limit(
            self,
            _context: LifecycleMemoryLimitContext,
        ) -> None:
            raise ValueError("policy failed")

    async def on_job(_context: LifecycleJobContext) -> None:
        raise RuntimeError("isolate exceeded its memory limit")

    async def on_error(error: BaseException) -> None:
        reported.append(str(error))
        await report_gate.block()

    ctx = fakes.FakeCtx()
    lifecycle = Lifecycle(
        ctx,
        host=object(),
        on_job=on_job,
        on_error=on_error,
        reset_alarm=resets.append,
    )
    lifecycle.use(PolicyCapability())
    await lifecycle.jobs.push(
        LifecycleJobPushOptions(
            id="memory",
            fn="work",
            time=100,
            retry={"maxAttempts": 1},
        )
    )

    alarm = asyncio.create_task(lifecycle.alarm())
    await report_gate.wait_until_blocked()
    alarm.cancel()

    with pytest.raises(asyncio.CancelledError):
        await alarm

    assert reported == ["policy failed"]
    assert resets == ["alarm memory-limit strike 1/3"]
    assert ctx.storage.alarm_time_ms == 31_000
