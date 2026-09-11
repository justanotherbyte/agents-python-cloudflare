from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime

import fakes
import pytest

import agents.lifecycle._runtime as lifecycle_module
import agents.schedules as schedules_module
from agents.lifecycle import (
    Lifecycle,
    LifecycleMemoryLimitContext,
    LifecycleRouteAddress,
    LifecycleRouteEnvelope,
    get_current_lifecycle_context,
)
from agents.lifecycle.jobs import LifecycleJob
from agents.schedules import (
    RetryOptions,
    ScheduleCriteria,
    ScheduleOptions,
    ScheduleTimeRange,
    Scheduler,
)
from agents.core.utils import MISSING


async def _scheduler(
    monkeypatch,
    callbacks,
    *,
    now=1_700_000_000_000,
    retry=None,
    on_error=None,
):
    monkeypatch.setattr(schedules_module, "now_ms", lambda: now)
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: now)
    ctx = fakes.FakeCtx()
    host = object()
    events = []
    scheduler = Scheduler(callbacks, retry=retry, on_error=on_error)
    lifecycle = Lifecycle(ctx, host=host, event_listeners=(events.append,))
    lifecycle.use(scheduler)
    await lifecycle.start()
    return scheduler, lifecycle, ctx, host, events


@pytest.mark.asyncio
async def test_delay_date_cron_and_interval_use_shared_job_storage(monkeypatch):
    scheduler, _, ctx, _, _ = await _scheduler(
        monkeypatch,
        {"run": lambda payload, schedule: None},
        now=1_704_067_200_500,
    )

    delayed = await scheduler.set(2.25, "run", {"kind": "delay"})
    dated = await scheduler.set(
        datetime(2024, 1, 1, 0, 1, 30, 900_000, tzinfo=UTC),
        "run",
        {"kind": "date"},
    )
    cron = await scheduler.set("*/5 * * * *", "run", {"kind": "cron"})
    interval = await scheduler.every(12.5, "run", {"kind": "interval"})

    assert delayed.type == "delayed"
    assert delayed.time == 1_704_067_202
    assert delayed.delay_in_seconds == 2.25
    assert dated.type == "scheduled"
    assert dated.time == 1_704_067_290
    assert cron.type == "cron"
    assert cron.time == 1_704_067_500
    assert interval.type == "interval"
    assert interval.time == 1_704_067_213
    assert interval.interval_seconds == 12.5
    assert all(len(schedule.id) == 9 for schedule in (delayed, dated, cron, interval))

    rows = ctx.storage.sql.exec(
        "SELECT capability, fn, time, singleflight, hung_timeout_seconds "
        "FROM cf_agents_jobs ORDER BY time"
    ).toArray()
    assert {row["capability"] for row in rows} == {"scheduler"}
    assert {row["fn"] for row in rows} == {"run"}
    assert {row["hung_timeout_seconds"] for row in rows} == {30}
    assert [row["singleflight"] for row in rows].count(1) == 1
    tables = ctx.storage.sql.exec(
        "SELECT name FROM sqlite_master WHERE name = 'cf_agents_schedules'"
    ).toArray()
    assert tables == []


@pytest.mark.asyncio
async def test_schedule_payload_rejects_non_javascript_safe_integers(monkeypatch):
    scheduler, _, _, _, _ = await _scheduler(
        monkeypatch,
        {"run": lambda payload, schedule: None},
    )

    with pytest.raises(ValueError, match="JavaScript-safe"):
        await scheduler.set(1, "run", {"nested": [2**53]})

    await scheduler.every(10, "run", None)
    with pytest.raises(ValueError, match="JavaScript-safe"):
        await scheduler.every(10, "run", 10**400)


@pytest.mark.asyncio
async def test_sealed_recovery_loop_notifies_every_reachable_facet_owner():
    root_address = LifecycleRouteAddress("root", '[{"name":"root"}]')
    owner_data, owner_key = schedules_module._canonical_agent_owner_path(
        '[{"className":"Agent","name":"facet"}]'
    )
    facet_address = LifecycleRouteAddress(owner_key, owner_data)
    lifecycles = {}

    async def transport(envelope: LifecycleRouteEnvelope):
        lifecycle = lifecycles[envelope.target.key]
        return await lifecycle.route(
            version=envelope.version,
            source=envelope.source,
            target=envelope.target,
            capability_id=envelope.capability_id,
            payload=envelope.payload,
        )

    root_scheduler = Scheduler({})
    root_lifecycle = Lifecycle(
        fakes.FakeCtx(),
        host=object(),
        root_route_address=root_address,
        route_transport=transport,
        owns_physical_alarm=True,
    )
    root_lifecycle.use(root_scheduler)
    facet_policies = []
    facet_scheduler = Scheduler({})
    facet_lifecycle = Lifecycle(
        fakes.FakeCtx(),
        host=object(),
        route_address=facet_address,
        root_route_address=root_address,
        route_transport=transport,
        owns_physical_alarm=False,
        on_alarm_memory_limit=facet_policies.append,
    )
    facet_lifecycle.use(facet_scheduler)
    lifecycles[root_address.key] = root_lifecycle
    lifecycles[facet_address.key] = facet_lifecycle
    purged = LifecycleJob(
        id="legacy-recovery",
        capability="scheduler",
        fn="_chatRecoveryContinue",
        time=1_000,
        payload={
            "type": "delayed",
            "delayInSeconds": 1,
            "owner_path": owner_data,
            "owner_path_key": owner_key,
        },
        payload_present=True,
        retry=None,
        singleflight=False,
        exclusive=False,
        recovery_loop=True,
        created_at=1_000,
    )
    bad_data, bad_key = schedules_module._canonical_agent_owner_path(
        '[{"className":"Agent","name":"missing"}]'
    )
    executing = LifecycleJob(
        id="tracked-recovery",
        capability="scheduler",
        fn="_chatRecoveryRetry",
        time=1_000,
        payload={
            "type": "delayed",
            "delayInSeconds": 1,
            "owner_path": bad_data,
            "owner_path_key": bad_key,
        },
        payload_present=True,
        retry=None,
        singleflight=False,
        exclusive=False,
        recovery_loop=True,
        created_at=1_000,
    )

    with pytest.raises(KeyError):
        await root_scheduler.on_memory_limit(
            LifecycleMemoryLimitContext(
                sealed=True,
                next_time=None,
                executing=executing,
                purged_recovery_loop_jobs=(purged,),
            )
        )

    assert len(facet_policies) == 1
    assert facet_policies[0].sealed
    assert facet_policies[0].executing is None


@pytest.mark.asyncio
async def test_payload_presence_and_retry_projection_are_exact(monkeypatch):
    scheduler, _, ctx, _, _ = await _scheduler(
        monkeypatch,
        {"run": lambda payload, schedule: None},
        retry=RetryOptions(max_attempts=4),
    )
    options = ScheduleOptions(
        retry=RetryOptions(base_delay_ms=250),
        idempotent=False,
    )

    omitted = await scheduler.set(1, "run", options=options)
    explicit_null = await scheduler.set(1, "run", None, options)

    rows = ctx.storage.sql.exec(
        "SELECT id, payload, retry_options FROM cf_agents_jobs ORDER BY id"
    ).toArray()
    by_id = {row["id"]: row for row in rows}
    omitted_payload = json.loads(by_id[omitted.id]["payload"])
    null_payload = json.loads(by_id[explicit_null.id]["payload"])
    assert "payload" not in omitted_payload
    assert null_payload["payload"] is None
    assert omitted_payload["retry"] == {"baseDelayMs": 250}
    assert json.loads(by_id[omitted.id]["retry_options"]) == {
        "maxAttempts": 4,
        "baseDelayMs": 250,
        "maxDelayMs": 3000,
    }
    assert omitted.retry == RetryOptions(base_delay_ms=250)
    assert omitted.payload is MISSING
    assert explicit_null.payload is None


@pytest.mark.asyncio
async def test_deduplication_defaults_and_matching_axes(monkeypatch):
    scheduler, _, ctx, _, events = await _scheduler(
        monkeypatch,
        {"run": lambda payload, schedule: None},
    )

    first = await scheduler.every(30, "run", {"key": 1})
    duplicate = await scheduler.every(
        30,
        "run",
        {"key": 1},
        ScheduleOptions(retry=RetryOptions(max_attempts=8)),
    )
    changed_interval = await scheduler.every(31, "run", {"key": 1})
    one_shot = await scheduler.set(10, "run", {"key": 1})
    second_one_shot = await scheduler.set(20, "run", {"key": 1})
    idempotent = await scheduler.set(
        30,
        "run",
        {"same": True},
        ScheduleOptions(idempotent=True),
    )
    joined = await scheduler.set(
        90,
        "run",
        {"same": True},
        ScheduleOptions(idempotent=True),
    )

    assert duplicate.id == first.id
    assert duplicate.retry is None
    assert changed_interval.id != first.id
    assert second_one_shot.id != one_shot.id
    assert joined.id == idempotent.id
    assert joined.time == idempotent.time
    assert len(ctx.storage.sql.exec("SELECT id FROM cf_agents_jobs").toArray()) == 5
    assert [event.type for event in events].count("schedule:create") == 5


@pytest.mark.asyncio
async def test_dedup_hit_rearms_a_lost_alarm(monkeypatch):
    scheduler, _, ctx, _, _ = await _scheduler(
        monkeypatch,
        {"run": lambda payload, schedule: None},
    )
    original = await scheduler.every(30, "run")
    await ctx.storage.deleteAlarm()

    duplicate = await scheduler.every(30, "run")

    assert duplicate.id == original.id
    assert ctx.storage.alarm_time_ms == original.time * 1_000


@pytest.mark.asyncio
async def test_deduplication_uses_javascript_json_semantics(monkeypatch):
    scheduler, _, ctx, _, _ = await _scheduler(
        monkeypatch,
        {"run": lambda payload, schedule: None},
    )

    first = await scheduler.every(30, "run", {"2": -0.0, "1": 1.0})
    duplicate = await scheduler.every(30, "run", {"1": 1, "2": 0})
    long_key = "9" * 5_000
    long_key_first = await scheduler.every(40, "run", {long_key: 1})
    long_key_duplicate = await scheduler.every(40, "run", {long_key: 1.0})

    assert duplicate.id == first.id
    assert long_key_duplicate.id == long_key_first.id
    assert len(ctx.storage.sql.exec("SELECT id FROM cf_agents_jobs").toArray()) == 2


@pytest.mark.asyncio
async def test_get_list_criteria_and_cancel_are_owner_scoped(monkeypatch):
    scheduler, _, _, _, events = await _scheduler(
        monkeypatch,
        {"run": lambda payload, schedule: None},
        now=1_700_000_000_000,
    )
    first = await scheduler.set(10, "run")
    second = await scheduler.every(20, "run")
    await scheduler.set(30, "run")

    assert await scheduler.get(first.id) == first
    assert await scheduler.get("missing") is None
    assert await scheduler.list(
        ScheduleCriteria(
            time_range=ScheduleTimeRange(
                start=datetime.fromtimestamp(first.time, tz=UTC),
                end=datetime.fromtimestamp(second.time, tz=UTC),
            )
        )
    ) == (first, second)
    assert await scheduler.list(ScheduleCriteria(type="interval")) == (second,)
    assert await scheduler.cancel(second.id) is True
    assert await scheduler.cancel(second.id) is False
    assert [event.type for event in events].count("schedule:cancel") == 1


@pytest.mark.asyncio
async def test_one_shot_executes_in_host_context_and_is_consumed(monkeypatch):
    calls = []

    def callback(payload, schedule):
        context = get_current_lifecycle_context()
        calls.append((payload, schedule, context.host))

    scheduler, lifecycle, _, host, events = await _scheduler(
        monkeypatch,
        {"run": callback},
    )
    scheduled = await scheduler.set(0, "run", {"value": 1})

    await lifecycle.alarm()

    assert calls == [({"value": 1}, scheduled, host)]
    assert await scheduler.get(scheduled.id) is None
    assert [event.type for event in events][-1] == "schedule:execute"


@pytest.mark.asyncio
async def test_omitted_payload_reaches_callback_as_missing(monkeypatch):
    calls = []
    scheduler, lifecycle, _, _, _ = await _scheduler(
        monkeypatch,
        {"run": lambda payload, schedule: calls.append((payload, schedule.payload))},
    )
    await scheduler.set(0, "run")

    await lifecycle.alarm()

    assert calls == [(MISSING, MISSING)]


@pytest.mark.asyncio
async def test_callback_retries_and_terminal_error_observer_isolated(
    monkeypatch,
):
    async def no_sleep(*args):
        pass

    monkeypatch.setattr(
        schedules_module,
        "now_ms",
        lambda: 1_700_000_000_000,
    )
    import agents.lifecycle._job_driver as driver_module

    monkeypatch.setattr(driver_module, "_retry_sleep", no_sleep)
    attempts = 0
    observed = []

    async def callback(payload, schedule):
        nonlocal attempts
        attempts += 1
        raise ValueError("failed")

    async def on_error(error):
        observed.append(error)
        raise RuntimeError("observer failed")

    scheduler, lifecycle, _, _, events = await _scheduler(
        monkeypatch,
        {"run": callback},
        retry=RetryOptions(max_attempts=2),
        on_error=on_error,
    )
    scheduled = await scheduler.set(0, "run")

    await lifecycle.alarm()

    assert attempts == 2
    assert len(observed) == 1
    assert isinstance(observed[0], ValueError)
    assert await scheduler.get(scheduled.id) is None
    assert [event.type for event in events][-3:] == [
        "schedule:execute",
        "schedule:retry",
        "schedule:error",
    ]


@pytest.mark.asyncio
async def test_terminal_error_observer_cancellation_propagates(monkeypatch):
    async def callback(payload, schedule):
        raise ValueError("failed")

    async def on_error(error):
        raise asyncio.CancelledError

    scheduler, lifecycle, _, _, _ = await _scheduler(
        monkeypatch,
        {"run": callback},
        retry=RetryOptions(max_attempts=1),
        on_error=on_error,
    )
    scheduled = await scheduler.set(0, "run")

    with pytest.raises(asyncio.CancelledError):
        await lifecycle.alarm()

    assert await scheduler.get(scheduled.id) == scheduled


@pytest.mark.asyncio
async def test_recurring_jobs_advance_from_completion_time(monkeypatch):
    scheduler, lifecycle, _, _, _ = await _scheduler(
        monkeypatch,
        {"run": lambda payload, schedule: None},
        now=1_700_000_000_000,
    )
    interval = await scheduler.every(10, "run")
    monkeypatch.setattr(schedules_module, "now_ms", lambda: 1_700_000_015_250)
    lifecycle._job_driver._clock = lambda: 1_700_000_010_000

    await lifecycle.alarm()

    advanced = await scheduler.get(interval.id)
    assert advanced is not None
    assert advanced.time == 1_700_000_025


@pytest.mark.asyncio
async def test_cron_supports_target_fields_aliases_and_rejects_extensions(monkeypatch):
    scheduler, _, _, _, _ = await _scheduler(
        monkeypatch,
        {"run": lambda payload, schedule: None},
        now=1_704_067_200_500,
    )

    seconds = await scheduler.set("*/10 * * * * *", "run")
    daily = await scheduler.set("@daily", "run")
    tabbed = await scheduler.set("5\t10 * * * *", "run")
    leading_tab = await scheduler.set("\t5 * * * *", "run")
    leading_bom = await scheduler.set("\ufeff5 * * * *", "run")

    assert seconds.time == 1_704_067_210
    assert daily.time == 1_704_153_600
    assert tabbed.time == 1_704_067_500
    assert leading_tab.time == 1_704_067_500
    assert leading_bom.time == 1_704_067_500
    with pytest.raises(ValueError, match="invalid cron"):
        await scheduler.set("0 0 L * *", "run")
    with pytest.raises(ValueError, match="invalid cron"):
        await scheduler.set("١ * * * *", "run")
    with pytest.raises(ValueError, match="invalid cron"):
        await scheduler.set("\u00855 * * * *", "run")


@pytest.mark.asyncio
async def test_cron_combines_restricted_day_and_weekday_with_or(monkeypatch):
    current = datetime(2026, 2, 1, tzinfo=UTC)
    scheduler, _, _, _, _ = await _scheduler(
        monkeypatch,
        {"run": lambda payload, schedule: None},
        now=int(current.timestamp() * 1_000),
    )

    scheduled = await scheduler.set("0 0 31 2 1", "run")

    assert scheduled.time == int(datetime(2026, 2, 2, tzinfo=UTC).timestamp())


@pytest.mark.asyncio
async def test_cron_uses_the_target_five_year_search_horizon(monkeypatch):
    current = datetime(2097, 3, 1, tzinfo=UTC)
    scheduler, _, _, _, _ = await _scheduler(
        monkeypatch,
        {"run": lambda payload, schedule: None},
        now=int(current.timestamp() * 1_000),
    )

    with pytest.raises(ValueError, match="invalid cron"):
        await scheduler.set("0 0 29 2 *", "run")


@pytest.mark.asyncio
async def test_validation_rejects_unknown_callbacks_and_invalid_inputs(monkeypatch):
    scheduler, _, _, _, _ = await _scheduler(
        monkeypatch,
        {"run": lambda payload, schedule: None},
    )

    with pytest.raises(ValueError, match="unknown scheduled callback"):
        await scheduler.set(1, "missing")
    with pytest.raises(TypeError, match="bool"):
        await scheduler.set(True, "run")
    with pytest.raises(ValueError, match="positive finite"):
        await scheduler.every(0, "run")
    with pytest.raises(ValueError, match="cannot exceed"):
        await scheduler.every(2_592_001, "run")
    with pytest.raises(ValueError, match="five or six fields"):
        await scheduler.set("* * * *", "run")
    with pytest.raises(ValueError, match="invalid cron"):
        await scheduler.set("bad * * * *", "run")
    with pytest.raises(ValueError, match="max_attempts"):
        await scheduler.set(
            1,
            "run",
            options=ScheduleOptions(retry=RetryOptions(max_attempts=0)),
        )
    with pytest.raises(ValueError, match="max_attempts"):
        await scheduler.set(
            1,
            "run",
            options=ScheduleOptions(retry=RetryOptions(max_attempts=10**400)),
        )
    with pytest.raises(ValueError):
        await scheduler.set(1, "run", float("nan"))
    with pytest.raises(ValueError, match="finite"):
        await scheduler.set(10**400, "run")
    with pytest.raises(ValueError, match="invalid timestamp"):
        await scheduler.set(1e308, "run")
    delay = (2**63 - 1 - 1_700_000_000_000) // 1_000
    boundary = await scheduler.set(delay, "run")
    assert boundary.time == (1_700_000_000_000 + delay * 1_000) // 1_000


@pytest.mark.asyncio
async def test_malformed_persisted_timing_is_reported_before_callback(monkeypatch):
    calls = []
    scheduler, lifecycle, ctx, _, events = await _scheduler(
        monkeypatch,
        {"run": lambda payload, schedule: calls.append(payload)},
    )
    scheduled = await scheduler.every(10, "run", {"value": 1})
    ctx.storage.sql.exec(
        "UPDATE cf_agents_jobs SET time = ?, payload = ? WHERE id = ?",
        1_700_000_000_000,
        '{"type":"interval","intervalSeconds":0}',
        scheduled.id,
    )

    assert await scheduler.get(scheduled.id) is None
    assert await scheduler.list() == ()
    await lifecycle.alarm()

    assert calls == []
    assert await scheduler.get(scheduled.id) is None
    assert events[-1].type == "schedule:error"


@pytest.mark.asyncio
async def test_corrupt_schedules_remain_cancellable_and_do_not_block_scans(monkeypatch):
    scheduler, _, ctx, _, _ = await _scheduler(
        monkeypatch,
        {"run": lambda payload, schedule: None},
    )
    malformed = await scheduler.every(10, "run")
    ctx.storage.sql.exec(
        "UPDATE cf_agents_jobs SET payload = ? WHERE id = ?",
        '{"type":"interval","intervalSeconds":0}',
        malformed.id,
    )
    corrupt = await scheduler.every(20, "run")
    ctx.storage.sql.exec(
        "UPDATE cf_agents_jobs SET payload = '{' WHERE id = ?",
        corrupt.id,
    )
    oversized = await scheduler.every(25, "run")
    ctx.storage.sql.exec(
        "UPDATE cf_agents_jobs SET retry_options = ? WHERE id = ?",
        '{"baseDelayMs":' + "9" * 400 + "}",
        oversized.id,
    )
    invalid_time = await scheduler.every(27, "run")
    ctx.storage.sql.exec(
        "UPDATE cf_agents_jobs SET time = 'bad' WHERE id = ?",
        invalid_time.id,
    )
    deeply_nested = await scheduler.every(28, "run")
    ctx.storage.sql.exec(
        "UPDATE cf_agents_jobs SET payload = ? WHERE id = ?",
        "[" * 10_000 + "null" + "]" * 10_000,
        deeply_nested.id,
    )

    assert await scheduler.get(malformed.id) is None
    assert await scheduler.get(corrupt.id) is None
    assert await scheduler.get(oversized.id) == oversized
    assert await scheduler.get(invalid_time.id) is None
    assert await scheduler.get(deeply_nested.id) is None
    assert await scheduler.list() == (oversized,)
    replacement = await scheduler.every(30, "run")

    assert await scheduler.cancel(malformed.id) is True
    assert await scheduler.cancel(corrupt.id) is True
    assert await scheduler.cancel(oversized.id) is True
    assert await scheduler.cancel(invalid_time.id) is True
    assert await scheduler.cancel(deeply_nested.id) is True
    assert await scheduler.list() == (replacement,)


@pytest.mark.asyncio
async def test_invalid_retry_defaults_surface_during_execution(monkeypatch):
    calls = []
    observed = []
    scheduler, lifecycle, _, _, events = await _scheduler(
        monkeypatch,
        {"run": lambda payload, schedule: calls.append(payload)},
        retry=RetryOptions(max_attempts=0),
        on_error=observed.append,
    )

    scheduled = await scheduler.set(0, "run")
    overridden = await scheduler.set(
        0,
        "run",
        options=ScheduleOptions(retry=RetryOptions(base_delay_ms=200)),
    )
    assert await scheduler.get(scheduled.id) == scheduled
    assert await scheduler.get(overridden.id) == overridden

    await lifecycle.alarm()

    assert calls == []
    assert len(observed) == 2
    assert all(isinstance(error, ValueError) for error in observed)
    assert [event.type for event in events] == [
        "schedule:create",
        "schedule:create",
        "schedule:error",
        "schedule:error",
    ]
    assert all(event.payload["attempts"] == 0 for event in events[-2:])
    assert await scheduler.list() == ()


def test_generated_schedule_ids_use_url_safe_nanoid_shape():
    ids = {schedules_module._schedule_id() for _ in range(100)}
    alphabet = set("_-0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")

    assert len(ids) == 100
    assert all(len(id) == 9 for id in ids)
    assert all(set(id) <= alphabet for id in ids)


def test_routed_schedule_results_validate_complete_shape():
    valid = {
        "id": "schedule",
        "callback": "run",
        "type": "interval",
        "time": 1,
        "intervalSeconds": 10,
    }

    assert schedules_module._require_schedule_from_route(valid).id == "schedule"
    for invalid in (
        {**valid, "id": None},
        {**valid, "callback": None},
        {**valid, "time": True},
        {**valid, "time": 2**63},
        {**valid, "intervalSeconds": 0},
    ):
        with pytest.raises((TypeError, ValueError)):
            schedules_module._require_schedule_from_route(invalid)


@pytest.mark.asyncio
async def test_callbacks_mapping_is_copied(monkeypatch):
    callbacks = {"run": lambda payload, schedule: None}
    monkeypatch.setattr(schedules_module, "now_ms", lambda: 1_700_000_000_000)
    ctx = fakes.FakeCtx()
    scheduler = Scheduler(callbacks)
    callbacks.clear()
    lifecycle = Lifecycle(ctx, host=object())
    lifecycle.use(scheduler)
    await lifecycle.start()

    assert (await scheduler.set(1, "run")).callback == "run"


@pytest.mark.asyncio
async def test_startup_one_shot_warning_occurs_once_per_callback(monkeypatch):
    monkeypatch.setattr(schedules_module, "now_ms", lambda: 1_700_000_000_000)
    ctx = fakes.FakeCtx()
    scheduler = Scheduler({"run": lambda payload, schedule: None})

    async def on_start():
        await scheduler.set(10, "run")
        await scheduler.set(20, "run")

    lifecycle = Lifecycle(ctx, host=object(), on_start=on_start)
    lifecycle.use(scheduler)

    with pytest.warns(UserWarning, match="every wake") as warnings_seen:
        await lifecycle.start()

    assert len(warnings_seen) == 1
    assert len(await scheduler.list()) == 2
