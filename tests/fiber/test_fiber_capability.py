from __future__ import annotations

import asyncio
import json
import types

import fakes
import pytest
import workers

import agents.lifecycle._runtime as lifecycle_module
import agents.lifecycle.fiber as fiber_module
from agents import Agent, ChatMessageType, FiberRecoveryContext, FiberRecoveryResult
from agents.core.schema import CORE_SCHEMA_VERSION
from agents.lifecycle.fiber import (
    INTERNAL_FIBER_PREFIX,
    FiberCapability,
)
from agents.lifecycle import (
    Lifecycle,
    LifecycleRouteAddress,
    get_current_lifecycle_context,
)


def _install(
    capability: FiberCapability,
    *,
    host: object | None = None,
    retain_work=None,
):
    ctx = fakes.FakeCtx()
    lifecycle = Lifecycle(
        ctx,
        host=host or object(),
        retain_work=retain_work,
    )
    lifecycle.use(capability)
    return ctx, lifecycle


def test_constructor_and_installation_do_not_touch_storage():
    class SilentContext:
        @property
        def storage(self):
            raise AssertionError("storage accessed")

    capability = FiberCapability()
    lifecycle = Lifecycle(SilentContext(), host=object())
    lifecycle.use(capability)

    assert capability._prepared is False


def test_agent_binds_default_capabilities_to_one_lifecycle():
    agent = Agent(fakes.FakeCtx(), types.SimpleNamespace())

    assert agent._lifecycle._bound_lifecycle(agent.scheduler) is agent._lifecycle
    assert agent._lifecycle._bound_lifecycle(agent.tasks) is agent._lifecycle
    assert agent._lifecycle._bound_lifecycle(agent.mcp) is agent._lifecycle
    assert agent._lifecycle._bound_lifecycle(agent._websockets) is agent._lifecycle
    assert agent._lifecycle._bound_lifecycle(agent._fiber) is agent._lifecycle
    assert [item.capability for item in agent._lifecycle._fallbacks] == [
        agent._websockets
    ]
    assert agent._lifecycle._seal_registrations() == (
        agent.scheduler,
        agent.tasks,
        agent.mcp,
        agent._fiber,
        agent._websockets,
    )


@pytest.mark.asyncio
async def test_agent_on_start_can_invoke_fiber_api_without_recursing():
    class StartupAgent(Agent):
        started_fiber = None

        async def on_start(self):
            async def body(_context):
                return "started"

            self.started_fiber = await self.run_fiber("startup", body)

    agent = fakes.build_agent(cls=StartupAgent)

    await asyncio.wait_for(agent._ensure_initialized(), timeout=1)

    assert agent.started_fiber == "started"


@pytest.mark.asyncio
async def test_recovery_callback_can_inspect_fiber_without_recursing():
    class InspectingRecoveryAgent(Agent):
        recovered_inspection = None

        async def on_fiber_recovered(self, context):
            self.recovered_inspection = await self.inspect_fiber(context.id)

    agent = fakes.build_agent(cls=InspectingRecoveryAgent)
    agent.sql(
        "INSERT INTO cf_agents_fibers (fiber_id, name, status, created_at) "
        "VALUES ('recovering', 'work', 'running', 1)"
    )
    agent.sql(
        "INSERT INTO cf_agents_runs (id, name, snapshot, created_at) "
        "VALUES ('recovering', 'work', NULL, 1)"
    )

    await asyncio.wait_for(agent._ensure_initialized(), timeout=1)

    assert agent.recovered_inspection is not None
    assert agent.recovered_inspection.status == "interrupted"


@pytest.mark.asyncio
async def test_plain_lifecycle_host_runs_and_prepares_fibers():
    capability = FiberCapability()
    host = object()
    ctx, lifecycle = _install(capability, host=host)
    callback_hosts = []

    async def body(fiber_context):
        current = get_current_lifecycle_context()
        callback_hosts.append(None if current is None else current.host)
        fiber_context.stash({"step": 1})
        return "complete"

    assert await capability.run_fiber("plain", body) == "complete"
    assert callback_hosts == [host]
    assert ctx.storage.sql.exec(
        "SELECT name FROM sqlite_master WHERE name IN "
        "('cf_agents_runs', 'cf_agents_fibers') ORDER BY name"
    ).toArray() == [
        {"name": "cf_agents_fibers"},
        {"name": "cf_agents_runs"},
    ]
    assert lifecycle._ready is True


@pytest.mark.asyncio
async def test_plain_keep_alive_without_retained_work_rejects_before_mutation():
    capability = FiberCapability()
    ctx, _lifecycle = _install(capability)
    alarm_history = ctx.storage.alarm_history_ms

    with pytest.raises(RuntimeError, match="retained work is not available"):
        await capability.keep_alive()

    assert capability._keep_alive_refs == 0
    assert (
        ctx.storage.sql.exec(
            "SELECT name FROM sqlite_master WHERE name = 'cf_agents_jobs'"
        ).toArray()
        == []
    )
    assert ctx.storage.alarm_history_ms == alarm_history

    async def body(_context):
        return "complete"

    assert await capability.run_fiber("awaited", body) == "complete"


@pytest.mark.asyncio
async def test_plain_keep_alive_while_cleans_up_without_retained_work(monkeypatch):
    monkeypatch.setattr(fiber_module, "now_ms", lambda: 1_000)
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)
    capability = FiberCapability(keep_alive_interval_ms=lambda: 100)
    ctx, _lifecycle = _install(capability)
    observed = []

    async def body():
        observed.append(
            (
                capability._keep_alive_refs,
                ctx.storage.sql.exec("SELECT id FROM cf_agents_jobs").toArray(),
                ctx.storage.alarm_time_ms,
            )
        )
        return "complete"

    assert await capability.keep_alive_while(body) == "complete"

    assert observed == [
        (
            1,
            [{"id": "__cf_internal_fibers_maintenance"}],
            1_100,
        )
    ]
    assert capability._keep_alive_refs == 0
    assert ctx.storage.sql.exec("SELECT id FROM cf_agents_jobs").toArray() == []
    assert ctx.storage.alarm_time_ms is None


@pytest.mark.asyncio
async def test_facet_allows_awaited_fibers_without_owning_jobs_or_alarm():
    agent = Agent(
        fakes.FakeCtx(name="cf-agents:v2:leaf:digest"),
        types.SimpleNamespace(),
    )
    agent.detached_fibers_enabled = True

    async def body(_context):
        return "complete"

    assert await agent.run_fiber("awaited", body) == "complete"
    with pytest.raises(RuntimeError, match="maintenance is unavailable"):
        await agent.keep_alive()
    with pytest.raises(RuntimeError, match="maintenance is unavailable"):
        await agent.start_fiber("detached", body)
    with pytest.raises(RuntimeError, match="owner-keyed root routing"):
        await agent._schedule_next_alarm()
    with pytest.raises(RuntimeError, match="owner-keyed root routing"):
        await agent._lifecycle.alarm()

    assert agent._lifecycle._owns_physical_alarm is False
    assert agent._fiber.lifecycle._lifecycle is agent._lifecycle
    assert agent._websockets.lifecycle._lifecycle is agent._lifecycle
    assert agent._lifecycle._route_address is None
    assert agent._lifecycle._root_route_address is None
    assert (
        agent.sql("SELECT name FROM sqlite_master WHERE name = 'cf_agents_jobs'") == []
    )
    assert agent.ctx.storage.alarm_history_ms == ()


@pytest.mark.asyncio
async def test_facet_recovery_stays_gated_without_root_maintenance():
    recovered = []

    class RecoveryFacet(Agent):
        async def on_fiber_recovered(self, context):
            recovered.append(context.id)
            return FiberRecoveryResult(status="completed")

    agent = fakes.build_agent(
        cls=RecoveryFacet,
        name="cf-agents:v2:leaf:digest",
    )
    agent.sql(
        "INSERT INTO cf_agents_fibers (fiber_id, name, status, created_at) "
        "VALUES ('orphan', 'work', 'running', 1)"
    )
    agent.sql(
        "INSERT INTO cf_agents_runs (id, name, snapshot, created_at) "
        "VALUES ('orphan', 'work', NULL, 1)"
    )

    await agent._ensure_initialized()

    async def body(_context):
        return None

    with pytest.raises(RuntimeError, match="without routed recovery"):
        await agent.start_fiber(
            "work",
            body,
            fiber_id="orphan",
            wait_for_completion=True,
        )

    assert recovered == []
    assert (await agent.inspect_fiber("orphan")).status == "running"
    assert agent.sql("SELECT id FROM cf_agents_runs") == [{"id": "orphan"}]
    assert (
        agent.sql("SELECT name FROM sqlite_master WHERE name = 'cf_agents_jobs'") == []
    )
    assert agent.ctx.storage.alarm_history_ms == ()


@pytest.mark.asyncio
async def test_unconfigured_facet_chat_turn_does_not_request_a_local_alarm():
    replies = []

    def reply(_options):
        replies.append("called")
        return "hello"

    agent = fakes.build_chat_agent(
        reply,
        name="cf-agents:v2:chat:digest",
    )
    connection = fakes.FakeConnection(id="requester")
    agent._connections[connection.id] = connection
    body = json.dumps(
        {
            "messages": [
                {
                    "id": "message",
                    "role": "user",
                    "parts": [{"type": "text", "text": "hi"}],
                }
            ]
        }
    )

    await agent._handle_use_chat_request(
        connection,
        {"id": "request", "init": {"method": "POST", "body": body}},
    )

    assert replies == []
    assert any(
        frame.get("type") == ChatMessageType.USE_CHAT_RESPONSE
        and frame.get("id") == "request"
        and frame.get("done") is True
        and frame.get("error") is True
        for frame in connection.frames
    )
    assert (
        agent.sql("SELECT name FROM sqlite_master WHERE name = 'cf_agents_jobs'") == []
    )
    assert agent.ctx.storage.alarm_history_ms == ()


@pytest.mark.asyncio
async def test_preparation_failure_degrades_startup_and_retries_on_operation(
    monkeypatch,
):
    prepare = fiber_module.prepare_fiber_schema
    attempts = 0

    def fail_once(sql):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("storage unavailable")
        prepare(sql)

    monkeypatch.setattr(fiber_module, "prepare_fiber_schema", fail_once)
    capability = FiberCapability()
    ctx, lifecycle = _install(capability)
    prepare(lambda query, *params: ctx.storage.sql.exec(query, *params).toArray())
    ctx.storage.sql.exec(
        "INSERT INTO cf_agents_runs (id, name, snapshot, created_at) "
        "VALUES ('orphan', 'work', NULL, 1)"
    )

    await lifecycle.start()

    assert capability._prepared is False
    assert await capability.inspect_fiber("missing") is None
    assert capability._prepared is True
    assert attempts == 2
    assert ctx.storage.sql.exec("SELECT id FROM cf_agents_runs").toArray() == []
    assert ctx.storage.sql.exec("SELECT id FROM cf_agents_jobs").toArray() == []


def test_shared_future_schema_marker_prevents_fiber_schema_writes():
    capability = FiberCapability(shared_schema=True)
    ctx, _lifecycle = _install(capability)
    ctx.storage.sql.exec(
        "CREATE TABLE cf_agents_state (id TEXT PRIMARY KEY NOT NULL, state TEXT)"
    )
    ctx.storage.sql.exec(
        "INSERT INTO cf_agents_state (id, state) VALUES ('cf_schema_version', ?)",
        str(CORE_SCHEMA_VERSION + 1),
    )

    capability._prepare()

    assert capability._prepared is True
    assert (
        ctx.storage.sql.exec(
            "SELECT name FROM sqlite_master WHERE name IN "
            "('cf_agents_runs', 'cf_agents_fibers')"
        ).toArray()
        == []
    )


@pytest.mark.asyncio
async def test_agent_settings_are_read_live_for_each_first_lease(monkeypatch):
    current_time = 1_000
    monkeypatch.setattr(fiber_module, "now_ms", lambda: current_time)
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: current_time)
    recorder = fakes.WaitUntilRecorder()
    agent = Agent(fakes.FakeCtx(), types.SimpleNamespace())
    monkeypatch.setattr(workers, "waitUntil", recorder)

    agent.keep_alive_interval_ms = 100
    first_dispose = await agent.keep_alive()
    first = await agent._fiber.lifecycle.jobs.get("__cf_internal_fibers_maintenance")
    assert first is not None and first.time == 1_100
    first_dispose()
    await recorder.drain()

    agent.keep_alive_interval_ms = 250
    second_dispose = await agent.keep_alive()
    second = await agent._fiber.lifecycle.jobs.get("__cf_internal_fibers_maintenance")
    assert second is not None and second.time == 1_250
    second_dispose()
    await recorder.drain()


def test_agent_non_interval_fiber_settings_are_read_live():
    agent = Agent(fakes.FakeCtx(), types.SimpleNamespace())

    agent.detached_fibers_enabled = True
    agent.fiber_recovery_scan_deadline_ms = 11
    agent.fiber_recovery_max_age_ms = 22
    agent.fiber_recovery_hook_timeout_ms = 33
    agent.fiber_recovery_max_backoff_ms = 44

    assert agent._fiber._get_detached_fibers_enabled() is True
    assert agent._fiber._get_recovery_scan_deadline_ms() == 11
    assert agent._fiber._get_recovery_max_age_ms() == 22
    assert agent._fiber._get_recovery_hook_timeout_ms() == 33
    assert agent._fiber._get_recovery_max_backoff_ms() == 44


@pytest.mark.asyncio
async def test_startup_reconciles_one_maintenance_job_once(monkeypatch):
    current_time = 1_000
    monkeypatch.setattr(fiber_module, "now_ms", lambda: current_time)
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: current_time)
    capability = FiberCapability()
    ctx, lifecycle = _install(capability)

    from agents.lifecycle.fiber_schema import prepare_fiber_schema

    prepare_fiber_schema(
        lambda query, *params: ctx.storage.sql.exec(query, *params).toArray()
    )
    ctx.storage.sql.exec(
        "INSERT INTO cf_agents_runs (id, name, snapshot, created_at) "
        "VALUES ('internal', ?, NULL, 1)",
        f"{INTERNAL_FIBER_PREFIX}unknown",
    )

    await lifecycle.start()
    await lifecycle.start()

    rows = ctx.storage.sql.exec(
        "SELECT id, capability, recovery_loop FROM cf_agents_jobs"
    ).toArray()
    assert rows == [
        {
            "id": "__cf_internal_fibers_maintenance",
            "capability": "fibers",
            "recovery_loop": 1,
        }
    ]
    assert capability._recovery_no_progress_scans == 1


@pytest.mark.asyncio
async def test_first_lease_rolls_back_when_scheduling_fails(monkeypatch):
    monkeypatch.setattr(fiber_module, "now_ms", lambda: 1_000)
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)
    capability = FiberCapability()
    ctx, lifecycle = _install(
        capability,
        retain_work=fakes.WaitUntilRecorder(),
    )
    await lifecycle.start()

    async def reject_alarm(_deadline):
        raise RuntimeError("alarm unavailable")

    ctx.storage.setAlarm = reject_alarm

    with pytest.raises(RuntimeError, match="alarm unavailable"):
        await capability.keep_alive()

    assert capability._keep_alive_refs == 0
    assert (
        ctx.storage.sql.exec(
            "SELECT id FROM cf_agents_jobs WHERE capability = 'fibers'"
        ).toArray()
        == []
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("column", ["payload", "retry_options"])
async def test_keep_alive_release_cancels_corrupt_maintenance_job(
    monkeypatch,
    column,
):
    monkeypatch.setattr(fiber_module, "now_ms", lambda: 1_000)
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)
    recorder = fakes.WaitUntilRecorder()
    capability = FiberCapability(keep_alive_interval_ms=lambda: 100)
    ctx, lifecycle = _install(capability, retain_work=recorder)
    await lifecycle.start()
    dispose = await capability.keep_alive()
    ctx.storage.sql.exec(
        f"UPDATE cf_agents_jobs SET {column} = '{{' "
        "WHERE id = '__cf_internal_fibers_maintenance'"
    )

    dispose()
    await recorder.drain()

    assert capability._keep_alive_refs == 0
    assert ctx.storage.sql.exec("SELECT id FROM cf_agents_jobs").toArray() == []
    assert ctx.storage.alarm_time_ms is None


@pytest.mark.asyncio
@pytest.mark.parametrize("column", ["payload", "retry_options"])
async def test_keep_alive_repairs_corrupt_maintenance_job(monkeypatch, column):
    monkeypatch.setattr(fiber_module, "now_ms", lambda: 1_000)
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)
    recorder = fakes.WaitUntilRecorder()
    capability = FiberCapability(keep_alive_interval_ms=lambda: 100)
    ctx, lifecycle = _install(capability, retain_work=recorder)
    await lifecycle.start()
    ctx.storage.sql.exec(
        "INSERT INTO cf_agents_jobs "
        "(id, capability, fn, time, recovery_loop) VALUES "
        "('__cf_internal_fibers_maintenance', 'fibers', "
        "'__cf_internal_fibers_maintenance', 5_000, 1)"
    )
    ctx.storage.sql.exec(
        f"UPDATE cf_agents_jobs SET {column} = '{{' "
        "WHERE id = '__cf_internal_fibers_maintenance'"
    )

    dispose = await capability.keep_alive()

    job = await capability.lifecycle.jobs.get("__cf_internal_fibers_maintenance")
    assert job is not None
    assert job.time == 1_100
    dispose()
    await recorder.drain()


@pytest.mark.asyncio
async def test_concurrent_first_lease_failure_does_not_rollback_successor(monkeypatch):
    monkeypatch.setattr(fiber_module, "now_ms", lambda: 1_000)
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)
    capability = FiberCapability(keep_alive_interval_ms=lambda: 100)
    recorder = fakes.WaitUntilRecorder()
    ctx, lifecycle = _install(capability, retain_work=recorder)
    await lifecycle.start()
    first_arm_started = asyncio.Event()
    release_first_arm = asyncio.Event()
    original_set_alarm = ctx.storage.setAlarm
    arms = 0

    async def set_alarm(deadline):
        nonlocal arms
        arms += 1
        if arms == 1:
            first_arm_started.set()
            await release_first_arm.wait()
            raise RuntimeError("first arm failed")
        await original_set_alarm(deadline)

    ctx.storage.setAlarm = set_alarm
    first = asyncio.create_task(capability.keep_alive())
    await first_arm_started.wait()
    second = asyncio.create_task(capability.keep_alive())
    await asyncio.sleep(0)
    release_first_arm.set()

    with pytest.raises(RuntimeError, match="first arm failed"):
        await first
    dispose_second = await second

    assert capability._keep_alive_refs == 1
    assert len(capability._active_lease_generations) == 1
    assert ctx.storage.alarm_time_ms == 1_100
    assert ctx.storage.sql.exec(
        "SELECT id FROM cf_agents_jobs WHERE capability = 'fibers'"
    ).toArray() == [{"id": "__cf_internal_fibers_maintenance"}]
    dispose_second()
    await recorder.drain()


@pytest.mark.asyncio
async def test_detached_handoff_rejection_rolls_back_without_starting_runner():
    submitted = 0

    def reject(_awaitable):
        raise RuntimeError("handoff rejected")

    capability = FiberCapability(detached_fibers_enabled=lambda: True)
    ctx, lifecycle = _install(capability, retain_work=reject)
    await lifecycle.start()

    async def body(_context):
        nonlocal submitted
        submitted += 1

    with pytest.raises(RuntimeError, match="handoff rejected"):
        await capability.start_fiber("detached", body)

    assert submitted == 0
    assert capability._fiber_active_ids == set()
    assert capability._active_lease_generations == set()
    assert capability._keep_alive_refs == 0
    assert ctx.storage.sql.exec("SELECT fiber_id FROM cf_agents_fibers").toArray() == []
    assert ctx.storage.sql.exec("SELECT id FROM cf_agents_jobs").toArray() == []
    assert ctx.storage.alarm_time_ms is None


@pytest.mark.asyncio
async def test_detached_alarm_failure_removes_pending_row_and_job(monkeypatch):
    monkeypatch.setattr(fiber_module, "now_ms", lambda: 1_000)
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)
    recorder = fakes.WaitUntilRecorder()
    capability = FiberCapability(
        detached_fibers_enabled=lambda: True,
        keep_alive_interval_ms=lambda: 100,
    )
    ctx, lifecycle = _install(capability, retain_work=recorder)
    await lifecycle.start()

    async def reject_alarm(_deadline):
        raise RuntimeError("alarm unavailable")

    ctx.storage.setAlarm = reject_alarm

    async def body(_context):
        raise AssertionError("rejected work must not run")

    with pytest.raises(RuntimeError, match="alarm unavailable"):
        await capability.start_fiber("detached", body)

    assert ctx.storage.sql.exec("SELECT fiber_id FROM cf_agents_fibers").toArray() == []
    assert ctx.storage.sql.exec("SELECT id FROM cf_agents_jobs").toArray() == []
    assert capability._active_lease_generations == set()
    assert recorder.records == []


@pytest.mark.asyncio
async def test_cancel_pending_detached_fiber_releases_transferred_lease(monkeypatch):
    monkeypatch.setattr(fiber_module, "now_ms", lambda: 1_000)
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)
    recorder = fakes.WaitUntilRecorder()
    capability = FiberCapability(
        detached_fibers_enabled=lambda: True,
        keep_alive_interval_ms=lambda: 100,
    )
    ctx, _lifecycle = _install(capability, retain_work=recorder)
    body_calls = 0

    async def body(_context):
        nonlocal body_calls
        body_calls += 1

    started = await capability.start_fiber("detached", body)
    assert await capability.cancel_fiber(started.fiber_id, "cancelled") is True

    assert (await capability.inspect_fiber(started.fiber_id)).status == "aborted"
    assert capability._pending_detached_leases == {}
    assert capability._fiber_active_ids == set()
    assert capability._active_lease_generations == set()
    assert ctx.storage.sql.exec("SELECT id FROM cf_agents_jobs").toArray() == []
    assert ctx.storage.alarm_time_ms is None

    await recorder.drain()

    assert body_calls == 0
    assert (await capability.inspect_fiber(started.fiber_id)).status == "aborted"
    assert ctx.storage.sql.exec("SELECT id FROM cf_agents_jobs").toArray() == []
    assert ctx.storage.alarm_time_ms is None


@pytest.mark.asyncio
async def test_cancel_detached_fiber_racing_first_alarm_acquisition(monkeypatch):
    monkeypatch.setattr(fiber_module, "now_ms", lambda: 1_000)
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)
    recorder = fakes.WaitUntilRecorder()
    capability = FiberCapability(
        detached_fibers_enabled=lambda: True,
        keep_alive_interval_ms=lambda: 100,
    )
    ctx, _lifecycle = _install(capability, retain_work=recorder)
    gate = fakes.AsyncGate()
    ctx.storage.pause_next_alarm(gate)
    body_calls = 0

    async def body(_context):
        nonlocal body_calls
        body_calls += 1

    accepting = asyncio.create_task(
        capability.start_fiber(
            "detached",
            body,
            fiber_id="cancel-race",
            idempotency_key="cancel-race-key",
        )
    )
    await gate.wait_until_blocked()

    assert await capability.cancel_fiber_by_key("cancel-race-key", "cancelled") is True
    gate.release()
    started = await accepting

    assert started.status == "aborted"
    assert body_calls == 0
    assert recorder.records == []
    assert capability._pending_detached_leases == {}
    assert capability._fiber_active_ids == set()
    assert capability._active_lease_generations == set()
    assert ctx.storage.sql.exec("SELECT id FROM cf_agents_jobs").toArray() == []
    assert ctx.storage.alarm_time_ms is None


@pytest.mark.asyncio
async def test_retained_managed_runner_executes_in_host_context():
    host = object()
    recorder = fakes.WaitUntilRecorder()
    capability = FiberCapability(detached_fibers_enabled=lambda: True)
    _ctx, _lifecycle = _install(
        capability,
        host=host,
        retain_work=recorder,
    )
    callback_hosts = []

    async def body(_context):
        current = get_current_lifecycle_context()
        callback_hosts.append(None if current is None else current.host)

    started = await capability.start_fiber("detached", body)
    await recorder.drain()

    assert callback_hosts == [host]
    inspection = await capability.inspect_fiber(started.fiber_id)
    assert inspection is not None and inspection.status == "completed"


@pytest.mark.asyncio
async def test_recovery_callback_runs_in_plain_host_context():
    host = object()
    seen = []

    async def recovered(context):
        current = get_current_lifecycle_context()
        seen.append((context.id, None if current is None else current.host))

    capability = FiberCapability(on_fiber_recovered=recovered)
    ctx, lifecycle = _install(capability, host=host)
    await lifecycle.start()
    ctx.storage.sql.exec(
        "INSERT INTO cf_agents_runs (id, name, snapshot, created_at) "
        "VALUES ('orphan', 'work', NULL, 1)"
    )

    await capability._check_run_fibers()

    assert seen == [("orphan", host)]


@pytest.mark.asyncio
async def test_recovery_timeout_restores_nested_host_context():
    host = object()
    timeout_ms = 1_000
    seen = []

    async def recovered(_context):
        current = get_current_lifecycle_context()
        seen.append(None if current is None else current.host)
        try:
            await asyncio.Event().wait()
        finally:
            current = get_current_lifecycle_context()
            seen.append(None if current is None else current.host)

    capability = FiberCapability(
        on_internal_fiber_recovery=recovered,
        recovery_hook_timeout_ms=lambda: timeout_ms,
    )
    _ctx, lifecycle = _install(capability, host=host)
    await lifecycle.start()
    timeout_ms = 1
    recovery = FiberRecoveryContext("orphan", "work", None, 1)

    async def outer():
        assert get_current_lifecycle_context().host is host
        assert await capability._run_fiber_recovery_hook(recovery, None) is False
        assert get_current_lifecycle_context().host is host

    await capability.lifecycle.run_in_host_context(outer)

    assert seen == [host, host]
    assert get_current_lifecycle_context() is None


@pytest.mark.asyncio
async def test_recovery_timeout_does_not_bound_user_hook():
    release = asyncio.Event()

    async def recovered(_context):
        await release.wait()
        return FiberRecoveryResult(status="completed")

    capability = FiberCapability(
        on_fiber_recovered=recovered,
        recovery_hook_timeout_ms=lambda: 1,
    )
    _ctx, lifecycle = _install(capability)
    await lifecycle.start()
    recovery = FiberRecoveryContext("orphan", "work", None, 1)
    running = asyncio.create_task(capability._run_fiber_recovery_hook(recovery, None))
    await asyncio.sleep(0.01)

    assert not running.done()
    release.set()
    assert await running is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("initial_status", "run_backed"),
    [("pending", True), ("running", False)],
    ids=["run-backed", "ledger-only"],
)
async def test_memory_limit_retry_restores_recoverable_work_before_sealing(
    monkeypatch,
    initial_status,
    run_backed,
):
    current_time = 1_000
    monkeypatch.setattr(fiber_module, "now_ms", lambda: current_time)
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: current_time)
    recovery_calls = 0

    async def recover(_context):
        nonlocal recovery_calls
        recovery_calls += 1
        raise RuntimeError(
            "Durable Object's isolate exceeded its memory limit and was reset."
        )

    ctx = fakes.FakeCtx()
    setup_capability = FiberCapability(keep_alive_interval_ms=lambda: 100)
    setup = Lifecycle(ctx, host=object())
    setup.use(setup_capability)
    await setup.start()
    ctx.storage.sql.exec(
        "INSERT INTO cf_agents_fibers "
        "(fiber_id, name, status, snapshot, metadata_json, error_message, "
        "created_at, completed_at) VALUES "
        "('manual', 'manual-work', 'interrupted', '\"manual-snapshot\"', "
        "'{\"source\":\"manual\"}', 'manual-error', 1, 77)"
    )
    ctx.storage.sql.exec(
        "INSERT INTO cf_agents_fibers "
        "(fiber_id, name, status, error_message, created_at, completed_at) "
        "VALUES ('current', 'work', ?, 'prior-error', 1, 123)",
        initial_status,
    )
    if run_backed:
        ctx.storage.sql.exec(
            "INSERT INTO cf_agents_runs (id, name, snapshot, created_at) "
            "VALUES ('current', 'work', NULL, 1)"
        )
    await setup_capability.lifecycle.jobs.push(
        lifecycle_module.LifecycleJobPushOptions(
            id="__cf_internal_fibers_maintenance",
            fn="__cf_internal_fibers_maintenance",
            time=100,
            retry={"maxAttempts": 1},
            recovery_loop=True,
        )
    )

    def fresh_activation():
        capability = FiberCapability(
            keep_alive_interval_ms=lambda: 100,
            on_internal_fiber_recovery=recover,
        )
        lifecycle = Lifecycle(
            ctx,
            host=object(),
            max_alarm_memory_limit_strikes=2,
            reset_alarm=lambda _reason: None,
        )
        lifecycle.use(capability)
        return lifecycle

    await fresh_activation().alarm()

    assert recovery_calls == 1
    assert ctx.storage.sql.exec(
        "SELECT status, error_message, completed_at FROM cf_agents_fibers "
        "WHERE fiber_id = 'current'"
    ).toArray() == [
        {
            "status": initial_status,
            "error_message": "prior-error",
            "completed_at": 123,
        }
    ]
    expected_runs = [{"id": "current"}] if run_backed else []
    assert (
        ctx.storage.sql.exec("SELECT id FROM cf_agents_runs").toArray() == expected_runs
    )
    assert ctx.storage.sql.exec(
        "SELECT id, time, running FROM cf_agents_jobs"
    ).toArray() == [
        {
            "id": "__cf_internal_fibers_maintenance",
            "time": 31_000,
            "running": 0,
        }
    ]
    assert ctx.storage.alarm_time_ms == 31_000
    assert ctx.storage.sql.exec(
        "SELECT status, snapshot, metadata_json, error_message, completed_at "
        "FROM cf_agents_fibers WHERE fiber_id = 'manual'"
    ).toArray() == [
        {
            "status": "interrupted",
            "snapshot": '"manual-snapshot"',
            "metadata_json": '{"source":"manual"}',
            "error_message": "manual-error",
            "completed_at": 77,
        }
    ]

    ordinary_activation = fresh_activation()
    await ordinary_activation.start()

    assert recovery_calls == 1
    assert ctx.storage.sql.exec(
        "SELECT id, time, running FROM cf_agents_jobs"
    ).toArray() == [
        {
            "id": "__cf_internal_fibers_maintenance",
            "time": 31_000,
            "running": 0,
        }
    ]
    assert ctx.storage.alarm_time_ms == 31_000

    active_capability = ordinary_activation._capabilities_by_id["fibers"]

    async def foreground(_context):
        return "complete"

    assert await active_capability.run_fiber("foreground", foreground) == "complete"
    with pytest.raises(RuntimeError, match="during memory backoff"):
        await active_capability.start_fiber(
            "work",
            foreground,
            fiber_id="current",
            wait_for_completion=True,
        )

    assert recovery_calls == 1
    assert ctx.storage.sql.exec(
        "SELECT id, time, running FROM cf_agents_jobs"
    ).toArray() == [
        {
            "id": "__cf_internal_fibers_maintenance",
            "time": 31_000,
            "running": 0,
        }
    ]
    assert ctx.storage.alarm_time_ms == 31_000

    current_time = 31_000
    await fresh_activation().alarm()

    assert recovery_calls == 2
    assert ctx.storage.sql.exec(
        "SELECT status, error_message, completed_at FROM cf_agents_fibers "
        "WHERE fiber_id = 'current'"
    ).toArray() == [
        {
            "status": "error",
            "error_message": fiber_module._MEMORY_LIMIT_ERROR,
            "completed_at": 31_000,
        },
    ]
    assert ctx.storage.sql.exec(
        "SELECT status, snapshot, metadata_json, error_message, completed_at "
        "FROM cf_agents_fibers WHERE fiber_id = 'manual'"
    ).toArray() == [
        {
            "status": "interrupted",
            "snapshot": '"manual-snapshot"',
            "metadata_json": '{"source":"manual"}',
            "error_message": "manual-error",
            "completed_at": 77,
        }
    ]
    assert ctx.storage.sql.exec("SELECT id FROM cf_agents_runs").toArray() == []
    assert ctx.storage.sql.exec("SELECT id FROM cf_agents_jobs").toArray() == []

    reincarnated = Lifecycle(ctx, host=object())
    reincarnated_capability = FiberCapability(keep_alive_interval_ms=lambda: 100)
    reincarnated.use(reincarnated_capability)
    await reincarnated.start()

    assert ctx.storage.sql.exec("SELECT id FROM cf_agents_jobs").toArray() == []
    assert ctx.storage.alarm_time_ms is None


@pytest.mark.asyncio
async def test_startup_platform_failure_keeps_wake_for_fresh_alarm_retry(monkeypatch):
    current_time = 1_000
    monkeypatch.setattr(fiber_module, "now_ms", lambda: current_time)
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: current_time)
    recovery_calls = []
    ctx = fakes.FakeCtx()
    setup_capability = FiberCapability(keep_alive_interval_ms=lambda: 100)
    setup = Lifecycle(ctx, host=object())
    setup.use(setup_capability)
    await setup.start()
    ctx.storage.sql.exec(
        "INSERT INTO cf_agents_fibers "
        "(fiber_id, name, status, created_at, completed_at) "
        "VALUES ('current', 'work', 'pending', 1, 123)"
    )
    ctx.storage.sql.exec(
        "INSERT INTO cf_agents_runs (id, name, snapshot, created_at) "
        "VALUES ('current', 'work', NULL, 1)"
    )

    async def recover(context):
        recovery_calls.append(context.id)
        raise RuntimeError("network connection lost")

    def fresh_activation():
        capability = FiberCapability(
            keep_alive_interval_ms=lambda: 100,
            on_internal_fiber_recovery=recover,
        )
        lifecycle = Lifecycle(ctx, host=object())
        lifecycle.use(capability)
        return lifecycle

    with pytest.raises(RuntimeError, match="network connection lost"):
        await fresh_activation().alarm()

    assert recovery_calls == ["current"]
    assert ctx.storage.sql.exec(
        "SELECT status, completed_at FROM cf_agents_fibers WHERE fiber_id = 'current'"
    ).toArray() == [{"status": "pending", "completed_at": 123}]
    assert ctx.storage.sql.exec(
        "SELECT id, time, retry_options, running FROM cf_agents_jobs"
    ).toArray() == [
        {
            "id": "__cf_internal_fibers_maintenance",
            "time": 1_100,
            "retry_options": '{"maxAttempts":1}',
            "running": 0,
        }
    ]
    assert ctx.storage.alarm_time_ms == 1_100

    current_time = 1_100
    with pytest.raises(RuntimeError, match="network connection lost"):
        await fresh_activation().alarm()

    assert recovery_calls == ["current", "current"]
    assert ctx.storage.sql.exec(
        "SELECT status, completed_at FROM cf_agents_fibers WHERE fiber_id = 'current'"
    ).toArray() == [{"status": "pending", "completed_at": 123}]
    assert ctx.storage.sql.exec(
        "SELECT id, running, execution_started_at FROM cf_agents_jobs"
    ).toArray() == [
        {
            "id": "__cf_internal_fibers_maintenance",
            "running": 0,
            "execution_started_at": None,
        }
    ]
    assert ctx.storage.alarm_time_ms == 1_101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("payload", "{"),
        ("retry_options", "{"),
        ("retry_options", None),
        ("time", "invalid"),
        ("time", 1.5),
    ],
    ids=[
        "malformed-payload",
        "malformed-retry",
        "null-retry",
        "text-time",
        "float-time",
    ],
)
async def test_due_maintenance_job_is_canonicalized_before_dispatch(
    monkeypatch,
    column,
    value,
):
    monkeypatch.setattr(fiber_module, "now_ms", lambda: 1_000)
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)
    ctx = fakes.FakeCtx()
    setup_capability = FiberCapability(keep_alive_interval_ms=lambda: 100)
    setup = Lifecycle(ctx, host=object())
    setup.use(setup_capability)
    await setup.start()
    ctx.storage.sql.exec(
        "INSERT INTO cf_agents_fibers "
        "(fiber_id, name, status, created_at) "
        "VALUES ('current', 'work', 'pending', 1)"
    )
    ctx.storage.sql.exec(
        "INSERT INTO cf_agents_runs (id, name, snapshot, created_at) "
        "VALUES ('current', 'work', NULL, 1)"
    )
    await setup_capability.lifecycle.jobs.push(
        lifecycle_module.LifecycleJobPushOptions(
            id="__cf_internal_fibers_maintenance",
            fn="legacy",
            time=1_000,
            retry={"maxAttempts": 9},
            recovery_loop=True,
        )
    )
    ctx.storage.sql.exec(
        f"UPDATE cf_agents_jobs SET {column} = ?, fn = 'legacy', "
        "singleflight = 1, hung_timeout_seconds = 7, exclusive = 1, "
        "recovery_loop = 0, running = 1, execution_started_at = 900 "
        "WHERE id = '__cf_internal_fibers_maintenance'",
        value,
    )
    observed_jobs = []

    async def recover(_context):
        observed_jobs.extend(
            ctx.storage.sql.exec(
                "SELECT fn, time, payload, retry_options, singleflight, "
                "hung_timeout_seconds, exclusive, recovery_loop, running, "
                "execution_started_at FROM cf_agents_jobs"
            ).toArray()
        )
        return FiberRecoveryResult(status="completed")

    capability = FiberCapability(
        keep_alive_interval_ms=lambda: 100,
        on_fiber_recovered=recover,
        defer_startup_recovery_to_alarm=lambda: True,
    )
    lifecycle = Lifecycle(ctx, host=object())
    lifecycle.use(capability)

    await lifecycle.alarm()

    assert observed_jobs == [
        {
            "fn": "__cf_internal_fibers_maintenance",
            "time": 1_000,
            "payload": None,
            "retry_options": '{"maxAttempts":1}',
            "singleflight": 0,
            "hung_timeout_seconds": None,
            "exclusive": 0,
            "recovery_loop": 1,
            "running": 1,
            "execution_started_at": 1_000,
        }
    ]
    assert (await capability.inspect_fiber("current")).status == "completed"
    assert ctx.storage.sql.exec("SELECT id FROM cf_agents_runs").toArray() == []
    assert ctx.storage.sql.exec("SELECT id FROM cf_agents_jobs").toArray() == []
    assert ctx.storage.alarm_time_ms is None


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_time", ["invalid", 1.5])
async def test_invalid_maintenance_time_is_rebuilt_before_startup_recovery(
    monkeypatch,
    invalid_time,
):
    monkeypatch.setattr(fiber_module, "now_ms", lambda: 1_000)
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)
    ctx = fakes.FakeCtx()
    setup_capability = FiberCapability(keep_alive_interval_ms=lambda: 100)
    setup = Lifecycle(ctx, host=object())
    setup.use(setup_capability)
    await setup.start()
    ctx.storage.sql.exec(
        "INSERT INTO cf_agents_fibers "
        "(fiber_id, name, status, created_at) "
        "VALUES ('current', 'work', 'pending', 1)"
    )
    ctx.storage.sql.exec(
        "INSERT INTO cf_agents_runs (id, name, snapshot, created_at) "
        "VALUES ('current', 'work', NULL, 1)"
    )
    await setup_capability.lifecycle.jobs.push(
        lifecycle_module.LifecycleJobPushOptions(
            id="__cf_internal_fibers_maintenance",
            fn="legacy",
            time=5_000,
            retry={"maxAttempts": 9},
            recovery_loop=True,
        )
    )
    ctx.storage.sql.exec(
        "UPDATE cf_agents_jobs SET time = ? "
        "WHERE id = '__cf_internal_fibers_maintenance'",
        invalid_time,
    )
    observed_times = []

    async def recover(_context):
        observed_times.extend(
            row["time"]
            for row in ctx.storage.sql.exec("SELECT time FROM cf_agents_jobs").toArray()
        )
        return FiberRecoveryResult(status="completed")

    capability = FiberCapability(
        keep_alive_interval_ms=lambda: 100,
        on_fiber_recovered=recover,
    )
    lifecycle = Lifecycle(ctx, host=object())
    lifecycle.use(capability)

    await lifecycle.start()

    assert observed_times == [1_100]
    assert (await capability.inspect_fiber("current")).status == "completed"
    assert ctx.storage.sql.exec("SELECT id FROM cf_agents_runs").toArray() == []
    assert ctx.storage.sql.exec("SELECT id FROM cf_agents_jobs").toArray() == []
    assert ctx.storage.alarm_time_ms is None


@pytest.mark.asyncio
async def test_due_maintenance_job_recovers_before_host_start(monkeypatch):
    monkeypatch.setattr(fiber_module, "now_ms", lambda: 1_000)
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)
    ctx = fakes.FakeCtx()
    setup_capability = FiberCapability(keep_alive_interval_ms=lambda: 100)
    setup = Lifecycle(ctx, host=object())
    setup.use(setup_capability)
    await setup.start()
    ctx.storage.sql.exec(
        "INSERT INTO cf_agents_fibers "
        "(fiber_id, name, status, created_at) "
        "VALUES ('current', 'work', 'pending', 1)"
    )
    ctx.storage.sql.exec(
        "INSERT INTO cf_agents_runs (id, name, snapshot, created_at) "
        "VALUES ('current', 'work', NULL, 1)"
    )
    await setup_capability.lifecycle.jobs.push(
        lifecycle_module.LifecycleJobPushOptions(
            id="__cf_internal_fibers_maintenance",
            fn="legacy",
            time=1_000,
            retry={"maxAttempts": 9},
            recovery_loop=True,
        )
    )
    await ctx.storage.deleteAlarm()
    events = []

    async def recover(context):
        events.append(("recovery", context.id))
        return FiberRecoveryResult(status="completed")

    async def host_start():
        events.append(("host", None))

    capability = FiberCapability(
        keep_alive_interval_ms=lambda: 100,
        on_fiber_recovered=recover,
    )
    lifecycle = Lifecycle(ctx, host=object(), on_start=host_start)
    lifecycle.use(capability)

    await lifecycle.start()

    assert events == [("recovery", "current"), ("host", None)]
    assert ctx.storage.sql.exec("SELECT id FROM cf_agents_runs").toArray() == []
    assert ctx.storage.sql.exec("SELECT id FROM cf_agents_jobs").toArray() == []
    assert ctx.storage.alarm_time_ms is None


@pytest.mark.asyncio
async def test_startup_memory_limit_seals_work_in_same_alarm(monkeypatch):
    monkeypatch.setattr(fiber_module, "now_ms", lambda: 1_000)
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)
    ctx = fakes.FakeCtx()
    setup_capability = FiberCapability(keep_alive_interval_ms=lambda: 100)
    setup = Lifecycle(ctx, host=object())
    setup.use(setup_capability)
    await setup.start()
    ctx.storage.sql.exec(
        "INSERT INTO cf_agents_fibers "
        "(fiber_id, name, status, created_at) "
        "VALUES ('current', 'work', 'pending', 1)"
    )
    ctx.storage.sql.exec(
        "INSERT INTO cf_agents_runs (id, name, snapshot, created_at) "
        "VALUES ('current', 'work', NULL, 1)"
    )
    recovery_calls = 0

    async def recover(_context):
        nonlocal recovery_calls
        recovery_calls += 1
        raise RuntimeError(
            "Durable Object's isolate exceeded its memory limit and was reset."
        )

    capability = FiberCapability(
        keep_alive_interval_ms=lambda: 100,
        on_internal_fiber_recovery=recover,
    )
    lifecycle = Lifecycle(
        ctx,
        host=object(),
        max_alarm_memory_limit_strikes=1,
        reset_alarm=lambda _reason: None,
    )
    lifecycle.use(capability)

    await lifecycle.alarm()

    assert recovery_calls == 1
    assert capability._prepared is True
    assert ctx.storage.sql.exec(
        "SELECT status, error_message, completed_at FROM cf_agents_fibers "
        "WHERE fiber_id = 'current'"
    ).toArray() == [
        {
            "status": "error",
            "error_message": fiber_module._MEMORY_LIMIT_ERROR,
            "completed_at": 1_000,
        }
    ]
    assert ctx.storage.sql.exec("SELECT id FROM cf_agents_runs").toArray() == []
    assert ctx.storage.sql.exec("SELECT id FROM cf_agents_jobs").toArray() == []
    assert ctx.storage.alarm_time_ms is None


@pytest.mark.asyncio
async def test_cold_disposal_terminalizes_persisted_fiber_before_job_cleanup(
    monkeypatch,
):
    monkeypatch.setattr(fiber_module, "now_ms", lambda: 1_000)
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)
    ctx = fakes.FakeCtx()
    setup_capability = FiberCapability(keep_alive_interval_ms=lambda: 100)
    setup = Lifecycle(ctx, host=object())
    setup.use(setup_capability)
    await setup.start()
    ctx.storage.sql.exec(
        "INSERT INTO cf_agents_fibers "
        "(fiber_id, name, status, created_at) "
        "VALUES ('current', 'work', 'pending', 1)"
    )
    ctx.storage.sql.exec(
        "INSERT INTO cf_agents_runs (id, name, snapshot, created_at) "
        "VALUES ('current', 'work', NULL, 1)"
    )
    await setup_capability.lifecycle.jobs.push(
        lifecycle_module.LifecycleJobPushOptions(
            id="__cf_internal_fibers_maintenance",
            fn="__cf_internal_fibers_maintenance",
            time=5_000,
            retry={"maxAttempts": 1},
            recovery_loop=True,
        )
    )
    capability = FiberCapability(keep_alive_interval_ms=lambda: 100)
    lifecycle = Lifecycle(ctx, host=object())
    lifecycle.use(capability)

    await lifecycle.dispose()

    assert ctx.storage.sql.exec(
        "SELECT status, error_message, completed_at FROM cf_agents_fibers"
    ).toArray() == [
        {
            "status": "aborted",
            "error_message": fiber_module._DISPOSED_ERROR,
            "completed_at": 1_000,
        }
    ]
    assert ctx.storage.sql.exec("SELECT id FROM cf_agents_runs").toArray() == []
    assert ctx.storage.sql.exec("SELECT id FROM cf_agents_jobs").toArray() == []
    assert ctx.storage.alarm_time_ms is None


@pytest.mark.asyncio
async def test_cold_facet_disposal_terminalizes_without_maintenance(monkeypatch):
    monkeypatch.setattr(fiber_module, "now_ms", lambda: 1_000)
    ctx = fakes.FakeCtx()
    from agents.lifecycle.fiber_schema import prepare_fiber_schema

    prepare_fiber_schema(
        lambda query, *params: ctx.storage.sql.exec(query, *params).toArray()
    )
    ctx.storage.sql.exec(
        "INSERT INTO cf_agents_fibers "
        "(fiber_id, name, status, created_at) "
        "VALUES ('current', 'work', 'pending', 1)"
    )
    ctx.storage.sql.exec(
        "INSERT INTO cf_agents_runs (id, name, snapshot, created_at) "
        "VALUES ('current', 'work', NULL, 1)"
    )
    capability = FiberCapability(maintenance_enabled=lambda: False)
    lifecycle = Lifecycle(
        ctx,
        host=object(),
        route_address=LifecycleRouteAddress("root/facet", "facet-data"),
        root_route_address=LifecycleRouteAddress("root", "root-data"),
    )
    lifecycle.use(capability)

    await lifecycle.dispose()

    assert ctx.storage.sql.exec(
        "SELECT status, error_message, completed_at FROM cf_agents_fibers"
    ).toArray() == [
        {
            "status": "aborted",
            "error_message": fiber_module._DISPOSED_ERROR,
            "completed_at": 1_000,
        }
    ]
    assert ctx.storage.sql.exec("SELECT id FROM cf_agents_runs").toArray() == []
    assert (
        ctx.storage.sql.exec(
            "SELECT name FROM sqlite_master WHERE name = 'cf_agents_jobs'"
        ).toArray()
        == []
    )
    assert ctx.storage.alarm_history_ms == ()


@pytest.mark.asyncio
async def test_agent_disposal_reports_fiber_cleanup_failure(monkeypatch):
    errors = []

    class ReportingAgent(Agent):
        async def on_error(self, error, connection=None):
            errors.append(error)

    agent = fakes.build_agent(cls=ReportingAgent)
    await agent._ensure_initialized()
    failure = RuntimeError("terminalization failed")

    def fail_terminalization(_reason, *, status):
        raise failure

    monkeypatch.setattr(
        agent._fiber,
        "_terminalize_local_work",
        fail_terminalization,
    )

    await agent._lifecycle.dispose()

    assert errors == [failure]


@pytest.mark.asyncio
async def test_keep_alive_disposer_does_not_retain_during_lifecycle_disposal():
    recorder = fakes.WaitUntilRecorder()
    capability = FiberCapability()
    _ctx, lifecycle = _install(capability, retain_work=recorder)
    dispose_lease = await capability.keep_alive()
    entered = asyncio.Event()
    release = asyncio.Event()

    async def hold_operation():
        entered.set()
        await release.wait()

    holding = asyncio.create_task(
        capability.lifecycle.run_in_host_context(hold_operation)
    )
    await entered.wait()
    disposing = asyncio.create_task(lifecycle.dispose())
    await asyncio.sleep(0)

    assert lifecycle._disposing is True
    dispose_lease()
    assert capability._keep_alive_refs == 0
    assert recorder.records == []

    release.set()
    await holding
    await disposing


@pytest.mark.asyncio
async def test_cold_disposal_failure_preserves_maintenance_wake(monkeypatch):
    monkeypatch.setattr(fiber_module, "now_ms", lambda: 1_000)
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)
    ctx = fakes.FakeCtx()
    setup_capability = FiberCapability(keep_alive_interval_ms=lambda: 100)
    setup = Lifecycle(ctx, host=object())
    setup.use(setup_capability)
    await setup.start()
    ctx.storage.sql.exec(
        "INSERT INTO cf_agents_fibers "
        "(fiber_id, name, status, created_at) "
        "VALUES ('current', 'work', 'pending', 1)"
    )
    ctx.storage.sql.exec(
        "INSERT INTO cf_agents_runs (id, name, snapshot, created_at) "
        "VALUES ('current', 'work', NULL, 1)"
    )
    await setup_capability.lifecycle.jobs.push(
        lifecycle_module.LifecycleJobPushOptions(
            id="__cf_internal_fibers_maintenance",
            fn="__cf_internal_fibers_maintenance",
            time=5_000,
            retry={"maxAttempts": 1},
            recovery_loop=True,
        )
    )
    errors = []

    async def report(error):
        errors.append(error)

    capability = FiberCapability(keep_alive_interval_ms=lambda: 100)
    lifecycle = Lifecycle(ctx, host=object(), on_error=report)
    lifecycle.use(capability)
    original_exec = ctx.storage.sql.exec

    def fail_terminalization(query, *params):
        if query.startswith("UPDATE cf_agents_fibers SET status"):
            raise RuntimeError("terminalization failed")
        return original_exec(query, *params)

    ctx.storage.sql.exec = fail_terminalization

    await lifecycle.dispose()

    assert len(errors) == 1
    assert str(errors[0]) == "terminalization failed"
    assert original_exec(
        "SELECT status FROM cf_agents_fibers WHERE fiber_id = 'current'"
    ).toArray() == [{"status": "pending"}]
    assert original_exec("SELECT id FROM cf_agents_runs").toArray() == [
        {"id": "current"}
    ]
    assert original_exec("SELECT id FROM cf_agents_jobs").toArray() == [
        {"id": "__cf_internal_fibers_maintenance"}
    ]
    assert ctx.storage.alarm_time_ms == 5_000


@pytest.mark.asyncio
async def test_disposal_cancels_fiber_maintenance_and_clears_alarm(monkeypatch):
    monkeypatch.setattr(fiber_module, "now_ms", lambda: 1_000)
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)
    recorder = fakes.WaitUntilRecorder()
    capability = FiberCapability(keep_alive_interval_ms=lambda: 100)
    ctx, lifecycle = _install(capability, retain_work=recorder)
    dispose_lease = await capability.keep_alive()

    await lifecycle.dispose()
    dispose_lease()

    assert ctx.storage.sql.exec("SELECT id FROM cf_agents_jobs").toArray() == []
    assert ctx.storage.alarm_time_ms is None
    assert recorder.records == []


@pytest.mark.asyncio
async def test_disposal_preserves_unrelated_job_and_rearms_its_alarm(monkeypatch):
    monkeypatch.setattr(fiber_module, "now_ms", lambda: 1_000)
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)
    recorder = fakes.WaitUntilRecorder()
    capability = FiberCapability(keep_alive_interval_ms=lambda: 100)
    ctx, lifecycle = _install(capability, retain_work=recorder)
    await lifecycle.jobs.push(
        lifecycle_module.LifecycleJobPushOptions(
            id="unrelated",
            fn="host-work",
            time=5_000,
        )
    )
    await capability.keep_alive()

    await lifecycle.dispose()

    assert ctx.storage.sql.exec(
        "SELECT id, capability FROM cf_agents_jobs"
    ).toArray() == [{"id": "unrelated", "capability": "host"}]
    assert ctx.storage.alarm_time_ms == 5_000


@pytest.mark.asyncio
async def test_disposal_rearms_remaining_host_deadline(monkeypatch):
    monkeypatch.setattr(fiber_module, "now_ms", lambda: 1_000)
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)
    recorder = fakes.WaitUntilRecorder()
    capability = FiberCapability(keep_alive_interval_ms=lambda: 100)
    ctx = fakes.FakeCtx()
    lifecycle = Lifecycle(
        ctx,
        host=object(),
        retain_work=recorder,
        alarm_deadline=lambda _now: 5_000,
    )
    lifecycle.use(capability)
    await capability.keep_alive()

    await lifecycle.dispose()

    assert ctx.storage.sql.exec("SELECT id FROM cf_agents_jobs").toArray() == []
    assert ctx.storage.alarm_time_ms == 5_000


@pytest.mark.asyncio
async def test_disposal_waits_for_fiber_execution_and_cleanup():
    capability = FiberCapability(keep_alive_interval_ms=lambda: 100)
    ctx, lifecycle = _install(capability)
    started = asyncio.Event()
    release = asyncio.Event()

    async def body(_context):
        started.set()
        await release.wait()
        return "complete"

    running = asyncio.create_task(capability.run_fiber("work", body))
    await started.wait()
    disposal = asyncio.create_task(lifecycle.dispose())
    await asyncio.sleep(0)

    assert not disposal.done()
    assert ctx.storage.sql.exec("SELECT id FROM cf_agents_runs").toArray()

    release.set()
    assert await running == "complete"
    await disposal

    assert ctx.storage.sql.exec("SELECT id FROM cf_agents_runs").toArray() == []
    assert capability._disposed is True
    assert capability._fiber_active_ids == set()
    assert capability._keep_alive_refs == 0


@pytest.mark.asyncio
async def test_disposal_fences_retained_runner_waiters_and_late_disposer():
    recorder = fakes.WaitUntilRecorder()
    callback_calls = 0
    capability = FiberCapability(
        detached_fibers_enabled=lambda: True,
        keep_alive_interval_ms=lambda: 100,
    )
    ctx, lifecycle = _install(capability, retain_work=recorder)

    async def body(_context):
        nonlocal callback_calls
        callback_calls += 1

    started = await capability.start_fiber("detached", body)
    waiter = capability._wait_for_managed_fiber_terminal(started.fiber_id)
    dispose_lease = await capability.keep_alive()
    retained_count = len(recorder.records)

    await lifecycle.dispose()
    dispose_lease()
    await waiter
    await recorder.drain()

    row = ctx.storage.sql.exec(
        "SELECT status, error_message FROM cf_agents_fibers WHERE fiber_id = ?",
        started.fiber_id,
    ).toArray()
    assert row == [{"status": "aborted", "error_message": "fiber capability disposed"}]
    assert callback_calls == 0
    assert len(recorder.records) == retained_count
    assert capability._managed_terminal_waiters == {}
    assert capability._active_lease_generations == set()
