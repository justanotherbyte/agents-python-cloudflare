from __future__ import annotations

import asyncio
import types

import fakes
import pytest

import agents.lifecycle._runtime as lifecycle_module
import agents.lifecycle.fiber as fiber_module
from agents import Agent, FiberRecoveryResult


@pytest.fixture
def short_heartbeat_agent(monkeypatch):
    monkeypatch.setattr(fiber_module, "now_ms", lambda: 1_000)
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)
    agent = fakes.build_agent()
    agent.keep_alive_interval_ms = 100
    return agent


def _seed_orphan(agent, fiber_id: str = "orphan") -> None:
    agent.sql(
        "INSERT INTO cf_agents_runs (id, name, snapshot, created_at) "
        "VALUES (?, ?, NULL, ?)",
        fiber_id,
        "job",
        1,
    )


@pytest.mark.asyncio
async def test_nested_leases_arm_once_and_last_idempotent_disposal_rearms(
    short_heartbeat_agent, wait_until
):
    agent = short_heartbeat_agent

    dispose_first = await agent.keep_alive()
    dispose_second = await agent.keep_alive()

    assert agent._fiber._keep_alive_refs == 2
    assert agent.ctx.storage.alarm_time_ms == 1_100
    assert agent.ctx.storage.alarm_history_ms == (
        ("delete", None),
        ("set", 1_100),
    )

    dispose_first()
    dispose_first()
    assert agent._fiber._keep_alive_refs == 1
    assert wait_until.records == []

    dispose_second()
    assert agent._fiber._keep_alive_refs == 0
    assert len(wait_until.coros) == 1
    await wait_until.drain()

    assert agent.ctx.storage.alarm_time_ms is None
    assert agent.ctx.storage.alarm_history_ms == (
        ("delete", None),
        ("set", 1_100),
        ("delete", None),
    )

    dispose_second()
    assert len(wait_until.records) == 1
    assert agent.ctx.storage.alarm_history_ms == (
        ("delete", None),
        ("set", 1_100),
        ("delete", None),
    )


@pytest.mark.asyncio
async def test_last_disposal_preserves_earlier_alarm_than_recovery_deadline(
    short_heartbeat_agent, wait_until
):
    agent = short_heartbeat_agent
    await agent._ensure_initialized()
    agent.fiber_recovery_max_backoff_ms = 250
    agent._fiber._recovery_no_progress_scans = 2
    _seed_orphan(agent)

    dispose = await agent.keep_alive()
    assert agent.ctx.storage.alarm_time_ms == 1_100

    dispose()
    await wait_until.drain()

    assert agent.ctx.storage.alarm_time_ms == 1_100
    assert agent.ctx.storage.alarm_history_ms == (
        ("delete", None),
        ("set", 1_100),
        ("set", 1_100),
    )


@pytest.mark.asyncio
async def test_recovery_deadline_wins_when_earlier_than_heartbeat(
    short_heartbeat_agent, wait_until
):
    agent = short_heartbeat_agent
    await agent._ensure_initialized()
    agent.fiber_recovery_max_backoff_ms = 40
    _seed_orphan(agent)

    dispose = await agent.keep_alive()

    assert agent.ctx.storage.alarm_time_ms == 1_040
    dispose()
    await wait_until.drain()
    assert agent.ctx.storage.alarm_history_ms == (
        ("delete", None),
        ("set", 1_040),
        ("set", 1_040),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("fails", [False, True], ids=["return", "error"])
async def test_keep_alive_while_releases_on_return_and_error(
    short_heartbeat_agent, wait_until, fails
):
    agent = short_heartbeat_agent
    observed = []

    async def body():
        observed.append(
            (agent._fiber._keep_alive_refs, agent.ctx.storage.alarm_time_ms)
        )
        if fails:
            raise RuntimeError("work failed")
        return "result"

    if fails:
        with pytest.raises(RuntimeError, match="work failed"):
            await agent.keep_alive_while(body)
    else:
        assert await agent.keep_alive_while(body) == "result"

    assert observed == [(1, 1_100)]
    assert agent._fiber._keep_alive_refs == 0
    assert wait_until.coros == ()
    assert agent.ctx.storage.alarm_history_ms == (
        ("delete", None),
        ("set", 1_100),
        ("delete", None),
    )


@pytest.mark.asyncio
async def test_keep_alive_while_releases_on_cancellation(
    short_heartbeat_agent, wait_until
):
    agent = short_heartbeat_agent
    started = asyncio.Event()
    release = asyncio.Event()

    async def body():
        started.set()
        await release.wait()

    running = asyncio.create_task(agent.keep_alive_while(body))
    try:
        await asyncio.wait_for(started.wait(), timeout=1)
        assert agent._fiber._keep_alive_refs == 1
        assert agent.ctx.storage.alarm_time_ms == 1_100

        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running

        assert agent._fiber._keep_alive_refs == 0
        assert wait_until.coros == ()
        assert agent.ctx.storage.alarm_history_ms == (
            ("delete", None),
            ("set", 1_100),
            ("delete", None),
        )
    finally:
        release.set()
        if not running.done():
            running.cancel()
        await asyncio.gather(running, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("fails", [False, True], ids=["return", "error"])
async def test_run_fiber_releases_its_lease_on_return_and_error(
    short_heartbeat_agent, fails
):
    agent = short_heartbeat_agent
    observed = []

    async def body(_ctx):
        observed.append(
            (agent._fiber._keep_alive_refs, agent.ctx.storage.alarm_time_ms)
        )
        if fails:
            raise RuntimeError("fiber failed")
        return "result"

    if fails:
        with pytest.raises(RuntimeError, match="fiber failed"):
            await agent.run_fiber("job", body)
    else:
        assert await agent.run_fiber("job", body) == "result"

    assert observed == [(1, 1_100)]
    assert agent._fiber._keep_alive_refs == 0
    assert agent.sql("SELECT id FROM cf_agents_runs") == []
    assert agent.ctx.storage.alarm_time_ms is None
    assert agent.ctx.storage.alarm_history_ms == (
        ("delete", None),
        ("set", 1_100),
        ("delete", None),
    )


@pytest.mark.asyncio
async def test_eviction_discards_an_outstanding_in_memory_lease(monkeypatch):
    monkeypatch.setattr(fiber_module, "now_ms", lambda: 1_000)
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)
    runtime = fakes.FakeDurableObjectRuntime()
    agent = Agent(runtime.new_context(), types.SimpleNamespace())
    agent.keep_alive_interval_ms = 100

    await agent.keep_alive()
    assert agent._fiber._keep_alive_refs == 1
    assert agent.ctx.storage.alarm_time_ms == 1_100

    reincarnated = Agent(runtime.evict(), types.SimpleNamespace())

    assert reincarnated._fiber._keep_alive_refs == 0
    assert reincarnated.ctx.storage.alarm_time_ms == 1_100


@pytest.mark.asyncio
async def test_cancelled_fiber_releases_lease_and_recovers_after_eviction(monkeypatch):
    current_time = 1_000
    monkeypatch.setattr(fiber_module, "now_ms", lambda: current_time)
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: current_time)
    runtime = fakes.FakeDurableObjectRuntime()
    started = asyncio.Event()
    fiber_ids = []
    recovered = []

    class RecoveryAgent(Agent):
        async def on_fiber_recovered(self, ctx):
            recovered.append(ctx.id)

    agent = RecoveryAgent(runtime.new_context(), types.SimpleNamespace())
    agent.keep_alive_interval_ms = 100

    async def body(ctx):
        fiber_ids.append(ctx.id)
        started.set()
        await asyncio.Event().wait()

    running = asyncio.create_task(agent.run_fiber("job", body))
    try:
        await asyncio.wait_for(started.wait(), timeout=1)
        assert agent._fiber._keep_alive_refs == 1
        assert agent.ctx.storage.alarm_time_ms == 1_100

        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running

        assert agent._fiber._keep_alive_refs == 0
        assert agent.sql("SELECT id FROM cf_agents_runs") == [{"id": fiber_ids[0]}]
        assert agent.ctx.storage.alarm_time_ms == 1_100

        reincarnated = RecoveryAgent(runtime.evict(), types.SimpleNamespace())
        reincarnated.keep_alive_interval_ms = 100
        assert reincarnated._fiber._keep_alive_refs == 0
        assert reincarnated.ctx.storage.alarm_time_ms == 1_100

        current_time = 1_100
        await reincarnated.alarm()

        assert recovered == fiber_ids
        assert reincarnated.sql("SELECT id FROM cf_agents_runs") == []
        assert reincarnated.ctx.storage.alarm_time_ms is None
        assert (
            reincarnated.sql(
                "SELECT id FROM cf_agents_jobs WHERE capability = 'fibers'"
            )
            == []
        )
    finally:
        if not running.done():
            running.cancel()
        await asyncio.gather(running, return_exceptions=True)


@pytest.mark.asyncio
async def test_cancellation_during_first_alarm_arm_preserves_recovery_intent(
    monkeypatch,
):
    monkeypatch.setattr(fiber_module, "now_ms", lambda: 1_000)
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)
    runtime = fakes.FakeDurableObjectRuntime()
    recovered = []

    class RecoveryAgent(Agent):
        async def on_fiber_recovered(self, context):
            recovered.append(context.id)
            return FiberRecoveryResult(status="completed")

    agent = RecoveryAgent(runtime.new_context(), types.SimpleNamespace())
    agent.keep_alive_interval_ms = 100
    gate = fakes.AsyncGate()
    agent.ctx.storage.pause_next_alarm(gate)

    async def body(_context):
        raise AssertionError("body should not start before the alarm arm")

    running = asyncio.create_task(
        agent.start_fiber("managed", body, wait_for_completion=True)
    )
    await gate.wait_until_blocked()
    running.cancel()
    await asyncio.sleep(0)
    gate.release()
    with pytest.raises(asyncio.CancelledError):
        await running

    ledger = agent.sql(
        "SELECT fiber_id, status FROM cf_agents_fibers WHERE name = 'managed'"
    )
    assert len(ledger) == 1
    fiber_id = ledger[0]["fiber_id"]
    assert ledger[0]["status"] == "running"
    assert agent.sql("SELECT id FROM cf_agents_runs") == [{"id": fiber_id}]
    assert agent.sql("SELECT id FROM cf_agents_jobs WHERE capability = 'fibers'") == [
        {"id": "__cf_internal_fibers_maintenance"}
    ]
    assert agent.ctx.storage.alarm_time_ms == 1_100
    assert agent._fiber._fiber_active_ids == set()

    reincarnated = RecoveryAgent(runtime.evict(), types.SimpleNamespace())
    reincarnated.keep_alive_interval_ms = 100
    await reincarnated._ensure_initialized()

    assert recovered == [fiber_id]
    assert reincarnated.sql("SELECT id FROM cf_agents_runs") == []
    assert reincarnated.sql("SELECT id FROM cf_agents_jobs") == []
    assert (await reincarnated.inspect_fiber(fiber_id)).status == "completed"


@pytest.mark.asyncio
async def test_detached_acceptance_survives_eviction_before_runner_starts(
    monkeypatch,
    wait_until,
):
    current_time = 1_000
    monkeypatch.setattr(fiber_module, "now_ms", lambda: current_time)
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: current_time)
    runtime = fakes.FakeDurableObjectRuntime()
    recovered = []
    body_calls = 0

    class RecoveryAgent(Agent):
        async def on_fiber_recovered(self, context):
            recovered.append(context.id)
            return FiberRecoveryResult(status="completed")

    agent = RecoveryAgent(runtime.new_context(), types.SimpleNamespace())
    agent.detached_fibers_enabled = True
    agent.keep_alive_interval_ms = 100

    async def body(_context):
        nonlocal body_calls
        body_calls += 1

    accepted = await agent.start_fiber("detached", body)

    assert body_calls == 0
    assert len(wait_until.coros) == 1
    assert agent.sql(
        "SELECT fiber_id, status FROM cf_agents_fibers WHERE fiber_id = ?",
        accepted.fiber_id,
    ) == [{"fiber_id": accepted.fiber_id, "status": "pending"}]
    assert agent.sql("SELECT id FROM cf_agents_jobs") == [
        {"id": "__cf_internal_fibers_maintenance"}
    ]
    assert agent.ctx.storage.alarm_time_ms == 1_100

    reincarnated = RecoveryAgent(runtime.evict(), types.SimpleNamespace())
    reincarnated.detached_fibers_enabled = True
    reincarnated.keep_alive_interval_ms = 100
    current_time = 1_100
    await reincarnated.alarm()

    assert recovered == [accepted.fiber_id]
    assert body_calls == 0
    assert reincarnated.sql("SELECT id FROM cf_agents_runs") == []
    assert reincarnated.sql("SELECT id FROM cf_agents_jobs") == []
    assert (await reincarnated.inspect_fiber(accepted.fiber_id)).status == "completed"


@pytest.mark.asyncio
async def test_cancelled_detached_acceptance_preserves_durable_recovery(
    monkeypatch,
    wait_until,
):
    current_time = 1_000
    monkeypatch.setattr(fiber_module, "now_ms", lambda: current_time)
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: current_time)
    runtime = fakes.FakeDurableObjectRuntime()
    recovered = []
    body_calls = 0

    class RecoveryAgent(Agent):
        async def on_fiber_recovered(self, context):
            recovered.append(context.id)
            return FiberRecoveryResult(status="completed")

    agent = RecoveryAgent(runtime.new_context(), types.SimpleNamespace())
    agent.detached_fibers_enabled = True
    agent.keep_alive_interval_ms = 100
    gate = fakes.AsyncGate()
    agent.ctx.storage.pause_next_alarm(gate)

    async def body(_context):
        nonlocal body_calls
        body_calls += 1

    accepting = asyncio.create_task(agent.start_fiber("detached", body))
    await gate.wait_until_blocked()
    accepting.cancel()
    await asyncio.sleep(0)
    gate.release()
    with pytest.raises(asyncio.CancelledError):
        await accepting

    ledger = agent.sql(
        "SELECT fiber_id, status FROM cf_agents_fibers WHERE name = 'detached'"
    )
    assert len(ledger) == 1
    fiber_id = ledger[0]["fiber_id"]
    assert ledger[0]["status"] == "pending"
    assert body_calls == 0
    assert wait_until.coros == ()
    assert agent.sql("SELECT id FROM cf_agents_jobs") == [
        {"id": "__cf_internal_fibers_maintenance"}
    ]
    assert agent.ctx.storage.alarm_time_ms == 1_100

    reincarnated = RecoveryAgent(runtime.evict(), types.SimpleNamespace())
    reincarnated.detached_fibers_enabled = True
    reincarnated.keep_alive_interval_ms = 100
    current_time = 1_100
    await reincarnated.alarm()

    assert recovered == [fiber_id]
    assert reincarnated.sql("SELECT id FROM cf_agents_runs") == []
    assert reincarnated.sql("SELECT id FROM cf_agents_jobs") == []
    assert (await reincarnated.inspect_fiber(fiber_id)).status == "completed"


@pytest.mark.asyncio
async def test_maintenance_job_recovers_an_orphan_after_startup(monkeypatch):
    current_time = 1_000
    monkeypatch.setattr(fiber_module, "now_ms", lambda: current_time)
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: current_time)
    recovered = []

    class RecoveryAgent(Agent):
        async def on_fiber_recovered(self, ctx):
            recovered.append(ctx.id)

    agent = fakes.build_agent(cls=RecoveryAgent)
    agent.keep_alive_interval_ms = 100
    await agent._ensure_initialized()
    agent._fiber._recovery_no_progress_scans = 0
    _seed_orphan(agent)
    await agent._fiber._synchronize_maintenance_intent()

    current_time = 1_100
    await agent.alarm()

    assert recovered == ["orphan"]
    assert agent.sql("SELECT id FROM cf_agents_runs") == []
    assert agent.ctx.storage.alarm_time_ms is None
    assert agent.ctx.storage.alarm_history_ms[-2:] == (
        ("set", 31_100),
        ("delete", None),
    )
