from __future__ import annotations

import fakes
import pytest
from workers import Request, Response

import agents.lifecycle._runtime as lifecycle_module
import agents.lifecycle.fiber as fiber_module
from agents import Agent
from agents.core.error import HookError
from agents.core.protocol import MessageType
from agents.lifecycle.websockets import ConnectionContext


@pytest.mark.asyncio
async def test_incomplete_websocket_headers_reach_the_request_hook():
    class UserAgent(Agent):
        async def on_request(self, request):
            return Response("ordinary")

    agent = fakes.build_agent(cls=UserAgent)

    response = await agent.fetch(
        Request("https://example.com/", headers={"Upgrade": "websocket"})
    )

    assert response.status == 200
    assert response.body == "ordinary"


@pytest.mark.asyncio
async def test_connect_override_cannot_skip_framework_handshake():
    class UserAgent(Agent):
        def __init__(self, ctx, env):
            super().__init__(ctx, env)
            self.connected = False

        async def on_connect(self, connection, ctx):
            self.connected = True

    agent = fakes.build_agent(cls=UserAgent)
    connection = fakes.FakeConnection()
    ctx = ConnectionContext(Request())

    await agent._dispatch_connect(connection, ctx)

    assert [frame["type"] for frame in connection.frames[:3]] == [
        MessageType.CF_AGENT_IDENTITY,
        MessageType.CF_AGENT_STATE,
        MessageType.CF_AGENT_MCP_SERVERS,
    ]
    assert agent.connected is True


@pytest.mark.asyncio
async def test_alarm_rearms_when_user_hook_fails():
    class FailingAlarm(Agent):
        async def on_error(self, error, connection=None):
            pass

        async def on_alarm(self):
            raise ValueError("alarm failed")

    agent = fakes.build_agent(cls=FailingAlarm)
    await agent.keep_alive()

    with pytest.raises(HookError):
        await agent._alarm_impl()

    assert agent.ctx.storage.alarm_time_ms is not None


@pytest.mark.asyncio
async def test_alarm_rearm_failure_does_not_mask_hook_failure():
    class FailingAlarmAndSchedule(Agent):
        def __init__(self, ctx, env):
            super().__init__(ctx, env)
            self.errors = []

        async def on_error(self, error, connection=None):
            self.errors.append(error)

        async def on_alarm(self):
            raise ValueError("alarm failed")

    agent = fakes.build_agent(cls=FailingAlarmAndSchedule)
    await agent.keep_alive()

    async def fail_rearm(_deadline):
        raise RuntimeError("schedule failed")

    agent.ctx.storage.setAlarm = fail_rearm

    with pytest.raises(HookError) as excinfo:
        await agent._alarm_impl()

    assert str(excinfo.value.original) == "alarm failed"
    assert len(agent.errors) == 2
    assert agent.errors[0] is excinfo.value
    assert isinstance(agent.errors[1], RuntimeError)
    assert str(agent.errors[1]) == "schedule failed"


@pytest.mark.asyncio
async def test_standalone_alarm_rearm_failure_propagates():
    agent = fakes.build_agent()
    await agent.keep_alive()

    async def fail_rearm(_deadline):
        raise RuntimeError("schedule failed")

    agent.ctx.storage.setAlarm = fail_rearm

    with pytest.raises(RuntimeError, match="schedule failed"):
        await agent._alarm_impl()


@pytest.mark.asyncio
async def test_public_alarm_seals_startup_fiber_memory_failure(monkeypatch):
    monkeypatch.setattr(fiber_module, "now_ms", lambda: 1_000)
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)

    class MemoryRecoveryAgent(Agent):
        recovery_calls = 0

        async def on_fiber_recovered(self, _context):
            self.recovery_calls += 1
            raise RuntimeError(
                "Durable Object's isolate exceeded its memory limit and was reset."
            )

    agent = fakes.build_agent(cls=MemoryRecoveryAgent)
    agent.keep_alive_interval_ms = 100
    agent._lifecycle._max_alarm_memory_limit_strikes = 1
    agent._lifecycle._reset_alarm = lambda _reason: None
    agent.sql(
        "INSERT INTO cf_agents_fibers "
        "(fiber_id, name, status, created_at) "
        "VALUES ('current', 'work', 'pending', 1)"
    )
    agent.sql(
        "INSERT INTO cf_agents_runs (id, name, snapshot, created_at) "
        "VALUES ('current', 'work', NULL, 1)"
    )

    await agent.alarm()

    assert agent.recovery_calls == 1
    assert agent.sql(
        "SELECT status, error_message, completed_at FROM cf_agents_fibers "
        "WHERE fiber_id = 'current'"
    ) == [
        {
            "status": "error",
            "error_message": fiber_module._MEMORY_LIMIT_ERROR,
            "completed_at": 1_000,
        }
    ]
    assert agent.sql("SELECT id FROM cf_agents_runs") == []
    assert agent.sql("SELECT id FROM cf_agents_jobs") == []
    assert agent.ctx.storage.alarm_time_ms is None
    assert agent._alarm_entry_in_progress is False


@pytest.mark.asyncio
async def test_agent_wires_the_host_memory_limit_hook():
    policies = []

    class MemoryPolicyAgent(Agent):
        async def on_alarm_memory_limit(self, context):
            policies.append(context)

    agent = fakes.build_agent(cls=MemoryPolicyAgent)
    context = lifecycle_module.LifecycleMemoryLimitContext(
        sealed=True,
        next_time=None,
        executing=None,
        purged_recovery_loop_jobs=(),
    )

    await agent._lifecycle._notify_host_memory_limit(context)

    assert policies == [context]


def test_agent_does_not_evaluate_the_memory_limit_hook_during_construction():
    class RaisingDescriptor:
        def __get__(self, instance, owner):
            raise RuntimeError("descriptor evaluated")

    class DescriptorAgent(Agent):
        on_alarm_memory_limit = RaisingDescriptor()

    agent = fakes.build_agent(cls=DescriptorAgent)

    assert isinstance(agent, DescriptorAgent)


@pytest.mark.asyncio
async def test_public_host_alarm_preserves_future_fiber_job(monkeypatch):
    monkeypatch.setattr(fiber_module, "now_ms", lambda: 1_000)
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)

    class HostDeadlineAgent(Agent):
        recovery_calls = 0

        def _collect_alarm_deadline(self, now, current):
            deadline = super()._collect_alarm_deadline(now, current)
            return 1_000 if deadline is None else min(deadline, 1_000)

        async def on_fiber_recovered(self, _context):
            self.recovery_calls += 1

    agent = fakes.build_agent(cls=HostDeadlineAgent)
    agent.sql(
        "INSERT INTO cf_agents_fibers "
        "(fiber_id, name, status, created_at) "
        "VALUES ('current', 'work', 'pending', 1)"
    )
    agent.sql(
        "INSERT INTO cf_agents_runs (id, name, snapshot, created_at) "
        "VALUES ('current', 'work', NULL, 1)"
    )
    agent._lifecycle._job_queue.prepare()
    agent.sql(
        "INSERT INTO cf_agents_jobs "
        "(id, capability, fn, time, retry_options, recovery_loop) VALUES "
        "('__cf_internal_fibers_maintenance', 'fibers', 'legacy', 5_000, "
        "'{\"maxAttempts\":9}', 1)"
    )
    await agent.ctx.storage.setAlarm(1_000)

    await agent.alarm()

    assert agent.recovery_calls == 0
    assert agent.sql(
        "SELECT fn, time, payload, retry_options, recovery_loop, running "
        "FROM cf_agents_jobs"
    ) == [
        {
            "fn": "__cf_internal_fibers_maintenance",
            "time": 5_000,
            "payload": None,
            "retry_options": '{"maxAttempts":1}',
            "recovery_loop": 1,
            "running": 0,
        }
    ]
    assert agent.sql(
        "SELECT fiber_id, status FROM cf_agents_fibers WHERE fiber_id = 'current'"
    ) == [{"fiber_id": "current", "status": "pending"}]
    assert agent.sql("SELECT id FROM cf_agents_runs") == [{"id": "current"}]
    assert agent._alarm_entry_in_progress is False


@pytest.mark.asyncio
async def test_on_start_lease_does_not_postpone_due_recovery(
    monkeypatch,
    wait_until,
):
    monkeypatch.setattr(fiber_module, "now_ms", lambda: 1_000)
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)
    events = []
    conn = fakes.new_sqlite()
    seed = fakes.build_agent(conn=conn)
    seed._lifecycle._job_queue.prepare()
    seed.sql(
        "INSERT INTO cf_agents_fibers "
        "(fiber_id, name, status, created_at) "
        "VALUES ('current', 'work', 'pending', 1)"
    )
    seed.sql(
        "INSERT INTO cf_agents_runs (id, name, snapshot, created_at) "
        "VALUES ('current', 'work', NULL, 1)"
    )
    seed.sql(
        "INSERT INTO cf_agents_jobs "
        "(id, capability, fn, time, retry_options, recovery_loop) VALUES "
        "('__cf_internal_fibers_maintenance', 'fibers', "
        "'__cf_internal_fibers_maintenance', 1_000, "
        "'{\"maxAttempts\":1}', 1)"
    )

    class LeaseOnStartAgent(Agent):
        recovery_calls = 0

        async def on_start(self):
            events.append("start")
            dispose = await self.keep_alive()
            dispose()

        async def on_fiber_recovered(self, _context):
            self.recovery_calls += 1
            events.append("recovery")
            return fiber_module.FiberRecoveryResult(status="completed")

    agent = fakes.build_agent(cls=LeaseOnStartAgent, conn=conn)
    agent.keep_alive_interval_ms = 100

    await agent.alarm()
    await wait_until.drain()

    assert agent.recovery_calls == 1
    assert events == ["recovery", "start"]
    assert agent.sql(
        "SELECT status FROM cf_agents_fibers WHERE fiber_id = 'current'"
    ) == [{"status": "completed"}]
    assert agent.sql("SELECT id FROM cf_agents_runs") == []
    assert agent.sql("SELECT id FROM cf_agents_jobs") == []
    assert agent.ctx.storage.alarm_time_ms is None
