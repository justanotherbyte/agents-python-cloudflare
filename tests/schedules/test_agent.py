from __future__ import annotations

import fakes
import pytest

import agents.schedules as schedules_module
from agents import Agent
from agents.lifecycle import get_current_lifecycle_context
from agents.schedules import (
    RetryOptions,
    ScheduleOptions,
    Scheduler,
    scheduler_callback,
)


def test_schedules_module_exports_the_experimental_public_surface():
    assert set(schedules_module.__all__) == {
        "RetryOptions",
        "Schedule",
        "ScheduleCriteria",
        "ScheduleOptions",
        "ScheduleTimeRange",
        "Scheduler",
        "SchedulerCallbacks",
        "SchedulerEventType",
        "SchedulerHandlers",
        "SchedulerOptions",
        "SchedulerPayload",
        "scheduler_callback",
    }


@pytest.mark.asyncio
async def test_agent_installs_scheduler_and_delegates_public_operations():
    class ScheduledAgent(Agent):
        @scheduler_callback()
        async def remind(self, payload, schedule):
            pass

    agent = fakes.build_agent(cls=ScheduledAgent)

    assert isinstance(agent.scheduler, Scheduler)
    scheduled = await agent.schedule(
        60,
        "remind",
        {"message": "check"},
        ScheduleOptions(idempotent=True),
    )
    assert await agent.get_schedule_by_id(scheduled.id) == scheduled
    assert await agent.list_schedules() == (scheduled,)
    assert await agent.cancel_schedule(scheduled.id) is True


@pytest.mark.asyncio
async def test_discovery_honors_shadowing_and_redecoration():
    class Base(Agent):
        @scheduler_callback()
        def inherited(self, payload, schedule):
            pass

        @scheduler_callback()
        def replaced(self, payload, schedule):
            pass

    class Shadowed(Base):
        def inherited(self, payload, schedule):
            pass

        @scheduler_callback()
        def replaced(self, payload, schedule):
            pass

    agent = fakes.build_agent(cls=Shadowed)

    with pytest.raises(ValueError, match="unknown scheduled callback"):
        await agent.schedule(1, "inherited")
    assert (await agent.schedule(1, "replaced")).callback == "replaced"


def test_discovery_does_not_evaluate_descriptors_and_maps_are_fresh():
    evaluated = []

    class ScheduledAgent(Agent):
        @property
        def dangerous(self):
            evaluated.append(True)
            raise RuntimeError("descriptor evaluated")

        @scheduler_callback()
        def callback(self, payload, schedule):
            pass

    first = fakes.build_agent(cls=ScheduledAgent)
    second = fakes.build_agent(cls=ScheduledAgent)

    assert evaluated == []
    assert first.scheduler._callbacks is not second.scheduler._callbacks
    first.scheduler._callbacks.clear()
    assert "callback" in second.scheduler._callbacks


@pytest.mark.asyncio
async def test_static_and_class_method_callbacks_bind_without_descriptor_lookup():
    calls = []

    class ScheduledAgent(Agent):
        @scheduler_callback()
        @staticmethod
        def static(payload, schedule):
            calls.append(("static", payload))

        @scheduler_callback()
        @classmethod
        def class_callback(cls, payload, schedule):
            calls.append((cls.__name__, payload))

    agent = fakes.build_agent(cls=ScheduledAgent)
    static = await agent.schedule(0, "static", 1)
    class_callback = await agent.schedule(0, "class_callback", 2)

    await agent.alarm()

    assert sorted(calls) == [("ScheduledAgent", 2), ("static", 1)]
    assert await agent.get_schedule_by_id(static.id) is None
    assert await agent.get_schedule_by_id(class_callback.id) is None


@pytest.mark.asyncio
async def test_scheduler_errors_reach_agent_on_error_in_host_context():
    observed = []

    class ScheduledAgent(Agent):
        @scheduler_callback()
        async def fail(self, payload, schedule):
            raise ValueError("failed")

        async def on_error(self, error, connection=None):
            observed.append((error, connection, get_current_lifecycle_context().host))

    agent = fakes.build_agent(cls=ScheduledAgent)
    await agent.schedule(
        0,
        "fail",
        options=ScheduleOptions(retry=RetryOptions(max_attempts=1)),
    )

    await agent.alarm()

    assert len(observed) == 1
    error, connection, host = observed[0]
    assert isinstance(error, ValueError)
    assert connection is None
    assert host is agent
