from __future__ import annotations

import types

import fakes
import pytest
from workers import DurableObject

from agents import AIChatAgent, Agent
from agents.chat.resumable_stream import StreamStorageUnavailable
from agents.core.agent import STATE_ROW_ID
from agents.core.agent_tools import AgentToolRuns
from agents.lifecycle import Lifecycle
from agents.lifecycle.fiber import FiberCapability
from agents.lifecycle.websockets import WebSockets
from agents.schedules import Scheduler
from agents.sessions import Sessions
from agents.tasks import Tasks


@pytest.mark.parametrize("agent_cls", [Agent, AIChatAgent])
def test_public_constructors_survive_complete_sql_outage(agent_cls, monkeypatch):
    ctx = fakes.FakeCtx()
    failures = 0

    def fail_sql(_query, *_params):
        nonlocal failures
        failures += 1
        raise RuntimeError("storage unavailable")

    monkeypatch.setattr(ctx.storage.sql, "exec", fail_sql)

    agent = agent_cls(ctx, types.SimpleNamespace())

    assert failures == 0
    assert agent.state == {}
    if isinstance(agent, AIChatAgent):
        assert agent.messages == []
        assert isinstance(agent.sessions, Sessions)
        assert agent.sessions.lifecycle._lifecycle is agent._lifecycle


@pytest.mark.asyncio
async def test_chat_stream_preparation_failure_retries_on_next_startup(monkeypatch):
    ctx = fakes.FakeCtx()
    agent = AIChatAgent(ctx, types.SimpleNamespace())
    original_exec = ctx.storage.sql.exec
    attempts = 0

    def fail_stream_prepare_once(query, *params):
        nonlocal attempts
        if "CREATE TABLE IF NOT EXISTS cf_ai_chat_stream_chunks" in query:
            attempts += 1
            if attempts == 1:
                raise RuntimeError("storage unavailable")
        return original_exec(query, *params)

    monkeypatch.setattr(ctx.storage.sql, "exec", fail_stream_prepare_once)

    with pytest.raises(StreamStorageUnavailable, match="stream storage read failed"):
        await agent._ensure_initialized()

    await agent._ensure_initialized()

    assert attempts == 2
    assert agent._lifecycle._ready is True


def test_unnamed_durable_object_id_keeps_constructor_reachable():
    ctx = fakes.FakeCtx()
    ctx.id.name = None

    agent = Agent(ctx, types.SimpleNamespace())

    assert agent._lifecycle._route_address is None
    assert agent._lifecycle._root_route_address is None


def test_state_read_failure_keeps_initial_state_reachable(monkeypatch):
    class InitialStateAgent(Agent):
        def initial_state(self):
            return {"ready": True}

    ctx = fakes.FakeCtx()
    original_exec = ctx.storage.sql.exec
    failures = 0

    def fail_state_read(query, *params):
        nonlocal failures
        if query.startswith("SELECT state") and params == (STATE_ROW_ID,):
            failures += 1
            raise RuntimeError("state read failed")
        return original_exec(query, *params)

    monkeypatch.setattr(ctx.storage.sql, "exec", fail_state_read)

    agent = InitialStateAgent(ctx, types.SimpleNamespace())

    assert failures == 0
    agent._hydrate_state()
    assert failures == 1
    assert agent.state == {"ready": True}


@pytest.mark.parametrize(
    "stored",
    [None, "not json", "null", "7", "[]"],
    ids=["sql-null", "malformed", "json-null", "scalar", "array"],
)
def test_corrupt_state_rows_do_not_block_reincarnation(stored):
    class InitialStateAgent(Agent):
        def initial_state(self):
            return {"fallback": True}

    conn = fakes.new_sqlite()
    fakes.build_agent(conn=conn)
    conn.execute(
        "INSERT OR REPLACE INTO cf_agents_state (id, state) VALUES (?, ?)",
        (STATE_ROW_ID, stored),
    )

    reincarnated = fakes.build_agent(cls=InitialStateAgent, conn=conn)

    assert reincarnated.state == {"fallback": True}


def test_agent_is_a_direct_durable_object_composition_root():
    agent = fakes.build_agent()

    assert Agent.__bases__ == (DurableObject,)
    assert isinstance(agent._lifecycle, Lifecycle)
    assert isinstance(agent._websockets, WebSockets)
    assert isinstance(agent._fiber, FiberCapability)
    assert isinstance(agent._agent_tool_runs, AgentToolRuns)
    assert isinstance(agent.scheduler, Scheduler)
    assert isinstance(agent.tasks, Tasks)
    assert agent._websockets.lifecycle._lifecycle is agent._lifecycle
    assert agent._fiber.lifecycle._lifecycle is agent._lifecycle
    assert agent.scheduler.lifecycle._lifecycle is agent._lifecycle
    assert agent.tasks.lifecycle._lifecycle is agent._lifecycle


@pytest.mark.parametrize(
    "name",
    [
        "_cf_route_lifecycle",
        "alarm",
        "webSocketMessage",
        "webSocketClose",
        "webSocketError",
    ],
)
def test_reserved_runtime_handlers_cannot_be_replaced(name):
    with pytest.raises(TypeError, match=name):
        type("CollidingAgent", (Agent,), {name: property(lambda self: None)})


def test_reserved_runtime_handlers_cannot_come_from_left_hand_mixins():
    class AlarmMixin:
        async def alarm(self): ...

    with pytest.raises(TypeError, match="alarm"):
        type("CollidingAgent", (AlarmMixin, Agent), {})


@pytest.mark.parametrize("behavior", ["raises", "wrong-shape"])
def test_invalid_initial_state_does_not_block_construction(behavior):
    class InvalidInitialStateAgent(Agent):
        def initial_state(self):
            if behavior == "raises":
                raise RuntimeError("initial state failed")
            return []

    agent = fakes.build_agent(cls=InvalidInitialStateAgent)

    assert agent.state == {}


def test_socket_hydration_keeps_valid_neighbors_around_safe_malformed_entries():
    first = fakes.FakeSocket({"id": "first", "state": {}, "tags": []})
    no_attachment = fakes.FakeSocket()
    non_string_id = fakes.FakeSocket({"id": 7, "state": {}, "tags": []})
    second = fakes.FakeSocket({"id": "second", "state": {}, "tags": []})
    ctx = fakes.FakeCtx(websockets=[first, no_attachment, non_string_id, second])

    agent = Agent(ctx, types.SimpleNamespace())

    assert [connection.id for connection in agent.get_connections()] == [
        "first",
        "second",
    ]


def test_construction_does_not_run_lifecycle_hooks_or_outbound_work(wait_until):
    calls = []

    class QuietChatAgent(AIChatAgent):
        def _on_startup(self):
            calls.append("framework-startup")

        def on_start(self):
            calls.append("start")

        def on_connect(self, connection, ctx):
            calls.append("connect")

        def on_message(self, connection, message):
            calls.append("message")

        def on_close(self, connection, code, reason, was_clean):
            calls.append("close")

        def on_error(self, error, connection=None):
            calls.append("error")

        def on_request(self, request):
            calls.append("request")

        def on_alarm(self):
            calls.append("alarm")

        def get_connection_tags(self, connection, ctx):
            calls.append("connection-tags")
            return []

        def on_chat_message(self, options):
            calls.append("chat")

        def on_fiber_recovered(self, ctx):
            calls.append("fiber")

    socket = fakes.FakeSocket({"id": "socket-1", "state": {}, "tags": []})
    ctx = fakes.FakeCtx(websockets=[socket])

    QuietChatAgent(ctx, types.SimpleNamespace())

    assert calls == []
    assert socket.sent == []
    assert ctx.accepted_websockets == []
    assert ctx.storage.alarm_history_ms == ()
    assert wait_until.records == []
