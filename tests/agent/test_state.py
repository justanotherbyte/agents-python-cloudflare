"""State sync: `_handle_state_update`, `_set_state_internal`, the `state`
property, and `_hydrate_state`.

Storage is real in-memory sqlite, so persistence, the copy-on-read guard, and
the degrade-don't-raise hydrate rules are exercised for real. A client update
broadcasts to everyone but the sender (its own copy is already optimistic); a
server update reaches everyone. A non-dict or NaN state is rejected before it
can reach storage — storing it would make every later state read raise and lock
clients out (see AGENTS.md).
"""

from __future__ import annotations

import types

import fakes

from agents import Agent
from agents.core.agent import STATE_ROW_ID
from agents.core.protocol import state_error_frame, state_frame


def _agent_with_two_connections():
    agent = fakes.build_agent()
    sender = fakes.FakeConnection(id="sender")
    other = fakes.FakeConnection(id="other")
    agent._connections[sender.id] = sender
    agent._connections[other.id] = other
    return agent, sender, other


# ── client and server updates ────────────────────────────────────────────────


def test_client_update_persists_and_broadcasts_excluding_sender():
    agent, sender, other = _agent_with_two_connections()

    agent._handle_state_update(sender, {"state": {"a": 1}})

    assert agent.state == {"a": 1}
    # The client applies its own set_state optimistically, so echoing the frame
    # back to the sender would fight its local copy.
    assert other.frames == [state_frame({"a": 1})]
    assert sender.sent == []


def test_server_update_broadcasts_to_everyone():
    agent, sender, other = _agent_with_two_connections()

    agent.set_state({"a": 2})

    assert agent.state == {"a": 2}
    assert state_frame({"a": 2}) in sender.frames
    assert state_frame({"a": 2}) in other.frames


# ── rejection paths ──────────────────────────────────────────────────────────


def test_non_dict_update_is_rejected_and_leaves_state_unchanged():
    agent, sender, _ = _agent_with_two_connections()
    before = agent.state

    agent._handle_state_update(sender, {"state": 123})

    # A fixed string, so nothing about why it failed leaks back to the client.
    assert state_error_frame("State update rejected") in sender.frames
    assert agent.state == before


def test_nan_is_rejected_before_assignment():
    agent, sender, _ = _agent_with_two_connections()
    agent.set_state({"ok": 1})
    sender.sent.clear()

    agent._handle_state_update(sender, {"state": {"bad": float("nan")}})

    # dumps_wire runs before the assignment and rejects NaN, so the prior state
    # stays live in memory rather than being clobbered by an unwritable value.
    assert state_error_frame("State update rejected") in sender.frames
    assert agent.state == {"ok": 1}


# ── the state property ───────────────────────────────────────────────────────


def test_state_returns_a_copy():
    agent = fakes.build_agent()
    agent.set_state({"a": 1})

    s = agent.state
    s["z"] = 9

    assert "z" not in agent.state


def test_cold_set_state_bootstraps_only_state_storage():
    agent = Agent(fakes.FakeCtx(), types.SimpleNamespace())

    agent.set_state({"ready": True})

    assert agent.state == {"ready": True}
    assert agent._read_state_cell(STATE_ROW_ID) == '{"ready":true}'
    assert agent._lifecycle._ready is False


# ── hydrate ──────────────────────────────────────────────────────────────────


def test_hydrate_restores_stored_state_across_instances():
    conn = fakes.new_sqlite()
    env = types.SimpleNamespace()

    a1 = fakes.build_agent(conn=conn, env=env)
    a1.set_state({"n": 1})

    a2 = fakes.build_agent(conn=conn, env=env)
    assert a2.state == {"n": 1}


def test_initial_state_is_the_fallback_when_no_row_stored():
    class InitialStateAgent(Agent):
        def initial_state(self):
            return {"d": 1}

    agent = fakes.build_agent(cls=InitialStateAgent)
    assert agent.state == {"d": 1}


def test_unparseable_row_is_overwritten_without_raising():
    agent = fakes.build_agent()
    agent.sql(
        "INSERT OR REPLACE INTO cf_agents_state (id, state) VALUES (?, ?)",
        STATE_ROW_ID,
        "not json",
    )

    # Must not raise: startup runs on every wake, so a bad row has to heal rather
    # than brick the object.
    agent._hydrate_state()

    assert agent.state == {}
    rows = agent.sql("SELECT state FROM cf_agents_state WHERE id = ?", STATE_ROW_ID)
    # The row was rewritten, so reading it back is valid JSON again.
    assert rows[0]["state"] == "{}"
