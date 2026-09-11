"""RPC dispatch: `_dispatch_message`'s rpc branch, `_handle_rpc`, and the
`rpc_callable` registration.

These drive a REAL Agent subclass over a FakeConnection, so the wire shape of
each frame (which keys are present, whether a terminal is owned by the stream)
is checked against the source rather than a mock. The client branches on key
presence, so `done`/`result` being present-or-absent is as load-bearing as the
values — see AGENTS.md, wire invariants.
"""

from __future__ import annotations

import json

import fakes
import pytest

from agents import Agent, rpc_callable


class RpcAgent(Agent):
    data = 5  # a non-callable attribute, for the "not callable" case

    @rpc_callable()
    def add(self, a, b):
        return a + b

    @rpc_callable()
    async def aecho(self, x):
        return x

    @rpc_callable(streaming=True)
    async def stream(self, resp, n):
        for i in range(n):
            resp.send(i)
        resp.end()

    @rpc_callable()
    def boom(self):
        raise RuntimeError("kaboom")

    @rpc_callable(streaming=True)
    async def boom_stream(self, resp):
        raise RuntimeError("mid-stream")

    async def on_message(self, connection, message):
        self.seen = getattr(self, "seen", [])
        self.seen.append(message)


def _agent() -> RpcAgent:
    return fakes.build_agent(cls=RpcAgent)


# ── non-streaming results ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_add_emits_single_terminal_with_result():
    agent = _agent()
    conn = fakes.FakeConnection()

    await agent._handle_rpc(conn, {"id": "1", "method": "add", "args": [1, 2]})

    assert conn.frames == [
        {"type": "rpc", "id": "1", "success": True, "done": True, "result": 3}
    ]


@pytest.mark.asyncio
async def test_async_return_of_none_keeps_result_key():
    # An omitted result and an explicit null are different answers to the client,
    # so a method returning None still carries result: null.
    agent = _agent()
    conn = fakes.FakeConnection()

    await agent._handle_rpc(conn, {"id": "2", "method": "aecho", "args": [None]})

    assert len(conn.frames) == 1
    frame = conn.frames[0]
    assert frame["done"] is True
    assert "result" in frame
    assert frame["result"] is None


# ── unresolved methods ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_missing_method_errors_without_terminal_keys():
    agent = _agent()
    conn = fakes.FakeConnection()

    await agent._handle_rpc(conn, {"id": "3", "method": "nope", "args": []})

    assert len(conn.frames) == 1
    frame = conn.frames[0]
    assert frame["success"] is False
    assert "does not exist" in frame["error"]
    # An error frame carries neither done nor result — that absence is how the
    # client tells it from a terminal.
    assert "done" not in frame
    assert "result" not in frame


@pytest.mark.asyncio
async def test_non_callable_attribute_errors():
    # DISCREPANCY: the plan expected `data` (a plain int attribute) to error with
    # "not callable", but the source only takes the "is not callable" branch when
    # the attribute is itself callable. A non-callable value falls through to the
    # same "does not exist" message a missing name gets. Asserting actual behaviour.
    agent = _agent()
    conn = fakes.FakeConnection()

    await agent._handle_rpc(conn, {"id": "4", "method": "data", "args": []})

    assert len(conn.frames) == 1
    frame = conn.frames[0]
    assert frame["success"] is False
    assert "does not exist" in frame["error"]
    assert "done" not in frame
    assert "result" not in frame


@pytest.mark.asyncio
async def test_undecorated_method_is_not_callable_over_rpc():
    # The genuinely-distinct "not callable" branch: a real method that exists and
    # is callable but was never exposed with @rpc_callable.
    agent = _agent()
    conn = fakes.FakeConnection()

    await agent._handle_rpc(conn, {"id": "5", "method": "on_message", "args": []})

    assert len(conn.frames) == 1
    frame = conn.frames[0]
    assert frame["success"] is False
    assert "not callable" in frame["error"]


@pytest.mark.asyncio
async def test_raising_method_emits_one_error_and_does_not_reach_on_error():
    # Agent.on_error re-raises, so had the RPC path routed the failure through it,
    # this await would raise. It returning cleanly with a single error frame is the
    # assertion that RPC owns its own notification channel and stays off on_error.
    agent = _agent()
    conn = fakes.FakeConnection()

    await agent._handle_rpc(conn, {"id": "b", "method": "boom", "args": []})

    assert len(conn.frames) == 1
    frame = conn.frames[0]
    assert frame["success"] is False
    assert frame["error"] == "kaboom"
    assert "done" not in frame
    assert "result" not in frame


# ── streaming ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_streaming_emits_chunks_then_one_terminal():
    agent = _agent()
    conn = fakes.FakeConnection()

    await agent._handle_rpc(conn, {"id": "s", "method": "stream", "args": [3]})

    frames = conn.frames
    assert len(frames) == 4

    for i, chunk in enumerate(frames[:3]):
        assert chunk["done"] is False
        assert chunk["result"] == i

    terminal = frames[3]
    assert terminal["done"] is True
    # end() was called with no final chunk, so the terminal omits result.
    assert "result" not in terminal
    # The stream owns the single terminal; the dispatcher adds none of its own.
    assert sum(f.get("done") is True for f in frames) == 1


@pytest.mark.asyncio
async def test_streaming_method_raising_gets_error_terminal_from_dispatcher():
    agent = _agent()
    conn = fakes.FakeConnection()

    await agent._handle_rpc(conn, {"id": "bs", "method": "boom_stream", "args": []})

    frames = conn.frames
    # Nothing was sent before the raise, so the error terminal is the only frame.
    assert len(frames) == 1
    terminal = frames[-1]
    assert terminal["success"] is False
    assert terminal["error"] == "mid-stream"
    assert "done" not in terminal
    assert "result" not in terminal


# ── _dispatch_message routing ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_dispatch_routes_well_formed_rpc_frame():
    agent = _agent()
    conn = fakes.FakeConnection()

    await agent._dispatch_message(
        conn, json.dumps({"type": "rpc", "id": "x", "method": "add", "args": [1, 2]})
    )

    assert conn.frames == [
        {"type": "rpc", "id": "x", "success": True, "done": True, "result": 3}
    ]


@pytest.mark.asyncio
async def test_dispatch_falls_through_malformed_rpc_to_on_message():
    # args is a string, not a list, so the frame is wrongly shaped and must reach
    # on_message rather than collect an rpc_error the caller never asked for.
    agent = _agent()
    conn = fakes.FakeConnection()
    raw = json.dumps({"type": "rpc", "id": "x", "method": "add", "args": "nope"})

    await agent._dispatch_message(conn, raw)

    assert conn.frames == []
    assert agent.seen == [raw]
