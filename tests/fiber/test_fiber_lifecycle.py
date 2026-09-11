"""Durable-execution lifecycle: run_fiber, start_fiber, stash, cancel_fiber.

These drive a real Agent through its installed FiberCapability over FakeCtx-backed
sqlite, so the run/ledger tables, the finally-delete on completion, and the
conditional ledger UPDATEs are exercised against the source rather than a mock.

wait_for_completion=True runs a fiber inline via await, so it is deterministic.
The default workers.waitUntil stub CLOSES a background coroutine unstarted, so
start_fiber(wait_for_completion=False) leaves a 'pending' ledger row without
ever running the fiber — which is what the cancel test needs.

The CancelledError-preserves-the-run-row case is deliberately excluded: its only
consumer is the deferred recovery scan (see design/TEST_SUITE.md, Phase 3).
"""

from __future__ import annotations

import json

import pytest
import workers


@pytest.mark.asyncio
async def test_run_fiber_runs_and_deletes_run_row(make_agent):
    agent = make_agent()
    seen = []

    async def body(ctx):
        seen.append(ctx.id)
        return "R"

    assert await agent.run_fiber("job", body) == "R"
    assert seen  # the callback ran
    assert agent.sql("SELECT id FROM cf_agents_runs") == []  # deleted on success


@pytest.mark.asyncio
async def test_run_fiber_propagates_failure_and_deletes_run_row(make_agent):
    agent = make_agent()

    async def boom(ctx):
        raise ValueError("x")

    with pytest.raises(ValueError):
        await agent.run_fiber("job", boom)

    # A normal exception is not CancelledError, so the finally-delete still fires.
    assert agent.sql("SELECT id FROM cf_agents_runs") == []


@pytest.mark.asyncio
async def test_stash_outside_a_fiber_raises(make_agent):
    agent = make_agent()
    with pytest.raises(RuntimeError):
        agent.stash({"a": 1})


@pytest.mark.asyncio
async def test_stash_inside_a_fiber_writes_the_snapshot(make_agent):
    agent = make_agent()

    async def body(ctx):
        agent.stash({"k": 1})
        # Read the snapshot before returning: the run row is deleted on completion.
        rows = agent.sql("SELECT snapshot FROM cf_agents_runs WHERE id = ?", ctx.id)
        assert json.loads(rows[0]["snapshot"]) == {"k": 1}

    await agent.run_fiber("job", body)


@pytest.mark.asyncio
async def test_start_fiber_wait_for_completion_settles_completed(make_agent):
    agent = make_agent()

    async def body(ctx):
        return "R"

    res = await agent.start_fiber("job", body, wait_for_completion=True)
    assert res.accepted is True
    assert res.status == "completed"

    row = agent.sql(
        "SELECT status FROM cf_agents_fibers WHERE fiber_id = ?", res.fiber_id
    )
    assert row[0]["status"] == "completed"


@pytest.mark.asyncio
async def test_awaited_fibers_do_not_use_wait_until(make_agent, monkeypatch):
    agent = make_agent()

    def unavailable(coro):
        coro.close()
        raise RuntimeError("waitUntil unavailable")

    monkeypatch.setattr(workers, "waitUntil", unavailable)

    async def body(ctx):
        return ctx.id

    assert await agent.run_fiber("inline", body)
    result = await agent.start_fiber("managed", body, wait_for_completion=True)
    assert result.status == "completed"

    with pytest.raises(RuntimeError, match="detached fibers are disabled"):
        await agent.start_fiber("gated", body)

    agent.detached_fibers_enabled = True
    with pytest.raises(RuntimeError, match="waitUntil unavailable"):
        await agent.start_fiber("detached", body)


@pytest.mark.asyncio
async def test_idempotency_key_reuse_does_not_rerun(make_agent):
    agent = make_agent()
    calls = []

    async def once(ctx):
        calls.append(1)
        return "a"

    r1 = await agent.start_fiber(
        "job", once, idempotency_key="k1", wait_for_completion=True
    )

    async def twice(ctx):
        calls.append(2)
        return "b"

    r2 = await agent.start_fiber(
        "job", twice, idempotency_key="k1", wait_for_completion=True
    )

    assert r1.accepted is True and r2.accepted is False
    assert calls == [1]  # the second body never ran


@pytest.mark.asyncio
async def test_conflicting_fiber_id_and_idempotency_key_raises(make_agent):
    agent = make_agent()

    async def body(ctx):
        return "R"

    await agent.start_fiber("n", body, fiber_id="A", wait_for_completion=True)
    await agent.start_fiber(
        "n", body, fiber_id="B", idempotency_key="k", wait_for_completion=True
    )

    # fiber_id "A" resolves to one row, idempotency_key "k" to another: a conflict.
    with pytest.raises(ValueError):
        await agent.start_fiber(
            "n",
            body,
            fiber_id="A",
            idempotency_key="k",
            wait_for_completion=True,
        )


@pytest.mark.asyncio
async def test_unknown_fiber_id_with_a_key_pointing_elsewhere_raises(make_agent):
    agent = make_agent()

    async def body(ctx):
        return "R"

    await agent.start_fiber(
        "n", body, fiber_id="B", idempotency_key="k", wait_for_completion=True
    )

    # The other half of the conflict check: fiber_id "C" has no row at all, so only
    # the key resolves — and it resolves to "B". Passing both still cannot mean two
    # different fibers.
    with pytest.raises(ValueError):
        await agent.start_fiber(
            "n",
            body,
            fiber_id="C",
            idempotency_key="k",
            wait_for_completion=True,
        )


@pytest.mark.asyncio
async def test_cancel_fiber_settles_pending_and_is_idempotent(make_agent):
    agent = make_agent()

    async def body(ctx):
        return "R"

    # Default waitUntil closes the background coroutine unstarted, so the ledger
    # row stays 'pending' and never runs.
    agent.detached_fibers_enabled = True
    res = await agent.start_fiber("job", body)  # wait_for_completion defaults to False
    assert res.accepted is True

    assert await agent.cancel_fiber(res.fiber_id) is True
    row = agent.sql(
        "SELECT status FROM cf_agents_fibers WHERE fiber_id = ?", res.fiber_id
    )
    assert row[0]["status"] == "aborted"

    assert await agent.cancel_fiber(res.fiber_id) is False  # already settled
    assert await agent.cancel_fiber("does-not-exist") is False
