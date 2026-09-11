"""The recovery scan: `_check_run_fibers` and the two passes under it.

Recovery reconciles two tables against each other. `cf_agents_runs` holds a row
for every fiber currently executing and `cf_agents_fibers` is the managed
ledger, so a wake finds three kinds of wreckage: a run row whose execution is
gone (orphan), a non-terminal ledger row that never got a run row at all
(ledger-only), and a run row whose ledger entry has already settled (stale).

An orphan is produced here by the code rather than seeded, because the state
that matters is the one an eviction actually leaves. A fiber body raising
`CancelledError` takes the same path an eviction does — `_run_fiber_internal`
sets `preserve` and skips its finally-delete — so the run row and the ledger row
are left exactly as a real eviction leaves them, including whether a `stash`
reached both tables. That is what the two snapshot cases below pin: the orphan
UPDATE overwrites the ledger snapshot from the run row, and it is only safe to
do so because a managed `stash` writes both tables in the same call.

`_check_run_fibers` is awaited directly; nothing here goes through an alarm.
"""

from __future__ import annotations

import asyncio

import pytest

from agents import Agent, FiberRecoveryResult
from agents.core.utils import now_ms
from agents.lifecycle.fiber import INTERNAL_FIBER_PREFIX

_RUN = "INSERT INTO cf_agents_runs (id, name, snapshot, created_at) VALUES (?, ?, ?, ?)"

_LEDGER = (
    "INSERT INTO cf_agents_fibers (fiber_id, idempotency_key, name, status, "
    "snapshot, metadata_json, error_message, created_at, started_at, "
    "completed_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)


class Recovering(Agent):
    """Records every ctx the recovery hook is handed, and can be made to fail.

    The scalar defaults sit on the class so a fixture can override them per test;
    the list cannot, or every instance would share one.
    """

    recovery_result: FiberRecoveryResult | None = None
    recovery_error: str = ""

    async def on_fiber_recovered(self, ctx):
        self.recovered.append(ctx)
        if self.recovery_error:
            raise RuntimeError(self.recovery_error)
        return self.recovery_result


@pytest.fixture
def recording_agent(make_agent):
    def build(result: FiberRecoveryResult | None = None, *, raises: str = ""):
        agent = make_agent(Recovering)
        agent.recovered = []
        agent.recovery_result = result
        agent.recovery_error = raises
        return agent

    return build


def seed_run(
    agent, fiber_id: str, *, name: str = "job", snapshot=None, created_at=None
):
    agent.sql(_RUN, fiber_id, name, snapshot, created_at or now_ms())


def seed_ledger(
    agent,
    fiber_id: str,
    *,
    name: str = "job",
    status: str = "running",
    snapshot=None,
    metadata_json=None,
    idempotency_key=None,
    created_at=None,
):
    agent.sql(
        _LEDGER,
        fiber_id,
        idempotency_key,
        name,
        status,
        snapshot,
        metadata_json,
        None,
        created_at or now_ms(),
        None,
        None,
    )


def run_ids(agent) -> list[str]:
    return [row["id"] for row in agent.sql("SELECT id FROM cf_agents_runs")]


def ledger(agent, fiber_id: str) -> dict:
    rows = agent.sql(
        "SELECT status, snapshot, error_message, completed_at FROM cf_agents_fibers "
        "WHERE fiber_id = ?",
        fiber_id,
    )
    return rows[0]


async def evict_mid_fiber(agent, *, stash=None) -> str:
    """Run a managed fiber that is cancelled mid-flight, and return its id.

    CancelledError is the eviction path: the run row survives and the ledger row
    is left non-terminal, which is precisely what the scan has to clean up.
    """
    captured: list[str] = []

    async def body(ctx):
        captured.append(ctx.id)
        if stash is not None:
            agent.stash(stash)
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await agent.start_fiber("job", body, wait_for_completion=True)

    return captured[0]


# ── what an eviction leaves behind ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_cancelled_fiber_preserves_its_run_row(make_agent):
    agent = make_agent()
    fiber_id = await evict_mid_fiber(agent, stash={"step": 2})

    # The finally-delete is skipped on CancelledError, because the surviving run
    # row is the whole recovery mechanism.
    assert run_ids(agent) == [fiber_id]
    assert ledger(agent, fiber_id)["status"] == "running"
    # And the in-memory marker is gone, so the next scan sees it as an orphan.
    assert fiber_id not in agent._fiber._fiber_active_ids


@pytest.mark.asyncio
async def test_a_stash_reaches_the_run_row_and_the_ledger_row_together(make_agent):
    agent = make_agent()
    fiber_id = await evict_mid_fiber(agent, stash={"step": 2})

    run = agent.sql("SELECT snapshot FROM cf_agents_runs WHERE id = ?", fiber_id)
    # The invariant the orphan UPDATE relies on: a managed stash writes both
    # tables in one call, so the run row's snapshot is never staler than the
    # ledger's and overwriting from it cannot lose a checkpoint.
    assert run[0]["snapshot"] == ledger(agent, fiber_id)["snapshot"] == '{"step":2}'


@pytest.mark.asyncio
async def test_a_fiber_that_never_stashed_leaves_both_snapshots_null(make_agent):
    agent = make_agent()
    fiber_id = await evict_mid_fiber(agent)

    run = agent.sql("SELECT snapshot FROM cf_agents_runs WHERE id = ?", fiber_id)
    assert run[0]["snapshot"] is None
    assert ledger(agent, fiber_id)["snapshot"] is None


# ── orphan run rows ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_orphan_run_row_is_interrupted_and_handed_to_the_hook(recording_agent):
    agent = recording_agent()
    fiber_id = await evict_mid_fiber(agent, stash={"step": 2})

    await agent._fiber._check_run_fibers()

    row = ledger(agent, fiber_id)
    assert row["status"] == "interrupted"
    assert row["completed_at"] is not None
    assert row["snapshot"] == '{"step":2}'
    # The run row is the marker for "still executing", so it goes once the ledger
    # carries the outcome.
    assert run_ids(agent) == []

    (ctx,) = agent.recovered
    assert ctx.id == fiber_id
    assert ctx.status == "interrupted"
    assert ctx.snapshot == {"step": 2}


@pytest.mark.asyncio
async def test_orphan_run_row_with_no_stash_keeps_the_ledger_snapshot_null(
    recording_agent,
):
    agent = recording_agent()
    fiber_id = await evict_mid_fiber(agent)

    await agent._fiber._check_run_fibers()

    # Nothing checkpointed, so there is nothing to restore from — and the hook is
    # told that rather than being handed a stale snapshot.
    assert ledger(agent, fiber_id)["snapshot"] is None
    assert ledger(agent, fiber_id)["status"] == "interrupted"
    assert agent.recovered[0].snapshot is None


@pytest.mark.asyncio
async def test_orphan_run_row_carries_the_ledger_identity_into_the_hook(
    recording_agent,
):
    agent = recording_agent()
    seed_run(agent, "f1")
    seed_ledger(agent, "f1", idempotency_key="k1", metadata_json='{"who":"me"}')

    await agent._fiber._check_run_fibers()

    (ctx,) = agent.recovered
    # The run row holds neither of these, so they have to come off the ledger row.
    assert ctx.idempotency_key == "k1"
    assert ctx.metadata == {"who": "me"}


@pytest.mark.asyncio
async def test_a_recovery_result_settles_the_interrupted_row(recording_agent):
    agent = recording_agent(FiberRecoveryResult(status="completed", snapshot={"ok": 1}))
    seed_run(agent, "f1")
    seed_ledger(agent, "f1")

    await agent._fiber._check_run_fibers()

    row = ledger(agent, "f1")
    assert row["status"] == "completed"
    assert row["snapshot"] == '{"ok":1}'


@pytest.mark.asyncio
async def test_invalid_managed_recovery_status_marks_the_row_error(recording_agent):
    agent = recording_agent(FiberRecoveryResult(status="done"))
    seed_run(agent, "f1")
    seed_ledger(agent, "f1")

    await agent._fiber._check_run_fibers()

    row = ledger(agent, "f1")
    assert row["status"] == "error"
    assert "invalid fiber recovery status 'done'" in row["error_message"]
    assert run_ids(agent) == []


@pytest.mark.asyncio
async def test_a_raising_hook_marks_the_row_error_and_still_clears_the_run_row(
    recording_agent,
):
    agent = recording_agent(raises="nope")
    seed_run(agent, "f1")
    seed_ledger(agent, "f1")

    await agent._fiber._check_run_fibers()

    row = ledger(agent, "f1")
    # The ledger is the failure's own notification channel, so the error lands
    # there rather than going to on_error.
    assert row["status"] == "error"
    assert row["error_message"] == "nope"
    # Managed rows clean up regardless: the ledger now holds a terminal status, so
    # a retained run row would only be rediscovered as an orphan forever.
    assert run_ids(agent) == []


@pytest.mark.asyncio
async def test_a_stale_run_row_for_a_settled_ledger_row_is_only_deleted(
    recording_agent,
):
    agent = recording_agent()
    seed_run(agent, "f1")
    seed_ledger(agent, "f1", status="completed")

    await agent._fiber._check_run_fibers()

    assert run_ids(agent) == []
    # Already terminal, so there is nothing to recover and no hook to run.
    assert ledger(agent, "f1")["status"] == "completed"
    assert agent.recovered == []


@pytest.mark.asyncio
async def test_an_unmanaged_orphan_run_row_reaches_the_hook_and_is_deleted(
    recording_agent,
):
    agent = recording_agent()
    seed_run(agent, "f1", snapshot='{"u":1}')

    await agent._fiber._check_run_fibers()

    (ctx,) = agent.recovered
    assert ctx.id == "f1"
    assert ctx.snapshot == {"u": 1}
    # run_fiber writes no ledger row, so there is no status to report — only the
    # run row, which the hook returning without raising retires.
    assert ctx.status is None
    assert run_ids(agent) == []


@pytest.mark.asyncio
async def test_invalid_unmanaged_recovery_status_keeps_run_row_for_retry(
    recording_agent,
):
    agent = recording_agent(FiberRecoveryResult(status="done"))
    seed_run(agent, "f1")

    await agent._fiber._check_run_fibers()

    assert run_ids(agent) == ["f1"]


@pytest.mark.asyncio
async def test_resolve_fiber_rejects_invalid_status_without_mutation(recording_agent):
    agent = recording_agent()
    seed_ledger(agent, "f1", status="interrupted")

    with pytest.raises(ValueError, match="invalid fiber recovery status 'done'"):
        await agent.resolve_fiber("f1", FiberRecoveryResult(status="done"))

    assert ledger(agent, "f1")["status"] == "interrupted"


@pytest.mark.asyncio
async def test_a_run_row_for_a_live_fiber_is_left_alone(recording_agent):
    agent = recording_agent()
    seed_run(agent, "f1")
    seed_ledger(agent, "f1")
    agent._fiber._fiber_active_ids.add("f1")

    await agent._fiber._check_run_fibers()

    # Still executing in this isolate, so it is not wreckage.
    assert run_ids(agent) == ["f1"]
    assert ledger(agent, "f1")["status"] == "running"
    assert agent.recovered == []


# ── ledger-only rows ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_ledger_only_row_is_finalized_interrupted(recording_agent):
    agent = recording_agent()
    seed_ledger(agent, "f1", status="pending", snapshot='{"keep":1}')

    await agent._fiber._check_run_fibers()

    row = ledger(agent, "f1")
    assert row["status"] == "interrupted"
    assert row["completed_at"] is not None
    # No run row ever existed, so there is no newer snapshot to overwrite with.
    assert row["snapshot"] == '{"keep":1}'

    (ctx,) = agent.recovered
    assert ctx.id == "f1"
    assert ctx.status == "interrupted"
    assert ctx.snapshot == {"keep": 1}


@pytest.mark.asyncio
async def test_a_ledger_only_row_for_a_live_fiber_is_left_alone(recording_agent):
    agent = recording_agent()
    seed_ledger(agent, "f1", status="pending")
    agent._fiber._fiber_active_ids.add("f1")

    await agent._fiber._check_run_fibers()

    assert ledger(agent, "f1")["status"] == "pending"
    assert agent.recovered == []


@pytest.mark.asyncio
async def test_a_scan_leaves_a_just_started_background_fiber_alone(
    make_agent, wait_until
):
    agent = make_agent()
    ran: list[str] = []

    async def body(ctx):
        ran.append(ctx.id)
        return "R"

    # The recorder captures the coroutine without running it, so the fiber sits
    # scheduled-but-not-started — the window between waitUntil and the coroutine's
    # first slice, where its run row does not exist yet.
    agent.detached_fibers_enabled = True
    res = await agent.start_fiber("job", body)
    assert run_ids(agent) == []

    await agent._fiber._check_run_fibers()

    # Claimed in memory at start, so the scan must not interrupt a fiber about to run.
    assert ledger(agent, res.fiber_id)["status"] == "pending"

    await wait_until.drain()

    assert ran == [res.fiber_id]
    assert ledger(agent, res.fiber_id)["status"] == "completed"


@pytest.mark.asyncio
async def test_a_terminal_ledger_row_with_no_run_row_is_not_touched(recording_agent):
    agent = recording_agent()
    seed_ledger(agent, "f1", status="completed")

    await agent._fiber._check_run_fibers()

    # The join only picks up pending and running, so a settled row is not revisited.
    assert ledger(agent, "f1")["status"] == "completed"
    assert agent.recovered == []


# ── framework fibers ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_an_internal_fiber_is_kept_from_the_user_hook(recording_agent):
    agent = recording_agent()
    seed_run(agent, "i1", name=f"{INTERNAL_FIBER_PREFIX}chat")

    await agent._fiber._check_run_fibers()

    # A fiber this port does not drive: the user hook knows nothing about it, so it
    # is left for a runtime that does rather than handed over or discarded.
    assert agent.recovered == []
    assert run_ids(agent) == ["i1"]


@pytest.mark.asyncio
async def test_an_internal_fiber_ages_out_of_the_run_table(recording_agent):
    agent = recording_agent()
    seed_run(agent, "i1", name=f"{INTERNAL_FIBER_PREFIX}chat", created_at=1_000)

    # Unclaimed forever is a leak, so the retention window retires it. created_at
    # is epoch 1970 here, far past any plausible window.
    await agent._fiber._check_run_fibers()

    assert run_ids(agent) == []
    assert agent.recovered == []


@pytest.mark.asyncio
async def test_a_zero_max_age_keeps_an_internal_fiber_indefinitely(recording_agent):
    agent = recording_agent()
    agent.fiber_recovery_max_age_ms = 0
    seed_run(agent, "i1", name=f"{INTERNAL_FIBER_PREFIX}chat", created_at=1_000)

    await agent._fiber._check_run_fibers()

    assert run_ids(agent) == ["i1"]


# ── scan bookkeeping ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_scan_already_running_is_a_no_op(recording_agent):
    agent = recording_agent()
    agent._fiber._fiber_recovery_in_progress = True
    seed_run(agent, "f1")

    await agent._fiber._check_run_fibers()

    # Re-entering would double-report a recovery and race its own deletes.
    assert run_ids(agent) == ["f1"]


@pytest.mark.asyncio
async def test_reentrant_deferred_recovery_queues_a_follow_up_scan(recording_agent):
    agent = recording_agent()
    ready = False
    recovered = []

    def is_target(ctx):
        return ctx.name == "target"

    async def recover(ctx):
        nonlocal ready
        recovered.append(ctx.name)
        if ctx.name == "trigger":
            ready = True
            await agent._fiber.resume_deferred_recovery(is_target)
        return FiberRecoveryResult(status="completed")

    agent._fiber._on_recovered = recover
    agent._fiber._defer_internal_recovery = lambda ctx: is_target(ctx) and not ready
    seed_run(agent, "target", name="target")
    seed_run(agent, "trigger", name="trigger")

    await asyncio.wait_for(agent._fiber._check_run_fibers(), timeout=1)

    assert recovered == ["trigger", "target"]
    assert run_ids(agent) == []
    assert agent.recovered == []


@pytest.mark.asyncio
async def test_waiting_for_an_unowned_fiber_during_recovery_fails_fast(
    recording_agent,
):
    agent = recording_agent()
    seed_ledger(agent, "f1", status="pending")
    agent._fiber._fiber_recovery_in_progress = True

    with pytest.raises(RuntimeError, match="during fiber recovery"):
        await agent._fiber._wait_for_managed_fiber("f1")


@pytest.mark.asyncio
async def test_an_empty_scan_does_not_lengthen_the_backoff_streak(
    recording_agent,
):
    agent = recording_agent()

    await agent._fiber._check_run_fibers()
    assert agent._fiber._recovery_no_progress_scans == 0
    await agent._fiber._check_run_fibers()
    assert agent._fiber._recovery_no_progress_scans == 0


@pytest.mark.asyncio
async def test_a_scan_that_recovered_something_resets_the_streak(recording_agent):
    agent = recording_agent()
    agent._fiber._recovery_no_progress_scans = 4
    seed_run(agent, "f1")
    seed_ledger(agent, "f1")

    await agent._fiber._check_run_fibers()

    # Streak 0 keeps the next pass at the base interval, so a draining multi-pass
    # recovery is not slowed down by the poison-hook backoff.
    assert agent._fiber._recovery_no_progress_scans == 0
