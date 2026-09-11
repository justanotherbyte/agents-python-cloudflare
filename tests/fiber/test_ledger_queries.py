"""The read side of the fiber ledger: list_fibers, delete_fibers, inspect_*.

Rows are seeded straight into `cf_agents_fibers` with `agent.sql`, so a query is
tested against whatever it finds rather than against what a lifecycle path
happens to leave behind. That is the point: `list_fibers` and `delete_fibers`
each fan a status filter into one query per status, take that many rows from
each, then merge, re-sort and re-clamp in Python. Only real sqlite ordering can
show whether the merged answer equals what a single query would return.

The two clamps are asserted with more rows than the ceiling, because a clamp is
invisible until the row count passes it.
"""

from __future__ import annotations

import json

import pytest

_INSERT = (
    "INSERT INTO cf_agents_fibers (fiber_id, idempotency_key, name, status, "
    "snapshot, metadata_json, error_message, created_at, started_at, "
    "completed_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)


def seed(
    agent,
    fiber_id: str,
    *,
    name: str = "job",
    status: str = "completed",
    created_at: int = 1_000,
    completed_at: int | None = None,
    idempotency_key: str | None = None,
    snapshot: str | None = None,
    metadata_json: str | None = None,
    error_message: str | None = None,
    started_at: int | None = None,
) -> None:
    agent.sql(
        _INSERT,
        fiber_id,
        idempotency_key,
        name,
        status,
        snapshot,
        metadata_json,
        error_message,
        created_at,
        started_at,
        completed_at,
    )


def ids(inspections) -> list[str]:
    return [i.fiber_id for i in inspections]


def remaining(agent) -> set[str]:
    rows = agent.sql("SELECT fiber_id FROM cf_agents_fibers")
    return {row["fiber_id"] for row in rows}


# ── list_fibers ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_fibers_unfiltered_returns_every_row(make_agent):
    agent = make_agent()
    seed(agent, "a", status="pending")
    seed(agent, "b", status="error")
    seed(agent, "c", status="completed")

    assert set(ids(await agent.list_fibers())) == {"a", "b", "c"}


@pytest.mark.asyncio
async def test_list_fibers_orders_by_created_at_then_fiber_id_descending(make_agent):
    agent = make_agent()
    seed(agent, "a", created_at=100)
    seed(agent, "b", created_at=100)
    seed(agent, "c", created_at=200)

    # created_at DESC first, then fiber_id DESC to break the tie. The tie-break is
    # on the primary key, so the order is total and can be asserted exactly.
    assert ids(await agent.list_fibers()) == ["c", "b", "a"]


@pytest.mark.asyncio
async def test_list_fibers_status_as_a_string_filters_to_that_status(make_agent):
    agent = make_agent()
    seed(agent, "a", status="completed")
    seed(agent, "b", status="error")

    assert ids(await agent.list_fibers(status="error")) == ["b"]


@pytest.mark.asyncio
async def test_list_fibers_status_as_a_list_unions_the_statuses(make_agent):
    agent = make_agent()
    seed(agent, "a", status="completed")
    seed(agent, "b", status="error")
    seed(agent, "c", status="pending")

    got = await agent.list_fibers(status=["completed", "error"])
    assert set(ids(got)) == {"a", "b"}


@pytest.mark.asyncio
async def test_list_fibers_merges_per_status_results_into_one_ordering(make_agent):
    agent = make_agent()
    seed(agent, "a", status="completed", created_at=400)
    seed(agent, "b", status="error", created_at=300)
    seed(agent, "c", status="completed", created_at=200)
    seed(agent, "d", status="error", created_at=100)

    got = await agent.list_fibers(status=["completed", "error"])
    # Interleaved by created_at across both statuses, not grouped by status.
    assert ids(got) == ["a", "b", "c", "d"]


@pytest.mark.asyncio
async def test_list_fibers_limit_applies_across_the_merged_statuses(make_agent):
    agent = make_agent()
    seed(agent, "a", status="completed", created_at=400)
    seed(agent, "b", status="error", created_at=300)
    seed(agent, "c", status="completed", created_at=200)
    seed(agent, "d", status="error", created_at=100)

    # Each status query takes up to the limit, so the merge has to re-clamp or a
    # two-status filter would return twice as many rows as asked for.
    assert ids(await agent.list_fibers(status=["completed", "error"], limit=2)) == [
        "a",
        "b",
    ]


@pytest.mark.asyncio
async def test_list_fibers_name_only_filters_to_that_name(make_agent):
    agent = make_agent()
    seed(agent, "a", name="import")
    seed(agent, "b", name="export")

    assert ids(await agent.list_fibers(name="export")) == ["b"]


@pytest.mark.asyncio
async def test_list_fibers_status_and_name_both_apply(make_agent):
    agent = make_agent()
    seed(agent, "a", name="import", status="completed")
    seed(agent, "b", name="import", status="error")
    seed(agent, "c", name="export", status="error")

    assert ids(await agent.list_fibers(status="error", name="import")) == ["b"]


@pytest.mark.asyncio
async def test_list_fibers_defaults_to_fifty_rows(make_agent):
    agent = make_agent()
    for i in range(60):
        seed(agent, f"f{i:03d}", created_at=1_000 + i)

    assert len(await agent.list_fibers()) == 50


@pytest.mark.asyncio
async def test_list_fibers_clamps_the_limit_to_one_hundred(make_agent):
    agent = make_agent()
    for i in range(110):
        seed(agent, f"f{i:03d}", created_at=1_000 + i)

    assert len(await agent.list_fibers(limit=10_000)) == 100
    # And the same ceiling once a status filter routes through the merge.
    assert len(await agent.list_fibers(status="completed", limit=10_000)) == 100


@pytest.mark.asyncio
async def test_list_fibers_clamps_a_zero_or_negative_limit_to_one(make_agent):
    agent = make_agent()
    seed(agent, "a", created_at=100)
    seed(agent, "b", created_at=200)

    assert ids(await agent.list_fibers(limit=0)) == ["b"]
    assert ids(await agent.list_fibers(limit=-5)) == ["b"]


# ── delete_fibers ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_delete_fibers_defaults_to_completed_aborted_and_error(make_agent):
    agent = make_agent()
    seed(agent, "done", status="completed", completed_at=100)
    seed(agent, "stopped", status="aborted", completed_at=100)
    seed(agent, "failed", status="error", completed_at=100)
    seed(agent, "cut", status="interrupted", completed_at=100)
    seed(agent, "live", status="running")

    assert await agent.delete_fibers() == 3
    # interrupted is terminal but is deliberately not swept by default: it is the
    # status a recovery hook still has to answer for.
    assert remaining(agent) == {"cut", "live"}


@pytest.mark.asyncio
async def test_delete_fibers_accepts_interrupted_when_asked_for_explicitly(make_agent):
    agent = make_agent()
    seed(agent, "cut", status="interrupted", completed_at=100)

    assert await agent.delete_fibers(status="interrupted") == 1
    assert remaining(agent) == set()


@pytest.mark.asyncio
async def test_delete_fibers_with_only_non_terminal_statuses_deletes_nothing(
    make_agent,
):
    agent = make_agent()
    seed(agent, "live", status="running")
    seed(agent, "queued", status="pending")

    assert await agent.delete_fibers(status=["running", "pending"]) == 0
    assert remaining(agent) == {"live", "queued"}


@pytest.mark.asyncio
async def test_delete_fibers_settled_before_keeps_newer_and_null_settled_rows(
    make_agent,
):
    agent = make_agent()
    seed(agent, "old", status="completed", completed_at=100)
    seed(agent, "new", status="completed", completed_at=900)
    seed(agent, "unsettled", status="completed", completed_at=None)

    assert await agent.delete_fibers(settled_before=500) == 1
    # completed_at IS NOT NULL rides along with the cutoff, so a terminal row that
    # never recorded a settle time is out of range rather than treated as ancient.
    assert remaining(agent) == {"new", "unsettled"}


@pytest.mark.asyncio
async def test_delete_fibers_takes_the_oldest_settled_rows_first(make_agent):
    agent = make_agent()
    seed(agent, "third", status="completed", completed_at=300)
    seed(agent, "first", status="completed", completed_at=100)
    seed(agent, "second", status="completed", completed_at=200)

    assert await agent.delete_fibers(limit=2) == 2
    assert remaining(agent) == {"third"}


@pytest.mark.asyncio
async def test_delete_fibers_orders_null_settled_rows_ahead_of_settled_ones(
    make_agent,
):
    agent = make_agent()
    seed(agent, "unsettled", status="completed", completed_at=None, created_at=50)
    seed(agent, "settled", status="completed", completed_at=100, created_at=60)

    # Without settled_before there is no IS NOT NULL predicate, so a NULL
    # completed_at sorts first — SQL puts NULLs first ascending and the Python
    # re-sort maps them to 0.
    assert await agent.delete_fibers(limit=1) == 1
    assert remaining(agent) == {"settled"}


@pytest.mark.asyncio
async def test_delete_fibers_limit_and_ordering_span_the_merged_statuses(make_agent):
    agent = make_agent()
    seed(agent, "old-done", status="completed", completed_at=100)
    seed(agent, "new-done", status="completed", completed_at=400)
    seed(agent, "old-failed", status="error", completed_at=200)
    seed(agent, "new-failed", status="error", completed_at=500)

    # The two oldest rows are one from each status, and each status holds a row
    # older than the other status's newest. So concatenating the per-status
    # results in either order and taking two would sweep the wrong pair — only a
    # re-sort across the merge picks these two.
    assert await agent.delete_fibers(status=["completed", "error"], limit=2) == 2
    assert remaining(agent) == {"new-done", "new-failed"}


@pytest.mark.asyncio
async def test_delete_fibers_defaults_to_one_hundred_rows(make_agent):
    agent = make_agent()
    for i in range(110):
        seed(agent, f"f{i:03d}", status="completed", completed_at=1_000 + i)

    assert await agent.delete_fibers() == 100
    assert len(remaining(agent)) == 10


@pytest.mark.asyncio
async def test_delete_fibers_clamps_the_limit_to_five_hundred(make_agent):
    agent = make_agent()
    for i in range(510):
        seed(agent, f"f{i:03d}", status="completed", completed_at=1_000 + i)

    assert await agent.delete_fibers(limit=10_000) == 500
    assert len(remaining(agent)) == 10


@pytest.mark.asyncio
async def test_delete_fibers_clamps_a_zero_limit_to_one(make_agent):
    agent = make_agent()
    seed(agent, "a", status="completed", completed_at=100)
    seed(agent, "b", status="completed", completed_at=200)

    assert await agent.delete_fibers(limit=0) == 1
    assert remaining(agent) == {"b"}


# ── inspect ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_inspect_fiber_maps_every_column_onto_the_inspection(make_agent):
    agent = make_agent()
    seed(
        agent,
        "f1",
        name="import",
        status="error",
        created_at=100,
        started_at=150,
        completed_at=200,
        idempotency_key="k1",
        snapshot=json.dumps({"step": 3}),
        metadata_json=json.dumps({"who": "me"}),
        error_message="boom",
    )

    got = await agent.inspect_fiber("f1")
    assert got.fiber_id == "f1"
    assert got.name == "import"
    assert got.status == "error"
    assert got.created_at == 100
    assert got.started_at == 150
    # completed_at is surfaced as settled_at, and error_message as error.
    assert got.settled_at == 200
    assert got.error == "boom"
    assert got.idempotency_key == "k1"
    assert got.snapshot == {"step": 3}
    assert got.metadata == {"who": "me"}


@pytest.mark.asyncio
async def test_inspect_fiber_returns_none_for_an_unknown_id(make_agent):
    agent = make_agent()
    seed(agent, "f1")

    assert await agent.inspect_fiber("nope") is None


@pytest.mark.asyncio
async def test_inspect_fiber_by_key_finds_the_row_and_misses_cleanly(make_agent):
    agent = make_agent()
    seed(agent, "f1", idempotency_key="k1")
    seed(agent, "f2", idempotency_key=None)

    found = await agent.inspect_fiber_by_key("k1")
    assert found.fiber_id == "f1"
    assert await agent.inspect_fiber_by_key("k2") is None
