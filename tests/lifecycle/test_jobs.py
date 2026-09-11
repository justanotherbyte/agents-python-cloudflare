from __future__ import annotations

import asyncio
import json

import fakes
import pytest

import agents.lifecycle._runtime as lifecycle_module
from agents.lifecycle import (
    Lifecycle,
    LifecycleCapability,
    LifecycleJobPushOptions,
)


EXPECTED_COLUMNS = [
    "id",
    "capability",
    "fn",
    "time",
    "payload",
    "retry_options",
    "singleflight",
    "hung_timeout_seconds",
    "exclusive",
    "recovery_loop",
    "running",
    "execution_started_at",
    "created_at",
]


class WorkerCapability(LifecycleCapability):
    capability_id = "worker"


@pytest.mark.asyncio
async def test_jobs_prepare_exact_schema_and_enforce_owner_scoped_crud(monkeypatch):
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)
    ctx = fakes.FakeCtx()
    lifecycle = Lifecycle(ctx, host=object())
    worker = WorkerCapability()
    lifecycle.use(worker)

    await lifecycle.start()

    columns = ctx.storage.sql.exec("PRAGMA table_info(cf_agents_jobs)").toArray()
    assert [column["name"] for column in columns] == EXPECTED_COLUMNS
    assert [
        (
            column["name"],
            column["type"],
            column["notnull"],
            column["dflt_value"],
            column["pk"],
        )
        for column in columns
    ] == [
        ("id", "TEXT", 1, None, 1),
        ("capability", "TEXT", 1, None, 0),
        ("fn", "TEXT", 1, None, 0),
        ("time", "INTEGER", 1, None, 0),
        ("payload", "TEXT", 0, None, 0),
        ("retry_options", "TEXT", 0, None, 0),
        ("singleflight", "INTEGER", 1, "0", 0),
        ("hung_timeout_seconds", "INTEGER", 0, None, 0),
        ("exclusive", "INTEGER", 1, "0", 0),
        ("recovery_loop", "INTEGER", 1, "0", 0),
        ("running", "INTEGER", 1, "0", 0),
        ("execution_started_at", "INTEGER", 0, None, 0),
        ("created_at", "INTEGER", 1, "unixepoch()", 0),
    ]
    table_sql = ctx.storage.sql.exec(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'cf_agents_jobs'"
    ).toArray()[0]["sql"]
    assert " ".join(table_sql.split()) == (
        "CREATE TABLE cf_agents_jobs ( "
        "id TEXT PRIMARY KEY NOT NULL, capability TEXT NOT NULL, "
        "fn TEXT NOT NULL, time INTEGER NOT NULL, payload TEXT, "
        "retry_options TEXT, singleflight INTEGER NOT NULL DEFAULT 0, "
        "hung_timeout_seconds INTEGER, exclusive INTEGER NOT NULL DEFAULT 0, "
        "recovery_loop INTEGER NOT NULL DEFAULT 0, "
        "running INTEGER NOT NULL DEFAULT 0, execution_started_at INTEGER, "
        "created_at INTEGER NOT NULL DEFAULT (unixepoch()) ) WITHOUT ROWID"
    )
    assert (
        ctx.storage.sql.exec(
            "SELECT name FROM sqlite_master WHERE type = 'index' "
            "AND tbl_name = 'cf_agents_jobs'"
        ).toArray()
        == []
    )

    original = await lifecycle.jobs.push(
        LifecycleJobPushOptions(
            id="shared-id",
            fn="host-work",
            time=2_000,
            payload={"value": 1},
        )
    )

    row = ctx.storage.sql.exec(
        "SELECT * FROM cf_agents_jobs WHERE id = 'shared-id'"
    ).toArray()[0]
    assert row["capability"] == "host"
    assert row["payload"] == '{"value":1}'
    assert row["retry_options"] is None
    assert row["singleflight"] == 0
    assert row["hung_timeout_seconds"] is None
    assert row["exclusive"] == 0
    assert row["recovery_loop"] == 0
    assert row["running"] == 0
    assert row["execution_started_at"] is None
    assert original.payload == {"value": 1}
    assert original.retry is None
    assert original.created_at == row["created_at"]

    assert await worker.lifecycle.jobs.get("shared-id") is None
    assert not await worker.lifecycle.jobs.cancel("shared-id")
    assert not await worker.lifecycle.jobs.reschedule("shared-id", 3_000)
    with pytest.raises(RuntimeError, match="already belongs to.*host"):
        await worker.lifecycle.jobs.push(
            LifecycleJobPushOptions(
                id="shared-id",
                fn="worker-work",
                time=3_000,
            )
        )

    ctx.storage.sql.exec(
        "UPDATE cf_agents_jobs SET running = 1, execution_started_at = 10 "
        "WHERE id = 'shared-id'"
    )
    replaced = await lifecycle.jobs.push(
        LifecycleJobPushOptions(
            id="shared-id",
            fn="replacement",
            time=4_000,
            payload={"value": 2},
        )
    )

    assert replaced.created_at == original.created_at
    assert replaced.fn == "replacement"
    row = ctx.storage.sql.exec(
        "SELECT running, execution_started_at FROM cf_agents_jobs "
        "WHERE id = 'shared-id'"
    ).toArray()[0]
    assert row == {"running": 0, "execution_started_at": None}
    assert [job.id for job in await lifecycle.jobs.list()] == ["shared-id"]
    assert await lifecycle.jobs.reschedule("shared-id", 5_000)
    rescheduled = await lifecycle.jobs.get("shared-id")
    assert rescheduled is not None
    assert rescheduled.time == 5_000
    assert await lifecycle.jobs.cancel("shared-id")
    assert not await lifecycle.jobs.cancel("shared-id")
    assert ctx.storage.alarm_time_ms is None


@pytest.mark.asyncio
async def test_job_payload_and_retry_json_are_strict(monkeypatch):
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)
    ctx = fakes.FakeCtx()
    lifecycle = Lifecycle(ctx, host=object())

    with pytest.raises(ValueError, match="JSON"):
        await lifecycle.jobs.push(
            LifecycleJobPushOptions(
                id="nan-payload",
                fn="work",
                time=2_000,
                payload={"value": float("nan")},
            )
        )

    assert await lifecycle.jobs.get("nan-payload") is None

    await lifecycle.jobs.push(
        LifecycleJobPushOptions(
            id="explicit-null",
            fn="work",
            time=2_000,
            payload=None,
        )
    )
    await lifecycle.jobs.push(
        LifecycleJobPushOptions(id="absent", fn="work", time=3_000)
    )
    assert ctx.storage.sql.exec(
        "SELECT id, payload FROM cf_agents_jobs ORDER BY time"
    ).toArray() == [
        {"id": "explicit-null", "payload": "null"},
        {"id": "absent", "payload": None},
    ]
    explicit_null = await lifecycle.jobs.get("explicit-null")
    absent = await lifecycle.jobs.get("absent")
    assert explicit_null is not None
    assert explicit_null.payload is None
    assert explicit_null.payload_present
    assert absent is not None
    assert absent.payload is None
    assert not absent.payload_present

    ctx.storage.sql.exec(
        "INSERT INTO cf_agents_jobs "
        "(id, capability, fn, time, payload, retry_options) "
        "VALUES ('bad-payload', 'host', 'work', 4000, '{\"value\":NaN}', NULL)"
    )
    with pytest.raises(ValueError, match="invalid JSON constant"):
        await lifecycle.jobs.get("bad-payload")
    assert await lifecycle.jobs.reschedule("bad-payload", 5_000)
    assert await lifecycle.jobs.cancel("bad-payload")

    ctx.storage.sql.exec(
        "INSERT INTO cf_agents_jobs "
        "(id, capability, fn, time, payload, retry_options) "
        "VALUES ('bad-retry', 'host', 'work', 4000, NULL, '{\"value\":Infinity}')"
    )
    with pytest.raises(ValueError, match="invalid JSON constant"):
        await lifecycle.jobs.get("bad-retry")
    ctx.storage.sql.exec(
        "INSERT INTO cf_agents_jobs "
        "(id, capability, fn, time, payload, retry_options) "
        "VALUES (?, 'host', 'work', 4000, ?, NULL)",
        "deep-payload",
        "[" * 10_000 + "null" + "]" * 10_000,
    )
    with pytest.raises(ValueError, match="nesting limit"):
        await lifecycle.jobs.get("deep-payload")
    with pytest.raises(ValueError, match="invalid JSON constant"):
        await lifecycle.jobs.list()
    assert [job.id for job in await lifecycle.jobs.list(skip_invalid=True)] == [
        "explicit-null",
        "absent",
    ]
    with pytest.raises(ValueError, match="invalid job time"):
        await lifecycle.jobs.push(LifecycleJobPushOptions(fn="work", time=10**400))


@pytest.mark.asyncio
async def test_rearm_ignores_a_lone_job_with_a_corrupt_time(monkeypatch):
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)
    ctx = fakes.FakeCtx()
    lifecycle = Lifecycle(ctx, host=object())
    await lifecycle.start()
    job = await lifecycle.jobs.push(
        LifecycleJobPushOptions(id="corrupt-time", fn="work", time=4_000)
    )
    ctx.storage.sql.exec(
        "UPDATE cf_agents_jobs SET time = 'bad' WHERE id = ?",
        job.id,
    )

    await lifecycle.jobs.rearm()

    assert ctx.storage.alarm_time_ms is None


@pytest.mark.asyncio
async def test_rearm_serializes_an_older_calculation_before_a_newer_push(monkeypatch):
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)
    ctx = fakes.FakeCtx()
    lifecycle = Lifecycle(ctx, host=object())
    await lifecycle.start()
    gate = fakes.AsyncGate()
    ctx.storage.pause_next_alarm(gate)

    later = asyncio.create_task(
        lifecycle.jobs.push(LifecycleJobPushOptions(id="later", fn="work", time=5_000))
    )
    await gate.wait_until_blocked()
    earlier = asyncio.create_task(
        lifecycle.jobs.push(
            LifecycleJobPushOptions(id="earlier", fn="work", time=4_000)
        )
    )
    await asyncio.sleep(0)

    gate.release()
    await asyncio.gather(later, earlier)

    assert ctx.storage.alarm_time_ms == 4_000


@pytest.mark.asyncio
async def test_job_mutation_finishes_rearming_before_propagating_cancellation(
    monkeypatch,
):
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)
    ctx = fakes.FakeCtx()
    lifecycle = Lifecycle(ctx, host=object())
    await lifecycle.start()
    gate = fakes.AsyncGate()
    ctx.storage.pause_next_alarm(gate)
    push = asyncio.create_task(
        lifecycle.jobs.push(
            LifecycleJobPushOptions(id="durable", fn="work", time=4_000)
        )
    )
    await gate.wait_until_blocked()

    push.cancel()
    await asyncio.sleep(0)

    assert not push.done()
    gate.release()
    with pytest.raises(asyncio.CancelledError):
        await push
    assert ctx.storage.alarm_time_ms == 4_000
    assert await lifecycle.jobs.get("durable") is not None


@pytest.mark.asyncio
async def test_disposal_waits_for_an_in_flight_rearm(monkeypatch):
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)
    ctx = fakes.FakeCtx()
    lifecycle = Lifecycle(ctx, host=object())
    await lifecycle.start()
    push = asyncio.create_task(
        lifecycle.jobs.push(
            LifecycleJobPushOptions(id="durable", fn="work", time=4_000)
        )
    )
    disposal = asyncio.create_task(lifecycle.dispose())

    await asyncio.gather(push, disposal)

    assert ctx.storage.alarm_time_ms == 4_000


@pytest.mark.asyncio
async def test_job_execution_options_round_trip_and_validate(monkeypatch):
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)
    ctx = fakes.FakeCtx()
    lifecycle = Lifecycle(ctx, host=object())

    job = await lifecycle.jobs.push(
        LifecycleJobPushOptions(
            id="configured",
            fn="work",
            time=4_000,
            retry={
                "maxAttempts": 4,
                "baseDelayMs": 25,
                "maxDelayMs": 500,
            },
            singleflight=True,
            hung_timeout_seconds=12,
            exclusive=True,
            recovery_loop=True,
        )
    )

    row = ctx.storage.sql.exec(
        "SELECT * FROM cf_agents_jobs WHERE id = 'configured'"
    ).toArray()[0]
    assert json.loads(row["retry_options"]) == job.retry
    assert row["singleflight"] == 1
    assert row["hung_timeout_seconds"] == 12
    assert row["exclusive"] == 1
    assert row["recovery_loop"] == 1
    assert job.singleflight
    assert job.exclusive
    assert job.recovery_loop

    invalid_options = [
        {"retry": {"maxAttempts": 0}},
        {"retry": {"maxAttempts": 1.5}},
        {"retry": {"maxAttempts": 10**400}},
        {"retry": {"baseDelayMs": 0}},
        {"retry": {"baseDelayMs": 20, "maxDelayMs": 10}},
        {"hung_timeout_seconds": 0},
        {"hung_timeout_seconds": 10**400},
    ]
    for index, values in enumerate(invalid_options):
        with pytest.raises(ValueError):
            await lifecycle.jobs.push(
                LifecycleJobPushOptions(
                    id=f"invalid-{index}",
                    fn="work",
                    time=4_000,
                    **values,
                )
            )


@pytest.mark.asyncio
async def test_alarm_candidates_honor_exclusive_and_singleflight_rows(monkeypatch):
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)
    ctx = fakes.FakeCtx()
    lifecycle = Lifecycle(ctx, host=object())
    await lifecycle.jobs.push(
        LifecycleJobPushOptions(id="ordinary", fn="work", time=100)
    )
    await lifecycle.jobs.push(
        LifecycleJobPushOptions(
            id="exclusive",
            fn="work",
            time=5_000,
            exclusive=True,
        )
    )

    assert lifecycle._job_queue.next_alarm_time(1_000) == 5_000

    assert await lifecycle.jobs.cancel("exclusive")
    assert lifecycle._job_queue.next_alarm_time(1_000) == 1_001
    assert await lifecycle.jobs.cancel("ordinary")
    await lifecycle.jobs.push(
        LifecycleJobPushOptions(
            id="singleflight",
            fn="work",
            time=100,
            singleflight=True,
            hung_timeout_seconds=2,
        )
    )
    ctx.storage.sql.exec(
        "UPDATE cf_agents_jobs SET running = 1, execution_started_at = 900 "
        "WHERE id = 'singleflight'"
    )

    assert lifecycle._job_queue.next_alarm_time(1_000) == 2_900

    assert await lifecycle.jobs.cancel("singleflight")
    await lifecycle.jobs.push(
        LifecycleJobPushOptions(
            id="exclusive-singleflight",
            fn="work",
            time=100,
            singleflight=True,
            hung_timeout_seconds=2,
            exclusive=True,
        )
    )
    ctx.storage.sql.exec(
        "UPDATE cf_agents_jobs SET running = 1, execution_started_at = 900 "
        "WHERE id = 'exclusive-singleflight'"
    )

    assert lifecycle._job_queue.next_alarm_time(1_000) == 2_900
