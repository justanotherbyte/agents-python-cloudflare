from __future__ import annotations

import json

import fakes
import pytest

import agents.schedules as schedules_module
from agents.lifecycle import Lifecycle
from agents.schedules import RetryOptions, Scheduler


def _legacy_schema(ctx):
    ctx.storage.sql.exec(
        """
        CREATE TABLE cf_agents_schedules (
          id TEXT PRIMARY KEY,
          callback TEXT NOT NULL,
          payload TEXT,
          type TEXT NOT NULL,
          time REAL NOT NULL,
          delayInSeconds REAL,
          cron TEXT,
          intervalSeconds REAL,
          retry_options TEXT,
          owner_path TEXT,
          owner_path_key TEXT
        )
        """
    )


async def _start(ctx):
    scheduler = Scheduler({})
    lifecycle = Lifecycle(ctx, host=object())
    lifecycle.use(scheduler)
    await lifecycle.start()
    return scheduler, lifecycle


@pytest.mark.asyncio
async def test_migration_converts_units_flags_owners_and_drops_obsolete_rows():
    ctx = fakes.FakeCtx()
    _legacy_schema(ctx)
    rows = [
        (
            "delay",
            "run",
            '{"value":1}',
            "delayed",
            1_700_000_010,
            10,
            None,
            None,
            '{"maxAttempts":5}',
            '[ {"name":"root", "className":"RootAgent"}, '
            '{"name":"leaf", "className":"ScheduledChild"} ]',
            "RootAgent:root/ScheduledChild:leaf",
        ),
        (
            "interval",
            "repeat",
            None,
            "interval",
            1_700_000_020,
            None,
            None,
            20,
            None,
            None,
            None,
        ),
        (
            "recovery",
            "_chatRecoveryContinue",
            "null",
            "scheduled",
            1_700_000_030,
            None,
            None,
            None,
            None,
            None,
            None,
        ),
        (
            "heartbeat",
            "_cf_keepAliveHeartbeat",
            None,
            "scheduled",
            1_700_000_040,
            None,
            None,
            None,
            None,
            None,
            None,
        ),
    ]
    for row in rows:
        ctx.storage.sql.exec(
            "INSERT INTO cf_agents_schedules VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            *row,
        )

    await _start(ctx)

    migrated = ctx.storage.sql.exec(
        "SELECT * FROM cf_agents_jobs ORDER BY id"
    ).toArray()
    by_id = {row["id"]: row for row in migrated}
    assert set(by_id) == {"delay", "interval", "recovery"}
    assert by_id["delay"]["time"] == 1_700_000_010_000
    assert json.loads(by_id["delay"]["payload"]) == {
        "type": "delayed",
        "owner_path": (
            '[{"className":"RootAgent","name":"root"},'
            '{"className":"ScheduledChild","name":"leaf"}]'
        ),
        "owner_path_key": "RootAgent:root/ScheduledChild:leaf",
        "payload": {"value": 1},
        "retry": {"maxAttempts": 5},
        "delayInSeconds": 10,
    }
    assert json.loads(by_id["delay"]["retry_options"])["maxAttempts"] == 5
    assert by_id["interval"]["singleflight"] == 1
    assert by_id["recovery"]["recovery_loop"] == 1
    assert (
        ctx.storage.sql.exec(
            "SELECT name FROM sqlite_master WHERE name = 'cf_agents_schedules'"
        ).toArray()
        == []
    )
    assert await ctx.storage.get("cf_agents:schedules_schema_version") == 2


@pytest.mark.asyncio
async def test_pre_interval_schema_migrates_without_optional_columns():
    ctx = fakes.FakeCtx()
    ctx.storage.sql.exec(
        """
        CREATE TABLE cf_agents_schedules (
          id TEXT PRIMARY KEY,
          callback TEXT NOT NULL,
          payload TEXT,
          type TEXT NOT NULL,
          time INTEGER NOT NULL
        )
        """
    )
    ctx.storage.sql.exec(
        "INSERT INTO cf_agents_schedules VALUES (?, ?, ?, ?, ?)",
        "old",
        "run",
        '"payload"',
        "scheduled",
        123,
    )

    await _start(ctx)

    [row] = ctx.storage.sql.exec(
        "SELECT id, time, payload FROM cf_agents_jobs"
    ).toArray()
    assert row["id"] == "old"
    assert row["time"] == 123_000
    assert json.loads(row["payload"])["payload"] == "payload"


@pytest.mark.asyncio
async def test_malformed_row_keeps_source_and_marker_for_retry():
    ctx = fakes.FakeCtx()
    _legacy_schema(ctx)
    ctx.storage.sql.exec(
        "INSERT INTO cf_agents_schedules "
        "(id, callback, payload, type, time) VALUES (?, ?, ?, ?, ?)",
        "broken",
        "run",
        "not json",
        "scheduled",
        123,
    )
    ctx.storage.sql.exec(
        "INSERT INTO cf_agents_schedules "
        "(id, callback, payload, type, time) VALUES (?, ?, ?, ?, ?)",
        "valid",
        "run",
        "null",
        "scheduled",
        124,
    )
    scheduler = Scheduler({})
    events = []
    lifecycle = Lifecycle(ctx, host=object(), event_listeners=(events.append,))
    lifecycle.use(scheduler)

    await lifecycle.start()

    assert ctx.storage.sql.exec(
        "SELECT id FROM cf_agents_schedules ORDER BY id"
    ).toArray() == [{"id": "broken"}, {"id": "valid"}]
    assert ctx.storage.sql.exec("SELECT id FROM cf_agents_jobs").toArray() == []
    assert await ctx.storage.get("cf_agents:schedules_schema_version") is None
    assert events[-1].type == "schedule:error"
    assert events[-1].payload["id"] == "broken"

    ctx.storage.sql.exec(
        "UPDATE cf_agents_schedules SET payload = ? WHERE id = ?",
        "null",
        "broken",
    )
    await _start(ctx)

    assert ctx.storage.sql.exec(
        "SELECT id FROM cf_agents_jobs ORDER BY id"
    ).toArray() == [{"id": "broken"}, {"id": "valid"}]
    assert await ctx.storage.get("cf_agents:schedules_schema_version") == 2


@pytest.mark.asyncio
async def test_deeply_nested_payload_keeps_legacy_source():
    ctx = fakes.FakeCtx()
    _legacy_schema(ctx)
    ctx.storage.sql.exec(
        "INSERT INTO cf_agents_schedules "
        "(id, callback, payload, type, time) VALUES (?, ?, ?, ?, ?)",
        "deep",
        "run",
        "[" * 10_000 + "null" + "]" * 10_000,
        "scheduled",
        123,
    )

    await _start(ctx)

    assert ctx.storage.sql.exec("SELECT id FROM cf_agents_jobs").toArray() == []
    assert ctx.storage.sql.exec("SELECT id FROM cf_agents_schedules").toArray() == [
        {"id": "deep"}
    ]
    assert await ctx.storage.get("cf_agents:schedules_schema_version") is None


@pytest.mark.asyncio
async def test_migration_treats_invalid_retry_json_as_absent():
    ctx = fakes.FakeCtx()
    _legacy_schema(ctx)
    ctx.storage.sql.exec(
        "INSERT INTO cf_agents_schedules "
        "(id, callback, payload, type, time, retry_options) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        "retry",
        "run",
        "null",
        "scheduled",
        123,
        "not json",
    )

    await _start(ctx)

    [row] = ctx.storage.sql.exec(
        "SELECT retry_options FROM cf_agents_jobs WHERE id = 'retry'"
    ).toArray()
    assert json.loads(row["retry_options"]) == {
        "maxAttempts": 3,
        "baseDelayMs": 100,
        "maxDelayMs": 3000,
    }


@pytest.mark.asyncio
async def test_malformed_delayed_metadata_prevents_migration_commit():
    ctx = fakes.FakeCtx()
    _legacy_schema(ctx)
    ctx.storage.sql.exec(
        "INSERT INTO cf_agents_schedules "
        "(id, callback, payload, type, time, delayInSeconds) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        "delay",
        "run",
        "null",
        "delayed",
        123,
        "bad",
    )

    await _start(ctx)

    assert ctx.storage.sql.exec("SELECT id FROM cf_agents_jobs").toArray() == []
    assert ctx.storage.sql.exec("SELECT id FROM cf_agents_schedules").toArray() == [
        {"id": "delay"}
    ]
    assert await ctx.storage.get("cf_agents:schedules_schema_version") is None


@pytest.mark.asyncio
async def test_routed_legacy_row_without_owner_key_retains_source():
    ctx = fakes.FakeCtx()
    _legacy_schema(ctx)
    ctx.storage.sql.exec(
        "INSERT INTO cf_agents_schedules "
        "(id, callback, payload, type, time, owner_path) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        "routed",
        "run",
        "null",
        "scheduled",
        123,
        '[{"className":"RootAgent","name":"root"}]',
    )

    await _start(ctx)

    assert ctx.storage.sql.exec("SELECT id FROM cf_agents_jobs").toArray() == []
    assert ctx.storage.sql.exec("SELECT id FROM cf_agents_schedules").toArray() == [
        {"id": "routed"}
    ]
    assert await ctx.storage.get("cf_agents:schedules_schema_version") is None


@pytest.mark.asyncio
async def test_routed_legacy_row_with_mismatched_owner_key_retains_source():
    ctx = fakes.FakeCtx()
    _legacy_schema(ctx)
    ctx.storage.sql.exec(
        "INSERT INTO cf_agents_schedules "
        "(id, callback, payload, type, time, owner_path, owner_path_key) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        "routed",
        "run",
        "null",
        "scheduled",
        1_700_000_000,
        '[{"className":"RootAgent","name":"root"},'
        '{"className":"ChildAgent","name":"leaf"}]',
        "RootAgent:root/ChildAgent:other",
    )

    await _start(ctx)

    assert ctx.storage.sql.exec("SELECT id FROM cf_agents_jobs").toArray() == []
    assert ctx.storage.sql.exec("SELECT id FROM cf_agents_schedules").toArray() == [
        {"id": "routed"}
    ]
    assert await ctx.storage.get("cf_agents:schedules_schema_version") is None


@pytest.mark.asyncio
async def test_retry_override_incompatible_with_current_defaults_migrates():
    ctx = fakes.FakeCtx()
    _legacy_schema(ctx)
    ctx.storage.sql.exec(
        "INSERT INTO cf_agents_schedules "
        "(id, callback, payload, type, time, retry_options) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        "retry",
        "run",
        "null",
        "scheduled",
        123,
        '{"baseDelayMs":5000}',
    )
    scheduler = Scheduler({}, retry=RetryOptions(max_delay_ms=3_000))
    lifecycle = Lifecycle(ctx, host=object())
    lifecycle.use(scheduler)

    await lifecycle.start()

    [row] = ctx.storage.sql.exec("SELECT retry_options FROM cf_agents_jobs").toArray()
    assert json.loads(row["retry_options"]) == {
        "maxAttempts": 3,
        "baseDelayMs": 5_000,
        "maxDelayMs": 3_000,
    }
    assert (
        ctx.storage.sql.exec(
            "SELECT name FROM sqlite_master WHERE name = 'cf_agents_schedules'"
        ).toArray()
        == []
    )
    assert await ctx.storage.get("cf_agents:schedules_schema_version") == 2


@pytest.mark.asyncio
async def test_semantically_invalid_retry_migrates_for_execution_time_reporting():
    ctx = fakes.FakeCtx()
    _legacy_schema(ctx)
    ctx.storage.sql.exec(
        "INSERT INTO cf_agents_schedules "
        "(id, callback, payload, type, time, retry_options) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        "retry",
        "run",
        "null",
        "scheduled",
        123,
        '{"maxAttempts":0}',
    )

    await _start(ctx)

    [row] = ctx.storage.sql.exec(
        "SELECT id, retry_options FROM cf_agents_jobs"
    ).toArray()
    assert row["id"] == "retry"
    assert json.loads(row["retry_options"]) == {
        "maxAttempts": 0,
        "baseDelayMs": 100,
        "maxDelayMs": 3_000,
    }
    assert (
        ctx.storage.sql.exec(
            "SELECT name FROM sqlite_master WHERE name = 'cf_agents_schedules'"
        ).toArray()
        == []
    )
    assert await ctx.storage.get("cf_agents:schedules_schema_version") == 2


@pytest.mark.asyncio
async def test_migration_collision_distinguishes_booleans_from_numbers():
    ctx = fakes.FakeCtx()
    scheduler = Scheduler({"run": lambda payload, schedule: None})
    lifecycle = Lifecycle(ctx, host=object())
    lifecycle.use(scheduler)
    await lifecycle.start()
    scheduled = await scheduler.every(10, "run", {"value": True})
    job = await scheduler.lifecycle.jobs._get_unvalidated_retry(scheduled.id)
    assert job is not None
    payload = {**job.payload, "payload": {"value": 1}}

    with pytest.raises(ValueError, match="id collision"):
        await scheduler._import_legacy_schedules(
            [
                schedules_module._PreparedLegacySchedule(
                    id=job.id,
                    callback=job.fn,
                    time_ms=job.time,
                    payload=payload,
                    retry=job.retry,
                    singleflight=job.singleflight,
                    recovery_loop=job.recovery_loop,
                )
            ]
        )


@pytest.mark.asyncio
async def test_oversized_time_keeps_legacy_source():
    ctx = fakes.FakeCtx()
    _legacy_schema(ctx)
    ctx.storage.sql.exec(
        "INSERT INTO cf_agents_schedules "
        "(id, callback, payload, type, time) VALUES (?, ?, ?, ?, ?)",
        "oversized",
        "run",
        "null",
        "scheduled",
        1e20,
    )

    await _start(ctx)

    assert ctx.storage.sql.exec("SELECT id FROM cf_agents_jobs").toArray() == []
    assert ctx.storage.sql.exec("SELECT id FROM cf_agents_schedules").toArray() == [
        {"id": "oversized"}
    ]
    assert await ctx.storage.get("cf_agents:schedules_schema_version") is None


@pytest.mark.asyncio
async def test_oversized_schema_marker_degrades_to_version_zero():
    ctx = fakes.FakeCtx()
    await ctx.storage.put("cf_agents:schedules_schema_version", 10**400)

    await _start(ctx)

    assert await ctx.storage.get("cf_agents:schedules_schema_version") == 2


@pytest.mark.asyncio
async def test_future_marker_leaves_legacy_table_untouched():
    ctx = fakes.FakeCtx()
    _legacy_schema(ctx)
    await ctx.storage.put("cf_agents:schedules_schema_version", 3)

    await _start(ctx)

    assert ctx.storage.sql.exec(
        "SELECT name FROM sqlite_master WHERE name = 'cf_agents_schedules'"
    ).toArray() == [{"name": "cf_agents_schedules"}]
    assert await ctx.storage.get("cf_agents:schedules_schema_version") == 3
