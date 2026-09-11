from __future__ import annotations

import asyncio

import fakes
import pytest


@pytest.mark.asyncio
async def test_storage_kv_is_ordered_and_copy_isolated():
    storage = fakes.FakeStorage(fakes.new_sqlite())

    await storage.put("jobs:b", {"attempt": 2})
    await storage.put("jobs:a", {"attempt": 1})

    value = await storage.get("jobs:a")
    value["attempt"] = 9

    assert await storage.get("jobs:a") == {"attempt": 1}
    assert await storage.list({"prefix": "jobs:"}) == {
        "jobs:a": {"attempt": 1},
        "jobs:b": {"attempt": 2},
    }
    assert await storage.delete("jobs:a") is True
    assert await storage.get("jobs:a") is None


@pytest.mark.asyncio
async def test_storage_kv_supports_bulk_operations_and_list_options():
    storage = fakes.FakeStorage(fakes.new_sqlite())
    await storage.put({"item:c": 3, "item:a": 1, "item:b": 2, "other": 0})

    assert await storage.get(["item:c", "missing", "item:a"]) == {
        "item:a": 1,
        "item:c": 3,
    }
    assert await storage.list(
        {"prefix": "item:", "startAfter": "item:a", "reverse": True, "limit": 1}
    ) == {"item:c": 3}
    assert await storage.delete(["item:a", "missing", "item:c"]) == 2


@pytest.mark.asyncio
async def test_storage_kv_rejects_invalid_put_and_list_options():
    storage = fakes.FakeStorage(fakes.new_sqlite())

    with pytest.raises(TypeError, match="put requires a value"):
        await storage.put("missing-value")
    with pytest.raises(TypeError, match="start and startAfter"):
        await storage.list({"start": "a", "startAfter": "b"})


def test_storage_transaction_sync_commits_or_rolls_back_sql():
    storage = fakes.FakeStorage(fakes.new_sqlite())
    storage.sql.exec("CREATE TABLE events (value TEXT NOT NULL)")

    def commit() -> str:
        storage.sql.exec("INSERT INTO events VALUES ('kept')")
        return "ok"

    result = storage.transactionSync(commit)

    def fail() -> None:
        storage.sql.exec("INSERT INTO events VALUES ('discarded')")
        raise RuntimeError("stop")

    with pytest.raises(RuntimeError, match="stop"):
        storage.transactionSync(fail)

    assert result == "ok"
    rows = storage.sql.exec("SELECT value FROM events").toArray()
    assert rows == [{"value": "kept"}]


def test_storage_transaction_sync_rejects_async_callbacks():
    storage = fakes.FakeStorage(fakes.new_sqlite())
    storage.sql.exec("CREATE TABLE events (value TEXT NOT NULL)")

    async def invalid_callback() -> None:
        pass

    callback_result = invalid_callback()

    def return_coroutine():
        storage.sql.exec("INSERT INTO events VALUES ('discarded')")
        return callback_result

    with pytest.raises(TypeError, match="synchronous callback"):
        storage.transactionSync(return_coroutine)

    assert callback_result.cr_frame is None
    assert storage.sql.exec("SELECT value FROM events").toArray() == []


@pytest.mark.asyncio
async def test_storage_alarm_gate_exposes_commit_order():
    storage = fakes.FakeStorage(fakes.new_sqlite())
    gate = fakes.AsyncGate()
    storage.pause_next_alarm(gate)

    older_write = asyncio.create_task(storage.setAlarm(100))
    await gate.wait_until_blocked()
    await storage.setAlarm(50)
    gate.release()
    await older_write

    assert await storage.getAlarm() == 100
    assert storage.alarm_history_ms == (("set", 50), ("set", 100))

    await storage.deleteAlarm()
    assert await storage.getAlarm() is None
    assert storage.alarm_history_ms[-1] == ("delete", None)


@pytest.mark.asyncio
async def test_durable_runtime_preserves_storage_and_sockets_across_eviction():
    runtime = fakes.FakeDurableObjectRuntime("agent-1")
    first = runtime.new_context()
    socket = fakes.FakeSocket({"id": "socket-1"})

    await first.storage.put("ready", True)
    first.storage.sql.exec("CREATE TABLE values_table (value INTEGER NOT NULL)")
    first.storage.sql.exec("INSERT INTO values_table VALUES (7)")
    first.acceptWebSocket(socket, ["room:one"])
    first.exports = object()

    second = runtime.evict()

    assert second is not first
    assert second.id.name == "agent-1"
    assert await second.storage.get("ready") is True
    assert second.storage.sql.exec("SELECT value FROM values_table").toArray() == [
        {"value": 7}
    ]
    assert second.getWebSockets() == [socket]
    assert second.getWebSockets("room:one") == [socket]
    assert second.getWebSockets("room:two") == []
    assert second.accepted_websockets == []
    assert second.exports is None
    assert runtime.evictions == 1


@pytest.mark.asyncio
async def test_route_transport_delivers_a_versioned_owned_envelope():
    transport = fakes.FakeRouteTransport("root", max_payload_bytes=64)
    transport.register("root/facet-a", "tasks", lambda payload: payload["task_id"])

    result = await transport.route(
        version=1,
        source="root",
        target="root/facet-a",
        capability_id="tasks",
        payload={"task_id": "task-1"},
    )

    assert result == "task-1"
    assert transport.calls == (
        {
            "version": 1,
            "source": "root",
            "target": "root/facet-a",
            "capability_id": "tasks",
            "payload": {"task_id": "task-1"},
        },
    )


@pytest.mark.asyncio
async def test_route_transport_rejects_invalid_envelopes():
    transport = fakes.FakeRouteTransport("root", max_payload_bytes=16)
    transport.register("root/facet-a", "tasks", lambda payload: payload)

    with pytest.raises(ValueError, match="route version"):
        await transport.route(
            version=2,
            source="root",
            target="root/facet-a",
            capability_id="tasks",
            payload={},
        )
    with pytest.raises(PermissionError, match="route address"):
        await transport.route(
            version=1,
            source="other",
            target="root/facet-a",
            capability_id="tasks",
            payload={},
        )
    with pytest.raises(PermissionError, match="route address"):
        await transport.route(
            version=1,
            source="root",
            target="other/facet-a",
            capability_id="tasks",
            payload={},
        )
    with pytest.raises(LookupError, match="capability"):
        await transport.route(
            version=1,
            source="root",
            target="root/facet-a",
            capability_id="missing",
            payload={},
        )
    with pytest.raises(ValueError, match="route payload"):
        await transport.route(
            version=1,
            source="root",
            target="root/facet-a",
            capability_id="tasks",
            payload={"value": "x" * 20},
        )

    assert transport.calls == ()


def test_resource_tracker_enforces_one_owner_and_idempotent_release():
    tracker = fakes.FakeResourceTracker()
    release = tracker.claim("proxy", "facet-a", "root")

    with pytest.raises(RuntimeError, match="already claimed"):
        tracker.claim("proxy", "facet-a", "other")

    active = tracker.active
    active.clear()
    assert tracker.active == {("proxy", "facet-a"): "root"}
    release()
    release()

    assert tracker.active == {}
    assert tracker.events == (
        ("claim", "proxy", "facet-a", "root"),
        ("release", "proxy", "facet-a", "root"),
    )


@pytest.mark.asyncio
async def test_wait_until_records_owner_context_and_completion():
    recorder = fakes.WaitUntilRecorder(owner="root", context="alarm")

    async def work() -> int:
        return 7

    recorder(work())

    assert recorder.records == [
        {
            "owner": "root",
            "context": "alarm",
            "status": "pending",
            "result": None,
            "error": None,
        }
    ]

    records = recorder.records
    records[0]["status"] = "changed"
    assert recorder.records[0]["status"] == "pending"

    assert await recorder.drain_next() == 7
    assert recorder.records[0]["status"] == "completed"
    assert recorder.records[0]["result"] == 7


@pytest.mark.asyncio
async def test_wait_until_records_failure():
    recorder = fakes.WaitUntilRecorder(owner="facet-a", context="task")
    error = RuntimeError("lost context")

    async def fail() -> None:
        raise error

    recorder(fail())

    with pytest.raises(RuntimeError, match="lost context"):
        await recorder.drain_next()

    assert recorder.records[0]["status"] == "failed"
    assert recorder.records[0]["error"] is error


def test_wait_until_cancels_pending_work():
    recorder = fakes.WaitUntilRecorder(owner="root", context="shutdown")

    async def pending() -> None:
        raise AssertionError("cancelled work must not run")

    recorder(pending())
    try:
        recorder.cancel_all()
        assert recorder.coros == ()
        assert recorder.records[0]["status"] == "cancelled"
    finally:
        recorder.close()


@pytest.mark.asyncio
async def test_wait_until_cancels_running_work():
    recorder = fakes.WaitUntilRecorder(owner="root", context="shutdown")
    started = asyncio.Event()
    release = asyncio.Event()

    async def running() -> None:
        started.set()
        await release.wait()

    recorder(running())
    drain = asyncio.create_task(recorder.drain_next())
    await started.wait()

    recorder.cancel_all()

    with pytest.raises(asyncio.CancelledError):
        await drain
    assert recorder.records[0]["status"] == "cancelled"
