from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import cast

import fakes
import pytest

import agents.lifecycle._runtime as lifecycle_module
import agents.tasks as tasks_module
from agents import Agent
from agents.lifecycle import (
    Lifecycle,
    LifecycleMemoryLimitContext,
    LifecycleRouteAddress,
    LifecycleRouteEnvelope,
    get_current_lifecycle_context,
)
from agents.lifecycle.jobs import LifecycleJob, LifecycleJobPushOptions
from agents.tasks import (
    MAX_SERIALIZED_BYTES,
    NonRetryableError,
    TASK_UNDEFINED,
    TaskDeleteOptions,
    TaskListOptions,
    TaskRunState,
    TaskSerializationError,
    TaskStepConfig,
    TaskStepRetryOptions,
    TaskStorageError,
    TaskWakeCollisionError,
    Tasks,
    task_definition,
)


async def _started(definitions=None, *, events=None):
    ctx = fakes.FakeCtx()
    tasks = Tasks(definitions)
    lifecycle = Lifecycle(
        ctx,
        host=object(),
        event_listeners=(() if events is None else (events.append,)),
    )
    lifecycle.use(tasks)
    await lifecycle.start()
    return tasks, lifecycle, ctx


async def _retained_started(
    definitions=None,
    *,
    events=None,
    retries=None,
    step_timeout=None,
    on_error=None,
):
    ctx = fakes.FakeCtx()
    retained = fakes.WaitUntilRecorder()
    host = object()
    tasks = Tasks(
        definitions,
        retries=retries,
        step_timeout=step_timeout,
        on_error=on_error,
    )
    lifecycle = Lifecycle(
        ctx,
        host=host,
        retain_work=retained,
        event_listeners=(() if events is None else (events.append,)),
    )
    lifecycle.use(tasks)
    await lifecycle.start()
    return tasks, lifecycle, ctx, retained, host


def _insert_run(
    ctx,
    run_id,
    state,
    *,
    definition="report",
    result=None,
    error_name=None,
    error_message=None,
    status_message=None,
    metadata=None,
    idempotency_key=None,
    retain=1,
    attempt=0,
    generation=None,
    next_at=None,
    wait_reason=None,
    cancel_requested=0,
    cancel_reason=None,
    created_at=100,
    started_at=None,
    updated_at=200,
    settled_at=None,
):
    ctx.storage.sql.exec(
        """
        INSERT INTO cf_agents_task_runs
          (run_id, definition, input, state, result, error_name, error_message,
           status_message, metadata, idempotency_key, retain, attempt,
           generation, next_at, wait_reason, cancel_requested, cancel_reason,
           created_at, started_at, updated_at, settled_at)
        VALUES (?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        run_id,
        definition,
        state,
        result,
        error_name,
        error_message,
        status_message,
        metadata,
        idempotency_key,
        retain,
        attempt,
        generation,
        next_at,
        wait_reason,
        cancel_requested,
        cancel_reason,
        created_at,
        started_at,
        updated_at,
        settled_at,
    )


def test_tasks_module_exports_the_experimental_public_surface():
    assert set(tasks_module.__all__) == {
        "CancelledTaskRunSnapshot",
        "CompletedTaskRunSnapshot",
        "DuplicateTaskStepError",
        "FailedTaskRunSnapshot",
        "MAX_SERIALIZED_BYTES",
        "MissingTaskDefinitionError",
        "NonRetryableError",
        "PendingTaskRunSnapshot",
        "RunningTaskRunSnapshot",
        "TASK_UNDEFINED",
        "Task",
        "TaskBackoff",
        "TaskCallbacks",
        "TaskDeleteOptions",
        "TaskDurationString",
        "TaskDurationUnit",
        "TaskError",
        "TaskEventType",
        "TaskHandlers",
        "TaskInterruptedStep",
        "TaskJson",
        "TaskListOptions",
        "TaskMemoryLimitSealed",
        "TaskReceipt",
        "TaskReplayDivergedError",
        "TaskRunOptions",
        "TaskRunSnapshot",
        "TaskRunState",
        "TaskSerializationError",
        "TaskStep",
        "TaskStepAttempt",
        "TaskStepConfig",
        "TaskStepRetryOptions",
        "TaskStorageError",
        "TaskTerminalState",
        "TaskUndefined",
        "TaskValue",
        "TaskWaitReason",
        "TaskWakeCollisionError",
        "Tasks",
        "TasksOptions",
        "WaitingTaskRunSnapshot",
        "task_definition",
    }


@pytest.mark.asyncio
async def test_schema_matches_the_shared_contract_and_stamps_last():
    _, _, ctx = await _started({})

    columns = ctx.storage.sql.exec("PRAGMA table_info(cf_agents_task_runs)").toArray()
    assert [column["name"] for column in columns] == [
        "run_id",
        "definition",
        "input",
        "state",
        "result",
        "error_name",
        "error_message",
        "status_message",
        "metadata",
        "idempotency_key",
        "retain",
        "attempt",
        "generation",
        "next_at",
        "wait_reason",
        "cancel_requested",
        "cancel_reason",
        "created_at",
        "started_at",
        "updated_at",
        "settled_at",
    ]
    assert ctx.storage.sql.exec(
        "PRAGMA index_info(cf_agents_task_runs_definition)"
    ).toArray() == [
        {"seqno": 0, "cid": 1, "name": "definition"},
        {"seqno": 1, "cid": 17, "name": "created_at"},
    ]
    assert [
        column["name"]
        for column in ctx.storage.sql.exec(
            "PRAGMA table_info(cf_agents_task_steps)"
        ).toArray()
    ] == [
        "run_id",
        "step_name",
        "kind",
        "state",
        "result",
        "error_name",
        "error_message",
        "attempt",
        "next_at",
        "created_at",
        "started_at",
        "updated_at",
        "completed_at",
    ]
    assert await ctx.storage.get("cf_agents:tasks_schema_version") == 1

    ctx.storage.sql.exec(
        """
        INSERT INTO cf_agents_task_runs
          (run_id, definition, state, created_at, updated_at)
        VALUES ('defaults', 'report', 'pending', 1, 1)
        """
    )
    [defaults] = ctx.storage.sql.exec(
        "SELECT retain, attempt, cancel_requested FROM cf_agents_task_runs"
    ).toArray()
    assert defaults == {"retain": 1, "attempt": 0, "cancel_requested": 0}
    with pytest.raises(Exception, match="CHECK constraint failed"):
        ctx.storage.sql.exec(
            """
            INSERT INTO cf_agents_task_runs
              (run_id, definition, state, created_at, updated_at)
            VALUES ('invalid', 'report', 'unknown', 1, 1)
            """
        )
    with pytest.raises(Exception, match="no such column: rowid"):
        ctx.storage.sql.exec("SELECT rowid FROM cf_agents_task_runs")


@pytest.mark.asyncio
async def test_failed_preparation_does_not_stamp_the_schema(monkeypatch):
    ctx = fakes.FakeCtx()
    tasks = Tasks({})
    lifecycle = Lifecycle(ctx, host=object())
    lifecycle.use(tasks)

    def fail_prepare(store):
        store._sql.execute(
            "CREATE TABLE cf_agents_task_runs_partial (run_id TEXT PRIMARY KEY)"
        )
        raise RuntimeError("interrupted")

    monkeypatch.setattr(tasks_module._TaskStore, "prepare", fail_prepare)

    with pytest.raises(RuntimeError, match="interrupted"):
        await lifecycle.start()

    assert await ctx.storage.get("cf_agents:tasks_schema_version") is None
    assert ctx.storage.sql.exec(
        "SELECT name FROM sqlite_master WHERE name = 'cf_agents_task_runs_partial'"
    ).toArray() == [{"name": "cf_agents_task_runs_partial"}]


@pytest.mark.asyncio
async def test_current_schema_marker_reconciles_missing_tables():
    ctx = fakes.FakeCtx()
    await ctx.storage.put("cf_agents:tasks_schema_version", 1)
    tasks = Tasks({})
    lifecycle = Lifecycle(ctx, host=object())
    lifecycle.use(tasks)

    await lifecycle.start()

    tables = ctx.storage.sql.exec(
        "SELECT name FROM sqlite_master WHERE name LIKE 'cf_agents_task_%' "
        "ORDER BY name"
    ).toArray()
    assert tables == [
        {"name": "cf_agents_task_runs"},
        {"name": "cf_agents_task_runs_definition"},
        {"name": "cf_agents_task_steps"},
    ]


@pytest.mark.asyncio
async def test_malformed_current_schema_disables_only_tasks():
    ctx = fakes.FakeCtx()
    ctx.storage.sql.exec("CREATE TABLE cf_agents_task_runs (run_id TEXT PRIMARY KEY)")
    await ctx.storage.put("cf_agents:tasks_schema_version", 1)
    errors = []
    tasks = Tasks({}, on_error=errors.append)
    lifecycle = Lifecycle(ctx, host=object())
    lifecycle.use(tasks)

    await lifecycle.start()

    assert len(errors) == 1
    assert isinstance(errors[0], TaskStorageError)
    with pytest.raises(TaskStorageError, match="do not match schema"):
        await tasks.get("run")


@pytest.mark.asyncio
async def test_future_schema_marker_is_not_downgraded():
    ctx = fakes.FakeCtx()
    await ctx.storage.put("cf_agents:tasks_schema_version", 2)
    tasks = Tasks({})
    lifecycle = Lifecycle(ctx, host=object())
    lifecycle.use(tasks)

    await lifecycle.start()

    assert await ctx.storage.get("cf_agents:tasks_schema_version") == 2
    assert (
        ctx.storage.sql.exec(
            "SELECT name FROM sqlite_master WHERE name LIKE 'cf_agents_task_%'"
        ).toArray()
        == []
    )


def test_definition_maps_are_copied_and_names_use_utf16_bounds():
    def callback(input, step):
        pass

    definitions = {"report": callback}
    tasks = Tasks(definitions)
    definitions.clear()

    assert tasks.handle("report").name == "report"
    with pytest.raises(ValueError, match="unknown Task definition"):
        tasks.handle("missing")

    longest = "\U0001f600" * 128
    assert Tasks({longest: callback}).handle(longest).name == longest
    too_long = longest + "\U0001f600"
    with pytest.raises(ValueError, match="cannot exceed 256"):
        Tasks({too_long: callback}).handle(too_long)

    with pytest.raises(ValueError, match="non-empty"):
        tasks.handle("")


@pytest.mark.asyncio
async def test_reserved_definitions_use_only_the_private_aperture():
    def callback(input, step):
        pass

    tasks = Tasks({"report": callback})
    tasks._register_reserved_definition("__cf_chat", callback)

    with pytest.raises(ValueError, match="public Task definitions"):
        tasks.handle("__cf_chat")
    with pytest.raises(ValueError, match="'__cf' prefix"):
        tasks._register_reserved_definition("ordinary", callback)
    with pytest.raises(ValueError, match="already registered"):
        tasks._register_reserved_definition("__cf_chat", callback)

    ctx = fakes.FakeCtx()
    lifecycle = Lifecycle(ctx, host=object())
    lifecycle.use(tasks)
    receipt = await tasks._run_reserved("__cf_chat", run_id="reserved")

    assert receipt.run_id == "reserved"
    assert (await tasks.get("reserved")).definition == "__cf_chat"


@pytest.mark.asyncio
async def test_acceptance_persists_json_and_joins_existing_runs(monkeypatch):
    events = []
    calls = []
    tasks, _, ctx = await _started(
        {
            "report": lambda input, step: calls.append(input),
            "other": lambda input, step: None,
        },
        events=events,
    )
    monkeypatch.setattr(tasks_module, "now_ms", lambda: 1_700_000_000_000)

    first = await tasks.run(
        "report",
        {"account": 7},
        run_id="run-1",
        idempotency_key="daily",
        metadata={"source": "test"},
        retain=False,
    )
    joined_by_key = await tasks.run(
        "report",
        {"ignored": True},
        run_id="unused",
        idempotency_key="daily",
    )
    joined_by_id = await tasks.run("report", run_id="run-1")

    assert first.accepted is True
    assert first.state == "pending"
    assert joined_by_key.run_id == "run-1"
    assert joined_by_key.accepted is False
    assert joined_by_id.accepted is False
    assert calls == []
    [row] = ctx.storage.sql.exec(
        "SELECT input, metadata, retain, next_at FROM cf_agents_task_runs"
    ).toArray()
    assert row == {
        "input": '{"account":7}',
        "metadata": '{"source":"test"}',
        "retain": 0,
        "next_at": 1_700_000_000_000,
    }
    assert [(event.type, event.payload) for event in events] == [
        (
            "task:accepted",
            {"runId": "run-1", "definition": "report", "accepted": True},
        )
    ]

    with pytest.raises(ValueError, match="conflicting idempotency key"):
        await tasks.run("report", run_id="run-1", idempotency_key="other-key")
    with pytest.raises(ValueError, match="belongs to definition"):
        await tasks.run("other", idempotency_key="daily")


@pytest.mark.asyncio
async def test_acceptance_rejects_invalid_or_oversized_json():
    tasks, _, ctx = await _started({"report": lambda input, step: None})

    with pytest.raises(TaskSerializationError):
        await tasks.run("report", float("nan"), run_id="nan")
    with pytest.raises(TaskSerializationError, match="serialized size"):
        await tasks.run("report", "x" * MAX_SERIALIZED_BYTES, run_id="large")
    with pytest.raises(TaskSerializationError, match="keys must be strings"):
        await tasks.run(
            "report",
            run_id="metadata",
            metadata=cast(dict, {1: "invalid"}),
        )
    with pytest.raises(TaskSerializationError, match="keys must be strings"):
        await tasks.run("report", {"nested": {1: "invalid"}}, run_id="input")
    with pytest.raises(TaskSerializationError, match="keys must be strings"):
        await tasks.run(
            "report",
            run_id="nested-metadata",
            metadata=cast(dict, {"nested": {1: "invalid"}}),
        )
    cycle = []
    cycle.append(cycle)
    with pytest.raises(TaskSerializationError, match="Circular reference"):
        await tasks.run("report", cycle, run_id="cycle")

    assert (
        ctx.storage.sql.exec("SELECT run_id FROM cf_agents_task_runs").toArray() == []
    )


@pytest.mark.asyncio
async def test_acceptance_sizes_utf8_and_escapes_lone_surrogates():
    tasks, _, ctx = await _started({"report": lambda input, step: None})

    await tasks.run("report", "\U0001f600" * 200_000, run_id="emoji")
    await tasks.run("report", "\ud800", run_id="surrogate")

    assert ctx.storage.sql.exec(
        "SELECT run_id, input FROM cf_agents_task_runs ORDER BY run_id"
    ).toArray() == [
        {"run_id": "emoji", "input": '"' + "\U0001f600" * 200_000 + '"'},
        {"run_id": "surrogate", "input": '"\\ud800"'},
    ]


@pytest.mark.asyncio
async def test_snapshots_project_each_state_and_mixed_runtime_nulls():
    tasks, _, ctx = await _started({"report": lambda input, step: None})
    _insert_run(ctx, "pending", "pending", metadata='{"source":"ts"}')
    _insert_run(ctx, "running", "running", attempt=2, started_at=0)
    _insert_run(
        ctx,
        "waiting",
        "waiting",
        status_message="retrying",
        next_at=0,
        wait_reason=None,
    )
    _insert_run(ctx, "complete-null", "completed", result="null", settled_at=0)
    _insert_run(ctx, "complete-undefined", "completed", result=None)
    _insert_run(
        ctx,
        "failed",
        "failed",
        error_name=None,
        error_message=None,
    )
    _insert_run(
        ctx,
        "failed-empty",
        "failed",
        error_name="",
        error_message="",
    )
    _insert_run(
        ctx,
        "cancelled",
        "cancelled",
        cancel_reason="stopped",
        settled_at=300,
    )

    assert (await tasks.get("pending")).metadata == {"source": "ts"}
    running = await tasks.get("running")
    assert (running.attempt, running.started_at) == (2, 0)
    waiting = await tasks.get("waiting")
    assert (waiting.reason, waiting.wake_at, waiting.status_message) == (
        "sleep",
        0,
        "retrying",
    )
    assert (await tasks.get("complete-null")).result is None
    assert (await tasks.get("complete-undefined")).result is TASK_UNDEFINED
    failed = await tasks.get("failed")
    assert (failed.error.name, failed.error.message) == ("Error", "Task run failed")
    failed_empty = await tasks.get("failed-empty")
    assert (failed_empty.error.name, failed_empty.error.message) == ("", "")
    cancelled = await tasks.get("cancelled")
    assert (cancelled.reason, cancelled.settled_at) == ("stopped", 300)


@pytest.mark.asyncio
async def test_handles_scope_lookup_and_cancellation_to_one_definition(monkeypatch):
    events = []
    tasks, _, ctx = await _started(
        {"one": lambda input, step: None, "two": lambda input, step: None},
        events=events,
    )
    monkeypatch.setattr(tasks_module, "now_ms", lambda: 500)
    _insert_run(
        ctx,
        "two-run",
        "waiting",
        definition="two",
        idempotency_key="two-key",
        next_at=600,
    )
    await tasks.lifecycle.jobs.push(
        LifecycleJobPushOptions(
            id="task:two-run",
            fn="wake",
            time=600,
            payload={"runId": "two-run"},
        )
    )
    one = tasks.handle("one")
    two = tasks.handle("two")

    assert await one.get("two-run") is None
    assert await one.get_by_idempotency_key("two-key") is None
    assert await one.cancel("two-run") is False
    assert await two.cancel("two-run", "stop") is True
    assert await two.cancel("two-run", "again") is False
    snapshot = await two.get("two-run")
    assert (snapshot.state, snapshot.reason, snapshot.settled_at) == (
        "cancelled",
        "stop",
        500,
    )
    assert events[-1].type == "task:cancelled"
    assert events[-1].payload == {
        "runId": "two-run",
        "definition": "two",
        "reason": "stop",
    }
    assert await tasks.lifecycle.jobs.get("task:two-run") is None


@pytest.mark.asyncio
async def test_cancel_retry_repairs_a_stale_root_wake(monkeypatch):
    events = []
    tasks, lifecycle, ctx = await _started(
        {"report": lambda input, step: None},
        events=events,
    )
    _insert_run(ctx, "run", "waiting", next_at=600)
    await tasks.lifecycle.jobs.push(
        LifecycleJobPushOptions(
            id="task:run",
            fn="wake",
            time=600,
            payload={"runId": "run"},
        )
    )
    original_cancel = lifecycle._cancel_job
    calls = 0

    async def fail_once(owner, job_id):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("cleanup failed")
        return await original_cancel(owner, job_id)

    monkeypatch.setattr(lifecycle, "_cancel_job", fail_once)

    with pytest.raises(RuntimeError, match="cleanup failed"):
        await tasks.cancel("run", "stop")

    assert (await tasks.get("run")).state == "cancelled"
    assert events[-1].type == "task:cancelled"
    assert await tasks.lifecycle.jobs.get("task:run") is not None
    assert await tasks.cancel("run", "stop") is False
    assert await tasks.lifecycle.jobs.get("task:run") is None


@pytest.mark.asyncio
async def test_list_is_newest_first_filtered_and_bounded():
    tasks, _, ctx = await _started(
        {"one": lambda input, step: None, "two": lambda input, step: None}
    )
    _insert_run(ctx, "a", "pending", definition="one", created_at=100)
    _insert_run(ctx, "b", "cancelled", definition="one", created_at=100)
    _insert_run(ctx, "c", "cancelled", definition="two", created_at=101)

    assert [snapshot.run_id for snapshot in await tasks.list()] == ["c", "b", "a"]
    assert [
        snapshot.run_id
        for snapshot in await tasks.list(
            TaskListOptions(definition="one", state="cancelled")
        )
    ] == ["b"]
    assert [
        snapshot.run_id
        for snapshot in await tasks.list(
            TaskListOptions(state=("pending", "cancelled"), limit=2)
        )
    ] == ["c", "b"]

    for limit in (0, 101, True):
        with pytest.raises((TypeError, ValueError)):
            await tasks.list(TaskListOptions(limit=cast(int, limit)))
    with pytest.raises(ValueError, match="unknown Task state"):
        await tasks.list(TaskListOptions(state=cast(TaskRunState, "unknown")))


@pytest.mark.asyncio
async def test_delete_removes_oldest_terminal_runs_and_their_steps():
    events = []
    tasks, _, ctx = await _started(
        {"report": lambda input, step: None},
        events=events,
    )
    _insert_run(ctx, "complete", "completed", settled_at=100)
    _insert_run(ctx, "failed", "failed", settled_at=200)
    _insert_run(ctx, "cancelled", "cancelled", settled_at=300)
    _insert_run(ctx, "pending", "pending", settled_at=None)
    for run_id in ("complete", "failed", "pending"):
        ctx.storage.sql.exec(
            """
            INSERT INTO cf_agents_task_steps
              (run_id, step_name, kind, state, attempt, created_at, updated_at)
            VALUES (?, 'work', 'do', 'completed', 1, 1, 1)
            """,
            run_id,
        )

    deleted = await tasks.delete(
        TaskDeleteOptions(
            state=("completed", "failed"),
            settled_before=datetime.fromtimestamp(0.25, tz=UTC),
            limit=1,
        )
    )

    assert deleted == 1
    assert ctx.storage.sql.exec(
        "SELECT run_id FROM cf_agents_task_runs ORDER BY run_id"
    ).toArray() == [
        {"run_id": "cancelled"},
        {"run_id": "failed"},
        {"run_id": "pending"},
    ]
    assert ctx.storage.sql.exec(
        "SELECT run_id FROM cf_agents_task_steps ORDER BY run_id"
    ).toArray() == [{"run_id": "failed"}, {"run_id": "pending"}]
    assert events[-1].type == "task:deleted"
    assert events[-1].payload == {
        "runId": "complete",
        "definition": "report",
    }

    assert await tasks.delete(TaskDeleteOptions(state=())) == 0
    with pytest.raises(ValueError, match="non-terminal"):
        await tasks.delete(TaskDeleteOptions(state=cast(tuple, ("pending",))))


@pytest.mark.asyncio
async def test_delete_rolls_back_journals_when_a_run_delete_fails(monkeypatch):
    tasks, _, ctx = await _started({"report": lambda input, step: None})
    _insert_run(ctx, "complete", "completed", settled_at=100)
    ctx.storage.sql.exec(
        """
        INSERT INTO cf_agents_task_steps
          (run_id, step_name, kind, state, attempt, created_at, updated_at)
        VALUES ('complete', 'work', 'do', 'completed', 1, 1, 1)
        """
    )
    original_execute = tasks._task_store()._sql.execute

    class FailingSql:
        def execute(self, query, *params):
            if query.startswith("DELETE FROM cf_agents_task_runs"):
                raise RuntimeError("delete failed")
            return original_execute(query, *params)

    monkeypatch.setattr(tasks._task_store(), "_sql", FailingSql())

    with pytest.raises(RuntimeError, match="delete failed"):
        await tasks.delete()

    assert ctx.storage.sql.exec("SELECT run_id FROM cf_agents_task_runs").toArray() == [
        {"run_id": "complete"}
    ]
    assert ctx.storage.sql.exec(
        "SELECT run_id FROM cf_agents_task_steps"
    ).toArray() == [{"run_id": "complete"}]


@pytest.mark.asyncio
async def test_agent_discovers_fresh_task_definitions_without_descriptors():
    class HostileDescriptor:
        def __get__(self, instance, owner):
            raise AssertionError("descriptor evaluated")

    class DefinedAgent(Agent):
        hostile = HostileDescriptor()

        @task_definition()
        async def report(self, input, step):
            return input

    first = fakes.build_agent(cls=Agent)
    second = fakes.build_agent(cls=Agent)
    defined = fakes.build_agent(cls=DefinedAgent)

    assert isinstance(first.tasks, Tasks)
    assert first.tasks is not second.tasks
    assert first.tasks._definitions == {}
    assert first.tasks._definitions is not second.tasks._definitions
    assert set(defined.tasks._definitions) == {"report"}
    assert await defined.tasks._definitions["report"]("done", cast(object, None)) == (
        "done"
    )
    with pytest.raises(ValueError, match="unknown Task definition"):
        await first.tasks.run("missing")


def test_task_definition_overrides_require_redecoration():
    class BaseAgent(Agent):
        @task_definition()
        async def report(self, input, step):
            return "base"

    class HiddenAgent(BaseAgent):
        async def report(self, input, step):
            return "hidden"

    class RestoredAgent(BaseAgent):
        @task_definition()
        async def report(self, input, step):
            return "restored"

    assert fakes.build_agent(cls=BaseAgent).tasks._definitions.keys() == {"report"}
    assert fakes.build_agent(cls=HiddenAgent).tasks._definitions == {}
    assert fakes.build_agent(cls=RestoredAgent).tasks._definitions.keys() == {"report"}


@pytest.mark.asyncio
async def test_do_replays_completed_steps_and_retries_from_the_frontier(monkeypatch):
    clock = [1_000]
    events = []
    handler_calls = []
    stable_calls = []
    stable_results = []
    flaky_attempts = []
    swallowed_errors = []

    async def report(input, step):
        handler_calls.append((input, step.interrupted))

        @step.do("stable")
        async def stable():
            stable_calls.append("called")
            return (input["value"],)

        stable_value = await stable()
        stable_results.append(stable_value)
        await step.status(f"stable={stable_value}")

        async def flaky(attempt):
            flaky_attempts.append(
                (attempt.attempt, attempt.idempotency_key, attempt.signal.aborted)
            )
            if attempt.attempt == 1:
                raise RuntimeError("try again")
            return stable_value[0] * 2

        try:
            return await step.do(
                "flaky",
                TaskStepConfig(
                    retries=TaskStepRetryOptions(
                        limit=2,
                        delay=100,
                        backoff="constant",
                    )
                ),
                flaky,
            )
        except Exception as error:
            swallowed_errors.append(error)
            return -1

    tasks, lifecycle, ctx, retained, _ = await _retained_started(
        {"report": report},
        events=events,
    )
    monkeypatch.setattr(tasks_module, "now_ms", lambda: clock[0])

    receipt = await tasks.run("report", {"value": 7}, run_id="replay")
    assert receipt.accepted is True
    await retained.drain()

    waiting = await tasks.get("replay")
    assert waiting is not None
    assert waiting.state == "waiting"
    assert waiting.wake_at == 1_100
    retry_job = await tasks.lifecycle.jobs.get("task:replay")
    assert retry_job is not None
    assert retry_job.fn == "wake"
    assert retry_job.time == 1_100
    assert retry_job.payload == {"runId": "replay"}
    assert retry_job.retry == {"maxAttempts": 1}
    assert ctx.storage.sql.exec(
        "SELECT status_message FROM cf_agents_task_runs WHERE run_id = 'replay'"
    ).toArray() == [{"status_message": "stable=[7]"}]
    assert ctx.storage.sql.exec(
        """
        SELECT state, error_name, error_message, completed_at
        FROM cf_agents_task_steps
        WHERE run_id = 'replay' AND step_name = 'flaky'
        """
    ).toArray() == [
        {
            "state": "waiting",
            "error_name": None,
            "error_message": None,
            "completed_at": None,
        }
    ]

    clock[0] = 1_100
    joined = await tasks.run("report", {"value": 999}, run_id="replay")
    assert joined.accepted is False
    assert joined.state == "waiting"
    await retained.drain()

    completed = await tasks.get("replay")
    assert completed is not None
    assert completed.state == "completed"
    assert completed.result == 14
    assert await tasks.lifecycle.jobs.get("task:replay") is None
    assert handler_calls == [({"value": 7}, None), ({"value": 7}, None)]
    assert stable_calls == ["called"]
    assert stable_results == [[7], [7]]
    assert flaky_attempts == [
        (1, "replay:flaky", False),
        (2, "replay:flaky", False),
    ]
    assert swallowed_errors == []
    assert ctx.storage.sql.exec(
        """
        SELECT step_name, state, attempt, result
        FROM cf_agents_task_steps
        WHERE run_id = 'replay'
        ORDER BY step_name
        """
    ).toArray() == [
        {"step_name": "flaky", "state": "completed", "attempt": 2, "result": "14"},
        {
            "step_name": "stable",
            "state": "completed",
            "attempt": 1,
            "result": "[7]",
        },
    ]
    assert [event.type for event in events] == [
        "task:accepted",
        "task:attempt:started",
        "task:step:started",
        "task:step:completed",
        "task:step:started",
        "task:waiting",
        "task:attempt:started",
        "task:step:retry",
        "task:step:started",
        "task:step:completed",
        "task:completed",
    ]


@pytest.mark.asyncio
async def test_interrupted_running_step_replays_with_durable_evidence(monkeypatch):
    clock = [2_000]
    events = []
    interruptions = []
    callback_attempts = []

    async def report(input, step):
        interruptions.append(step.interrupted)

        async def resume(attempt):
            callback_attempts.append(attempt.attempt)
            context = get_current_lifecycle_context()
            assert context is not None
            return "recovered"

        return await step.do("lost", resume)

    tasks, _, ctx = await _started({"report": report}, events=events)
    monkeypatch.setattr(tasks_module, "now_ms", lambda: clock[0])
    await tasks.run("report", run_id="interrupted")
    ctx.storage.sql.exec(
        """
        UPDATE cf_agents_task_runs
        SET state = 'running', attempt = 1, next_at = NULL, started_at = 1_000
        WHERE run_id = 'interrupted'
        """
    )
    ctx.storage.sql.exec(
        """
        INSERT INTO cf_agents_task_steps
          (run_id, step_name, kind, state, attempt, created_at, started_at, updated_at)
        VALUES ('interrupted', 'lost', 'do', 'running', 1, 1_000, 1_000, 1_000)
        """
    )

    await tasks._execute_run("interrupted")

    completed = await tasks.get("interrupted")
    assert completed is not None
    assert completed.state == "completed"
    assert completed.result == "recovered"
    assert interruptions == [tasks_module.TaskInterruptedStep(name="lost", attempt=1)]
    assert callback_attempts == [2]
    assert [event.type for event in events] == [
        "task:accepted",
        "task:attempt:interrupted",
        "task:attempt:started",
        "task:step:started",
        "task:step:completed",
        "task:completed",
    ]
    assert ctx.storage.sql.exec(
        "SELECT state, attempt FROM cf_agents_task_steps WHERE run_id = 'interrupted'"
    ).toArray() == [{"state": "completed", "attempt": 2}]


@pytest.mark.asyncio
async def test_non_retryable_step_failure_settles_once_and_notifies_on_error():
    callback_attempts = []
    observed = []

    async def on_error(error):
        context = get_current_lifecycle_context()
        assert context is not None
        observed.append(error)

    async def report(input, step):
        async def stop(attempt):
            callback_attempts.append(attempt.attempt)
            raise NonRetryableError("stop now")

        return await step.do("stop", stop)

    tasks, _, ctx, retained, _ = await _retained_started(
        {"report": report},
        on_error=on_error,
    )
    await tasks.run("report", run_id="failed")
    await retained.drain()

    failed = await tasks.get("failed")
    assert failed is not None
    assert failed.state == "failed"
    assert failed.error.name == "NonRetryableError"
    assert failed.error.message == "stop now"
    assert callback_attempts == [1]
    assert len(observed) == 1
    assert isinstance(observed[0], NonRetryableError)
    assert ctx.storage.sql.exec(
        """
        SELECT state, attempt, error_name, error_message
        FROM cf_agents_task_steps WHERE run_id = 'failed'
        """
    ).toArray() == [
        {
            "state": "failed",
            "attempt": 1,
            "error_name": "NonRetryableError",
            "error_message": "stop now",
        }
    ]


@pytest.mark.asyncio
async def test_step_result_serialization_failure_skips_retries():
    callback_attempts = []

    async def report(input, step):
        def invalid(attempt):
            callback_attempts.append(attempt.attempt)
            return object()

        return await step.do("invalid", invalid)

    tasks, _, ctx, retained, _ = await _retained_started({"report": report})
    await tasks.run("report", run_id="serialization")
    await retained.drain()

    failed = await tasks.get("serialization")
    assert failed is not None
    assert failed.state == "failed"
    assert failed.error.name == "TaskSerializationError"
    assert callback_attempts == [1]
    assert ctx.storage.sql.exec(
        "SELECT state, attempt, error_name, completed_at FROM cf_agents_task_steps"
    ).toArray() == [
        {
            "state": "failed",
            "attempt": 1,
            "error_name": "TaskSerializationError",
            "completed_at": None,
        }
    ]
    assert not tasks._task_store().complete_step(
        "serialization",
        "stale-generation",
        "invalid",
        1,
        '"late"',
        999,
    )
    assert not tasks._task_store().settle_completed(
        "serialization",
        "stale-generation",
        '"late"',
        999,
    )


@pytest.mark.asyncio
async def test_final_result_serialization_failure_is_persisted():
    tasks, _, _, retained, _ = await _retained_started(
        {"report": lambda input, step: object()}
    )
    await tasks.run("report", run_id="invalid-result")
    await retained.drain()

    failed = await tasks.get("invalid-result")
    assert failed is not None
    assert failed.state == "failed"
    assert failed.error.name == "TaskSerializationError"


@pytest.mark.asyncio
async def test_duplicate_step_name_fails_before_second_callback_runs():
    callback_results = []

    async def report(input, step):
        callback_results.append(await step.do("same", lambda attempt: 1))
        return await step.do("same", lambda attempt: 2)

    tasks, _, ctx, retained, _ = await _retained_started({"report": report})
    await tasks.run("report", run_id="duplicate")
    await retained.drain()

    failed = await tasks.get("duplicate")
    assert failed is not None
    assert failed.state == "failed"
    assert failed.error.name == "DuplicateTaskStepError"
    assert callback_results == [1]
    assert ctx.storage.sql.exec(
        "SELECT state, attempt, result FROM cf_agents_task_steps"
    ).toArray() == [{"state": "completed", "attempt": 1, "result": "1"}]


@pytest.mark.asyncio
async def test_step_timeout_aborts_the_signal_and_settles_the_run():
    aborts = []

    async def report(input, step):
        async def slow(attempt):
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                aborts.append((attempt.signal.aborted, attempt.signal.reason))
                await asyncio.sleep(0)
                return "late"

        return await step.do("slow", slow)

    tasks, _, _, retained, _ = await _retained_started(
        {"report": report},
        retries=TaskStepRetryOptions(limit=1),
        step_timeout=0,
    )
    await tasks.run("report", run_id="timeout")
    await retained.drain()

    failed = await tasks.get("timeout")
    assert failed is not None
    assert failed.state == "failed"
    assert failed.error.name == "TimeoutError"
    assert len(aborts) == 1
    assert aborts[0][0] is True
    assert isinstance(aborts[0][1], TimeoutError)


@pytest.mark.parametrize(
    ("backoff", "failed_attempt", "expected"),
    [
        ("constant", 4, 1_000),
        ("linear", 4, 4_000),
        ("exponential", 4, 8_000),
        ("exponential", 10_000, 86_400_000),
    ],
)
def test_retry_backoff_and_cap(backoff, failed_attempt, expected):
    policy = tasks_module._default_step_policy(
        TaskStepRetryOptions(delay="1 second", backoff=backoff),
        "5 minutes",
    )

    assert tasks_module._retry_delay(policy, failed_attempt) == expected
    assert policy.timeout_ms == 300_000


@pytest.mark.parametrize("value", [True, 0, -1])
def test_invalid_retry_limits_are_rejected_when_a_step_resolves_policy(value):
    tasks = Tasks(retries=TaskStepRetryOptions(limit=cast(int, value)))

    with pytest.raises(ValueError, match="integer >= 1"):
        tasks_module._default_step_policy(tasks._retries, tasks._step_timeout)


@pytest.mark.asyncio
async def test_platform_failure_leaves_the_run_and_step_replayable(monkeypatch):
    observed = []
    attempts = []

    async def report(input, step):
        def reset(attempt):
            attempts.append(attempt.attempt)
            if attempt.attempt == 1:
                raise RuntimeError("network connection lost")
            return "recovered"

        return await step.do("reset", reset)

    tasks, lifecycle, ctx, retained, _ = await _retained_started(
        {"report": report},
        on_error=observed.append,
    )
    clock = [1_000]
    monkeypatch.setattr(tasks_module, "now_ms", lambda: clock[0])
    await tasks.run("report", run_id="platform")

    with pytest.raises(RuntimeError, match="network connection lost"):
        await retained.drain()

    running = await tasks.get("platform")
    assert running is not None
    assert running.state == "running"
    assert observed == []
    assert ctx.storage.sql.exec(
        "SELECT state, attempt FROM cf_agents_task_steps WHERE run_id = 'platform'"
    ).toArray() == [{"state": "running", "attempt": 1}]

    deadline = tasks._task_store().authoritative_deadline("platform")
    assert deadline is not None
    clock[0] = deadline
    await lifecycle.alarm()

    completed = await tasks.get("platform")
    assert completed is not None
    assert completed.state == "completed"
    assert completed.result == "recovered"
    assert attempts == [1, 2]


@pytest.mark.asyncio
async def test_rejected_warm_submission_does_not_reverse_durable_acceptance():
    def reject(awaitable):
        raise RuntimeError("waitUntil rejected")

    ctx = fakes.FakeCtx()
    tasks = Tasks({"report": lambda input, step: "unused"})
    lifecycle = Lifecycle(ctx, host=object(), retain_work=reject)
    lifecycle.use(tasks)
    await lifecycle.start()

    receipt = await tasks.run("report", run_id="durable")

    assert receipt.accepted is True
    pending = await tasks.get("durable")
    assert pending is not None
    assert pending.state == "pending"


@pytest.mark.asyncio
async def test_no_retained_work_executes_from_the_durable_alarm():
    calls = []
    tasks, lifecycle, ctx = await _started(
        {"report": lambda input, step: calls.append("ran") or "done"}
    )

    await tasks.run("report", run_id="alarm-only", retain=False)
    await lifecycle.alarm()

    assert calls == ["ran"]
    assert await tasks.get("alarm-only") is None
    assert (
        ctx.storage.sql.exec(
            "SELECT id FROM cf_agents_jobs WHERE capability = 'tasks'"
        ).toArray()
        == []
    )


@pytest.mark.asyncio
async def test_task_json_rejects_integers_that_typescript_cannot_round_trip():
    tasks, _, _ = await _started({"report": lambda input, step: input})

    await tasks.run("report", 2**53 - 1, run_id="safe")
    with pytest.raises(TaskSerializationError, match="safe-integer"):
        await tasks.run("report", {"nested": [2**53]}, run_id="unsafe")


@pytest.mark.asyncio
async def test_sleep_reuses_its_first_deadline_and_alarm_completes_the_run(monkeypatch):
    clock = [10_000]
    events = []
    durations = ["1 second", "1 week"]

    async def report(input, step):
        await step.sleep("pause", durations.pop(0))
        return "awake"

    tasks, lifecycle, ctx, retained, _ = await _retained_started(
        {"report": report},
        events=events,
    )
    monkeypatch.setattr(tasks_module, "now_ms", lambda: clock[0])

    await tasks.run("report", run_id="sleeper")
    await retained.drain()

    waiting = await tasks.get("sleeper")
    assert waiting is not None
    assert waiting.state == "waiting"
    assert waiting.reason == "sleep"
    assert waiting.wake_at == 11_000
    sleep_job = await tasks.lifecycle.jobs.get("task:sleeper")
    assert sleep_job is not None
    assert sleep_job.time == 11_000
    assert ctx.storage.sql.exec(
        """
        SELECT kind, state, attempt, next_at, started_at, completed_at
        FROM cf_agents_task_steps WHERE run_id = 'sleeper'
        """
    ).toArray() == [
        {
            "kind": "sleep",
            "state": "waiting",
            "attempt": 0,
            "next_at": 11_000,
            "started_at": None,
            "completed_at": None,
        }
    ]

    clock[0] = 11_000
    await lifecycle.alarm()

    completed = await tasks.get("sleeper")
    assert completed is not None
    assert completed.state == "completed"
    assert completed.result == "awake"
    assert durations == []
    assert await tasks.lifecycle.jobs.get("task:sleeper") is None
    assert ctx.storage.sql.exec(
        """
        SELECT state, attempt, next_at, completed_at
        FROM cf_agents_task_steps WHERE run_id = 'sleeper'
        """
    ).toArray() == [
        {
            "state": "completed",
            "attempt": 0,
            "next_at": None,
            "completed_at": 11_000,
        }
    ]
    assert [event.type for event in events] == [
        "task:accepted",
        "task:attempt:started",
        "task:waiting",
        "task:attempt:started",
        "task:completed",
    ]


@pytest.mark.asyncio
async def test_elapsed_sleeps_are_born_completed_without_step_events(monkeypatch):
    clock = [10_000]
    events = []

    async def report(input, step):
        await step.sleep("zero", 0)
        await step.sleep_until("past", datetime.fromtimestamp(1, tz=UTC))
        return "done"

    tasks, lifecycle, ctx, retained, _ = await _retained_started(
        {"report": report},
        events=events,
    )
    monkeypatch.setattr(tasks_module, "now_ms", lambda: clock[0])

    await tasks.run("report", run_id="elapsed")
    await retained.drain()

    completed = await tasks.get("elapsed")
    assert completed is not None
    assert completed.state == "completed"
    assert await tasks.lifecycle.jobs.get("task:elapsed") is None
    assert ctx.storage.sql.exec(
        """
        SELECT step_name, kind, state, attempt, next_at, completed_at
        FROM cf_agents_task_steps ORDER BY step_name
        """
    ).toArray() == [
        {
            "step_name": "past",
            "kind": "sleep",
            "state": "completed",
            "attempt": 0,
            "next_at": None,
            "completed_at": 10_000,
        },
        {
            "step_name": "zero",
            "kind": "sleep",
            "state": "completed",
            "attempt": 0,
            "next_at": None,
            "completed_at": 10_000,
        },
    ]
    assert [event.type for event in events] == [
        "task:accepted",
        "task:attempt:started",
        "task:completed",
    ]


@pytest.mark.asyncio
async def test_long_replay_refreshes_the_claim_mirror_before_new_work(monkeypatch):
    clock = [1_000]
    claim_deadlines = []
    lifecycle = None

    async def report(input, step):
        async def first(attempt):
            job = await tasks.lifecycle.jobs.get("task:refresh")
            claim_deadlines.append(job.time)
            clock[0] = 16_000
            return 1

        async def second(attempt):
            job = await tasks.lifecycle.jobs.get("task:refresh")
            claim_deadlines.append(job.time)
            return 2

        await step.do("first", first)
        return await step.do("second", second)

    tasks, lifecycle, _, retained, _ = await _retained_started({"report": report})
    monkeypatch.setattr(tasks_module, "now_ms", lambda: clock[0])

    await tasks.run("report", run_id="refresh")
    await retained.drain()

    assert claim_deadlines == [331_000, 346_000]
    completed = await tasks.get("refresh")
    assert completed is not None
    assert completed.state == "completed"


@pytest.mark.asyncio
async def test_startup_repairs_an_unarmed_exact_wake_without_duplicate_rows(
    monkeypatch,
):
    clock = [2_000_000_000_000]
    runtime = fakes.FakeDurableObjectRuntime()
    first_ctx = runtime.new_context()
    first_tasks = Tasks({"report": lambda input, step: "done"})
    first_lifecycle = Lifecycle(first_ctx, host=object())
    first_lifecycle.use(first_tasks)
    monkeypatch.setattr(tasks_module, "now_ms", lambda: clock[0])

    await first_tasks.run("report", run_id="cold")
    assert await first_ctx.storage.getAlarm() == clock[0]
    await first_ctx.storage.deleteAlarm()
    assert await first_ctx.storage.getAlarm() is None

    second_ctx = runtime.evict()
    second_tasks = Tasks({"report": lambda input, step: "done"})
    second_lifecycle = Lifecycle(second_ctx, host=object())
    second_lifecycle.use(second_tasks)
    await second_lifecycle.start()

    assert await second_ctx.storage.getAlarm() == clock[0]
    assert second_ctx.storage.sql.exec(
        "SELECT id, time FROM cf_agents_jobs WHERE capability = 'tasks'"
    ).toArray() == [{"id": "task:cold", "time": clock[0]}]


@pytest.mark.asyncio
async def test_startup_repairs_interrupted_and_missing_deadlines(monkeypatch):
    clock = [70_000]
    runtime = fakes.FakeDurableObjectRuntime()
    first_ctx = runtime.new_context()
    first_tasks = Tasks({})
    first_lifecycle = Lifecycle(first_ctx, host=object())
    first_lifecycle.use(first_tasks)
    await first_lifecycle.start()
    _insert_run(first_ctx, "pending", "pending", next_at=None)
    _insert_run(first_ctx, "running", "running", attempt=1, next_at=999_999)
    monkeypatch.setattr(tasks_module, "now_ms", lambda: clock[0])

    second_ctx = runtime.evict()
    second_tasks = Tasks({})
    second_lifecycle = Lifecycle(second_ctx, host=object())
    second_lifecycle.use(second_tasks)
    await second_lifecycle.start()

    assert second_ctx.storage.sql.exec(
        "SELECT run_id, next_at FROM cf_agents_task_runs ORDER BY run_id"
    ).toArray() == [
        {"run_id": "pending", "next_at": 70_000},
        {"run_id": "running", "next_at": 70_000},
    ]
    assert second_ctx.storage.sql.exec(
        """
        SELECT id, time FROM cf_agents_jobs
        WHERE capability = 'tasks' ORDER BY id
        """
    ).toArray() == [
        {"id": "task:pending", "time": 70_000},
        {"id": "task:running", "time": 70_000},
    ]


@pytest.mark.asyncio
async def test_joined_run_repairs_its_missing_wake(monkeypatch):
    clock = [80_000]
    tasks, _, _ = await _started({"report": lambda input, step: "done"})
    monkeypatch.setattr(tasks_module, "now_ms", lambda: clock[0])
    await tasks.run("report", run_id="joined")
    assert await tasks.lifecycle.jobs.cancel("task:joined") is True

    receipt = await tasks.run("report", run_id="joined")

    assert receipt.accepted is False
    repaired = await tasks.lifecycle.jobs.get("task:joined")
    assert repaired is not None
    assert repaired.time == 80_000


@pytest.mark.asyncio
async def test_non_alarm_owner_startup_does_not_open_the_root_job_queue():
    ctx = fakes.FakeCtx()
    retained_calls = 0

    def reject_retained_work(awaitable):
        nonlocal retained_calls
        retained_calls += 1
        raise RuntimeError("retention rejected")

    tasks = Tasks({})
    lifecycle = Lifecycle(
        ctx,
        host=object(),
        retain_work=reject_retained_work,
        owns_physical_alarm=False,
    )
    lifecycle.use(tasks)

    await lifecycle.start()

    assert retained_calls == 0
    assert (
        ctx.storage.sql.exec(
            "SELECT name FROM sqlite_master WHERE name = 'cf_agents_jobs'"
        ).toArray()
        == []
    )


@pytest.mark.asyncio
async def test_facet_reconciliation_rereads_each_authoritative_deadline():
    entered = asyncio.Event()
    release = asyncio.Event()
    payloads = []

    async def transport(envelope: LifecycleRouteEnvelope):
        payloads.append(envelope.payload)
        if len(payloads) == 1:
            entered.set()
            await release.wait()
        return True

    ctx = fakes.FakeCtx()
    tasks = Tasks({})
    lifecycle = Lifecycle(
        ctx,
        host=object(),
        route_address=LifecycleRouteAddress("root/facet", "[]"),
        root_route_address=LifecycleRouteAddress("root", "[]"),
        route_transport=transport,
        owns_physical_alarm=False,
    )
    lifecycle.use(tasks)
    await lifecycle.start()
    store = tasks._task_store()
    for run_id in ("first", "second"):
        store.insert_pending(
            run_id=run_id,
            definition="report",
            input_json=None,
            metadata_json=None,
            idempotency_key=None,
            retain=True,
            created_at=10_000,
        )
    tasks._reconciliation_pending = True

    flush = asyncio.create_task(tasks._flush_reconciliation())
    await entered.wait()
    ctx.storage.sql.exec(
        """
        UPDATE cf_agents_task_runs
        SET state = 'completed', next_at = NULL, settled_at = 10001
        WHERE run_id = 'second'
        """
    )
    release.set()
    await flush

    assert payloads == [
        {"type": "repair", "runId": "first", "nextAt": 10_000},
        {"type": "repair", "runId": "second", "nextAt": None},
    ]


@pytest.mark.asyncio
async def test_root_alarm_routes_one_wake_to_a_facet_without_reentry(monkeypatch):
    clock = [20_000]
    root_address = LifecycleRouteAddress("root", '[{"name":"root"}]')
    facet_address = LifecycleRouteAddress(
        "root/facet",
        '[{"name":"root"},{"name":"facet"}]',
    )
    lifecycles = {}
    envelopes = []

    async def transport(envelope: LifecycleRouteEnvelope):
        envelopes.append(envelope)
        lifecycle = lifecycles[envelope.target.key]
        return await lifecycle.route(
            version=envelope.version,
            source=envelope.source,
            target=envelope.target,
            capability_id=envelope.capability_id,
            payload=envelope.payload,
        )

    async def report(input, step):
        await step.sleep("pause", "1 second")
        return "facet awake"

    root_ctx = fakes.FakeCtx()
    root_tasks = Tasks({"root": lambda input, step: "root awake"})
    root_lifecycle = Lifecycle(
        root_ctx,
        host=object(),
        root_route_address=root_address,
        route_transport=transport,
        owns_physical_alarm=True,
    )
    root_lifecycle.use(root_tasks)
    facet_runtime = fakes.FakeDurableObjectRuntime()
    facet_ctx = facet_runtime.new_context()
    retained = fakes.WaitUntilRecorder()
    facet_tasks = Tasks({"report": report})
    facet_lifecycle = Lifecycle(
        facet_ctx,
        host=object(),
        retain_work=retained,
        route_address=facet_address,
        root_route_address=root_address,
        route_transport=transport,
        owns_physical_alarm=False,
    )
    facet_lifecycle.use(facet_tasks)
    lifecycles[root_address.key] = root_lifecycle
    lifecycles[facet_address.key] = facet_lifecycle
    monkeypatch.setattr(tasks_module, "now_ms", lambda: clock[0])

    await root_tasks.run("root", run_id="shared")
    await facet_tasks.run("report", run_id="shared")
    await retained.drain()

    routed_job_id = "task:root/facet:shared"
    routed_job = await root_tasks.lifecycle.jobs.get(routed_job_id)
    assert routed_job is not None
    assert routed_job.time == 20_000
    assert await root_tasks.lifecycle.jobs.get("task:shared") is not None

    await root_lifecycle.alarm()

    routed_job = await root_tasks.lifecycle.jobs.get(routed_job_id)
    assert routed_job is not None
    assert routed_job.time == 21_000
    assert routed_job.payload == {
        "runId": "shared",
        "owner_path": facet_address.data,
        "owner_path_key": facet_address.key,
    }
    assert await root_tasks.lifecycle.jobs.get("task:shared") is None
    assert (
        facet_ctx.storage.sql.exec(
            "SELECT name FROM sqlite_master WHERE name = 'cf_agents_jobs'"
        ).toArray()
        == []
    )

    assert await root_tasks.lifecycle.jobs.cancel(routed_job_id) is True
    cold_facet_ctx = facet_runtime.evict()
    cold_retained = fakes.WaitUntilRecorder()
    cold_facet_tasks = Tasks({"report": report})
    cold_facet_lifecycle = Lifecycle(
        cold_facet_ctx,
        host=object(),
        retain_work=cold_retained,
        route_address=facet_address,
        root_route_address=root_address,
        route_transport=transport,
        owns_physical_alarm=False,
    )
    cold_facet_lifecycle.use(cold_facet_tasks)
    lifecycles[facet_address.key] = cold_facet_lifecycle
    await cold_facet_lifecycle.start()
    assert await root_tasks.lifecycle.jobs.get(routed_job_id) is None

    await cold_retained.drain()
    repaired_job = await root_tasks.lifecycle.jobs.get(routed_job_id)
    assert repaired_job is not None
    assert repaired_job.time == 21_000

    routes_before_alarm = len(envelopes)
    clock[0] = 21_000
    await root_lifecycle.alarm()

    assert len(envelopes) == routes_before_alarm + 1
    completed = await cold_facet_tasks.get("shared")
    assert completed is not None
    assert completed.state == "completed"
    assert completed.result == "facet awake"
    assert await root_tasks.lifecycle.jobs.get(routed_job_id) is None
    root_completed = await root_tasks.get("shared")
    assert root_completed is not None
    assert root_completed.state == "completed"
    assert root_completed.result == "root awake"


@pytest.mark.asyncio
async def test_root_and_facet_wake_id_collisions_reject_the_later_run():
    root_address = LifecycleRouteAddress("root", '[{"name":"root"}]')
    lifecycles = {}

    async def transport(envelope: LifecycleRouteEnvelope):
        lifecycle = lifecycles[envelope.target.key]
        return await lifecycle.route(
            version=envelope.version,
            source=envelope.source,
            target=envelope.target,
            capability_id=envelope.capability_id,
            payload=envelope.payload,
        )

    root_tasks = Tasks({"report": lambda input, step: "root"})
    root_lifecycle = Lifecycle(
        fakes.FakeCtx(),
        host=object(),
        root_route_address=root_address,
        route_transport=transport,
        owns_physical_alarm=True,
    )
    root_lifecycle.use(root_tasks)
    lifecycles[root_address.key] = root_lifecycle

    def facet(owner_key):
        address = LifecycleRouteAddress(
            owner_key,
            f'{{"name":"{owner_key}"}}',
        )
        tasks = Tasks({"report": lambda input, step: "facet"})
        lifecycle = Lifecycle(
            fakes.FakeCtx(),
            host=object(),
            route_address=address,
            root_route_address=root_address,
            route_transport=transport,
            owns_physical_alarm=False,
        )
        lifecycle.use(tasks)
        lifecycles[address.key] = lifecycle
        return tasks

    first_facet = facet("first")
    root_receipt = await root_tasks.run("report", run_id="first:shared")
    with pytest.raises(TaskWakeCollisionError, match="task:first:shared"):
        await first_facet.run("report", run_id="shared")
    assert await root_tasks.get(root_receipt.run_id) is not None
    assert await first_facet.get("shared") is None

    second_facet = facet("second")
    facet_receipt = await second_facet.run("report", run_id="shared")
    with pytest.raises(TaskWakeCollisionError, match="task:second:shared"):
        await root_tasks.run("report", run_id="second:shared")
    assert await second_facet.get(facet_receipt.run_id) is not None
    assert await root_tasks.get("second:shared") is None


@pytest.mark.asyncio
async def test_stale_generation_cannot_mutate_run_or_step():
    tasks, _, ctx = await _started({"report": lambda input, step: None})
    _insert_run(
        ctx,
        "fenced",
        "running",
        attempt=1,
        generation="current",
        next_at=500,
        started_at=100,
    )
    ctx.storage.sql.exec(
        """
        INSERT INTO cf_agents_task_steps
          (run_id, step_name, kind, state, attempt, next_at,
           created_at, started_at, updated_at)
        VALUES
          ('fenced', 'work', 'do', 'running', 1, NULL, 100, 100, 100),
          ('fenced', 'nap', 'sleep', 'waiting', 0, 500, 100, NULL, 100)
        """
    )
    store = tasks._task_store()

    assert not store.refresh_claim("fenced", "stale", 600, 200)
    assert not store.extend_active_claim("fenced", "stale", 600, 200)
    assert not store.insert_step("fenced", "stale", "late", 200)
    assert not store.insert_completed_sleep("fenced", "stale", "elapsed", 200)
    assert not store.park_new_sleep("fenced", "stale", "later", 600, 200)
    assert not store.park_existing_sleep("fenced", "stale", 600, 200)
    assert not store.complete_sleep("fenced", "stale", "nap", 200)
    assert store.restart_step("fenced", "stale", "work", 200) is None
    assert not store.complete_step("fenced", "stale", "work", 1, "1", 200)
    assert not store.fail_step(
        "fenced",
        "stale",
        "work",
        1,
        "RuntimeError",
        "late",
        200,
    )
    assert not store.park_step_retry("fenced", "stale", "work", 1, 600, 200)
    assert not store.update_status("fenced", "stale", "late", 200)
    assert not store.settle_completed("fenced", "stale", "1", 200)
    assert not store.settle_failed(
        "fenced",
        "stale",
        "RuntimeError",
        "late",
        200,
    )
    assert not store.request_cancel("fenced", "stale", "late", 200)
    assert not store.settle_cancelled_claim("fenced", "stale", "late", 200)
    assert not store.backoff_memory_claim("fenced", "stale", 600, 200)
    assert not store.settle_failed_unclaimed(
        "fenced",
        "RuntimeError",
        "late",
        200,
    )

    assert ctx.storage.sql.exec(
        """
        SELECT state, attempt, generation, next_at, status_message,
               cancel_requested, settled_at, updated_at
        FROM cf_agents_task_runs WHERE run_id = 'fenced'
        """
    ).toArray() == [
        {
            "state": "running",
            "attempt": 1,
            "generation": "current",
            "next_at": 500,
            "status_message": None,
            "cancel_requested": 0,
            "settled_at": None,
            "updated_at": 200,
        }
    ]
    assert ctx.storage.sql.exec(
        """
        SELECT step_name, state, attempt, next_at, result, error_name
        FROM cf_agents_task_steps ORDER BY step_name
        """
    ).toArray() == [
        {
            "step_name": "nap",
            "state": "waiting",
            "attempt": 0,
            "next_at": 500,
            "result": None,
            "error_name": None,
        },
        {
            "step_name": "work",
            "state": "running",
            "attempt": 1,
            "next_at": None,
            "result": None,
            "error_name": None,
        },
    ]


@pytest.mark.asyncio
async def test_cold_claimed_rows_settle_cancellation_and_missing_definition():
    events = []
    tasks, lifecycle, ctx = await _started(
        {"report": lambda input, step: "done"},
        events=events,
    )
    _insert_run(
        ctx,
        "cancelled-claim",
        "running",
        generation="old-cancel",
        next_at=100,
        cancel_requested=1,
        cancel_reason="stop",
    )
    _insert_run(
        ctx,
        "missing-claim",
        "running",
        definition="removed",
        generation="old-missing",
        next_at=100,
    )
    _insert_run(
        ctx,
        "invalid-policy",
        "running",
        generation="old-policy",
        next_at=100,
    )
    tasks._step_timeout = -1
    for run_id in ("cancelled-claim", "missing-claim", "invalid-policy"):
        await tasks.lifecycle.jobs.push(
            LifecycleJobPushOptions(
                id=f"task:{run_id}",
                fn="wake",
                time=100,
                payload={"runId": run_id},
                retry={"maxAttempts": 1},
            )
        )

    await lifecycle.alarm()

    cancelled = await tasks.get("cancelled-claim")
    assert cancelled is not None
    assert cancelled.state == "cancelled"
    failed = await tasks.get("missing-claim")
    assert failed is not None
    assert failed.state == "failed"
    assert failed.error.name == "MissingTaskDefinitionError"
    invalid = await tasks.get("invalid-policy")
    assert invalid is not None
    assert invalid.state == "failed"
    assert invalid.error.name == "ValueError"
    assert [
        event.type
        for event in events
        if event.type in {"task:cancelled", "task:failed"}
    ] == [
        "task:cancelled",
        "task:failed",
        "task:failed",
    ]
    assert (
        ctx.storage.sql.exec(
            "SELECT id FROM cf_agents_jobs WHERE capability = 'tasks'"
        ).toArray()
        == []
    )


@pytest.mark.asyncio
async def test_duplicate_alarm_extends_one_canonical_active_attempt(monkeypatch):
    clock = [1_000]
    entered = asyncio.Event()
    release = asyncio.Event()

    async def report(input, step):
        entered.set()
        await release.wait()
        return "done"

    ctx = fakes.FakeCtx()
    retained = fakes.WaitUntilRecorder()
    tasks = Tasks({"report": report})
    lifecycle = Lifecycle(ctx, host=object(), retain_work=retained)
    lifecycle.use(tasks)
    await lifecycle.start()
    tasks._task_store().insert_pending(
        run_id="long",
        definition="report",
        input_json=None,
        metadata_json=None,
        idempotency_key=None,
        retain=True,
        created_at=clock[0],
    )
    await tasks._sync_wake("long")
    monkeypatch.setattr(tasks_module, "now_ms", lambda: clock[0])
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: clock[0])
    monkeypatch.setattr(tasks_module, "_DISPATCH_BUDGET_SECONDS", 0)

    await lifecycle.alarm()
    await entered.wait()
    first_active = tasks._active_runs["long"]
    first_row = tasks._task_store().get("long")
    assert first_row is not None
    assert first_row.attempt == 1
    assert first_row.next_at == 331_000
    assert len(retained.coros) == 1

    clock[0] = 331_000
    await lifecycle.alarm()

    extended = tasks._task_store().get("long")
    assert extended is not None
    assert extended.attempt == 1
    assert extended.generation == first_row.generation
    assert extended.next_at == 661_000
    assert tasks._active_runs["long"].task is first_active.task
    assert len(retained.coros) == 1

    release.set()
    await retained.drain()
    completed = await tasks.get("long")
    assert completed is not None
    assert completed.state == "completed"
    assert await tasks.lifecycle.jobs.get("task:long") is None


@pytest.mark.asyncio
async def test_step_timeout_override_extends_the_claim_before_user_code(monkeypatch):
    clock = [1_000]
    deadlines = []

    async def report(input, step):
        async def work(attempt):
            job = await tasks.lifecycle.jobs.get("task:long-timeout")
            deadlines.append(job.time)
            return "done"

        return await step.do(
            "work",
            TaskStepConfig(timeout=100_000),
            work,
        )

    tasks, _, _, retained, _ = await _retained_started(
        {"report": report},
        step_timeout=1_000,
    )
    monkeypatch.setattr(tasks_module, "now_ms", lambda: clock[0])

    await tasks.run("report", run_id="long-timeout")
    await retained.drain()

    assert deadlines == [131_000]


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_event", ["task:attempt:started", "task:step:started"])
async def test_cancellation_from_start_event_prevents_user_code(cancel_event):
    definition_calls = []
    callback_calls = []
    tasks = None

    async def listener(event):
        if event.type == cancel_event:
            assert tasks is not None
            assert await tasks.cancel("event-cancel", "listener") is True

    async def report(input, step):
        definition_calls.append("called")

        async def work(attempt):
            callback_calls.append("called")
            return "done"

        return await step.do("work", work)

    ctx = fakes.FakeCtx()
    retained = fakes.WaitUntilRecorder()
    tasks = Tasks({"report": report})
    lifecycle = Lifecycle(
        ctx,
        host=object(),
        retain_work=retained,
        event_listeners=(listener,),
    )
    lifecycle.use(tasks)
    await lifecycle.start()

    await tasks.run("report", run_id="event-cancel")
    await retained.drain()

    cancelled = await tasks.get("event-cancel")
    assert cancelled is not None
    assert cancelled.state == "cancelled"
    assert callback_calls == []
    assert definition_calls == (
        [] if cancel_event == "task:attempt:started" else ["called"]
    )


@pytest.mark.asyncio
async def test_active_cancellation_wins_before_the_next_step():
    entered = asyncio.Event()
    release = asyncio.Event()
    events = []

    async def report(input, step):
        entered.set()
        await release.wait()
        await step.status("too late")
        return "too late"

    tasks, _, _, retained, _ = await _retained_started(
        {"report": report},
        events=events,
    )
    await tasks.run("report", run_id="cancel-before-step")
    draining = asyncio.create_task(retained.drain_next())
    await entered.wait()

    assert await tasks.cancel("cancel-before-step", "stop") is True
    running = tasks._task_store().get("cancel-before-step")
    assert running is not None
    assert running.state == "running"
    assert running.cancel_requested == 1
    assert running.generation is not None

    release.set()
    await draining
    cancelled = await tasks.get("cancel-before-step")
    assert cancelled is not None
    assert cancelled.state == "cancelled"
    assert cancelled.reason == "stop"
    assert [event.type for event in events].count("task:cancelled") == 1
    assert all(event.type != "task:completed" for event in events)


@pytest.mark.asyncio
async def test_active_cancellation_aborts_the_running_step():
    entered = asyncio.Event()
    observed = []

    async def report(input, step):
        async def blocked(attempt):
            entered.set()
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                observed.append(
                    (
                        attempt.signal.aborted,
                        getattr(attempt.signal.reason, "reason", None),
                    )
                )
                raise

        return await step.do("blocked", blocked)

    tasks, _, _, retained, _ = await _retained_started({"report": report})
    await tasks.run("report", run_id="cancel-step")
    draining = asyncio.create_task(retained.drain_next())
    await entered.wait()

    assert await tasks.cancel("cancel-step", "operator") is True
    await draining
    await retained.drain()

    cancelled = await tasks.get("cancel-step")
    assert cancelled is not None
    assert cancelled.state == "cancelled"
    assert cancelled.reason == "operator"
    assert observed == [(True, "operator")]


@pytest.mark.asyncio
async def test_unretained_terminal_event_precedes_atomic_journal_cleanup():
    ctx = fakes.FakeCtx()
    retained = fakes.WaitUntilRecorder()
    visible_at_terminal = []

    async def listener(event):
        if event.type not in {"task:completed", "task:failed", "task:cancelled"}:
            return
        run_id = event.payload["runId"]
        rows = ctx.storage.sql.exec(
            "SELECT run_id FROM cf_agents_task_runs WHERE run_id = ?",
            run_id,
        ).toArray()
        visible_at_terminal.append((event.type, run_id, bool(rows)))

    async def fail(input, step):
        raise RuntimeError("failed")

    async def sleep(input, step):
        await step.sleep("pause", "1 hour")

    tasks = Tasks(
        {
            "complete": lambda input, step: "done",
            "fail": fail,
            "sleep": sleep,
        }
    )
    lifecycle = Lifecycle(
        ctx,
        host=object(),
        retain_work=retained,
        event_listeners=(listener,),
    )
    lifecycle.use(tasks)
    await lifecycle.start()

    await tasks.run("complete", run_id="complete", retain=False)
    await retained.drain()
    await tasks.run("fail", run_id="fail", retain=False)
    await retained.drain()
    await tasks.run("sleep", run_id="cancel", retain=False)
    await retained.drain()
    assert await tasks.cancel("cancel", "stop") is True

    assert visible_at_terminal == [
        ("task:completed", "complete", True),
        ("task:failed", "fail", True),
        ("task:cancelled", "cancel", True),
    ]
    assert (
        ctx.storage.sql.exec("SELECT run_id FROM cf_agents_task_runs").toArray() == []
    )
    assert (
        ctx.storage.sql.exec("SELECT run_id FROM cf_agents_task_steps").toArray() == []
    )
    assert (
        ctx.storage.sql.exec(
            "SELECT id FROM cf_agents_jobs WHERE capability = 'tasks'"
        ).toArray()
        == []
    )


@pytest.mark.asyncio
async def test_startup_cleans_an_interrupted_unretained_terminal_journal():
    runtime = fakes.FakeDurableObjectRuntime()
    first_ctx = runtime.new_context()
    first_tasks = Tasks({})
    first_lifecycle = Lifecycle(first_ctx, host=object())
    first_lifecycle.use(first_tasks)
    await first_lifecycle.start()
    _insert_run(
        first_ctx,
        "orphan",
        "completed",
        result='"done"',
        retain=0,
        settled_at=200,
    )
    first_ctx.storage.sql.exec(
        """
        INSERT INTO cf_agents_task_steps
          (run_id, step_name, kind, state, attempt, result,
           created_at, updated_at, completed_at)
        VALUES ('orphan', 'work', 'do', 'completed', 1, '1', 100, 200, 200)
        """
    )
    await first_tasks.lifecycle.jobs.push(
        LifecycleJobPushOptions(
            id="task:orphan",
            fn="wake",
            time=300,
            payload={"runId": "orphan"},
            retry={"maxAttempts": 1},
        )
    )
    events = []

    second_ctx = runtime.evict()
    second_tasks = Tasks({})
    second_lifecycle = Lifecycle(
        second_ctx,
        host=object(),
        event_listeners=(events.append,),
    )
    second_lifecycle.use(second_tasks)
    await second_lifecycle.start()

    assert [event.type for event in events] == ["task:completed"]
    assert (
        second_ctx.storage.sql.exec("SELECT run_id FROM cf_agents_task_runs").toArray()
        == []
    )
    assert (
        second_ctx.storage.sql.exec("SELECT run_id FROM cf_agents_task_steps").toArray()
        == []
    )
    assert await second_tasks.lifecycle.jobs.get("task:orphan") is None


@pytest.mark.asyncio
async def test_cold_facet_cleanup_cancels_its_root_mirror():
    root_address = LifecycleRouteAddress("root", '[{"name":"root"}]')
    facet_address = LifecycleRouteAddress("root/facet", '[{"name":"facet"}]')
    lifecycles = {}

    async def transport(envelope: LifecycleRouteEnvelope):
        lifecycle = lifecycles[envelope.target.key]
        return await lifecycle.route(
            version=envelope.version,
            source=envelope.source,
            target=envelope.target,
            capability_id=envelope.capability_id,
            payload=envelope.payload,
        )

    root_tasks = Tasks({})
    root_lifecycle = Lifecycle(
        fakes.FakeCtx(),
        host=object(),
        root_route_address=root_address,
        route_transport=transport,
        owns_physical_alarm=True,
    )
    root_lifecycle.use(root_tasks)
    facet_runtime = fakes.FakeDurableObjectRuntime()
    first_ctx = facet_runtime.new_context()
    first_tasks = Tasks({})
    first_lifecycle = Lifecycle(
        first_ctx,
        host=object(),
        route_address=facet_address,
        root_route_address=root_address,
        route_transport=transport,
        owns_physical_alarm=False,
    )
    first_lifecycle.use(first_tasks)
    lifecycles[root_address.key] = root_lifecycle
    lifecycles[facet_address.key] = first_lifecycle
    await root_lifecycle.start()
    await first_lifecycle.start()
    _insert_run(
        first_ctx,
        "orphan",
        "completed",
        result='"done"',
        retain=0,
        settled_at=200,
    )
    await root_tasks.lifecycle.jobs.push(
        LifecycleJobPushOptions(
            id="task:root/facet:orphan",
            fn="wake",
            time=300,
            payload={
                "runId": "orphan",
                "owner_path": facet_address.data,
                "owner_path_key": facet_address.key,
            },
            retry={"maxAttempts": 1},
        )
    )

    cold_ctx = facet_runtime.evict()
    cold_tasks = Tasks({})
    cold_lifecycle = Lifecycle(
        cold_ctx,
        host=object(),
        route_address=facet_address,
        root_route_address=root_address,
        route_transport=transport,
        owns_physical_alarm=False,
    )
    cold_lifecycle.use(cold_tasks)
    lifecycles[facet_address.key] = cold_lifecycle
    await cold_lifecycle.start()

    assert await root_tasks.lifecycle.jobs.get("task:root/facet:orphan") is not None
    assert cold_ctx.storage.sql.exec(
        "SELECT run_id FROM cf_agents_task_runs WHERE run_id = 'orphan'"
    ).toArray() == [{"run_id": "orphan"}]

    retry_ctx = facet_runtime.evict()
    retry_tasks = Tasks({})
    retry_lifecycle = Lifecycle(
        retry_ctx,
        host=object(),
        route_address=facet_address,
        root_route_address=root_address,
        route_transport=transport,
        owns_physical_alarm=False,
    )
    retry_lifecycle.use(retry_tasks)
    lifecycles[facet_address.key] = retry_lifecycle
    await retry_lifecycle.start()
    await root_lifecycle.alarm()
    assert await root_tasks.lifecycle.jobs.get("task:root/facet:orphan") is None
    assert (
        retry_ctx.storage.sql.exec(
            "SELECT run_id FROM cf_agents_task_runs WHERE run_id = 'orphan'"
        ).toArray()
        == []
    )


@pytest.mark.asyncio
async def test_failed_alarm_handoff_cancels_the_canonical_task(monkeypatch):
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def report(input, step):
        entered.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    def reject(awaitable):
        raise RuntimeError("waitUntil rejected")

    ctx = fakes.FakeCtx()
    tasks = Tasks({"report": report})
    lifecycle = Lifecycle(ctx, host=object(), retain_work=reject)
    lifecycle.use(tasks)
    await lifecycle.start()
    tasks._task_store().insert_pending(
        run_id="rejected-handoff",
        definition="report",
        input_json=None,
        metadata_json=None,
        idempotency_key=None,
        retain=True,
        created_at=1_000,
    )
    await tasks._sync_wake("rejected-handoff")
    monkeypatch.setattr(tasks_module, "_DISPATCH_BUDGET_SECONDS", 0)

    await lifecycle.alarm()

    assert entered.is_set()
    assert cancelled.is_set()
    assert "rejected-handoff" not in tasks._active_runs
    assert "rejected-handoff" not in tasks._execution_tasks
    running = await tasks.get("rejected-handoff")
    assert running is not None
    assert running.state == "running"


@pytest.mark.asyncio
async def test_memory_limit_backoff_then_seals_the_active_run(monkeypatch):
    clock = [1_000]
    events = []
    host_policies: list[LifecycleMemoryLimitContext] = []

    async def report(input, step):
        raise RuntimeError("Durable Object's isolate exceeded its memory limit")

    monkeypatch.setattr(tasks_module, "now_ms", lambda: clock[0])
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: clock[0])
    ctx = fakes.FakeCtx()
    tasks = Tasks({"report": report})
    lifecycle = Lifecycle(
        ctx,
        host=object(),
        event_listeners=(events.append,),
        on_alarm_memory_limit=host_policies.append,
        max_alarm_memory_limit_strikes=2,
        reset_alarm=lambda reason: None,
    )
    lifecycle.use(tasks)

    await tasks.run("report", run_id="memory")
    await lifecycle.alarm()

    backed_off = tasks._task_store().get("memory")
    assert backed_off is not None
    assert backed_off.state == "running"
    assert backed_off.attempt == 1
    assert backed_off.generation is None
    assert backed_off.wait_reason == "memory"
    assert backed_off.next_at == 31_000
    assert not host_policies[0].sealed

    clock[0] = 31_000
    await lifecycle.alarm()

    sealed = await tasks.get("memory")
    assert sealed is not None
    assert sealed.state == "failed"
    assert sealed.error.name == "TaskMemoryLimitSealed"
    assert sealed.settled_at == 31_000
    assert host_policies[1].sealed
    assert [event.type for event in events].count("task:failed") == 1
    assert await tasks.lifecycle.jobs.get("task:memory") is None


@pytest.mark.asyncio
async def test_tracked_memory_failure_is_attributed_after_alarm_handoff(monkeypatch):
    clock = [1_000]
    entered = asyncio.Event()
    release = asyncio.Event()

    async def report(input, step):
        entered.set()
        await release.wait()
        raise RuntimeError("isolate exceeded its memory limit")

    monkeypatch.setattr(tasks_module, "now_ms", lambda: clock[0])
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: clock[0])
    monkeypatch.setattr(tasks_module, "_DISPATCH_BUDGET_SECONDS", 0)
    ctx = fakes.FakeCtx()
    retained = fakes.WaitUntilRecorder()
    tasks = Tasks({"report": report})
    lifecycle = Lifecycle(
        ctx,
        host=object(),
        retain_work=retained,
        max_alarm_memory_limit_strikes=2,
        reset_alarm=lambda reason: None,
    )
    lifecycle.use(tasks)
    await lifecycle.start()
    tasks._task_store().insert_pending(
        run_id="tracked-memory",
        definition="report",
        input_json=None,
        metadata_json=None,
        idempotency_key=None,
        retain=True,
        created_at=clock[0],
    )
    await tasks._sync_wake("tracked-memory")

    await lifecycle.alarm()
    await entered.wait()
    assert len(retained.coros) == 1

    first = tasks._task_store().get("tracked-memory")
    assert first is not None
    clock[0] = first.next_at
    release.set()
    active_task = tasks._active_runs["tracked-memory"].task
    while not active_task.done():
        await asyncio.sleep(0)
    await lifecycle.alarm()

    await retained.drain()

    backed_off = tasks._task_store().get("tracked-memory")
    assert backed_off is not None
    assert backed_off.state == "running"
    assert backed_off.generation is None
    assert backed_off.wait_reason == "memory"
    assert backed_off.next_at == clock[0] + 30_000
    assert await tasks.lifecycle.storage.get("cf_agents:oom_alarm_strikes") == 1


@pytest.mark.asyncio
async def test_active_cancellation_wins_over_tracked_memory_sealing(monkeypatch):
    entered = asyncio.Event()
    release = asyncio.Event()

    async def report(input, step):
        entered.set()
        await release.wait()
        raise RuntimeError("isolate exceeded its memory limit")

    ctx = fakes.FakeCtx()
    retained = fakes.WaitUntilRecorder()
    tasks = Tasks({"report": report})
    lifecycle = Lifecycle(
        ctx,
        host=object(),
        retain_work=retained,
        max_alarm_memory_limit_strikes=1,
        reset_alarm=lambda reason: None,
    )
    lifecycle.use(tasks)
    await lifecycle.start()
    tasks._task_store().insert_pending(
        run_id="cancel-memory",
        definition="report",
        input_json=None,
        metadata_json=None,
        idempotency_key=None,
        retain=True,
        created_at=1_000,
    )
    await tasks._sync_wake("cancel-memory")
    monkeypatch.setattr(tasks_module, "_DISPATCH_BUDGET_SECONDS", 0)

    await lifecycle.alarm()
    await entered.wait()
    assert await tasks.cancel("cancel-memory", "operator") is True

    release.set()
    await retained.drain()

    cancelled = await tasks.get("cancel-memory")
    assert cancelled is not None
    assert cancelled.state == "cancelled"
    assert cancelled.reason == "operator"
    assert await tasks.lifecycle.jobs.get("task:cancel-memory") is None


@pytest.mark.asyncio
async def test_duplicate_routed_alarm_extends_the_facet_claim(monkeypatch):
    clock = [1_000]
    entered = asyncio.Event()
    release = asyncio.Event()
    root_address = LifecycleRouteAddress("root", '[{"name":"root"}]')
    facet_address = LifecycleRouteAddress(
        "root/facet",
        '[{"name":"root"},{"name":"facet"}]',
    )
    lifecycles = {}

    async def transport(envelope: LifecycleRouteEnvelope):
        lifecycle = lifecycles[envelope.target.key]
        return await lifecycle.route(
            version=envelope.version,
            source=envelope.source,
            target=envelope.target,
            capability_id=envelope.capability_id,
            payload=envelope.payload,
        )

    async def report(input, step):
        entered.set()
        await release.wait()
        return "done"

    root_ctx = fakes.FakeCtx()
    root_retained = fakes.WaitUntilRecorder()
    root_tasks = Tasks({})
    root_lifecycle = Lifecycle(
        root_ctx,
        host=object(),
        retain_work=root_retained,
        root_route_address=root_address,
        route_transport=transport,
        owns_physical_alarm=True,
    )
    root_lifecycle.use(root_tasks)
    facet_ctx = fakes.FakeCtx()
    facet_tasks = Tasks({"report": report})
    facet_lifecycle = Lifecycle(
        facet_ctx,
        host=object(),
        route_address=facet_address,
        root_route_address=root_address,
        route_transport=transport,
        owns_physical_alarm=False,
    )
    facet_lifecycle.use(facet_tasks)
    lifecycles[root_address.key] = root_lifecycle
    lifecycles[facet_address.key] = facet_lifecycle
    monkeypatch.setattr(tasks_module, "now_ms", lambda: clock[0])
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: clock[0])
    monkeypatch.setattr(tasks_module, "_DISPATCH_BUDGET_SECONDS", 0)

    await facet_tasks.run("report", run_id="routed-long")
    await root_lifecycle.alarm()
    await entered.wait()

    first = facet_tasks._task_store().get("routed-long")
    assert first is not None
    assert first.attempt == 1
    assert first.next_at == 331_000
    assert len(root_retained.coros) == 1

    clock[0] = 331_000
    await root_lifecycle.alarm()

    extended = facet_tasks._task_store().get("routed-long")
    assert extended is not None
    assert extended.attempt == 1
    assert extended.generation == first.generation
    assert extended.next_at == 661_000
    assert len(root_retained.coros) == 1

    release.set()
    await root_retained.drain()
    completed = await facet_tasks.get("routed-long")
    assert completed is not None
    assert completed.state == "completed"
    assert await root_tasks.lifecycle.jobs.get("task:root/facet:routed-long") is None


@pytest.mark.asyncio
async def test_completed_routed_handoff_keeps_its_memory_generation(monkeypatch):
    clock = [1_000]
    entered = asyncio.Event()
    release = asyncio.Event()
    root_address = LifecycleRouteAddress("root", '[{"name":"root"}]')
    facet_address = LifecycleRouteAddress("root/facet", '[{"name":"facet"}]')
    lifecycles = {}

    async def transport(envelope: LifecycleRouteEnvelope):
        lifecycle = lifecycles[envelope.target.key]
        return await lifecycle.route(
            version=envelope.version,
            source=envelope.source,
            target=envelope.target,
            capability_id=envelope.capability_id,
            payload=envelope.payload,
        )

    async def report(input, step):
        entered.set()
        await release.wait()
        raise RuntimeError("isolate exceeded its memory limit")

    monkeypatch.setattr(tasks_module, "now_ms", lambda: clock[0])
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: clock[0])
    monkeypatch.setattr(tasks_module, "_DISPATCH_BUDGET_SECONDS", 0)
    root_retained = fakes.WaitUntilRecorder()
    root_tasks = Tasks({})
    root_lifecycle = Lifecycle(
        fakes.FakeCtx(),
        host=object(),
        retain_work=root_retained,
        root_route_address=root_address,
        route_transport=transport,
        owns_physical_alarm=True,
        max_alarm_memory_limit_strikes=2,
        reset_alarm=lambda reason: None,
    )
    root_lifecycle.use(root_tasks)
    facet_tasks = Tasks({"report": report})
    facet_lifecycle = Lifecycle(
        fakes.FakeCtx(),
        host=object(),
        route_address=facet_address,
        root_route_address=root_address,
        route_transport=transport,
        owns_physical_alarm=False,
    )
    facet_lifecycle.use(facet_tasks)
    lifecycles[root_address.key] = root_lifecycle
    lifecycles[facet_address.key] = facet_lifecycle

    await facet_tasks.run("report", run_id="routed-tracked-memory")
    await root_lifecycle.alarm()
    await entered.wait()
    key = (facet_address.key, "routed-tracked-memory")
    routed_task = root_tasks._routed_wakes[key]
    first = facet_tasks._task_store().get("routed-tracked-memory")
    assert first is not None
    assert first.generation == root_tasks._routed_wake_generations[key]

    release.set()
    while not routed_task.done():
        await asyncio.sleep(0)
    assert root_tasks.lifecycle._alarm_work_is_tracked(routed_task)
    clock[0] = first.next_at
    await root_lifecycle.alarm()

    assert root_tasks._routed_wakes[key] is routed_task
    assert root_tasks._routed_wake_generations[key] == first.generation

    await root_retained.drain()

    backed_off = facet_tasks._task_store().get("routed-tracked-memory")
    assert backed_off is not None
    assert backed_off.state == "running"
    assert backed_off.attempt == 1
    assert backed_off.generation is None
    assert backed_off.wait_reason == "memory"
    assert backed_off.next_at == clock[0] + 30_000
    assert await root_tasks.lifecycle.storage.get("cf_agents:oom_alarm_strikes") == 1


@pytest.mark.asyncio
async def test_nested_facet_run_is_returned_as_a_root_mirror_repair(monkeypatch):
    clock = [1_000]
    root_address = LifecycleRouteAddress("root", '[{"name":"root"}]')
    facet_address = LifecycleRouteAddress(
        "root/facet",
        '[{"name":"root"},{"name":"facet"}]',
    )
    lifecycles = {}
    child_calls = []
    facet_tasks = None
    route_depth = 0
    max_route_depth = 0

    async def transport(envelope: LifecycleRouteEnvelope):
        nonlocal max_route_depth, route_depth
        route_depth += 1
        max_route_depth = max(max_route_depth, route_depth)
        try:
            lifecycle = lifecycles[envelope.target.key]
            return await lifecycle.route(
                version=envelope.version,
                source=envelope.source,
                target=envelope.target,
                capability_id=envelope.capability_id,
                payload=envelope.payload,
            )
        finally:
            route_depth -= 1

    async def parent(input, step):
        assert facet_tasks is not None
        await facet_tasks.run("child", run_id="nested")
        await facet_tasks.run("child", run_id="sibling")
        assert await facet_tasks.cancel("sibling", "parent") is True
        return "parent done"

    async def child(input, step):
        child_calls.append("ran")
        return "child done"

    root_tasks = Tasks({})
    root_lifecycle = Lifecycle(
        fakes.FakeCtx(),
        host=object(),
        root_route_address=root_address,
        route_transport=transport,
        owns_physical_alarm=True,
    )
    root_lifecycle.use(root_tasks)
    facet_retained = fakes.WaitUntilRecorder()
    facet_tasks = Tasks({"parent": parent, "child": child})
    facet_lifecycle = Lifecycle(
        fakes.FakeCtx(),
        host=object(),
        retain_work=facet_retained,
        route_address=facet_address,
        root_route_address=root_address,
        route_transport=transport,
        owns_physical_alarm=False,
    )
    facet_lifecycle.use(facet_tasks)
    lifecycles[root_address.key] = root_lifecycle
    lifecycles[facet_address.key] = facet_lifecycle
    monkeypatch.setattr(tasks_module, "now_ms", lambda: clock[0])
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: clock[0])

    await facet_tasks.run("child", run_id="sibling")
    await facet_tasks.run("parent", run_id="parent")
    await root_lifecycle.alarm()

    nested_job = await root_tasks.lifecycle.jobs.get("task:root/facet:nested")
    assert nested_job is not None
    assert nested_job.time == clock[0]
    sibling = await facet_tasks.get("sibling")
    assert sibling is not None
    assert sibling.state == "cancelled"
    assert await root_tasks.lifecycle.jobs.get("task:root/facet:sibling") is None

    await root_lifecycle.alarm()

    nested = await facet_tasks.get("nested")
    assert nested is not None
    assert nested.state == "completed"
    assert child_calls == ["ran"]
    assert facet_retained.coros == ()
    assert max_route_depth == 1
    await facet_retained.drain()


@pytest.mark.asyncio
async def test_stale_routed_memory_policy_cannot_seal_a_new_generation():
    facet_address = LifecycleRouteAddress("root/facet", '[{"name":"facet"}]')
    host_policies = []
    ctx = fakes.FakeCtx()
    tasks = Tasks({})
    lifecycle = Lifecycle(
        ctx,
        host=object(),
        route_address=facet_address,
        root_route_address=LifecycleRouteAddress("root", '[{"name":"root"}]'),
        owns_physical_alarm=False,
        on_alarm_memory_limit=host_policies.append,
    )
    lifecycle.use(tasks)
    await lifecycle.start()
    _insert_run(
        ctx,
        "new-attempt",
        "running",
        generation="new-generation",
        next_at=500,
    )

    response = await lifecycle.route(
        version=1,
        source=None,
        target=facet_address,
        capability_id="tasks",
        payload={
            "type": "memory",
            "runId": "new-attempt",
            "generation": "old-generation",
            "sealed": True,
            "nextAt": None,
        },
    )

    assert response == {"nextAt": 500, "status": "stale"}
    current = tasks._task_store().get("new-attempt")
    assert current is not None
    assert current.state == "running"
    assert current.generation == "new-generation"
    assert host_policies == []

    async def transport(envelope: LifecycleRouteEnvelope):
        return await lifecycle.route(
            version=envelope.version,
            source=envelope.source,
            target=envelope.target,
            capability_id=envelope.capability_id,
            payload=envelope.payload,
        )

    root_tasks = Tasks({})
    root_lifecycle = Lifecycle(
        fakes.FakeCtx(),
        host=object(),
        root_route_address=LifecycleRouteAddress("root", '[{"name":"root"}]'),
        route_transport=transport,
        owns_physical_alarm=True,
    )
    root_lifecycle.use(root_tasks)
    ctx.storage.sql.exec(
        """
        UPDATE cf_agents_task_runs
        SET generation = NULL, wait_reason = 'memory', next_at = 600
        WHERE run_id = 'new-attempt'
        """
    )

    deadline, status = await root_tasks._deliver_routed_memory_policy(
        facet_address,
        "new-attempt",
        "old-generation",
        False,
        600,
    )

    assert (deadline, status) == (600, "already")
    mirror = await root_tasks.lifecycle.jobs.get("task:root/facet:new-attempt")
    assert mirror is not None
    assert mirror.time == 600

    ctx.storage.sql.exec(
        """
        UPDATE cf_agents_task_runs
        SET state = 'failed', wait_reason = NULL, next_at = NULL,
            error_name = 'TaskMemoryLimitSealed', settled_at = 700
        WHERE run_id = 'new-attempt'
        """
    )
    deadline, status = await root_tasks._deliver_routed_memory_policy(
        facet_address,
        "new-attempt",
        "old-generation",
        True,
        None,
    )
    assert (deadline, status) == (None, "already")
    assert await root_tasks.lifecycle.jobs.get("task:root/facet:new-attempt") is None


@pytest.mark.asyncio
async def test_routed_preclaim_memory_failure_defers_instead_of_hot_looping(
    monkeypatch,
):
    clock = [1_000]
    root_address = LifecycleRouteAddress("root", '[{"name":"root"}]')
    facet_address = LifecycleRouteAddress("root/facet", '[{"name":"facet"}]')
    lifecycles = {}
    fail_wake = [True]
    facet_policies = []

    async def transport(envelope: LifecycleRouteEnvelope):
        if (
            isinstance(envelope.payload, dict)
            and envelope.payload.get("type") == "wake"
            and fail_wake[0]
        ):
            fail_wake[0] = False
            raise RuntimeError("isolate exceeded its memory limit")
        lifecycle = lifecycles[envelope.target.key]
        return await lifecycle.route(
            version=envelope.version,
            source=envelope.source,
            target=envelope.target,
            capability_id=envelope.capability_id,
            payload=envelope.payload,
        )

    monkeypatch.setattr(tasks_module, "now_ms", lambda: clock[0])
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: clock[0])
    root_tasks = Tasks({})
    root_lifecycle = Lifecycle(
        fakes.FakeCtx(),
        host=object(),
        root_route_address=root_address,
        route_transport=transport,
        owns_physical_alarm=True,
        max_alarm_memory_limit_strikes=1,
        reset_alarm=lambda reason: None,
    )
    root_lifecycle.use(root_tasks)
    facet_tasks = Tasks({"report": lambda input, step: "done"})
    facet_lifecycle = Lifecycle(
        fakes.FakeCtx(),
        host=object(),
        route_address=facet_address,
        root_route_address=root_address,
        route_transport=transport,
        owns_physical_alarm=False,
        on_alarm_memory_limit=facet_policies.append,
    )
    facet_lifecycle.use(facet_tasks)
    lifecycles[root_address.key] = root_lifecycle
    lifecycles[facet_address.key] = facet_lifecycle

    await facet_tasks.run("report", run_id="preclaim")
    await root_lifecycle.alarm()

    pending = await facet_tasks.get("preclaim")
    assert pending is not None
    assert pending.state == "pending"
    assert facet_tasks._task_store().get("preclaim").attempt == 0
    assert facet_policies == []
    deferred = await root_tasks.lifecycle.jobs.get("task:root/facet:preclaim")
    assert deferred is not None
    assert deferred.time == 31_000

    clock[0] = 31_000
    await root_lifecycle.alarm()

    completed = await facet_tasks.get("preclaim")
    assert completed is not None
    assert completed.state == "completed"


@pytest.mark.asyncio
async def test_cancelled_memory_policy_persistence_restores_the_root_mirror(
    monkeypatch,
):
    clock = [1_000]
    owner = LifecycleRouteAddress("root/facet", '[{"name":"facet"}]')
    ctx = fakes.FakeCtx()
    tasks = Tasks({})
    lifecycle = Lifecycle(ctx, host=object())
    lifecycle.use(tasks)
    await lifecycle.start()
    monkeypatch.setattr(tasks_module, "now_ms", lambda: clock[0])
    original_put = ctx.storage.put
    attempts = 0

    async def cancelling_put(key, value):
        nonlocal attempts
        if key.startswith(tasks_module._MEMORY_POLICY_PREFIX):
            attempts += 1
            if attempts == 1:
                raise asyncio.CancelledError
        await original_put(key, value)

    ctx.storage.put = cancelling_put
    tasks._routed_wake_generations[(owner.key, "cancelled-policy")] = "generation"
    job = LifecycleJob(
        id="task:root/facet:cancelled-policy",
        capability="tasks",
        fn="wake",
        time=clock[0],
        payload={
            "runId": "cancelled-policy",
            "owner_path": owner.data,
            "owner_path_key": owner.key,
        },
        payload_present=True,
        retry={"maxAttempts": 1},
        singleflight=False,
        exclusive=False,
        recovery_loop=False,
        created_at=clock[0],
    )
    context = LifecycleMemoryLimitContext(
        sealed=True,
        next_time=None,
        executing=job,
        purged_recovery_loop_jobs=(),
    )

    with pytest.raises(asyncio.CancelledError):
        await tasks.on_memory_limit(context)

    policies = await tasks.lifecycle.storage.list(
        prefix=tasks_module._MEMORY_POLICY_PREFIX
    )
    assert len(policies) == 1
    assert attempts == 2
    retry = await tasks.lifecycle.jobs.get(job.id)
    assert retry is not None
    assert retry.time == 31_000


@pytest.mark.asyncio
async def test_routed_memory_seal_reaches_the_facet_row_and_host(monkeypatch):
    clock = [1_000]
    root_address = LifecycleRouteAddress("root", '[{"name":"root"}]')
    facet_address = LifecycleRouteAddress(
        "root/facet",
        '[{"name":"root"},{"name":"facet"}]',
    )
    lifecycles = {}
    facet_policies: list[LifecycleMemoryLimitContext] = []
    fail_seal_deliveries = [2]

    async def transport(envelope: LifecycleRouteEnvelope):
        if (
            isinstance(envelope.payload, dict)
            and envelope.payload.get("type") == "memory"
            and envelope.payload.get("sealed") is True
            and fail_seal_deliveries[0] > 0
        ):
            fail_seal_deliveries[0] -= 1
            raise RuntimeError("memory policy delivery failed")
        lifecycle = lifecycles[envelope.target.key]
        return await lifecycle.route(
            version=envelope.version,
            source=envelope.source,
            target=envelope.target,
            capability_id=envelope.capability_id,
            payload=envelope.payload,
        )

    async def report(input, step):
        raise RuntimeError("Durable Object's isolate exceeded its memory limit")

    monkeypatch.setattr(tasks_module, "now_ms", lambda: clock[0])
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: clock[0])
    root_runtime = fakes.FakeDurableObjectRuntime()
    root_ctx = root_runtime.new_context()
    original_put = root_ctx.storage.put
    policy_put_attempts = 0

    async def flaky_policy_put(key, value):
        nonlocal policy_put_attempts
        if key.startswith(tasks_module._MEMORY_POLICY_PREFIX):
            policy_put_attempts += 1
            if policy_put_attempts == 1:
                raise RuntimeError("memory policy persistence failed")
        await original_put(key, value)

    root_ctx.storage.put = flaky_policy_put
    root_tasks = Tasks({})
    root_lifecycle = Lifecycle(
        root_ctx,
        host=object(),
        root_route_address=root_address,
        route_transport=transport,
        owns_physical_alarm=True,
        max_alarm_memory_limit_strikes=2,
        reset_alarm=lambda reason: None,
    )
    root_lifecycle.use(root_tasks)
    facet_ctx = fakes.FakeCtx()
    facet_tasks = Tasks({"report": report})
    facet_lifecycle = Lifecycle(
        facet_ctx,
        host=object(),
        route_address=facet_address,
        root_route_address=root_address,
        route_transport=transport,
        owns_physical_alarm=False,
        on_alarm_memory_limit=facet_policies.append,
    )
    facet_lifecycle.use(facet_tasks)
    lifecycles[root_address.key] = root_lifecycle
    lifecycles[facet_address.key] = facet_lifecycle

    await facet_tasks.run("report", run_id="routed-memory")
    await root_lifecycle.alarm()

    backed_off = facet_tasks._task_store().get("routed-memory")
    assert backed_off is not None
    assert backed_off.state == "running"
    assert backed_off.generation is None
    assert backed_off.wait_reason == "memory"
    assert backed_off.next_at == 31_000
    assert len(facet_policies) == 1
    assert not facet_policies[0].sealed

    clock[0] = 31_000
    await root_lifecycle.alarm()

    pending_seal = facet_tasks._task_store().get("routed-memory")
    assert pending_seal is not None
    assert pending_seal.state == "running"
    assert pending_seal.generation is not None
    policies = await root_tasks.lifecycle.storage.list(
        prefix=tasks_module._MEMORY_POLICY_PREFIX
    )
    assert len(policies) == 1
    [policy] = policies.values()
    assert policy["generation"] == pending_seal.generation
    retry = await root_tasks.lifecycle.jobs.get("task:root/facet:routed-memory")
    assert retry is not None
    assert retry.time == 61_000
    assert policy_put_attempts == 3

    cold_root_tasks = Tasks({})
    cold_root_lifecycle = Lifecycle(
        root_runtime.evict(),
        host=object(),
        root_route_address=root_address,
        route_transport=transport,
        owns_physical_alarm=True,
        max_alarm_memory_limit_strikes=2,
        reset_alarm=lambda reason: None,
    )
    cold_root_lifecycle.use(cold_root_tasks)
    lifecycles[root_address.key] = cold_root_lifecycle
    await cold_root_lifecycle.start()

    clock[0] = 61_000
    await cold_root_lifecycle.alarm()

    retry = await cold_root_tasks.lifecycle.jobs.get("task:root/facet:routed-memory")
    assert retry is not None
    assert retry.time == 91_000

    clock[0] = 91_000
    await cold_root_lifecycle.alarm()

    sealed = await facet_tasks.get("routed-memory")
    assert sealed is not None
    assert sealed.state == "failed"
    assert sealed.error.name == "TaskMemoryLimitSealed"
    assert len(facet_policies) == 2
    assert facet_policies[1].sealed
    assert facet_policies[1].executing is None
    assert (
        await cold_root_tasks.lifecycle.jobs.get("task:root/facet:routed-memory")
        is None
    )
    assert (
        await cold_root_tasks.lifecycle.storage.list(
            prefix=tasks_module._MEMORY_POLICY_PREFIX
        )
        == {}
    )
    assert cold_root_tasks._routed_wake_generations == {}


@pytest.mark.asyncio
async def test_startup_rearms_a_persisted_routed_memory_policy(monkeypatch):
    clock = [5_000]
    runtime = fakes.FakeDurableObjectRuntime()
    owner = LifecycleRouteAddress("root/facet", '[{"name":"facet"}]')
    first_tasks = Tasks({})
    first_lifecycle = Lifecycle(runtime.new_context(), host=object())
    first_lifecycle.use(first_tasks)
    await first_lifecycle.start()
    await first_tasks._persist_memory_policy(
        owner,
        "sealed",
        "struck-generation",
        sealed=True,
        next_at=None,
    )
    monkeypatch.setattr(tasks_module, "now_ms", lambda: clock[0])

    second_tasks = Tasks({})
    second_lifecycle = Lifecycle(runtime.evict(), host=object())
    second_lifecycle.use(second_tasks)
    await second_lifecycle.start()

    job = await second_tasks.lifecycle.jobs.get("task:root/facet:sealed")
    assert job is not None
    assert job.time == 5_000
    assert job.payload == {
        "runId": "sealed",
        "owner_path": owner.data,
        "owner_path_key": owner.key,
    }


@pytest.mark.parametrize("value", [-1, float("inf"), True, "1 fortnight"])
def test_invalid_sleep_durations_are_rejected(value):
    with pytest.raises((TypeError, ValueError)):
        tasks_module._parse_task_duration(value, "sleep duration")
