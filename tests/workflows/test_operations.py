from __future__ import annotations

from collections.abc import Mapping

import pytest

from agents.workflows import (
    AgentWorkflowRootOrigin,
    RunWorkflowOptions,
    WorkflowCallbackHandlers,
    WorkflowCompleteCallback,
    WorkflowErrorCallback,
    WorkflowErrorInfo,
    WorkflowEventCallback,
    WorkflowEventPayload,
    WorkflowInstanceStatus,
    WorkflowLedger,
    WorkflowOperations,
    WorkflowProgressCallback,
    _WorkersWorkflowRuntime,
)
from tests.workflows.test_ledger import Sql


class Runtime:
    def __init__(self) -> None:
        self.bindings = {"REPORTS", "RENAMED"}
        self.creates: list[tuple[str, str, dict[str, object], object]] = []
        self.events: list[tuple[str, str, dict[str, object]]] = []
        self.controls: list[tuple[str, str, str]] = []
        self.statuses: dict[str, WorkflowInstanceStatus] = {}

    def has_binding(self, workflow_name: str) -> bool:
        return workflow_name in self.bindings

    async def create(
        self,
        workflow_name: str,
        workflow_id: str,
        params: Mapping[str, object],
        retention: object,
    ) -> str:
        self.creates.append((workflow_name, workflow_id, dict(params), retention))
        return workflow_id

    async def status(
        self, workflow_name: str, workflow_id: str
    ) -> WorkflowInstanceStatus:
        return self.statuses.get(workflow_id, WorkflowInstanceStatus(status="running"))

    async def send_event(
        self,
        workflow_name: str,
        workflow_id: str,
        event: Mapping[str, object],
    ) -> None:
        self.events.append((workflow_name, workflow_id, dict(event)))

    async def control(self, action: str, workflow_name: str, workflow_id: str) -> None:
        self.controls.append((action, workflow_name, workflow_id))
        statuses = {
            "pause": "paused",
            "resume": "running",
            "restart": "queued",
            "terminate": "terminated",
        }
        self.statuses[workflow_id] = WorkflowInstanceStatus(status=statuses[action])


class Clock:
    seconds = 200

    def __call__(self) -> int:
        return self.seconds


def operations(*, callbacks=None, runtime=None, sleep=None):
    sql = Sql()
    clock = Clock()
    row_ids = iter(f"row-{index}" for index in range(20))
    ledger = WorkflowLedger(
        sql,
        id_factory=lambda: next(row_ids),
        now_seconds=clock,
    )
    runtime = runtime or Runtime()
    options = {} if sleep is None else {"sleep": sleep}
    emitted = []
    service = WorkflowOperations(
        ledger,
        runtime,
        AgentWorkflowRootOrigin(binding="RootAgents", name="tenant-7"),
        callbacks=callbacks,
        emit=lambda event, payload: emitted.append((event, payload)),
        id_factory=lambda: "generated",
        **options,
    )
    return service, ledger, runtime, emitted, clock


class RetryableError(RuntimeError):
    retryable = True
    overloaded = False


class RetryingRuntime(Runtime):
    def __init__(self, *, event_failures=0, control_failures=0) -> None:
        super().__init__()
        self.event_failures = event_failures
        self.control_failures = control_failures
        self.event_attempts = 0
        self.control_attempts = 0

    async def send_event(self, workflow_name, workflow_id, event) -> None:
        self.event_attempts += 1
        if self.event_attempts <= self.event_failures:
            raise RetryableError("temporary event failure")
        await super().send_event(workflow_name, workflow_id, event)

    async def control(self, action, workflow_name, workflow_id):
        self.control_attempts += 1
        if self.control_attempts <= self.control_failures:
            raise RetryableError("temporary control failure")
        await super().control(action, workflow_name, workflow_id)


class NativeWorkflowInstance:
    def __init__(self) -> None:
        self.events = []
        self.controls = []
        self.destroyed = 0

    async def status(self):
        return {
            "status": "running",
            "output": {"records": 7},
            "customField": "preserved",
        }

    async def sendEvent(self, event):
        self.events.append(dict(event))

    async def pause(self):
        self.controls.append("pause")

    def destroy(self):
        self.destroyed += 1


class NativeWorkflowBinding:
    def __init__(self) -> None:
        self.created = []
        self.instance = NativeWorkflowInstance()

    async def create(self, options):
        self.created.append(dict(options))
        return {"id": options["id"], "platformField": "preserved"}

    async def get(self, workflow_id):
        assert workflow_id == "wf-1"
        return self.instance


@pytest.mark.asyncio
async def test_run_injects_versioned_origin_and_tracks_the_created_instance():
    service, ledger, runtime, emitted, _ = operations()
    retention = {"successRetention": "1 day", "errorRetention": "2 weeks"}

    workflow_id = await service.run_workflow(
        "REPORTS",
        {"report": 42},
        RunWorkflowOptions(
            id="wf-custom",
            metadata={"tenant": "a"},
            retention=retention,
        ),
    )

    assert workflow_id == "wf-custom"
    assert runtime.creates == [
        (
            "REPORTS",
            "wf-custom",
            {
                "report": 42,
                "__agentName": "tenant-7",
                "__agentBinding": "RootAgents",
                "__workflowName": "REPORTS",
                "__agentOrigin": {
                    "kind": "agent",
                    "version": 1,
                    "binding": "RootAgents",
                    "name": "tenant-7",
                },
            },
            retention,
        )
    ]
    assert ledger.get(workflow_id).metadata == {"tenant": "a"}
    assert emitted == [
        ("workflow:start", {"workflowId": "wf-custom", "workflowName": "REPORTS"})
    ]


@pytest.mark.asyncio
async def test_status_refresh_events_approvals_and_controls_use_the_tracked_binding():
    service, ledger, runtime, emitted, _ = operations()
    ledger.track("wf-1", "REPORTS")
    runtime.statuses["wf-1"] = WorkflowInstanceStatus(status="waiting")

    assert await service.get_workflow_status(
        "REPORTS", "wf-1"
    ) == WorkflowInstanceStatus(status="waiting")
    assert ledger.get("wf-1").status == "waiting"

    await service.send_workflow_event(
        "REPORTS", "wf-1", WorkflowEventPayload(type="signal", payload={"n": 1})
    )
    await service.approve_workflow("wf-1", reason="safe", metadata={"reviewer": "sam"})
    await service.reject_workflow("wf-1", reason="expired")
    await service.pause_workflow("wf-1")
    await service.resume_workflow("wf-1")
    await service.terminate_workflow("wf-1")

    assert runtime.events == [
        ("REPORTS", "wf-1", {"type": "signal", "payload": {"n": 1}}),
        (
            "REPORTS",
            "wf-1",
            {
                "type": "approval",
                "payload": {
                    "approved": True,
                    "reason": "safe",
                    "metadata": {"reviewer": "sam"},
                },
            },
        ),
        (
            "REPORTS",
            "wf-1",
            {
                "type": "approval",
                "payload": {"approved": False, "reason": "expired"},
            },
        ),
    ]
    assert runtime.controls == [
        ("pause", "REPORTS", "wf-1"),
        ("resume", "REPORTS", "wf-1"),
        ("terminate", "REPORTS", "wf-1"),
    ]
    assert ledger.get("wf-1").status == "terminated"
    assert [event for event, _ in emitted] == [
        "workflow:event",
        "workflow:event",
        "workflow:approved",
        "workflow:event",
        "workflow:rejected",
        "workflow:paused",
        "workflow:resumed",
        "workflow:terminated",
    ]


@pytest.mark.asyncio
async def test_retryable_sends_and_controls_retry_at_most_three_times():
    delays = []

    async def sleep(delay):
        delays.append(delay)

    runtime = RetryingRuntime(event_failures=2, control_failures=2)
    service, ledger, _, _, _ = operations(runtime=runtime, sleep=sleep)
    ledger.track("wf-1", "REPORTS")

    await service.send_workflow_event(
        "REPORTS", "wf-1", WorkflowEventPayload(type="signal", payload={})
    )
    await service.pause_workflow("wf-1")

    assert runtime.event_attempts == 3
    assert runtime.control_attempts == 3
    assert delays == [0.2, 0.4, 0.2, 0.4]

    exhausted = RetryingRuntime(event_failures=3)
    service, _, _, _, _ = operations(runtime=exhausted, sleep=sleep)
    with pytest.raises(RetryableError, match="temporary event failure"):
        await service.send_workflow_event(
            "REPORTS", "wf-2", WorkflowEventPayload(type="signal", payload={})
        )
    assert exhausted.event_attempts == 3


@pytest.mark.asyncio
async def test_workers_binding_runtime_preserves_native_status_output():
    binding = NativeWorkflowBinding()
    env = type("Env", (), {"REPORTS": binding})()
    runtime = _WorkersWorkflowRuntime(env)

    assert runtime.has_binding("REPORTS") is True
    assert runtime.has_binding("MISSING") is False
    assert await runtime.create("REPORTS", "wf-1", {"task": 7}, None) == "wf-1"
    status = await runtime.status("REPORTS", "wf-1")
    await runtime.send_event(
        "REPORTS", "wf-1", {"type": "signal", "payload": {"ok": True}}
    )
    await runtime.control("pause", "REPORTS", "wf-1")
    controlled = await runtime.status("REPORTS", "wf-1")

    assert binding.created == [{"id": "wf-1", "params": {"task": 7}}]
    assert status.native == {
        "status": "running",
        "output": {"records": 7},
        "customField": "preserved",
    }
    assert status["output"] == {"records": 7}
    assert controlled.native == status.native
    assert binding.instance.events == [{"type": "signal", "payload": {"ok": True}}]
    assert binding.instance.controls == ["pause"]
    assert binding.instance.destroyed == 4


@pytest.mark.asyncio
async def test_for_agent_builds_operations_with_the_workers_runtime():
    binding = NativeWorkflowBinding()
    env = type("Env", (), {"REPORTS": binding})()
    sql = Sql()
    service = WorkflowOperations.for_agent(
        sql.execute,
        env,
        AgentWorkflowRootOrigin(binding="RootAgents", name="tenant-7"),
    )

    workflow_id = await service.run_workflow(
        "REPORTS",
        {"task": 7},
        RunWorkflowOptions(id="wf-1"),
    )

    assert workflow_id == "wf-1"


@pytest.mark.asyncio
async def test_restart_resets_tracking_by_default_or_preserves_creation_time():
    service, ledger, runtime, _, clock = operations()
    ledger.track("wf-1", "REPORTS")
    clock.seconds = 300

    await service.restart_workflow("wf-1")
    reset = ledger.get("wf-1")
    assert reset.status == "queued"
    assert int(reset.created_at.timestamp()) == 300

    clock.seconds = 400
    await service.restart_workflow("wf-1", reset_tracking=False)
    preserved = ledger.get("wf-1")
    assert preserved.status == "queued"
    assert int(preserved.created_at.timestamp()) == 300


@pytest.mark.asyncio
async def test_missing_rows_and_bindings_fail_before_runtime_control():
    service, ledger, runtime, _, _ = operations()

    with pytest.raises(ValueError, match="not found in tracking table"):
        await service.pause_workflow("missing")

    ledger.track("wf-1", "REMOVED")
    with pytest.raises(ValueError, match="binding 'REMOVED' not found"):
        await service.pause_workflow("wf-1")
    assert runtime.controls == []


@pytest.mark.asyncio
async def test_callbacks_update_the_ledger_before_user_hooks_and_guard_terminal_races():
    observations = []
    ledger_ref = None

    async def on_progress(name, workflow_id, progress):
        observations.append(("progress", ledger_ref.get(workflow_id).status, progress))

    async def on_complete(name, workflow_id, result):
        observations.append(("complete", ledger_ref.get(workflow_id).status, result))

    async def on_error(name, workflow_id, error):
        observations.append(("error", ledger_ref.get(workflow_id).status, error))

    async def on_event(name, workflow_id, event):
        observations.append(("event", ledger_ref.get(workflow_id).status, event))

    service, ledger_ref, _, _, _ = operations(
        callbacks=WorkflowCallbackHandlers(
            on_progress=on_progress,
            on_complete=on_complete,
            on_error=on_error,
            on_event=on_event,
        )
    )
    ledger_ref.track("wf-1", "REPORTS")

    await service.handle_callback(
        WorkflowProgressCallback(
            workflow_name="REPORTS",
            workflow_id="wf-1",
            timestamp=1,
            progress={"percent": 0.5},
        )
    )
    await service.handle_callback(
        WorkflowEventCallback(
            workflow_name="REPORTS",
            workflow_id="wf-1",
            timestamp=2,
            event={"kind": "audit"},
        )
    )
    await service.handle_callback(
        WorkflowCompleteCallback(
            workflow_name="REPORTS",
            workflow_id="wf-1",
            timestamp=3,
            result={"ok": True},
        )
    )
    ledger_ref.update_status("wf-1", WorkflowInstanceStatus(status="paused"))
    await service.handle_callback(
        WorkflowErrorCallback(
            workflow_name="REPORTS",
            workflow_id="wf-1",
            timestamp=4,
            error="late failure",
        )
    )

    assert observations == [
        ("progress", "running", {"percent": 0.5}),
        ("event", "running", {"kind": "audit"}),
        ("complete", "complete", {"ok": True}),
        ("error", "paused", "late failure"),
    ]
    assert ledger_ref.get("wf-1").status == "paused"


def test_binding_migration_requires_a_live_destination_binding():
    service, ledger, _, _, _ = operations()
    ledger.track("wf-1", "OLD")

    assert service.migrate_workflow_binding("OLD", "RENAMED") == 1
    assert ledger.get("wf-1").workflow_name == "RENAMED"
    with pytest.raises(ValueError, match="binding 'MISSING' not found"):
        service.migrate_workflow_binding("RENAMED", "MISSING")


def test_operations_exposes_bulk_tracking_cleanup_for_the_agent_delegate():
    service, ledger, _, _, _ = operations()
    ledger.track("wf-complete", "REPORTS", {"tenant.id": "a"})
    ledger.track("wf-running", "REPORTS", {"tenant.id": "a"})
    ledger.update_status("wf-complete", WorkflowInstanceStatus(status="complete"))

    assert (
        service.delete_workflows(
            status="complete",
            metadata={"tenant.id": "a"},
        )
        == 1
    )
    assert service.get_workflow("wf-complete") is None
    assert service.get_workflow("wf-running") is not None


@pytest.mark.asyncio
async def test_rpc_callback_wire_is_decoded_at_the_operations_boundary():
    observed = []
    service, ledger, _, _, _ = operations(
        callbacks=WorkflowCallbackHandlers(
            on_progress=lambda name, workflow_id, progress: observed.append(
                (name, workflow_id, progress)
            )
        )
    )
    ledger.track("wf-1", "REPORTS")

    await service.handle_callback(
        {
            "workflowName": "REPORTS",
            "workflowId": "wf-1",
            "type": "progress",
            "timestamp": 1234,
            "progress": {"percent": 0.25},
        }
    )

    assert ledger.get("wf-1").status == "running"
    assert observed == [("REPORTS", "wf-1", {"percent": 0.25})]
