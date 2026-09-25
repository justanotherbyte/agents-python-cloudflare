from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping

import pytest
from workers import WorkflowEntrypoint

from agents.workflows import (
    AgentWorkflow,
    AgentWorkflowEvent,
    WorkflowCompleteCallback,
    WorkflowRejectedError,
    WorkflowStepEvent,
    WaitForApprovalOptions,
    decode_workflow_callback,
    encode_workflow_callback,
    _WorkersWorkflowStepRuntime,
)
from tests.workflows.test_origins import Resolver


def test_agent_workflow_is_a_native_workflow_entrypoint():
    assert issubclass(AgentWorkflow, WorkflowEntrypoint)


def test_omitted_completion_result_stays_absent_on_the_rpc_wire():
    callback = WorkflowCompleteCallback(
        workflow_name="REPORTS",
        workflow_id="wf-1",
        timestamp=1234,
    )

    wire = encode_workflow_callback(callback)

    assert wire == {
        "workflowName": "REPORTS",
        "workflowId": "wf-1",
        "type": "complete",
        "timestamp": 1234,
    }
    assert encode_workflow_callback(decode_workflow_callback(wire)) == wire
    assert (
        encode_workflow_callback(
            WorkflowCompleteCallback(
                workflow_name="REPORTS",
                workflow_id="wf-1",
                timestamp=1234,
                result=None,
            )
        )["result"]
        is None
    )


class AgentStub:
    def __init__(self) -> None:
        self.callbacks: list[dict[str, object]] = []
        self.state_actions: list[tuple[object, ...]] = []
        self.broadcasts: list[object] = []
        self.calls: list[tuple[str, tuple[object, ...]]] = []
        self.fail_callback = False
        self.destroyed = 0

    async def _workflow_handleCallback(self, callback: Mapping[str, object]) -> None:
        if self.fail_callback:
            raise RuntimeError("agent unavailable")
        self.callbacks.append(dict(callback))

    async def _workflow_updateState(self, *args: object) -> None:
        self.state_actions.append(args)

    async def _workflow_broadcast(self, message: object) -> None:
        self.broadcasts.append(message)

    async def record_result(self, *args: object) -> None:
        self.calls.append(("record_result", args))

    def destroy(self) -> None:
        self.destroyed += 1


class RootResolver(Resolver):
    def __init__(self) -> None:
        super().__init__()
        self.root = AgentStub()


class NativeStep:
    def __init__(self) -> None:
        self.native_api = "preserved"


class StepRuntime:
    def __init__(self, *, run_durable: bool = True) -> None:
        self.run_durable = run_durable
        self.names: list[str] = []
        self.waits: list[tuple[str, str, object]] = []
        self.wait_event = WorkflowStepEvent(
            payload={"approved": True, "metadata": {"by": "sam"}}
        )
        self.released_events: list[WorkflowStepEvent] = []

    def install(
        self,
        step: object,
        name: str,
        callback: Callable[..., Awaitable[object]],
    ) -> None:
        setattr(step, name, callback)

    async def durable(
        self,
        step: object,
        name: str,
        callback: Callable[[], Awaitable[object]],
    ) -> object:
        self.names.append(name)
        if self.run_durable:
            return await callback()
        return None

    async def wait_for_event(
        self,
        step: object,
        step_name: str,
        event_type: str,
        timeout: object,
    ) -> WorkflowStepEvent:
        self.waits.append((step_name, event_type, timeout))
        return self.wait_event

    def release_event(self, event: WorkflowStepEvent) -> None:
        self.released_events.append(event)


def event(payload: Mapping[str, object]) -> AgentWorkflowEvent[Mapping[str, object]]:
    return AgentWorkflowEvent(
        instance_id="wf-1",
        native={"instanceId": "wf-1", "timestamp": 99},
        payload={
            **payload,
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
    )


class ReportingWorkflow(AgentWorkflow):
    def __init__(self, resolver, step_runtime) -> None:
        super().__init__(
            object(),
            object(),
            resolver=resolver,
            step_runtime=step_runtime,
            now_ms=lambda: 1234,
        )
        self.received_step = None

    async def run(self, workflow_event, step):
        self.received_step = step
        assert workflow_event.payload == {"task": 7}
        assert workflow_event["timestamp"] == 99
        assert workflow_event.timestamp == 99
        assert self.workflow_id == "wf-1"
        assert self.workflow_name == "REPORTS"
        await self.agent.record_result("wf-1")
        await self.report_progress({"percent": 0.5})
        await self.broadcast_to_clients({"kind": "progress"})
        await step.report_complete({"ok": True})
        await step.send_event({"kind": "audit"})
        await step.update_agent_state({"count": 1})
        await step.merge_agent_state({"status": "done"})
        await step.reset_agent_state()
        return "finished"


@pytest.mark.asyncio
async def test_execute_preserves_the_native_step_and_installs_durable_agent_helpers():
    resolver = RootResolver()
    step_runtime = StepRuntime()
    workflow = ReportingWorkflow(resolver, step_runtime)
    step = NativeStep()

    result = await workflow.execute(event({"task": 7}), step)

    assert result == "finished"
    assert workflow.received_step is step
    assert step.native_api == "preserved"
    assert step_runtime.names == [
        "__agent_reportComplete_0",
        "__agent_sendEvent_1",
        "__agent_updateState_2",
        "__agent_mergeState_3",
        "__agent_resetState_4",
    ]
    assert resolver.root.callbacks == [
        {
            "workflowName": "REPORTS",
            "workflowId": "wf-1",
            "type": "progress",
            "progress": {"percent": 0.5},
            "timestamp": 1234,
        },
        {
            "workflowName": "REPORTS",
            "workflowId": "wf-1",
            "type": "complete",
            "result": {"ok": True},
            "timestamp": 1234,
        },
        {
            "workflowName": "REPORTS",
            "workflowId": "wf-1",
            "type": "event",
            "event": {"kind": "audit"},
            "timestamp": 1234,
        },
    ]
    assert resolver.root.state_actions == [
        ("set", {"count": 1}),
        ("merge", {"status": "done"}),
        ("reset",),
    ]
    assert resolver.root.broadcasts == [{"kind": "progress"}]
    assert resolver.released == [resolver.root]
    with pytest.raises(RuntimeError, match="not initialized"):
        _ = workflow.agent


@pytest.mark.asyncio
async def test_durable_helpers_do_not_pre_evaluate_their_callbacks():
    resolver = RootResolver()
    step_runtime = StepRuntime(run_durable=False)
    workflow = ReportingWorkflow(resolver, step_runtime)

    await workflow.execute(event({"task": 7}), NativeStep())

    assert [callback["type"] for callback in resolver.root.callbacks] == ["progress"]
    assert resolver.root.state_actions == []
    assert step_runtime.names == [
        "__agent_reportComplete_0",
        "__agent_sendEvent_1",
        "__agent_updateState_2",
        "__agent_mergeState_3",
        "__agent_resetState_4",
    ]


class ApprovalWorkflow(AgentWorkflow):
    async def run(self, workflow_event, step):
        return await self.wait_for_approval(
            step,
            WaitForApprovalOptions(
                step_name="review",
                event_type="decision",
                timeout="7 days",
            ),
        )


@pytest.mark.asyncio
async def test_wait_for_approval_returns_metadata_and_releases_the_native_event():
    resolver = RootResolver()
    step_runtime = StepRuntime()
    workflow = ApprovalWorkflow(
        object(),
        object(),
        resolver=resolver,
        step_runtime=step_runtime,
        now_ms=lambda: 1234,
    )

    result = await workflow.execute(event({}), NativeStep())

    assert result == {"by": "sam"}
    assert step_runtime.waits == [("review", "decision", "7 days")]
    assert step_runtime.released_events == [step_runtime.wait_event]


@pytest.mark.asyncio
async def test_rejection_reports_durably_once_then_raises_and_releases_event():
    resolver = RootResolver()
    step_runtime = StepRuntime()
    step_runtime.wait_event = WorkflowStepEvent(
        payload={"approved": False, "reason": "budget exceeded"}
    )
    workflow = ApprovalWorkflow(
        object(),
        object(),
        resolver=resolver,
        step_runtime=step_runtime,
        now_ms=lambda: 1234,
    )

    with pytest.raises(WorkflowRejectedError) as caught:
        await workflow.execute(event({}), NativeStep())

    assert caught.value.reason == "budget exceeded"
    assert caught.value.workflow_id == "wf-1"
    assert step_runtime.names == ["__agent_reportError_0"]
    assert [callback["type"] for callback in resolver.root.callbacks] == ["error"]
    assert step_runtime.released_events == [step_runtime.wait_event]


class FailingWorkflow(AgentWorkflow):
    async def run(self, workflow_event, step):
        raise LookupError("original failure")


@pytest.mark.asyncio
async def test_unhandled_error_reports_once_and_notification_failure_never_masks_it():
    resolver = RootResolver()
    resolver.root.fail_callback = True
    workflow = FailingWorkflow(
        object(),
        object(),
        resolver=resolver,
        step_runtime=StepRuntime(),
        now_ms=lambda: 1234,
    )

    with pytest.raises(LookupError, match="original failure"):
        await workflow.execute(event({}), NativeStep())

    assert resolver.released == [resolver.root]


@pytest.mark.asyncio
async def test_release_failure_does_not_mask_user_result():
    class FailingReleaseResolver(RootResolver):
        async def release(self, agent):
            self.released.append(agent)
            raise RuntimeError("release failed")

    class SuccessfulWorkflow(AgentWorkflow):
        async def run(self, workflow_event, step):
            return "complete"

    resolver = FailingReleaseResolver()
    workflow = SuccessfulWorkflow(
        object(), object(), resolver=resolver, step_runtime=StepRuntime()
    )

    assert await workflow.execute(event({}), NativeStep()) == "complete"
    assert resolver.released == [resolver.root]


@pytest.mark.asyncio
async def test_unsupported_persisted_origin_fails_before_user_code():
    resolver = RootResolver()
    workflow = FailingWorkflow(
        object(), object(), resolver=resolver, step_runtime=StepRuntime()
    )
    bad_event = event({})
    bad_event.payload["__agentOrigin"] = {"kind": "agent", "version": 2}

    with pytest.raises(ValueError, match="unsupported workflow origin version"):
        await workflow.execute(bad_event, NativeStep())

    assert resolver.resolved == []


@pytest.mark.asyncio
async def test_legacy_agent_fields_resolve_when_versioned_origin_is_absent():
    resolver = RootResolver()

    class LegacyWorkflow(AgentWorkflow):
        async def run(self, workflow_event, step):
            return workflow_event.payload

    workflow = LegacyWorkflow(
        object(), object(), resolver=resolver, step_runtime=StepRuntime()
    )
    legacy_event = event({"task": 7})
    del legacy_event.payload["__agentOrigin"]

    assert await workflow.execute(legacy_event, NativeStep()) == {"task": 7}
    assert resolver.resolved == [("RootAgents", "tenant-7")]


@pytest.mark.asyncio
async def test_subclass_super_run_calls_each_defining_implementation_once():
    calls = []

    class ParentWorkflow(AgentWorkflow):
        async def run(self, workflow_event, step):
            calls.append(("parent", workflow_event.payload))
            return "parent"

    class ChildWorkflow(ParentWorkflow):
        async def run(self, workflow_event, step):
            calls.append(("child", workflow_event.payload))
            return f"child:{await super().run(workflow_event, step)}"

    workflow = ChildWorkflow(
        object(), object(), resolver=RootResolver(), step_runtime=StepRuntime()
    )

    assert await workflow.run(event({"task": 7}), NativeStep()) == "child:parent"
    assert calls == [("child", {"task": 7}), ("parent", {"task": 7})]


class Namespace:
    def __init__(self, stub: AgentStub) -> None:
        self.stub = stub
        self.ids: list[str] = []

    def idFromName(self, name: str) -> str:
        self.ids.append(name)
        return f"id:{name}"

    def get(self, id: str) -> AgentStub:
        assert id == "id:tenant-7"
        return self.stub


class NativeDecoratorStep(NativeStep):
    def __init__(self) -> None:
        super().__init__()
        self.names: list[str] = []

    def do(self, name: str):
        self.names.append(name)

        def decorate(callback):
            async def operation():
                return await callback()

            return operation

        return decorate


@pytest.mark.asyncio
async def test_native_wait_event_preserves_platform_fields():
    class WaitingStep:
        async def wait_for_event(self, name, event_type, *, timeout):
            assert (name, event_type, timeout) == ("review", "approval", "1 day")
            return {
                "payload": {"approved": True},
                "timestamp": 99,
                "attempt": 2,
            }

    result = await _WorkersWorkflowStepRuntime().wait_for_event(
        WaitingStep(), "review", "approval", "1 day"
    )

    assert result.payload == {"approved": True}
    assert result["timestamp"] == 99
    assert result["attempt"] == 2


class NativeAdapterWorkflow(AgentWorkflow):
    async def run(self, workflow_event, step):
        assert workflow_event.timestamp == 99
        assert workflow_event["attempt"] == 2
        await step.report_complete({"native": True})
        return workflow_event.payload


@pytest.mark.asyncio
async def test_native_entrypoint_uses_decorator_step_without_replacing_it():
    stub = AgentStub()
    namespace = Namespace(stub)
    env = type("Env", (), {"RootAgents": namespace})()
    workflow = NativeAdapterWorkflow(object(), env, now_ms=lambda: 1234)
    step = NativeDecoratorStep()

    result = await workflow.run(
        {
            "instanceId": "wf-1",
            "timestamp": 99,
            "attempt": 2,
            "payload": {
                "task": 7,
                "__workflowName": "REPORTS",
                "__agentOrigin": {
                    "kind": "agent",
                    "version": 1,
                    "binding": "RootAgents",
                    "name": "tenant-7",
                },
            },
        },
        step,
    )

    assert result == {"task": 7}
    assert step.native_api == "preserved"
    assert step.names == ["__agent_reportComplete_0"]
    assert namespace.ids == ["tenant-7"]
    assert stub.callbacks[0]["type"] == "complete"
    assert stub.destroyed == 1
