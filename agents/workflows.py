from __future__ import annotations

import asyncio
import base64
import binascii
import inspect
import json
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Any, Literal, NotRequired, Protocol, TypedDict, cast
from uuid import uuid4

from workers import WorkflowEntrypoint

from .core.utils import dumps_wire


type WorkflowStatus = Literal[
    "queued",
    "running",
    "paused",
    "errored",
    "terminated",
    "complete",
    "waiting",
    "waitingForPause",
    "unknown",
]
type WorkflowOrder = Literal["asc", "desc"]
type WorkflowMetadataScalar = str | int | float | bool
type WorkflowCallbackType = Literal["progress", "complete", "error", "event"]

_WORKFLOW_STATUSES = frozenset(
    {
        "queued",
        "running",
        "paused",
        "errored",
        "terminated",
        "complete",
        "waiting",
        "waitingForPause",
        "unknown",
    }
)
_TERMINAL_WORKFLOW_STATUSES = frozenset({"complete", "errored", "terminated"})
_WORKFLOW_UNSET = object()


class DefaultProgress(TypedDict):
    """Common progress fields; callers may include domain-specific keys."""

    step: NotRequired[str]
    status: NotRequired[Literal["pending", "running", "complete", "error"]]
    message: NotRequired[str]
    percent: NotRequired[int | float]


@dataclass(frozen=True, slots=True)
class AgentWorkflowEvent[ParamsT](Mapping[str, object]):
    """Workflow event with cleaned params and all native event fields."""

    instance_id: str
    payload: ParamsT
    native: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        native = dict(self.native)
        native["instanceId"] = self.instance_id
        native["payload"] = self.payload
        object.__setattr__(self, "native", MappingProxyType(native))

    def __getitem__(self, key: str) -> object:
        return self.native[key]

    def __iter__(self):
        return iter(self.native)

    def __len__(self) -> int:
        return len(self.native)

    def __getattr__(self, name: str) -> object:
        try:
            return self.native[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def with_payload[NewParamsT](
        self, payload: NewParamsT
    ) -> AgentWorkflowEvent[NewParamsT]:
        """Replace params without discarding platform-owned event fields."""

        return AgentWorkflowEvent(
            instance_id=self.instance_id,
            payload=payload,
            native=self.native,
        )


@dataclass(frozen=True, slots=True)
class WorkflowStepEvent(Mapping[str, object]):
    """Approval payload plus the complete native step-event result."""

    payload: Mapping[str, object]
    native: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        native = dict(self.native)
        native["payload"] = self.payload
        object.__setattr__(self, "native", MappingProxyType(native))

    def __getitem__(self, key: str) -> object:
        return self.native[key]

    def __iter__(self):
        return iter(self.native)

    def __len__(self) -> int:
        return len(self.native)


@dataclass(frozen=True, slots=True, kw_only=True)
class WaitForApprovalOptions:
    """Configure the durable event used by approval waiting."""

    step_name: str = "wait-for-approval"
    timeout: object = None
    event_type: str = "approval"


@dataclass(frozen=True, slots=True, kw_only=True)
class ApprovalEventPayload:
    """Standard payload sent by Workflow approval controls."""

    approved: bool
    reason: str | None = None
    metadata: Mapping[str, object] | None = None


class WorkflowRejectedError(Exception):
    """Report rejection of a Workflow waiting for Agent approval."""

    def __init__(self, reason: str | None = None, workflow_id: str | None = None):
        self.reason = reason
        self.workflow_id = workflow_id
        message = f"Workflow rejected: {reason}" if reason else "Workflow rejected"
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class WorkflowErrorInfo:
    """Describe an error returned by a native Workflow instance."""

    name: str
    message: str


@dataclass(frozen=True, slots=True)
class WorkflowInstanceStatus(Mapping[str, object]):
    """Parsed status fields plus the complete native status result."""

    status: WorkflowStatus
    error: WorkflowErrorInfo | None = None
    native: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_status(self.status)
        native = dict(self.native)
        native.setdefault("status", self.status)
        if self.error is not None:
            native.setdefault(
                "error",
                {"name": self.error.name, "message": self.error.message},
            )
        object.__setattr__(self, "native", MappingProxyType(native))

    def __getitem__(self, key: str) -> object:
        return self.native[key]

    def __iter__(self):
        return iter(self.native)

    def __len__(self) -> int:
        return len(self.native)


@dataclass(frozen=True, slots=True)
class WorkflowInfo:
    """Parsed view of one tracked Workflow row."""

    id: str
    workflow_id: str
    workflow_name: str
    status: WorkflowStatus
    metadata: dict[str, object] | None
    error: WorkflowErrorInfo | None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None


@dataclass(frozen=True, slots=True, kw_only=True)
class WorkflowQueryCriteria:
    """Filter and paginate tracked Workflows by a total created-time order."""

    status: WorkflowStatus | Sequence[WorkflowStatus] | None = None
    workflow_name: str | None = None
    metadata: Mapping[str, WorkflowMetadataScalar] | None = None
    limit: int = 50
    order_by: WorkflowOrder = "desc"
    cursor: str | None = None

    def __post_init__(self) -> None:
        if self.status is not None and not isinstance(self.status, str):
            object.__setattr__(self, "status", tuple(self.status))
        if self.metadata is not None:
            object.__setattr__(
                self,
                "metadata",
                MappingProxyType(dict(self.metadata)),
            )


@dataclass(frozen=True, slots=True)
class WorkflowPage:
    """One cursor page plus the total count before pagination."""

    workflows: tuple[WorkflowInfo, ...]
    total: int
    next_cursor: str | None


class WorkflowTrackingRow(TypedDict):
    """Persisted row shape shared with the TypeScript Workflow tracker."""

    id: str
    workflow_id: str
    workflow_name: str
    status: WorkflowStatus
    metadata: str | None
    error_name: str | None
    error_message: str | None
    created_at: int
    updated_at: int
    completed_at: int | None


@dataclass(frozen=True, slots=True, kw_only=True)
class RunWorkflowOptions:
    """Configure Workflow identity, tracking metadata, origin, and retention."""

    id: str | None = None
    metadata: Mapping[str, object] | None = None
    agent_binding: str | None = None
    retention: object = None

    def __post_init__(self) -> None:
        if self.metadata is not None:
            object.__setattr__(
                self,
                "metadata",
                MappingProxyType(dict(self.metadata)),
            )


@dataclass(frozen=True, slots=True)
class WorkflowEventPayload:
    """Describe an event sent to a native Workflow instance."""

    type: str
    payload: object

    def __post_init__(self) -> None:
        _require_non_empty(self.type, "workflow event type")


@dataclass(frozen=True, slots=True, kw_only=True)
class WorkflowProgressCallback:
    """Report non-durable progress from a Workflow to its Agent."""

    workflow_name: str
    workflow_id: str
    timestamp: int
    progress: object


@dataclass(frozen=True, slots=True, kw_only=True)
class WorkflowCompleteCallback:
    """Durably report successful Workflow completion."""

    workflow_name: str
    workflow_id: str
    timestamp: int
    result: object = _WORKFLOW_UNSET


@dataclass(frozen=True, slots=True, kw_only=True)
class WorkflowErrorCallback:
    """Durably report a Workflow error message."""

    workflow_name: str
    workflow_id: str
    timestamp: int
    error: str


@dataclass(frozen=True, slots=True, kw_only=True)
class WorkflowEventCallback:
    """Durably report an application event from a Workflow."""

    workflow_name: str
    workflow_id: str
    timestamp: int
    event: object


type WorkflowCallback = (
    WorkflowProgressCallback
    | WorkflowCompleteCallback
    | WorkflowErrorCallback
    | WorkflowEventCallback
)
type WorkflowCallbackHandler = Callable[..., object | Awaitable[object]]


@dataclass(frozen=True, slots=True, kw_only=True)
class WorkflowCallbackHandlers:
    """User hooks invoked after their corresponding ledger transition."""

    on_progress: WorkflowCallbackHandler | None = None
    on_complete: WorkflowCallbackHandler | None = None
    on_error: WorkflowCallbackHandler | None = None
    on_event: WorkflowCallbackHandler | None = None


type WorkflowControlAction = Literal["pause", "resume", "restart", "terminate"]
type _WorkflowOriginFactory = Callable[[str | None], AgentWorkflowOrigin]


class WorkflowRuntime(Protocol):
    """Semantic adapter over native Workers Workflow bindings and instances."""

    def has_binding(self, workflow_name: str) -> bool: ...

    async def create(
        self,
        workflow_name: str,
        workflow_id: str,
        params: Mapping[str, object],
        retention: object,
    ) -> str: ...

    async def status(
        self,
        workflow_name: str,
        workflow_id: str,
    ) -> WorkflowInstanceStatus: ...

    async def send_event(
        self,
        workflow_name: str,
        workflow_id: str,
        event: Mapping[str, object],
    ) -> None: ...

    async def control(
        self,
        action: WorkflowControlAction,
        workflow_name: str,
        workflow_id: str,
    ) -> None: ...


class _WorkersWorkflowRuntime:
    """Adapt native Workflow bindings through their documented narrow methods."""

    def __init__(self, env: object) -> None:
        self._env = env

    def has_binding(self, workflow_name: str) -> bool:
        try:
            self._binding(workflow_name)
        except ValueError:
            return False
        return True

    async def create(
        self,
        workflow_name: str,
        workflow_id: str,
        params: Mapping[str, object],
        retention: object,
    ) -> str:
        binding = self._binding(workflow_name)
        options: dict[str, object] = {
            "id": workflow_id,
            "params": dict(params),
        }
        if retention is not None:
            options["retention"] = retention
        created = await _maybe_await(binding.create(options))
        try:
            instance_id = _native_member(created, "id")
            return _require_non_empty(instance_id, "Workflow instance ID")
        finally:
            await _maybe_await(_destroy_native_proxy(created))

    async def status(
        self,
        workflow_name: str,
        workflow_id: str,
    ) -> WorkflowInstanceStatus:
        instance = await self._instance(workflow_name, workflow_id)
        try:
            return _coerce_instance_status(await _maybe_await(instance.status()))
        finally:
            await _maybe_await(_destroy_native_proxy(instance))

    async def send_event(
        self,
        workflow_name: str,
        workflow_id: str,
        event: Mapping[str, object],
    ) -> None:
        instance = await self._instance(workflow_name, workflow_id)
        try:
            await _maybe_await(instance.sendEvent(dict(event)))
        finally:
            await _maybe_await(_destroy_native_proxy(instance))

    async def control(
        self,
        action: WorkflowControlAction,
        workflow_name: str,
        workflow_id: str,
    ) -> None:
        instance = await self._instance(workflow_name, workflow_id)
        try:
            control = getattr(instance, action)
            await _maybe_await(control())
        finally:
            await _maybe_await(_destroy_native_proxy(instance))

    def _binding(self, workflow_name: str) -> Any:
        binding = _environment_member(self._env, workflow_name)
        if binding is None or not callable(getattr(binding, "create", None)):
            raise ValueError(
                f"Workflow binding '{workflow_name}' not found in environment"
            )
        if not callable(getattr(binding, "get", None)):
            raise ValueError(
                f"Workflow binding '{workflow_name}' not found in environment"
            )
        return binding

    async def _instance(self, workflow_name: str, workflow_id: str) -> Any:
        binding = self._binding(workflow_name)
        return await _maybe_await(binding.get(workflow_id))


class WorkflowSql(Protocol):
    """Minimal SQL service required by the Workflow ledger."""

    def execute(self, query: str, *params: object) -> list[dict[str, object]]: ...


class _CallableWorkflowSql:
    def __init__(
        self,
        execute: Callable[..., list[dict[str, object]]],
    ) -> None:
        self._execute = execute

    def execute(self, query: str, *params: object) -> list[dict[str, object]]:
        return self._execute(query, *params)


class WorkflowLedger:
    """Own tracking-row parsing, mutation, filtering, and pagination."""

    def __init__(
        self,
        sql: WorkflowSql,
        *,
        id_factory: Callable[[], str] | None = None,
        now_seconds: Callable[[], int] | None = None,
    ) -> None:
        self._sql = sql
        self._id_factory = id_factory or (lambda: uuid4().hex)
        self._now_seconds = now_seconds or (lambda: int(time.time()))

    def track(
        self,
        workflow_id: str,
        workflow_name: str,
        metadata: Mapping[str, object] | None = None,
    ) -> WorkflowInfo:
        """Insert a queued Workflow tracking row."""

        _require_non_empty(workflow_id, "workflow_id")
        _require_non_empty(workflow_name, "workflow_name")
        encoded_metadata = dumps_wire(dict(metadata)) if metadata is not None else None
        now = self._now_seconds()
        try:
            self._sql.execute(
                "INSERT INTO cf_agents_workflows "
                "(id, workflow_id, workflow_name, status, metadata, "
                "created_at, updated_at) VALUES (?, ?, ?, 'queued', ?, ?, ?)",
                self._id_factory(),
                workflow_id,
                workflow_name,
                encoded_metadata,
                now,
                now,
            )
        except Exception as error:
            if "UNIQUE constraint failed" in str(error):
                raise ValueError(
                    f'Workflow with ID "{workflow_id}" is already being tracked'
                ) from error
            raise
        tracked = self.get(workflow_id)
        if tracked is None:
            raise RuntimeError(f"failed to track Workflow {workflow_id!r}")
        return tracked

    def get(self, workflow_id: str) -> WorkflowInfo | None:
        """Return one tracked Workflow by native instance ID."""

        rows = self._sql.execute(
            "SELECT * FROM cf_agents_workflows WHERE workflow_id = ?",
            workflow_id,
        )
        return _row_to_info(rows[0]) if rows else None

    def list(self, criteria: WorkflowQueryCriteria | None = None) -> WorkflowPage:
        """Return a filtered keyset page and total matching row count."""

        criteria = criteria or WorkflowQueryCriteria()
        where, params = _criteria_where(criteria)
        count_rows = self._sql.execute(
            "SELECT COUNT(*) AS count FROM cf_agents_workflows" + where,
            *params,
        )
        total = cast(int, count_rows[0]["count"] if count_rows else 0)

        page_where = where
        page_params = list(params)
        if criteria.cursor is not None:
            created_at, workflow_id = _decode_cursor(criteria.cursor)
            comparator = ">" if criteria.order_by == "asc" else "<"
            page_where += (
                f" AND (created_at {comparator} ? OR "
                f"(created_at = ? AND workflow_id {comparator} ?))"
            )
            page_params.extend((created_at, created_at, workflow_id))

        direction = "ASC" if criteria.order_by == "asc" else "DESC"
        limit = max(1, min(criteria.limit, 100))
        rows = self._sql.execute(
            "SELECT * FROM cf_agents_workflows"
            + page_where
            + f" ORDER BY created_at {direction}, workflow_id {direction} LIMIT ?",
            *page_params,
            limit + 1,
        )
        has_more = len(rows) > limit
        selected = rows[:limit]
        workflows = tuple(_row_to_info(row) for row in selected)
        next_cursor = _encode_cursor(workflows[-1]) if has_more and workflows else None
        return WorkflowPage(workflows, total, next_cursor)

    def update_status(
        self,
        workflow_id: str,
        status: WorkflowInstanceStatus,
    ) -> bool:
        """Refresh status, error, and terminal timestamp from the native runtime."""

        if self.get(workflow_id) is None:
            return False
        now = self._now_seconds()
        completed_at = now if status.status in _TERMINAL_WORKFLOW_STATUSES else None
        self._sql.execute(
            "UPDATE cf_agents_workflows SET status = ?, error_name = ?, "
            "error_message = ?, updated_at = ?, completed_at = ? "
            "WHERE workflow_id = ?",
            status.status,
            None if status.error is None else status.error.name,
            None if status.error is None else status.error.message,
            now,
            completed_at,
            workflow_id,
        )
        return True

    def apply_callback(self, callback: WorkflowCallback) -> None:
        """Apply the callback transition before user code observes it."""

        now = self._now_seconds()
        if isinstance(callback, WorkflowProgressCallback):
            self._sql.execute(
                "UPDATE cf_agents_workflows SET status = 'running', updated_at = ? "
                "WHERE workflow_id = ? AND status IN ('queued', 'waiting')",
                now,
                callback.workflow_id,
            )
            return
        if isinstance(callback, WorkflowCompleteCallback):
            self._sql.execute(
                "UPDATE cf_agents_workflows SET status = 'complete', "
                "updated_at = ?, completed_at = ? WHERE workflow_id = ? "
                "AND status NOT IN ('terminated', 'paused')",
                now,
                now,
                callback.workflow_id,
            )
            return
        if isinstance(callback, WorkflowErrorCallback):
            self._sql.execute(
                "UPDATE cf_agents_workflows SET status = 'errored', "
                "updated_at = ?, completed_at = ?, error_name = 'WorkflowError', "
                "error_message = ? WHERE workflow_id = ? "
                "AND status NOT IN ('terminated', 'paused')",
                now,
                now,
                callback.error,
                callback.workflow_id,
            )

    def reset_for_restart(self, workflow_id: str) -> None:
        """Reset tracking fields for a fresh native restart."""

        now = self._now_seconds()
        self._sql.execute(
            "UPDATE cf_agents_workflows SET status = 'queued', created_at = ?, "
            "updated_at = ?, completed_at = NULL, error_name = NULL, "
            "error_message = NULL WHERE workflow_id = ?",
            now,
            now,
            workflow_id,
        )

    def delete(self, workflow_id: str) -> bool:
        """Delete one tracking row without controlling the native instance."""

        if self.get(workflow_id) is None:
            return False
        self._sql.execute(
            "DELETE FROM cf_agents_workflows WHERE workflow_id = ?",
            workflow_id,
        )
        return True

    def delete_many(
        self,
        *,
        status: WorkflowStatus | Sequence[WorkflowStatus] | None = None,
        workflow_name: str | None = None,
        metadata: Mapping[str, WorkflowMetadataScalar] | None = None,
        created_before: datetime | None = None,
    ) -> int:
        """Delete tracking rows matching exact filters and return their count."""

        criteria = WorkflowQueryCriteria(
            status=status,
            workflow_name=workflow_name,
            metadata=metadata,
        )
        where, params = _criteria_where(criteria)
        if created_before is not None:
            where += " AND created_at < ?"
            params.append(int(created_before.timestamp()))
        rows = self._sql.execute(
            "SELECT COUNT(*) AS count FROM cf_agents_workflows" + where,
            *params,
        )
        count = cast(int, rows[0]["count"] if rows else 0)
        if count:
            self._sql.execute("DELETE FROM cf_agents_workflows" + where, *params)
        return count

    def migrate_binding(self, old_name: str, new_name: str) -> int:
        """Rename tracked binding references after runtime validation by the caller."""

        rows = self._sql.execute(
            "SELECT COUNT(*) AS count FROM cf_agents_workflows WHERE workflow_name = ?",
            old_name,
        )
        count = cast(int, rows[0]["count"] if rows else 0)
        if count:
            self._sql.execute(
                "UPDATE cf_agents_workflows SET workflow_name = ? "
                "WHERE workflow_name = ?",
                new_name,
                old_name,
            )
        return count


class WorkflowOperations:
    """Compose Workflow runtime operations with durable Agent-local tracking."""

    def __init__(
        self,
        ledger: WorkflowLedger,
        runtime: WorkflowRuntime,
        origin: AgentWorkflowOrigin | _WorkflowOriginFactory,
        *,
        callbacks: WorkflowCallbackHandlers | None = None,
        emit: Callable[[str, object], object | Awaitable[object]] | None = None,
        id_factory: Callable[[], str] | None = None,
        sleep: Callable[[float], Awaitable[object]] = asyncio.sleep,
    ) -> None:
        self._ledger = ledger
        self._runtime = runtime
        self._origin = origin
        self._callbacks = callbacks or WorkflowCallbackHandlers()
        self._emit = emit
        self._id_factory = id_factory or (lambda: uuid4().hex)
        self._sleep = sleep

    @classmethod
    def for_agent(
        cls,
        sql: WorkflowSql | Callable[..., list[dict[str, object]]],
        env: object,
        origin: AgentWorkflowOrigin | _WorkflowOriginFactory,
        *,
        callbacks: WorkflowCallbackHandlers | None = None,
        emit: Callable[[str, object], object | Awaitable[object]] | None = None,
        id_factory: Callable[[], str] | None = None,
        sleep: Callable[[float], Awaitable[object]] = asyncio.sleep,
    ) -> WorkflowOperations:
        """Build the production Agent composition over native Workers bindings."""

        if callable(getattr(sql, "execute", None)):
            ledger_sql = cast(WorkflowSql, sql)
        else:
            ledger_sql = _CallableWorkflowSql(
                cast(Callable[..., list[dict[str, object]]], sql)
            )
        return cls(
            WorkflowLedger(ledger_sql),
            _WorkersWorkflowRuntime(env),
            origin,
            callbacks=callbacks,
            emit=emit,
            id_factory=id_factory,
            sleep=sleep,
        )

    async def run_workflow(
        self,
        workflow_name: str,
        params: Mapping[str, object],
        options: RunWorkflowOptions | None = None,
    ) -> str:
        """Create a native Workflow and track it in the originating Agent."""

        self._require_binding(workflow_name)
        options = options or RunWorkflowOptions()
        if callable(self._origin):
            origin = cast(_WorkflowOriginFactory, self._origin)(options.agent_binding)
        else:
            origin = _origin_with_binding(
                cast(AgentWorkflowOrigin, self._origin),
                options.agent_binding,
            )
        workflow_id = options.id or f"wf_{self._id_factory()}"
        augmented = dict(params)
        augmented.update(
            {
                "__agentName": _origin_name(origin),
                "__agentBinding": _origin_binding(origin),
                "__workflowName": workflow_name,
                "__agentOrigin": encode_workflow_origin(origin),
            }
        )
        instance_id = await self._runtime.create(
            workflow_name,
            workflow_id,
            augmented,
            options.retention,
        )
        self._ledger.track(instance_id, workflow_name, options.metadata)
        await self._emit_event(
            "workflow:start",
            {"workflowId": instance_id, "workflowName": workflow_name},
        )
        return instance_id

    def get_workflow(self, workflow_id: str) -> WorkflowInfo | None:
        """Return one Agent-local Workflow tracking record."""

        return self._ledger.get(workflow_id)

    def get_workflows(
        self,
        criteria: WorkflowQueryCriteria | None = None,
    ) -> WorkflowPage:
        """Return one Agent-local filtered Workflow page."""

        return self._ledger.list(criteria)

    async def get_workflow_status(
        self,
        workflow_name: str,
        workflow_id: str,
    ) -> WorkflowInstanceStatus:
        """Refresh and return native instance status."""

        self._require_binding(workflow_name)
        status = await self._runtime.status(workflow_name, workflow_id)
        self._ledger.update_status(workflow_id, status)
        return status

    async def send_workflow_event(
        self,
        workflow_name: str,
        workflow_id: str,
        event: WorkflowEventPayload,
    ) -> None:
        """Send an event to a native Workflow instance."""

        self._require_binding(workflow_name)
        await self._retry_runtime_call(
            lambda: self._runtime.send_event(
                workflow_name,
                workflow_id,
                {"type": event.type, "payload": event.payload},
            )
        )
        await self._emit_event(
            "workflow:event",
            {"workflowId": workflow_id, "eventType": event.type},
        )

    async def approve_workflow(
        self,
        workflow_id: str,
        *,
        reason: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        """Send the standard approval event to a tracked Workflow."""

        info = self._require_tracked(workflow_id)
        payload: dict[str, object] = {"approved": True}
        if reason is not None:
            payload["reason"] = reason
        if metadata is not None:
            payload["metadata"] = dict(metadata)
        await self.send_workflow_event(
            info.workflow_name,
            workflow_id,
            WorkflowEventPayload(type="approval", payload=payload),
        )
        event: dict[str, object] = {"workflowId": workflow_id}
        if reason is not None:
            event["reason"] = reason
        await self._emit_event("workflow:approved", event)

    async def reject_workflow(
        self,
        workflow_id: str,
        *,
        reason: str | None = None,
    ) -> None:
        """Send the standard rejection event to a tracked Workflow."""

        info = self._require_tracked(workflow_id)
        payload: dict[str, object] = {"approved": False}
        if reason is not None:
            payload["reason"] = reason
        await self.send_workflow_event(
            info.workflow_name,
            workflow_id,
            WorkflowEventPayload(type="approval", payload=payload),
        )
        event: dict[str, object] = {"workflowId": workflow_id}
        if reason is not None:
            event["reason"] = reason
        await self._emit_event("workflow:rejected", event)

    async def pause_workflow(self, workflow_id: str) -> None:
        """Pause a tracked native Workflow and refresh tracking."""

        await self._control("pause", workflow_id)

    async def resume_workflow(self, workflow_id: str) -> None:
        """Resume a tracked native Workflow and refresh tracking."""

        await self._control("resume", workflow_id)

    async def terminate_workflow(self, workflow_id: str) -> None:
        """Terminate a tracked native Workflow and refresh tracking."""

        await self._control("terminate", workflow_id)

    async def restart_workflow(
        self,
        workflow_id: str,
        *,
        reset_tracking: bool = True,
    ) -> None:
        """Restart a tracked Workflow, optionally preserving its creation time."""

        info = self._require_tracked(workflow_id)
        self._require_binding(info.workflow_name)
        await self._retry_runtime_call(
            lambda: self._runtime.control("restart", info.workflow_name, workflow_id)
        )
        if reset_tracking:
            self._ledger.reset_for_restart(workflow_id)
        else:
            status = await self._runtime.status(info.workflow_name, workflow_id)
            self._ledger.update_status(workflow_id, status)
        await self._emit_event(
            "workflow:restarted",
            {"workflowId": workflow_id, "workflowName": info.workflow_name},
        )

    async def handle_callback(
        self,
        callback: WorkflowCallback | Mapping[str, object],
    ) -> None:
        """Apply a Workflow callback and then invoke its user hook."""

        if isinstance(callback, Mapping):
            callback = decode_workflow_callback(callback)
        self._ledger.apply_callback(callback)
        if isinstance(callback, WorkflowProgressCallback):
            await _call_optional(
                self._callbacks.on_progress,
                callback.workflow_name,
                callback.workflow_id,
                callback.progress,
            )
        elif isinstance(callback, WorkflowCompleteCallback):
            await _call_optional(
                self._callbacks.on_complete,
                callback.workflow_name,
                callback.workflow_id,
                None if callback.result is _WORKFLOW_UNSET else callback.result,
            )
        elif isinstance(callback, WorkflowErrorCallback):
            await _call_optional(
                self._callbacks.on_error,
                callback.workflow_name,
                callback.workflow_id,
                callback.error,
            )
        else:
            await _call_optional(
                self._callbacks.on_event,
                callback.workflow_name,
                callback.workflow_id,
                callback.event,
            )

    def delete_workflow(self, workflow_id: str) -> bool:
        """Delete one Agent-local tracking row."""

        return self._ledger.delete(workflow_id)

    def delete_workflows(
        self,
        *,
        status: WorkflowStatus | Sequence[WorkflowStatus] | None = None,
        workflow_name: str | None = None,
        metadata: Mapping[str, WorkflowMetadataScalar] | None = None,
        created_before: datetime | None = None,
    ) -> int:
        """Delete Agent-local tracking rows matching exact criteria."""

        return self._ledger.delete_many(
            status=status,
            workflow_name=workflow_name,
            metadata=metadata,
            created_before=created_before,
        )

    def migrate_workflow_binding(self, old_name: str, new_name: str) -> int:
        """Validate and rename a binding in retained tracking rows."""

        self._require_binding(new_name)
        return self._ledger.migrate_binding(old_name, new_name)

    async def _control(
        self,
        action: WorkflowControlAction,
        workflow_id: str,
    ) -> None:
        info = self._require_tracked(workflow_id)
        self._require_binding(info.workflow_name)
        await self._retry_runtime_call(
            lambda: self._runtime.control(
                action,
                info.workflow_name,
                workflow_id,
            )
        )
        status = await self._runtime.status(info.workflow_name, workflow_id)
        self._ledger.update_status(workflow_id, status)
        await self._emit_event(
            f"workflow:{'terminated' if action == 'terminate' else action + 'd'}",
            {"workflowId": workflow_id, "workflowName": info.workflow_name},
        )

    def _require_tracked(self, workflow_id: str) -> WorkflowInfo:
        info = self._ledger.get(workflow_id)
        if info is None:
            raise ValueError(f"Workflow {workflow_id} not found in tracking table")
        return info

    def _require_binding(self, workflow_name: str) -> None:
        if not self._runtime.has_binding(workflow_name):
            raise ValueError(
                f"Workflow binding '{workflow_name}' not found in environment"
            )

    async def _emit_event(self, event: str, payload: object) -> None:
        await _call_optional(self._emit, event, payload)

    async def _retry_runtime_call[ResultT](
        self,
        operation: Callable[[], Awaitable[ResultT]],
    ) -> ResultT:
        for attempt in range(3):
            try:
                return await operation()
            except BaseException as error:
                if attempt == 2 or not _is_retryable_workflow_error(error):
                    raise
                await self._sleep(0.2 * (2**attempt))
        raise AssertionError("unreachable Workflow retry state")


def _origin_with_binding(
    origin: AgentWorkflowOrigin,
    binding: str | None,
) -> AgentWorkflowOrigin:
    if binding is None:
        return origin
    if isinstance(origin, AgentWorkflowRootOrigin):
        return AgentWorkflowRootOrigin(binding=binding, name=origin.name)
    return AgentWorkflowFacetOrigin(root_binding=binding, path=origin.path)


def _origin_name(origin: AgentWorkflowOrigin) -> str:
    if isinstance(origin, AgentWorkflowRootOrigin):
        return origin.name
    return origin.path[-1].name


def _origin_binding(origin: AgentWorkflowOrigin) -> str:
    if isinstance(origin, AgentWorkflowRootOrigin):
        return origin.binding
    return origin.root_binding


async def _call_optional(
    callback: Callable[..., object | Awaitable[object]] | None,
    *args: object,
) -> object | None:
    if callback is None:
        return None
    result = callback(*args)
    return await result if inspect.isawaitable(result) else result


async def _maybe_await(value: object) -> Any:
    return await value if inspect.isawaitable(value) else value


def _environment_member(env: object, name: str) -> object | None:
    if isinstance(env, Mapping):
        return env.get(name)
    return getattr(env, name, None)


def _native_member(value: object, name: str) -> object:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _destroy_native_proxy(value: object) -> object:
    destroy = getattr(value, "destroy", None)
    return destroy() if callable(destroy) else None


def _coerce_instance_status(value: object) -> WorkflowInstanceStatus:
    if isinstance(value, WorkflowInstanceStatus):
        return value
    if not isinstance(value, Mapping):
        raise ValueError("native Workflow status must be an object")
    status = value.get("status")
    _validate_status(status)
    error = None
    raw_error = value.get("error")
    if isinstance(raw_error, Mapping):
        name = raw_error.get("name")
        message = raw_error.get("message")
        if isinstance(name, str) and isinstance(message, str):
            error = WorkflowErrorInfo(name=name, message=message)
    return WorkflowInstanceStatus(
        status=cast(WorkflowStatus, status),
        error=error,
        native=dict(value),
    )


def _is_retryable_workflow_error(error: BaseException) -> bool:
    return (
        bool(getattr(error, "retryable", False))
        and not bool(getattr(error, "overloaded", False))
        and "Durable Object is overloaded" not in str(error)
    )


def _criteria_where(
    criteria: WorkflowQueryCriteria,
) -> tuple[str, list[object]]:
    if type(criteria.limit) is not int:
        raise TypeError("workflow query limit must be an integer")
    if criteria.order_by not in ("asc", "desc"):
        raise ValueError("workflow query order_by must be 'asc' or 'desc'")
    clauses = ["1=1"]
    params: list[object] = []
    statuses = _normalize_statuses(criteria.status)
    if statuses:
        clauses.append("status IN (" + ", ".join("?" for _ in statuses) + ")")
        params.extend(statuses)
    if criteria.workflow_name is not None:
        clauses.append("workflow_name = ?")
        params.append(criteria.workflow_name)
    if criteria.metadata is not None:
        for key, value in criteria.metadata.items():
            if not isinstance(key, str):
                raise TypeError("workflow metadata filter keys must be strings")
            if isinstance(value, bool):
                sql_value: object = int(value)
            elif isinstance(value, (str, int, float)):
                sql_value = value
            else:
                raise TypeError("workflow metadata filters must contain scalar values")
            clauses.append(
                "EXISTS (SELECT 1 FROM json_each(metadata) AS workflow_metadata "
                "WHERE workflow_metadata.key = ? AND workflow_metadata.value = ?)"
            )
            params.extend((key, sql_value))
    return " WHERE " + " AND ".join(clauses), params


def _normalize_statuses(
    status: WorkflowStatus | Sequence[WorkflowStatus] | None,
) -> tuple[WorkflowStatus, ...]:
    if status is None:
        return ()
    statuses = (status,) if isinstance(status, str) else tuple(status)
    for item in statuses:
        _validate_status(item)
    return cast(tuple[WorkflowStatus, ...], statuses)


def _validate_status(status: object) -> None:
    if status not in _WORKFLOW_STATUSES:
        raise ValueError(f"unknown Workflow status: {status!r}")


def _row_to_info(row: Mapping[str, object]) -> WorkflowInfo:
    metadata = None
    if row["metadata"] is not None:
        try:
            parsed = json.loads(cast(str, row["metadata"]))
        except (TypeError, ValueError) as error:
            raise ValueError("persisted Workflow metadata is invalid JSON") from error
        if not isinstance(parsed, dict):
            raise ValueError("persisted Workflow metadata must be an object")
        metadata = parsed
    error = None
    if row["error_name"] is not None:
        error = WorkflowErrorInfo(
            name=cast(str, row["error_name"]),
            message=cast(str, row["error_message"] or ""),
        )
    status = cast(WorkflowStatus, row["status"])
    _validate_status(status)
    return WorkflowInfo(
        id=cast(str, row["id"]),
        workflow_id=cast(str, row["workflow_id"]),
        workflow_name=cast(str, row["workflow_name"]),
        status=status,
        metadata=metadata,
        error=error,
        created_at=datetime.fromtimestamp(cast(int, row["created_at"]), UTC),
        updated_at=datetime.fromtimestamp(cast(int, row["updated_at"]), UTC),
        completed_at=(
            None
            if row["completed_at"] is None
            else datetime.fromtimestamp(cast(int, row["completed_at"]), UTC)
        ),
    )


def _encode_cursor(workflow: WorkflowInfo) -> str:
    payload = dumps_wire(
        {"c": int(workflow.created_at.timestamp()), "i": workflow.workflow_id}
    )
    return base64.b64encode(payload.encode()).decode()


def _decode_cursor(cursor: str) -> tuple[int, str]:
    try:
        raw = base64.b64decode(cursor, validate=True)
        value = json.loads(raw)
        if (
            not isinstance(value, dict)
            or set(value) != {"c", "i"}
            or type(value["c"]) is not int
            or not isinstance(value["i"], str)
        ):
            raise ValueError
    except (
        binascii.Error,
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
    ) as error:
        raise ValueError(
            "Invalid pagination cursor. The cursor may be malformed or corrupted."
        ) from error
    return value["c"], value["i"]


def _require_non_empty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


@dataclass(frozen=True, slots=True)
class AgentWorkflowPathStep:
    """Identify one root-first Agent path segment."""

    class_name: str
    name: str

    def __post_init__(self) -> None:
        if not self.class_name or not self.name:
            raise ValueError("workflow origin path fields must be non-empty strings")


@dataclass(frozen=True, slots=True)
class AgentWorkflowRootOrigin:
    """Persist the named root Agent that started a Workflow."""

    binding: str
    name: str

    def __post_init__(self) -> None:
        if not self.binding or not self.name:
            raise ValueError("root workflow origin fields must be non-empty strings")


@dataclass(frozen=True, slots=True)
class AgentWorkflowFacetOrigin:
    """Persist the root binding and root-first path of an Agent facet."""

    root_binding: str
    path: tuple[AgentWorkflowPathStep, ...]

    def __post_init__(self) -> None:
        if not self.root_binding:
            raise ValueError("facet workflow root binding must be non-empty")
        if not self.path:
            raise ValueError("facet workflow origin requires a non-empty path")
        object.__setattr__(self, "path", tuple(self.path))


type AgentWorkflowOrigin = AgentWorkflowRootOrigin | AgentWorkflowFacetOrigin


class WorkflowAgentResolver(Protocol):
    """Resolve root Agent RPC and invoke a root-first facet path."""

    def resolve_root(self, binding: str, name: str) -> object | Awaitable[object]: ...

    def invoke_agent_path(
        self,
        root: object,
        path: tuple[AgentWorkflowPathStep, ...],
        method: str,
        args: tuple[object, ...],
    ) -> object | Awaitable[object]: ...

    def release(self, agent: object) -> object | Awaitable[object]: ...


class AgentWorkflowStep(Protocol):
    """Agent helpers installed alongside a native Workflow step's full API."""

    async def report_complete(self, result: object = _WORKFLOW_UNSET) -> None: ...

    async def report_error(self, error: BaseException | str) -> None: ...

    async def send_event(self, event: object) -> None: ...

    async def update_agent_state(self, state: object) -> None: ...

    async def merge_agent_state(self, state: Mapping[str, object]) -> None: ...

    async def reset_agent_state(self) -> None: ...


class WorkflowStepRuntime(Protocol):
    """Adapt native durable-step calls without fixing unavailable signatures."""

    def install(
        self,
        step: object,
        name: str,
        callback: Callable[..., Awaitable[object]],
    ) -> None: ...

    async def durable(
        self,
        step: object,
        name: str,
        callback: Callable[[], Awaitable[object]],
    ) -> object: ...

    async def wait_for_event(
        self,
        step: object,
        step_name: str,
        event_type: str,
        timeout: object,
    ) -> WorkflowStepEvent: ...

    def release_event(self, event: WorkflowStepEvent) -> None: ...


def encode_workflow_origin(origin: AgentWorkflowOrigin) -> dict[str, object]:
    """Encode the version-1 cross-runtime Workflow origin."""

    if isinstance(origin, AgentWorkflowRootOrigin):
        return {
            "kind": "agent",
            "version": 1,
            "binding": origin.binding,
            "name": origin.name,
        }
    return {
        "kind": "facet",
        "version": 1,
        "rootBinding": origin.root_binding,
        "path": [
            {"className": step.class_name, "name": step.name} for step in origin.path
        ],
    }


def decode_workflow_origin(value: object) -> AgentWorkflowOrigin:
    """Decode a persisted origin and reject versions this build cannot read."""

    if not isinstance(value, Mapping):
        raise ValueError("workflow origin must be an object")
    version = value.get("version")
    if type(version) is not int or version != 1:
        raise ValueError(f"unsupported workflow origin version: {version!r}")
    kind = value.get("kind")
    if kind == "agent":
        return AgentWorkflowRootOrigin(
            binding=_required_string(value, "binding"),
            name=_required_string(value, "name"),
        )
    if kind == "facet":
        raw_path = value.get("path")
        if not isinstance(raw_path, list) or not raw_path:
            raise ValueError("facet workflow origin requires a non-empty path")
        path = []
        for item in raw_path:
            if not isinstance(item, Mapping):
                raise ValueError("workflow origin path entries must be objects")
            path.append(
                AgentWorkflowPathStep(
                    class_name=_required_string(item, "className"),
                    name=_required_string(item, "name"),
                )
            )
        return AgentWorkflowFacetOrigin(
            root_binding=_required_string(value, "rootBinding"),
            path=tuple(path),
        )
    raise ValueError(f"unknown workflow origin kind: {kind!r}")


def _required_string(value: Mapping[Any, object], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise ValueError(f"workflow origin {key} must be a non-empty string")
    return item


class RpcOnlyAgentStub:
    """Forward facet Agent methods through a root path RPC aperture."""

    __slots__ = ("_path", "_resolver", "_root")

    def __init__(
        self,
        root: object,
        path: tuple[AgentWorkflowPathStep, ...],
        resolver: WorkflowAgentResolver,
    ) -> None:
        self._root = root
        self._path = path
        self._resolver = resolver

    def fetch(self, *_args: object, **_kwargs: object) -> None:
        raise RuntimeError(
            "AgentWorkflow.agent for facet origins is an RPC-only stub; "
            "fetch() is not supported"
        )

    def __getattr__(self, method: str):
        if method.startswith("__"):
            raise AttributeError(method)

        async def invoke(*args: object) -> object:
            result = self._resolver.invoke_agent_path(
                self._root,
                self._path,
                method,
                args,
            )
            if isinstance(result, Awaitable):
                return await result
            return result

        return invoke


async def resolve_workflow_agent(
    origin: AgentWorkflowOrigin,
    resolver: WorkflowAgentResolver,
) -> object:
    """Resolve a root stub or an RPC-only facet stub from persisted identity."""

    if isinstance(origin, AgentWorkflowRootOrigin):
        result = resolver.resolve_root(origin.binding, origin.name)
        return await result if isinstance(result, Awaitable) else result
    root = origin.path[0]
    result = resolver.resolve_root(origin.root_binding, root.name)
    root_stub = await result if isinstance(result, Awaitable) else result
    return RpcOnlyAgentStub(root_stub, origin.path, resolver)


def encode_workflow_callback(callback: WorkflowCallback) -> dict[str, object]:
    """Encode a Workflow callback using the cross-runtime field names."""

    wire: dict[str, object] = {
        "workflowName": callback.workflow_name,
        "workflowId": callback.workflow_id,
        "timestamp": callback.timestamp,
    }
    if isinstance(callback, WorkflowProgressCallback):
        wire.update(type="progress", progress=callback.progress)
    elif isinstance(callback, WorkflowCompleteCallback):
        wire["type"] = "complete"
        if callback.result is not _WORKFLOW_UNSET:
            wire["result"] = callback.result
    elif isinstance(callback, WorkflowErrorCallback):
        wire.update(type="error", error=callback.error)
    else:
        wire.update(type="event", event=callback.event)
    return wire


def decode_workflow_callback(value: object) -> WorkflowCallback:
    """Decode and validate a callback received over Agent RPC."""

    if not isinstance(value, Mapping):
        raise ValueError("Workflow callback must be an object")
    common = {
        "workflow_name": _required_string(value, "workflowName"),
        "workflow_id": _required_string(value, "workflowId"),
        "timestamp": _required_timestamp(value.get("timestamp")),
    }
    callback_type = value.get("type")
    if callback_type == "progress" and "progress" in value:
        return WorkflowProgressCallback(**common, progress=value["progress"])
    if callback_type == "complete":
        result = value["result"] if "result" in value else _WORKFLOW_UNSET
        return WorkflowCompleteCallback(**common, result=result)
    if callback_type == "error" and isinstance(value.get("error"), str):
        return WorkflowErrorCallback(**common, error=cast(str, value["error"]))
    if callback_type == "event" and "event" in value:
        return WorkflowEventCallback(**common, event=value["event"])
    raise ValueError(f"invalid Workflow callback type: {callback_type!r}")


def _required_timestamp(value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("Workflow callback timestamp must be a non-negative integer")
    return value


class _WorkersWorkflowAgentResolver:
    def __init__(self, env: object) -> None:
        self._env = env

    async def resolve_root(self, binding: str, name: str) -> object:
        namespace = getattr(self._env, binding, None)
        if namespace is None:
            raise ValueError(f"Agent binding '{binding}' not found in environment")
        stub = namespace.get(namespace.idFromName(name))
        try:
            initialize = getattr(stub, "__unsafe_ensureInitialized", None)
            if callable(initialize):
                for attempt in range(3):
                    try:
                        result = initialize()
                        if inspect.isawaitable(result):
                            await result
                        break
                    except BaseException as error:
                        if attempt == 2 or not _is_retryable_workflow_error(error):
                            raise
                        await asyncio.sleep(0.2 * (2**attempt))
            return stub
        except BaseException:
            try:
                cleanup = _destroy_native_proxy(stub)
                if inspect.isawaitable(cleanup):
                    await cleanup
            except BaseException:
                pass
            raise

    def invoke_agent_path(
        self,
        root: object,
        path: tuple[AgentWorkflowPathStep, ...],
        method: str,
        args: tuple[object, ...],
    ) -> object:
        invoke = getattr(root, "_cf_invokeAgentPath")
        wire_path = [{"className": step.class_name, "name": step.name} for step in path]
        return invoke(wire_path, method, list(args))

    def release(self, agent: object) -> object:
        return _destroy_native_proxy(agent)


class _WorkersWorkflowStepRuntime:
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
        decorate = getattr(step, "do")
        operation = decorate(name)(callback)
        result = operation()
        return await result if inspect.isawaitable(result) else result

    async def wait_for_event(
        self,
        step: object,
        step_name: str,
        event_type: str,
        timeout: object,
    ) -> WorkflowStepEvent:
        wait = getattr(step, "wait_for_event")
        if timeout is None:
            result = wait(step_name, event_type)
        else:
            result = wait(step_name, event_type, timeout=timeout)
        if inspect.isawaitable(result):
            result = await result
        if not isinstance(result, Mapping) or not isinstance(
            result.get("payload"), Mapping
        ):
            raise ValueError("native Workflow event returned an invalid payload")
        return WorkflowStepEvent(
            payload=cast(Mapping[str, object], result["payload"]),
            native=dict(result),
        )

    def release_event(self, event: WorkflowStepEvent) -> None:
        return None


def _coerce_workflow_event(
    event: AgentWorkflowEvent[Mapping[str, object]] | Mapping[str, object],
) -> AgentWorkflowEvent[Mapping[str, object]]:
    if isinstance(event, AgentWorkflowEvent):
        if not isinstance(event.payload, Mapping):
            raise ValueError("AgentWorkflow event payload must be an object")
        return event
    payload = event.get("payload")
    instance_id = event.get("instanceId")
    if not isinstance(payload, Mapping) or not isinstance(instance_id, str):
        raise ValueError("native AgentWorkflow event is invalid")
    return AgentWorkflowEvent(
        instance_id=instance_id,
        payload=payload,
        native=dict(event),
    )


def _workflow_event_origin(payload: Mapping[str, object]) -> AgentWorkflowOrigin:
    raw_origin = payload.get("__agentOrigin", _WORKFLOW_UNSET)
    if raw_origin is not _WORKFLOW_UNSET and raw_origin is not None:
        return decode_workflow_origin(raw_origin)
    return AgentWorkflowRootOrigin(
        binding=_required_string(payload, "__agentBinding"),
        name=_required_string(payload, "__agentName"),
    )


def _workflow_user_run(cls: type) -> tuple[type, Callable[..., object]]:
    for defining_class in cls.__mro__:
        user_run = defining_class.__dict__.get("_agent_workflow_user_run")
        if callable(user_run):
            return defining_class, user_run
    raise NotImplementedError("AgentWorkflow subclasses must define run()")


class AgentWorkflow(WorkflowEntrypoint):
    """Native Workers Workflow entry point with originating-Agent integration."""

    def __init__(
        self,
        ctx: object,
        env: object,
        *,
        resolver: WorkflowAgentResolver | None = None,
        step_runtime: WorkflowStepRuntime | None = None,
        now_ms: Callable[[], int] | None = None,
    ) -> None:
        super().__init__(ctx, env)
        self._resolver = resolver or _WorkersWorkflowAgentResolver(env)
        self._step_runtime = step_runtime or _WorkersWorkflowStepRuntime()
        self._now_ms = now_ms or (lambda: int(time.time() * 1_000))
        self._agent: object | None = None
        self._workflow_id: str | None = None
        self._workflow_name: str | None = None
        self._error_reported = False
        self._step_counter = 0
        self._active_run_class: type | None = None

    def __init_subclass__(cls, **kwargs: object) -> None:
        user_run = cls.__dict__.get("run")
        if user_run is not None:
            setattr(cls, "_agent_workflow_user_run", user_run)

            async def run(self: AgentWorkflow, event: object, step: object) -> object:
                if self._active_run_class is not None:
                    result = user_run(self, event, step)
                    return await result if inspect.isawaitable(result) else result
                return await self.execute(
                    cast(Mapping[str, object], event),
                    step,
                    _run_definition=(cls, user_run),
                )

            setattr(cls, "run", run)
        super().__init_subclass__(**kwargs)

    @property
    def agent(self) -> Any:
        """Return the originating Agent RPC stub during Workflow execution."""

        if self._agent is None:
            raise RuntimeError(
                "Agent not initialized; access AgentWorkflow.agent only inside run()"
            )
        return self._agent

    @property
    def workflow_id(self) -> str:
        """Return the current native Workflow instance ID."""

        if self._workflow_id is None:
            raise RuntimeError("Workflow is not initialized")
        return self._workflow_id

    @property
    def workflow_name(self) -> str:
        """Return the current Workflow binding name."""

        if self._workflow_name is None:
            raise RuntimeError("Workflow is not initialized")
        return self._workflow_name

    async def execute[ResultT](
        self,
        event: AgentWorkflowEvent[Mapping[str, object]] | Mapping[str, object],
        step: object,
        *,
        _run_definition: tuple[type, Callable[..., object]] | None = None,
    ) -> ResultT:
        """Resolve the origin, extend the native step, and invoke user code."""

        event = _coerce_workflow_event(event)
        payload = dict(event.payload)
        origin = _workflow_event_origin(payload)
        workflow_name = _required_string(payload, "__workflowName")
        payload.pop("__agentOrigin", None)
        payload.pop("__workflowName", None)
        payload.pop("__agentName", None)
        payload.pop("__agentBinding", None)

        agent = await resolve_workflow_agent(origin, self._resolver)
        previous_run_class = self._active_run_class
        try:
            self._agent = agent
            self._workflow_id = _require_non_empty(
                event.instance_id, "workflow instance ID"
            )
            self._workflow_name = workflow_name
            self._error_reported = False
            self._step_counter = 0
            cleaned = event.with_payload(payload)
            defining_class, user_run = _run_definition or _workflow_user_run(type(self))
            self._active_run_class = defining_class
            self._install_step_helpers(step)
            extended = self.extend_step(step, cleaned)
            if extended is not step:
                raise TypeError("extend_step must preserve the native Workflow step")
            try:
                result = user_run(self, cleaned, step)
                if inspect.isawaitable(result):
                    result = await result
                return cast(ResultT, result)
            except BaseException as error:
                await self._auto_report_error(error)
                raise
        finally:
            self._active_run_class = previous_run_class
            self._agent = None
            self._workflow_id = None
            self._workflow_name = None
            release_target = (
                agent._root if isinstance(agent, RpcOnlyAgentStub) else agent
            )
            try:
                released = self._resolver.release(release_target)
                if inspect.isawaitable(released):
                    await released
            except BaseException:
                pass

    def extend_step(self, step: object, event: AgentWorkflowEvent[object]) -> object:
        """Add domain helpers while preserving the native step object."""

        return step

    async def notify_agent(self, callback: WorkflowCallback) -> None:
        """Send one callback to the originating Agent over RPC."""

        await _invoke_agent(
            self.agent,
            "_workflow_handleCallback",
            encode_workflow_callback(callback),
        )

    async def report_progress(self, progress: object) -> None:
        """Report non-durable progress to the originating Agent."""

        await self.notify_agent(
            WorkflowProgressCallback(
                workflow_name=self.workflow_name,
                workflow_id=self.workflow_id,
                timestamp=self._now_ms(),
                progress=progress,
            )
        )

    async def broadcast_to_clients(self, message: object) -> None:
        """Broadcast a non-durable message through the originating Agent."""

        await _invoke_agent(self.agent, "_workflow_broadcast", message)

    async def wait_for_approval(
        self,
        step: object,
        options: WaitForApprovalOptions | None = None,
    ) -> object:
        """Wait for approval metadata or durably report and raise rejection."""

        options = options or WaitForApprovalOptions()
        event = await self._step_runtime.wait_for_event(
            step,
            options.step_name,
            options.event_type,
            options.timeout,
        )
        try:
            approved = event.payload.get("approved")
            if type(approved) is not bool:
                raise ValueError("Workflow approval event requires approved: bool")
            if not approved:
                reason = event.payload.get("reason")
                if reason is not None and not isinstance(reason, str):
                    raise ValueError("Workflow rejection reason must be a string")
                report_error = getattr(step, "report_error")
                await report_error(reason or "Workflow rejected")
                raise WorkflowRejectedError(reason, self.workflow_id)
            return event.payload.get("metadata")
        finally:
            self._step_runtime.release_event(event)

    def _install_step_helpers(self, step: object) -> None:
        async def report_complete(result: object = _WORKFLOW_UNSET) -> None:
            async def notify() -> object:
                await self.notify_agent(
                    WorkflowCompleteCallback(
                        workflow_name=self.workflow_name,
                        workflow_id=self.workflow_id,
                        timestamp=self._now_ms(),
                        result=result,
                    )
                )
                return None

            await self._durable(step, "reportComplete", notify)

        async def report_error(error: BaseException | str) -> None:
            self._error_reported = True
            message = str(error)

            async def notify() -> object:
                await self.notify_agent(
                    WorkflowErrorCallback(
                        workflow_name=self.workflow_name,
                        workflow_id=self.workflow_id,
                        timestamp=self._now_ms(),
                        error=message,
                    )
                )
                return None

            await self._durable(step, "reportError", notify)

        async def send_event(event: object) -> None:
            async def notify() -> object:
                await self.notify_agent(
                    WorkflowEventCallback(
                        workflow_name=self.workflow_name,
                        workflow_id=self.workflow_id,
                        timestamp=self._now_ms(),
                        event=event,
                    )
                )
                return None

            await self._durable(step, "sendEvent", notify)

        async def update_agent_state(state: object) -> None:
            async def update() -> object:
                await _invoke_agent(self.agent, "_workflow_updateState", "set", state)
                return None

            await self._durable(step, "updateState", update)

        async def merge_agent_state(state: Mapping[str, object]) -> None:
            async def merge() -> object:
                await _invoke_agent(
                    self.agent,
                    "_workflow_updateState",
                    "merge",
                    dict(state),
                )
                return None

            await self._durable(step, "mergeState", merge)

        async def reset_agent_state() -> None:
            async def reset() -> object:
                await _invoke_agent(self.agent, "_workflow_updateState", "reset")
                return None

            await self._durable(step, "resetState", reset)

        helpers = {
            "report_complete": report_complete,
            "report_error": report_error,
            "send_event": send_event,
            "update_agent_state": update_agent_state,
            "merge_agent_state": merge_agent_state,
            "reset_agent_state": reset_agent_state,
        }
        for name, helper in helpers.items():
            self._step_runtime.install(step, name, helper)

    async def _durable(
        self,
        step: object,
        operation: str,
        callback: Callable[[], Awaitable[object]],
    ) -> object:
        name = f"__agent_{operation}_{self._step_counter}"
        self._step_counter += 1
        return await self._step_runtime.durable(step, name, callback)

    async def _auto_report_error(self, error: BaseException) -> None:
        if self._error_reported:
            return
        self._error_reported = True
        try:
            await self.notify_agent(
                WorkflowErrorCallback(
                    workflow_name=self.workflow_name,
                    workflow_id=self.workflow_id,
                    timestamp=self._now_ms(),
                    error=str(error),
                )
            )
        except BaseException:
            pass


async def _invoke_agent(agent: object, method: str, *args: object) -> object:
    callback = getattr(agent, method)
    result = callback(*args)
    return await result if inspect.isawaitable(result) else result


__all__ = (
    "AgentWorkflow",
    "AgentWorkflowEvent",
    "AgentWorkflowFacetOrigin",
    "AgentWorkflowOrigin",
    "AgentWorkflowPathStep",
    "AgentWorkflowRootOrigin",
    "AgentWorkflowStep",
    "ApprovalEventPayload",
    "DefaultProgress",
    "RpcOnlyAgentStub",
    "RunWorkflowOptions",
    "WorkflowAgentResolver",
    "WorkflowCallback",
    "WorkflowCallbackHandler",
    "WorkflowCallbackHandlers",
    "WorkflowCallbackType",
    "WorkflowCompleteCallback",
    "WorkflowControlAction",
    "WorkflowErrorInfo",
    "WorkflowErrorCallback",
    "WorkflowEventCallback",
    "WorkflowEventPayload",
    "WorkflowInfo",
    "WorkflowInstanceStatus",
    "WorkflowLedger",
    "WorkflowMetadataScalar",
    "WorkflowOrder",
    "WorkflowOperations",
    "WorkflowPage",
    "WorkflowProgressCallback",
    "WorkflowQueryCriteria",
    "WorkflowRuntime",
    "WorkflowSql",
    "WorkflowStatus",
    "WorkflowStepEvent",
    "WorkflowStepRuntime",
    "WorkflowTrackingRow",
    "WorkflowRejectedError",
    "WaitForApprovalOptions",
    "decode_workflow_callback",
    "decode_workflow_origin",
    "encode_workflow_callback",
    "encode_workflow_origin",
    "resolve_workflow_agent",
)
