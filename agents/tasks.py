from __future__ import annotations

import asyncio
import json
import math
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import wraps
from types import MappingProxyType
from typing import Any, Literal, TypeVar, TypedDict, cast, overload

from .core._discovery import (
    deferred_method,
    static_definition_function,
    static_mro_members,
)
from .core._wire import strict_json_loads
from .core.schema import parse_schema_version
from .core.utils import MISSING, gen_id, now_ms
from ._task_store import _TaskRunRow, _TaskStore
from .lifecycle import (
    LifecycleCapability,
    LifecycleRouteAddress,
    LifecycleRouteContext,
)
from .lifecycle._job_driver import _is_memory_limit_reset, _is_platform_failure
from .lifecycle.jobs import (
    LifecycleJob,
    LifecycleJobContext,
    LifecycleJobOutcome,
    LifecycleJobPushOptions,
    LifecycleJobReschedule,
    LifecycleMemoryLimitContext,
)


class TaskUndefined:
    """Python representation of a Task value stored as SQL NULL."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "<TASK_UNDEFINED>"


TASK_UNDEFINED = TaskUndefined()


type TaskJson = (
    str | int | float | bool | None | list["TaskJson"] | dict[str, "TaskJson"]
)
type TaskValue = TaskJson | TaskUndefined
type TaskDurationUnit = Literal["second", "minute", "hour", "day", "week"]
type TaskDurationString = str
type TaskRunState = Literal[
    "pending", "running", "waiting", "completed", "failed", "cancelled"
]
type TaskTerminalState = Literal["completed", "failed", "cancelled"]
type TaskWaitReason = Literal["sleep", "retry"]
type TaskBackoff = Literal["constant", "linear", "exponential"]
type TaskEventType = Literal[
    "task:accepted",
    "task:attempt:started",
    "task:attempt:interrupted",
    "task:step:started",
    "task:step:retry",
    "task:step:completed",
    "task:waiting",
    "task:completed",
    "task:failed",
    "task:cancelled",
    "task:deleted",
]
type _TaskDefinition = Callable[[Any, "TaskStep"], TaskValue | Awaitable[TaskValue]]
type TaskHandlers = Mapping[str, _TaskDefinition]
type TaskCallbacks = TaskHandlers


class _LocalTaskWakePayload(TypedDict):
    runId: str


class _RoutedTaskWakePayload(TypedDict):
    runId: str
    owner_path: str
    owner_path_key: str


type _TaskWakePayload = _LocalTaskWakePayload | _RoutedTaskWakePayload


class _TaskMemoryPolicyRecord(TypedDict):
    runId: str
    owner_path: str
    owner_path_key: str
    generation: str
    sealed: bool
    nextAt: int | None


class _TaskRepairRecord(TypedDict):
    runId: str
    nextAt: int | None


class _RoutedTaskWakeResult(TypedDict):
    nextAt: int | None
    repairs: list[_TaskRepairRecord]
    cleanups: list[str]


class _TaskDeadlineRoute(TypedDict):
    type: Literal["sync", "repair"]
    runId: str
    nextAt: int | None


class _TaskWakeRoute(TypedDict):
    type: Literal["wake"]
    runId: str
    generation: str


class _TaskRunRoute(TypedDict):
    type: Literal["extend"]
    runId: str


class _TaskCleanupRoute(TypedDict):
    type: Literal["cleanup"]
    runIds: list[str]


class _TaskMemoryRoute(TypedDict):
    type: Literal["memory"]
    runId: str
    generation: str
    sealed: bool
    nextAt: int | None


class _TaskDeadlineResult(TypedDict):
    nextAt: int | None


class _TaskMemoryResult(TypedDict):
    nextAt: int | None
    status: _TaskMemoryPolicyStatus


def _task_deadline_route(
    route_type: Literal["sync", "repair"],
    run_id: str,
    next_at: int | None,
) -> _TaskDeadlineRoute:
    return _TaskDeadlineRoute(type=route_type, runId=run_id, nextAt=next_at)


def _task_wake_route(run_id: str, generation: str) -> _TaskWakeRoute:
    return _TaskWakeRoute(type="wake", runId=run_id, generation=generation)


def _task_run_route(run_id: str) -> _TaskRunRoute:
    return _TaskRunRoute(type="extend", runId=run_id)


def _task_cleanup_route(run_ids: Sequence[str]) -> _TaskCleanupRoute:
    return _TaskCleanupRoute(type="cleanup", runIds=list(run_ids))


def _task_memory_route(
    run_id: str,
    generation: str,
    sealed: bool,
    next_at: int | None,
) -> _TaskMemoryRoute:
    return _TaskMemoryRoute(
        type="memory",
        runId=run_id,
        generation=generation,
        sealed=sealed,
        nextAt=next_at,
    )


def _task_deadline_result_wire(next_at: int | None) -> _TaskDeadlineResult:
    return _TaskDeadlineResult(nextAt=next_at)


def _task_memory_result_wire(
    next_at: int | None,
    status: _TaskMemoryPolicyStatus,
) -> _TaskMemoryResult:
    return _TaskMemoryResult(nextAt=next_at, status=status)


MAX_SERIALIZED_BYTES = 1_048_576
_TASK_DEFINITION_METADATA_ATTR = "__cf_task_definition__"
_TASK_DEFINITION_METADATA = object()
_TASK_SCHEMA_VERSION_KEY = "cf_agents:tasks_schema_version"
_CURRENT_TASK_SCHEMA_VERSION = 1
_MAX_DEFINITION_NAME_LENGTH = 256
_DEFAULT_LIST_LIMIT = 100
_MAX_LIST_LIMIT = 100
_RUN_STATES = frozenset(
    {"pending", "running", "waiting", "completed", "failed", "cancelled"}
)
_TERMINAL_STATES = frozenset({"completed", "failed", "cancelled"})
_MAX_STEPS_PER_RUN = 10_000
_MAX_RETRY_DELAY_MS = 24 * 60 * 60 * 1_000
_MAX_SAFE_INTEGER = 2**53 - 1
_MAX_SQLITE_INTEGER = 2**63 - 1
_CLAIM_SLACK_MS = 30_000
_DISPATCH_BUDGET_SECONDS = 5
_MEMORY_POLICY_PREFIX = "cf_agents:tasks_memory:"
_MEMORY_POLICY_RETRY_MS = 30_000
_WAKE_RETRY = {"maxAttempts": 1}
_DURATION_PATTERN = re.compile(r"^(\d+(?:\.\d+)?)\s+(second|minute|hour|day|week)s?$")
_DURATION_UNIT_MS = {
    "second": 1_000,
    "minute": 60_000,
    "hour": 3_600_000,
    "day": 86_400_000,
    "week": 604_800_000,
}
_CURRENT_ROUTED_TASK_WAKE: ContextVar[set[str] | None] = ContextVar(
    "current_routed_task_wake",
    default=None,
)
type _TaskMemoryPolicyStatus = Literal["applied", "already", "preclaim", "stale"]
_F = TypeVar("_F", bound=Callable[..., Any])


def task_definition() -> Callable[[_F], _F]:
    """Mark an Agent method as a replayable Task definition."""

    def decorator(definition: _F) -> _F:
        function = static_definition_function(definition)
        if function is None:
            raise TypeError("task_definition can only decorate methods")
        state = object.__getattribute__(definition, "__dict__")
        state[_TASK_DEFINITION_METADATA_ATTR] = _TASK_DEFINITION_METADATA
        return definition

    return decorator


def _discover_task_definitions(
    instance: object,
    members: Mapping[str, object] | None = None,
) -> dict[str, _TaskDefinition]:
    """Build a fresh Agent Task registry without evaluating descriptors."""
    definitions: dict[str, _TaskDefinition] = {}
    for name, member in (members or static_mro_members(instance)).items():
        function = static_definition_function(member)
        if function is None:
            continue
        metadata = None
        for candidate in (member, function):
            state = object.__getattribute__(candidate, "__dict__")
            if type(state) is dict:
                metadata = state.get(_TASK_DEFINITION_METADATA_ATTR)
            if metadata is _TASK_DEFINITION_METADATA:
                break
        if metadata is _TASK_DEFINITION_METADATA:
            definitions[name] = cast(_TaskDefinition, deferred_method(member, instance))
    return definitions


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskStepRetryOptions:
    """Configure retry count, delay, and backoff for Task steps."""

    limit: int | None = None
    delay: int | float | TaskDurationString | None = None
    backoff: TaskBackoff | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskStepConfig:
    """Override retry and timeout policy for one Task step."""

    retries: TaskStepRetryOptions | None = None
    timeout: int | float | TaskDurationString | None = None


class TaskSignal:
    """Expose cooperative attempt cancellation to Task callbacks."""

    __slots__ = ("_event", "_reason")

    def __init__(self) -> None:
        self._event = asyncio.Event()
        self._reason: BaseException | None = None

    @property
    def aborted(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> BaseException | None:
        return self._reason

    def abort(self, reason: BaseException) -> None:
        if self._event.is_set():
            return
        self._reason = reason
        self._event.set()

    async def wait(self) -> None:
        await self._event.wait()

    def throw_if_aborted(self) -> None:
        if self._reason is not None:
            raise self._reason


@dataclass(frozen=True, slots=True)
class TaskStepAttempt:
    """Describe one step attempt and its stable cancellation signal."""

    attempt: int
    idempotency_key: str
    signal: TaskSignal


@dataclass(frozen=True, slots=True)
class TaskInterruptedStep:
    """Identify the first interrupted step observed during replay."""

    name: str
    attempt: int


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskRunOptions:
    """Configure Task run identity, metadata, and terminal retention."""

    run_id: str | None = None
    idempotency_key: str | None = None
    metadata: Mapping[str, TaskJson] | None = None
    retain: bool = True

    def __post_init__(self) -> None:
        if self.metadata is not None:
            object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskListOptions:
    """Filter newest-first Task run inspection."""

    definition: str | None = None
    state: TaskRunState | Sequence[TaskRunState] | None = None
    limit: int = _DEFAULT_LIST_LIMIT

    def __post_init__(self) -> None:
        if self.state is not None and not isinstance(self.state, str):
            object.__setattr__(self, "state", tuple(self.state))


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskDeleteOptions:
    """Bound deletion of retained terminal Task journals."""

    state: Sequence[TaskTerminalState] | None = None
    settled_before: datetime | None = None
    limit: int = _DEFAULT_LIST_LIMIT

    def __post_init__(self) -> None:
        if self.state is not None:
            object.__setattr__(self, "state", tuple(self.state))


@dataclass(frozen=True, slots=True, kw_only=True)
class TasksOptions:
    """Configure Task definitions and default step policy."""

    definitions: TaskHandlers | None = None
    retries: TaskStepRetryOptions | None = None
    step_timeout: int | float | TaskDurationString | None = None
    on_error: Callable[[BaseException], object | Awaitable[object]] | None = None

    def __post_init__(self) -> None:
        if self.definitions is not None:
            object.__setattr__(
                self,
                "definitions",
                MappingProxyType(dict(self.definitions)),
            )


@dataclass(frozen=True, slots=True)
class TaskReceipt:
    """Report durable acceptance or joining of a Task run."""

    run_id: str
    definition: str
    accepted: bool
    state: TaskRunState
    created_at: int


@dataclass(frozen=True, slots=True)
class TaskError:
    """Describe a persisted Task failure."""

    name: str
    message: str


@dataclass(frozen=True, slots=True)
class PendingTaskRunSnapshot:
    """Describe a Task accepted but not yet claimed."""

    run_id: str
    definition: str
    state: Literal["pending"]
    created_at: int
    metadata: dict[str, TaskJson] | None = None


@dataclass(frozen=True, slots=True)
class RunningTaskRunSnapshot:
    """Describe the currently claimed Task attempt."""

    run_id: str
    definition: str
    state: Literal["running"]
    attempt: int
    started_at: int
    created_at: int
    status_message: str | None = None
    metadata: dict[str, TaskJson] | None = None


@dataclass(frozen=True, slots=True)
class WaitingTaskRunSnapshot:
    """Describe a Task parked on a durable deadline."""

    run_id: str
    definition: str
    state: Literal["waiting"]
    reason: TaskWaitReason
    wake_at: int
    created_at: int
    status_message: str | None = None
    metadata: dict[str, TaskJson] | None = None


@dataclass(frozen=True, slots=True)
class CompletedTaskRunSnapshot[OutputT]:
    """Describe a successfully completed Task run."""

    run_id: str
    definition: str
    state: Literal["completed"]
    result: OutputT | TaskUndefined
    created_at: int
    settled_at: int
    metadata: dict[str, TaskJson] | None = None


@dataclass(frozen=True, slots=True)
class FailedTaskRunSnapshot:
    """Describe a failed Task run and its persisted error."""

    run_id: str
    definition: str
    state: Literal["failed"]
    error: TaskError
    created_at: int
    settled_at: int
    metadata: dict[str, TaskJson] | None = None


@dataclass(frozen=True, slots=True)
class CancelledTaskRunSnapshot:
    """Describe a cooperatively cancelled Task run."""

    run_id: str
    definition: str
    state: Literal["cancelled"]
    created_at: int
    settled_at: int
    reason: str | None = None
    metadata: dict[str, TaskJson] | None = None


type TaskRunSnapshot[OutputT] = (
    PendingTaskRunSnapshot
    | RunningTaskRunSnapshot
    | WaitingTaskRunSnapshot
    | CompletedTaskRunSnapshot[OutputT]
    | FailedTaskRunSnapshot
    | CancelledTaskRunSnapshot
)


class NonRetryableError(Exception):
    """Mark an application failure that must not retry."""


class DuplicateTaskStepError(Exception):
    """Report reuse of a durable step name in one replay."""

    def __init__(self, step_name: str) -> None:
        self.step_name = step_name
        super().__init__(
            f'Step name "{step_name}" was already used in this run. Step names '
            "are durable journal keys; suffix loop steps with a stable index."
        )


class TaskReplayDivergedError(Exception):
    """Report code that no longer matches its persisted step journal."""

    def __init__(self, step_name: str, detail: str) -> None:
        self.step_name = step_name
        super().__init__(
            f'Replay diverged from the journal at step "{step_name}": {detail}. '
            "Version the definition name instead of changing an in-flight run."
        )


class MissingTaskDefinitionError(Exception):
    """Report an in-flight run whose definition is not registered."""

    def __init__(self, definition: str) -> None:
        self.definition = definition
        super().__init__(
            f'No Task definition named "{definition}" is registered. Re-register '
            "the definition so its in-flight runs can finish."
        )


class TaskWakeCollisionError(Exception):
    """Reject a wake ID already owned by a different root or facet run."""

    def __init__(self, job_id: str) -> None:
        self.job_id = job_id
        super().__init__(f"Task wake ID collides with another run: {job_id!r}")


class TaskStorageError(Exception):
    """Raised when persisted Task tables do not match the current schema."""


class TaskSerializationError(Exception):
    """Report a Task value that cannot cross the durable JSON boundary."""

    def __init__(self, context: str, detail: str) -> None:
        super().__init__(f"Cannot serialize {context}: {detail}")


class TaskMemoryLimitSealed(Exception):
    """Report a run sealed after repeated alarm memory resets."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        super().__init__(
            f'Task run "{run_id}" was sealed after repeated memory-limit resets'
        )


class Task[InputT, OutputT]:
    """Provide definition-scoped Task operations."""

    def __init__(self, tasks: Tasks, name: str) -> None:
        self._tasks = tasks
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    async def run(
        self,
        input: InputT | None = None,
        *,
        run_id: str | None = None,
        idempotency_key: str | None = None,
        metadata: Mapping[str, TaskJson] | None = None,
        retain: bool = True,
    ) -> TaskReceipt:
        """Accept or join a run for this definition."""
        return await self._tasks.run(
            self._name,
            input,
            run_id=run_id,
            idempotency_key=idempotency_key,
            metadata=metadata,
            retain=retain,
        )

    async def get(self, run_id: str) -> TaskRunSnapshot[OutputT] | None:
        """Return this definition's run snapshot when present."""
        return cast(
            TaskRunSnapshot[OutputT] | None,
            await self._tasks._snapshot(run_id, self._name),
        )

    async def get_by_idempotency_key(
        self,
        key: str,
    ) -> TaskRunSnapshot[OutputT] | None:
        """Look up this definition's run by idempotency key."""
        return cast(
            TaskRunSnapshot[OutputT] | None,
            await self._tasks._snapshot_by_idempotency_key(key, self._name),
        )

    async def cancel(self, run_id: str, reason: str | None = None) -> bool:
        """Request cancellation when the run belongs to this definition."""
        return await self._tasks._cancel_scoped(run_id, self._name, reason)


@dataclass(frozen=True, slots=True)
class _ResolvedStepPolicy:
    retry_limit: int
    retry_delay_ms: int
    backoff: TaskBackoff
    timeout_ms: int


class _TaskSuspension(BaseException):
    def __init__(self, wake_at: int, reason: TaskWaitReason) -> None:
        self.wake_at = wake_at
        self.reason = reason
        super().__init__(f"Task {reason} is waiting until {wake_at}")


class _TaskAttemptSuperseded(BaseException):
    pass


class _TaskCancellation(BaseException):
    def __init__(self, reason: str | None) -> None:
        self.reason = reason
        super().__init__(reason or "Task run was cancelled")


@dataclass(slots=True)
class _ActiveTaskAttempt:
    generation: str
    task: asyncio.Task[None]
    signal: TaskSignal
    memory_failed: bool = False


class _PersistedTaskError(Exception):
    def __init__(self, name: str, message: str) -> None:
        self.name = name
        super().__init__(message)


class TaskStep:
    """Replay named work and durable sleeps against a Task journal."""

    def __init__(
        self,
        tasks: Tasks,
        run_id: str,
        definition: str,
        generation: str,
        attempt_signal: TaskSignal,
        *,
        starts_live: bool,
        interrupted: TaskInterruptedStep | None,
    ) -> None:
        self._tasks = tasks
        self._run_id = run_id
        self._definition = definition
        self._generation = generation
        self._attempt_signal = attempt_signal
        self._live = starts_live
        self._interrupted = interrupted
        self._used_names: set[str] = set()
        self._last_status: str | None = None

    @property
    def interrupted(self) -> TaskInterruptedStep | None:
        return self._interrupted

    @overload
    def do[T: TaskValue](
        self,
        name: str,
        callback: Callable[[TaskStepAttempt], T | Awaitable[T]],
        /,
    ) -> Awaitable[T]: ...

    @overload
    def do[T: TaskValue](
        self,
        name: str,
        config: TaskStepConfig,
        callback: Callable[[TaskStepAttempt], T | Awaitable[T]],
        /,
    ) -> Awaitable[T]: ...

    @overload
    def do[T: TaskValue](
        self,
        name: str,
        config: TaskStepConfig | None = None,
        /,
    ) -> Callable[
        [Callable[[], T | Awaitable[T]]],
        Callable[[], Awaitable[T]],
    ]: ...

    def do(
        self,
        name: str,
        config_or_callback: TaskStepConfig
        | Callable[[TaskStepAttempt], Any | Awaitable[Any]]
        | None = None,
        callback: Callable[[TaskStepAttempt], Any | Awaitable[Any]] | None = None,
        /,
    ) -> Awaitable[Any] | Callable[[Callable[..., Any]], Callable[[], Awaitable[Any]]]:
        """Replay named deferred work, raising when the journal diverges."""
        if callable(config_or_callback) and callback is None:
            return self._do(name, None, config_or_callback)
        if callback is not None:
            if config_or_callback is not None and not isinstance(
                config_or_callback, TaskStepConfig
            ):
                raise TypeError("Task step configuration must be TaskStepConfig")
            return self._do(name, config_or_callback, callback)
        if config_or_callback is not None and not isinstance(
            config_or_callback, TaskStepConfig
        ):
            raise TypeError("Task step configuration must be TaskStepConfig")

        def decorate(
            wrapped: Callable[[], Any | Awaitable[Any]],
        ) -> Callable[[], Awaitable[Any]]:
            def invoke(_: TaskStepAttempt) -> Any | Awaitable[Any]:
                return wrapped()

            @wraps(wrapped)
            async def deferred() -> Any:
                return await self._do(name, config_or_callback, invoke)

            return deferred

        return decorate

    async def sleep(self, name: str, duration: int | float | str) -> None:
        """Park the run for a durable relative duration."""
        duration_ms = _parse_task_duration(duration, "sleep duration")
        await self._sleep_at(name, lambda: now_ms() + duration_ms)

    async def sleep_until(self, name: str, when: int | float | datetime) -> None:
        """Park the run until an epoch-millisecond or datetime deadline."""
        wake_at = _sleep_until_ms(when, name)
        await self._sleep_at(name, lambda: wake_at)

    async def status(self, message: str) -> None:
        """Persist a live-attempt status message once per distinct value."""
        self._tasks._ensure_attempt_current(self._run_id, self._generation)
        message = str(message)
        if not self._live or message == self._last_status:
            return
        now = now_ms()
        if not self._tasks._task_store().update_status(
            self._run_id,
            self._generation,
            message,
            now,
        ):
            raise _TaskAttemptSuperseded
        self._last_status = message

    def idempotency_key(self, name: str) -> str:
        """Return a stable run-and-step-scoped idempotency key."""
        return f"{self._run_id}:{name}"

    def _enter_step(self, name: str) -> None:
        _validate_step_name(name)
        if name in self._used_names:
            raise DuplicateTaskStepError(name)
        self._used_names.add(name)
        self._tasks._ensure_attempt_current(self._run_id, self._generation)

    async def _sleep_at(self, name: str, wake_time: Callable[[], int]) -> None:
        self._enter_step(name)
        store = self._tasks._task_store()
        row = store.get_step(self._run_id, name)
        if row is None:
            self._live = True
            wake_at = wake_time()
            if type(wake_at) is not int or wake_at > _MAX_SQLITE_INTEGER:
                raise ValueError(f"Invalid sleep deadline for step {name!r}")
            now = now_ms()
            if wake_at <= now:
                if not store.insert_completed_sleep(
                    self._run_id,
                    self._generation,
                    name,
                    now,
                ):
                    raise _TaskAttemptSuperseded
                return
            if not store.park_new_sleep(
                self._run_id,
                self._generation,
                name,
                wake_at,
                now,
            ):
                raise _TaskAttemptSuperseded
            await self._tasks._sync_wake(self._run_id)
            raise _TaskSuspension(wake_at, "sleep")
        if row.kind != "sleep":
            raise TaskReplayDivergedError(
                name,
                f"the recorded kind is {row.kind!r}, not 'sleep'",
            )
        if row.state == "completed":
            return
        self._live = True
        now = now_ms()
        wake_at = row.next_at or 0
        if wake_at > now:
            if not store.park_existing_sleep(
                self._run_id,
                self._generation,
                wake_at,
                now,
            ):
                raise _TaskAttemptSuperseded
            await self._tasks._sync_wake(self._run_id)
            raise _TaskSuspension(wake_at, "sleep")
        if not store.complete_sleep(
            self._run_id,
            self._generation,
            name,
            now,
        ):
            raise _TaskAttemptSuperseded

    async def _do(
        self,
        name: str,
        config: TaskStepConfig | None,
        callback: Callable[[TaskStepAttempt], Any | Awaitable[Any]],
    ) -> Any:
        policy = self._tasks._step_policy(config)
        self._enter_step(name)
        store = self._tasks._task_store()

        row = store.get_step(self._run_id, name)
        emit_retry = False
        if row is not None:
            if row.kind != "do":
                raise TaskReplayDivergedError(
                    name,
                    f"the recorded kind is {row.kind!r}, not 'do'",
                )
            if row.state == "completed":
                return _deserialize_task_value(row.result, f"Task step {name!r} result")
            if row.state == "failed":
                raise _PersistedTaskError(
                    row.error_name or "Error",
                    row.error_message or "Task step failed",
                )
            now = now_ms()
            if row.state == "waiting" and row.next_at is not None and row.next_at > now:
                raise _TaskSuspension(row.next_at, "retry")
            emit_retry = row.state == "waiting"
            row = store.restart_step(self._run_id, self._generation, name, now)
            if row is None:
                raise _TaskAttemptSuperseded
        else:
            if store.count_steps(self._run_id) >= _MAX_STEPS_PER_RUN:
                raise RuntimeError(
                    f"Task run {self._run_id!r} exceeded {_MAX_STEPS_PER_RUN} steps"
                )
            now = now_ms()
            if not store.insert_step(self._run_id, self._generation, name, now):
                raise _TaskAttemptSuperseded
            row = store.get_step(self._run_id, name)
            if row is None:
                raise RuntimeError(f"Task step {name!r} was not durably recorded")

        self._live = True
        await self._tasks._refresh_claim(
            self._run_id,
            self._generation,
            policy.timeout_ms,
        )
        step_attempt = TaskStepAttempt(
            attempt=row.attempt,
            idempotency_key=f"{self._run_id}:{name}",
            signal=TaskSignal(),
        )
        if emit_retry:
            await self._tasks._emit(
                "task:step:retry",
                {
                    "runId": self._run_id,
                    "definition": self._definition,
                    "step": name,
                    "attempt": row.attempt,
                },
            )
        await self._tasks._emit(
            "task:step:started",
            {
                "runId": self._run_id,
                "definition": self._definition,
                "step": name,
                "attempt": row.attempt,
            },
        )
        self._tasks._ensure_attempt_current(self._run_id, self._generation)

        try:
            result = await self._tasks._invoke_step_callback(
                callback,
                step_attempt,
                name,
                policy.timeout_ms,
                self._attempt_signal,
            )
            result_json = _serialize_task_value(result, f"Task step {name!r} result")
            replay_result = _deserialize_task_value(
                result_json,
                f"Task step {name!r} result",
            )
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            if isinstance(error, _TaskCancellation):
                raise
            if _is_platform_failure(error):
                raise
            error_name, error_message = _error_summary(error)
            now = now_ms()
            if not _is_non_retryable(error) and row.attempt < policy.retry_limit:
                wake_at = now + _retry_delay(policy, row.attempt)
                if not store.park_step_retry(
                    self._run_id,
                    self._generation,
                    name,
                    row.attempt,
                    wake_at,
                    now,
                ):
                    raise _TaskAttemptSuperseded from error
                await self._tasks._sync_wake(self._run_id)
                raise _TaskSuspension(wake_at, "retry") from error
            if not store.fail_step(
                self._run_id,
                self._generation,
                name,
                row.attempt,
                error_name,
                error_message,
                now,
            ):
                raise _TaskAttemptSuperseded from error
            raise

        completed_at = now_ms()
        if not store.complete_step(
            self._run_id,
            self._generation,
            name,
            row.attempt,
            result_json,
            completed_at,
        ):
            raise _TaskAttemptSuperseded
        await self._tasks._emit(
            "task:step:completed",
            {
                "runId": self._run_id,
                "definition": self._definition,
                "step": name,
                "attempt": row.attempt,
            },
        )
        return replay_result


class Tasks(LifecycleCapability):
    """Run replayable definitions over a generation-fenced durable journal."""

    capability_id = "tasks"

    def __init__(
        self,
        definitions: TaskHandlers | None = None,
        *,
        retries: TaskStepRetryOptions | None = None,
        step_timeout: int | float | TaskDurationString | None = None,
        on_error: Callable[[BaseException], object | Awaitable[object]] | None = None,
    ) -> None:
        self._definitions = dict(definitions or {})
        self._reserved_definitions: dict[str, _TaskDefinition] = {}
        self._retries = retries
        self._step_timeout = step_timeout
        self._on_error = on_error
        self._store: _TaskStore | None = None
        self._active_runs: dict[str, _ActiveTaskAttempt] = {}
        self._execution_tasks: dict[str, asyncio.Task[None]] = {}
        self._routed_wakes: dict[tuple[str, str], asyncio.Task[int | None]] = {}
        self._routed_wake_generations: dict[tuple[str, str], str] = {}
        self._submitted_runs: set[str] = set()
        self._reconciliation_pending = False
        self._reconciliation_run_ids: set[str] = set()
        self._startup_error: TaskStorageError | None = None

    async def on_start(self) -> None:
        version = await self.lifecycle.storage.get(_TASK_SCHEMA_VERSION_KEY)
        parsed_version = parse_schema_version(version)
        if parsed_version > _CURRENT_TASK_SCHEMA_VERSION:
            return
        store = self._task_store()
        if not store.schema_is_compatible():
            self._startup_error = TaskStorageError(
                "persisted Task tables do not match schema version 1"
            )
            await self._observe_error(self._startup_error)
            return
        store.prepare()
        if parsed_version < _CURRENT_TASK_SCHEMA_VERSION:
            await self.lifecycle.storage.put(
                _TASK_SCHEMA_VERSION_KEY,
                _CURRENT_TASK_SCHEMA_VERSION,
            )
        cleaned_run_ids = await self._cleanup_unretained_terminals()
        store.reconcile_deadlines(now_ms())
        has_deadlines = bool(store.non_terminal_deadlines())
        if self.lifecycle.owns_physical_alarm:
            for run_id in cleaned_run_ids:
                await self._sync_root_wake(None, run_id, None)
                store.delete_unretained_terminal(run_id)
            await self._sync_all_wakes()
        elif has_deadlines or cleaned_run_ids:
            self._reconciliation_run_ids.update(cleaned_run_ids)
            self._reconciliation_pending = True
            if self.lifecycle.retained_work.available:
                self.lifecycle.retained_work.retain(self._flush_after_start)

    async def _cleanup_unretained_terminals(self) -> tuple[str, ...]:
        store = self._task_store()
        rows = store.unretained_terminal_runs()
        for row in rows:
            if row.state == "completed":
                await self._emit(
                    "task:completed",
                    {"runId": row.run_id, "definition": row.definition},
                )
            elif row.state == "failed":
                await self._emit(
                    "task:failed",
                    {
                        "runId": row.run_id,
                        "definition": row.definition,
                        "error": row.error_name or "Error",
                    },
                )
            else:
                await self._emit(
                    "task:cancelled",
                    {
                        "runId": row.run_id,
                        "definition": row.definition,
                        "reason": row.cancel_reason,
                    },
                )
        return tuple(row.run_id for row in rows)

    def handle[InputT, OutputT](self, definition: str) -> Task[InputT, OutputT]:
        """Return a definition-scoped Task handle or reject an unknown name."""
        self._validate_public_definition(definition)
        return Task(self, definition)

    async def run(
        self,
        definition: str,
        input: object = None,
        *,
        run_id: str | None = None,
        idempotency_key: str | None = None,
        metadata: Mapping[str, TaskJson] | None = None,
        retain: bool = True,
    ) -> TaskReceipt:
        """Accept or join a run; invalid input, identity, or mirrors raise."""
        self._validate_public_definition(definition)
        return await self._accept(
            definition,
            input,
            run_id=run_id,
            idempotency_key=idempotency_key,
            metadata=metadata,
            retain=retain,
        )

    async def get(self, run_id: str) -> TaskRunSnapshot[object] | None:
        """Return a run snapshot, or None when it is absent."""
        return await self._snapshot(run_id)

    async def get_by_idempotency_key(
        self,
        key: str,
    ) -> TaskRunSnapshot[object] | None:
        """Return the run assigned to an idempotency key when present."""
        return await self._snapshot_by_idempotency_key(key)

    async def list(
        self,
        options: TaskListOptions | None = None,
    ) -> tuple[TaskRunSnapshot[object], ...]:
        """Return bounded newest-first snapshots matching the supplied filters."""
        await self._ready()
        options = options or TaskListOptions()
        limit = _validate_limit(options.limit)
        states = _normalize_states(options.state)
        return tuple(
            _snapshot(row)
            for row in self._task_store().list(
                definition=options.definition,
                states=states,
                limit=limit,
            )
        )

    async def cancel(self, run_id: str, reason: str | None = None) -> bool:
        """Cooperatively cancel a pending, waiting, or active run."""
        await self._ready()
        row = self._task_store().get(run_id)
        if row is None:
            return False
        if row.state in _TERMINAL_STATES:
            await self._sync_wake(run_id)
            return False
        store = self._task_store()
        active = self._active_runs.get(run_id)
        if row.generation is not None:
            requested_at = now_ms()
            if active is not None and active.generation == row.generation:
                if store.request_cancel(
                    run_id,
                    row.generation,
                    reason,
                    requested_at,
                ):
                    active.signal.abort(_TaskCancellation(reason))
                    await self._sync_wake(run_id)
                    return True
            elif store.settle_cancelled_claim(
                run_id,
                row.generation,
                reason,
                requested_at,
            ):
                await self._after_cancelled(row, reason)
                return True
            return False

        settled_at = now_ms()
        if not store.cancel_parked(run_id, reason, settled_at):
            return False
        await self._after_cancelled(row, reason)
        return True

    async def delete(self, options: TaskDeleteOptions | None = None) -> int:
        """Delete a bounded set of retained terminal journals after commit."""
        await self._ready()
        options = options or TaskDeleteOptions()
        limit = _validate_limit(options.limit)
        states = _normalize_terminal_states(options.state)
        settled_before = _datetime_ms(options.settled_before)
        deleted = self._task_store().delete_terminal(
            states=states,
            settled_before=settled_before,
            limit=limit,
        )
        for run_id, definition in deleted:
            await self._emit(
                "task:deleted",
                {"runId": run_id, "definition": definition},
            )
            await self._sync_wake(run_id)
        return len(deleted)

    def _submit_run(self, run_id: str) -> None:
        if run_id in self._execution_tasks or run_id in self._submitted_runs:
            return
        if _CURRENT_ROUTED_TASK_WAKE.get() is not None:
            return
        if self.lifecycle.routes.source is not None:
            return
        if not self.lifecycle.retained_work.available:
            return
        self._submitted_runs.add(run_id)
        try:
            self.lifecycle.retained_work.retain(lambda: self._run_submitted(run_id))
        except BaseException:
            self._submitted_runs.discard(run_id)
            return

    async def _run_submitted(self, run_id: str) -> None:
        try:
            await self._execution_task(run_id)
        finally:
            self._submitted_runs.discard(run_id)

    def _execution_task(
        self,
        run_id: str,
        generation: str | None = None,
    ) -> asyncio.Task[None]:
        active = self._active_runs.get(run_id)
        if active is not None:
            return active.task
        existing = self._execution_tasks.get(run_id)
        if existing is not None and not existing.done():
            return existing
        task = asyncio.create_task(self._execute_run(run_id, generation))
        self._execution_tasks[run_id] = task

        def forget(completed: asyncio.Task[None]) -> None:
            if self._execution_tasks.get(run_id) is completed:
                self._execution_tasks.pop(run_id, None)

        task.add_done_callback(forget)
        return task

    async def _execute_run(
        self,
        run_id: str,
        claim_generation: str | None = None,
    ) -> None:
        await self.lifecycle.ready()
        store = self._task_store()
        row = store.get(run_id)
        if row is None or row.state in _TERMINAL_STATES:
            return
        if row.cancel_requested:
            if row.generation is None:
                await self._settle_unclaimed_cancelled(row, row.cancel_reason)
            else:
                await self._settle_cancelled_claim(
                    row,
                    row.generation,
                    row.cancel_reason,
                )
            return
        now = now_ms()
        if row.next_at is not None and row.next_at > now:
            return

        definition = self._resolve_definition(row.definition)
        if definition is None:
            error = MissingTaskDefinitionError(row.definition)
            if row.generation is None:
                await self._settle_unclaimed_failure(row, error)
            else:
                await self._settle_failure(row, row.generation, error)
            return

        interrupted: TaskInterruptedStep | None = None
        if row.state == "running":
            running_step = store.newest_running_step(run_id)
            if running_step is not None:
                interrupted = TaskInterruptedStep(
                    name=running_step.step_name,
                    attempt=running_step.attempt,
                )
            await self._emit(
                "task:attempt:interrupted",
                {
                    "runId": run_id,
                    "definition": row.definition,
                    "attempt": row.attempt,
                    "step": interrupted.name if interrupted is not None else None,
                },
            )

        try:
            claim_deadline = now + self._claim_timeout_ms()
            _validate_deadline(claim_deadline, "Task claim deadline")
        except BaseException as error:
            if _is_platform_failure(error):
                raise
            if row.generation is None:
                await self._settle_unclaimed_failure(row, error)
            else:
                await self._settle_failure(row, row.generation, error)
            return
        generation = claim_generation or gen_id()
        claimed = store.claim_run(run_id, generation, now, claim_deadline)
        if claimed is None:
            return
        current_task = asyncio.current_task()
        if current_task is None:
            raise RuntimeError("Task execution requires an asyncio Task")
        active = _ActiveTaskAttempt(
            generation,
            cast(asyncio.Task[None], current_task),
            TaskSignal(),
        )
        self._active_runs[run_id] = active
        try:
            await self._sync_wake(run_id)
            await self._emit(
                "task:attempt:started",
                {
                    "runId": run_id,
                    "definition": row.definition,
                    "attempt": claimed.attempt,
                },
            )
            self._ensure_attempt_current(run_id, generation)
            step = TaskStep(
                self,
                run_id,
                row.definition,
                generation,
                active.signal,
                starts_live=claimed.attempt == 1,
                interrupted=interrupted,
            )
            input_value = _deserialize_task_value(row.input, "Task input")
            result = await self.lifecycle.run_in_host_context(
                lambda: definition(input_value, step)
            )
            result_json = _serialize_task_value(
                result,
                f"result of Task definition {row.definition!r}",
            )
        except _TaskSuspension as suspension:
            current = store.get(run_id)
            if current is not None and current.state == "cancelled":
                return
            await self._emit(
                "task:waiting",
                {
                    "runId": run_id,
                    "definition": row.definition,
                    "reason": suspension.reason,
                    "wakeAt": suspension.wake_at,
                },
            )
        except _TaskAttemptSuperseded:
            await self._settle_requested_cancellation(row, generation)
            return
        except _TaskCancellation as cancellation:
            await self._settle_cancelled_claim(row, generation, cancellation.reason)
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            if _is_platform_failure(error):
                if _is_memory_limit_reset(error):
                    active.memory_failed = True
                raise
            await self._settle_failure(row, generation, error)
        else:
            settled_at = now_ms()
            if store.settle_completed(run_id, generation, result_json, settled_at):
                await self._emit(
                    "task:completed",
                    {"runId": run_id, "definition": row.definition},
                )
                store.delete_unretained_terminal(run_id)
                await self._sync_wake(run_id)
            else:
                await self._settle_requested_cancellation(row, generation)
        finally:
            if not active.memory_failed and self._active_runs.get(run_id) is active:
                self._active_runs.pop(run_id, None)

    async def _settle_unclaimed_failure(
        self,
        row: _TaskRunRow,
        error: BaseException,
    ) -> None:
        error_name, error_message = _error_summary(error)
        if self._task_store().settle_failed_unclaimed(
            row.run_id,
            error_name,
            error_message,
            now_ms(),
        ):
            await self._emit(
                "task:failed",
                {
                    "runId": row.run_id,
                    "definition": row.definition,
                    "error": error_name,
                },
            )
            self._task_store().delete_unretained_terminal(row.run_id)
            await self._sync_wake(row.run_id)
        await self._observe_error(error)

    async def _settle_failure(
        self,
        row: _TaskRunRow,
        generation: str,
        error: BaseException,
    ) -> None:
        error_name, error_message = _error_summary(error)
        if self._task_store().settle_failed(
            row.run_id,
            generation,
            error_name,
            error_message,
            now_ms(),
        ):
            await self._emit(
                "task:failed",
                {
                    "runId": row.run_id,
                    "definition": row.definition,
                    "error": error_name,
                },
            )
            self._task_store().delete_unretained_terminal(row.run_id)
            await self._sync_wake(row.run_id)
        else:
            await self._settle_requested_cancellation(row, generation)
        await self._observe_error(error)

    async def _settle_unclaimed_cancelled(
        self,
        row: _TaskRunRow,
        reason: str | None,
    ) -> None:
        if self._task_store().cancel_parked(row.run_id, reason, now_ms()):
            await self._after_cancelled(row, reason)

    async def _settle_cancelled_claim(
        self,
        row: _TaskRunRow,
        generation: str,
        reason: str | None,
    ) -> None:
        if self._task_store().settle_cancelled_claim(
            row.run_id,
            generation,
            reason,
            now_ms(),
        ):
            await self._after_cancelled(row, reason)

    async def _settle_requested_cancellation(
        self,
        row: _TaskRunRow,
        generation: str,
    ) -> None:
        current = self._task_store().get(row.run_id)
        if (
            current is None
            or current.generation != generation
            or not current.cancel_requested
        ):
            return
        await self._settle_cancelled_claim(
            row,
            generation,
            current.cancel_reason,
        )

    async def _after_cancelled(
        self,
        row: _TaskRunRow,
        reason: str | None,
        *,
        sync_wake: bool = True,
    ) -> None:
        await self._emit(
            "task:cancelled",
            {"runId": row.run_id, "definition": row.definition, "reason": reason},
        )
        self._task_store().delete_unretained_terminal(row.run_id)
        if sync_wake:
            await self._sync_wake(row.run_id)

    async def _observe_error(self, error: BaseException) -> None:
        on_error = self._on_error
        if on_error is None:
            return
        try:
            await self.lifecycle.run_in_host_context(lambda: on_error(error))
        except BaseException:
            return

    def _resolve_definition(self, name: str) -> _TaskDefinition | None:
        definition = self._definitions.get(name)
        if definition is not None:
            return definition
        return self._reserved_definitions.get(name)

    def _ensure_attempt_current(self, run_id: str, generation: str) -> None:
        row = self._task_store().get(run_id)
        if row is None or row.state != "running" or row.generation != generation:
            raise _TaskAttemptSuperseded
        if row.cancel_requested:
            raise _TaskCancellation(row.cancel_reason)

    def _step_policy(self, config: TaskStepConfig | None) -> _ResolvedStepPolicy:
        defaults = _default_step_policy(self._retries, self._step_timeout)
        return _resolve_step_policy(defaults, config)

    async def _invoke_step_callback(
        self,
        callback: Callable[[TaskStepAttempt], Any | Awaitable[Any]],
        attempt: TaskStepAttempt,
        name: str,
        timeout_ms: int,
        attempt_signal: TaskSignal,
    ) -> Any:
        if attempt_signal.aborted:
            raise attempt_signal.reason or _TaskCancellation(None)
        callback_task = asyncio.create_task(
            self.lifecycle.run_in_host_context(lambda: callback(attempt))
        )
        timeout_task = asyncio.create_task(asyncio.sleep(timeout_ms / 1_000))
        cancellation_task = asyncio.create_task(attempt_signal.wait())
        try:
            completed, _ = await asyncio.wait(
                {callback_task, timeout_task, cancellation_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
        except asyncio.CancelledError as error:
            attempt.signal.abort(error)
            callback_task.cancel()
            timeout_task.cancel()
            cancellation_task.cancel()
            await asyncio.gather(
                callback_task,
                timeout_task,
                cancellation_task,
                return_exceptions=True,
            )
            raise
        if callback_task in completed:
            timeout_task.cancel()
            cancellation_task.cancel()
            await asyncio.gather(
                timeout_task,
                cancellation_task,
                return_exceptions=True,
            )
            return await callback_task

        if cancellation_task in completed:
            timeout_task.cancel()
            reason = attempt_signal.reason or _TaskCancellation(None)
            attempt.signal.abort(reason)
            callback_task.cancel()
            await asyncio.gather(timeout_task, return_exceptions=True)
            await self._retain_late_callback(callback_task)
            raise reason

        error = TimeoutError(
            f'Step "{name}" attempt {attempt.attempt} timed out after {timeout_ms}ms'
        )
        attempt.signal.abort(error)
        callback_task.cancel()
        cancellation_task.cancel()
        await asyncio.gather(cancellation_task, return_exceptions=True)
        await self._retain_late_callback(callback_task)
        raise error

    async def _retain_late_callback(self, callback_task: asyncio.Task[Any]) -> None:
        try:
            self.lifecycle.retained_work.retain(
                lambda: _observe_late_callback(callback_task)
            )
        except BaseException:
            await asyncio.gather(callback_task, return_exceptions=True)

    async def _ready(self) -> None:
        await self.lifecycle.ready()
        if self._startup_error is not None:
            raise self._startup_error
        await self._flush_reconciliation()

    async def _flush_reconciliation(self) -> None:
        if not self._reconciliation_pending:
            return
        run_ids = {run_id for run_id, _ in self._task_store().non_terminal_deadlines()}
        run_ids.update(self._reconciliation_run_ids)
        for run_id in sorted(run_ids):
            next_at = self._task_store().authoritative_deadline(run_id)
            await self.lifecycle.routes.to_root(
                _task_deadline_route("repair", run_id, next_at)
            )
            if run_id in self._reconciliation_run_ids:
                self._task_store().delete_unretained_terminal(run_id)
                self._reconciliation_run_ids.discard(run_id)
        self._reconciliation_run_ids.clear()
        self._reconciliation_pending = False

    async def _flush_after_start(self) -> None:
        await self.lifecycle.ready()
        await self._flush_reconciliation()

    def _claim_timeout_ms(self) -> int:
        return _default_step_policy(self._retries, self._step_timeout).timeout_ms + (
            _CLAIM_SLACK_MS
        )

    async def _refresh_claim(
        self,
        run_id: str,
        generation: str,
        timeout_ms: int,
    ) -> None:
        current_time = now_ms()
        deadline = current_time + timeout_ms + _CLAIM_SLACK_MS
        _validate_deadline(deadline, "Task claim deadline")
        if not self._task_store().refresh_claim(
            run_id,
            generation,
            deadline,
            current_time,
        ):
            raise _TaskAttemptSuperseded
        await self._sync_wake(run_id)

    async def _sync_wake(self, run_id: str) -> None:
        routed_updates = _CURRENT_ROUTED_TASK_WAKE.get()
        if routed_updates is not None:
            routed_updates.add(run_id)
            return
        next_at = self._task_store().authoritative_deadline(run_id)
        owner = self.lifecycle.routes.source
        if owner is not None:
            await self.lifecycle.routes.to_root(
                _task_deadline_route("sync", run_id, next_at)
            )
            return
        await self._sync_root_wake(None, run_id, next_at)

    async def _repair_wake(self, run_id: str) -> None:
        routed_updates = _CURRENT_ROUTED_TASK_WAKE.get()
        if routed_updates is not None:
            routed_updates.add(run_id)
            return
        next_at = self._task_store().authoritative_deadline(run_id)
        owner = self.lifecycle.routes.source
        if owner is not None:
            await self.lifecycle.routes.to_root(
                _task_deadline_route("repair", run_id, next_at)
            )
            return
        await self._sync_root_wake(None, run_id, next_at, repair=True)
        await self.lifecycle.jobs.rearm()

    async def _sync_all_wakes(self) -> None:
        if self.lifecycle.routes.source is not None:
            raise RuntimeError("only the root can synchronize all Task wakes")
        for run_id, next_at in self._task_store().non_terminal_deadlines():
            await self._sync_root_wake(None, run_id, next_at, repair=True)
        await self._sync_pending_memory_policies()
        await self.lifecycle.jobs.rearm()

    async def _sync_pending_memory_policies(self) -> None:
        policies = await self.lifecycle.storage.list(prefix=_MEMORY_POLICY_PREFIX)
        for key, value in policies.items():
            try:
                owner, run_id, _generation, _sealed, next_at = (
                    _stored_task_memory_policy(
                        key,
                        value,
                    )
                )
            except ValueError as error:
                await self.lifecycle.storage.delete(key)
                await self._observe_error(error)
                continue
            retry_at = next_at if next_at is not None else now_ms()
            await self._sync_root_wake(
                owner,
                run_id,
                retry_at,
                repair=True,
            )

    async def _sync_root_wake(
        self,
        owner: LifecycleRouteAddress | None,
        run_id: str,
        next_at: int | None,
        *,
        repair: bool = False,
    ) -> None:
        job_id = _task_job_id(owner, run_id)
        try:
            existing = await self.lifecycle.jobs._get_unvalidated_retry(job_id)
        except (TypeError, ValueError):
            existing = None
        if existing is not None:
            try:
                existing_run_id, existing_owner = _task_job_identity(existing)
            except (TypeError, ValueError):
                pass
            else:
                if existing_run_id != run_id or existing_owner != owner:
                    raise TaskWakeCollisionError(job_id)
        if next_at is None:
            await self.lifecycle.jobs.cancel(job_id)
            return
        payload = _task_job_payload(owner, run_id)
        if repair:
            try:
                existing = await self.lifecycle.jobs.get(job_id)
            except (TypeError, ValueError):
                existing = None
            if existing is not None and _task_job_matches(
                existing,
                job_id,
                next_at,
                payload,
            ):
                return
        await self.lifecycle.jobs.push(
            LifecycleJobPushOptions(
                id=job_id,
                fn="wake",
                time=next_at,
                payload=payload,
                retry=dict(_WAKE_RETRY),
            )
        )

    async def _await_dispatch_budget(self, task: asyncio.Task[Any]) -> bool:
        budget = asyncio.create_task(asyncio.sleep(_DISPATCH_BUDGET_SECONDS))
        try:
            completed, _ = await asyncio.wait(
                {task, budget},
                return_when=asyncio.FIRST_COMPLETED,
            )
        except asyncio.CancelledError:
            try:
                handed_off = self.lifecycle.track_alarm_work(task)
            except BaseException:
                handed_off = False
            if not handed_off:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            raise
        finally:
            if not budget.done():
                budget.cancel()
            await asyncio.gather(budget, return_exceptions=True)
        if task in completed:
            await task
            return True
        try:
            handed_off = self.lifecycle.track_alarm_work(task)
        except BaseException:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            raise
        if handed_off:
            return False
        await task
        return True

    async def _extend_active_claim(
        self,
        run_id: str,
        *,
        sync_wake: bool = True,
    ) -> None:
        active = self._active_runs.get(run_id)
        if active is None or active.task.done():
            return
        current_time = now_ms()
        deadline = current_time + self._claim_timeout_ms()
        _validate_deadline(deadline, "Task claim deadline")
        extended = self._task_store().extend_active_claim(
            run_id,
            active.generation,
            deadline,
            current_time,
        )
        if extended and sync_wake:
            await self._sync_wake(run_id)

    def _routed_wake_task(
        self,
        owner: LifecycleRouteAddress,
        run_id: str,
    ) -> asyncio.Task[int | None]:
        key = (owner.key, run_id)
        existing = self._routed_wakes.get(key)
        if existing is not None:
            if not existing.done() or self.lifecycle._alarm_work_is_tracked(existing):
                return existing
            self._routed_wakes.pop(key, None)
        generation = gen_id()
        self._routed_wake_generations[key] = generation
        task = asyncio.create_task(self._run_routed_wake(owner, run_id, generation))
        self._routed_wakes[key] = task

        def forget(completed: asyncio.Task[int | None]) -> None:
            if not completed.cancelled():
                error = completed.exception()
                if (
                    error is not None
                    and _is_memory_limit_reset(error)
                    and self.lifecycle._alarm_work_is_tracked(completed)
                ):
                    return
            if self._routed_wakes.get(key) is completed:
                self._routed_wakes.pop(key, None)

        task.add_done_callback(forget)
        return task

    async def _run_routed_wake(
        self,
        owner: LifecycleRouteAddress,
        run_id: str,
        generation: str,
    ) -> int | None:
        key = (owner.key, run_id)
        try:
            response = await self.lifecycle.routes.to(
                owner,
                _task_wake_route(run_id, generation),
            )
        except BaseException as error:
            if not _is_memory_limit_reset(error):
                if self._routed_wake_generations.get(key) == generation:
                    self._routed_wake_generations.pop(key, None)
            raise
        current_deadline, repairs, cleanups = _routed_deadlines(response)
        await self._sync_root_wake(owner, run_id, current_deadline)
        for routed_run_id, next_at in repairs:
            await self._sync_root_wake(
                owner,
                routed_run_id,
                next_at,
                repair=True,
            )
        if repairs:
            await self.lifecycle.jobs.rearm()
        if cleanups:
            await self.lifecycle.routes.to(
                owner,
                _task_cleanup_route(cleanups),
            )
        if self._routed_wake_generations.get(key) == generation:
            self._routed_wake_generations.pop(key, None)
        return current_deadline

    async def on_job(self, context: LifecycleJobContext) -> LifecycleJobOutcome:
        try:
            run_id, owner = _task_job_identity(context.job)
        except (TypeError, ValueError) as error:
            await self._observe_error(error)
            return None

        if owner is not None:
            try:
                policy = await self._pending_memory_policy(owner, run_id)
                if policy is not None:
                    try:
                        (
                            current_deadline,
                            _status,
                        ) = await self._deliver_routed_memory_policy(
                            owner,
                            run_id,
                            *policy,
                        )
                    except asyncio.CancelledError:
                        raise
                    except BaseException as error:
                        if not _is_platform_failure(error):
                            await self._observe_error(error)
                        retry_at = max(
                            policy[2] or 0,
                            now_ms() + _MEMORY_POLICY_RETRY_MS,
                        )
                        await self._sync_root_wake(owner, run_id, retry_at)
                        return LifecycleJobReschedule(retry_at)
                    else:
                        await self._clear_memory_policy(owner, run_id)
                        self._forget_routed_attempt(
                            (owner.key, run_id),
                            policy[0],
                        )
                else:
                    key = (owner.key, run_id)
                    existing = self._routed_wakes.get(key)
                    current_deadline = None
                    if existing is not None and not existing.done():
                        response = await self.lifecycle.routes.to(
                            owner,
                            _task_run_route(run_id),
                        )
                        extended_deadline = _task_deadline_result(
                            response,
                            "routed Task claim",
                        )
                        await self._sync_root_wake(
                            owner,
                            run_id,
                            extended_deadline,
                        )
                        current_deadline = extended_deadline
                    task = self._routed_wake_task(owner, run_id)
                    if self.lifecycle._alarm_work_is_tracked(task):
                        if current_deadline is None:
                            current_deadline = now_ms() + self._claim_timeout_ms()
                        return LifecycleJobReschedule(current_deadline)
                    handoff_deadline = now_ms() + self._claim_timeout_ms()
                    completed = await self._await_dispatch_budget(task)
                    current_deadline = task.result() if completed else handoff_deadline
                if current_deadline is None:
                    return None
                return LifecycleJobReschedule(current_deadline)
            except asyncio.CancelledError:
                raise
            except BaseException as error:
                if _is_platform_failure(error):
                    raise
                await self._observe_error(error)
                return "yield"

        await self._extend_active_claim(run_id)
        task = self._execution_task(run_id)
        if self.lifecycle._alarm_work_is_tracked(task):
            next_at = self._task_store().authoritative_deadline(run_id)
            return None if next_at is None else LifecycleJobReschedule(next_at)
        completed = await self._await_dispatch_budget(task)
        next_at = self._task_store().authoritative_deadline(run_id)
        if not completed and next_at is None:
            next_at = now_ms() + self._claim_timeout_ms()
        return None if next_at is None else LifecycleJobReschedule(next_at)

    async def on_job_error(
        self,
        context: LifecycleJobContext,
        error: BaseException,
    ) -> LifecycleJobOutcome:
        if _is_platform_failure(error):
            raise error
        await self._observe_error(error)
        return "yield"

    async def on_memory_limit(self, context: LifecycleMemoryLimitContext) -> None:
        job = context.executing
        if job is None or job.capability != self.capability_id:
            return
        try:
            run_id, owner = _task_job_identity(job)
        except (TypeError, ValueError) as error:
            await self._observe_error(error)
            return
        if owner is None:
            active = self._active_runs.get(run_id)
            row = self._task_store().get(run_id)
            expected_generation = active.generation if active is not None else None
            if expected_generation is None and row is not None:
                expected_generation = row.generation
            if expected_generation is None:
                return
            await self._apply_memory_policy(
                run_id,
                expected_generation,
                sealed=context.sealed,
                next_at=context.next_time,
                sync_wake=True,
            )
            return

        key = (owner.key, run_id)
        expected_generation = self._routed_wake_generations.get(key)
        if expected_generation is None:
            retry_at = context.next_time
            if retry_at is None:
                retry_at = now_ms() + _MEMORY_POLICY_RETRY_MS
            await self._sync_root_wake(owner, run_id, retry_at)
            return
        try:
            await self._persist_memory_policy(
                owner,
                run_id,
                expected_generation,
                sealed=context.sealed,
                next_at=context.next_time,
            )
        except asyncio.CancelledError:
            self._forget_routed_attempt(key, expected_generation)
            retry_at = max(
                context.next_time or 0,
                now_ms() + _MEMORY_POLICY_RETRY_MS,
            )
            await self._sync_root_wake(owner, run_id, retry_at)
            raise
        except BaseException as error:
            if not _is_platform_failure(error):
                await self._observe_error(error)
            retry_at = context.next_time
            if retry_at is None:
                retry_at = now_ms() + _MEMORY_POLICY_RETRY_MS
            await self._sync_root_wake(owner, run_id, retry_at)
            return
        self._forget_routed_attempt(key, expected_generation)
        try:
            await self._deliver_routed_memory_policy(
                owner,
                run_id,
                expected_generation,
                context.sealed,
                context.next_time,
            )
        except asyncio.CancelledError:
            retry_at = max(
                context.next_time or 0,
                now_ms() + _MEMORY_POLICY_RETRY_MS,
            )
            await self._sync_root_wake(owner, run_id, retry_at)
            raise
        except BaseException as error:
            if not _is_platform_failure(error):
                await self._observe_error(error)
            retry_at = max(
                context.next_time or 0,
                now_ms() + _MEMORY_POLICY_RETRY_MS,
            )
            await self._sync_root_wake(owner, run_id, retry_at)
            return
        await self._clear_memory_policy(owner, run_id)

    def _forget_routed_attempt(
        self,
        key: tuple[str, str],
        generation: str,
    ) -> None:
        if self._routed_wake_generations.get(key) == generation:
            self._routed_wake_generations.pop(key, None)
        task = self._routed_wakes.get(key)
        if task is not None and task.done():
            self._routed_wakes.pop(key, None)

    async def _apply_memory_policy(
        self,
        run_id: str,
        expected_generation: str,
        *,
        sealed: bool,
        next_at: int | None,
        sync_wake: bool = False,
    ) -> tuple[int | None, _TaskMemoryPolicyStatus]:
        store = self._task_store()
        row = store.get(run_id)
        if row is None:
            return None, "already"
        if row.state in _TERMINAL_STATES:
            return None, "already"
        if row.generation != expected_generation:
            status: _TaskMemoryPolicyStatus = "stale"
            if row.generation is None:
                status = (
                    "already"
                    if row.state == "running" and row.wait_reason == "memory"
                    else "preclaim"
                )
            return store.authoritative_deadline(run_id), status
        active = self._active_runs.get(run_id)
        if (
            active is not None
            and active.generation == expected_generation
            and not active.memory_failed
        ):
            return store.authoritative_deadline(run_id), "stale"

        current_time = now_ms()
        if row.cancel_requested:
            if store.settle_cancelled_claim(
                run_id,
                expected_generation,
                row.cancel_reason,
                current_time,
            ):
                await self._after_cancelled(
                    row,
                    row.cancel_reason,
                    sync_wake=sync_wake,
                )
        elif sealed:
            error = TaskMemoryLimitSealed(run_id)
            error_name, error_message = _error_summary(error)
            settled = store.settle_failed(
                run_id,
                expected_generation,
                error_name,
                error_message,
                current_time,
            )
            if settled:
                await self._emit(
                    "task:failed",
                    {
                        "runId": run_id,
                        "definition": row.definition,
                        "error": error_name,
                    },
                )
                store.delete_unretained_terminal(run_id)
                if sync_wake:
                    await self._sync_wake(run_id)
                await self._observe_error(error)
            else:
                await self._settle_requested_cancellation(
                    row,
                    expected_generation,
                )
        elif next_at is not None:
            backed_off = store.backoff_memory_claim(
                run_id,
                expected_generation,
                next_at,
                current_time,
            )
            if backed_off:
                if sync_wake:
                    await self._sync_wake(run_id)
            else:
                await self._settle_requested_cancellation(
                    row,
                    expected_generation,
                )

        if active is not None and self._active_runs.get(run_id) is active:
            self._active_runs.pop(run_id, None)
        return store.authoritative_deadline(run_id), "applied"

    async def _persist_memory_policy(
        self,
        owner: LifecycleRouteAddress,
        run_id: str,
        generation: str,
        *,
        sealed: bool,
        next_at: int | None,
    ) -> None:
        key = _memory_policy_key(owner, run_id)
        value = _TaskMemoryPolicyRecord(
            runId=run_id,
            owner_path=owner.data,
            owner_path_key=owner.key,
            generation=generation,
            sealed=sealed,
            nextAt=next_at,
        )
        last_error: BaseException | None = None
        cancelled: asyncio.CancelledError | None = None
        persisted = False
        for _ in range(3):
            try:
                await self.lifecycle.storage.put(key, value)
            except asyncio.CancelledError as error:
                cancelled = error
            except BaseException as error:
                last_error = error
            else:
                persisted = True
                break
        if persisted:
            if cancelled is not None:
                raise cancelled
            return
        if cancelled is not None and last_error is None:
            raise cancelled
        if last_error is not None:
            raise last_error

    async def _pending_memory_policy(
        self,
        owner: LifecycleRouteAddress,
        run_id: str,
    ) -> tuple[str, bool, int | None] | None:
        value = await self.lifecycle.storage.get(_memory_policy_key(owner, run_id))
        if value is None:
            return None
        try:
            return _task_memory_policy(value, owner, run_id)
        except ValueError as error:
            await self._clear_memory_policy(owner, run_id)
            await self._observe_error(error)
            return None

    async def _deliver_routed_memory_policy(
        self,
        owner: LifecycleRouteAddress,
        run_id: str,
        generation: str,
        sealed: bool,
        next_at: int | None,
    ) -> tuple[int | None, _TaskMemoryPolicyStatus]:
        response = await self.lifecycle.routes.to(
            owner,
            _task_memory_route(run_id, generation, sealed, next_at),
        )
        current_deadline, status = _task_memory_result(response)
        if status == "preclaim":
            current_deadline = max(
                next_at or 0,
                now_ms() + _MEMORY_POLICY_RETRY_MS,
            )
        await self._sync_root_wake(owner, run_id, current_deadline)
        return current_deadline, status

    async def _clear_memory_policy(
        self,
        owner: LifecycleRouteAddress,
        run_id: str,
    ) -> None:
        await self.lifecycle.storage.delete(_memory_policy_key(owner, run_id))

    async def on_route(self, context: LifecycleRouteContext) -> object:
        message = context.payload
        if not isinstance(message, dict) or type(message.get("type")) is not str:
            raise ValueError("invalid Tasks route message")
        route_type = message["type"]
        local_owner = self.lifecycle.routes.source
        if route_type in ("sync", "repair"):
            if local_owner is not None or context.source is None:
                raise PermissionError("Task wake sync must route from a facet to root")
            run_id, next_at = _task_sync_message(message)
            await self._sync_root_wake(
                context.source,
                run_id,
                next_at,
                repair=route_type == "repair",
            )
            if route_type == "repair":
                await self.lifecycle.jobs.rearm()
            return True
        if route_type == "memory":
            if local_owner is None or context.source is not None:
                raise PermissionError(
                    "Task memory policy must route from root to a facet"
                )
            run_id, generation, sealed, next_at = _task_memory_message(message)
            current_deadline, status = await self._apply_memory_policy(
                run_id,
                generation,
                sealed=sealed,
                next_at=next_at,
                sync_wake=False,
            )
            if status == "applied":
                await self.lifecycle._notify_host_memory_limit(
                    LifecycleMemoryLimitContext(
                        sealed=sealed,
                        next_time=next_at,
                        executing=None,
                        purged_recovery_loop_jobs=(),
                    )
                )
            return _task_memory_result_wire(current_deadline, status)
        if route_type == "cleanup":
            if local_owner is None or context.source is not None:
                raise PermissionError(
                    "Task cleanup acknowledgement must route from root to a facet"
                )
            if set(message) != {"type", "runIds"} or not isinstance(
                message["runIds"], list
            ):
                raise ValueError("invalid routed Task cleanup acknowledgement")
            run_ids = tuple(_required_run_id(value) for value in message["runIds"])
            for cleanup_run_id in run_ids:
                self._task_store().delete_unretained_terminal(cleanup_run_id)
                self._reconciliation_run_ids.discard(cleanup_run_id)
            return True
        if route_type == "extend":
            if local_owner is None or context.source is not None:
                raise PermissionError(
                    "Task claim extension must route from root to a facet"
                )
            if set(message) != {"type", "runId"}:
                raise ValueError("invalid routed Task claim extension")
            run_id = _required_run_id(message.get("runId"))
            await self._extend_active_claim(run_id, sync_wake=False)
            return _task_deadline_result_wire(
                self._task_store().authoritative_deadline(run_id)
            )
        if route_type != "wake":
            raise ValueError(f"unknown Tasks route message: {route_type!r}")
        if local_owner is None or context.source is not None:
            raise PermissionError("Task wake must route from root to a facet")
        if set(message) != {"type", "runId", "generation"}:
            raise ValueError("invalid routed Task wake")
        run_id = _required_run_id(message.get("runId"))
        generation = _required_generation(message.get("generation"))
        was_pending = self._reconciliation_pending
        self._reconciliation_pending = False
        routed_updates: set[str] = set()
        token = _CURRENT_ROUTED_TASK_WAKE.set(routed_updates)
        try:
            await self._extend_active_claim(run_id)
            await self._execution_task(run_id, generation)
            current_deadline = self._task_store().authoritative_deadline(run_id)
            repair_ids = routed_updates - {run_id}
            cleanup_ids: tuple[str, ...] = ()
            if was_pending:
                cleanup_ids = tuple(sorted(self._reconciliation_run_ids))
                repair_ids.update(cleanup_ids)
                repair_ids.discard(run_id)
                repair_ids.update(
                    routed_run_id
                    for routed_run_id, _ in (
                        self._task_store().non_terminal_deadlines()
                    )
                    if routed_run_id != run_id
                )
            repairs = tuple(
                (
                    routed_run_id,
                    self._task_store().authoritative_deadline(routed_run_id),
                )
                for routed_run_id in sorted(repair_ids)
            )
        except BaseException:
            if was_pending:
                self._reconciliation_pending = True
            raise
        finally:
            _CURRENT_ROUTED_TASK_WAKE.reset(token)
        return _RoutedTaskWakeResult(
            nextAt=current_deadline,
            repairs=[
                _TaskRepairRecord(runId=routed_run_id, nextAt=next_at)
                for routed_run_id, next_at in repairs
            ],
            cleanups=list(cleanup_ids),
        )

    async def cleanup_route_prefix(
        self,
        prefix: str,
        owner_path: str | None = None,
    ) -> None:
        tables = self.lifecycle.sql.execute(
            "SELECT 1 FROM sqlite_master "
            "WHERE type = 'table' AND name = 'cf_agents_jobs'"
        )
        rows = (
            self.lifecycle.sql.execute(
                "SELECT id, payload FROM cf_agents_jobs WHERE capability = ?",
                self.capability_id,
            )
            if tables
            else []
        )
        for row in rows:
            try:
                payload = _strict_task_job_payload(row["payload"])
            except (TypeError, ValueError):
                continue
            owner_key = payload.get("owner_path_key")
            key_matches = isinstance(owner_key, str) and (
                owner_key == prefix or owner_key.startswith(f"{prefix}/")
            )
            path_matches = owner_path is not None and _owner_path_has_prefix(
                payload.get("owner_path"),
                owner_path,
            )
            if key_matches or path_matches:
                await self.lifecycle.jobs.cancel(cast(str, row["id"]))

        policies = await self.lifecycle.storage.list(prefix=_MEMORY_POLICY_PREFIX)
        delete_keys = []
        for key, value in policies.items():
            if not isinstance(value, dict):
                continue
            owner_key = value.get("owner_path_key")
            key_matches = isinstance(owner_key, str) and (
                owner_key == prefix or owner_key.startswith(f"{prefix}/")
            )
            path_matches = owner_path is not None and _owner_path_has_prefix(
                value.get("owner_path"),
                owner_path,
            )
            if key_matches or path_matches:
                delete_keys.append(key)
        if delete_keys:
            await self.lifecycle.storage.delete(delete_keys)

    def _register_reserved_definition(
        self,
        name: str,
        definition: _TaskDefinition,
    ) -> None:
        _validate_definition_name(name)
        if not name.startswith("__cf"):
            raise ValueError("reserved Task definitions must use the '__cf' prefix")
        if name in self._definitions or name in self._reserved_definitions:
            raise ValueError(f"Task definition {name!r} is already registered")
        self._reserved_definitions[name] = definition

    async def _run_reserved(
        self,
        definition: str,
        input: object = None,
        *,
        run_id: str | None = None,
        idempotency_key: str | None = None,
        metadata: Mapping[str, TaskJson] | None = None,
        retain: bool = True,
    ) -> TaskReceipt:
        if definition not in self._reserved_definitions:
            raise ValueError(f"unknown reserved Task definition: {definition!r}")
        return await self._accept(
            definition,
            input,
            run_id=run_id,
            idempotency_key=idempotency_key,
            metadata=metadata,
            retain=retain,
        )

    async def _snapshot(
        self,
        run_id: str,
        definition: str | None = None,
    ) -> TaskRunSnapshot[object] | None:
        await self._ready()
        row = self._task_store().get(run_id)
        if row is None or definition is not None and row.definition != definition:
            return None
        return _snapshot(row)

    async def _snapshot_by_idempotency_key(
        self,
        key: str,
        definition: str | None = None,
    ) -> TaskRunSnapshot[object] | None:
        await self._ready()
        row = self._task_store().get_by_idempotency_key(key)
        if row is None or definition is not None and row.definition != definition:
            return None
        return _snapshot(row)

    async def _cancel_scoped(
        self,
        run_id: str,
        definition: str,
        reason: str | None,
    ) -> bool:
        await self._ready()
        row = self._task_store().get(run_id)
        if row is None or row.definition != definition:
            return False
        return await self.cancel(run_id, reason)

    async def _accept(
        self,
        definition: str,
        input: object,
        *,
        run_id: str | None,
        idempotency_key: str | None,
        metadata: Mapping[str, TaskJson] | None,
        retain: bool,
    ) -> TaskReceipt:
        await self._ready()
        _validate_optional_identifier(run_id, "run_id")
        _validate_optional_identifier(idempotency_key, "idempotency_key")
        if type(retain) is not bool:
            raise TypeError("retain must be a bool")
        _default_step_policy(self._retries, self._step_timeout)

        input_json = _serialize_task_value(
            input,
            f"input for Task definition {definition!r}",
        )
        metadata_json = _serialize_metadata(metadata, definition)
        store = self._task_store()
        existing = store.get(run_id) if run_id is not None else None
        if existing is None and idempotency_key is not None:
            existing = store.get_by_idempotency_key(idempotency_key)
        if existing is not None:
            receipt = self._joined_receipt(existing, definition, idempotency_key)
            await self._repair_wake(existing.run_id)
            if existing.state not in _TERMINAL_STATES:
                self._submit_run(existing.run_id)
            return receipt

        accepted_id = run_id or f"task_{gen_id()}"
        created_at = now_ms()
        store.insert_pending(
            run_id=accepted_id,
            definition=definition,
            input_json=input_json,
            metadata_json=metadata_json,
            idempotency_key=idempotency_key,
            retain=retain,
            created_at=created_at,
        )
        try:
            await self._sync_wake(accepted_id)
        except TaskWakeCollisionError:
            store.delete_unaccepted(accepted_id)
            raise
        await self._emit(
            "task:accepted",
            {"runId": accepted_id, "definition": definition, "accepted": True},
        )
        self._submit_run(accepted_id)
        return TaskReceipt(
            run_id=accepted_id,
            definition=definition,
            accepted=True,
            state="pending",
            created_at=created_at,
        )

    def _joined_receipt(
        self,
        row: _TaskRunRow,
        definition: str,
        idempotency_key: str | None,
    ) -> TaskReceipt:
        if row.definition != definition:
            raise ValueError(
                f"Task run {row.run_id!r} belongs to definition "
                f"{row.definition!r}, not {definition!r}"
            )
        if idempotency_key is not None and row.idempotency_key != idempotency_key:
            raise ValueError(
                f"Task run {row.run_id!r} has a conflicting idempotency key"
            )
        return TaskReceipt(
            run_id=row.run_id,
            definition=row.definition,
            accepted=False,
            state=row.state,
            created_at=row.created_at,
        )

    def _validate_public_definition(self, definition: str) -> None:
        _validate_definition_name(definition)
        if definition.startswith("__cf"):
            raise ValueError("public Task definitions cannot use the '__cf' prefix")
        if definition not in self._definitions:
            raise ValueError(f"unknown Task definition: {definition!r}")

    def _task_store(self) -> _TaskStore:
        if self._store is None:
            self._store = _TaskStore(
                self.lifecycle.sql,
                self.lifecycle.storage.transaction_sync,
            )
        return self._store

    async def _emit(self, type: TaskEventType, payload: object) -> None:
        await self.lifecycle.events.emit(type, payload)


def _sleep_until_ms(value: int | float | datetime, step_name: str) -> int:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        try:
            milliseconds = value.timestamp() * 1_000
        except (OverflowError, OSError, ValueError) as error:
            raise ValueError(
                f"Invalid sleep_until time for step {step_name!r}"
            ) from error
    elif isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("sleep_until time must be milliseconds or a datetime")
    else:
        milliseconds = value
    if not math.isfinite(milliseconds):
        raise ValueError(f"Invalid sleep_until time for step {step_name!r}")
    wake_at = math.floor(milliseconds)
    if wake_at > _MAX_SQLITE_INTEGER:
        raise ValueError(f"Invalid sleep_until time for step {step_name!r}")
    return wake_at


def _validate_deadline(value: int, context: str) -> None:
    if type(value) is not int or value < 0 or value > _MAX_SQLITE_INTEGER:
        raise ValueError(f"Invalid {context}")


def _task_job_id(owner: LifecycleRouteAddress | None, run_id: str) -> str:
    return f"task:{run_id}" if owner is None else f"task:{owner.key}:{run_id}"


def _task_job_payload(
    owner: LifecycleRouteAddress | None,
    run_id: str,
) -> _TaskWakePayload:
    if owner is None:
        return _LocalTaskWakePayload(runId=run_id)
    return _RoutedTaskWakePayload(
        runId=run_id,
        owner_path=owner.data,
        owner_path_key=owner.key,
    )


def _task_job_matches(
    job: LifecycleJob,
    job_id: str,
    next_at: int,
    payload: Mapping[str, object],
) -> bool:
    return (
        job.id == job_id
        and job.fn == "wake"
        and job.time == next_at
        and job.payload_present
        and job.payload == payload
        and job.retry == _WAKE_RETRY
        and not job.singleflight
        and not job.exclusive
        and not job.recovery_loop
    )


def _task_job_identity(
    job: LifecycleJob,
) -> tuple[str, LifecycleRouteAddress | None]:
    if job.capability != "tasks" or job.fn != "wake" or not job.payload_present:
        raise ValueError("invalid Task wake job")
    if not isinstance(job.payload, dict):
        raise ValueError("Task wake payload must be an object")
    payload = job.payload
    if set(payload) == {"runId"}:
        run_id = _required_run_id(payload.get("runId"))
        owner = None
    elif set(payload) == {"runId", "owner_path", "owner_path_key"}:
        run_id = _required_run_id(payload.get("runId"))
        owner_path = payload.get("owner_path")
        owner_key = payload.get("owner_path_key")
        if not isinstance(owner_path, str) or not isinstance(owner_key, str):
            raise ValueError("invalid routed Task wake owner")
        owner = LifecycleRouteAddress(owner_key, owner_path)
    else:
        raise ValueError("invalid Task wake payload")
    if job.id != _task_job_id(owner, run_id):
        raise ValueError("Task wake job ID does not match its payload")
    return run_id, owner


def _task_sync_message(message: Mapping[str, object]) -> tuple[str, int | None]:
    if set(message) != {"type", "runId", "nextAt"}:
        raise ValueError("invalid Task wake sync")
    run_id = _required_run_id(message.get("runId"))
    next_at = message.get("nextAt")
    if next_at is not None:
        _validate_deadline(cast(int, next_at), "Task wake deadline")
        next_at = cast(int, next_at)
    return run_id, next_at


def _task_memory_message(
    message: Mapping[str, object],
) -> tuple[str, str, bool, int | None]:
    if set(message) != {"type", "runId", "generation", "sealed", "nextAt"}:
        raise ValueError("invalid Task memory policy")
    run_id = _required_run_id(message.get("runId"))
    generation = _required_generation(message.get("generation"))
    return (
        run_id,
        generation,
        *_task_memory_state(
            message.get("sealed"),
            message.get("nextAt"),
        ),
    )


def _task_memory_result(
    value: object,
) -> tuple[int | None, _TaskMemoryPolicyStatus]:
    if not isinstance(value, dict) or set(value) != {"nextAt", "status"}:
        raise ValueError("invalid routed Task memory result")
    status = value["status"]
    if status not in {"applied", "already", "preclaim", "stale"}:
        raise ValueError("invalid routed Task memory result status")
    return _task_deadline_result(
        {"nextAt": value["nextAt"]},
        "routed Task memory",
    ), cast(_TaskMemoryPolicyStatus, status)


def _task_deadline_result(value: object, context: str) -> int | None:
    if not isinstance(value, dict) or set(value) != {"nextAt"}:
        raise ValueError(f"invalid {context} result")
    next_at = value["nextAt"]
    if next_at is not None:
        _validate_deadline(cast(int, next_at), f"{context} deadline")
        return cast(int, next_at)
    return None


def _task_memory_policy(
    value: object,
    owner: LifecycleRouteAddress,
    run_id: str,
) -> tuple[str, bool, int | None]:
    expected = {
        "runId",
        "owner_path",
        "owner_path_key",
        "generation",
        "sealed",
        "nextAt",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("invalid stored Task memory policy")
    if (
        value.get("runId") != run_id
        or value.get("owner_path") != owner.data
        or value.get("owner_path_key") != owner.key
    ):
        raise ValueError("stored Task memory policy does not match its wake")
    generation = _required_generation(value.get("generation"))
    sealed, next_at = _task_memory_state(value.get("sealed"), value.get("nextAt"))
    return generation, sealed, next_at


def _stored_task_memory_policy(
    key: str,
    value: object,
) -> tuple[LifecycleRouteAddress, str, str, bool, int | None]:
    if not isinstance(value, dict):
        raise ValueError("invalid stored Task memory policy")
    run_id = _required_run_id(value.get("runId"))
    owner_path = value.get("owner_path")
    owner_key = value.get("owner_path_key")
    if not isinstance(owner_path, str) or not isinstance(owner_key, str):
        raise ValueError("invalid stored Task memory policy owner")
    owner = LifecycleRouteAddress(owner_key, owner_path)
    if key != _memory_policy_key(owner, run_id):
        raise ValueError("stored Task memory policy key does not match its wake")
    generation, sealed, next_at = _task_memory_policy(value, owner, run_id)
    return owner, run_id, generation, sealed, next_at


def _task_memory_state(
    sealed_value: object,
    next_value: object,
) -> tuple[bool, int | None]:
    if type(sealed_value) is not bool:
        raise ValueError("Task memory sealed must be a bool")
    sealed = sealed_value
    if next_value is None:
        if not sealed:
            raise ValueError("unsealed Task memory policy requires a deadline")
        return sealed, None
    if sealed:
        raise ValueError("sealed Task memory policy cannot carry a deadline")
    _validate_deadline(cast(int, next_value), "Task memory deadline")
    return sealed, cast(int, next_value)


def _memory_policy_key(owner: LifecycleRouteAddress, run_id: str) -> str:
    return f"{_MEMORY_POLICY_PREFIX}{_task_job_id(owner, run_id)}"


def _required_run_id(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("Task run ID must be a non-empty string")
    return value


def _required_generation(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("Task generation must be a non-empty string")
    return value


def _routed_deadlines(
    value: object,
) -> tuple[int | None, tuple[tuple[str, int | None], ...], tuple[str, ...]]:
    if not isinstance(value, dict) or set(value) != {
        "nextAt",
        "repairs",
        "cleanups",
    }:
        raise ValueError("invalid routed Task wake result")
    current_deadline = value["nextAt"]
    if current_deadline is not None:
        _validate_deadline(cast(int, current_deadline), "routed Task deadline")
        current_deadline = cast(int, current_deadline)
    repairs = value["repairs"]
    if not isinstance(repairs, list):
        raise ValueError("routed Task repairs must be a list")
    resolved: list[tuple[str, int | None]] = []
    seen: set[str] = set()
    for item in repairs:
        if not isinstance(item, dict) or set(item) != {"runId", "nextAt"}:
            raise ValueError("invalid routed Task deadline")
        run_id = _required_run_id(item.get("runId"))
        if run_id in seen:
            raise ValueError("duplicate routed Task deadline")
        next_at = item.get("nextAt")
        if next_at is not None:
            _validate_deadline(cast(int, next_at), "routed Task deadline")
            next_at = cast(int, next_at)
        seen.add(run_id)
        resolved.append((run_id, next_at))
    cleanups = value["cleanups"]
    if not isinstance(cleanups, list):
        raise ValueError("routed Task cleanups must be a list")
    resolved_cleanups = tuple(_required_run_id(item) for item in cleanups)
    if len(set(resolved_cleanups)) != len(resolved_cleanups):
        raise ValueError("duplicate routed Task cleanup")
    return current_deadline, tuple(resolved), resolved_cleanups


def _strict_task_job_payload(value: object) -> dict[str, object]:
    if not isinstance(value, str):
        raise TypeError("stored Task wake payload must be JSON")
    try:
        decoded = strict_json_loads(value, "stored Task wake payload")
    except ValueError as error:
        raise ValueError("invalid stored Task wake payload") from error
    if not isinstance(decoded, dict):
        raise ValueError("stored Task wake payload must be an object")
    return decoded


def _owner_path_has_prefix(candidate: object, prefix: str) -> bool:
    if not isinstance(candidate, str):
        return False
    try:
        candidate_path = strict_json_loads(candidate, "Task owner path")
        prefix_path = strict_json_loads(prefix, "Task owner path prefix")
    except ValueError:
        return False
    return (
        isinstance(candidate_path, list)
        and isinstance(prefix_path, list)
        and candidate_path[: len(prefix_path)] == prefix_path
    )


def _default_step_policy(
    retries: TaskStepRetryOptions | None,
    step_timeout: int | float | TaskDurationString | None,
) -> _ResolvedStepPolicy:
    retry_limit = retries.limit if retries is not None else None
    if retry_limit is None:
        retry_limit = 5
    _validate_retry_limit(retry_limit)
    retry_delay = retries.delay if retries is not None else None
    retry_delay_ms = (
        1_000
        if retry_delay is None
        else _parse_task_duration(retry_delay, "retries.delay")
    )
    backoff = retries.backoff if retries is not None else None
    if backoff is None:
        backoff = "exponential"
    _validate_backoff(backoff)
    timeout_ms = (
        300_000
        if step_timeout is None
        else _parse_task_duration(step_timeout, "step_timeout")
    )
    return _ResolvedStepPolicy(
        retry_limit=retry_limit,
        retry_delay_ms=retry_delay_ms,
        backoff=backoff,
        timeout_ms=timeout_ms,
    )


def _resolve_step_policy(
    defaults: _ResolvedStepPolicy,
    config: TaskStepConfig | None,
) -> _ResolvedStepPolicy:
    if config is None:
        return defaults
    retries = config.retries
    retry_limit = retries.limit if retries is not None else None
    if retry_limit is None:
        retry_limit = defaults.retry_limit
    _validate_retry_limit(retry_limit)
    delay = retries.delay if retries is not None else None
    retry_delay_ms = (
        defaults.retry_delay_ms
        if delay is None
        else _parse_task_duration(delay, "step retries.delay")
    )
    backoff = retries.backoff if retries is not None else None
    if backoff is None:
        backoff = defaults.backoff
    _validate_backoff(backoff)
    timeout_ms = (
        defaults.timeout_ms
        if config.timeout is None
        else _parse_task_duration(config.timeout, "step timeout")
    )
    return _ResolvedStepPolicy(
        retry_limit=retry_limit,
        retry_delay_ms=retry_delay_ms,
        backoff=backoff,
        timeout_ms=timeout_ms,
    )


def _validate_retry_limit(value: int) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(
            f"Invalid step retries.limit: expected an integer >= 1, got {value!r}"
        )


def _validate_backoff(value: str) -> None:
    if value not in ("constant", "linear", "exponential"):
        raise ValueError(f"Invalid step retries.backoff: {value!r}")


def _parse_task_duration(value: int | float | str, context: str) -> int:
    if type(value) in (int, float):
        numeric = cast(int | float, value)
        if not math.isfinite(numeric) or numeric < 0:
            raise ValueError(
                f"Invalid {context}: expected non-negative milliseconds, got {value!r}"
            )
        return math.floor(numeric)
    if not isinstance(value, str):
        raise TypeError(f"Invalid {context}: expected milliseconds or a duration")
    match = _DURATION_PATTERN.fullmatch(value.strip())
    if match is None:
        raise ValueError(
            f'Invalid {context}: expected milliseconds or a duration like "10 seconds"'
        )
    milliseconds = float(match.group(1)) * _DURATION_UNIT_MS[match.group(2)]
    if not math.isfinite(milliseconds):
        raise ValueError(f"Invalid {context}: duration is too large")
    return math.floor(milliseconds)


def _validate_step_name(name: str) -> None:
    if not isinstance(name, str) or not name:
        raise ValueError("Task step names must be non-empty strings")
    length = len(name.encode("utf-16-le", errors="surrogatepass")) // 2
    if length > _MAX_DEFINITION_NAME_LENGTH:
        raise ValueError(
            f"Task step names cannot exceed {_MAX_DEFINITION_NAME_LENGTH} characters"
        )
    if name.startswith("__cf"):
        raise ValueError("Task step names cannot use the reserved '__cf' prefix")


def _retry_delay(policy: _ResolvedStepPolicy, failed_attempt: int) -> int:
    delay = policy.retry_delay_ms
    if policy.backoff == "linear":
        delay *= failed_attempt
    elif policy.backoff == "exponential":
        if delay == 0:
            return 0
        exponent = max(0, failed_attempt - 1)
        if exponent >= math.ceil(_MAX_RETRY_DELAY_MS / delay).bit_length():
            return _MAX_RETRY_DELAY_MS
        delay *= 2**exponent
    return min(delay, _MAX_RETRY_DELAY_MS)


def _error_summary(error: BaseException) -> tuple[str, str]:
    name = getattr(error, "name", type(error).__name__)
    if not isinstance(name, str) or not name:
        name = type(error).__name__
    return name, str(error)


def _is_non_retryable(error: BaseException) -> bool:
    name, _ = _error_summary(error)
    return isinstance(error, (NonRetryableError, TaskSerializationError)) or (
        name == "NonRetryableError"
    )


async def _observe_late_callback(task: asyncio.Task[Any]) -> None:
    try:
        await task
    except BaseException:
        return


def _snapshot(row: _TaskRunRow) -> TaskRunSnapshot[object]:
    metadata = _deserialize_metadata(row.metadata)
    if row.state == "pending":
        return PendingTaskRunSnapshot(
            run_id=row.run_id,
            definition=row.definition,
            state="pending",
            created_at=row.created_at,
            metadata=metadata,
        )
    if row.state == "running":
        return RunningTaskRunSnapshot(
            run_id=row.run_id,
            definition=row.definition,
            state="running",
            attempt=row.attempt,
            started_at=(
                row.started_at if row.started_at is not None else row.created_at
            ),
            created_at=row.created_at,
            status_message=row.status_message,
            metadata=metadata,
        )
    if row.state == "waiting":
        return WaitingTaskRunSnapshot(
            run_id=row.run_id,
            definition=row.definition,
            state="waiting",
            reason=cast(TaskWaitReason, row.wait_reason or "sleep"),
            wake_at=row.next_at if row.next_at is not None else row.updated_at,
            created_at=row.created_at,
            status_message=row.status_message,
            metadata=metadata,
        )
    if row.state == "completed":
        return CompletedTaskRunSnapshot(
            run_id=row.run_id,
            definition=row.definition,
            state="completed",
            result=_deserialize_task_value(row.result, "Task result"),
            created_at=row.created_at,
            settled_at=(
                row.settled_at if row.settled_at is not None else row.updated_at
            ),
            metadata=metadata,
        )
    if row.state == "failed":
        return FailedTaskRunSnapshot(
            run_id=row.run_id,
            definition=row.definition,
            state="failed",
            error=TaskError(
                name=(row.error_name if row.error_name is not None else "Error"),
                message=(
                    row.error_message
                    if row.error_message is not None
                    else "Task run failed"
                ),
            ),
            created_at=row.created_at,
            settled_at=(
                row.settled_at if row.settled_at is not None else row.updated_at
            ),
            metadata=metadata,
        )
    return CancelledTaskRunSnapshot(
        run_id=row.run_id,
        definition=row.definition,
        state="cancelled",
        created_at=row.created_at,
        settled_at=row.settled_at if row.settled_at is not None else row.updated_at,
        reason=row.cancel_reason,
        metadata=metadata,
    )


def _validate_definition_name(name: str) -> None:
    if not isinstance(name, str) or not name:
        raise ValueError("Task definition names must be non-empty strings")
    length = len(name.encode("utf-16-le", errors="surrogatepass")) // 2
    if length > _MAX_DEFINITION_NAME_LENGTH:
        raise ValueError(
            f"Task definition names cannot exceed {_MAX_DEFINITION_NAME_LENGTH} "
            "characters"
        )


def _validate_optional_identifier(value: str | None, name: str) -> None:
    if value is None:
        return
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value:
        raise ValueError(f"{name} must be non-empty")


def _validate_limit(value: int) -> int:
    if type(value) is not int:
        raise TypeError("Task list limits must be integers")
    if value < 1 or value > _MAX_LIST_LIMIT:
        raise ValueError(f"Task list limits must be between 1 and {_MAX_LIST_LIMIT}")
    return value


def _normalize_states(
    value: TaskRunState | Sequence[TaskRunState] | None,
) -> tuple[TaskRunState, ...]:
    if value is None:
        values: tuple[TaskRunState, ...] = ()
    elif isinstance(value, str):
        values = (cast(TaskRunState, value),)
    else:
        values = tuple(value)
    for state in values:
        if state not in _RUN_STATES:
            raise ValueError(f"unknown Task state: {state!r}")
    return values


def _normalize_terminal_states(
    value: Sequence[TaskTerminalState] | None,
) -> tuple[TaskTerminalState, ...]:
    values = (
        tuple(value)
        if value is not None
        else (
            "completed",
            "failed",
            "cancelled",
        )
    )
    for state in values:
        if state not in _TERMINAL_STATES:
            raise ValueError(f"cannot delete non-terminal Task state: {state!r}")
    return values


def _datetime_ms(value: datetime | None) -> int | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    milliseconds = value.timestamp() * 1_000
    if not math.isfinite(milliseconds) or milliseconds < 0:
        raise ValueError("settled_before must have a non-negative timestamp")
    return math.floor(milliseconds)


def _serialize_metadata(
    value: Mapping[str, TaskJson] | None,
    definition: str,
) -> str | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("Task metadata must be a mapping")
    metadata = dict(value)
    if any(not isinstance(key, str) for key in metadata):
        raise TaskSerializationError("Task metadata", "all keys must be strings")
    return _serialize_task_value(
        metadata,
        f"metadata for Task definition {definition!r}",
    )


def _serialize_task_value(value: object, context: str) -> str | None:
    if value is MISSING or isinstance(value, TaskUndefined):
        return None
    try:
        _validate_json_keys(value, context, set())
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    except (RecursionError, TypeError, ValueError) as error:
        raise TaskSerializationError(context, str(error)) from error
    encoded = _escape_json_surrogates(encoded)
    size = len(encoded.encode("utf-8"))
    if size > MAX_SERIALIZED_BYTES:
        raise TaskSerializationError(
            context,
            f"serialized size {size} bytes exceeds the "
            f"{MAX_SERIALIZED_BYTES}-byte limit",
        )
    return encoded


def _deserialize_task_value(value: str | None, context: str) -> object:
    if value is None:
        return TASK_UNDEFINED
    try:
        return strict_json_loads(value, f"persisted {context}")
    except ValueError as error:
        raise ValueError(f"invalid persisted {context}") from error


def _deserialize_metadata(value: str | None) -> dict[str, TaskJson] | None:
    if value is None:
        return None
    decoded = _deserialize_task_value(value, "Task metadata")
    if not isinstance(decoded, dict) or any(
        not isinstance(key, str) for key in decoded
    ):
        raise ValueError("persisted Task metadata must be a JSON object")
    return decoded


def _validate_json_keys(value: object, context: str, seen: set[int]) -> None:
    if isinstance(value, int) and not isinstance(value, bool):
        if value < -_MAX_SAFE_INTEGER or value > _MAX_SAFE_INTEGER:
            raise TaskSerializationError(
                context,
                "integers must fit in the JavaScript safe-integer range",
            )
        return
    if isinstance(value, dict):
        identity = id(value)
        if identity in seen:
            return
        seen.add(identity)
        for key, item in value.items():
            if not isinstance(key, str):
                raise TaskSerializationError(context, "all object keys must be strings")
            _validate_json_keys(item, context, seen)
        seen.remove(identity)
        return
    if isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in seen:
            return
        seen.add(identity)
        for item in value:
            _validate_json_keys(item, context, seen)
        seen.remove(identity)


def _escape_json_surrogates(value: str) -> str:
    escaped: list[str] = []
    index = 0
    while index < len(value):
        codepoint = ord(value[index])
        if 0xD800 <= codepoint <= 0xDBFF and index + 1 < len(value):
            low = ord(value[index + 1])
            if 0xDC00 <= low <= 0xDFFF:
                escaped.append(
                    chr(0x10000 + (codepoint - 0xD800) * 0x400 + low - 0xDC00)
                )
                index += 2
                continue
        if 0xD800 <= codepoint <= 0xDFFF:
            escaped.append(f"\\u{codepoint:04x}")
        else:
            escaped.append(value[index])
        index += 1
    return "".join(escaped)


__all__ = [
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
]
