from __future__ import annotations

import asyncio
import calendar
import inspect
import math
import re
import secrets
import warnings
from bisect import bisect_right
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Any, Literal, NotRequired, TypeVar, TypedDict, cast
from urllib.parse import quote

from .core._discovery import (
    deferred_method,
    static_definition_function,
    static_mro_members,
)
from .core._wire import strict_json_loads
from .core.utils import MISSING, dumps_wire, now_ms
from .lifecycle import (
    LifecycleCapability,
    LifecycleRouteAddress,
    LifecycleRouteContext,
)
from .lifecycle._job_driver import (
    _is_code_update_reset,
    _is_platform_failure,
    _retry_sleep,
)
from .lifecycle.jobs import (
    LifecycleJob,
    LifecycleJobContext,
    LifecycleJobOutcome,
    LifecycleJobPushOptions,
    LifecycleJobReschedule,
    LifecycleMemoryLimitContext,
    _validate_retry as _validate_lifecycle_retry,
)

type ScheduleType = Literal["scheduled", "delayed", "cron", "interval"]
type SchedulerEventType = Literal[
    "schedule:create",
    "schedule:cancel",
    "schedule:execute",
    "schedule:retry",
    "schedule:error",
    "schedule:duplicate_warning",
]
type SchedulerCallback = Callable[[object, "Schedule"], object | Awaitable[object]]
type SchedulerCallbacks = Mapping[str, SchedulerCallback]
type SchedulerHandlers = SchedulerCallbacks
type SchedulerPayload = object
type _LegacyScheduleRows = list["_PreparedLegacySchedule"]
type _LegacyJobOptions = list[LifecycleJobPushOptions]

__all__ = (
    "RetryOptions",
    "Schedule",
    "ScheduleCriteria",
    "ScheduleOptions",
    "ScheduleTimeRange",
    "Scheduler",
    "SchedulerCallbacks",
    "SchedulerEventType",
    "SchedulerHandlers",
    "SchedulerOptions",
    "SchedulerPayload",
    "scheduler_callback",
)


class _ScheduleJobPayload(TypedDict):
    type: ScheduleType
    owner_path: str | None
    owner_path_key: str | None
    payload: NotRequired[object]
    retry: NotRequired[dict[str, object]]
    delayInSeconds: NotRequired[int | float]
    cron: NotRequired[str]
    intervalSeconds: NotRequired[int | float]


class _ScheduleOptionsWire(TypedDict):
    retry: NotRequired[dict[str, object]]
    idempotent: NotRequired[bool]


class _ScheduleSetRoute(TypedDict):
    type: Literal["schedule"]
    when: object
    callback: str
    payload: NotRequired[object]
    options: NotRequired[_ScheduleOptionsWire]


class _ScheduleEveryRoute(TypedDict):
    type: Literal["every"]
    intervalSeconds: int | float
    callback: str
    payload: NotRequired[object]
    options: NotRequired[_ScheduleOptionsWire]


class _ScheduleIdRoute(TypedDict):
    type: Literal["get", "cancel"]
    id: str


class _ScheduleListRoute(TypedDict):
    type: Literal["list"]
    criteria: dict[str, object]


class _ScheduleDispatchRoute(TypedDict):
    type: Literal["dispatch"]
    id: str
    fn: str
    job: _ScheduleJobPayload
    time: int
    retry: dict[str, object] | None


class _ScheduleMemoryRoute(TypedDict):
    type: Literal["memory"]
    sealed: Literal[True]
    nextAt: None


class _ScheduleWire(TypedDict):
    id: str
    callback: str
    type: ScheduleType
    time: int
    payload: NotRequired[object]
    retry: NotRequired[dict[str, object]]
    delayInSeconds: NotRequired[int | float]
    cron: NotRequired[str]
    intervalSeconds: NotRequired[int | float]


class _ScheduleInsertResult(TypedDict):
    schedule: _ScheduleWire
    created: bool


class _ScheduleCancelResult(TypedDict):
    ok: bool
    callback: NotRequired[str]


_SCHEDULE_SCHEMA_VERSION_KEY = "cf_agents:schedules_schema_version"
_CURRENT_SCHEDULE_SCHEMA_VERSION = 2
_MAX_INTERVAL_SECONDS = 30 * 24 * 60 * 60
_MAX_SAFE_INTEGER = 2**53 - 1
_DEFAULT_RETRY = {
    "maxAttempts": 3,
    "baseDelayMs": 100,
    "maxDelayMs": 3_000,
}
_LEGACY_RECOVERY_LOOP_CALLBACKS = frozenset(
    {"_chatRecoveryContinue", "_chatRecoveryRetry"}
)
_SCHEDULER_METADATA_ATTR = "__agents_scheduler_callback__"
_MAX_SQLITE_INTEGER = 2**63 - 1
_URI_SAFE = "-_.!~*'()"
_ECMASCRIPT_WHITESPACE = (
    "\u0009\u000a\u000b\u000c\u000d\u0020\u00a0\u1680"
    "\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a"
    "\u2028\u2029\u202f\u205f\u3000\ufeff"
)
_CRON_NICKNAMES = {
    "@yearly": "0 0 1 1 *",
    "@annually": "0 0 1 1 *",
    "@monthly": "0 0 1 * *",
    "@weekly": "0 0 * * 0",
    "@daily": "0 0 * * *",
    "@hourly": "0 * * * *",
    "@minutely": "* * * * *",
}
_MONTH_ALIASES = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}
_WEEKDAY_ALIASES = {
    "mon": 1,
    "tue": 2,
    "wed": 3,
    "thu": 4,
    "fri": 5,
    "sat": 6,
    "sun": 7,
}
_CRON_RANGE = re.compile(r"^(?:([0-9A-Za-z]+)-([0-9A-Za-z]+)|(\*))(?:/([0-9]+))?$")
_INTEGER_PREFIX = re.compile(r"^[+-]?[0-9]+")


@dataclass(frozen=True, slots=True, kw_only=True)
class RetryOptions:
    """Override retry attempts and millisecond backoff bounds."""

    max_attempts: int | None = None
    base_delay_ms: int | float | None = None
    max_delay_ms: int | float | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class ScheduleOptions:
    """Configure retry and deduplication for a schedule."""

    retry: RetryOptions | None = None
    idempotent: bool | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class ScheduleTimeRange:
    """Filter schedules by an inclusive datetime range."""

    start: datetime | None = None
    end: datetime | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class ScheduleCriteria:
    """Filter schedules by ID, timing type, and time range."""

    id: str | None = None
    type: ScheduleType | None = None
    time_range: ScheduleTimeRange | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class SchedulerOptions:
    """Configure a Scheduler registry, retries, timeout, and error hook."""

    callbacks: SchedulerCallbacks | None = None
    retry: RetryOptions | None = None
    hung_schedule_timeout_seconds: int = 30
    on_error: Callable[[BaseException], object | Awaitable[object]] | None = None

    def __post_init__(self) -> None:
        if self.callbacks is not None:
            object.__setattr__(
                self,
                "callbacks",
                MappingProxyType(dict(self.callbacks)),
            )


@dataclass(frozen=True, slots=True)
class Schedule:
    """Describe one persisted Lifecycle-backed schedule."""

    id: str
    callback: str
    payload: object
    retry: RetryOptions | None
    type: ScheduleType
    time: int
    delay_in_seconds: int | float | None = None
    cron: str | None = None
    interval_seconds: int | float | None = None


@dataclass(frozen=True)
class _PreparedLegacySchedule:
    id: str
    callback: str
    time_ms: int
    payload: _ScheduleJobPayload
    retry: dict[str, object]
    singleflight: bool
    recovery_loop: bool


_SCHEDULER_CALLBACK_METADATA = object()


@dataclass(frozen=True, slots=True)
class _ParsedCron:
    seconds: tuple[int, ...]
    minutes: tuple[int, ...]
    hours: tuple[int, ...]
    days: tuple[int, ...]
    months: tuple[int, ...]
    weekdays: tuple[int, ...]


_F = TypeVar("_F", bound=Callable[..., Any])


def scheduler_callback() -> Callable[[_F], _F]:
    """Mark an Agent method for static Scheduler discovery."""

    def decorator(callback: _F) -> _F:
        function = static_definition_function(callback)
        if function is None:
            raise TypeError("scheduler_callback can only decorate methods")
        state = object.__getattribute__(callback, "__dict__")
        state[_SCHEDULER_METADATA_ATTR] = _SCHEDULER_CALLBACK_METADATA
        return callback

    return decorator


def _discover_scheduler_callbacks(
    instance: object,
    members: Mapping[str, object] | None = None,
) -> dict[str, SchedulerCallback]:
    callbacks: dict[str, SchedulerCallback] = {}
    for name, member in (members or static_mro_members(instance)).items():
        if _scheduler_callback_metadata(member) is None:
            continue
        callbacks[name] = cast(SchedulerCallback, deferred_method(member, instance))
    return callbacks


def _schedule_route_message(
    when: datetime | str | int | float,
    callback: str,
    payload: object,
    options: ScheduleOptions,
) -> _ScheduleSetRoute:
    message = _ScheduleSetRoute(
        type="schedule",
        when={"date": when.isoformat()} if isinstance(when, datetime) else when,
        callback=callback,
    )
    if payload is not MISSING:
        message["payload"] = payload
    encoded_options = _options_to_route(options)
    if encoded_options:
        message["options"] = encoded_options
    return message


def _every_route_message(
    interval_seconds: int | float,
    callback: str,
    payload: object,
    options: ScheduleOptions,
) -> _ScheduleEveryRoute:
    message = _ScheduleEveryRoute(
        type="every",
        intervalSeconds=interval_seconds,
        callback=callback,
    )
    if payload is not MISSING:
        message["payload"] = payload
    encoded_options = _options_to_route(options)
    if encoded_options:
        message["options"] = encoded_options
    return message


def _options_to_route(options: ScheduleOptions) -> _ScheduleOptionsWire:
    encoded = _ScheduleOptionsWire()
    retry = _retry_to_wire(options.retry)
    if retry is not None:
        encoded["retry"] = retry
    if options.idempotent is not None:
        encoded["idempotent"] = options.idempotent
    return encoded


def _schedule_id_route(
    route_type: Literal["get", "cancel"],
    schedule_id: str,
) -> _ScheduleIdRoute:
    return _ScheduleIdRoute(type=route_type, id=schedule_id)


def _schedule_list_route(criteria: ScheduleCriteria) -> _ScheduleListRoute:
    return _ScheduleListRoute(type="list", criteria=_criteria_to_route(criteria))


def _schedule_dispatch_route(
    job: LifecycleJob, timing: _ScheduleJobPayload
) -> _ScheduleDispatchRoute:
    return _ScheduleDispatchRoute(
        type="dispatch",
        id=job.id,
        fn=job.fn,
        job=timing,
        time=job.time,
        retry=job.retry,
    )


def _schedule_memory_route() -> _ScheduleMemoryRoute:
    return _ScheduleMemoryRoute(type="memory", sealed=True, nextAt=None)


def _options_from_route(value: object) -> ScheduleOptions:
    if value is None:
        return ScheduleOptions()
    if not isinstance(value, dict):
        raise ValueError("routed schedule options must be an object")
    unknown = set(value) - {"retry", "idempotent"}
    if unknown:
        raise ValueError(f"unknown routed schedule option: {sorted(unknown)[0]}")
    return ScheduleOptions(
        retry=_retry_from_wire(value.get("retry")),
        idempotent=cast(bool | None, value.get("idempotent")),
    )


def _when_from_route(value: object) -> datetime | str | int | float:
    if isinstance(value, dict) and set(value) == {"date"}:
        encoded = value["date"]
        if not isinstance(encoded, str):
            raise ValueError("routed schedule date must be a string")
        try:
            return datetime.fromisoformat(encoded)
        except ValueError as error:
            raise ValueError("routed schedule date is invalid") from error
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError("routed schedule time is invalid")
    return value


def _criteria_to_route(criteria: ScheduleCriteria) -> dict[str, object]:
    encoded: dict[str, object] = {}
    if criteria.id is not None:
        encoded["id"] = criteria.id
    if criteria.type is not None:
        encoded["type"] = criteria.type
    if criteria.time_range is not None:
        encoded["timeRange"] = {
            "start": (
                criteria.time_range.start.isoformat()
                if criteria.time_range.start is not None
                else None
            ),
            "end": (
                criteria.time_range.end.isoformat()
                if criteria.time_range.end is not None
                else None
            ),
        }
    return encoded


def _criteria_from_route(value: object) -> ScheduleCriteria:
    if value is None:
        return ScheduleCriteria()
    if not isinstance(value, dict):
        raise ValueError("routed schedule criteria must be an object")
    time_range = value.get("timeRange")
    parsed_range = None
    if time_range is not None:
        if not isinstance(time_range, dict):
            raise ValueError("routed schedule time range must be an object")
        parsed_range = ScheduleTimeRange(
            start=_optional_route_datetime(time_range.get("start")),
            end=_optional_route_datetime(time_range.get("end")),
        )
    return ScheduleCriteria(
        id=cast(str | None, value.get("id")),
        type=cast(ScheduleType | None, value.get("type")),
        time_range=parsed_range,
    )


def _optional_route_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("routed schedule range date must be a string")
    try:
        return datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError("routed schedule range date is invalid") from error


def _schedule_to_route(schedule: Schedule) -> _ScheduleWire:
    encoded = _ScheduleWire(
        id=schedule.id,
        callback=schedule.callback,
        type=schedule.type,
        time=schedule.time,
    )
    if schedule.payload is not MISSING:
        encoded["payload"] = schedule.payload
    retry = _retry_to_wire(schedule.retry)
    if retry is not None:
        encoded["retry"] = retry
    if schedule.delay_in_seconds is not None:
        encoded["delayInSeconds"] = schedule.delay_in_seconds
    if schedule.cron is not None:
        encoded["cron"] = schedule.cron
    if schedule.interval_seconds is not None:
        encoded["intervalSeconds"] = schedule.interval_seconds
    return encoded


def _schedule_insert_result(
    schedule: Schedule,
    created: bool,
) -> _ScheduleInsertResult:
    return _ScheduleInsertResult(
        schedule=_schedule_to_route(schedule),
        created=created,
    )


def _schedule_cancel_result(
    cancelled: bool,
    callback: str | None = None,
) -> _ScheduleCancelResult:
    result = _ScheduleCancelResult(ok=cancelled)
    if callback is not None:
        result["callback"] = callback
    return result


def _schedule_from_route(value: object) -> Schedule | None:
    if value is None:
        return None
    return _require_schedule_from_route(value)


def _require_schedule_from_route(value: object) -> Schedule:
    if not isinstance(value, dict):
        raise TypeError("routed schedule returned an invalid result")
    schedule_id = _route_id(value)
    callback = _route_callback(value)
    scheduled_time = value.get("time")
    if (
        type(scheduled_time) is not int
        or scheduled_time < 0
        or scheduled_time > _MAX_SQLITE_INTEGER // 1_000
    ):
        raise ValueError("routed schedule returned an invalid time")
    _validate_scheduler_payload(value, schedule_id)
    return Schedule(
        id=schedule_id,
        callback=callback,
        payload=value.get("payload", MISSING),
        retry=_retry_from_wire(value.get("retry")),
        type=cast(ScheduleType, value["type"]),
        time=scheduled_time,
        delay_in_seconds=cast(int | float | None, value.get("delayInSeconds")),
        cron=cast(str | None, value.get("cron")),
        interval_seconds=cast(int | float | None, value.get("intervalSeconds")),
    )


def _insert_result_from_route(value: object) -> tuple[Schedule, bool]:
    if not isinstance(value, dict) or type(value.get("created")) is not bool:
        raise TypeError("routed schedule insertion returned an invalid result")
    return _require_schedule_from_route(value.get("schedule")), value["created"]


def _route_callback(message: Mapping[str, object], *, key: str = "callback") -> str:
    callback = message.get(key)
    if not isinstance(callback, str) or not callback:
        raise ValueError("routed schedule callback must be a non-empty string")
    return callback


def _route_id(message: Mapping[str, object]) -> str:
    schedule_id = message.get("id")
    if not isinstance(schedule_id, str) or not schedule_id:
        raise ValueError("routed schedule id must be a non-empty string")
    return schedule_id


def _scheduler_callback_metadata(member: object) -> object | None:
    function = static_definition_function(member)
    if function is None:
        return None
    for candidate in (member, function):
        state = object.__getattribute__(candidate, "__dict__")
        if type(state) is dict:
            metadata = state.get(_SCHEDULER_METADATA_ATTR)
            if metadata is _SCHEDULER_CALLBACK_METADATA:
                return metadata
    return None


class Scheduler(LifecycleCapability):
    """Create and manage root-owned one-shot and recurring schedules."""

    capability_id = "scheduler"

    def __init__(
        self,
        callbacks: SchedulerCallbacks | None = None,
        *,
        retry: RetryOptions | None = None,
        hung_schedule_timeout_seconds: int = 30,
        on_error: Callable[[BaseException], object | Awaitable[object]] | None = None,
    ) -> None:
        self._callbacks = dict(callbacks or {})
        self._retry_defaults = _merge_retry(retry)
        self._hung_schedule_timeout_seconds = hung_schedule_timeout_seconds
        self._on_error = on_error
        self._warned_startup_callbacks: set[str] = set()
        self._executing_routed: set[str] = set()
        self._cancelled_routed: set[str] = set()

    async def on_start(self) -> None:
        self._warned_startup_callbacks.clear()
        if self.lifecycle.routes.source is not None:
            return
        version = await self.lifecycle.storage.get(_SCHEDULE_SCHEMA_VERSION_KEY)
        if _schema_version(version) >= _CURRENT_SCHEDULE_SCHEMA_VERSION:
            return
        if not await self._migrate_legacy_schedules():
            return
        await self.lifecycle.storage.put(
            _SCHEDULE_SCHEMA_VERSION_KEY,
            _CURRENT_SCHEDULE_SCHEMA_VERSION,
        )

    async def set(
        self,
        when: datetime | str | int | float,
        callback: str,
        payload: object = MISSING,
        options: ScheduleOptions | None = None,
    ) -> Schedule:
        """Create or deduplicate a schedule; invalid callbacks or timing raise."""
        await self.lifecycle.ready()
        self._validate_callback(callback)
        options = options or ScheduleOptions()
        _validate_options(options, self._retry_defaults)
        _validate_schedule_payload(payload)
        timing = _parse_when(when, now_ms(), callback)
        self._warn_for_startup_one_shot(timing, callback, options)
        source = self.lifecycle.routes.source
        if source is None:
            schedule, created = await self._insert(
                None,
                timing,
                callback,
                payload,
                options,
            )
        else:
            result = await self.lifecycle.routes.to_root(
                _schedule_route_message(when, callback, payload, options)
            )
            schedule, created = _insert_result_from_route(result)
        if created:
            await self._emit(
                "schedule:create", {"callback": callback, "id": schedule.id}
            )
        return schedule

    async def every(
        self,
        interval_seconds: int | float,
        callback: str,
        payload: object = MISSING,
        options: ScheduleOptions | None = None,
    ) -> Schedule:
        """Create an interval schedule; invalid callbacks or intervals raise."""
        await self.lifecycle.ready()
        self._validate_callback(callback)
        options = options or ScheduleOptions()
        _validate_options(options, self._retry_defaults)
        _validate_schedule_payload(payload)
        timing = _parse_interval(interval_seconds, now_ms())
        source = self.lifecycle.routes.source
        if source is None:
            schedule, created = await self._insert(
                None,
                timing,
                callback,
                payload,
                options,
            )
        else:
            result = await self.lifecycle.routes.to_root(
                _every_route_message(interval_seconds, callback, payload, options)
            )
            schedule, created = _insert_result_from_route(result)
        if created:
            await self._emit(
                "schedule:create", {"callback": callback, "id": schedule.id}
            )
        return schedule

    async def get(self, id: str) -> Schedule | None:
        """Return an owner-visible schedule, or None when it is absent."""
        await self.lifecycle.ready()
        source = self.lifecycle.routes.source
        if source is not None:
            return _schedule_from_route(
                await self.lifecycle.routes.to_root(_schedule_id_route("get", id))
            )
        return await self._get_for_owner(None, id)

    async def _get_for_owner(
        self,
        owner: LifecycleRouteAddress | None,
        id: str,
    ) -> Schedule | None:
        try:
            job = await self.lifecycle.jobs._get_unvalidated_retry(id)
        except (TypeError, ValueError):
            return None
        if job is None:
            return None
        timing = _scheduler_payload(job)
        if timing is None or not _owner_matches(timing, owner):
            return None
        return _job_to_schedule(job, timing)

    async def list(
        self,
        criteria: ScheduleCriteria | None = None,
    ) -> tuple[Schedule, ...]:
        """Return owner-visible schedules matching the supplied criteria."""
        await self.lifecycle.ready()
        criteria = criteria or ScheduleCriteria()
        _validate_criteria(criteria)
        source = self.lifecycle.routes.source
        if source is not None:
            result = await self.lifecycle.routes.to_root(_schedule_list_route(criteria))
            if not isinstance(result, list):
                raise TypeError("routed schedule list returned an invalid result")
            return tuple(_require_schedule_from_route(item) for item in result)
        return await self._list_for_owner(None, criteria)

    async def _list_for_owner(
        self,
        owner: LifecycleRouteAddress | None,
        criteria: ScheduleCriteria,
    ) -> tuple[Schedule, ...]:
        start, end = _criteria_range(criteria.time_range)
        schedules: list[Schedule] = []
        for job, timing in await self._owned_jobs():
            if not _owner_matches(timing, owner):
                continue
            if criteria.id is not None and job.id != criteria.id:
                continue
            if criteria.type is not None and timing["type"] != criteria.type:
                continue
            seconds = job.time // 1_000
            if start is not None and seconds < start:
                continue
            if end is not None and seconds > end:
                continue
            schedules.append(_job_to_schedule(job, timing))
        return tuple(schedules)

    async def cancel(self, id: str) -> bool:
        """Cancel an owner-visible schedule and report whether it existed."""
        await self.lifecycle.ready()
        source = self.lifecycle.routes.source
        if source is not None:
            result = await self.lifecycle.routes.to_root(
                _schedule_id_route("cancel", id)
            )
            if not isinstance(result, dict) or type(result.get("ok")) is not bool:
                raise TypeError(
                    "routed schedule cancellation returned an invalid result"
                )
            callback = result.get("callback")
            if result["ok"] and isinstance(callback, str):
                if id in self._executing_routed:
                    self._cancelled_routed.add(id)
                await self._emit("schedule:cancel", {"callback": callback, "id": id})
            return cast(bool, result["ok"])
        result = await self._cancel_for_owner(None, id)
        if result["ok"] and isinstance(result.get("callback"), str):
            await self._emit(
                "schedule:cancel",
                {"callback": result["callback"], "id": id},
            )
        return result["ok"]

    async def _cancel_for_owner(
        self,
        owner: LifecycleRouteAddress | None,
        id: str,
    ) -> _ScheduleCancelResult:
        try:
            job = await self.lifecycle.jobs._get_unvalidated_retry(id)
        except (TypeError, ValueError):
            job = None
        if owner is not None and job is None:
            return _schedule_cancel_result(False)
        if job is not None and not _job_owner_matches(job, owner):
            return _schedule_cancel_result(False)
        cancelled = await self.lifecycle.jobs.cancel(id)
        callback = job.fn if cancelled and job is not None else None
        return _schedule_cancel_result(cancelled, callback)

    async def on_job(self, context: LifecycleJobContext) -> LifecycleJobOutcome:
        job = context.job
        try:
            timing = _require_scheduler_payload(job)
        except (TypeError, ValueError) as error:
            await self._report_schedule_error(job.fn, job.id, error, 0)
            return None

        owner = _owner_address(timing)
        if owner is not None:
            try:
                await self.lifecycle.routes.to(
                    owner,
                    _schedule_dispatch_route(job, timing),
                )
            except asyncio.CancelledError:
                raise
            except BaseException as error:
                if _is_platform_failure(error):
                    raise
                await self._report_schedule_error(job.fn, job.id, error, 0)
                return "yield"
            return _recurrence_outcome(timing, now_ms())

        event: SchedulerEventType = (
            "schedule:execute" if context.attempt == 1 else "schedule:retry"
        )
        event_payload: dict[str, object] = {"callback": job.fn, "id": job.id}
        if context.attempt > 1:
            event_payload.update(
                attempt=context.attempt,
                maxAttempts=cast(
                    int, (job.retry or self._retry_defaults)["maxAttempts"]
                ),
            )
        await self._emit(event, event_payload)

        callback = self._callbacks.get(job.fn)
        if callback is None:
            return _recurrence_outcome(timing, now_ms())
        schedule = _job_to_schedule(job, timing)
        payload = timing.get("payload", MISSING)
        await self.lifecycle.run_in_host_context(lambda: callback(payload, schedule))
        return _recurrence_outcome(timing, now_ms())

    async def on_job_error(
        self,
        context: LifecycleJobContext,
        error: BaseException,
    ) -> LifecycleJobOutcome:
        timing = _scheduler_payload(context.job)
        if timing is None:
            return None
        attempts = context.attempt
        await self._report_schedule_error(
            context.job.fn,
            context.job.id,
            error,
            attempts,
        )
        return _recurrence_outcome(timing, now_ms())

    async def on_memory_limit(self, context: LifecycleMemoryLimitContext) -> None:
        if not context.sealed or self.lifecycle.routes.source is not None:
            return
        owners: dict[str, LifecycleRouteAddress] = {}
        jobs = list(context.purged_recovery_loop_jobs)
        if context.executing is not None and context.executing.recovery_loop:
            jobs.append(context.executing)
        for job in jobs:
            if job.capability != self.capability_id:
                continue
            timing = _scheduler_payload(job)
            owner = _owner_address(timing) if timing is not None else None
            if owner is not None:
                owners[owner.key] = owner
        results = await asyncio.gather(
            *(
                self.lifecycle.routes.to(
                    owner,
                    _schedule_memory_route(),
                )
                for owner in owners.values()
            ),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, BaseException):
                raise result

    async def on_route(self, context: LifecycleRouteContext) -> object:
        message = context.payload
        if not isinstance(message, dict) or type(message.get("type")) is not str:
            raise ValueError("invalid Scheduler route message")
        route_type = message["type"]
        local_owner = self.lifecycle.routes.source
        if route_type == "dispatch":
            if local_owner is None or context.source is not None:
                raise PermissionError(
                    "schedule dispatch must route from root to a facet"
                )
        elif route_type == "memory":
            if local_owner is None or context.source is not None:
                raise PermissionError(
                    "schedule memory policy must route from root to a facet"
                )
        elif local_owner is not None or context.source is None:
            raise PermissionError("schedule operations must route from a facet to root")
        if route_type == "schedule":
            callback = _route_callback(message)
            options = _options_from_route(message.get("options"))
            _validate_options(options, self._retry_defaults)
            timing = _parse_when(
                _when_from_route(message.get("when")),
                now_ms(),
                callback,
            )
            payload = message.get("payload", MISSING)
            _validate_schedule_payload(payload)
            schedule, created = await self._insert(
                context.source,
                timing,
                callback,
                payload,
                options,
            )
            return _schedule_insert_result(schedule, created)
        if route_type == "every":
            callback = _route_callback(message)
            options = _options_from_route(message.get("options"))
            _validate_options(options, self._retry_defaults)
            timing = _parse_interval(
                cast(int | float, message.get("intervalSeconds")),
                now_ms(),
            )
            payload = message.get("payload", MISSING)
            _validate_schedule_payload(payload)
            schedule, created = await self._insert(
                context.source,
                timing,
                callback,
                payload,
                options,
            )
            return _schedule_insert_result(schedule, created)
        if route_type == "get":
            schedule = await self._get_for_owner(context.source, _route_id(message))
            return None if schedule is None else _schedule_to_route(schedule)
        if route_type == "list":
            criteria = _criteria_from_route(message.get("criteria"))
            _validate_criteria(criteria)
            schedules = await self._list_for_owner(context.source, criteria)
            return [_schedule_to_route(schedule) for schedule in schedules]
        if route_type == "cancel":
            return await self._cancel_for_owner(context.source, _route_id(message))
        if route_type == "dispatch":
            await self._execute_routed(message)
            return True
        if route_type == "memory":
            if set(message) != {"type", "sealed", "nextAt"}:
                raise ValueError("invalid routed schedule memory policy")
            if message["sealed"] is not True or message["nextAt"] is not None:
                raise ValueError("invalid routed schedule memory policy")
            await self.lifecycle._notify_host_memory_limit(
                LifecycleMemoryLimitContext(
                    sealed=True,
                    next_time=None,
                    executing=None,
                    purged_recovery_loop_jobs=(),
                )
            )
            return True
        raise ValueError(f"unknown Scheduler route message: {route_type!r}")

    async def cleanup_route_prefix(
        self,
        prefix: str,
        owner_path: str | None = None,
    ) -> None:
        tables = self.lifecycle.sql.execute(
            "SELECT 1 FROM sqlite_master "
            "WHERE type = 'table' AND name = 'cf_agents_jobs'"
        )
        if not tables:
            return
        rows = self.lifecycle.sql.execute(
            "SELECT id, fn, payload FROM cf_agents_jobs WHERE capability = ?",
            self.capability_id,
        )
        for row in rows:
            try:
                timing = strict_json_loads(row["payload"], "schedule payload")
            except (TypeError, ValueError):
                continue
            if not isinstance(timing, dict):
                continue
            owner_key = timing.get("owner_path_key")
            key_matches = isinstance(owner_key, str) and (
                owner_key == prefix or owner_key.startswith(f"{prefix}/")
            )
            path_matches = owner_path is not None and _owner_path_has_prefix(
                timing.get("owner_path"), owner_path
            )
            if not key_matches and not path_matches:
                continue
            await self._emit(
                "schedule:cancel",
                {"callback": row["fn"], "id": row["id"]},
            )
            await self.lifecycle.jobs.cancel(row["id"])

    async def _execute_routed(self, message: Mapping[str, object]) -> None:
        schedule_id = _route_id(message)
        callback_name = _route_callback(message, key="fn")
        timing = message.get("job")
        if not isinstance(timing, dict):
            raise ValueError("routed schedule job must be an object")
        _validate_scheduler_payload(timing, schedule_id)
        retry_options = _retry_from_wire(message.get("retry"))
        _validate_options(ScheduleOptions(retry=retry_options), self._retry_defaults)
        retry = _merge_retry(retry_options, self._retry_defaults)
        _validate_lifecycle_retry(retry)
        max_attempts = cast(int, retry["maxAttempts"])
        due = message.get("time")
        if type(due) is not int or due < 0 or due > _MAX_SQLITE_INTEGER:
            raise ValueError("routed schedule time must be a valid integer")
        schedule = _schedule_from_routed_job(
            schedule_id,
            callback_name,
            timing,
            due,
        )
        self._executing_routed.add(schedule_id)
        try:
            await self._emit(
                "schedule:execute",
                {"callback": callback_name, "id": schedule_id},
            )
            callback = self._callbacks.get(callback_name)
            if callback is None:
                return
            for attempt in range(1, max_attempts + 1):
                if schedule_id in self._cancelled_routed:
                    return
                try:
                    if attempt > 1:
                        await self._emit(
                            "schedule:retry",
                            {
                                "callback": callback_name,
                                "id": schedule_id,
                                "attempt": attempt,
                                "maxAttempts": max_attempts,
                            },
                        )
                        if schedule_id in self._cancelled_routed:
                            return
                    payload = timing.get("payload", MISSING)
                    await self.lifecycle.run_in_host_context(
                        lambda: callback(payload, schedule)
                    )
                    return
                except asyncio.CancelledError:
                    raise
                except BaseException as error:
                    if _is_code_update_reset(error):
                        raise
                    if schedule_id in self._cancelled_routed:
                        await self._report_schedule_error(
                            callback_name,
                            schedule_id,
                            error,
                            attempt,
                        )
                        return
                    if attempt == max_attempts:
                        raise
                    await _retry_sleep(
                        attempt,
                        cast(int | float, retry["baseDelayMs"]),
                        cast(int | float, retry["maxDelayMs"]),
                    )
                    if schedule_id in self._cancelled_routed:
                        await self._report_schedule_error(
                            callback_name,
                            schedule_id,
                            error,
                            attempt,
                        )
                        return
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            if _is_platform_failure(error):
                raise
            await self._report_schedule_error(
                callback_name,
                schedule_id,
                error,
                max_attempts,
            )
        finally:
            self._executing_routed.discard(schedule_id)
            self._cancelled_routed.discard(schedule_id)

    def _validate_callback(self, callback: str) -> None:
        if type(callback) is not str:
            raise TypeError("callback must be a string")
        if callback not in self._callbacks:
            raise ValueError(f"unknown scheduled callback: {callback!r}")

    async def _insert(
        self,
        owner: LifecycleRouteAddress | None,
        timing: dict[str, object],
        callback: str,
        payload: object,
        options: ScheduleOptions,
    ) -> tuple[Schedule, bool]:
        recurring = timing["type"] in ("cron", "interval")
        idempotent = (
            options.idempotent is not False if recurring else options.idempotent is True
        )
        if idempotent:
            existing = await self._find_matching(owner, timing, callback, payload)
            if existing is not None:
                await self.lifecycle.jobs.rearm()
                return existing, False

        job_payload = _job_payload(
            timing,
            payload,
            options.retry,
            owner_path=None if owner is None else owner.data,
            owner_path_key=None if owner is None else owner.key,
        )
        dumps_wire(job_payload)
        job = await self.lifecycle.jobs._push_unvalidated_retry(
            LifecycleJobPushOptions(
                id=_schedule_id(),
                fn=callback,
                time=cast(int, timing["time"]) * 1_000,
                payload=job_payload,
                retry=_job_retry(options.retry, self._retry_defaults),
                singleflight=timing["type"] == "interval",
                hung_timeout_seconds=self._hung_schedule_timeout_seconds,
            )
        )
        return _job_to_schedule(job, job_payload), True

    async def _find_matching(
        self,
        owner: LifecycleRouteAddress | None,
        timing: dict[str, object],
        callback: str,
        payload: object,
    ) -> Schedule | None:
        payload_token = _payload_token(payload)
        for job, candidate in await self._owned_jobs():
            if not _owner_matches(candidate, owner):
                continue
            if job.fn != callback or candidate["type"] != timing["type"]:
                continue
            if _payload_token(candidate.get("payload", MISSING)) != payload_token:
                continue
            if timing["type"] == "cron" and candidate.get("cron") != timing["cron"]:
                continue
            if (
                timing["type"] == "interval"
                and candidate.get("intervalSeconds") != timing["intervalSeconds"]
            ):
                continue
            return _job_to_schedule(job, candidate)
        return None

    async def _owned_jobs(
        self,
    ) -> tuple[tuple[LifecycleJob, _ScheduleJobPayload], ...]:
        owned = []
        for job in await self.lifecycle.jobs._list_unvalidated_retry():
            timing = _scheduler_payload(job)
            if timing is not None:
                owned.append((job, timing))
        return tuple(owned)

    def _warn_for_startup_one_shot(
        self,
        timing: dict[str, object],
        callback: str,
        options: ScheduleOptions,
    ) -> None:
        if timing["type"] not in ("scheduled", "delayed"):
            return
        if not self.lifecycle.starting or options.idempotent is not None:
            return
        if callback in self._warned_startup_callbacks:
            return
        self._warned_startup_callbacks.add(callback)
        warnings.warn(
            f"scheduling {callback!r} during startup without idempotent=True "
            "creates a new row on every wake",
            UserWarning,
            stacklevel=3,
        )

    async def _migrate_legacy_schedules(self) -> bool:
        if not self._has_legacy_schedule_table():
            return True
        prepared = await self._prepare_legacy_schedules()
        if prepared is None:
            return False
        await self._import_legacy_schedules(prepared)
        self.lifecycle.sql.execute("DROP TABLE cf_agents_schedules")
        return True

    def _has_legacy_schedule_table(self) -> bool:
        tables = self.lifecycle.sql.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name = 'cf_agents_schedules'"
        )
        return bool(tables)

    async def _prepare_legacy_schedules(self) -> _LegacyScheduleRows | None:
        rows = self.lifecycle.sql.execute("SELECT * FROM cf_agents_schedules")
        prepared: _LegacyScheduleRows = []
        malformed = False
        for row in rows:
            callback = row["callback"]
            if callback == "_cf_keepAliveHeartbeat":
                continue
            try:
                schedule_id = row.get("id")
                if type(schedule_id) is not str or not schedule_id:
                    raise ValueError("legacy schedule id must be a non-empty string")
                if type(callback) is not str or not callback.strip():
                    raise ValueError("legacy callback must be a non-empty string")
                payload = (
                    MISSING
                    if row.get("payload") is None
                    else strict_json_loads(row["payload"], "schedule payload")
                )
                retry = _legacy_retry(row.get("retry_options"))
                timing = _legacy_timing(row)
                owner_path = row.get("owner_path")
                owner_path_key = row.get("owner_path_key")
                if isinstance(owner_path, str) and isinstance(owner_path_key, str):
                    owner_path, canonical_key = _canonical_agent_owner_path(owner_path)
                    if owner_path_key != canonical_key:
                        raise ValueError("legacy routed schedule owner key mismatch")
                    owner_path_key = canonical_key
                elif owner_path is not None or owner_path_key is not None:
                    raise ValueError("legacy routed schedule has incomplete ownership")
                job_payload = _job_payload(
                    timing,
                    payload,
                    retry,
                    owner_path=owner_path,
                    owner_path_key=owner_path_key,
                )
                _validate_schedule_payload(payload)
                dumps_wire(job_payload)
                _validate_scheduler_payload(job_payload, schedule_id)
                due = _legacy_time_ms(row["time"])
                resolved_retry = _job_retry(retry, self._retry_defaults)
            except (TypeError, ValueError) as error:
                malformed = True
                await self._report_schedule_error(
                    str(callback),
                    str(row.get("id", "")),
                    error,
                    0,
                )
                continue
            prepared.append(
                _PreparedLegacySchedule(
                    id=schedule_id,
                    callback=callback,
                    time_ms=due,
                    payload=job_payload,
                    retry=resolved_retry,
                    singleflight=timing["type"] == "interval",
                    recovery_loop=callback in _LEGACY_RECOVERY_LOOP_CALLBACKS,
                )
            )
        if malformed:
            return None
        return prepared

    async def _import_legacy_schedules(
        self,
        rows: _LegacyScheduleRows,
    ) -> None:
        pending: _LegacyJobOptions = []
        for row in rows:
            try:
                existing = await self.lifecycle.jobs._get_unvalidated_retry(row.id)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"cannot reconcile migrated schedule {row.id!r}"
                ) from error
            if existing is not None:
                if (
                    existing.fn == row.callback
                    and existing.time == row.time_ms
                    and existing.payload_present
                    and _js_json_token(existing.payload) == _js_json_token(row.payload)
                    and _js_json_token(existing.retry) == _js_json_token(row.retry)
                    and existing.singleflight is row.singleflight
                    and existing.recovery_loop is row.recovery_loop
                ):
                    continue
                raise ValueError(f"migrated schedule id collision: {row.id!r}")
            pending.append(
                LifecycleJobPushOptions(
                    id=row.id,
                    fn=row.callback,
                    time=row.time_ms,
                    payload=row.payload,
                    retry=row.retry,
                    singleflight=row.singleflight,
                    hung_timeout_seconds=self._hung_schedule_timeout_seconds,
                    recovery_loop=row.recovery_loop,
                )
            )
        for options in pending:
            await self.lifecycle.jobs._push_unvalidated_retry(options)

    async def _report_schedule_error(
        self,
        callback: str,
        id: str,
        error: BaseException,
        attempts: int,
    ) -> None:
        await self._emit(
            "schedule:error",
            {
                "callback": callback,
                "id": id,
                "error": str(error),
                "attempts": attempts,
            },
        )
        if self._on_error is None:
            return
        try:
            result = self._on_error(error)
            if inspect.isawaitable(result):
                await result
        except Exception:
            pass

    async def _emit(self, type: SchedulerEventType, payload: object) -> None:
        await self.lifecycle.events.emit(type, payload)


def _parse_when(
    when: datetime | str | int | float,
    current_ms: int,
    callback: str,
) -> dict[str, object]:
    if isinstance(when, datetime):
        return {"type": "scheduled", "time": _datetime_seconds(when)}
    if isinstance(when, bool):
        raise TypeError(f"invalid schedule type for {callback!r}: bool")
    if isinstance(when, (int, float)):
        if not _number_is_finite(when):
            raise ValueError("schedule delay must be finite")
        scheduled_ms = current_ms + when * 1_000
        if (
            not _number_is_finite(scheduled_ms)
            or scheduled_ms < 0
            or scheduled_ms > _MAX_SQLITE_INTEGER
        ):
            raise ValueError("schedule delay produces an invalid timestamp")
        scheduled_seconds = (
            scheduled_ms // 1_000
            if type(scheduled_ms) is int
            else math.floor(scheduled_ms / 1_000)
        )
        return {
            "type": "delayed",
            "time": scheduled_seconds,
            "delayInSeconds": when,
        }
    if isinstance(when, str):
        return {
            "type": "cron",
            "time": math.floor(_next_cron_time_ms(when, current_ms) / 1_000),
            "cron": when,
        }
    raise TypeError(f"invalid schedule type for {callback!r}: {type(when).__name__}")


def _parse_interval(
    interval_seconds: int | float, current_ms: int
) -> dict[str, object]:
    if (
        isinstance(interval_seconds, bool)
        or not isinstance(interval_seconds, (int, float))
        or not _number_is_finite(interval_seconds)
        or interval_seconds <= 0
    ):
        raise ValueError("interval_seconds must be a positive finite number")
    if interval_seconds > _MAX_INTERVAL_SECONDS:
        raise ValueError(
            f"interval_seconds cannot exceed {_MAX_INTERVAL_SECONDS} seconds"
        )
    return {
        "type": "interval",
        "time": math.floor((current_ms + interval_seconds * 1_000) / 1_000),
        "intervalSeconds": interval_seconds,
    }


def _next_cron_time_ms(expression: str, current_ms: int) -> int:
    parsed = _parse_cron_expression(expression)
    current = datetime.fromtimestamp(current_ms / 1_000, tz=UTC)
    try:
        following = _next_cron_date(parsed, current)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"invalid cron expression: {expression!r}") from error
    return math.floor(following.timestamp() * 1_000)


def _parse_cron_expression(expression: str) -> _ParsedCron:
    if type(expression) is not str:
        raise ValueError("cron expression must be a string")
    expanded = _CRON_NICKNAMES.get(expression.lower(), expression)
    elements = [element for element in expanded.split(" ") if element]
    if len(elements) not in (5, 6):
        raise ValueError("cron expressions must contain five or six fields")

    fields = elements if len(elements) == 6 else ["0", *elements]
    constraints = (
        (0, 59, None, False),
        (0, 59, None, False),
        (0, 23, None, False),
        (1, 31, None, False),
        (1, 12, _MONTH_ALIASES, False),
        (0, 7, _WEEKDAY_ALIASES, True),
    )
    parsed_fields = []
    try:
        for field, (minimum, maximum, aliases, weekday) in zip(
            fields,
            constraints,
            strict=True,
        ):
            values = _parse_cron_element(
                field,
                minimum,
                maximum,
                aliases or {},
                weekday=weekday,
            )
            if weekday:
                values = {value % 7 for value in values}
            parsed_fields.append(tuple(sorted(values)))
    except ValueError as error:
        raise ValueError(f"invalid cron expression: {expression!r}") from error
    return _ParsedCron(*parsed_fields)


def _next_cron_date(parsed: _ParsedCron, current: datetime) -> datetime:
    start_index = next(
        (index for index, month in enumerate(parsed.months) if month >= current.month),
        None,
    )
    first_year = current.year
    if start_index is None:
        start_index = 0
        first_year += 1

    for offset in range(len(parsed.months) * 5):
        position = start_index + offset
        year = first_year + position // len(parsed.months)
        month = parsed.months[position % len(parsed.months)]
        is_start_month = year == current.year and month == current.month
        day = _next_cron_day(
            parsed,
            year,
            month,
            current.day if is_start_month else 1,
        )
        is_start_day = is_start_month and day == current.day
        if day is not None and is_start_day:
            next_time = _next_cron_time_of_day(parsed, current)
            if next_time is not None:
                hour, minute, second = next_time
                return datetime(year, month, day, hour, minute, second, tzinfo=UTC)
            day = _next_cron_day(parsed, year, month, day + 1)
            is_start_day = False
        if day is not None and not is_start_day:
            return datetime(
                year,
                month,
                day,
                parsed.hours[0],
                parsed.minutes[0],
                parsed.seconds[0],
                tzinfo=UTC,
            )
    raise ValueError("no valid cron date found within five years")


def _next_cron_day(
    parsed: _ParsedCron,
    year: int,
    month: int,
    start_day: int,
) -> int | None:
    last_day = calendar.monthrange(year, month)[1]
    days_restricted = len(parsed.days) != 31
    weekdays_restricted = len(parsed.weekdays) != 7
    for day in range(start_day, last_day + 1):
        day_matches = day in parsed.days
        weekday = (datetime(year, month, day).weekday() + 1) % 7
        weekday_matches = weekday in parsed.weekdays
        if days_restricted and weekdays_restricted:
            allowed = day_matches or weekday_matches
        elif days_restricted:
            allowed = day_matches
        elif weekdays_restricted:
            allowed = weekday_matches
        else:
            allowed = True
        if allowed:
            return day
    return None


def _next_cron_time_of_day(
    parsed: _ParsedCron,
    current: datetime,
) -> tuple[int, int, int] | None:
    for hour in parsed.hours:
        if hour < current.hour:
            continue
        for minute in parsed.minutes:
            if hour == current.hour and minute < current.minute:
                continue
            second_index = 0
            if hour == current.hour and minute == current.minute:
                second_index = bisect_right(parsed.seconds, current.second)
            if second_index < len(parsed.seconds):
                return hour, minute, parsed.seconds[second_index]
    return None


def _parse_cron_element(
    element: str,
    minimum: int,
    maximum: int,
    aliases: Mapping[str, int],
    *,
    weekday: bool,
) -> set[int]:
    if element == "*":
        return set(range(minimum, maximum + 1))
    if "," in element:
        values: set[int] = set()
        for item in element.split(","):
            values.update(
                _parse_cron_element(
                    item,
                    minimum,
                    maximum,
                    aliases,
                    weekday=weekday,
                )
            )
        return values

    match = _CRON_RANGE.fullmatch(element)
    if match is None:
        return {_parse_cron_value(element, minimum, maximum, aliases)}

    start_text, end_text, wildcard, step_text = match.groups()
    if wildcard is not None:
        start = minimum
        end = maximum
    else:
        start = _parse_cron_value(start_text, minimum, maximum, aliases)
        end = _parse_cron_value(end_text, minimum, maximum, aliases)
    if weekday and start == 7 and end != 7:
        start = 0
    if start > end:
        raise ValueError("cron range start must not exceed its end")
    step = int(step_text) if step_text is not None else 1
    if step < 1:
        raise ValueError("cron step must be greater than zero")
    return set(range(start, end + 1, step))


def _parse_cron_value(
    value: str,
    minimum: int,
    maximum: int,
    aliases: Mapping[str, int],
) -> int:
    parsed = aliases.get(value.lower())
    if parsed is None:
        prefix = _INTEGER_PREFIX.match(value.lstrip(_ECMASCRIPT_WHITESPACE))
        if prefix is None:
            raise ValueError("cron value is not a number or alias")
        parsed = int(prefix.group())
    if parsed < minimum or parsed > maximum:
        raise ValueError("cron value is outside its allowed range")
    return parsed


def _datetime_seconds(value: datetime) -> int:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    seconds = value.timestamp()
    if not _number_is_finite(seconds) or seconds < 0:
        raise ValueError("scheduled datetime must have a non-negative timestamp")
    return math.floor(seconds)


def _criteria_range(
    value: ScheduleTimeRange | None,
) -> tuple[int | None, int | None]:
    if value is None:
        return None, None
    start = _datetime_seconds(value.start) if value.start is not None else 0
    end = _datetime_seconds(value.end) if value.end is not None else None
    if start is not None and end is not None and start > end:
        raise ValueError("time range start must not be after end")
    return start, end


def _validate_criteria(criteria: ScheduleCriteria) -> None:
    if criteria.type not in (None, "scheduled", "delayed", "cron", "interval"):
        raise ValueError(f"unknown schedule type: {criteria.type!r}")


def _validate_options(
    options: ScheduleOptions,
    defaults: dict[str, object],
) -> None:
    if options.idempotent is not None and type(options.idempotent) is not bool:
        raise TypeError("idempotent must be a boolean")
    if options.retry is not None:
        _validate_retry_fields(options.retry)
        base_delay = cast(
            int | float,
            options.retry.base_delay_ms
            if options.retry.base_delay_ms is not None
            else defaults["baseDelayMs"],
        )
        max_delay = cast(
            int | float,
            options.retry.max_delay_ms
            if options.retry.max_delay_ms is not None
            else defaults["maxDelayMs"],
        )
        if base_delay > max_delay:
            raise ValueError("retry.base_delay_ms must be <= retry.max_delay_ms")


def _merge_retry(
    retry: RetryOptions | None,
    defaults: Mapping[str, object] | None = None,
) -> dict[str, object]:
    defaults = _DEFAULT_RETRY if defaults is None else defaults
    resolved = dict(defaults)
    if retry is None:
        return resolved
    if retry.max_attempts is not None:
        resolved["maxAttempts"] = retry.max_attempts
    if retry.base_delay_ms is not None:
        resolved["baseDelayMs"] = retry.base_delay_ms
    if retry.max_delay_ms is not None:
        resolved["maxDelayMs"] = retry.max_delay_ms
    return resolved


def _job_retry(
    retry: RetryOptions | None,
    defaults: Mapping[str, object],
) -> dict[str, object]:
    return _merge_retry(retry, defaults)


def _positive_retry_number(value: object, name: str) -> int | float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not _number_is_finite(value)
        or value <= 0
    ):
        raise ValueError(f"retry.{name} must be > 0")
    return value


def _retry_to_wire(retry: RetryOptions | None) -> dict[str, object] | None:
    if retry is None:
        return None
    wire = {}
    if retry.max_attempts is not None:
        wire["maxAttempts"] = retry.max_attempts
    if retry.base_delay_ms is not None:
        wire["baseDelayMs"] = retry.base_delay_ms
    if retry.max_delay_ms is not None:
        wire["maxDelayMs"] = retry.max_delay_ms
    return wire


def _retry_from_wire(value: object) -> RetryOptions | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("schedule retry must be a JSON object")
    unknown = set(value) - set(_DEFAULT_RETRY)
    if unknown:
        raise ValueError(f"unknown schedule retry option: {sorted(unknown)[0]}")
    retry = RetryOptions(
        max_attempts=cast(int | None, value.get("maxAttempts")),
        base_delay_ms=cast(int | float | None, value.get("baseDelayMs")),
        max_delay_ms=cast(int | float | None, value.get("maxDelayMs")),
    )
    return retry


def _validate_retry_fields(retry: RetryOptions) -> None:
    if retry.max_attempts is not None and (
        type(retry.max_attempts) is not int
        or not _number_is_finite(retry.max_attempts)
        or retry.max_attempts < 1
    ):
        raise ValueError("retry.max_attempts must be an integer >= 1")
    if retry.base_delay_ms is not None:
        _positive_retry_number(retry.base_delay_ms, "base_delay_ms")
    if retry.max_delay_ms is not None:
        _positive_retry_number(retry.max_delay_ms, "max_delay_ms")
    if (
        retry.base_delay_ms is not None
        and retry.max_delay_ms is not None
        and retry.base_delay_ms > retry.max_delay_ms
    ):
        raise ValueError("retry.base_delay_ms must be <= retry.max_delay_ms")


def _job_payload(
    timing: Mapping[str, object],
    payload: object,
    retry: RetryOptions | None,
    *,
    owner_path: str | None = None,
    owner_path_key: str | None = None,
) -> _ScheduleJobPayload:
    encoded = _ScheduleJobPayload(
        type=cast(ScheduleType, timing["type"]),
        owner_path=owner_path,
        owner_path_key=owner_path_key,
    )
    if payload is not MISSING:
        encoded["payload"] = payload
    retry_wire = _retry_to_wire(retry)
    if retry_wire is not None:
        encoded["retry"] = retry_wire
    if "delayInSeconds" in timing:
        encoded["delayInSeconds"] = cast(int | float, timing["delayInSeconds"])
    if "cron" in timing:
        encoded["cron"] = cast(str, timing["cron"])
    if "intervalSeconds" in timing:
        encoded["intervalSeconds"] = cast(int | float, timing["intervalSeconds"])
    return encoded


def _scheduler_payload(job: LifecycleJob | None) -> _ScheduleJobPayload | None:
    if job is None:
        return None
    try:
        return _require_scheduler_payload(job)
    except (TypeError, ValueError):
        return None


def _require_scheduler_payload(job: LifecycleJob) -> _ScheduleJobPayload:
    if not job.payload_present or not isinstance(job.payload, dict):
        raise ValueError(f"malformed schedule job {job.id}: payload must be an object")
    timing = job.payload
    _validate_scheduler_payload(timing, job.id)
    return cast(_ScheduleJobPayload, timing)


def _validate_scheduler_payload(timing: Mapping[str, object], job_id: str) -> None:
    schedule_type = timing.get("type")
    if schedule_type not in ("scheduled", "delayed", "cron", "interval"):
        raise ValueError(f"malformed schedule job {job_id}: invalid type")
    owner_path = timing.get("owner_path")
    owner_key = timing.get("owner_path_key")
    if owner_path is None and owner_key is None:
        pass
    elif not isinstance(owner_path, str) or not isinstance(owner_key, str):
        raise ValueError(f"malformed schedule job {job_id}: invalid owner path")
    else:
        canonical_path, canonical_key = _canonical_agent_owner_path(owner_path)
        if canonical_path != owner_path or canonical_key != owner_key:
            raise ValueError(f"malformed schedule job {job_id}: noncanonical owner")
    if "retry" in timing:
        _retry_from_wire(timing["retry"])
    if schedule_type == "delayed":
        _finite_schedule_number(timing.get("delayInSeconds"), "delayInSeconds")
    elif schedule_type == "cron":
        _parse_cron_expression(cast(str, timing.get("cron")))
    elif schedule_type == "interval":
        interval = _finite_schedule_number(
            timing.get("intervalSeconds"),
            "intervalSeconds",
        )
        _parse_interval(interval, 0)


def _finite_schedule_number(value: object, name: str) -> int | float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not _number_is_finite(value)
    ):
        raise ValueError(f"schedule {name} must be a finite number")
    return value


def _job_to_schedule(job: LifecycleJob, timing: Mapping[str, object]) -> Schedule:
    return Schedule(
        id=job.id,
        callback=job.fn,
        payload=timing.get("payload", MISSING),
        retry=_retry_from_wire(timing.get("retry")),
        type=cast(ScheduleType, timing["type"]),
        time=job.time // 1_000,
        delay_in_seconds=cast(int | float | None, timing.get("delayInSeconds")),
        cron=cast(str | None, timing.get("cron")),
        interval_seconds=cast(int | float | None, timing.get("intervalSeconds")),
    )


def _schedule_from_routed_job(
    schedule_id: str,
    callback: str,
    timing: Mapping[str, object],
    due: int,
) -> Schedule:
    return Schedule(
        id=schedule_id,
        callback=callback,
        payload=timing.get("payload", MISSING),
        retry=_retry_from_wire(timing.get("retry")),
        type=cast(ScheduleType, timing["type"]),
        time=due // 1_000,
        delay_in_seconds=cast(int | float | None, timing.get("delayInSeconds")),
        cron=cast(str | None, timing.get("cron")),
        interval_seconds=cast(int | float | None, timing.get("intervalSeconds")),
    )


def _recurrence_outcome(
    timing: Mapping[str, object],
    current_ms: int,
) -> LifecycleJobOutcome:
    if timing["type"] == "cron":
        return LifecycleJobReschedule(
            _next_cron_time_ms(cast(str, timing["cron"]), current_ms)
        )
    if timing["type"] == "interval":
        interval = cast(int | float, timing["intervalSeconds"])
        return LifecycleJobReschedule(math.floor(current_ms + interval * 1_000))
    return None


def _owner_matches(
    timing: Mapping[str, object],
    owner: LifecycleRouteAddress | None,
) -> bool:
    owner_path = timing.get("owner_path")
    owner_key = timing.get("owner_path_key")
    if owner is None:
        return owner_path is None and owner_key is None
    return owner_path == owner.data and owner_key == owner.key


def _job_owner_matches(
    job: LifecycleJob,
    owner: LifecycleRouteAddress | None,
) -> bool:
    return (
        job.payload_present
        and isinstance(job.payload, dict)
        and _owner_matches(job.payload, owner)
    )


def _owner_path_has_prefix(candidate: object, prefix: str) -> bool:
    if not isinstance(candidate, str):
        return False
    try:
        candidate_path = strict_json_loads(candidate, "schedule owner path")
        prefix_path = strict_json_loads(prefix, "schedule owner path prefix")
    except ValueError:
        return False
    return (
        isinstance(candidate_path, list)
        and isinstance(prefix_path, list)
        and candidate_path[: len(prefix_path)] == prefix_path
    )


def _owner_address(
    timing: Mapping[str, object],
) -> LifecycleRouteAddress | None:
    owner_path = timing.get("owner_path")
    owner_key = timing.get("owner_path_key")
    if owner_path is None and owner_key is None:
        return None
    if not isinstance(owner_path, str) or not isinstance(owner_key, str):
        raise ValueError("malformed routed schedule owner")
    return LifecycleRouteAddress(owner_key, owner_path)


def _canonical_agent_owner_path(value: str) -> tuple[str, str]:
    path = strict_json_loads(value, "schedule owner path")
    if not isinstance(path, list) or not path:
        raise ValueError("schedule owner path must be a non-empty array")
    canonical = []
    encoded = []
    for step in path:
        if not isinstance(step, dict):
            raise ValueError("schedule owner path steps must be objects")
        class_name = step.get("className")
        name = step.get("name")
        if not isinstance(class_name, str) or not isinstance(name, str):
            raise ValueError("schedule owner path steps require className and name")
        canonical.append({"className": class_name, "name": name})
        encoded.append(
            f"{quote(class_name, safe=_URI_SAFE)}:{quote(name, safe=_URI_SAFE)}"
        )
    return dumps_wire(canonical), "/".join(encoded)


def _payload_token(payload: object) -> tuple[bool, object]:
    if payload is MISSING:
        return False, None
    parsed = strict_json_loads(dumps_wire(payload), "schedule payload")
    return True, _js_json_token(parsed)


def _js_json_token(value: object) -> object:
    if value is None:
        return ("null",)
    if type(value) is bool:
        return ("boolean", value)
    if type(value) is str:
        return ("string", value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            number = float(value)
        except OverflowError:
            number = math.inf
        return ("null",) if not math.isfinite(number) else ("number", number)
    if isinstance(value, list):
        return ("array", tuple(_js_json_token(item) for item in value))
    if isinstance(value, dict):
        indexed = []
        ordinary = []
        for key, item in value.items():
            index = _js_array_index(key)
            entry = (key, _js_json_token(item))
            if index is None:
                ordinary.append(entry)
            else:
                indexed.append((index, entry))
        indexed.sort(key=lambda entry: entry[0])
        entries = tuple(entry for _, entry in indexed) + tuple(ordinary)
        return ("object", entries)
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def _js_array_index(key: object) -> int | None:
    if type(key) is not str or not key:
        return None
    if key != "0" and key.startswith("0"):
        return None
    if not key.isascii() or not key.isdigit():
        return None
    if len(key) > 10:
        return None
    value = int(key)
    if value >= 2**32 - 1 or str(value) != key:
        return None
    return value


def _schedule_id() -> str:
    return secrets.token_urlsafe(7)[:9]


def _schema_version(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    if not _number_is_finite(value) or value < 0:
        return 0
    return math.floor(value)


def _legacy_time_ms(value: object) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not _number_is_finite(value)
        or value < 0
    ):
        raise ValueError(f"invalid legacy schedule time: {value!r}")
    milliseconds = value * 1_000
    if not _number_is_finite(milliseconds) or milliseconds > _MAX_SQLITE_INTEGER:
        raise ValueError(f"invalid legacy schedule time: {value!r}")
    return math.floor(milliseconds)


def _legacy_timing(row: Mapping[str, object]) -> dict[str, object]:
    schedule_type = row.get("type")
    if schedule_type not in ("scheduled", "delayed", "cron", "interval"):
        raise ValueError(f"invalid legacy schedule type: {schedule_type!r}")
    timing: dict[str, object] = {"type": schedule_type}
    fields = {
        "delayed": "delayInSeconds",
        "cron": "cron",
        "interval": "intervalSeconds",
    }
    field = fields.get(cast(str, schedule_type))
    if field is not None:
        value = row.get(field)
        if value is None:
            raise ValueError(f"legacy {schedule_type} schedule is missing {field}")
        timing[field] = value
    if schedule_type == "cron":
        _parse_cron_expression(cast(str, timing["cron"]))
    elif schedule_type == "interval":
        _parse_interval(
            _finite_schedule_number(timing["intervalSeconds"], "intervalSeconds"),
            0,
        )
    return timing


def _legacy_retry(
    value: object,
) -> RetryOptions | None:
    if value is None:
        return None
    try:
        parsed = strict_json_loads(value, "schedule retry")
    except ValueError:
        return None
    return _retry_from_wire(parsed)


def _number_is_finite(value: int | float) -> bool:
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _validate_safe_integers(value: object, seen: set[int]) -> None:
    if isinstance(value, int) and not isinstance(value, bool):
        if value < -_MAX_SAFE_INTEGER or value > _MAX_SAFE_INTEGER:
            raise ValueError("schedule payload integers must be JavaScript-safe")
        return
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in seen:
            return
        seen.add(identity)
        for item in value.values():
            _validate_safe_integers(item, seen)
        seen.remove(identity)
        return
    if isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in seen:
            return
        seen.add(identity)
        for item in value:
            _validate_safe_integers(item, seen)
        seen.remove(identity)


def _validate_schedule_payload(value: object) -> None:
    if value is MISSING:
        return
    _validate_safe_integers(value, set())
    dumps_wire(value)
