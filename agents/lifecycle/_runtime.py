from __future__ import annotations

import asyncio
import inspect
import json
import logging
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, TypeGuard, cast, overload

from js import Object, setTimeout  # ty: ignore[unresolved-import]
from pyodide.ffi import create_once_callable, to_js
from workers import Request, Response

from ..core._discovery import static_mro_members
from ..core.utils import now_ms
from ._job_driver import _JobDispatch, _LifecycleJobDriver
from .capability import LifecycleCapability, _SERVICE_SLOT
from .host_context import (
    _CURRENT_LIFECYCLE_CONTEXT,
    _call_maybe_async,
    _call_with_context,
    get_current_lifecycle_context,
)
from .jobs import (
    LifecycleJob,
    LifecycleJobContext,
    LifecycleJobOutcome,
    LifecycleJobPushOptions,
    LifecycleJobReschedule,
    LifecycleJobs,
    LifecycleMemoryLimitContext,
    _LifecycleJobQueue,
)
from .types import (
    CapabilityRequestContext,
    CapabilityWebSocketCloseContext,
    CapabilityWebSocketErrorContext,
    CapabilityWebSocketMessageContext,
    CapabilityWebSocketUpgradeContext,
    CurrentLifecycleContext,
    LifecycleEvent,
    LifecycleEvents,
    LifecycleHostContextScope,
    LifecycleRetainedWork,
    LifecycleRouteAddress,
    LifecycleRouteContext,
    LifecycleRouteEnvelope,
    LifecycleRoutes,
    LifecycleRouteTransport,
    LifecycleServices,
    LifecycleSockets,
    LifecycleSql,
    LifecycleStorage,
)


__all__ = [
    "CapabilityRequestContext",
    "CapabilityWebSocketCloseContext",
    "CapabilityWebSocketErrorContext",
    "CapabilityWebSocketMessageContext",
    "CapabilityWebSocketUpgradeContext",
    "CurrentLifecycleContext",
    "Lifecycle",
    "LifecycleCapability",
    "LifecycleEvent",
    "LifecycleEvents",
    "LifecycleHostContextScope",
    "LifecycleJob",
    "LifecycleJobContext",
    "LifecycleJobOutcome",
    "LifecycleJobPushOptions",
    "LifecycleJobReschedule",
    "LifecycleJobs",
    "LifecycleMemoryLimitContext",
    "LifecycleRetainedWork",
    "LifecycleRouteAddress",
    "LifecycleRouteContext",
    "LifecycleRouteEnvelope",
    "LifecycleRouteTransport",
    "LifecycleRoutes",
    "LifecycleServices",
    "LifecycleSockets",
    "LifecycleSql",
    "LifecycleStorage",
    "get_current_lifecycle_context",
]


_LOGGER = logging.getLogger(__name__)
_MISSING = object()
_ROUTE_VERSION = 1
_BUILTIN_CAPABILITY_IDS = (
    "websockets",
    "scheduler",
    "tasks",
    "sessions",
    "mcp",
    "fibers",
)


class _StorageServices:
    __slots__ = ("_ctx", "_lifecycle")

    def __init__(self, lifecycle: Lifecycle, ctx: object):
        self._lifecycle = lifecycle
        self._ctx = ctx

    @property
    def _storage(self) -> object:
        return getattr(self._ctx, "storage")

    @overload
    async def get(self, key: str) -> Any: ...

    @overload
    async def get(self, key: Sequence[str]) -> dict[str, Any]: ...

    async def get(self, key: str | Sequence[str]) -> Any:
        async with self._lifecycle._operation(continue_existing=True):
            storage_key = key if type(key) is str else list(key)
            return await _call_maybe_async(
                getattr(self._storage, "get"),
                storage_key,
            )

    @overload
    async def put(self, key: str, value: object) -> None: ...

    @overload
    async def put(self, key: dict[str, Any]) -> None: ...

    async def put(
        self,
        key: str | dict[str, Any],
        value: object = _MISSING,
    ) -> None:
        async with self._lifecycle._operation(continue_existing=True):
            put = getattr(self._storage, "put")
            if value is _MISSING:
                await _call_maybe_async(put, key)
            else:
                await _call_maybe_async(put, key, value)

    @overload
    async def delete(self, key: str) -> bool: ...

    @overload
    async def delete(self, key: Sequence[str]) -> int: ...

    async def delete(self, key: str | Sequence[str]) -> bool | int:
        async with self._lifecycle._operation(continue_existing=True):
            storage_key = key if type(key) is str else list(key)
            return await _call_maybe_async(
                getattr(self._storage, "delete"),
                storage_key,
            )

    async def list(
        self,
        *,
        prefix: str = "",
        start: str | None = None,
        start_after: str | None = None,
        end: str | None = None,
        reverse: bool = False,
        limit: int | None = None,
    ) -> dict[str, Any]:
        async with self._lifecycle._operation(continue_existing=True):
            options: dict[str, object] = {"prefix": prefix, "reverse": reverse}
            if start is not None:
                options["start"] = start
            if start_after is not None:
                options["startAfter"] = start_after
            if end is not None:
                options["end"] = end
            if limit is not None:
                options["limit"] = limit
            return await _call_maybe_async(getattr(self._storage, "list"), options)

    def transaction_sync[T](self, callback: Callable[[], T]) -> T:
        self._lifecycle._ensure_service_access()
        return getattr(self._storage, "transactionSync")(callback)


class _SqlServices:
    __slots__ = ("_ctx", "_lifecycle")

    def __init__(self, lifecycle: Lifecycle, ctx: object):
        self._lifecycle = lifecycle
        self._ctx = ctx

    def execute(self, query: str, *params: object) -> list[dict[str, Any]]:
        self._lifecycle._ensure_service_access()
        storage = getattr(self._ctx, "storage")
        sql = getattr(storage, "sql")
        cursor = getattr(sql, "exec")(query, *params)
        return cursor.toArray()


class _SocketServices:
    __slots__ = ("_ctx", "_lifecycle")

    def __init__(self, lifecycle: Lifecycle, ctx: object):
        self._lifecycle = lifecycle
        self._ctx = ctx

    def accept(
        self,
        websocket: object,
        *,
        tags: Sequence[str] = (),
    ) -> None:
        self._lifecycle._ensure_socket_service_access()
        js_tags = to_js(list(tags))
        getattr(self._ctx, "acceptWebSocket")(websocket, js_tags)

    def get(self, *, tag: str | None = None) -> tuple[object, ...]:
        self._lifecycle._ensure_socket_service_access()
        get_websockets = getattr(self._ctx, "getWebSockets")
        sockets = get_websockets() if tag is None else get_websockets(tag)
        return tuple(sockets)

    def serialize_attachment(self, websocket: object, value: object) -> None:
        self._lifecycle._ensure_socket_service_access()
        attachment = to_js(value, dict_converter=Object.fromEntries)
        getattr(websocket, "serializeAttachment")(attachment)

    def deserialize_attachment(self, websocket: object) -> object | None:
        self._lifecycle._ensure_socket_service_access()
        attachment = getattr(websocket, "deserializeAttachment")()
        if attachment is None:
            return None
        to_python = getattr(attachment, "to_py", None)
        return to_python() if callable(to_python) else attachment


@dataclass(slots=True)
class _Installation:
    capability: object
    capability_id: object


@dataclass(slots=True)
class _QueuedEvent:
    event: LifecycleEvent
    done: asyncio.Future[None]


type _StartHandler = Callable[[], object | Awaitable[object]]
type _PrepareHandler = Callable[[LifecycleSql], object | Awaitable[object]]
type _ResponseHandler = Callable[[Request], Response | Awaitable[Response]]
type _SocketHandler = Callable[[object], object | Awaitable[object]]
type _EventListener = Callable[[LifecycleEvent], object | Awaitable[object]]
type _ErrorHandler = Callable[[BaseException], object | Awaitable[object]]
type _RetainWork = Callable[[Awaitable[object]], object]
type _ResetAlarm = Callable[[str], object | Awaitable[object]]
type _JobHandler = Callable[
    [LifecycleJobContext],
    LifecycleJobOutcome | Awaitable[LifecycleJobOutcome],
]
type _MemoryLimitHandler = Callable[
    [LifecycleMemoryLimitContext],
    object | Awaitable[object],
]
type _AlarmDeadline = Callable[[int], int | None]


class _CapabilityEvents:
    __slots__ = ("_lifecycle", "_source")

    def __init__(self, lifecycle: Lifecycle, source: str):
        self._lifecycle = lifecycle
        self._source = source

    async def emit(self, type: str, payload: object) -> None:
        await self._lifecycle._emit_event(self._source, type, payload)


class _CapabilityRoutes:
    __slots__ = ("_lifecycle", "_owner")

    def __init__(self, lifecycle: Lifecycle, owner: str):
        self._lifecycle = lifecycle
        self._owner = owner

    @property
    def source(self) -> LifecycleRouteAddress | None:
        return self._lifecycle._route_address

    async def to_root(self, payload: object) -> object:
        target = self._lifecycle._root_route_address
        if self._lifecycle._route_address is None or target is None:
            async with self._lifecycle._operation():
                await self._lifecycle._start()
                return await self._lifecycle._route_local(
                    self._owner,
                    source=None,
                    payload=payload,
                )
        return await self._lifecycle._route_for(self._owner, target, payload)

    async def to(
        self,
        target: LifecycleRouteAddress,
        payload: object,
    ) -> object:
        return await self._lifecycle._route_for(self._owner, target, payload)


class _CapabilityRetainedWork:
    __slots__ = ("_lifecycle",)

    def __init__(self, lifecycle: Lifecycle):
        self._lifecycle = lifecycle

    @property
    def available(self) -> bool:
        return self._lifecycle._retained_work_available()

    def retain(self, factory: Callable[[], Awaitable[object]]) -> None:
        self._lifecycle._retain_work(factory)


class _CapabilityJobs:
    __slots__ = ("_lifecycle", "_owner")

    def __init__(self, lifecycle: Lifecycle, owner: str):
        self._lifecycle = lifecycle
        self._owner = owner

    async def push(self, options: LifecycleJobPushOptions) -> LifecycleJob:
        return await self._lifecycle._push_job(self._owner, options)

    async def _push_unvalidated_retry(
        self,
        options: LifecycleJobPushOptions,
    ) -> LifecycleJob:
        return await self._lifecycle._push_job(
            self._owner,
            options,
            validate_retry=False,
        )

    async def cancel(self, id: str) -> bool:
        return await self._lifecycle._cancel_job(self._owner, id)

    async def reschedule(self, id: str, time: int) -> bool:
        return await self._lifecycle._reschedule_job(self._owner, id, time)

    async def get(self, id: str) -> LifecycleJob | None:
        return await self._lifecycle._get_job(self._owner, id)

    async def _get_unvalidated_retry(self, id: str) -> LifecycleJob | None:
        return await self._lifecycle._get_job(
            self._owner,
            id,
            validate_retry=False,
        )

    async def list(
        self,
        *,
        skip_invalid: bool = False,
    ) -> tuple[LifecycleJob, ...]:
        return await self._lifecycle._list_jobs(
            self._owner,
            skip_invalid=skip_invalid,
        )

    async def _list_unvalidated_retry(self) -> tuple[LifecycleJob, ...]:
        return await self._lifecycle._list_jobs(
            self._owner,
            skip_invalid=True,
            validate_retry=False,
        )

    async def rearm(self) -> None:
        await self._lifecycle.rearm_alarm()


class _CapabilityServices:
    __slots__ = ("_events", "_jobs", "_lifecycle", "_retained_work", "_routes")

    def __init__(self, lifecycle: Lifecycle, capability_id: str):
        self._lifecycle = lifecycle
        self._events = _CapabilityEvents(lifecycle, capability_id)
        self._jobs = _CapabilityJobs(lifecycle, capability_id)
        self._routes = _CapabilityRoutes(lifecycle, capability_id)
        self._retained_work = _CapabilityRetainedWork(lifecycle)

    @property
    def storage(self) -> LifecycleStorage:
        return self._lifecycle._storage_services

    @property
    def sql(self) -> LifecycleSql:
        return self._lifecycle._sql_services

    @property
    def sockets(self) -> LifecycleSockets:
        return self._lifecycle._socket_services

    @property
    def events(self) -> LifecycleEvents:
        return self._events

    @property
    def routes(self) -> LifecycleRoutes:
        return self._routes

    @property
    def retained_work(self) -> LifecycleRetainedWork:
        return self._retained_work

    def track_alarm_work(self, awaitable: Awaitable[object]) -> bool:
        return self._lifecycle.track_alarm_work(awaitable)

    def _alarm_work_is_tracked(self, awaitable: Awaitable[object]) -> bool:
        return self._lifecycle._alarm_work_is_tracked(awaitable)

    async def _notify_host_memory_limit(
        self,
        context: LifecycleMemoryLimitContext,
    ) -> None:
        await self._lifecycle._notify_host_memory_limit(context)

    @property
    def jobs(self) -> LifecycleJobs:
        return self._jobs

    @property
    def owns_physical_alarm(self) -> bool:
        return self._lifecycle._owns_physical_alarm

    @property
    def starting(self) -> bool:
        return self._lifecycle._starting

    @property
    def is_ready(self) -> bool:
        return self._lifecycle._ready

    async def ready(self) -> None:
        await self._lifecycle._ready_for_capability()

    async def run_in_host_context[T](
        self,
        callback: Callable[[], T | Awaitable[T]],
        *,
        scope: LifecycleHostContextScope | None = None,
    ) -> T:
        return cast(T, await self._lifecycle._run_in_host_context(callback, scope))


class Lifecycle:
    """Experimental host-neutral capability lifecycle."""

    def __init__(
        self,
        ctx: object,
        *,
        host: object,
        prepare: _PrepareHandler | None = None,
        on_start: _StartHandler | None = None,
        on_request: _ResponseHandler | None = None,
        on_websocket_upgrade: _ResponseHandler | None = None,
        on_websocket_message: _SocketHandler | None = None,
        on_websocket_close: _SocketHandler | None = None,
        on_websocket_error: _SocketHandler | None = None,
        on_job: _JobHandler | None = None,
        on_alarm: _StartHandler | None = None,
        alarm_deadline: _AlarmDeadline | None = None,
        on_alarm_memory_limit: _MemoryLimitHandler | None = None,
        event_listeners: Sequence[_EventListener] = (),
        on_error: _ErrorHandler | None = None,
        route_address: LifecycleRouteAddress | None = None,
        root_route_address: LifecycleRouteAddress | None = None,
        route_transport: LifecycleRouteTransport | None = None,
        max_route_payload_bytes: int = 64 * 1024,
        retain_work: _RetainWork | None = None,
        max_alarm_memory_limit_strikes: int = 3,
        reset_alarm: _ResetAlarm | None = None,
        owns_physical_alarm: bool | None = None,
    ) -> None:
        self._ctx = ctx
        self._storage_services = _StorageServices(self, ctx)
        self._sql_services = _SqlServices(self, ctx)
        self._socket_services = _SocketServices(self, ctx)
        self._job_queue = _LifecycleJobQueue(self._execute_job_sql)
        self._jobs = _CapabilityJobs(self, "host")
        self._alarm_rearm_lock = asyncio.Lock()
        self._rearm_requested_during_start = False
        self._alarms_disabled = False
        self._host = host
        self._normal: list[_Installation] = []
        self._fallbacks: list[_Installation] = []
        self._installations: list[_Installation] = []
        self._registrations_closed = False
        self._sealed: tuple[LifecycleCapability, ...] | None = None
        self._capabilities_by_id: dict[str, LifecycleCapability] = {}
        self._ready = False
        self._starting = False
        self._startup_owner: asyncio.Task[Any] | None = None
        self._startup_attempt: asyncio.Future[None] | None = None
        self._disposing = False
        self._disposed = False
        self._dispose_owner: asyncio.Task[Any] | None = None
        self._dispose_cleanup_tasks: set[asyncio.Task[Any]] = set()
        self._disposal_job_cleanup_tasks: set[asyncio.Task[Any]] = set()
        self._dispose_attempt: asyncio.Future[None] | None = None
        self._active_operations = 0
        self._operation_tasks: dict[asyncio.Task[Any], int] = {}
        self._operations_idle = asyncio.Event()
        self._operations_idle.set()
        self._prepare = prepare
        self._host_start = on_start
        self._host_request = on_request
        self._host_websocket_upgrade = on_websocket_upgrade
        self._host_websocket_message = on_websocket_message
        self._host_websocket_close = on_websocket_close
        self._host_websocket_error = on_websocket_error
        self._host_job = on_job
        self._host_alarm = on_alarm
        self._host_alarm_deadline = alarm_deadline
        self._host_alarm_memory_limit = on_alarm_memory_limit
        self._event_listeners = tuple(event_listeners)
        self._pending_events: list[LifecycleEvent] = []
        self._event_queue: deque[_QueuedEvent] = deque()
        self._event_draining = False
        self._event_owner: asyncio.Task[Any] | None = None
        self._error_handler = on_error
        self._route_address = route_address
        self._root_route_address = root_route_address or route_address
        inferred_alarm_ownership = (
            root_route_address is None or route_address == root_route_address
        )
        self._owns_physical_alarm = (
            inferred_alarm_ownership
            if owns_physical_alarm is None
            else owns_physical_alarm
        )
        self._route_transport = route_transport
        self._max_route_payload_bytes = max_route_payload_bytes
        self._retain_work_handler = retain_work
        self._max_alarm_memory_limit_strikes = max_alarm_memory_limit_strikes
        self._reset_alarm = reset_alarm
        self._job_driver = _LifecycleJobDriver(
            queue=self._job_queue,
            storage=lambda: getattr(ctx, "storage"),
            clock=now_ms,
            disabled=lambda: self._alarms_disabled,
            resolve_dispatch=self._resolve_job_dispatch,
            run_host_alarm=self._run_host_alarm,
            arm_at=self._arm_alarm_at,
            rearm=self._rearm_alarm,
            report_error=self._report_secondary_error,
            memory_limit=self._run_memory_limit_policy,
            max_memory_limit_strikes=self._memory_limit_strike_limit,
            reset=self._reset_after_memory_limit,
            schedule_background=self._schedule_alarm_background,
        )

    def use(
        self,
        capability: LifecycleCapability,
        *,
        fallback: bool = False,
    ) -> None:
        """Register a normal capability or an ordered fallback."""
        if self._registrations_closed:
            raise RuntimeError("Lifecycle registration is closed")

        capability_id = static_mro_members(capability).get("capability_id")
        installation = _Installation(capability, capability_id)
        target = self._fallbacks if fallback else self._normal
        target.append(installation)
        self._installations.append(installation)

        if (
            self._is_capability(capability)
            and self._bound_lifecycle(capability) is None
        ):
            services = _CapabilityServices(self, cast(str, capability_id))
            _SERVICE_SLOT.__set__(capability, services)

    @property
    def jobs(self) -> LifecycleJobs:
        return self._jobs

    def _seal_registrations(self) -> tuple[LifecycleCapability, ...]:
        """Lock registration, validate it, and return normal then fallback order."""
        self._registrations_closed = True
        if self._sealed is not None:
            return self._sealed

        ordered = (*self._normal, *self._fallbacks)
        seen_instances: set[int] = set()
        seen_ids: set[str] = set()
        by_id: dict[str, LifecycleCapability] = {}

        for installation in ordered:
            capability = installation.capability
            if not self._is_capability(capability):
                raise ValueError("capability must inherit LifecycleCapability")
            instance_id = id(capability)
            if instance_id in seen_instances:
                raise ValueError("capability instance is already installed")
            seen_instances.add(instance_id)

            capability_id = installation.capability_id
            if type(capability_id) is not str or not capability_id.strip():
                raise ValueError("capability_id must be a non-empty string")
            if capability_id in seen_ids:
                raise ValueError(f"duplicate capability_id: {capability_id}")
            if capability_id == "host":
                raise ValueError("capability_id 'host' is reserved")
            seen_ids.add(capability_id)

            if self._bound_lifecycle(capability) is not self:
                raise ValueError("capability is already installed on another Lifecycle")
            by_id[capability_id] = capability

        sealed = tuple(cast(LifecycleCapability, item.capability) for item in ordered)
        self._sealed = sealed
        self._capabilities_by_id = by_id
        return sealed

    async def start(self) -> None:
        """Run or join the activation's retryable startup attempt."""
        async with self._operation():
            await self._start()

    async def ready(self) -> None:
        """Join startup unless called by the startup owner itself."""
        await self._ready_for_capability()

    async def _start(self) -> None:
        self._ensure_active()
        current_attempt = self._startup_attempt
        if current_attempt is not None:
            await asyncio.shield(current_attempt)
            return
        if self._ready:
            return

        attempt = asyncio.get_running_loop().create_future()
        self._startup_attempt = attempt
        self._startup_owner = asyncio.current_task()
        self._starting = True
        try:
            capabilities = self._seal_registrations()
            if self._owns_physical_alarm:
                self._memory_limit_strike_limit()
                self._job_queue.prepare()
            if self._prepare is not None:
                await _call_with_context(None, self._prepare, self._sql_services)
            for capability in capabilities:
                await _call_with_context(None, capability.on_start)
            if self._host_start is not None:
                await self._call_host(self._host_start)
            self._starting = False
            await self._flush_pending_events()
            self._ready = True
            if self._rearm_requested_during_start:
                await self._rearm_alarm()
                self._rearm_requested_during_start = False
        except BaseException as error:
            self._ready = False
            self._starting = False
            self._pending_events.clear()
            cancellation: asyncio.CancelledError | None = None
            if self._rearm_requested_during_start:
                try:
                    await self._rearm_alarm()
                except asyncio.CancelledError as rearm_cancellation:
                    cancellation = rearm_cancellation
                except BaseException as rearm_error:
                    try:
                        await self._report_secondary_error(rearm_error)
                    except asyncio.CancelledError as reporting_cancellation:
                        cancellation = reporting_cancellation
                else:
                    self._rearm_requested_during_start = False
            attempt.set_exception(cancellation or error)
            attempt.exception()
            self._startup_attempt = None
            self._startup_owner = None
            if cancellation is not None:
                raise cancellation
            raise

        self._startup_owner = None
        attempt.set_result(None)

    async def fetch(self, request: Request) -> Response:
        async with self._operation():
            upgrade = request.headers.get("Upgrade")
            if type(upgrade) is str and upgrade.lower() == "websocket":
                return await self._websocket_upgrade(request)

            return await self._request(request)

    async def request(self, request: Request) -> Response:
        async with self._operation():
            return await self._request(request)

    async def _request(self, request: Request) -> Response:
        await self._start()
        context = CapabilityRequestContext(request)
        response = await self._dispatch_response("on_request", context)
        if response is not None:
            return response
        if self._host_request is None:
            raise RuntimeError("no host handler for on_request")
        return await self._call_host(
            self._host_request,
            request,
            request=request,
        )

    async def websocket_upgrade(self, request: Request) -> Response:
        async with self._operation():
            return await self._websocket_upgrade(request)

    async def _websocket_upgrade(self, request: Request) -> Response:
        await self._start()
        context = CapabilityWebSocketUpgradeContext(request)
        response = await self._dispatch_response("on_websocket_upgrade", context)
        if response is not None:
            return response
        if self._host_websocket_upgrade is None:
            raise RuntimeError("no host handler for on_websocket_upgrade")
        return await self._call_host(
            self._host_websocket_upgrade,
            request,
            request=request,
        )

    async def websocket_message(self, websocket: object, message: object) -> bool:
        async with self._operation():
            await self._start()
            context = CapabilityWebSocketMessageContext(websocket, message)
            if await self._dispatch_socket("on_websocket_message", context):
                return True
            if self._host_websocket_message is None:
                return False
            result = await self._call_host(
                self._host_websocket_message,
                context,
                connection=websocket,
            )
            return result is True

    async def websocket_close(
        self,
        websocket: object,
        code: int,
        reason: str,
        was_clean: bool,
    ) -> bool:
        async with self._operation():
            await self._start()
            context = CapabilityWebSocketCloseContext(
                websocket,
                code,
                reason,
                was_clean,
            )
            if await self._dispatch_socket("on_websocket_close", context):
                return True
            if self._host_websocket_close is None:
                return False
            result = await self._call_host(
                self._host_websocket_close,
                context,
                connection=websocket,
            )
            return result is True

    async def websocket_error(self, websocket: object, error: object) -> bool:
        async with self._operation():
            await self._start()
            context = CapabilityWebSocketErrorContext(websocket, error)
            if await self._dispatch_socket("on_websocket_error", context):
                return True
            if self._host_websocket_error is None:
                return False
            result = await self._call_host(
                self._host_websocket_error,
                context,
                connection=websocket,
            )
            return result is True

    async def route(
        self,
        *,
        version: int,
        source: LifecycleRouteAddress | None,
        target: LifecycleRouteAddress,
        capability_id: str,
        payload: object,
    ) -> object:
        async with self._operation():
            await self._start()
            return await self._route_local(
                capability_id,
                source=source,
                payload=payload,
                version=version,
                target=target,
            )

    async def alarm(self) -> None:
        async with self._operation():
            self._ensure_root_jobs()
            await self._job_driver.run_alarm(self._start)

    def track_alarm_work(self, awaitable: Awaitable[object]) -> bool:
        return self._job_driver.track(awaitable)

    def _alarm_work_is_tracked(self, awaitable: Awaitable[object]) -> bool:
        return self._job_driver.is_tracked(awaitable)

    async def _notify_host_memory_limit(
        self,
        context: LifecycleMemoryLimitContext,
    ) -> None:
        if self._host_alarm_memory_limit is not None:
            await self._call_host(self._host_alarm_memory_limit, context)

    async def rearm_alarm(self) -> None:
        self._ensure_job_operation_access()
        async with self._operation(continue_existing=True):
            await self._ready_for_jobs()
            await self._rearm_alarm()

    def request_rearm_alarm(self) -> None:
        self._ensure_job_operation_access()
        self._retain_work(lambda: self.rearm_alarm())

    async def dispose(self) -> None:
        current_task = asyncio.current_task()
        reentrant = (
            current_task is self._dispose_owner
            or current_task in self._dispose_cleanup_tasks
            or current_task in self._operation_tasks
        )
        if reentrant:
            raise RuntimeError(
                "Lifecycle cannot be disposed during an active operation"
            )
        current_attempt = self._dispose_attempt
        if current_attempt is not None:
            await asyncio.shield(current_attempt)
            return
        if self._disposed:
            return

        attempt = asyncio.get_running_loop().create_future()
        self._dispose_attempt = attempt
        self._disposing = True
        self._dispose_owner = current_task
        cancellation: asyncio.CancelledError | None = None
        try:
            while not self._operations_idle.is_set():
                try:
                    await self._operations_idle.wait()
                except asyncio.CancelledError as error:
                    cancellation = cancellation or error
            self._seal_registrations()
            if self._owns_physical_alarm:
                try:
                    self._job_queue.prepare()
                except BaseException:
                    self._dispose_attempt = None
                    raise
            self._alarms_disabled = True
        except BaseException as error:
            self._disposing = False
            self._dispose_owner = None
            attempt.set_exception(error)
            attempt.exception()
            raise
        for installation in reversed(self._installations):
            capability = installation.capability
            if not self._is_capability(capability):
                continue
            error, hook_cancellation = await self._run_disposal_awaitable(
                _call_with_context(None, capability.on_dispose),
                allow_job_cancel=True,
            )
            cancellation = cancellation or hook_cancellation
            if error is not None:
                (
                    reporting_error,
                    reporting_cancellation,
                ) = await self._run_disposal_awaitable(self._report_error(error))
                cancellation = cancellation or reporting_cancellation
                if reporting_error is not None:
                    _LOGGER.error(
                        "Lifecycle error reporter failed",
                        exc_info=reporting_error,
                    )
        if self._owns_physical_alarm:
            error, rearm_cancellation = await self._run_disposal_awaitable(
                self._rearm_after_disposal()
            )
            cancellation = cancellation or rearm_cancellation
            if error is not None:
                (
                    reporting_error,
                    reporting_cancellation,
                ) = await self._run_disposal_awaitable(self._report_error(error))
                cancellation = cancellation or reporting_cancellation
                if reporting_error is not None:
                    _LOGGER.error(
                        "Lifecycle error reporter failed",
                        exc_info=reporting_error,
                    )
        self._disposed = True
        self._disposing = False
        self._dispose_owner = None
        attempt.set_result(None)
        if cancellation is not None:
            raise cancellation

    async def _run_disposal_awaitable(
        self,
        awaitable: Coroutine[Any, Any, object],
        *,
        allow_job_cancel: bool = False,
    ) -> tuple[Exception | None, asyncio.CancelledError | None]:
        task = asyncio.create_task(awaitable)
        self._dispose_cleanup_tasks.add(task)
        if allow_job_cancel:
            self._disposal_job_cleanup_tasks.add(task)
        cancellation: asyncio.CancelledError | None = None
        try:
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError as error:
                    cancellation = cancellation or error
                except Exception:
                    break
            try:
                task.result()
            except asyncio.CancelledError as error:
                cancellation = cancellation or error
            except Exception as error:
                return error, cancellation
            return None, cancellation
        finally:
            self._disposal_job_cleanup_tasks.discard(task)
            self._dispose_cleanup_tasks.discard(task)

    async def _dispatch_response(
        self,
        phase: str,
        context: object,
    ) -> Response | None:
        for capability in self._capabilities():
            result = await _call_with_context(
                None,
                getattr(capability, phase),
                context,
            )
            if self._is_response(result):
                return result
        return None

    async def _dispatch_socket(self, phase: str, context: object) -> bool:
        for capability in self._capabilities():
            result = await _call_with_context(
                None,
                getattr(capability, phase),
                context,
            )
            if result is True:
                return True
        return False

    async def _ready_for_capability(self) -> None:
        self._ensure_active()
        if asyncio.current_task() is self._startup_owner:
            return
        await self.start()

    async def _ready_for_jobs(self) -> None:
        self._ensure_root_jobs()
        if self._ready or asyncio.current_task() is self._startup_owner:
            return
        await self._start()

    def _ensure_root_jobs(self) -> None:
        if not self._owns_physical_alarm:
            raise RuntimeError("facet Lifecycle jobs require owner-keyed root routing")

    def _configure_routes(
        self,
        route_address: LifecycleRouteAddress,
        root_route_address: LifecycleRouteAddress,
    ) -> None:
        if (
            self._route_address == route_address
            and self._root_route_address == root_route_address
        ):
            return
        if self._ready:
            raise RuntimeError("Lifecycle routes must be configured before startup")
        self._route_address = route_address
        self._root_route_address = root_route_address

    async def _push_job(
        self,
        owner: str,
        options: LifecycleJobPushOptions,
        *,
        validate_retry: bool = True,
    ) -> LifecycleJob:
        self._ensure_job_operation_access()
        async with self._operation(continue_existing=True):
            await self._ready_for_jobs()
            job = self._job_queue.push(
                owner,
                options,
                validate_retry=validate_retry,
            )
            self._job_driver.note_mutation(job.id)
            await self._rearm_alarm()
            return job

    async def _cancel_job(self, owner: str, job_id: str) -> bool:
        if asyncio.current_task() in self._disposal_job_cleanup_tasks:
            self._ensure_root_jobs()
            cancelled = self._job_queue.cancel(owner, job_id)
            if cancelled:
                self._job_driver.note_mutation(job_id)
            return cancelled
        self._ensure_job_operation_access()
        async with self._operation(continue_existing=True):
            await self._ready_for_jobs()
            cancelled = self._job_queue.cancel(owner, job_id)
            if cancelled:
                self._job_driver.note_mutation(job_id)
            await self._rearm_alarm()
            return cancelled

    async def _reschedule_job(self, owner: str, job_id: str, time: int) -> bool:
        self._ensure_job_operation_access()
        async with self._operation(continue_existing=True):
            await self._ready_for_jobs()
            rescheduled = self._job_queue.reschedule(owner, job_id, time)
            if rescheduled:
                self._job_driver.note_mutation(job_id)
            await self._rearm_alarm()
            return rescheduled

    async def _get_job(
        self,
        owner: str,
        job_id: str,
        *,
        validate_retry: bool = True,
    ) -> LifecycleJob | None:
        self._ensure_job_operation_access()
        async with self._operation(continue_existing=True):
            await self._ready_for_jobs()
            return self._job_queue.get(
                owner,
                job_id,
                validate_retry=validate_retry,
            )

    async def _list_jobs(
        self,
        owner: str,
        *,
        skip_invalid: bool = False,
        validate_retry: bool = True,
    ) -> tuple[LifecycleJob, ...]:
        self._ensure_job_operation_access()
        async with self._operation(continue_existing=True):
            await self._ready_for_jobs()
            return self._job_queue.list(
                owner,
                skip_invalid=skip_invalid,
                validate_retry=validate_retry,
            )

    def _resolve_job_dispatch(self, owner: str) -> _JobDispatch | None:
        if owner == "host":
            host_job = self._host_job
            if host_job is None:
                return None

            async def run_host(context: LifecycleJobContext) -> LifecycleJobOutcome:
                return await self._call_host(host_job, context)

            return _JobDispatch(run_host)

        capability = self._capabilities_by_id.get(owner)
        if capability is None:
            return None
        definitions = static_mro_members(capability)
        if definitions.get("on_job") is LifecycleCapability.on_job:
            return None

        async def run_capability(
            context: LifecycleJobContext,
        ) -> LifecycleJobOutcome:
            return await _call_with_context(None, capability.on_job, context)

        error_handler = None
        if definitions.get("on_job_error") is not LifecycleCapability.on_job_error:

            async def run_error(
                context: LifecycleJobContext,
                error: BaseException,
            ) -> LifecycleJobOutcome:
                return await _call_with_context(
                    None,
                    capability.on_job_error,
                    context,
                    error,
                )

            error_handler = run_error
        return _JobDispatch(run_capability, error_handler)

    async def _run_host_alarm(self) -> None:
        if self._host_alarm is not None:
            await self._call_host(self._host_alarm)

    async def _run_memory_limit_policy(
        self,
        context: LifecycleMemoryLimitContext,
    ) -> None:
        capabilities = self._sealed or self._seal_registrations()
        cancelled: asyncio.CancelledError | None = None
        for capability in capabilities:
            try:
                await _call_with_context(None, capability.on_memory_limit, context)
            except asyncio.CancelledError as error:
                cancelled = error
            except BaseException as error:
                try:
                    await self._report_secondary_error(error)
                except asyncio.CancelledError as reporting_cancelled:
                    cancelled = reporting_cancelled
                except BaseException:
                    pass
        if self._host_alarm_memory_limit is not None:
            try:
                await self._call_host(self._host_alarm_memory_limit, context)
            except asyncio.CancelledError as error:
                cancelled = error
            except BaseException as error:
                try:
                    await self._report_secondary_error(error)
                except asyncio.CancelledError as reporting_cancelled:
                    cancelled = reporting_cancelled
                except BaseException:
                    pass
        if cancelled is not None:
            raise cancelled

    def _memory_limit_strike_limit(self) -> int:
        limit = self._max_alarm_memory_limit_strikes
        if type(limit) is not int or limit < 1:
            raise ValueError("max_alarm_memory_limit_strikes must be an integer >= 1")
        return limit

    async def _reset_after_memory_limit(self, reason: str) -> None:
        if self._reset_alarm is not None:
            await _call_maybe_async(self._reset_alarm, reason)
            return
        abort = getattr(self._ctx, "abort", None)
        if callable(abort):
            # The Python wrapper does not yet expose abort options.
            raw_context = getattr(self._ctx, "_ctx", self._ctx)
            options = to_js(
                {"retryAlarm": False},
                dict_converter=Object.fromEntries,
            )
            callback = create_once_callable(
                lambda: getattr(raw_context, "abort")(reason, options)
            )
            setTimeout(callback, 0)

    def _schedule_alarm_background(self, awaitable: Awaitable[object]) -> bool:
        if self._retain_work_handler is None:
            return False
        self._retain_work_handler(awaitable)
        return True

    async def _arm_alarm_at(self, deadline: int) -> None:
        async with self._alarm_rearm_lock:
            if self._alarms_disabled:
                return
            storage = getattr(self._ctx, "storage")
            await _call_maybe_async(getattr(storage, "setAlarm"), deadline)

    async def _rearm_alarm(self) -> None:
        if self._alarms_disabled:
            return
        if self._starting:
            self._rearm_requested_during_start = True
            return
        task = asyncio.create_task(self._perform_alarm_rearm())
        cancellation: asyncio.CancelledError | None = None
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError as error:
                cancellation = cancellation or error
            except Exception:
                break
        if cancellation is not None:
            try:
                task.result()
            except BaseException as rearm_error:
                await self._report_secondary_error(rearm_error)
            raise cancellation
        task.result()

    async def _perform_alarm_rearm(self) -> None:
        async with self._alarm_rearm_lock:
            if self._alarms_disabled:
                return
            alarm = self._next_alarm_deadline()
            storage = getattr(self._ctx, "storage")
            if alarm is None:
                await _call_maybe_async(getattr(storage, "deleteAlarm"))
            else:
                await _call_maybe_async(getattr(storage, "setAlarm"), alarm)

    def _next_alarm_deadline(self) -> int | None:
        current_time = now_ms()
        alarm = self._job_queue.next_alarm_time(current_time)
        if self._host_alarm_deadline is not None:
            host_alarm = self._host_alarm_deadline(current_time)
            if host_alarm is not None:
                alarm = host_alarm if alarm is None else min(alarm, host_alarm)
        return alarm

    async def _rearm_after_disposal(self) -> None:
        alarm = self._next_alarm_deadline()
        storage = getattr(self._ctx, "storage")
        current = await _call_maybe_async(getattr(storage, "getAlarm"))
        if current == alarm:
            return
        if alarm is None:
            await _call_maybe_async(getattr(storage, "deleteAlarm"))
        else:
            await _call_maybe_async(getattr(storage, "setAlarm"), alarm)

    def _execute_job_sql(
        self,
        query: str,
        *params: object,
    ) -> list[dict[str, Any]]:
        storage = getattr(self._ctx, "storage")
        sql = getattr(storage, "sql")
        return getattr(sql, "exec")(query, *params).toArray()

    async def _report_secondary_error(self, error: BaseException) -> None:
        if isinstance(error, asyncio.CancelledError):
            raise error
        if not isinstance(error, Exception):
            return
        try:
            await self._report_error(error)
        except asyncio.CancelledError:
            raise
        except BaseException:
            _LOGGER.error("Lifecycle error reporter failed", exc_info=error)

    async def _run_in_host_context[T](
        self,
        callback: Callable[[], T | Awaitable[T]],
        scope: LifecycleHostContextScope | None,
    ) -> T:
        async with self._operation():
            scope = scope or LifecycleHostContextScope()
            context = CurrentLifecycleContext(
                self._host,
                request=scope.request,
                connection=scope.connection,
            )
            return await _call_with_context(context, callback)

    async def _call_host(
        self,
        callback: Callable[..., Any],
        *args: object,
        request: Request | None = None,
        connection: object | None = None,
    ) -> Any:
        context = CurrentLifecycleContext(
            self._host,
            request=request,
            connection=connection,
        )
        return await _call_with_context(context, callback, *args)

    async def _emit_event(
        self,
        source: str,
        event_type: str,
        payload: object,
    ) -> None:
        async with self._operation():
            if type(source) is not str or not source.strip():
                raise ValueError("event source must be a non-empty string")
            if type(event_type) is not str or not event_type.strip():
                raise ValueError("event type must be a non-empty string")
            event = LifecycleEvent(source, event_type, payload)
            startup_is_buffering = self._starting or (
                not self._ready and self._startup_attempt is None
            )
            if startup_is_buffering:
                self._pending_events.append(event)
                return
            await self._queue_event(event)

    async def _flush_pending_events(self) -> None:
        pending = self._pending_events
        self._pending_events = []
        done: asyncio.Future[None] | None = None
        for event in pending:
            done = asyncio.get_running_loop().create_future()
            self._event_queue.append(_QueuedEvent(event, done))
        if done is not None:
            await self._drain_event_queue(done)

    async def _queue_event(self, event: LifecycleEvent) -> None:
        done = asyncio.get_running_loop().create_future()
        self._event_queue.append(_QueuedEvent(event, done))
        await self._drain_event_queue(done)

    async def _drain_event_queue(self, done: asyncio.Future[None]) -> None:
        current_task = asyncio.current_task()
        if self._event_draining:
            if current_task is self._event_owner:
                return
            await asyncio.shield(done)
            return

        self._event_draining = True
        self._event_owner = current_task
        try:
            while self._event_queue:
                queued = self._event_queue.popleft()
                try:
                    await self._deliver_event(queued.event)
                except BaseException as error:
                    self._fail_queued_event(queued, error)
                    while self._event_queue:
                        self._fail_queued_event(self._event_queue.popleft(), error)
                    raise
                if not queued.done.done():
                    queued.done.set_result(None)
        finally:
            self._event_owner = None
            self._event_draining = False

    async def _deliver_event(self, event: LifecycleEvent) -> None:
        for listener in self._event_listeners:
            try:
                await _call_with_context(None, listener, event)
            except Exception as error:
                await self._report_error(error)

    async def _report_error(self, error: BaseException) -> None:
        if self._error_handler is None:
            _LOGGER.error("Lifecycle callback failed", exc_info=error)
            return
        try:
            await self._call_host(self._error_handler, error)
        except Exception as reporting_error:
            _LOGGER.error("Lifecycle error reporter failed", exc_info=reporting_error)

    @staticmethod
    def _fail_queued_event(queued: _QueuedEvent, error: BaseException) -> None:
        if queued.done.done():
            return
        queued.done.set_exception(error)
        queued.done.exception()

    async def _route_for(
        self,
        capability_id: str,
        target: LifecycleRouteAddress,
        payload: object,
    ) -> object:
        async with self._operation():
            if (
                asyncio.current_task() is self._startup_owner
                and target != self._route_address
            ):
                raise RuntimeError(
                    "Lifecycle cannot route to another object during startup"
                )
            await self._start()
            source = self._route_address
            self._validate_route_payload(payload)
            envelope = LifecycleRouteEnvelope(
                _ROUTE_VERSION,
                source,
                target,
                capability_id,
                payload,
            )
            if target == source:
                return await self._route_local(
                    envelope.capability_id,
                    source=envelope.source,
                    payload=envelope.payload,
                    version=envelope.version,
                    target=envelope.target,
                )
            if self._route_transport is None:
                raise RuntimeError("Lifecycle has no route transport")
            return await _call_maybe_async(self._route_transport, envelope)

    async def _route_local(
        self,
        capability_id: str,
        *,
        source: LifecycleRouteAddress | None,
        payload: object,
        version: int = _ROUTE_VERSION,
        target: LifecycleRouteAddress | None = None,
    ) -> object:
        if type(version) is not int or version != _ROUTE_VERSION:
            raise ValueError(f"unknown route version: {version}")
        if target is not None:
            local_address = self._route_address or self._root_route_address
            if local_address is None or target != local_address:
                raise PermissionError("route target does not belong to this Lifecycle")
        self._validate_route_payload(payload)
        capability = self._capabilities_by_id.get(capability_id)
        if capability is None:
            raise LookupError(f"unknown route capability: {capability_id}")
        context = LifecycleRouteContext(source, payload)
        return await _call_with_context(None, capability.on_route, context)

    def _validate_route_payload(self, payload: object) -> None:
        encoded = json.dumps(
            payload,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        if len(encoded) > self._max_route_payload_bytes:
            raise ValueError("route payload exceeds the configured byte limit")

    def _retained_work_available(self) -> bool:
        return (
            self._retain_work_handler is not None
            and not self._disposing
            and not self._disposed
        )

    def _retain_work(self, factory: Callable[[], Awaitable[object]]) -> None:
        self._ensure_active()
        if self._retain_work_handler is None:
            raise RuntimeError("retained work is not available")

        retained = self._run_retained_work(factory)
        try:
            self._retain_work_handler(retained)
        except BaseException:
            self._close_awaitable(retained)
            raise

    async def _run_retained_work(
        self,
        factory: Callable[[], Awaitable[object]],
    ) -> object:
        token = _CURRENT_LIFECYCLE_CONTEXT.set(None)
        try:
            work = factory()
            if not inspect.isawaitable(work):
                raise TypeError("retained work factory must return an awaitable")
            return await work
        finally:
            _CURRENT_LIFECYCLE_CONTEXT.reset(token)

    def _capabilities(self) -> tuple[LifecycleCapability, ...]:
        if self._sealed is None:
            raise RuntimeError("Lifecycle is not ready")
        return self._sealed

    def _ensure_active(self) -> None:
        if self._disposed or self._disposing:
            raise RuntimeError("Lifecycle is disposed")

    def _ensure_socket_service_access(self) -> None:
        self._ensure_service_access()

    def _ensure_service_access(self) -> None:
        try:
            task = asyncio.current_task()
        except RuntimeError:
            task = None
        cleanup_or_in_flight = (
            task is self._dispose_owner
            or task in self._dispose_cleanup_tasks
            or task in self._operation_tasks
        )
        if self._disposed or (self._disposing and not cleanup_or_in_flight):
            raise RuntimeError("Lifecycle is disposed")

    def _ensure_job_operation_access(self) -> None:
        try:
            task = asyncio.current_task()
        except RuntimeError:
            task = None
        if self._disposed or (self._disposing and task not in self._operation_tasks):
            raise RuntimeError("Lifecycle is disposed")

    @asynccontextmanager
    async def _operation(
        self,
        *,
        continue_existing: bool = False,
    ) -> AsyncIterator[None]:
        task = asyncio.current_task()
        if task is None:
            raise RuntimeError("Lifecycle operation requires an asyncio task")
        if continue_existing:
            self._ensure_service_access()
        else:
            self._ensure_active()
        self._active_operations += 1
        self._operation_tasks[task] = self._operation_tasks.get(task, 0) + 1
        self._operations_idle.clear()
        try:
            yield
        finally:
            self._active_operations -= 1
            task_operations = self._operation_tasks[task] - 1
            if task_operations:
                self._operation_tasks[task] = task_operations
            else:
                del self._operation_tasks[task]
            if self._active_operations == 0:
                self._operations_idle.set()

    @staticmethod
    def _close_awaitable(awaitable: Awaitable[object]) -> None:
        cancel = getattr(awaitable, "cancel", None)
        if callable(cancel):
            cancel()
            return
        close = getattr(awaitable, "close", None)
        if callable(close):
            close()

    @staticmethod
    def _is_response(value: Any) -> TypeGuard[Response]:
        return issubclass(type(value), Response)

    @staticmethod
    def _bound_lifecycle(capability: LifecycleCapability) -> Lifecycle | None:
        try:
            services = _SERVICE_SLOT.__get__(capability, LifecycleCapability)
        except AttributeError:
            return None
        if type(services) is not _CapabilityServices:
            return None
        return object.__getattribute__(services, "_lifecycle")

    @staticmethod
    def _is_capability(capability: object) -> TypeGuard[LifecycleCapability]:
        return issubclass(type(capability), LifecycleCapability)
