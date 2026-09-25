from __future__ import annotations

import asyncio
import inspect
import traceback
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from contextlib import suppress
from contextvars import ContextVar
from datetime import datetime
from types import CoroutineType
from typing import Any, Literal, TypeVar, cast

import workers
from js import Object  # ty: ignore[unresolved-import]
from js import Request as JsRequest  # ty: ignore[unresolved-import]
from pyodide.ffi import create_proxy, to_js
from workers import DurableObject, Request, Response

from ..lifecycle import (
    Lifecycle,
    LifecycleMemoryLimitContext,
    LifecycleRouteAddress,
    LifecycleRouteEnvelope,
)
from ..lifecycle.fiber import (
    FiberCapability,
    FiberContext,
    FiberInspection,
    FiberRecoveryContext,
    FiberRecoveryResult,
    StartFiberResult,
)
from ..lifecycle.websockets import (
    Connection,
    ConnectionContext,
    JsonT,
    WebSockets,
    _current_websocket_connection,
    _prepare_tags,
)
from ..mcp.client import MCPClientManager, RPCTransportAdapter
from ..schedules import (
    Schedule,
    ScheduleCriteria,
    ScheduleOptions,
    Scheduler,
    _discover_scheduler_callbacks,
)
from ..tasks import Tasks, _discover_task_definitions
from ..workflows import (
    AgentWorkflowFacetOrigin,
    AgentWorkflowPathStep,
    AgentWorkflowRootOrigin,
    RunWorkflowOptions,
    WorkflowCallback,
    WorkflowCallbackHandlers,
    WorkflowEventPayload,
    WorkflowInfo,
    WorkflowInstanceStatus,
    WorkflowMetadataScalar,
    WorkflowOperations,
    WorkflowPage,
    WorkflowQueryCriteria,
    WorkflowStatus,
    decode_workflow_callback,
)
from ._discovery import deferred_method, static_definition_function, static_mro_members
from ._wire import strict_json_loads
from .agent_tools import AgentToolResult, AgentToolRuns
from .connection_state import (
    CF_NO_PROTOCOL_KEY,
    CF_READONLY_KEY,
    set_connection_state_flag,
)
from .error import HookError, RoutingException, RPCError
from .facets import (
    _FACET_KEY_SEP,
    FACET_ID_PREFIX,
    SUB_PREFIX,
    _agent_path_from_value,
    _agent_route_address,
    _facet_identity,
    _facet_key,
    _facet_logical_name,
    _FacetOperationGate,
    _LifecycleRouteStaleWire,
    _LifecycleRouteValueWire,
    _next_sub_hop,
    _parse_parent_path,
    _path_step,
    _route_address_path,
    _route_envelope_from_wire,
    _route_envelope_to_wire,
    _route_rpc_result,
    _StaleLifecycleRoute,
)
from .protocol import (
    McpServers,
    MessageType,
    PathStep,
    SubAgentRecord,
    SubAgentRef,
    identity_frame,
    mcp_servers_frame,
    rpc_error,
    rpc_result,
    state_error_frame,
    state_frame,
)
from .response import StreamingResponse
from .routing import (
    CorsT,
    _is_ws_req,
    camel_to_kebab,
    kebab_to_screaming,
    route_agent_request,
)
from .rpc import (
    _bind_rpc_member,
    _RPC_Func,
    _rpc_metadata,
    _static_callable,
    rpc_callable,
)
from .schema import (
    CORE_SCHEMA_VERSION,
    parse_schema_version,
    prepare_core_schema,
    prepare_core_state_schema,
    read_core_schema_version,
)
from .subagent_relay import (
    BufferedRelayConnection,
    RelayPathStep,
    RelaySession,
    RelayTarget,
)
from .utils import (
    MISSING,
    dumps_wire,
    error_message,
    loads_dict_or_none,
    now_ms,
    url_path,
    url_with_path,
)

_T = TypeVar("_T")


async def _maybe_coro(value: _T | CoroutineType[Any, Any, _T]) -> _T:  # noqa: UP047
    if inspect.iscoroutine(value):
        return cast(_T, await value)

    return cast(_T, value)


async def _call_maybe_async(fn: Callable[..., Any], *args: Any) -> Any:
    result = fn(*args)
    if inspect.isawaitable(result):
        return await result
    return result


def _rpc_to_python(value: object) -> object:
    to_python = getattr(value, "to_py", None)
    return to_python() if callable(to_python) else value


# "server" for server-initiated updates, or the Connection that sent the frame
StateSourceT = Connection | Literal["server"]

STATE_ROW_ID = "cf_state_row_id"

PARENT_PATH_ROW_ID = "cf_parent_path"
_RUNTIME_ENTRY_POINTS = frozenset(
    {
        "_cf_broadcastAgentPath",
        "_cf_invokeAgentPath",
        "_cf_route_lifecycle",
        "_workflow_broadcast",
        "_workflow_handleCallback",
        "_workflow_updateState",
        "alarm",
        "webSocketMessage",
        "webSocketClose",
        "webSocketError",
    }
)


class _AgentMCPBindingResolver:
    def __init__(self, env: object) -> None:
        self._env = env

    async def resolve(
        self,
        binding_name: str,
        name: str,
        props: Mapping[str, Any] | None,
    ) -> object:
        namespace = (
            self._env.get(binding_name)
            if isinstance(self._env, Mapping)
            else getattr(self._env, binding_name, None)
        )
        id_from_name = getattr(namespace, "idFromName", None)
        get = getattr(namespace, "get", None)
        if not callable(id_from_name) or not callable(get):
            raise ValueError(f'MCP Durable Object binding "{binding_name}" not found')

        stub = get(id_from_name(f"rpc:{name}"))
        try:
            initialize = getattr(stub, "__unsafe_ensureInitialized", None)
            if callable(initialize):
                if props is None:
                    await _call_maybe_async(initialize)
                else:
                    await _call_maybe_async(initialize, dict(props))
            return stub
        except BaseException:
            destroy = getattr(stub, "destroy", None)
            if callable(destroy):
                try:
                    cleanup = destroy()
                    if inspect.isawaitable(cleanup):
                        await cleanup
                except Exception:
                    pass
            raise


def _parse_version(raw: Any) -> int:
    return parse_schema_version(raw)


class Agent(DurableObject):
    keep_alive_interval_ms: int = 30_000
    detached_fibers_enabled = False
    fiber_recovery_scan_deadline_ms: int = 10_000
    fiber_recovery_max_age_ms: int = 24 * 60 * 60 * 1000
    fiber_recovery_hook_timeout_ms: int = 10_000
    fiber_recovery_max_backoff_ms: int = 5 * 60 * 1000
    _hide_relay_connections = True
    sub_agent_ws_max_frames = 1_000
    sub_agent_ws_max_bytes = 1024 * 1024
    sub_agent_ws_handler_timeout_s = 30
    agent_tool_recovery_grace_ms = 5 * 60 * 1000
    max_concurrent_agent_tools = 4

    def __init_subclass__(cls, **kwargs: Any) -> None:
        collisions: set[str] = set()
        for base in cls.__mro__:
            if base is Agent:
                break
            collisions.update(_RUNTIME_ENTRY_POINTS.intersection(base.__dict__))
        if collisions:
            names = ", ".join(sorted(collisions))
            raise TypeError(f"Agent runtime entry points cannot be overridden: {names}")
        super().__init_subclass__(**kwargs)

    def __init__(self, ctx, env):
        super().__init__(ctx, env)

        self._alarm_entry_in_progress = False
        self.__state: dict[str, Any] = {}
        self.__state_storage_ready = False
        self.__rpc_meths: dict[str, _RPC_Func] = {}
        self.__rpc_class_callables: set[str] = set()
        self.__handshake_state_connections: set[Connection] = set()
        self.__mcp_broadcast_ready = False
        self.__facet_startup_protocol_pending: tuple[str, str] | None = None
        self.__relay_startup_context: ContextVar[tuple[str, str] | None] = ContextVar(
            "agent_relay_startup_context",
            default=None,
        )

        # None means "not read yet": the ancestor chain loads on first use, keeping it
        # off the constructor path.
        self.__parent_path: list[PathStep] | None = None
        self.__sub_agents_ready = False

        # Keyed by facet key so resolving one child repeatedly reuses a single proxy —
        # see AGENTS.md, FFI traps.
        self.__facet_proxies: dict[str, Any] = {}
        self.__facet_operation_gates: dict[str, _FacetOperationGate] = {}
        self.__active_facet_deletions: set[str] = set()
        self.__pending_relay_targets: dict[int, RelayTarget] = {}

        members = static_mro_members(self)
        relay_max_frames = members.get("sub_agent_ws_max_frames", 1_000)
        if type(relay_max_frames) is not int:
            relay_max_frames = 1_000
        relay_max_bytes = members.get("sub_agent_ws_max_bytes", 1024 * 1024)
        if type(relay_max_bytes) is not int:
            relay_max_bytes = 1024 * 1024
        configured_timeout = members.get("sub_agent_ws_handler_timeout_s", 30)
        relay_timeout = (
            float(configured_timeout)
            if isinstance(configured_timeout, (int, float))
            and not isinstance(configured_timeout, bool)
            else 30.0
        )
        hide_relay_connections = members.get("_hide_relay_connections", True)
        if type(hide_relay_connections) is not bool:
            hide_relay_connections = True
        self.scheduler = Scheduler(
            _discover_scheduler_callbacks(self, members),
            on_error=self._report_scheduler_error,
        )
        self.tasks = Tasks(
            _discover_task_definitions(self, members),
            on_error=self._report_tasks_error,
        )
        self.mcp = MCPClientManager(
            type(self).__name__,
            "0.1.0",
            transports={"rpc": RPCTransportAdapter(_AgentMCPBindingResolver(env))},
        )
        self.mcp.add_state_listener(lambda: self._publish_mcp_state_change())
        self._workflows = WorkflowOperations.for_agent(
            self.sql,
            self.env,
            lambda agent_binding: Agent._workflow_origin(self, agent_binding),
            callbacks=WorkflowCallbackHandlers(
                on_progress=lambda *args: self.on_workflow_progress(*args),
                on_complete=lambda *args: self.on_workflow_complete(*args),
                on_error=lambda *args: self.on_workflow_error(*args),
                on_event=lambda *args: self.on_workflow_event(*args),
            ),
        )

        self._websockets = WebSockets(
            on_connect=self._dispatch_connect,
            on_message=self._dispatch_message,
            on_close=self._dispatch_close,
            on_error=lambda *args: self.on_error(*args),
            get_connection_tags=lambda *args: self.get_connection_tags(*args),
            socket_source=self._owned_websockets,
            ensure_ready=self._ensure_initialized,
            before_upgrade=self._before_websocket_upgrade,
            relay_target_for_request=self._websocket_relay_target_for_request,
            relay_target_is_valid=self._websocket_relay_target_is_valid,
            relay_forward=self._forward_websocket_relay,
            relay_max_frames=relay_max_frames,
            relay_max_bytes=relay_max_bytes,
            relay_timeout=relay_timeout,
            hide_relay_connections=hide_relay_connections,
        )
        route_address, root_route_address = self._lifecycle_route_addresses()
        self._lifecycle = Lifecycle(
            ctx,
            host=self,
            prepare=self._prepare_for_startup,
            on_start=self._lifecycle_host_start,
            on_request=self._lifecycle_host_request,
            on_alarm=self._lifecycle_host_alarm,
            on_alarm_memory_limit=self._lifecycle_host_memory_limit,
            alarm_deadline=lambda now: self._collect_alarm_deadline(now, None),
            on_error=self._report_lifecycle_error,
            retain_work=lambda awaitable: workers.waitUntil(awaitable),
            route_address=route_address,
            root_route_address=root_route_address,
            route_transport=self._route_lifecycle_transport,
            owns_physical_alarm=self._lifecycle_owns_physical_alarm(),
        )
        self._lifecycle.use(self.scheduler)
        self._lifecycle.use(self.tasks)
        self._lifecycle.use(self.mcp)
        self._lifecycle.use(self._websockets, fallback=True)
        self._connections = self._websockets._connections

        self._fiber = FiberCapability(
            on_fiber_recovered=lambda context: self.on_fiber_recovered(context),
            on_internal_fiber_recovery=lambda context: (
                self._handle_internal_fiber_recovery(context)
            ),
            keep_alive_interval_ms=lambda: self.keep_alive_interval_ms,
            detached_fibers_enabled=lambda: self.detached_fibers_enabled,
            recovery_scan_deadline_ms=lambda: self.fiber_recovery_scan_deadline_ms,
            recovery_max_age_ms=lambda: self.fiber_recovery_max_age_ms,
            recovery_hook_timeout_ms=lambda: self.fiber_recovery_hook_timeout_ms,
            recovery_max_backoff_ms=lambda: self.fiber_recovery_max_backoff_ms,
            maintenance_enabled=lambda: self._facet_name is None,
            defer_startup_recovery_to_alarm=lambda: (
                self._alarm_entry_in_progress
                and self._fiber._has_future_maintenance_wake()
            ),
            defer_internal_recovery=lambda context: self._defer_internal_fiber_recovery(
                context
            ),
            shared_schema=True,
        )
        self._lifecycle.use(self._fiber)
        self._agent_tool_runs = AgentToolRuns(
            sql=self.sql,
            publish=lambda frame: self.broadcast_json(frame),
            resolve_sub_agent=lambda class_name, name: self._resolve_sub_agent(
                class_name, name
            ),
            max_concurrent=lambda: self.max_concurrent_agent_tools,
            recovery_grace_ms=lambda: self.agent_tool_recovery_grace_ms,
        )

        for child_name, child in static_mro_members(self).items():
            if _static_callable(child):
                self.__rpc_class_callables.add(child_name)

            metadata = _rpc_metadata(child)
            if metadata is None:
                continue
            bound = _bind_rpc_member(child, self)
            self.__rpc_meths[child_name] = _RPC_Func(
                bound,
                streaming=metadata.streaming,
            )

    @property
    def _facet_name(self) -> str | None:
        return _facet_logical_name(self.ctx.id.name)

    def _lifecycle_route_addresses(
        self,
    ) -> tuple[LifecycleRouteAddress | None, LifecycleRouteAddress | None]:
        if _facet_logical_name(self.ctx.id.name) is not None:
            return None, None
        if not isinstance(self.ctx.id.name, str) or not self.ctx.id.name:
            return None, None
        root = _agent_route_address([_path_step(type(self).__name__, self.ctx.id.name)])
        return None, root

    def _configure_lifecycle_routes(self) -> None:
        if self._facet_name is None:
            return
        path = self.self_path
        if len(path) < 2:
            return
        self._lifecycle._configure_routes(
            _agent_route_address(path),
            _agent_route_address(path[:1]),
        )

    async def _route_lifecycle_transport(
        self,
        envelope: LifecycleRouteEnvelope,
    ) -> object:
        target_path = _route_address_path(envelope.target)
        if self._facet_name is not None and len(target_path) == 1:
            result = await self._root_agent_stub()._cf_route_lifecycle(
                _route_envelope_to_wire(envelope)
            )
            return _route_rpc_result(result)

        result = await self._route_lifecycle_envelope(envelope)
        if isinstance(result, _StaleLifecycleRoute):
            stale_path = result.path
            if len(stale_path) < 2 or target_path[: len(stale_path)] != stale_path:
                raise ValueError(
                    "stale Lifecycle route is outside the requested target"
                )
            if self._facet_name is not None:
                raise RuntimeError("stale Lifecycle routes must settle at the root")
            await self._cleanup_facet_prefix(stale_path)
            return False
        return result

    async def _route_lifecycle_envelope(
        self,
        envelope: LifecycleRouteEnvelope,
    ) -> object:
        self_path = self.self_path
        target_path = _route_address_path(envelope.target)
        if target_path[: len(self_path)] != self_path:
            raise PermissionError("Lifecycle route target is outside this Agent tree")
        if envelope.source is not None:
            source_path = _route_address_path(envelope.source)
            if source_path[:1] != self_path[:1]:
                raise PermissionError(
                    "Lifecycle route source is outside this Agent tree"
                )
        if len(target_path) == len(self_path):
            return await self._lifecycle.route(
                version=envelope.version,
                source=envelope.source,
                target=envelope.target,
                capability_id=envelope.capability_id,
                payload=envelope.payload,
            )

        # TODO: this definitely needs to be made clearer

        next_step = target_path[len(self_path)]
        class_name = next_step["className"]
        name = next_step["name"]
        if not self._has_sub_agent_row(class_name, name):
            return _StaleLifecycleRoute(target_path[: len(self_path) + 1])
        stub = await self._resolve_sub_agent(class_name, name)
        result = await stub._cf_route_lifecycle(_route_envelope_to_wire(envelope))
        routed = _route_rpc_result(result)
        if isinstance(routed, _StaleLifecycleRoute):
            existing_prefix = target_path[: len(self_path) + 1]
            if routed.path[: len(existing_prefix)] != existing_prefix or len(
                routed.path
            ) <= len(existing_prefix):
                raise ValueError("stale Lifecycle route does not follow the routed hop")
        return routed

    async def _cf_route_lifecycle(self, envelope_json: str) -> str:
        envelope = _route_envelope_from_wire(envelope_json)
        target_path = _route_address_path(envelope.target)
        if self._facet_name is None:
            if envelope.source is None or target_path != self.self_path:
                raise PermissionError("root route RPC only accepts facet-to-root work")
            source_path = _route_address_path(envelope.source)
            if len(source_path) < 2:
                raise PermissionError("root route RPC requires a facet source")
            source_facet = source_path[1]
            key = _facet_key(source_facet["className"], source_facet["name"])
            await self._ensure_initialized()
            gate = self.__facet_operation_gates.setdefault(key, _FacetOperationGate())
            gate.users += 1
            try:
                async with gate.lock:
                    if not self._has_sub_agent_row(
                        source_facet["className"], source_facet["name"]
                    ):
                        raise PermissionError(
                            "root route RPC source is no longer registered"
                        )
                    result = await self._route_lifecycle_envelope(envelope)
            finally:
                gate.users -= 1
                if gate.users == 0:
                    self.__facet_operation_gates.pop(key, None)
        elif envelope.source is not None:
            raise PermissionError("facet route RPC only accepts root dispatch")
        else:
            result = await self._route_lifecycle_envelope(envelope)
        if isinstance(result, _StaleLifecycleRoute):
            return dumps_wire(_LifecycleRouteStaleWire(type="stale", path=result.path))
        return dumps_wire(_LifecycleRouteValueWire(type="value", value=result))

    def _root_agent_stub(self) -> Any:
        parent_path = self.parent_path
        if not parent_path:
            raise RoutingException("facet routing requires a root parent")
        root = parent_path[0]
        namespace = self._facet_class_handle(root["className"])
        return namespace.get(namespace.idFromName(root["name"]))

    def _lifecycle_owns_physical_alarm(self) -> bool:
        return _facet_logical_name(self.ctx.id.name) is None

    def _alarm_initializes_through_lifecycle(self) -> bool:
        return True

    def _owned_websockets(self, tag: str | None = None):
        # A child shares its host, so getWebSockets() would hand it the parent's sockets
        # and first use would cross Durable Object I/O contexts.
        if self._facet_name is not None:
            return ()
        return self.ctx.getWebSockets() if tag is None else self.ctx.getWebSockets(tag)

    def _websocket_relay_target_for_request(
        self,
        request: Request,
    ) -> RelayTarget | None:
        return self.__pending_relay_targets.get(id(request))

    def _before_websocket_upgrade(self, request: Request) -> Response | None:
        if self._facet_name is not None:
            return Response("Not Found", status=404)
        return None

    def _websocket_relay_target_is_valid(self, target: RelayTarget) -> bool:
        if "path" not in target:
            hop = _next_sub_hop(url_path(target["url"]), is_child=False)
            return hop is not None and _FACET_KEY_SEP not in hop[1]
        path = self._relay_target_path(target)
        self_path = self.self_path
        return (
            path is not None
            and len(path) > len(self_path)
            and path[: len(self_path)] == self_path
        )

    async def _forward_websocket_relay(self, payload: dict[str, Any]) -> str:
        return await self._forward_ws_relay(payload, gate=False)

    @property
    def name(self) -> str:
        # A child's routing id encodes its whole ancestor path, but everything
        # downstream wants the name it was spawned under.
        return self._facet_name or self.ctx.id.name

    def sql(self, query: str, *params: Any) -> list[dict[str, Any]]:
        return self.ctx.storage.sql.exec(query, *params).toArray()

    async def _ensure_initialized(self) -> None:
        await self._lifecycle.ready()

    async def _prepare_for_startup(self, _sql: object) -> None:
        self._run_schema_migration()
        self._hydrate_state()
        self._configure_lifecycle_routes()

    def _run_schema_migration(self) -> None:
        prepare_core_schema(self.sql)
        if self._read_schema_version() > CORE_SCHEMA_VERSION:
            return
        self._prepare_agent_tool_compat_schema()

    def _prepare_agent_tool_compat_schema(self) -> None:
        self._agent_tool_runs.prepare()

    async def _report_error(
        self,
        error: BaseException,
        connection: Connection | None = None,
    ) -> None:
        await self._websockets.report_error(error, connection)

    async def _dispatch_hook(
        self,
        name: str,
        hook: Callable[..., Any],
        *args: Any,
        connection: Connection | None = None,
        propagate: bool,
    ) -> Any:
        return await self._websockets.dispatch_hook(
            name,
            hook,
            *args,
            connection=connection,
            propagate=propagate,
        )

    def _read_state_cell(self, row_id: str) -> Any:
        # MISSING for an absent row, so a caller can still tell that from a row holding
        # SQL NULL. Deliberately unguarded: two callers need a failed read to stay
        # distinct from an empty one, so the try/except belongs to them.
        rows = self.sql("SELECT state FROM cf_agents_state WHERE id = ?", row_id)
        return rows[0]["state"] if rows else MISSING

    def _write_state_cell(self, row_id: str, payload: str) -> None:
        # Upsert, not UPDATE: an UPDATE silently no-ops on a missing row, which would
        # look like a successful save until the next hydrate found nothing.
        self.sql(
            "INSERT OR REPLACE INTO cf_agents_state (id, state) VALUES (?, ?)",
            row_id,
            payload,
        )

    def _read_schema_version(self) -> int:
        return read_core_schema_version(self.sql)

    def _safe_initial_state(self) -> dict[str, Any]:
        # A bad initial state must not prevent startup from repairing persisted state.
        try:
            result = self.initial_state()
        except Exception:  # noqa: BLE001
            return {}

        if not isinstance(result, dict):
            return {}

        return result

    def _hydrate_state(self) -> None:
        # Safe default up front, so any failure below still leaves valid state.
        self.__state = self._safe_initial_state()

        try:
            raw = self._read_state_cell(STATE_ROW_ID)
        except Exception:  # noqa: BLE001
            # Keep the in-memory default and let the next startup retry storage.
            return

        self.__state_storage_ready = True

        if raw is MISSING:
            # No prior state. The first set_state writes the row; until then the in-
            # memory default is enough.
            return

        stored = loads_dict_or_none(raw)
        if stored is not None:
            self.__state = stored
            return

        # Present but unparseable, so overwrite rather than fail identically forever.
        # A missing table or invalid initial value must not prevent a later wake from
        # retrying startup.
        with suppress(Exception):
            self._write_state_cell(STATE_ROW_ID, dumps_wire(self.__state))

    def initial_state(self) -> dict[str, Any]:
        return {}

    def get_mcp_servers(self) -> McpServers:
        if not self.mcp._started:
            return McpServers(servers={}, tools=[], prompts=[], resources=[])
        return self.mcp.get_mcp_servers()

    async def add_mcp_server(
        self,
        server_id: str,
        *,
        url: str,
        name: str,
        callback_url: str = "",
        client_id: str | None = None,
        client: Mapping[str, Any] | None = None,
        transport: Mapping[str, Any] | None = None,
        retry: Mapping[str, Any] | None = None,
        binding_name: str | None = None,
        props: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Register and connect one HTTP or RPC MCP server."""

        await self._ensure_initialized()
        registered_id = await self.mcp.register_server(
            server_id,
            url=url,
            name=name,
            callback_url=callback_url,
            client_id=client_id,
            client=client,
            transport=transport,
            retry=retry,
            binding_name=binding_name,
            props=props,
        )
        result = await self.mcp.connect_to_server(registered_id)
        return {"id": registered_id, **result}

    async def remove_mcp_server(self, server_id: str) -> None:
        """Remove a durable MCP server registration and its live connection."""

        await self._ensure_initialized()
        await self.mcp.remove_server(server_id)

    def should_connection_be_readonly(
        self,
        connection: Connection,
        ctx: ConnectionContext,
    ) -> bool:
        return False

    def set_connection_readonly(
        self,
        connection: Connection,
        readonly: bool = True,
    ) -> None:
        raw = connection._get_raw_state()
        connection._set_raw_state(
            set_connection_state_flag(
                raw,
                CF_READONLY_KEY,
                True if readonly else None,
            )
        )

    def is_connection_readonly(self, connection: Connection) -> bool:
        raw = connection._get_raw_state()
        return isinstance(raw, dict) and bool(raw.get(CF_READONLY_KEY))

    def should_send_protocol_messages(
        self,
        connection: Connection,
        ctx: ConnectionContext,
    ) -> bool:
        return True

    def is_connection_protocol_enabled(self, connection: Connection) -> bool:
        raw = connection._get_raw_state()
        return not isinstance(raw, dict) or not bool(raw.get(CF_NO_PROTOCOL_KEY))

    def _set_connection_no_protocol(self, connection: Connection) -> None:
        raw = connection._get_raw_state()
        connection._set_raw_state(
            set_connection_state_flag(raw, CF_NO_PROTOCOL_KEY, True)
        )

    async def on_error(
        self,
        error: BaseException,
        connection: Connection | None = None,
    ) -> None:
        where = "server" if connection is None else f"connection {connection.id}"
        print(f"[Agent] error on {where}: {error}")
        traceback.print_exception(error)
        raise error

    async def _lifecycle_host_start(self) -> None:
        if self._facet_name is None:
            self._broadcast_mcp_servers()
            self.__mcp_broadcast_ready = True
        await self._retry_pending_facet_deletions()
        await self._dispatch_hook("on_start", self.on_start, propagate=True)
        if self._facet_name is not None:
            self.__mcp_broadcast_ready = True
            startup = self.__relay_startup_context.get()
            if startup is not None:
                self.__facet_startup_protocol_pending = startup
            else:
                self._retain_facet_protocol_snapshot(())

    async def _lifecycle_host_request(self, request: Request) -> Response:
        return await self._dispatch_hook(
            "on_request",
            self._dispatch_request,
            request,
            propagate=True,
        )

    async def _alarm_impl(self) -> None:
        await self._lifecycle.alarm()

    async def _lifecycle_host_alarm(self) -> None:
        await self._alarm_housekeeping()
        await self._dispatch_hook("on_alarm", self.on_alarm, propagate=True)

    async def _lifecycle_host_memory_limit(
        self,
        context: LifecycleMemoryLimitContext,
    ) -> None:
        await _call_maybe_async(self.on_alarm_memory_limit, context)

    async def _report_lifecycle_error(self, error: BaseException) -> None:
        try:
            await _call_maybe_async(self.on_error, error, None)
        except BaseException as raised:  # noqa: BLE001
            if raised is not error:
                traceback.print_exception(raised)

    async def _report_scheduler_error(self, error: BaseException) -> None:
        await self.scheduler.lifecycle.run_in_host_context(
            lambda: self.on_error(error, None)
        )

    async def _report_tasks_error(self, error: BaseException) -> None:
        await self.tasks.lifecycle.run_in_host_context(
            lambda: self.on_error(error, None)
        )

    async def _alarm_housekeeping(self) -> None: ...

    def _collect_alarm_deadline(self, now: int, current: int | None) -> int | None:
        return current

    async def _schedule_next_alarm(self) -> None:
        await self._lifecycle.rearm_alarm()

    def _spawn_reschedule(self) -> None:
        if self._facet_name is not None:
            return
        self._lifecycle.request_rearm_alarm()

    async def alarm(self) -> None:
        self._alarm_entry_in_progress = True
        try:
            await self._alarm_impl()
        finally:
            self._alarm_entry_in_progress = False

    async def _socket_entry(
        self,
        websocket: object,
        callback: Callable[[], Awaitable[bool]],
        *,
        close: tuple[int, str] | None = None,
    ) -> bool:
        try:
            return await callback()
        except HookError:
            pass
        except Exception as error:  # noqa: BLE001
            await self._report_error(error, self._websockets.adopt(websocket))

        connection = self._websockets.adopt(websocket)
        if close is not None and connection is not None:
            code, reason = close
            self._websockets._reciprocate_close(websocket, code, reason)
            self._websockets.discard(connection)
        return connection is not None

    async def webSocketMessage(self, websocket: object, message: object) -> bool:
        return await self._socket_entry(
            websocket,
            lambda: self._lifecycle.websocket_message(websocket, message),
        )

    async def webSocketClose(
        self,
        websocket: object,
        code: int,
        reason: str,
        was_clean: bool,
    ) -> bool:
        return await self._socket_entry(
            websocket,
            lambda: self._lifecycle.websocket_close(
                websocket,
                code,
                reason,
                was_clean,
            ),
            close=(code, reason),
        )

    async def webSocketError(self, websocket: object, error: object) -> bool:
        return await self._socket_entry(
            websocket,
            lambda: self._lifecycle.websocket_error(websocket, error),
        )

    def broadcast(self, data: str, exclude: Iterable[str] = ()) -> None:
        self._websockets.broadcast(data, exclude=exclude)

    def broadcast_json(self, data: JsonT, exclude: Iterable[str] = ()) -> None:
        self._websockets.broadcast_json(data, exclude=exclude)

    def get_connection(self, id: str) -> Connection | None:
        return self._websockets.get_connection(id)

    def get_connections(self, tag: str | None = None) -> list[Connection]:
        return list(self._websockets.get_connections(tag))

    async def on_start(self) -> None: ...

    async def on_connect(
        self,
        connection: Connection,
        ctx: ConnectionContext,
    ) -> None: ...

    async def on_message(self, connection: Connection, message: str) -> None: ...

    async def on_close(
        self,
        connection: Connection,
        code: int,
        reason: str,
        was_clean: bool,
    ) -> None: ...

    async def on_request(self, request: Request) -> Response:
        return Response(None, status=404)

    async def on_alarm(self) -> None: ...

    async def on_alarm_memory_limit(
        self,
        context: LifecycleMemoryLimitContext,
    ) -> None:
        """React after Lifecycle applies an alarm memory-limit policy."""

    def get_connection_tags(
        self,
        connection: Connection,
        ctx: ConnectionContext,
    ) -> list[str]:
        return []

    def _workflow_origin(
        self,
        agent_binding: str | None,
    ) -> AgentWorkflowRootOrigin | AgentWorkflowFacetOrigin:
        path = self.self_path
        root = path[0]
        binding = agent_binding or self._find_agent_binding_name(root["className"])
        if binding is None:
            raise ValueError(
                "Could not detect Agent binding name from class name. "
                "Pass it explicitly via RunWorkflowOptions.agent_binding"
            )
        if self._facet_name is None:
            return AgentWorkflowRootOrigin(binding=binding, name=self.name)
        return AgentWorkflowFacetOrigin(
            root_binding=binding,
            path=tuple(
                AgentWorkflowPathStep(
                    class_name=step["className"],
                    name=step["name"],
                )
                for step in path
            ),
        )

    def _find_agent_binding_name(self, class_name: str) -> str | None:
        expected = camel_to_kebab(class_name)
        env = getattr(self.env, "_env", self.env)
        try:
            keys = Object.keys(env)
        except Exception:  # noqa: BLE001
            return None
        for raw_key in keys:
            key = str(raw_key)
            if camel_to_kebab(key) != expected:
                continue
            try:
                value = env.get(key) if isinstance(env, Mapping) else getattr(env, key)
            except Exception:  # noqa: BLE001
                continue
            if callable(getattr(value, "idFromName", None)):
                return key
        return None

    async def run_workflow(
        self,
        workflow_name: str,
        params: Mapping[str, object],
        options: RunWorkflowOptions | None = None,
    ) -> str:
        """Start and track a native Workflow from this Agent or facet."""

        await self._ensure_initialized()
        return await self._workflows.run_workflow(workflow_name, params, options)

    def get_workflow(self, workflow_id: str) -> WorkflowInfo | None:
        """Return one locally tracked Workflow."""

        return self._workflows.get_workflow(workflow_id)

    def get_workflows(
        self,
        criteria: WorkflowQueryCriteria | None = None,
    ) -> WorkflowPage:
        """Return a page of locally tracked Workflows."""

        return self._workflows.get_workflows(criteria)

    async def get_workflow_status(
        self,
        workflow_name: str,
        workflow_id: str,
    ) -> WorkflowInstanceStatus:
        """Refresh and return a native Workflow's status."""

        await self._ensure_initialized()
        return await self._workflows.get_workflow_status(workflow_name, workflow_id)

    async def send_workflow_event(
        self,
        workflow_name: str,
        workflow_id: str,
        event: WorkflowEventPayload,
    ) -> None:
        """Send an event to a native Workflow instance."""

        await self._ensure_initialized()
        await self._workflows.send_workflow_event(workflow_name, workflow_id, event)

    async def approve_workflow(
        self,
        workflow_id: str,
        *,
        reason: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        """Approve a Workflow waiting for the standard approval event."""

        await self._ensure_initialized()
        await self._workflows.approve_workflow(
            workflow_id,
            reason=reason,
            metadata=metadata,
        )

    async def reject_workflow(
        self,
        workflow_id: str,
        *,
        reason: str | None = None,
    ) -> None:
        """Reject a Workflow waiting for the standard approval event."""

        await self._ensure_initialized()
        await self._workflows.reject_workflow(workflow_id, reason=reason)

    async def pause_workflow(self, workflow_id: str) -> None:
        """Pause a locally tracked Workflow."""

        await self._ensure_initialized()
        await self._workflows.pause_workflow(workflow_id)

    async def resume_workflow(self, workflow_id: str) -> None:
        """Resume a locally tracked Workflow."""

        await self._ensure_initialized()
        await self._workflows.resume_workflow(workflow_id)

    async def terminate_workflow(self, workflow_id: str) -> None:
        """Terminate a locally tracked Workflow."""

        await self._ensure_initialized()
        await self._workflows.terminate_workflow(workflow_id)

    async def restart_workflow(
        self,
        workflow_id: str,
        *,
        reset_tracking: bool = True,
    ) -> None:
        """Restart a locally tracked Workflow."""

        await self._ensure_initialized()
        await self._workflows.restart_workflow(
            workflow_id,
            reset_tracking=reset_tracking,
        )

    def delete_workflow(self, workflow_id: str) -> bool:
        """Delete one local Workflow tracking row."""

        return self._workflows.delete_workflow(workflow_id)

    def delete_workflows(
        self,
        *,
        status: WorkflowStatus | Sequence[WorkflowStatus] | None = None,
        workflow_name: str | None = None,
        metadata: Mapping[str, WorkflowMetadataScalar] | None = None,
        created_before: datetime | None = None,
    ) -> int:
        """Delete local Workflow tracking rows matching exact filters."""

        return self._workflows.delete_workflows(
            status=status,
            workflow_name=workflow_name,
            metadata=metadata,
            created_before=created_before,
        )

    def migrate_workflow_binding(self, old_name: str, new_name: str) -> int:
        """Rename a Workflow binding in retained tracking rows."""

        return self._workflows.migrate_workflow_binding(old_name, new_name)

    async def on_workflow_callback(self, callback: WorkflowCallback) -> None:
        """Apply one Workflow callback and dispatch its specific hook."""

        await self._workflows.handle_callback(callback)

    async def on_workflow_progress(
        self,
        workflow_name: str,
        workflow_id: str,
        progress: object,
    ) -> None: ...

    async def on_workflow_complete(
        self,
        workflow_name: str,
        workflow_id: str,
        result: object | None,
    ) -> None: ...

    async def on_workflow_error(
        self,
        workflow_name: str,
        workflow_id: str,
        error: str,
    ) -> None:
        print(f"Workflow error [{workflow_name}/{workflow_id}]: {error}")

    async def on_workflow_event(
        self,
        workflow_name: str,
        workflow_id: str,
        event: object,
    ) -> None: ...

    async def _workflow_handleCallback(self, callback: object) -> None:
        await self._ensure_initialized()
        value = _rpc_to_python(callback)
        if not isinstance(value, Mapping):
            raise TypeError("Workflow callback must be an object")
        decoded = decode_workflow_callback(value)
        await _call_maybe_async(self.on_workflow_callback, decoded)

    async def _workflow_broadcast(self, message: object) -> None:
        await self._ensure_initialized()
        payload = dumps_wire(_rpc_to_python(message))
        if self._facet_name is None:
            self.broadcast(payload)
            return
        await self._root_agent_stub()._cf_broadcastAgentPath(self.self_path, payload)

    async def _workflow_updateState(
        self,
        action: str,
        state: object = MISSING,
    ) -> None:
        await self._ensure_initialized()
        value = _rpc_to_python(state)
        if action == "reset":
            initial = self.initial_state()
            if not isinstance(initial, dict):
                raise TypeError("initial_state must return an object")
            self.set_state(initial)
            return
        if not isinstance(value, dict):
            raise TypeError("Workflow state updates must contain an object")
        if action == "set":
            self.set_state(value)
            return
        if action == "merge":
            self.set_state({**self.state, **value})
            return
        raise ValueError(f"Unknown Workflow state action: {action!r}")

    async def _dispatch_request(self, request: Request) -> Response:
        return await _call_maybe_async(self.on_request, request)

    async def _dispatch_close(
        self,
        connection: Connection,
        code: int,
        reason: str,
        was_clean: bool,
    ) -> None:
        await _call_maybe_async(self.on_close, connection, code, reason, was_clean)

    async def schedule(
        self,
        when: datetime | str | int | float,
        callback: str,
        payload: object = MISSING,
        options: ScheduleOptions | None = None,
    ) -> Schedule:
        """Create a date, delay, or cron schedule for a decorated callback."""
        return await self.scheduler.set(when, callback, payload, options)

    async def schedule_every(
        self,
        interval_seconds: int | float,
        callback: str,
        payload: object = MISSING,
        options: ScheduleOptions | None = None,
    ) -> Schedule:
        """Create a recurring interval schedule for a decorated callback."""
        return await self.scheduler.every(
            interval_seconds,
            callback,
            payload,
            options,
        )

    async def get_schedule_by_id(self, id: str) -> Schedule | None:
        """Return an owned schedule by ID when present."""
        return await self.scheduler.get(id)

    async def list_schedules(
        self,
        criteria: ScheduleCriteria | None = None,
    ) -> tuple[Schedule, ...]:
        """Return owned schedules matching the supplied criteria."""
        return await self.scheduler.list(criteria)

    async def cancel_schedule(self, id: str) -> bool:
        """Cancel an owned schedule and report whether it existed."""
        return await self.scheduler.cancel(id)

    def _delete_settled_fiber(self, fiber_id: str) -> None:
        self._fiber._delete_settled_fiber(fiber_id)

    async def run_fiber(
        self, name: str, fn: Callable[[FiberContext], Awaitable[Any]]
    ) -> Any:
        return await self._fiber.run_fiber(name, fn)

    def stash(self, data: Any) -> None:
        self._fiber.stash(data)

    async def keep_alive(self) -> Callable[[], None]:
        return await self._fiber.keep_alive()

    async def keep_alive_while(self, fn: Callable[[], Awaitable[Any]]) -> Any:
        return await self._fiber.keep_alive_while(fn)

    async def start_fiber(
        self,
        name: str,
        fn: Callable[[FiberContext], Awaitable[Any]],
        *,
        fiber_id: str | None = None,
        idempotency_key: str | None = None,
        metadata: dict[str, Any] | None = None,
        wait_for_completion: bool = False,
    ) -> StartFiberResult:
        return await self._fiber.start_fiber(
            name,
            fn,
            fiber_id=fiber_id,
            idempotency_key=idempotency_key,
            metadata=metadata,
            wait_for_completion=wait_for_completion,
        )

    async def inspect_fiber(self, fiber_id: str) -> FiberInspection | None:
        return await self._fiber.inspect_fiber(fiber_id)

    async def inspect_fiber_by_key(
        self, idempotency_key: str
    ) -> FiberInspection | None:
        return await self._fiber.inspect_fiber_by_key(idempotency_key)

    async def list_fibers(
        self,
        *,
        status: str | list[str] | None = None,
        name: str | None = None,
        limit: int | None = None,
    ) -> list[FiberInspection]:
        return await self._fiber.list_fibers(status=status, name=name, limit=limit)

    async def cancel_fiber(self, fiber_id: str, reason: str | None = None) -> bool:
        return await self._fiber.cancel_fiber(fiber_id, reason)

    async def cancel_fiber_by_key(
        self, idempotency_key: str, reason: str | None = None
    ) -> bool:
        return await self._fiber.cancel_fiber_by_key(idempotency_key, reason)

    async def resolve_fiber(self, fiber_id: str, result: FiberRecoveryResult) -> bool:
        return await self._fiber.resolve_fiber(fiber_id, result)

    async def delete_fibers(
        self,
        *,
        status: str | list[str] | None = None,
        settled_before: int | None = None,
        limit: int | None = None,
    ) -> int:
        return await self._fiber.delete_fibers(
            status=status,
            settled_before=settled_before,
            limit=limit,
        )

    async def on_fiber_recovered(
        self, context: FiberRecoveryContext
    ) -> FiberRecoveryResult | None:
        return None

    async def _handle_internal_fiber_recovery(self, ctx: FiberRecoveryContext) -> bool:
        return False

    def _defer_internal_fiber_recovery(self, ctx: FiberRecoveryContext) -> bool:
        return False

    async def run_agent_tool(
        self,
        cls: Any,
        *,
        input: Any,
        run_id: str | None = None,
        parent_tool_call_id: str | None = None,
        display_order: int = 0,
        input_preview: Any = MISSING,
        display: Any = MISSING,
        abort: asyncio.Event | None = None,
    ) -> AgentToolResult:
        await self._ensure_initialized()
        return await self._agent_tool_runs.run_agent_tool(
            cls,
            input=input,
            run_id=run_id,
            parent_tool_call_id=parent_tool_call_id,
            display_order=display_order,
            input_preview=input_preview,
            display=display,
            abort=abort,
        )

    def _replay_agent_tool_runs(self, connection: Connection) -> None:
        self._agent_tool_runs.replay(connection)

    async def _dispatch_connect(
        self,
        connection: Connection,
        ctx: ConnectionContext,
    ) -> None:
        if self.should_connection_be_readonly(connection, ctx):
            self.set_connection_readonly(connection)

        if self.should_send_protocol_messages(connection, ctx):
            agent = camel_to_kebab(self.__class__.__name__)
            connection.send_json(identity_frame(self.name, agent))
            already_resolving = connection in self.__handshake_state_connections
            self.__handshake_state_connections.add(connection)
            try:
                current_state = self.state
            finally:
                if not already_resolving:
                    self.__handshake_state_connections.discard(connection)
            connection.send_json(state_frame(current_state))
            connection.send_json(mcp_servers_frame(self.get_mcp_servers()))
        else:
            self._set_connection_no_protocol(connection)

        # After the identity/state handshake so a reconnecting client has rebuilt the
        # agent before its in-flight tool runs replay onto it. These frames buffer
        # before the 101 like the rest, so they cannot meet a closed socket.
        self._agent_tool_runs.replay(connection)
        await _call_maybe_async(self.on_connect, connection, ctx)

    async def _dispatch_message(self, connection: Connection, message: str) -> None:
        # not JSON, not an object, or a binary frame — hand it straight to the user
        data = loads_dict_or_none(message)
        if data is None:
            await _call_maybe_async(self.on_message, connection, message)
            return

        _type = data.get("type")

        # Presence, not type: a frame carrying an explicit null still counts as a state
        # update, so it earns an error frame instead of falling through to on_message.
        if _type == MessageType.CF_AGENT_STATE and "state" in data:
            self._handle_state_update(connection, data)
            return

        # Checked here rather than in _handle_rpc, so a wrongly shaped "rpc" frame
        # reaches the user's on_message instead of collecting an error it never asked
        # for, and _handle_rpc can rely on data["id"] being a string.
        if (
            _type == MessageType.RPC
            and isinstance(data.get("id"), str)
            and isinstance(data.get("method"), str)
            and isinstance(data.get("args"), list)
        ):
            await self._handle_rpc(connection, data)
            return

        await _call_maybe_async(self.on_message, connection, message)

    async def _forward_ws_relay(self, payload: dict[str, Any], *, gate: bool) -> str:
        url = payload.get("url")
        if not isinstance(url, str):
            raise RoutingException("sub-agent relay URL is missing")
        hop = _next_sub_hop(url_path(url), is_child=self._facet_name is not None)
        if hop is None:
            return await self._handle_ws_relay_leaf(payload)

        kebab, child_name, remaining = hop
        if _FACET_KEY_SEP in child_name:
            raise RoutingException("sub-agent name may not contain a null character")
        class_name = self._resolve_child_class(kebab)
        if class_name is None:
            raise RoutingException("sub-agent class is not exported")

        forwarded_url = url_with_path(url, remaining)
        headers = payload.get("headers")
        if not isinstance(headers, dict):
            headers = {}
        if gate and payload.get("event") == "connect":
            request = Request(url, headers=headers)
            decision = await self._dispatch_hook(
                "on_before_sub_agent",
                self.on_before_sub_agent,
                request,
                SubAgentRef(class_name=class_name, name=child_name),
                propagate=True,
            )
            if isinstance(decision, Response):
                return dumps_wire(
                    [
                        {
                            "type": "close",
                            "code": 1008,
                            "reason": "Sub-agent connection rejected",
                        }
                    ]
                )
            if isinstance(decision, Request):
                forwarded_url = url_with_path(decision.url, remaining)
                headers = dict(decision.headers)

        relay_id = payload.get("relayId")
        event = payload.get("event")
        context = (
            (relay_id, event)
            if isinstance(relay_id, str) and isinstance(event, str)
            else None
        )
        token = self.__relay_startup_context.set(context)
        try:
            child = await self._resolve_sub_agent(class_name, child_name)
        finally:
            self.__relay_startup_context.reset(token)
        forwarded = {**payload, "url": forwarded_url, "headers": headers}
        return await child._cf_ws_event(dumps_wire(forwarded))

    async def _cf_ws_event(self, payload_json: str) -> str:
        payload = loads_dict_or_none(payload_json)
        if payload is None:
            raise RoutingException("invalid sub-agent relay payload")
        relay_id = payload.get("relayId")
        event = payload.get("event")
        context = (
            (relay_id, event)
            if isinstance(relay_id, str) and isinstance(event, str)
            else None
        )
        token = self.__relay_startup_context.set(context)
        try:
            await self._ensure_initialized()
        finally:
            self.__relay_startup_context.reset(token)
        return await self._forward_ws_relay(payload, gate=True)

    async def _cf_invokeAgentPath(
        self,
        target_path: object,
        method: object,
        args: object,
    ) -> object:
        await self._ensure_initialized()
        target = _agent_path_from_value(_rpc_to_python(target_path))
        method = _rpc_to_python(method)
        args = _rpc_to_python(args)
        if not isinstance(method, str) or not isinstance(args, list):
            raise TypeError("Workflow path RPC requires a method and argument list")

        self_path = self.self_path
        if target[: len(self_path)] != self_path:
            raise ValueError(
                f"Workflow origin path does not descend from {self_path!r}"
            )
        if len(target) == len(self_path):
            member = static_mro_members(self).get(method)
            if method.startswith("__") or static_definition_function(member) is None:
                raise ValueError(
                    f'Workflow origin method "{method}" is not callable on '
                    f"{type(self).__name__}"
                )
            callback = deferred_method(member, self)
            return await _call_maybe_async(callback, *args)

        next_step = target[len(self_path)]
        class_name = next_step["className"]
        name = next_step["name"]
        if not self._has_sub_agent_row(class_name, name):
            raise ValueError(
                f'Workflow origin sub-agent {class_name} "{name}" no longer exists'
            )
        child = await self._resolve_sub_agent(class_name, name)
        return await _call_maybe_async(
            child._cf_invokeAgentPath,
            target,
            method,
            args,
        )

    async def _cf_broadcastAgentPath(
        self,
        target_path: object,
        message: object,
        exclude: object = (),
        protocol: object = False,
    ) -> None:
        await self._ensure_initialized()
        if self._facet_name is not None:
            raise ValueError("facet broadcasts must be routed through the root Agent")

        target = _agent_path_from_value(_rpc_to_python(target_path))
        message = _rpc_to_python(message)
        excluded = _rpc_to_python(exclude)
        protocol = _rpc_to_python(protocol)
        if not isinstance(message, str):
            raise TypeError("facet broadcast message must be a string")
        if not isinstance(excluded, (list, tuple)) or not all(
            isinstance(value, str) for value in excluded
        ):
            raise TypeError("facet broadcast exclusions must be physical relay keys")
        if not isinstance(protocol, bool):
            raise TypeError("facet protocol broadcast marker must be a boolean")

        self_path = self.self_path
        if target[: len(self_path)] != self_path or len(target) == len(self_path):
            raise ValueError(
                f"facet broadcast path does not descend from {self_path!r}"
            )

        def matches(relay: RelayTarget, connection: Connection) -> bool:
            return self._relay_target_path(relay) == target and (
                not protocol
                or (
                    connection not in self.__handshake_state_connections
                    and self.is_connection_protocol_enabled(connection)
                )
            )

        await self._websockets._broadcast_relays(message, matches, exclude=excluded)

    def _relay_target_path(self, target: RelayTarget) -> list[PathStep] | None:
        stored_path = target.get("path")
        stored = (
            [_path_step(step["className"], step["name"]) for step in stored_path]
            if stored_path is not None
            else None
        )
        path = self.self_path
        if stored is not None and stored[: len(path)] != path:
            return None
        remaining = url_path(target["url"])
        is_child = False
        while True:
            hop = _next_sub_hop(remaining, is_child=is_child)
            if hop is None:
                reconstructed = path if is_child else None
                if stored is None:
                    return reconstructed
                return stored if stored == reconstructed else None
            kebab, name, remaining = hop
            if stored is None:
                class_name = self._resolve_child_class(kebab)
                if class_name is None:
                    return None
            else:
                if len(path) >= len(stored):
                    return None
                step = stored[len(path)]
                if camel_to_kebab(step["className"]) != kebab or step["name"] != name:
                    return None
                class_name = step["className"]
            path.append(_path_step(class_name, name))
            is_child = True

    async def _handle_ws_relay_leaf(self, payload: dict[str, Any]) -> str:
        connection_id = payload.get("connectionId")
        relay_id = payload.get("relayId")
        event = payload.get("event")
        if (
            not isinstance(connection_id, str)
            or not isinstance(relay_id, str)
            or not isinstance(event, str)
        ):
            raise RoutingException("invalid sub-agent relay event")
        state = payload.get("state")
        tags = payload.get("tags")
        relay_tags = tags if isinstance(tags, list) else []
        connection = BufferedRelayConnection(
            connection_id,
            physical_key=relay_id,
            state=state,
            tags=[tag for tag in relay_tags if isinstance(tag, str)],
            max_frames=self.sub_agent_ws_max_frames,
            max_bytes=self.sub_agent_ws_max_bytes,
        )
        relay_connection = cast(Connection, connection)
        with RelaySession(self._connections, connection, key=relay_id):
            if event == "connect":
                url = payload.get("url")
                headers = payload.get("headers")
                request = Request(
                    url if isinstance(url, str) else "https://agents.invalid/",
                    headers=headers if isinstance(headers, dict) else {},
                )
                context = ConnectionContext(request)
                child_tags = await self._dispatch_hook(
                    "get_connection_tags",
                    self.get_connection_tags,
                    connection,
                    context,
                    connection=relay_connection,
                    propagate=True,
                )
                connection._insert_tags(_prepare_tags(connection.id, child_tags or ()))
                await self._dispatch_hook(
                    "on_connect",
                    self._dispatch_connect,
                    connection,
                    context,
                    connection=relay_connection,
                    propagate=True,
                )
            elif event == "message":
                message = payload.get("message")
                if not isinstance(message, str):
                    raise RoutingException("invalid sub-agent relay message")
                await self._dispatch_hook(
                    "on_message",
                    self._dispatch_message,
                    connection,
                    message,
                    connection=relay_connection,
                    propagate=False,
                )
            elif event == "close":
                await self._dispatch_hook(
                    "on_close",
                    self._dispatch_close,
                    connection,
                    int(payload.get("code") or 1000),
                    str(payload.get("reason") or ""),
                    bool(payload.get("wasClean")),
                    connection=relay_connection,
                    propagate=False,
                )
            elif event == "error":
                error = Exception(str(payload.get("error") or ""))
                await self._report_error(error, relay_connection)
            else:
                raise RoutingException("unknown sub-agent relay event")
            if self.__facet_startup_protocol_pending == (relay_id, event):
                self.__facet_startup_protocol_pending = None
                excluded = (relay_id,) if event == "connect" else ()
                self._retain_facet_protocol_snapshot(excluded)
            return connection.operation_log()

    async def _handle_rpc(self, connection: Connection, data: dict[str, Any]):
        # guaranteed well-formed by the check in _dispatch_message
        rpc_id: str = data["id"]
        method_name: str = data["method"]
        args: list[Any] = list(data["args"])

        try:
            method = self.__rpc_meths.get(method_name)
            if method is None:
                if method_name in self.__rpc_class_callables:
                    raise RPCError(f"Method {method_name} is not callable")

                raise RPCError(f"Method {method_name} does not exist")

            if method.streaming:
                stream = StreamingResponse(connection, rpc_id)
                try:
                    await _maybe_coro(method.func(stream, *args))
                except Exception as exc:  # noqa: BLE001
                    # A streaming call's terminal belongs to the stream, so the failure
                    # leaves through it — and only if the method had not already closed
                    # it.
                    if not stream.is_closed:
                        stream.error(error_message(exc))

                # the stream owns every frame this call produces, so falling through
                # would append a second terminal
                return

            result = await _maybe_coro(method.func(*args))

            connection.send_if_open(rpc_result(rpc_id, result))
        except Exception as exc:  # noqa: BLE001
            # Without this the client's call never settles: nothing reaches the wire and
            # its pending promise waits forever.
            connection.send_if_open(rpc_error(rpc_id, error_message(exc)))

    def _handle_state_update(
        self,
        connection: Connection,
        data: dict[str, Any],
    ) -> None:
        if self.is_connection_readonly(connection):
            connection.send_if_open(state_error_frame("Connection is readonly"))
            return
        try:
            state = data["state"]
            if not isinstance(state, dict):
                # A scalar here would make every later state read raise, including
                # the connection handshake, locking every future client out.
                raise TypeError(f"state must be an object, got {type(state).__name__}")

            self._set_state_internal(state, connection)
        except Exception:  # noqa: BLE001
            # the frame carries a fixed string, so nothing about why it failed leaks
            connection.send_if_open(state_error_frame("State update rejected"))

    def _set_state_internal(
        self,
        state: dict[str, Any],
        source: StateSourceT = "server",
    ) -> None:
        # Serialized first: a value the other runtime could not parse is rejected before
        # anything observes it, or the raise would leave that value live in memory and
        # absent from the row. Then assigned before the write, so a failing send cannot
        # leave the in-memory copy behind what SQLite already holds.
        serialized = dumps_wire(state)
        if not self.__state_storage_ready:
            prepare_core_state_schema(self.sql)
            self.__state_storage_ready = True
        self.__state = state
        self._write_state_cell(STATE_ROW_ID, serialized)

        # The client applies its own update optimistically, so only that physical
        # connection is excluded; another socket with the same public id still receives
        # the update.
        excluded = None if source == "server" else source
        self._broadcast_protocol(state_frame(state), exclude=excluded)

    def set_state(self, state: dict[str, Any]) -> None:
        connection = _current_websocket_connection(self._websockets)
        if (
            connection is not None
            and connection not in self.__handshake_state_connections
            and self.is_connection_readonly(cast(Connection, connection))
        ):
            raise RuntimeError("Connection is readonly")
        self._set_state_internal(state)

    def _broadcast_protocol(
        self,
        frame: dict[str, Any],
        *,
        exclude: Connection | None = None,
    ) -> None:
        relay_exclusions: list[str] = []
        for connection in self.get_connections():
            physical_key = getattr(connection, "_physical_key", None)
            if (
                connection is exclude
                or connection in self.__handshake_state_connections
                or not self.is_connection_protocol_enabled(connection)
            ):
                if connection is exclude and isinstance(physical_key, str):
                    relay_exclusions.append(physical_key)
                continue
            connection.send_if_open(frame)
            if isinstance(physical_key, str):
                relay_exclusions.append(physical_key)
        if self._facet_name is not None and self.__mcp_broadcast_ready:
            target = self.self_path
            payload = dumps_wire(frame)
            self._lifecycle._retain_work(
                lambda: self._root_agent_stub()._cf_broadcastAgentPath(
                    target,
                    payload,
                    relay_exclusions,
                    True,
                )
            )

    def _retain_facet_protocol_snapshot(self, excluded: Iterable[str]) -> None:
        target = self.self_path
        state_payload = dumps_wire(state_frame(self.state))
        mcp_payload = dumps_wire(mcp_servers_frame(self.get_mcp_servers()))
        relay_exclusions = list(excluded)

        async def publish() -> None:
            root = self._root_agent_stub()
            await root._cf_broadcastAgentPath(
                target,
                state_payload,
                relay_exclusions,
                True,
            )
            await root._cf_broadcastAgentPath(
                target,
                mcp_payload,
                relay_exclusions,
                True,
            )

        self._lifecycle._retain_work(publish)

    def _broadcast_mcp_servers(self) -> None:
        self._broadcast_protocol(mcp_servers_frame(self.get_mcp_servers()))

    def _publish_mcp_state_change(self) -> None:
        if self.__mcp_broadcast_ready:
            self._broadcast_mcp_servers()

    @property
    def state(self) -> dict[str, Any]:
        return self.__state.copy()

    # -- sub-agents ------------------------------------------------------
    #
    # A child Durable Object running on the parent's machine with its own SQLite. The
    # runtime builds it from a class handle read out of the worker's own exports, so a
    # child needs no binding and no migration — only to be exported.

    @property
    def parent_path(self) -> list[PathStep]:
        # Root-first, so the direct parent is last. Loaded on first access because a
        # child needs its own name before the socket map is built, but nothing needs the
        # ancestor chain until it spawns a grandchild or reaches back up.
        if self.__parent_path is None:
            self.__parent_path = self._read_parent_path()

        return list(self.__parent_path or ())

    @property
    def self_path(self) -> list[PathStep]:
        return [*self.parent_path, _path_step(type(self).__name__, self.name)]

    def _read_parent_path(self) -> list[PathStep] | None:
        # None means "could not read", which must not be cached as "no ancestors" or a
        # transient storage failure would convince a child it had no parent.
        try:
            raw = self._read_state_cell(PARENT_PATH_ROW_ID)
        except Exception:  # noqa: BLE001
            return None

        if raw is MISSING:
            return []

        return _parse_parent_path(raw)

    async def _cf_init_as_facet(
        self,
        name: str,
        parent_path_json: str,
        relay_id: str | None = None,
        relay_event: str | None = None,
    ) -> None:
        # Reached over RPC from the parent, so it runs inside this object's own isolate
        # and owns its storage writes. Deliberately not decorated: browser RPC must not
        # expose this internal bootstrap method.
        routed = self._facet_name
        if routed != name:
            raise RPCError(
                f"facet bootstrap mismatch: routing id decodes to {routed!r}, "
                f"parent passed {name!r}"
            )

        prepare_core_state_schema(self.sql)
        self._write_state_cell(PARENT_PATH_ROW_ID, parent_path_json)
        self.__parent_path = _parse_parent_path(parent_path_json)
        self._configure_lifecycle_routes()

        context = (
            (relay_id, relay_event)
            if isinstance(relay_id, str) and isinstance(relay_event, str)
            else None
        )
        token = self.__relay_startup_context.set(context)
        try:
            await self._ensure_initialized()
        finally:
            self.__relay_startup_context.reset(token)

    def _ensure_sub_agent_registry(self) -> None:
        # Lazy and outside the schema marker, so an agent that never spawns a child
        # never carries the table.
        if self.__sub_agents_ready:
            return

        self.sql("""
        CREATE TABLE IF NOT EXISTS cf_agents_sub_agents (
            class TEXT NOT NULL,
            name TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            PRIMARY KEY (class, name)
        )
        """)
        self.__sub_agents_ready = True

    def _root_namespace(self) -> Any:
        # Minted from the root of the tree, because a child class need not have a
        # binding and an intermediate parent is itself a child, so neither can hand out
        # ids.
        root = self.parent_path[0] if self.parent_path else None
        class_name = root["className"] if root else type(self).__name__

        namespace = self._facet_class_handle(class_name)
        if getattr(namespace, "idFromName", None) is None:
            # Only the root needs a binding; a class exported but not bound turns up
            # here without the namespace helpers.
            raise RoutingException(
                f"sub-agents need the root class {class_name!r} registered in "
                f"durable_objects.bindings; its export exposes no idFromName"
            )

        return namespace

    def _ctx_capability(self, attr: str) -> Any:
        value = getattr(self.ctx, attr, None)
        if value is None:
            raise RoutingException(
                f"sub-agents need ctx.{attr}, which this runtime does not expose "
                "— raise compatibility_date in wrangler.jsonc"
            )

        return value

    def _facet_class_handle(self, class_name: str) -> Any:
        exports = self._ctx_capability("exports")
        handle = getattr(exports, class_name, None)
        if handle is None:
            raise RoutingException(
                f"sub-agent class {class_name!r} is not exported from the worker "
                f"entry module"
            )

        return handle

    def _facets(self) -> Any:
        return self._ctx_capability("facets")

    def _validate_child(self, class_name: str, name: str) -> None:
        if _FACET_KEY_SEP in name:
            raise RoutingException("sub-agent name may not contain a null character")
        if not name:
            raise RoutingException("sub-agent name may not be empty")
        if camel_to_kebab(class_name) == SUB_PREFIX:
            raise RoutingException(
                f"sub-agent class {class_name!r} collides with the reserved "
                f"{SUB_PREFIX!r} URL separator"
            )

    async def sub_agent(self, cls: type, name: str) -> Any:
        """Get or create a named child agent and return an RPC stub for it.

        The first call for a name starts the child; later calls return the
        existing one. The stub is the runtime's own, so results come back as
        JavaScript values — call `.to_py()` on anything structured.
        """
        return await self._resolve_sub_agent(cls.__name__, name)

    async def _resolve_sub_agent(self, class_name: str, name: str) -> Any:
        # Takes the class name rather than the class, so a URL resolves through exactly
        # the same path as a direct spawn.
        self._validate_child(class_name, name)
        key = _facet_key(class_name, name)
        gate = self.__facet_operation_gates.setdefault(key, _FacetOperationGate())
        gate.users += 1
        try:
            async with gate.lock:
                if self._facet_name is None and self._has_pending_facet_deletion(
                    class_name, name
                ):
                    await self._complete_facet_deletion_unlocked(class_name, name)

                handle = self._facet_class_handle(class_name)
                facets = self._facets()
                parent_path = self.self_path
                identity = _facet_identity(
                    [*parent_path, _path_step(class_name, name)], name
                )
                facet_id = self._root_namespace().idFromName(identity)

                def start_options() -> Any:
                    return to_js(
                        {"class": handle, "id": facet_id},
                        dict_converter=Object.fromEntries,
                    )

                # The runtime retains this callback beyond facets.get, so the proxy is
                # cached until the corresponding facet is deleted.
                proxy = self.__facet_proxies.get(key)
                if proxy is None:
                    proxy = create_proxy(start_options)
                    self.__facet_proxies[key] = proxy

                stub = facets.get(key, proxy)
                # Recorded before bootstrap so a hard interruption cannot leave a child
                # with no row. A caught first-bootstrap failure rolls its row back.
                existed = self._has_sub_agent_row(class_name, name)
                self._record_sub_agent(class_name, name)
                try:
                    relay_context = self.__relay_startup_context.get()
                    args = (
                        (name, dumps_wire(parent_path))
                        if relay_context is None
                        else (
                            name,
                            dumps_wire(parent_path),
                            relay_context[0],
                            relay_context[1],
                        )
                    )
                    await stub._cf_init_as_facet(*args)
                except BaseException:
                    if not existed:
                        self._forget_sub_agent(class_name, name)
                    raise
                return stub
        finally:
            gate.users -= 1
            if gate.users == 0:
                self.__facet_operation_gates.pop(key, None)

    async def parent_agent(self, cls: type) -> Any:
        """Return an RPC stub for the agent that spawned this one."""
        path = self.parent_path
        if not path:
            raise RoutingException(
                f"{type(self).__name__} is not a sub-agent, so it has no parent"
            )

        parent = path[-1]
        if parent["className"] != cls.__name__:
            raise RoutingException(
                f"recorded parent class is {parent['className']!r}, "
                f"not {cls.__name__!r}"
            )
        if len(path) > 1:
            raise RoutingException(
                "parent_agent() only reaches a top-level parent; an intermediate "
                "sub-agent has no binding to address it by"
            )

        namespace = self._root_namespace()
        return namespace.get(namespace.idFromName(parent["name"]))

    def _record_sub_agent(self, class_name: str, name: str) -> None:
        self._ensure_sub_agent_registry()
        self.sql(
            "INSERT OR IGNORE INTO cf_agents_sub_agents (class, name, created_at) "
            "VALUES (?, ?, ?)",
            class_name,
            name,
            now_ms(),
        )

    def _forget_sub_agent(self, class_name: str, name: str) -> None:
        self._ensure_sub_agent_registry()
        self.sql(
            "DELETE FROM cf_agents_sub_agents WHERE class = ? AND name = ?",
            class_name,
            name,
        )

    def _facet_cleanup_table_exists(self) -> bool:
        return bool(
            self.sql(
                "SELECT 1 FROM sqlite_master "
                "WHERE type = 'table' AND name = 'cf_agents_facet_cleanup'"
            )
        )

    def _ensure_facet_cleanup_table(self) -> None:
        self.sql("""
        CREATE TABLE IF NOT EXISTS cf_agents_facet_cleanup (
            class TEXT NOT NULL,
            name TEXT NOT NULL,
            PRIMARY KEY (class, name)
        )
        """)

    def _record_facet_deletion(self, class_name: str, name: str) -> None:
        self._ensure_facet_cleanup_table()
        self.sql(
            "INSERT OR IGNORE INTO cf_agents_facet_cleanup (class, name) VALUES (?, ?)",
            class_name,
            name,
        )

    def _has_pending_facet_deletion(self, class_name: str, name: str) -> bool:
        if not self._facet_cleanup_table_exists():
            return False
        return bool(
            self.sql(
                "SELECT 1 FROM cf_agents_facet_cleanup WHERE class = ? AND name = ?",
                class_name,
                name,
            )
        )

    def _forget_facet_deletion(self, class_name: str, name: str) -> None:
        self.sql(
            "DELETE FROM cf_agents_facet_cleanup WHERE class = ? AND name = ?",
            class_name,
            name,
        )

    async def _retry_pending_facet_deletions(self) -> None:
        if self._facet_name is not None or not self._facet_cleanup_table_exists():
            return
        rows = self.sql("SELECT class, name FROM cf_agents_facet_cleanup")
        for row in rows:
            key = _facet_key(row["class"], row["name"])
            if key in self.__active_facet_deletions:
                continue
            try:
                await self._complete_facet_deletion(row["class"], row["name"])
            except asyncio.CancelledError:
                raise
            except BaseException as error:
                await self._report_lifecycle_error(error)

    def has_sub_agent(self, cls: type, name: str) -> bool:
        return self._has_sub_agent_row(cls.__name__, name)

    def _has_sub_agent_row(self, class_name: str, name: str) -> bool:
        self._ensure_sub_agent_registry()
        rows = self.sql(
            "SELECT 1 FROM cf_agents_sub_agents WHERE class = ? AND name = ?",
            class_name,
            name,
        )
        return bool(rows)

    def list_sub_agents(self, cls: type | None = None) -> list[SubAgentRecord]:
        """Children this agent has spawned, oldest first.

        Reflects what was recorded here, not what the runtime currently holds.
        """
        self._ensure_sub_agent_registry()
        if cls is None:
            rows = self.sql(
                "SELECT class, name, created_at FROM cf_agents_sub_agents "
                "ORDER BY created_at ASC"
            )
        else:
            rows = self.sql(
                "SELECT class, name, created_at FROM cf_agents_sub_agents "
                "WHERE class = ? ORDER BY created_at ASC",
                cls.__name__,
            )

        return [
            SubAgentRecord(
                class_name=row["class"],
                name=row["name"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def abort_sub_agent(self, cls: type, name: str, reason: str = "aborted") -> None:
        """Stop a child. Its storage survives and the next spawn restarts it."""
        self._facets().abort(_facet_key(cls.__name__, name), reason)

    async def _cleanup_facet_prefix(self, owner_path: list[PathStep]) -> None:
        if self._facet_name is not None:
            raise RuntimeError("facet cleanup must run on the root Agent")
        if len(owner_path) < 2 or owner_path[:1] != self.self_path[:1]:
            raise PermissionError("facet cleanup path is outside this Agent tree")
        address = _agent_route_address(owner_path)
        await self.scheduler.cleanup_route_prefix(address.key, address.data)
        await self.tasks.cleanup_route_prefix(address.key, address.data)
        facet_run_tables = self.sql(
            "SELECT 1 FROM sqlite_master "
            "WHERE type = 'table' AND name = 'cf_agents_facet_runs'"
        )
        if not facet_run_tables:
            return
        prefix = address.key
        rows = self.sql(
            "SELECT owner_path, owner_path_key, run_id FROM cf_agents_facet_runs"
        )
        for row in rows:
            owner_key = row["owner_path_key"]
            key_matches = isinstance(owner_key, str) and (
                owner_key == prefix or owner_key.startswith(f"{prefix}/")
            )
            try:
                row_path = _agent_path_from_value(
                    strict_json_loads(row["owner_path"], "facet owner path")
                )
            except (RecursionError, TypeError, ValueError):
                row_path = []
            if not key_matches and row_path[: len(owner_path)] != owner_path:
                continue
            self.sql(
                "DELETE FROM cf_agents_facet_runs "
                "WHERE owner_path_key = ? AND run_id = ?",
                owner_key,
                row["run_id"],
            )

    async def _complete_facet_deletion(
        self,
        class_name: str,
        name: str,
        *,
        record: bool = False,
    ) -> None:
        key = _facet_key(class_name, name)
        gate = self.__facet_operation_gates.setdefault(key, _FacetOperationGate())
        gate.users += 1
        try:
            async with gate.lock:
                if record:
                    self._record_facet_deletion(class_name, name)
                await self._complete_facet_deletion_unlocked(class_name, name)
        finally:
            gate.users -= 1
            if gate.users == 0:
                self.__facet_operation_gates.pop(key, None)

    async def _complete_facet_deletion_unlocked(
        self,
        class_name: str,
        name: str,
    ) -> None:
        key = _facet_key(class_name, name)
        self.__active_facet_deletions.add(key)
        try:
            self._facets().delete(key)
            try:
                self._forget_sub_agent(class_name, name)
                await self._cleanup_facet_prefix(
                    [*self.self_path, _path_step(class_name, name)]
                )
                self._forget_facet_deletion(class_name, name)
            finally:
                proxy = self.__facet_proxies.pop(key, None)
                if proxy is not None:
                    proxy.destroy()
        finally:
            self.__active_facet_deletions.discard(key)

    async def delete_sub_agent(self, cls: type, name: str) -> None:
        """Stop a child and destroy its storage. Irreversible, and transitive.

        Only a top-level agent can do this. The runtime refuses to destroy a
        facet's own sub-facets, and that refusal is deliberately not caught: the
        registry row would come off while the child's storage lived on, so the
        map would report a child that is still there. Use `abort_sub_agent` from
        a sub-agent, which works at any depth.

        A key the runtime does not hold is accepted, so repeat deletes and
        deletes of a child that was never spawned both settle quietly.
        """
        if self._facet_name is not None:
            raise RuntimeError("only a root may delete facet storage")
        self._validate_child(cls.__name__, name)
        await self._ensure_initialized()
        await self._complete_facet_deletion(cls.__name__, name, record=True)

    # -- sub-agent request routing ---------------------------------------

    async def on_before_sub_agent(
        self,
        request: Request,
        child: SubAgentRef,
    ) -> Request | Response | None:
        """Gate a request on its way to a child. Runs once per hop, in the parent.

        Return nothing to forward as-is, a `Request` to forward instead, or a
        `Response` to answer without waking the child. Permissive by default,
        so override this to keep a name from a URL spawning a child.

        The child always sees the path after its own `/sub/{class}/{name}`
        segment, so shape a forwarded request through its headers or body rather
        than its URL.
        """
        return None

    def _resolve_child_class(self, kebab: str) -> str | None:
        exports = getattr(self.ctx, "exports", None)
        if exports is None:
            return None

        for key in Object.keys(exports):
            if camel_to_kebab(key) == kebab:
                return key

        return None

    async def fetch(self, request: Request) -> Response:
        await self._ensure_initialized()

        path = url_path(request.url)
        hop = _next_sub_hop(path, is_child=self._facet_name is not None)
        if hop is None:
            if _is_ws_req(dict(request.headers)):
                return await self._lifecycle.websocket_upgrade(request)
            return await self._lifecycle.request(request)

        kebab, child_name, remaining = hop

        if _FACET_KEY_SEP in child_name:
            # A percent-encoded null would otherwise address a different child than the
            # path spells out.
            return Response("Bad Request", status=400)

        class_name = self._resolve_child_class(kebab)
        if class_name is None:
            return Response("Not Found", status=404)

        child = SubAgentRef(class_name=class_name, name=child_name)
        decision = await self._dispatch_hook(
            "on_before_sub_agent",
            self.on_before_sub_agent,
            request,
            child,
            propagate=True,
        )
        if isinstance(decision, Response):
            return decision

        forwarded = decision if isinstance(decision, Request) else request
        if _is_ws_req(dict(request.headers)):
            target = RelayTarget(
                url=url_with_path(forwarded.url, path),
                headers=dict(forwarded.headers),
            )
            target_path = self._relay_target_path(target)
            if target_path is not None:
                target["path"] = [
                    RelayPathStep(className=step["className"], name=step["name"])
                    for step in target_path
                ]
            self.__pending_relay_targets[id(forwarded)] = target
            try:
                return await self._lifecycle.websocket_upgrade(forwarded)
            finally:
                self.__pending_relay_targets.pop(id(forwarded), None)

        return await self._forward_to_sub_agent(
            forwarded, class_name, child_name, remaining
        )

    async def _forward_to_sub_agent(
        self,
        request: Request,
        class_name: str,
        child_name: str,
        remaining: str,
    ) -> Response:
        try:
            stub = await self._resolve_sub_agent(class_name, child_name)
        except RoutingException:
            # Flattened deliberately: the reason names worker exports, and a client
            # asking for a child that does not exist should not learn what else does.
            return Response("Not Found", status=404)

        # The child re-parses this to find its own next hop, so this marker has to be
        # gone and the rest left alone. Rebuilt because a request's URL is fixed once it
        # exists.
        rewritten = url_with_path(request.url, remaining)

        # Handed over as a stream: reading it here would pull the whole body into
        # memory, once per hop.
        forwarded = JsRequest.new(rewritten, request.js_object)
        return Response(await stub.fetch(forwarded))
