from __future__ import annotations

import asyncio
import inspect
import ipaddress
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, cast
from urllib.parse import parse_qs, parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

from workers import Response

from ...core.protocol import McpServers, mcp_servers_frame
from ...lifecycle import LifecycleCapability
from .oauth import DurableOAuthProvider
from .storage import (
    CURRENT_MCP_SCHEMA_VERSION,
    MCP_SCHEMA_VERSION_KEY,
    MCPKeyValueStorage,
    MCPSql,
    MCPServerStore,
    decode_server_options,
    encode_server_options,
    retry_options,
    with_session,
)
from .transports import HTTPTransportAdapter, OfficialMCPConnector
from .types import (
    MCPAbortSignal,
    MCPAIToolDescriptor,
    MCPAuthorizationRequired,
    MCPCatalog,
    MCPConnectionState,
    MCPElicitationHandlers,
    MCPError,
    MCPIsolateLostError,
    MCPOAuthCallbackPolicy,
    MCPOAuthCallbackResult,
    MCPOAuthProvider,
    MCPServerFilter,
    MCPServerRow,
    MCPStaleSessionError,
    MCPTransportContext,
    MCPTransportFactory,
    MCPTransportSession,
)

MCP_SERVER_ID_MAX_LENGTH = 64
_MAX_ERROR_LENGTH = 512
_BLOCKED_HOSTS = {"0.0.0.0", "::", "metadata.google.internal"}
_BLOCKED_HOST_SUFFIXES = (".internal", ".local", ".home.arpa")
_DEFAULT_CONNECT_TIMEOUT_MS = 30_000
_DEFAULT_DISCOVERY_TIMEOUT_MS = 15_000
_DEFAULT_RESTORE_TIMEOUT_MS = 60_000
_DEFAULT_RESTORE_CONCURRENCY = 4
_GOOD_CONNECTION_STATES = {
    MCPConnectionState.CONNECTED,
    MCPConnectionState.DISCOVERING,
    MCPConnectionState.READY,
}

type StateListener = Callable[[], object]
type Sleep = Callable[[float], Awaitable[object]]
type OAuthProviderFactory = Callable[
    [MCPKeyValueStorage, str, str, str, str | None], MCPOAuthProvider
]
type HostContextRunner = Callable[
    [Callable[[], object | Awaitable[object]]], Awaitable[object]
]


@dataclass(slots=True)
class MCPClientConnection:
    server: MCPServerRow
    options: dict[str, Any]
    oauth_provider: MCPOAuthProvider | None = None
    state: MCPConnectionState = MCPConnectionState.CONNECTING
    error: str | None = None
    catalog: MCPCatalog | None = None
    session: MCPTransportSession | None = None
    generation: int = 0
    step_up_attempts: int = 0


@dataclass(frozen=True, slots=True)
class _PendingConnection:
    generation: int
    task: asyncio.Task[None]


@dataclass(frozen=True, slots=True)
class _PendingDiscovery:
    generation: int
    task: asyncio.Task[bool]


class _StaleGeneration(Exception):
    pass


class _EventSignal:
    def __init__(self, event: asyncio.Event):
        self._event = event

    @property
    def aborted(self) -> bool:
        return self._event.is_set()

    async def wait(self) -> None:
        await self._event.wait()


class _CombinedSignal:
    def __init__(self, *signals: MCPAbortSignal):
        self._signals = signals

    @property
    def aborted(self) -> bool:
        return any(signal.aborted for signal in self._signals)

    async def wait(self) -> None:
        tasks = [asyncio.create_task(signal.wait()) for signal in self._signals]
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


class MCPClientManager(LifecycleCapability):
    """Durable, provider-neutral MCP client connection manager."""

    capability_id = "mcp"

    def __init__(
        self,
        name: str,
        version: str,
        *,
        storage: MCPKeyValueStorage | None = None,
        sql: MCPSql | None = None,
        transports: Mapping[str, MCPTransportFactory] | None = None,
        create_oauth_provider: OAuthProviderFactory | None = None,
        run_in_host_context: HostContextRunner | None = None,
        sleep: Sleep = asyncio.sleep,
        connect_timeout_ms: int = _DEFAULT_CONNECT_TIMEOUT_MS,
        discovery_timeout_ms: int = _DEFAULT_DISCOVERY_TIMEOUT_MS,
        restore_timeout_ms: int = _DEFAULT_RESTORE_TIMEOUT_MS,
        restore_concurrency: int = _DEFAULT_RESTORE_CONCURRENCY,
    ):
        self._name = name
        self._version = version
        self._storage_override = storage
        self._sql_override = sql
        self._store: MCPServerStore | None = None
        http = HTTPTransportAdapter(OfficialMCPConnector())
        self._transports: dict[str, MCPTransportFactory] = {
            "auto": http,
        }
        self._transports.update(transports or {})
        self._create_oauth_provider = create_oauth_provider
        self._host_context_runner = run_in_host_context
        self._sleep = sleep
        self.mcp_connections: dict[str, MCPClientConnection] = {}
        self._pending_connections: dict[str, _PendingConnection] = {}
        self._pending_discoveries: dict[str, _PendingDiscovery] = {}
        self._generations: dict[str, int] = {}
        self._listeners: set[StateListener] = set()
        self._elicitation_handlers: MCPElicitationHandlers | None = None
        self._oauth_callback_policy: MCPOAuthCallbackPolicy | None = None
        self._disposed = asyncio.Event()
        self._started = False
        self._connect_timeout_ms = _bounded_int(
            connect_timeout_ms, _DEFAULT_CONNECT_TIMEOUT_MS, 1, 300_000
        )
        self._discovery_timeout_ms = _bounded_int(
            discovery_timeout_ms, _DEFAULT_DISCOVERY_TIMEOUT_MS, 1, 300_000
        )
        self._restore_timeout_ms = _bounded_int(
            restore_timeout_ms, _DEFAULT_RESTORE_TIMEOUT_MS, 1, 600_000
        )
        self._restore_concurrency = _bounded_int(
            restore_concurrency, _DEFAULT_RESTORE_CONCURRENCY, 1, 32
        )

    async def on_start(self) -> None:
        storage, sql = self._resources()
        self._store = MCPServerStore(sql)
        raw_version = await storage.get(MCP_SCHEMA_VERSION_KEY)
        schema_version = (
            raw_version if type(raw_version) is int and raw_version >= 0 else 0
        )
        if schema_version < CURRENT_MCP_SCHEMA_VERSION:
            self._store.prepare()
            await storage.put(MCP_SCHEMA_VERSION_KEY, CURRENT_MCP_SCHEMA_VERSION)
        self._started = True
        await self.restore_connections()
        self._fire_state_changed()

    async def on_dispose(self) -> None:
        await self.dispose()

    async def on_request(self, context: object) -> Response | None:
        request = getattr(context, "request", None)
        if request is None or not self.is_callback_request(request):
            return None
        result = await self.handle_callback_request(request)
        return cast(Response, self.oauth_callback_response(result, str(request.url)))

    def add_state_listener(self, listener: StateListener) -> Callable[[], None]:
        """Subscribe Agent integration to catalog and state changes."""
        self._listeners.add(listener)

        def unsubscribe() -> None:
            self._listeners.discard(listener)

        return unsubscribe

    async def register_server(
        self,
        server_id: str,
        *,
        url: str,
        name: str,
        callback_url: str = "",
        client_id: str | None = None,
        auth_url: str | None = None,
        client: Mapping[str, Any] | None = None,
        transport: Mapping[str, Any] | None = None,
        retry: Mapping[str, Any] | None = None,
        binding_name: str | None = None,
        props: Mapping[str, Any] | None = None,
    ) -> str:
        self._require_started()
        server_id = normalize_server_id(server_id)
        _validate_server_url(url)
        if not name:
            raise ValueError("MCP server name must not be empty")
        generation = self._advance_generation(server_id)
        await self._cancel_pending(server_id)
        await self._cancel_discovery(server_id)
        existing = self.mcp_connections.pop(server_id, None)
        if existing is not None:
            await self._close_connection(existing, persist=False)
        options: dict[str, Any] = {
            "client": dict(client or {}),
            "transport": dict(transport or {}),
        }
        if retry is not None:
            options["retry"] = dict(retry)
        if binding_name is not None:
            options["bindingName"] = binding_name
        if props is not None:
            options["props"] = dict(props)
        capabilities = self._advertised_handler_capabilities()
        if capabilities is not None:
            options["capabilities"] = capabilities
        row = MCPServerRow(
            id=server_id,
            name=name,
            server_url=url,
            callback_url=callback_url,
            client_id=client_id,
            auth_url=auth_url,
            server_options=encode_server_options(options),
        )
        self._store_required().save(row)
        self.mcp_connections[server_id] = self._make_connection(row, generation)
        if auth_url:
            self.mcp_connections[server_id].state = MCPConnectionState.AUTHENTICATING
        self._fire_state_changed()
        return server_id

    async def register_rpc_server(
        self,
        server_id: str,
        *,
        name: str,
        rpc_name: str,
        binding_name: str,
        props: Mapping[str, Any] | None = None,
        retry: Mapping[str, Any] | None = None,
    ) -> str:
        return await self.register_server(
            server_id,
            url=f"rpc:{rpc_name}",
            name=name,
            transport={"type": "rpc"},
            retry=retry,
            binding_name=binding_name,
            props=props,
        )

    async def remove_server(self, server_id: str) -> None:
        self._require_started()
        self._advance_generation(server_id)
        await self._cancel_pending(server_id)
        await self._cancel_discovery(server_id)
        connection = self.mcp_connections.pop(server_id, None)
        if connection is not None:
            await self._close_connection(connection, persist=False)
        self._store_required().remove(server_id)
        self._fire_state_changed()

    async def migrate_server_id(
        self, old_id: str, new_id: str, client_name: str | None = None
    ) -> None:
        self._require_started()
        new_id = normalize_server_id(new_id)
        if new_id != old_id and (
            self._store_required().get(new_id) is not None
            or new_id in self.mcp_connections
        ):
            raise ValueError(f'MCP server id "{new_id}" is already in use')
        pending = self._pending_connections.get(old_id)
        if pending is not None:
            await asyncio.gather(asyncio.shield(pending.task), return_exceptions=True)
        await self._cancel_discovery(old_id)
        generation = self._advance_generation(new_id)
        self._advance_generation(old_id)
        migrated = self._store_required().migrate_id(old_id, new_id)
        storage, _ = self._resources()
        owner = client_name or self._name
        old_prefix = f"/{owner}/{old_id}/"
        new_prefix = f"/{owner}/{new_id}/"
        values = await storage.list(prefix=old_prefix)
        if values:
            await storage.put(
                {
                    new_prefix + key.removeprefix(old_prefix): value
                    for key, value in values.items()
                }
            )
            await storage.delete(tuple(values))
        connection = self.mcp_connections.pop(old_id, None)
        if connection is not None:
            connection.server = replace(connection.server, id=new_id)
            connection.generation = generation
            if connection.oauth_provider is not None:
                connection.oauth_provider.server_id = new_id
            self.mcp_connections[new_id] = connection
        if migrated or connection is not None:
            self._fire_state_changed()

    async def restore_connections(self) -> None:
        self._require_started()
        rows = self._store_required().list()
        semaphore = asyncio.Semaphore(self._restore_concurrency)

        async def restore(row: MCPServerRow) -> None:
            if row.id in self.mcp_connections:
                return
            generation = self._advance_generation(row.id)
            try:
                _validate_server_url(row.server_url)
                connection = self._make_connection(row, generation)
            except Exception as error:
                connection = MCPClientConnection(
                    row,
                    {},
                    state=MCPConnectionState.FAILED,
                    error=sanitize_error(error),
                    generation=generation,
                )
                self.mcp_connections[row.id] = connection
                return
            self.mcp_connections[row.id] = connection
            if row.auth_url:
                connection.state = MCPConnectionState.AUTHENTICATING
                return
            async with semaphore:
                try:
                    await self.establish_connection(row.id)
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    if self._is_current(connection, generation):
                        connection.state = MCPConnectionState.FAILED
                        connection.error = sanitize_error(error)

        tasks = [asyncio.create_task(restore(row)) for row in rows]
        if not tasks:
            return
        for row in rows:
            self._generations.setdefault(row.id, 0)
        try:
            async with asyncio.timeout(self._restore_timeout_ms / 1_000):
                await asyncio.gather(*tasks)
        except TimeoutError:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            for row in rows:
                pending = self._pending_connections.get(row.id)
                if pending is not None:
                    pending.task.cancel()
                connection = self.mcp_connections.get(row.id)
                if connection is not None and connection.state not in (
                    MCPConnectionState.READY,
                    MCPConnectionState.AUTHENTICATING,
                    MCPConnectionState.FAILED,
                ):
                    connection.state = MCPConnectionState.FAILED
                    connection.error = "MCP restoration deadline exceeded"
            pending_tasks = [
                pending.task
                for row in rows
                if (pending := self._pending_connections.pop(row.id, None)) is not None
            ]
            if pending_tasks:
                await asyncio.gather(*pending_tasks, return_exceptions=True)
            discovery_tasks = [
                pending.task
                for row in rows
                if (pending := self._pending_discoveries.pop(row.id, None)) is not None
            ]
            for task in discovery_tasks:
                task.cancel()
            if discovery_tasks:
                await asyncio.gather(*discovery_tasks, return_exceptions=True)

    async def establish_connection(
        self,
        server_id: str,
        *,
        authorization_params: Mapping[str, str] | None = None,
    ) -> None:
        self._require_started()
        connection = self._connection(server_id)
        generation = connection.generation
        existing = self._pending_connections.get(server_id)
        if existing is not None and existing.generation == generation:
            await asyncio.shield(existing.task)
            return
        task = asyncio.create_task(
            self._establish_connection(connection, generation, authorization_params)
        )
        pending = _PendingConnection(generation, task)
        self._pending_connections[server_id] = pending
        try:
            await asyncio.shield(task)
        finally:
            if self._pending_connections.get(server_id) is pending and task.done():
                self._pending_connections.pop(server_id, None)

    async def connect_to_server(self, server_id: str) -> dict[str, Any]:
        """Connect a registration and return its resulting state."""
        await self.establish_connection(server_id)
        connection = self._connection(server_id)
        result: dict[str, Any] = {"state": connection.state}
        if connection.state == MCPConnectionState.AUTHENTICATING:
            result["authUrl"] = connection.server.auth_url
            if connection.server.client_id is not None:
                result["clientId"] = connection.server.client_id
        elif connection.state == MCPConnectionState.FAILED:
            result["error"] = connection.error or "Unknown connection error"
        return result

    async def discover_if_connected(self, server_id: str) -> dict[str, Any] | None:
        """Refresh one connected catalog without reconnecting it."""
        connection = self.mcp_connections.get(server_id)
        if connection is None:
            return None
        success = await self.refresh_catalog(server_id)
        result: dict[str, Any] = {"success": success, "state": connection.state}
        if not success:
            result["error"] = connection.error or "MCP server is not connected"
        return result

    async def _establish_connection(
        self,
        connection: MCPClientConnection,
        generation: int,
        authorization_params: Mapping[str, str] | None,
    ) -> None:
        server_id = connection.server.id
        if connection.state in (
            MCPConnectionState.READY,
            MCPConnectionState.DISCOVERING,
        ):
            return
        retry = retry_options(connection.options)
        last_error: Exception | None = None
        max_attempts = 1 if authorization_params is not None else retry.max_attempts
        attempt = 0
        stale_recovered = False
        while attempt < max_attempts:
            try:
                await self._connect_once(connection, generation, authorization_params)
                if connection.state == MCPConnectionState.CONNECTED:
                    await self.refresh_catalog(server_id, generation=generation)
                return
            except MCPAuthorizationRequired as required:
                self._ensure_current(connection, generation)
                try:
                    await self._apply_authorization_required(connection, required)
                except MCPError as error:
                    last_error = error
                    break
                return
            except MCPStaleSessionError as error:
                if stale_recovered:
                    last_error = error
                    break
                stale_recovered = True
                connection.options = with_session(connection.options, None, None)
                connection.server = replace(
                    connection.server,
                    server_options=encode_server_options(connection.options),
                )
                self._store_required().save(connection.server)
                continue
            except _StaleGeneration:
                return
            except asyncio.CancelledError:
                raise
            except Exception as error:
                last_error = error
                attempt += 1
                if attempt < max_attempts:
                    delay_ms = min(
                        retry.max_delay_ms,
                        retry.base_delay_ms * (2 ** (attempt - 1)),
                    )
                    await self._sleep(delay_ms / 1_000)
        if self._is_current(connection, generation):
            connection.state = MCPConnectionState.FAILED
            connection.error = sanitize_error(last_error or "MCP connection failed")
            self._fire_state_changed()

    async def _connect_once(
        self,
        connection: MCPClientConnection,
        generation: int,
        authorization_params: Mapping[str, str] | None,
    ) -> None:
        self._ensure_current(connection, generation)
        if connection.session is not None:
            await connection.session.close()
            connection.session = None
        connection.state = MCPConnectionState.CONNECTING
        connection.error = None
        self._fire_state_changed()
        transport_options = dict(
            cast(Mapping[str, Any], connection.options.get("transport") or {})
        )
        for key in ("bindingName", "props"):
            if key in connection.options:
                transport_options[key] = connection.options[key]
        factory = self._transport_factory(connection.server, transport_options)
        client_options = dict(
            cast(Mapping[str, Any], connection.options.get("client") or {})
        )
        capability_seed = connection.options.get("capabilities")
        if isinstance(capability_seed, Mapping):
            configured = client_options.get("capabilities")
            client_options["capabilities"] = {
                **dict(capability_seed),
                **(dict(configured) if isinstance(configured, Mapping) else {}),
            }

        async def catalog_changed() -> object:
            if not self._is_current(connection, generation):
                return False
            return await self.refresh_catalog(
                connection.server.id, generation=generation
            )

        context = MCPTransportContext(
            server=connection.server,
            client_name=self._name,
            client_version=self._version,
            client_options=client_options,
            transport_options=transport_options,
            oauth_provider=connection.oauth_provider,
            elicit=lambda request, signal: self._handle_elicitation(
                connection.server.id, request, signal
            ),
            authorization_params=authorization_params,
            catalog_changed=catalog_changed,
            discover_result=cast(
                Mapping[str, Any] | None, connection.options.get("discoverResult")
            ),
        )
        async with asyncio.timeout(self._connect_timeout_ms / 1_000):
            session = await factory.open(context)
        if not self._is_current(connection, generation):
            await session.close()
            raise _StaleGeneration
        connection.session = session
        connection.state = MCPConnectionState.CONNECTED
        connection.server = replace(connection.server, auth_url=None)
        self._store_required().save(connection.server)
        self._persist_session(connection)
        self._fire_state_changed()

    async def refresh_catalog(
        self,
        server_id: str,
        *,
        generation: int | None = None,
        recover_stale: bool = True,
    ) -> bool:
        connection = self._connection(server_id)
        generation = connection.generation if generation is None else generation
        self._ensure_current(connection, generation)
        existing = self._pending_discoveries.get(server_id)
        if existing is not None and existing.generation == generation:
            return await asyncio.shield(existing.task)
        task = asyncio.create_task(
            self._refresh_catalog(connection, generation, recover_stale)
        )
        pending = _PendingDiscovery(generation, task)
        self._pending_discoveries[server_id] = pending
        try:
            return await asyncio.shield(task)
        finally:
            if self._pending_discoveries.get(server_id) is pending and task.done():
                self._pending_discoveries.pop(server_id, None)

    async def _refresh_catalog(
        self,
        connection: MCPClientConnection,
        generation: int,
        recover_stale: bool,
    ) -> bool:
        server_id = connection.server.id
        if (
            connection.session is None
            or connection.state not in _GOOD_CONNECTION_STATES
        ):
            return False
        connection.state = MCPConnectionState.DISCOVERING
        self._fire_state_changed()
        try:
            async with asyncio.timeout(self._discovery_timeout_ms / 1_000):
                catalog = await connection.session.discover()
            self._ensure_current(connection, generation)
            connection.catalog = catalog
        except MCPStaleSessionError:
            if not recover_stale or not self._is_current(connection, generation):
                raise
            await self._close_connection(connection, persist=True)
            await self._connect_once(connection, generation, None)
            return await self._refresh_catalog(connection, generation, False)
        except MCPAuthorizationRequired as required:
            self._ensure_current(connection, generation)
            try:
                await self._apply_authorization_required(connection, required)
            except MCPError as error:
                connection.state = MCPConnectionState.CONNECTED
                connection.error = sanitize_error(error)
                self._fire_state_changed()
            return False
        except _StaleGeneration:
            return False
        except asyncio.CancelledError:
            raise
        except Exception as error:
            if not self._is_current(connection, generation):
                return False
            connection.state = MCPConnectionState.CONNECTED
            connection.error = sanitize_error(error)
            self._fire_state_changed()
            return False
        connection.state = MCPConnectionState.READY
        connection.error = None
        connection.step_up_attempts = 0
        self._persist_session(connection)
        self._fire_state_changed()
        return True

    async def wait_for_connections(self, timeout_ms: int | None = None) -> None:
        tasks = tuple(pending.task for pending in self._pending_connections.values())
        if not tasks or timeout_ms is not None and timeout_ms <= 0:
            return
        settled = asyncio.gather(*tasks, return_exceptions=True)
        if timeout_ms is None:
            await settled
            return
        try:
            await asyncio.wait_for(asyncio.shield(settled), timeout_ms / 1_000)
        except TimeoutError:
            pass

    def list_servers(self) -> tuple[MCPServerRow, ...]:
        self._require_started()
        return self._store_required().list()

    def get_connection(self, server_id: str) -> MCPClientConnection | None:
        """Inspect one live connection without mutating it."""
        return self.mcp_connections.get(server_id)

    def list_tools(self, filter: MCPServerFilter | None = None) -> list[dict[str, Any]]:
        return self._namespaced("tools", filter)

    def list_prompts(
        self, filter: MCPServerFilter | None = None
    ) -> list[dict[str, Any]]:
        return self._namespaced("prompts", filter)

    def list_resources(
        self, filter: MCPServerFilter | None = None
    ) -> list[dict[str, Any]]:
        return self._namespaced("resources", filter)

    def list_resource_templates(
        self, filter: MCPServerFilter | None = None
    ) -> list[dict[str, Any]]:
        return self._namespaced("resource_templates", filter)

    def get_mcp_servers(self) -> McpServers:
        """Project the exact Agent handshake and catalog-update frame body."""
        self._require_started()
        servers: dict[str, Any] = {}
        for row in self._store_required().list():
            connection = self.mcp_connections.get(row.id)
            default_state = "authenticating" if row.auth_url else "not-connected"
            catalog = connection.catalog if connection is not None else None
            servers[row.id] = {
                "name": row.name,
                "server_url": row.server_url,
                "auth_url": row.auth_url,
                "state": connection.state if connection is not None else default_state,
                "error": sanitize_error(connection.error) if connection else None,
                "instructions": catalog.instructions if catalog else None,
                "capabilities": catalog.capabilities if catalog else None,
            }
        return McpServers(
            servers=servers,
            tools=self.list_tools(),
            prompts=self.list_prompts(),
            resources=self.list_resources(),
        )

    def get_catalog_frame(self) -> dict[str, Any]:
        return mcp_servers_frame(self.get_mcp_servers())

    def get_ai_tools(
        self, filter: MCPServerFilter | None = None
    ) -> dict[str, MCPAIToolDescriptor]:
        result: dict[str, MCPAIToolDescriptor] = {}
        for connection in self._filtered_connections(filter).values():
            catalog = connection.catalog
            if catalog is None:
                continue
            server_id = connection.server.id
            for tool in catalog.tools:
                name = tool.get("name")
                if not isinstance(name, str):
                    continue
                input_schema = tool.get("inputSchema", {"type": "object"})
                output_schema = tool.get("outputSchema")
                if not isinstance(input_schema, Mapping):
                    continue
                if output_schema is not None and not isinstance(output_schema, Mapping):
                    continue
                annotations = tool.get("annotations")
                title = tool.get("title")
                if not isinstance(title, str) and isinstance(annotations, Mapping):
                    candidate = annotations.get("title")
                    title = candidate if isinstance(candidate, str) else None
                description = tool.get("description")

                async def invoke(
                    arguments: Mapping[str, Any],
                    *,
                    _server_id: str = server_id,
                    _name: str = name,
                ) -> Mapping[str, Any]:
                    response = await self.call_tool(_server_id, _name, arguments)
                    if response.get("isError") is True:
                        raise MCPError(_tool_error(response))
                    return response

                key = f"tool_{server_id.replace('-', '')}_{name}"
                result[key] = MCPAIToolDescriptor(
                    input_schema=dict(input_schema),
                    output_schema=(
                        dict(output_schema)
                        if isinstance(output_schema, Mapping)
                        else None
                    ),
                    description=(description if isinstance(description, str) else None),
                    title=title if isinstance(title, str) else None,
                    execute=invoke,
                )
        return result

    async def call_tool(
        self,
        server_id: str,
        name: str,
        arguments: Mapping[str, Any],
        *,
        signal: MCPAbortSignal | None = None,
    ) -> Mapping[str, Any]:
        session = self._ready_session(server_id)
        unqualified = name.removeprefix(f"{server_id}.")
        try:
            return await session.call_tool(unqualified, arguments, signal=signal)
        except MCPAuthorizationRequired as required:
            await self._apply_authorization_required(
                self._connection(server_id), required
            )
            raise

    async def read_resource(
        self,
        server_id: str,
        params: Mapping[str, Any],
        *,
        signal: MCPAbortSignal | None = None,
    ) -> Mapping[str, Any]:
        try:
            return await self._ready_session(server_id).read_resource(
                params, signal=signal
            )
        except MCPAuthorizationRequired as required:
            await self._apply_authorization_required(
                self._connection(server_id), required
            )
            raise

    async def get_prompt(
        self,
        server_id: str,
        params: Mapping[str, Any],
        *,
        signal: MCPAbortSignal | None = None,
    ) -> Mapping[str, Any]:
        try:
            return await self._ready_session(server_id).get_prompt(
                params, signal=signal
            )
        except MCPAuthorizationRequired as required:
            await self._apply_authorization_required(
                self._connection(server_id), required
            )
            raise

    def configure_elicitation_handlers(
        self, handlers: MCPElicitationHandlers | None
    ) -> None:
        self._elicitation_handlers = (
            handlers if handlers and (handlers.form or handlers.url) else None
        )
        if self._started:
            capabilities = self._advertised_handler_capabilities()
            for row in self._store_required().list():
                options = decode_server_options(row.server_options)
                if capabilities is None:
                    options.pop("capabilities", None)
                else:
                    options["capabilities"] = capabilities
                self._store_required().save(
                    replace(row, server_options=encode_server_options(options))
                )

    def configure_oauth_callback(self, policy: MCPOAuthCallbackPolicy) -> None:
        self._oauth_callback_policy = policy

    def is_callback_request(self, request: object) -> bool:
        if getattr(request, "method", None) != "GET" or not self._started:
            return False
        request_url = urlsplit(str(getattr(request, "url", "")))
        query = _callback_params(request_url.query)
        if query is None:
            return False
        state = query.get("state")
        server_id = _server_id_from_state(state)
        if server_id is None:
            return False
        row = self._store_required().get(server_id)
        if row is None or not row.callback_url:
            return False
        callback = urlsplit(row.callback_url)
        return (
            callback.scheme,
            callback.netloc,
            callback.path,
        ) == (request_url.scheme, request_url.netloc, request_url.path)

    async def handle_callback_request(self, request: object) -> MCPOAuthCallbackResult:
        url = urlsplit(str(getattr(request, "url", "")))
        if getattr(request, "method", None) != "GET" or not self.is_callback_request(
            request
        ):
            return MCPOAuthCallbackResult(
                False, auth_error="Invalid OAuth callback URL"
            )
        params = _callback_params(url.query)
        if params is None:
            return MCPOAuthCallbackResult(
                False, auth_error="OAuth callback contains duplicate parameters"
            )
        state = params.get("state")
        server_id = _server_id_from_state(state)
        if state is None or server_id is None:
            return MCPOAuthCallbackResult(
                False, auth_error="Unauthorized: no state provided"
            )
        connection = self.mcp_connections.get(server_id)
        if connection is None:
            return MCPOAuthCallbackResult(
                False, server_id, f'No connection found for serverId "{server_id}".'
            )
        provider = connection.oauth_provider
        if provider is None:
            return self._fail_callback(
                connection, "MCP connection has no OAuth provider"
            )
        valid, state_error = await provider.check_state(state)
        if not valid:
            if connection.state in _GOOD_CONNECTION_STATES:
                return MCPOAuthCallbackResult(True, server_id)
            return MCPOAuthCallbackResult(
                False, server_id, sanitize_error(state_error or "Invalid state")
            )
        if connection.state in _GOOD_CONNECTION_STATES:
            await provider.consume_state(state)
            await _delete_code_verifier(provider, state)
            return MCPOAuthCallbackResult(True, server_id)
        external_error = params.get("error_description") or params.get("error")
        if external_error:
            await provider.consume_state(state)
            await _delete_code_verifier(provider, state)
            return self._fail_callback(connection, external_error)
        if not params.get("code"):
            await provider.consume_state(state)
            await _delete_code_verifier(provider, state)
            return self._fail_callback(connection, "Unauthorized: no code provided")
        if params.get("error"):
            await provider.consume_state(state)
            await _delete_code_verifier(provider, state)
            return self._fail_callback(connection, "OAuth callback is ambiguous")
        try:
            await self.establish_connection(server_id, authorization_params=params)
        finally:
            await provider.consume_state(state)
            await _delete_code_verifier(provider, state)
        if connection.state not in _GOOD_CONNECTION_STATES:
            if connection.state == MCPConnectionState.AUTHENTICATING:
                return MCPOAuthCallbackResult(
                    False, server_id, "Additional OAuth authorization is required"
                )
            return self._fail_callback(
                connection, connection.error or "OAuth connection failed"
            )
        return MCPOAuthCallbackResult(True, server_id)

    def oauth_callback_response(
        self, result: MCPOAuthCallbackResult, request_url: str
    ) -> object:
        policy = self._oauth_callback_policy
        if policy is not None and policy.custom_handler is not None:
            return policy.custom_handler(result)
        parsed = urlsplit(request_url)
        base_origin = urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))
        redirect = None
        if policy is not None:
            redirect = (
                policy.success_redirect
                if result.auth_success
                else policy.error_redirect
            )
        target = urljoin(base_origin, redirect) if redirect else base_origin
        if not result.auth_success and result.auth_error:
            target_url = urlsplit(target)
            query = parse_qs(target_url.query)
            query["error"] = [sanitize_error(result.auth_error) or "OAuth failed"]
            target = urlunsplit(
                (
                    target_url.scheme,
                    target_url.netloc,
                    target_url.path,
                    urlencode(query, doseq=True),
                    target_url.fragment,
                )
            )
        return Response(None, status=302, headers={"Location": target})

    async def dispose(self) -> None:
        if self._disposed.is_set():
            return
        self._disposed.set()
        pending = tuple(item.task for item in self._pending_connections.values())
        self._pending_connections.clear()
        discoveries = tuple(item.task for item in self._pending_discoveries.values())
        self._pending_discoveries.clear()
        for task in (*pending, *discoveries):
            task.cancel()
        if pending or discoveries:
            await asyncio.gather(*pending, *discoveries, return_exceptions=True)
        connections = tuple(self.mcp_connections.values())
        self.mcp_connections.clear()
        results = await asyncio.gather(
            *(self._close_connection(connection) for connection in connections),
            return_exceptions=True,
        )
        self._listeners.clear()
        failures = [result for result in results if isinstance(result, Exception)]
        if failures:
            raise ExceptionGroup("Failed to close MCP connections", failures)

    async def _handle_elicitation(
        self,
        server_id: str,
        request: Mapping[str, Any],
        signal: MCPAbortSignal | None,
    ) -> Mapping[str, Any]:
        params = request.get("params")
        mode = params.get("mode") if isinstance(params, Mapping) else None
        selected = "url" if mode == "url" else "form"
        handlers = self._elicitation_handlers
        handler = getattr(handlers, selected, None) if handlers else None
        if handler is None:
            raise MCPError(f"No MCP {selected}-mode elicitation handler configured")
        dispose_signal = _EventSignal(self._disposed)
        combined = _CombinedSignal(
            *(item for item in (signal, dispose_signal) if item is not None)
        )

        async def invoke() -> Mapping[str, Any]:
            def callback() -> object | Awaitable[object]:
                return handler(request, server_id, combined)

            if self._host_context_runner is not None:
                value = await self._host_context_runner(callback)
            else:
                try:
                    lifecycle = self.lifecycle
                except RuntimeError:
                    value = callback()
                    if inspect.isawaitable(value):
                        value = await value
                else:
                    value = await lifecycle.run_in_host_context(callback)
            if not isinstance(value, Mapping):
                raise TypeError("MCP elicitation handler must return an object")
            return dict(value)

        handler_task = asyncio.create_task(invoke())
        cancelled_task = asyncio.create_task(combined.wait())
        done, _ = await asyncio.wait(
            (handler_task, cancelled_task), return_when=asyncio.FIRST_COMPLETED
        )
        if handler_task in done:
            cancelled_task.cancel()
            await asyncio.gather(cancelled_task, return_exceptions=True)
            return handler_task.result()
        handler_task.cancel()
        await asyncio.gather(handler_task, return_exceptions=True)
        if self._disposed.is_set():
            raise MCPIsolateLostError(
                "MCP elicitation was interrupted by isolate disposal; retry the call"
            )
        raise asyncio.CancelledError

    def _make_connection(
        self, row: MCPServerRow, generation: int | None = None
    ) -> MCPClientConnection:
        options = decode_server_options(row.server_options)
        provider = None
        if row.callback_url:
            storage, _ = self._resources()
            if self._create_oauth_provider is None:
                provider = DurableOAuthProvider(
                    storage,
                    self._name,
                    row.callback_url,
                    server_id=row.id,
                    client_id=row.client_id,
                )
            else:
                provider = self._create_oauth_provider(
                    storage,
                    self._name,
                    row.callback_url,
                    row.id,
                    row.client_id,
                )
        return MCPClientConnection(
            row,
            options,
            provider,
            generation=(
                self._generations.get(row.id, 0) if generation is None else generation
            ),
        )

    def _transport_factory(
        self, row: MCPServerRow, options: Mapping[str, Any]
    ) -> MCPTransportFactory:
        transport = (
            "rpc"
            if row.server_url.startswith("rpc:")
            else str(options.get("type", "auto"))
        )
        factory = self._transports.get(transport)
        if factory is None and transport in ("streamable-http", "sse"):
            factory = self._transports.get("auto")
        if factory is None:
            raise MCPError(f'No MCP transport adapter configured for "{transport}"')
        return factory

    def _namespaced(
        self, attribute: str, filter: MCPServerFilter | None
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for server_id, connection in self._filtered_connections(filter).items():
            catalog = connection.catalog
            if catalog is None:
                continue
            for item in getattr(catalog, attribute):
                result.append({**item, "serverId": server_id})
        return result

    def _filtered_connections(
        self, filter: MCPServerFilter | None
    ) -> dict[str, MCPClientConnection]:
        if filter is None:
            return dict(self.mcp_connections)
        ids = _as_set(filter.server_id)
        names = _as_set(filter.server_name)
        states = _as_set(filter.state)
        return {
            server_id: connection
            for server_id, connection in self.mcp_connections.items()
            if (ids is None or server_id in ids)
            and (names is None or connection.server.name in names)
            and (states is None or connection.state in states)
        }

    def _persist_session(self, connection: MCPClientConnection) -> None:
        session = connection.session
        if session is None:
            return
        discover_result = None
        if connection.catalog is not None:
            discover_result = {
                "supportedVersions": [session.protocol_version]
                if session.protocol_version
                else [],
                "capabilities": connection.catalog.capabilities or {},
                "instructions": connection.catalog.instructions,
                "resultType": "complete",
            }
        connection.options = with_session(
            connection.options,
            session.session_id,
            session.protocol_version,
            discover_result,
        )
        connection.server = replace(
            connection.server,
            server_options=encode_server_options(connection.options),
        )
        self._store_required().save(connection.server)

    def _ready_session(self, server_id: str) -> MCPTransportSession:
        connection = self._connection(server_id)
        if connection.session is None or connection.state != MCPConnectionState.READY:
            raise MCPError(f'MCP server "{server_id}" is not ready')
        return connection.session

    def _connection(self, server_id: str) -> MCPClientConnection:
        connection = self.mcp_connections.get(server_id)
        if connection is None:
            raise LookupError(f'MCP server "{server_id}" is not registered')
        return connection

    async def _close_connection(
        self, connection: MCPClientConnection, *, persist: bool = True
    ) -> None:
        if connection.session is not None:
            session = connection.session
            connection.session = None
            try:
                await session.close()
            finally:
                connection.options = with_session(connection.options, None, None)
                connection.server = replace(
                    connection.server,
                    server_options=encode_server_options(connection.options),
                )
                if persist and self._store is not None:
                    self._store.save(connection.server)

    def _advance_generation(self, server_id: str) -> int:
        generation = self._generations.get(server_id, 0) + 1
        self._generations[server_id] = generation
        return generation

    def _is_current(self, connection: MCPClientConnection, generation: int) -> bool:
        return (
            connection.generation == generation
            and self._generations.get(connection.server.id) == generation
            and self.mcp_connections.get(connection.server.id) is connection
        )

    def _ensure_current(self, connection: MCPClientConnection, generation: int) -> None:
        if not self._is_current(connection, generation):
            raise _StaleGeneration

    async def _cancel_pending(self, server_id: str) -> None:
        pending = self._pending_connections.pop(server_id, None)
        if pending is None:
            return
        pending.task.cancel()
        await asyncio.gather(pending.task, return_exceptions=True)

    async def _cancel_discovery(self, server_id: str) -> None:
        pending = self._pending_discoveries.pop(server_id, None)
        if pending is None:
            return
        pending.task.cancel()
        await asyncio.gather(pending.task, return_exceptions=True)

    async def _apply_authorization_required(
        self,
        connection: MCPClientConnection,
        required: MCPAuthorizationRequired,
    ) -> None:
        if required.scope_step_up:
            policy = connection.options.get("transport") or {}
            if (
                isinstance(policy, Mapping)
                and policy.get("onInsufficientScope") == "throw"
            ):
                await self._cleanup_authorization_required(connection, required)
                raise MCPError("MCP server requires additional OAuth scope")
            connection.step_up_attempts += 1
            maximum = _bounded_int(
                policy.get("maxStepUpRetries") if isinstance(policy, Mapping) else None,
                1,
                0,
                10,
            )
            if connection.step_up_attempts > maximum:
                await self._cleanup_authorization_required(connection, required)
                raise MCPError("MCP OAuth scope step-up retry limit exceeded")
        connection.state = MCPConnectionState.AUTHENTICATING
        connection.error = None
        provider = connection.oauth_provider
        if provider is not None and required.client_id is not None:
            provider.client_id = required.client_id
        connection.server = replace(
            connection.server,
            auth_url=required.auth_url,
            client_id=required.client_id or connection.server.client_id,
        )
        self._store_required().save(connection.server)
        self._fire_state_changed()

    async def _cleanup_authorization_required(
        self,
        connection: MCPClientConnection,
        required: MCPAuthorizationRequired,
    ) -> None:
        provider = connection.oauth_provider
        if provider is None:
            return
        state = parse_qs(urlsplit(required.auth_url).query).get("state", [None])[0]
        if not isinstance(state, str):
            return
        await provider.consume_state(state)
        await _delete_code_verifier(provider, state)

    def _fail_callback(
        self, connection: MCPClientConnection, error: object
    ) -> MCPOAuthCallbackResult:
        connection.state = MCPConnectionState.FAILED
        connection.error = sanitize_error(error)
        connection.server = replace(connection.server, auth_url=None)
        self._store_required().save(connection.server)
        self._fire_state_changed()
        return MCPOAuthCallbackResult(False, connection.server.id, connection.error)

    def _advertised_handler_capabilities(self) -> dict[str, Any] | None:
        handlers = self._elicitation_handlers
        if handlers is None:
            return None
        elicitation: dict[str, Any] = {}
        if handlers.form is not None:
            elicitation["form"] = {}
        if handlers.url is not None:
            elicitation["url"] = {}
        return {"elicitation": elicitation} if elicitation else None

    def _resources(self) -> tuple[MCPKeyValueStorage, MCPSql]:
        if self._storage_override is not None and self._sql_override is not None:
            return self._storage_override, self._sql_override
        try:
            lifecycle = self.lifecycle
        except RuntimeError as error:
            raise RuntimeError(
                "MCPClientManager requires Lifecycle installation or injected storage"
            ) from error
        return cast(MCPKeyValueStorage, lifecycle.storage), lifecycle.sql

    def _store_required(self) -> MCPServerStore:
        if self._store is None:
            raise RuntimeError("MCPClientManager has not started")
        return self._store

    def _require_started(self) -> None:
        if not self._started:
            raise RuntimeError("MCPClientManager has not started")
        if self._disposed.is_set():
            raise RuntimeError("MCPClientManager is disposed")

    def _fire_state_changed(self) -> None:
        for listener in tuple(self._listeners):
            listener()


def normalize_server_id(value: str) -> str:
    """Normalize a stable server ID for storage and provider tool names."""
    if not isinstance(value, str):
        raise TypeError(f"normalize_server_id expected str, got {type(value).__name__}")
    normalized = re.sub(r"[^a-z0-9_-]+", "-", value.lower())
    normalized = re.sub(r"-+", "-", normalized).strip("-_")
    if not normalized or not normalized[0].isalpha() or not normalized[0].isascii():
        normalized = f"id-{normalized}".rstrip("-")
    normalized = normalized[:MCP_SERVER_ID_MAX_LENGTH].rstrip("-")
    return normalized


def sanitize_error(error: object) -> str | None:
    if error is None:
        return None
    value = re.sub(r"[\x00-\x1f\x7f]+", " ", str(error))
    value = " ".join(value.split())
    return value[:_MAX_ERROR_LENGTH]


def _validate_server_url(value: str) -> None:
    parsed = urlsplit(value)
    if parsed.scheme == "rpc" and parsed.path:
        return
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("MCP server URL must use http, https, or rpc")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("MCP server URL must not contain credentials")
    try:
        _ = parsed.port
    except ValueError as error:
        raise ValueError("MCP server URL has an invalid port") from error
    host = parsed.hostname.lower().rstrip(".")
    if host in ("localhost", "::1") or host.startswith("127."):
        return
    if host in _BLOCKED_HOSTS or host.endswith(_BLOCKED_HOST_SUFFIXES):
        raise ValueError("MCP client connections to internal addresses are blocked")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        if _looks_numeric_host(host):
            raise ValueError(
                "MCP client connections to ambiguous IP addresses are blocked"
            )
        return
    if not address.is_global:
        raise ValueError("MCP client connections to internal addresses are blocked")


def _server_id_from_state(state: object) -> str | None:
    if not isinstance(state, str):
        return None
    parts = state.split(".")
    return parts[1] if len(parts) == 2 and all(parts) else None


def _as_set(value: object) -> set[object] | None:
    if value is None:
        return None
    if isinstance(value, str) or isinstance(value, MCPConnectionState):
        return {value}
    if isinstance(value, Sequence):
        return set(value)
    return {value}


def _tool_error(result: Mapping[str, Any]) -> str:
    content = result.get("content")
    if isinstance(content, list) and content and isinstance(content[0], Mapping):
        first = content[0]
        if first.get("type") == "text" and isinstance(first.get("text"), str):
            return cast(str, first["text"])
    return "Tool call failed"


def _looks_numeric_host(host: str) -> bool:
    return bool(re.fullmatch(r"(?:0x[0-9a-f]+|[0-9a-f]*:[0-9a-f:.]+|[0-9.]+)", host))


def _callback_params(query: str) -> dict[str, str] | None:
    result: dict[str, str] = {}
    for key, value in parse_qsl(query, keep_blank_values=True):
        if key in result:
            return None
        result[key] = value
    return result


async def _delete_code_verifier(provider: MCPOAuthProvider, state: str) -> None:
    delete = getattr(provider, "delete_code_verifier", None)
    if callable(delete):
        value = delete(state)
        if inspect.isawaitable(value):
            await value


def _bounded_int(value: object, default: int, minimum: int, maximum: int) -> int:
    if type(value) is not int:
        return default
    return max(minimum, min(maximum, value))
