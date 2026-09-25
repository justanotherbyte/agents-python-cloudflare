from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from typing import Any, Protocol, cast

from .oauth import DurableOAuthProvider, create_official_oauth_provider
from .types import (
    MCPAbortSignal,
    MCPCatalog,
    MCPError,
    MCPRemoteError,
    MCPStaleSessionError,
    MCPTransportContext,
    MCPTransportFactory,
    MCPTransportNotSupported,
    MCPTransportSession,
)

_DEFAULT_LIST_MAX_PAGES = 100
_MAX_LIST_MAX_PAGES = 1_000
_MAX_RPC_CONTINUATIONS = 32
_CANCEL_CLEANUP_TIMEOUT_SECONDS = 0.1


class HTTPMCPConnector(Protocol):
    """Runtime seam that creates an official MCP HTTP or SSE client session."""

    async def open(
        self,
        *,
        transport: str,
        context: MCPTransportContext,
    ) -> MCPTransportSession: ...


class OfficialMCPConnector:
    """Create HTTP and SSE sessions with the official Python MCP SDK."""

    async def open(
        self, *, transport: str, context: MCPTransportContext
    ) -> MCPTransportSession:
        try:
            from mcp import Client
            from mcp_types import DiscoverResult, Implementation
        except ImportError as error:  # pragma: no cover - packaging integration seam
            raise RuntimeError("HTTP MCP transports require mcp>=2,<3") from error

        headers = _string_headers(context.transport_options.get("headers"))
        auth: Any = None
        if isinstance(context.oauth_provider, DurableOAuthProvider):
            auth = cast(
                Any,
                create_official_oauth_provider(
                    context.oauth_provider,
                    context.server.server_url,
                    context.authorization_params,
                    reauthorize_scope_step_up=context.transport_options.get(
                        "onInsufficientScope"
                    )
                    != "throw",
                ),
            )
        session_ids: dict[str, Any] = {
            "value": _optional_string(context.transport_options.get("sessionId"))
        }
        if transport == "streamable-http":
            sdk_transport = _streamable_transport(
                context.server.server_url, headers, auth, session_ids
            )
        elif transport == "sse":
            from mcp.client.sse import sse_client

            sdk_transport = sse_client(
                context.server.server_url,
                headers=headers,
                auth=auth,
                on_session_created=lambda value: session_ids.update(value=value),
            )
        else:
            raise ValueError(f"unsupported official MCP transport: {transport}")

        protocol_version = _optional_string(
            context.transport_options.get("protocolVersion")
        )
        prior_discover = (
            DiscoverResult.model_validate(context.discover_result)
            if context.discover_result is not None
            else None
        )
        mode = "legacy"
        if protocol_version == "2026-07-28":
            mode = protocol_version
        elif protocol_version is None:
            mode = "auto"

        transport_context = context

        async def elicit(context: object, params: object) -> Any:
            from mcp_types import ElicitResult

            request = {
                "method": "elicitation/create",
                "params": _model_dict(params),
            }
            return ElicitResult.model_validate(
                await transport_context.elicit(request, None)
            )

        async def message_handler(message: object) -> None:
            method = getattr(message, "method", None)
            if (
                method
                in {
                    "notifications/tools/list_changed",
                    "notifications/prompts/list_changed",
                    "notifications/resources/list_changed",
                }
                and context.catalog_changed is not None
            ):
                await context.catalog_changed()

        capabilities = context.client_options.get("capabilities")
        elicitation_enabled = isinstance(capabilities, Mapping) and isinstance(
            capabilities.get("elicitation"), Mapping
        )
        client = Client(
            cast(Any, sdk_transport),
            client_info=Implementation(
                name=context.client_name, version=context.client_version
            ),
            mode=mode,
            prior_discover=prior_discover,
            elicitation_callback=elicit if elicitation_enabled else None,
            message_handler=message_handler,
            input_required_max_rounds=_bounded_int(
                context.client_options.get("inputRequired"), 10, 1, 100
            ),
        )
        try:
            await client.__aenter__()
        except Exception as error:
            await client.__aexit__(type(error), error, error.__traceback__)
            if session_ids.get("value") and _stale_session(error):
                raise MCPStaleSessionError(
                    "Restored MCP session was terminated"
                ) from error
            if _transport_not_supported(error, session_ids.get("last_status")):
                raise MCPTransportNotSupported from error
            raise
        return cast(
            MCPTransportSession,
            OfficialMCPTransportSession(client, context, session_ids),
        )


class OfficialMCPTransportSession:
    """Provider-neutral projection over an entered official MCP Client."""

    def __init__(
        self,
        client: object,
        context: MCPTransportContext,
        session_ids: dict[str, str | None],
    ):
        self._client = client
        self._context = context
        self._session_ids = session_ids
        self._closed = False
        self._restored_session = bool(context.transport_options.get("sessionId"))
        self.protocol_version = _optional_string(
            getattr(client, "protocol_version", None)
        )

    @property
    def session_id(self) -> str | None:
        transport = self._session_ids.get("transport")
        value = getattr(transport, "session_id", None)
        return value if isinstance(value, str) else self._session_ids.get("value")

    async def discover(self) -> MCPCatalog:
        client = self._client
        capabilities = _model_dict(getattr(client, "server_capabilities", None))
        try:
            tools, prompts, resources, templates = await asyncio.gather(
                self._list("list_tools", "tools", bool(capabilities.get("tools"))),
                self._list(
                    "list_prompts", "prompts", bool(capabilities.get("prompts"))
                ),
                self._list(
                    "list_resources",
                    "resources",
                    bool(capabilities.get("resources")),
                ),
                self._list(
                    "list_resource_templates",
                    "resourceTemplates",
                    bool(capabilities.get("resources")),
                ),
            )
        except Exception as error:
            if self._restored_session and _stale_session(error):
                raise MCPStaleSessionError(
                    "Restored MCP session was terminated"
                ) from error
            raise
        return MCPCatalog(
            tools=tools,
            prompts=prompts,
            resources=resources,
            resource_templates=templates,
            capabilities=capabilities,
            instructions=getattr(client, "instructions", None),
        )

    async def call_tool(
        self,
        name: str,
        arguments: Mapping[str, Any],
        *,
        signal: MCPAbortSignal | None = None,
    ) -> Mapping[str, Any]:
        return await self._call(
            getattr(self._client, "call_tool")(name, dict(arguments)), signal
        )

    async def read_resource(
        self,
        params: Mapping[str, Any],
        *,
        signal: MCPAbortSignal | None = None,
    ) -> Mapping[str, Any]:
        uri = params.get("uri")
        if not isinstance(uri, str):
            raise ValueError("resources/read requires a string uri")
        return await self._call(getattr(self._client, "read_resource")(uri), signal)

    async def get_prompt(
        self,
        params: Mapping[str, Any],
        *,
        signal: MCPAbortSignal | None = None,
    ) -> Mapping[str, Any]:
        name = params.get("name")
        arguments = params.get("arguments")
        if not isinstance(name, str):
            raise ValueError("prompts/get requires a string name")
        if arguments is not None and not isinstance(arguments, Mapping):
            raise ValueError("prompts/get arguments must be an object")
        return await self._call(
            getattr(self._client, "get_prompt")(
                name, None if arguments is None else dict(arguments)
            ),
            signal,
        )

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await getattr(self._client, "__aexit__")(None, None, None)

    async def _list(
        self, method: str, key: str, supported: bool
    ) -> list[dict[str, Any]]:
        if not supported:
            return []
        max_pages = _bounded_int(
            self._context.client_options.get("listMaxPages"),
            _DEFAULT_LIST_MAX_PAGES,
            1,
            _MAX_LIST_MAX_PAGES,
        )
        cursor: str | None = None
        seen: set[str] = set()
        result: list[dict[str, Any]] = []
        for _ in range(max_pages):
            page = await getattr(self._client, method)(cursor=cursor)
            raw = _model_dict(page)
            values = raw.get(key, [])
            if not isinstance(values, list):
                raise MCPError(f"{method} returned an invalid {key} list")
            result.extend(_object_list(values, method))
            next_cursor = raw.get("nextCursor")
            if not isinstance(next_cursor, str) or not next_cursor:
                return result
            if next_cursor in seen:
                raise MCPError(f"{method} repeated pagination cursor")
            seen.add(next_cursor)
            cursor = next_cursor
        raise MCPError(f"{method} exceeded listMaxPages={max_pages}")

    async def _call(
        self, awaitable: Awaitable[object], signal: MCPAbortSignal | None
    ) -> Mapping[str, Any]:
        value = await _race_signal(awaitable, signal)
        return _model_dict(value)


@asynccontextmanager
async def _streamable_transport(
    url: str,
    headers: dict[str, str] | None,
    auth: Any,
    session_ids: dict[str, Any],
) -> AsyncIterator[tuple[object, object]]:
    import anyio
    from mcp.client.streamable_http import StreamableHTTPTransport
    from mcp.shared._context_streams import create_context_streams
    from mcp.shared._httpx_utils import create_mcp_http_client

    client = create_mcp_http_client(headers=headers, auth=auth)

    async def record_status(response: object) -> None:
        session_ids["last_status"] = getattr(response, "status_code", None)

    client.event_hooks.setdefault("response", []).append(record_status)
    transport = StreamableHTTPTransport(url)
    transport.session_id = session_ids["value"]
    session_ids["transport"] = transport
    async with client:
        read_writer, read_stream = create_context_streams(0)
        write_stream, write_reader = create_context_streams(0)
        async with (
            read_writer,
            read_stream,
            write_stream,
            write_reader,
            anyio.create_task_group() as task_group,
        ):
            task_group.start_soon(
                transport.post_writer,
                client,
                write_reader,
                read_writer,
                write_stream,
                lambda: task_group.start_soon(
                    transport.handle_get_stream, client, read_writer
                ),
                task_group,
            )
            try:
                yield read_stream, write_stream
            finally:
                session_ids["value"] = transport.session_id
                if transport.session_id:
                    await transport.terminate_session(client)
                task_group.cancel_scope.cancel()


class HTTPTransportAdapter(MCPTransportFactory):
    """Negotiate Streamable HTTP first and use SSE only when unsupported."""

    def __init__(self, connector: HTTPMCPConnector):
        self._connector = connector

    async def open(self, context: MCPTransportContext) -> MCPTransportSession:
        configured = str(context.transport_options.get("type", "auto"))
        if configured == "streamable-http" or configured == "sse":
            return await self._connector.open(transport=configured, context=context)
        if configured != "auto":
            raise ValueError(f"unsupported HTTP MCP transport: {configured}")
        try:
            return await self._connector.open(
                transport="streamable-http", context=context
            )
        except MCPTransportNotSupported:
            return await self._connector.open(transport="sse", context=context)


class RPCMCPBindingResolver(Protocol):
    """Resolve a same-Worker Durable Object registration to its RPC target."""

    def resolve(
        self,
        binding_name: str,
        name: str,
        props: Mapping[str, Any] | None,
    ) -> object | Awaitable[object]: ...


class RPCTransportAdapter(MCPTransportFactory):
    """Create standard JSON-RPC MCP sessions over a same-Worker RPC stub."""

    def __init__(self, resolver: RPCMCPBindingResolver):
        self._resolver = resolver

    async def open(self, context: MCPTransportContext) -> MCPTransportSession:
        options = context.transport_options
        binding_name = options.get("bindingName")
        if not isinstance(binding_name, str) or not binding_name:
            raise ValueError("restoring an RPC MCP server requires bindingName")
        if not context.server.server_url.startswith("rpc:"):
            raise ValueError("RPC MCP server URL must start with 'rpc:'")
        props = options.get("props")
        if props is not None and not isinstance(props, Mapping):
            raise ValueError("RPC MCP props must be an object")
        target = self._resolver.resolve(
            binding_name,
            context.server.server_url.removeprefix("rpc:"),
            cast(Mapping[str, Any] | None, props),
        )
        if inspect.isawaitable(target):
            target = await target
        session: RPCTransportSession | None = None
        try:
            session = RPCTransportSession(target, context)
            await session.start()
        except BaseException:
            if session is not None:
                await session.close()
            else:
                destroy = getattr(target, "destroy", None)
                if callable(destroy):
                    try:
                        destroy()
                    except Exception:
                        pass
            raise
        return session


class RPCTransportSession:
    """A request-correlated MCP client session over an Agent MCP RPC target."""

    session_id = None

    def __init__(self, target: object, context: MCPTransportContext):
        self._target: object | None = target
        handler = getattr(target, "handleMcpMessage", None)
        if callable(handler):
            self._handler = handler
            self._handler_accepts_signal = False
        else:
            handler = getattr(target, "handle_mcp_message", None)
            if not callable(handler):
                raise TypeError(
                    "RPC MCP target must expose handle_mcp_message or handleMcpMessage"
                )
            self._handler = handler
            self._handler_accepts_signal = True
        self._context = context
        self._next_id = 0
        self._closed = False
        self._capabilities: dict[str, Any] = {}
        self._instructions: str | None = None
        self.protocol_version: str | None = None

    async def start(self) -> None:
        supported = self._context.client_options.get("supportedProtocolVersions")
        protocol_version = (
            str(supported[0])
            if isinstance(supported, list) and supported
            else str(
                self._context.transport_options.get("protocolVersion", "2025-06-18")
            )
        )
        result = await self._request(
            "initialize",
            {
                "protocolVersion": protocol_version,
                "capabilities": dict(
                    cast(
                        Mapping[str, Any],
                        self._context.client_options.get("capabilities") or {},
                    )
                ),
                "clientInfo": {
                    "name": self._context.client_name,
                    "version": self._context.client_version,
                },
            },
        )
        negotiated = result.get("protocolVersion")
        self.protocol_version = (
            negotiated if isinstance(negotiated, str) else protocol_version
        )
        capabilities = result.get("capabilities")
        if isinstance(capabilities, Mapping):
            self._capabilities = dict(capabilities)
        instructions = result.get("instructions")
        if isinstance(instructions, str):
            self._instructions = instructions
        await self._notify("notifications/initialized", {})

    async def discover(self) -> MCPCatalog:
        tools: list[dict[str, Any]] = []
        prompts: list[dict[str, Any]] = []
        resources: list[dict[str, Any]] = []
        templates: list[dict[str, Any]] = []
        if "tools" in self._capabilities:
            tools = await self._list_pages("tools/list", "tools")
        if "prompts" in self._capabilities:
            prompts = await self._list_pages("prompts/list", "prompts")
        if "resources" in self._capabilities:
            resources = await self._list_pages("resources/list", "resources")
            templates = await self._list_pages(
                "resources/templates/list", "resourceTemplates"
            )
        return MCPCatalog(
            tools=tools,
            prompts=prompts,
            resources=resources,
            resource_templates=templates,
            capabilities=dict(self._capabilities),
            instructions=self._instructions,
        )

    async def call_tool(
        self,
        name: str,
        arguments: Mapping[str, Any],
        *,
        signal: MCPAbortSignal | None = None,
    ) -> Mapping[str, Any]:
        return await self._request(
            "tools/call", {"name": name, "arguments": dict(arguments)}, signal=signal
        )

    async def read_resource(
        self,
        params: Mapping[str, Any],
        *,
        signal: MCPAbortSignal | None = None,
    ) -> Mapping[str, Any]:
        return await self._request("resources/read", dict(params), signal=signal)

    async def get_prompt(
        self,
        params: Mapping[str, Any],
        *,
        signal: MCPAbortSignal | None = None,
    ) -> Mapping[str, Any]:
        return await self._request("prompts/get", dict(params), signal=signal)

    async def close(self) -> None:
        self._closed = True
        target = self._target
        self._target = None
        destroy = getattr(target, "destroy", None)
        if callable(destroy):
            destroy()

    async def _list_pages(self, method: str, key: str) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        cursor: str | None = None
        seen: set[str] = set()
        max_pages = _bounded_int(
            self._context.client_options.get("listMaxPages"),
            _DEFAULT_LIST_MAX_PAGES,
            1,
            _MAX_LIST_MAX_PAGES,
        )
        for _ in range(max_pages):
            params = {} if cursor is None else {"cursor": cursor}
            result = await self._request(method, params)
            page = result.get(key, [])
            if not isinstance(page, list):
                raise MCPError(f"{method} returned an invalid {key} list")
            for item in page:
                if not isinstance(item, Mapping):
                    raise MCPError(f"{method} returned a non-object item")
                items.append(dict(item))
            next_cursor = result.get("nextCursor")
            if not isinstance(next_cursor, str) or not next_cursor:
                return items
            if next_cursor in seen:
                raise MCPError(f"{method} repeated pagination cursor")
            seen.add(next_cursor)
            cursor = next_cursor
        raise MCPError(f"{method} exceeded listMaxPages={max_pages}")

    async def _request(
        self,
        method: str,
        params: Mapping[str, Any],
        *,
        signal: MCPAbortSignal | None = None,
    ) -> Mapping[str, Any]:
        self._ensure_open()
        self._next_id += 1
        request_id = self._next_id
        message = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": dict(params),
        }
        response = await self._send(message, signal)
        return await self._consume_response(response, request_id, signal)

    async def _consume_response(
        self,
        response: object,
        request_id: int,
        signal: MCPAbortSignal | None,
        continuations: int = 0,
    ) -> Mapping[str, Any]:
        if continuations > _MAX_RPC_CONTINUATIONS:
            raise MCPError("MCP RPC response exceeded continuation limit")
        messages = response if isinstance(response, list) else [response]
        for raw in messages:
            if not isinstance(raw, Mapping):
                continue
            if "method" in raw and "id" in raw:
                method = raw.get("method")
                params = raw.get("params")
                if method != "elicitation/create" or not isinstance(params, Mapping):
                    reply: dict[str, Any] = {
                        "jsonrpc": "2.0",
                        "id": raw["id"],
                        "error": {"code": -32601, "message": "Method not found"},
                    }
                else:
                    try:
                        elicited = await self._context.elicit(dict(raw), signal)
                        reply = {
                            "jsonrpc": "2.0",
                            "id": raw["id"],
                            "result": dict(elicited),
                        }
                    except Exception as error:
                        reply = {
                            "jsonrpc": "2.0",
                            "id": raw["id"],
                            "error": {"code": -32603, "message": str(error)},
                        }
                continuation = await self._send(reply, signal)
                return await self._consume_response(
                    continuation, request_id, signal, continuations + 1
                )
            if raw.get("id") != request_id:
                continue
            error = raw.get("error")
            if isinstance(error, Mapping):
                code = error.get("code")
                message = error.get("message")
                raise MCPRemoteError(
                    code if type(code) is int else -32603,
                    message if isinstance(message, str) else "MCP request failed",
                    error.get("data"),
                )
            result = raw.get("result")
            if not isinstance(result, Mapping):
                raise MCPError("MCP JSON-RPC response result must be an object")
            return dict(result)
        raise MCPError("MCP JSON-RPC response did not match the request")

    async def _notify(self, method: str, params: Mapping[str, Any]) -> None:
        await self._send(
            {"jsonrpc": "2.0", "method": method, "params": dict(params)}, None
        )

    async def _send(
        self, message: Mapping[str, Any], signal: MCPAbortSignal | None
    ) -> object:
        if signal is not None and signal.aborted:
            raise asyncio.CancelledError
        pending = _call_handler(
            self._handler,
            dict(message),
            signal,
            accepts_signal=self._handler_accepts_signal,
        )
        if signal is None:
            return await pending
        call_task = asyncio.create_task(pending)
        signal_task = asyncio.create_task(signal.wait())
        cancellation_sent = False
        try:
            done, _ = await asyncio.wait(
                (call_task, signal_task), return_when=asyncio.FIRST_COMPLETED
            )
            if call_task in done:
                return call_task.result()
            await self._cancel_remote_call(call_task, message)
            cancellation_sent = True
            raise asyncio.CancelledError
        except asyncio.CancelledError:
            if not cancellation_sent and not call_task.done():
                await self._cancel_remote_call(call_task, message)
            raise
        finally:
            if not call_task.done():
                await _bounded_cancel(call_task)
            await _bounded_cancel(signal_task)

    async def _cancel_remote_call(
        self,
        call_task: asyncio.Task[object],
        message: Mapping[str, Any],
    ) -> None:
        notification = _call_handler(
            self._handler,
            {
                "jsonrpc": "2.0",
                "method": "notifications/cancelled",
                "params": {"requestId": message.get("id")},
            },
            None,
            accepts_signal=self._handler_accepts_signal,
        )
        await asyncio.gather(
            _bounded_await(notification),
            _bounded_cancel(call_task),
            return_exceptions=True,
        )

    def _ensure_open(self) -> None:
        if self._closed:
            raise MCPError("MCP RPC transport is closed")


async def _call_handler(
    handler: Callable[..., object],
    message: Mapping[str, Any],
    signal: MCPAbortSignal | None,
    *,
    accepts_signal: bool,
) -> object:
    result = handler(message, signal) if accepts_signal else handler(message)
    return await result if inspect.isawaitable(result) else result


async def _bounded_cancel(task: asyncio.Task[Any]) -> None:
    task.cancel()
    done, _ = await asyncio.wait((task,), timeout=_CANCEL_CLEANUP_TIMEOUT_SECONDS)
    if task not in done:
        task.add_done_callback(_consume_task_result)
        return
    try:
        task.result()
    except (asyncio.CancelledError, Exception):
        pass


async def _bounded_await(awaitable: Awaitable[object]) -> None:
    task = asyncio.ensure_future(awaitable)
    done, _ = await asyncio.wait((task,), timeout=_CANCEL_CLEANUP_TIMEOUT_SECONDS)
    if task not in done:
        task.cancel()
        task.add_done_callback(_consume_task_result)
        return
    try:
        task.result()
    except (asyncio.CancelledError, Exception):
        pass


def _consume_task_result(task: asyncio.Future[object]) -> None:
    try:
        task.result()
    except (asyncio.CancelledError, Exception):
        pass


async def _race_signal(
    awaitable: Awaitable[object], signal: MCPAbortSignal | None
) -> object:
    if signal is None:
        return await awaitable
    if signal.aborted:
        if inspect.iscoroutine(awaitable):
            awaitable.close()
        raise asyncio.CancelledError
    call_task = asyncio.ensure_future(awaitable)
    signal_task = asyncio.create_task(signal.wait())
    done, _ = await asyncio.wait(
        (call_task, signal_task), return_when=asyncio.FIRST_COMPLETED
    )
    if call_task in done:
        await _bounded_cancel(signal_task)
        return call_task.result()
    await _bounded_cancel(call_task)
    raise asyncio.CancelledError


def _model_dict(value: object) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    model_dump = getattr(value, "model_dump", None)
    if not callable(model_dump):
        raise MCPError(
            f"Official MCP SDK returned {type(value).__name__}, not an object"
        )
    dumped = model_dump(by_alias=True, mode="json", exclude_none=True)
    if not isinstance(dumped, dict):
        raise MCPError("Official MCP SDK returned a non-object model")
    return dumped


def _object_list(values: list[object], method: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for value in values:
        try:
            result.append(_model_dict(value))
        except MCPError as error:
            raise MCPError(f"{method} returned a non-object item") from error
    return result


def _string_headers(value: object) -> dict[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("MCP transport headers must be an object")
    headers: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, str):
            raise ValueError("MCP transport headers must contain only strings")
        headers[key] = item
    return headers


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _bounded_int(value: object, default: int, minimum: int, maximum: int) -> int:
    if type(value) is not int:
        return default
    return max(minimum, min(maximum, value))


def _transport_not_supported(
    error: BaseException, recorded_status: object = None
) -> bool:
    status = recorded_status if type(recorded_status) is int else _status_code(error)
    if status in (404, 405, 501):
        return True
    return (
        getattr(error, "code", None) == -32601
        or "method not found" in str(error).lower()
    )


def _stale_session(error: BaseException) -> bool:
    status = _status_code(error)
    return status in (404, 410) or "session terminated" in str(error).lower()


def _status_code(error: BaseException) -> int | None:
    value = getattr(error, "status", None)
    if type(value) is int:
        return value
    response = getattr(error, "response", None)
    value = getattr(response, "status_code", None)
    return value if type(value) is int else None
