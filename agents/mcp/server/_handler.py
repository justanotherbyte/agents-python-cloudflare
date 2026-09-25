from __future__ import annotations

import asyncio
import inspect
import ipaddress
import json
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field
from http import HTTPMethod
from typing import Any, Literal, Protocol, TypeVar, cast
from urllib.parse import urlsplit

from mcp.server.auth.middleware.bearer_auth import AuthenticatedUser
from mcp.server.auth.middleware.auth_context import AuthContextMiddleware
from mcp.server.auth.provider import AccessToken
from mcp.server.transport_security import TransportSecuritySettings
from starlette.authentication import AuthCredentials
from starlette.types import ASGIApp, Message, Receive, Scope, Send
from workers import Response

_DEFAULT_CORS_HEADERS = (
    "Content-Type, Accept, Authorization, Last-Event-ID, mcp-session-id, "
    "MCP-Protocol-Version, Mcp-Method, Mcp-Name"
)
_LOCAL_HOSTNAMES = frozenset({"127.0.0.1", "::1", "localhost"})
_SUPPORTED_METHODS = frozenset({"GET", "POST", "DELETE"})
_ALLOW_MCP_METHODS = "GET, POST, DELETE"
_ALLOW_METHODS = "GET, POST, DELETE, OPTIONS"
_HOST_LABEL = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?$")


@dataclass(frozen=True, kw_only=True)
class CORSOptions:
    """CORS headers applied to every response from the MCP route."""

    origin: str = "*"
    headers: str = _DEFAULT_CORS_HEADERS
    methods: str = "GET, POST, DELETE, OPTIONS"
    expose_headers: str = "mcp-session-id"
    max_age: int = 86400


@dataclass(frozen=True, kw_only=True)
class MCPAuthContext:
    """Trusted application properties available while an MCP request runs."""

    props: Mapping[str, object]


@dataclass(frozen=True, kw_only=True)
class VerifiedMCPAuth:
    """OAuth data already verified by the Worker authentication boundary."""

    access_token: AccessToken
    props: Mapping[str, object]


class MCPAuthVerifier(Protocol):
    """Resolve trusted OAuth data from a Worker execution context."""

    def verify(
        self, execution_context: object
    ) -> VerifiedMCPAuth | None | Awaitable[VerifiedMCPAuth | None]: ...


@dataclass(frozen=True, kw_only=True)
class StatelessElicitation:
    """Shared upstream request-state security for stateless elicitation rounds.

    Pass an official ``mcp.server.mcpserver.RequestStateSecurity`` instance and
    use it when constructing each server in the factory. Sharing it lets a later
    HTTP request verify request state minted by an earlier server instance.
    """

    request_state_security: object


@dataclass(frozen=True, kw_only=True)
class MCPServerFactoryContext:
    """Request-scoped inputs supplied to the MCP server factory."""

    auth_info: AccessToken | None
    elicitation: StatelessElicitation | None


@dataclass(frozen=True, kw_only=True)
class MCPHandlerOptions:
    """Configuration for a stateless MCP Worker handler."""

    route: str = "/mcp"
    cors: CORSOptions | Literal[False] = field(default_factory=CORSOptions)
    allowed_hostnames: Sequence[str] | None = None
    allowed_origin_hostnames: Sequence[str] | Literal["*"] | None = None
    auth_context: MCPAuthContext | None = None
    auth_verifier: MCPAuthVerifier | None = None
    elicitation: StatelessElicitation | None = None
    json_response: bool = False
    max_request_body_size: int = 4 * 1024 * 1024
    on_error: Callable[[Exception], object] | None = None


class _MCPServerApplication(Protocol):
    def streamable_http_app(
        self,
        *,
        streamable_http_path: str,
        json_response: bool,
        stateless_http: bool,
        max_request_body_size: int,
        transport_security: TransportSecuritySettings,
    ) -> ASGIApp: ...


class _ASGIRuntime(Protocol):
    async def fetch(
        self,
        app: ASGIApp,
        request: _WorkerRequest,
        env: object,
        execution_context: object | None,
    ) -> object: ...


class _WorkersExecutionContext(Protocol):
    def waitUntil(self, other: Awaitable[Any]) -> None: ...


class _Headers(Protocol):
    def get(self, name: str) -> object | None: ...


class _WorkerRequest(Protocol):
    url: str
    method: str | HTTPMethod
    headers: _Headers


class _WorkersASGIRuntime:
    async def fetch(
        self,
        app: ASGIApp,
        request: _WorkerRequest,
        env: object,
        execution_context: object | None,
    ) -> object:
        from asgi import fetch as asgi_fetch

        # The stock bridge ends its lifespan when it returns a streaming
        # Response, before the ASGI request task necessarily finishes.
        app = _PerRequestLifespanApp(app)
        context = cast(_WorkersExecutionContext | None, execution_context)
        return await asgi_fetch(app, request, env, context)


class _PerRequestLifespanApp:
    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "lifespan":
            await self._bridge_lifespan(receive, send)
            return

        queue: asyncio.Queue[Message] = asyncio.Queue()
        startup = asyncio.get_running_loop().create_future()
        shutdown = asyncio.get_running_loop().create_future()
        state: dict[str, object] = {}

        async def lifespan_receive() -> Message:
            return await queue.get()

        async def lifespan_send(message: Message) -> None:
            if message["type"] == "lifespan.startup.complete":
                if not startup.done():
                    startup.set_result(None)
            elif message["type"] == "lifespan.startup.failed":
                if not startup.done():
                    startup.set_exception(
                        RuntimeError(message.get("message", "ASGI startup failed"))
                    )
            elif message["type"] == "lifespan.shutdown.complete":
                if not shutdown.done():
                    shutdown.set_result(None)
            elif message["type"] == "lifespan.shutdown.failed":
                if not shutdown.done():
                    shutdown.set_exception(
                        RuntimeError(message.get("message", "ASGI shutdown failed"))
                    )

        async def run_lifespan() -> None:
            try:
                await self._app(
                    {
                        "asgi": {"spec_version": "2.0", "version": "3.0"},
                        "state": state,
                        "type": "lifespan",
                    },
                    lifespan_receive,
                    lifespan_send,
                )
            except BaseException as error:
                if not startup.done():
                    startup.set_exception(error)
                elif (
                    not startup.cancelled()
                    and startup.exception() is None
                    and not shutdown.done()
                ):
                    shutdown.set_exception(error)
            else:
                if not startup.done():
                    startup.set_exception(
                        RuntimeError("ASGI lifespan ended before startup completed")
                    )
                elif (
                    not startup.cancelled()
                    and startup.exception() is None
                    and not shutdown.done()
                ):
                    shutdown.set_exception(
                        RuntimeError("ASGI lifespan ended before shutdown completed")
                    )

        lifespan_task = asyncio.create_task(run_lifespan())
        await queue.put({"type": "lifespan.startup"})
        started = False
        try:
            await startup
            started = True
            request_scope = dict(scope)
            request_scope["state"] = dict(state)
            await self._app(request_scope, receive, send)
        finally:
            if started:
                await queue.put({"type": "lifespan.shutdown"})
                await shutdown
            elif not lifespan_task.done():
                lifespan_task.cancel()
            await lifespan_task

    @staticmethod
    async def _bridge_lifespan(receive: Receive, send: Send) -> None:
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                await send({"type": "lifespan.startup.complete"})
            elif message["type"] == "lifespan.shutdown":
                await send({"type": "lifespan.shutdown.complete"})
                return


_CURRENT_AUTH_CONTEXT: ContextVar[MCPAuthContext | None] = ContextVar(
    "agents_mcp_auth_context", default=None
)
_ServerFactory = Callable[[MCPServerFactoryContext], _MCPServerApplication]
_T = TypeVar("_T")


def get_mcp_auth_context() -> MCPAuthContext | None:
    """Return the trusted auth properties for the current MCP request."""

    return _CURRENT_AUTH_CONTEXT.get()


class MCPHandler:
    """Callable stateless MCP handler with a Worker-style ``fetch`` face."""

    def __init__(
        self,
        factory: _ServerFactory,
        options: MCPHandlerOptions,
        runtime: _ASGIRuntime,
    ) -> None:
        if not callable(factory):
            raise TypeError("create_mcp_handler requires an MCP server factory")
        _validate_options(options)
        self._factory = factory
        self._options = options
        self._runtime = runtime

    async def __call__(
        self,
        request: _WorkerRequest,
        env: object = None,
        execution_context: object | None = None,
    ) -> object:
        return await self.fetch(request, env=env, execution_context=execution_context)

    async def fetch(
        self,
        request: _WorkerRequest,
        *,
        env: object = None,
        execution_context: object | None = None,
    ) -> object:
        cors_headers = _cors_headers(self._options.cors)
        try:
            request_url = urlsplit(str(request.url))
        except ValueError:
            return _json_error(-32600, "Invalid request URL", 400, cors_headers)
        if request_url.path != self._options.route:
            return _response("Not Found", 404, cors_headers)

        method = _method(request)
        if _request_aborted(request):
            return _response(None, 499, cors_headers)
        if method == "OPTIONS" and self._options.cors is not False:
            permitted_method = True
        else:
            permitted_method = method in _SUPPORTED_METHODS
        if not permitted_method:
            return _response(
                "Method Not Allowed",
                405,
                cors_headers,
                extra_headers={
                    "allow": (
                        _ALLOW_METHODS
                        if self._options.cors is not False
                        else _ALLOW_MCP_METHODS
                    )
                },
            )

        body_rejection = _body_rejection(request, self._options.max_request_body_size)
        if body_rejection is not None:
            status, message = body_rejection
            return _json_error(-32000, message, status, cors_headers)

        rejection = _security_rejection(request, self._options)
        if rejection is not None:
            status, message = rejection
            return _json_error(-32000, message, status, cors_headers)

        if method == "OPTIONS" and self._options.cors is not False:
            return _response(None, 200, cors_headers)

        try:
            verified = await _resolve_verified_auth(
                self._options.auth_verifier, execution_context
            )
            auth_info = verified.access_token if verified is not None else None
            auth_context = self._options.auth_context
            if auth_context is None and verified is not None:
                auth_context = MCPAuthContext(props=verified.props)

            token = _CURRENT_AUTH_CONTEXT.set(auth_context)
            try:
                server = self._factory(
                    MCPServerFactoryContext(
                        auth_info=auth_info,
                        elicitation=self._options.elicitation,
                    )
                )
                if inspect.isawaitable(server):
                    raise TypeError("The MCP server factory must be synchronous")
                app = server.streamable_http_app(
                    streamable_http_path=self._options.route,
                    json_response=self._options.json_response,
                    stateless_http=True,
                    max_request_body_size=self._options.max_request_body_size,
                    transport_security=TransportSecuritySettings(
                        enable_dns_rebinding_protection=False
                    ),
                )
                app = AuthContextMiddleware(app)
                app = _RequestBoundary(app, auth_info, auth_context, cors_headers)
                return await self._runtime.fetch(app, request, env, execution_context)
            finally:
                _CURRENT_AUTH_CONTEXT.reset(token)
        except Exception as error:
            _report_error(self._options.on_error, error)
            return _json_error(-32603, "Internal server error", 500, cors_headers)


class _RequestBoundary:
    def __init__(
        self,
        app: ASGIApp,
        auth_info: AccessToken | None,
        auth_context: MCPAuthContext | None,
        cors_headers: list[tuple[bytes, bytes]],
    ) -> None:
        self._app = app
        self._auth_info = auth_info
        self._auth_context = auth_context
        self._cors_headers = cors_headers

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        request_scope = dict(scope)
        if self._auth_info is not None:
            request_scope["auth"] = AuthCredentials(self._auth_info.scopes)
            request_scope["user"] = AuthenticatedUser(self._auth_info)

        async def send_with_cors(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = [
                    (name, value)
                    for name, value in message.get("headers", [])
                    if not name.lower().startswith(b"access-control-")
                ]
                headers.extend(self._cors_headers)
                message = {**message, "headers": headers}
            await send(message)

        token = _CURRENT_AUTH_CONTEXT.set(self._auth_context)
        try:
            await self._app(request_scope, receive, send_with_cors)
        finally:
            _CURRENT_AUTH_CONTEXT.reset(token)


def create_mcp_handler(
    factory: _ServerFactory,
    options: MCPHandlerOptions | None = None,
    *,
    _runtime: _ASGIRuntime | None = None,
) -> MCPHandler:
    """Create an isolated stateless MCP handler for a Python Worker."""

    return MCPHandler(
        factory,
        options or MCPHandlerOptions(),
        _runtime or _WorkersASGIRuntime(),
    )


def _validate_options(options: MCPHandlerOptions) -> None:
    route = urlsplit(options.route)
    if (
        not options.route.startswith("/")
        or route.path != options.route
        or route.query
        or route.fragment
    ):
        raise ValueError("MCP route must be an exact absolute pathname")
    if options.max_request_body_size <= 0:
        raise ValueError("max_request_body_size must be positive")
    for hostname in options.allowed_hostnames or ():
        _validate_hostname_option(hostname)
    if options.allowed_origin_hostnames != "*":
        for hostname in options.allowed_origin_hostnames or ():
            _validate_hostname_option(hostname)


def _validate_hostname_option(hostname: str) -> None:
    if not isinstance(hostname, str) or _canonical_hostname(hostname) is None:
        raise ValueError(
            f"Expected a hostname without a scheme or port, got {hostname!r}"
        )


async def _resolve_verified_auth(
    verifier: MCPAuthVerifier | None,
    execution_context: object | None,
) -> VerifiedMCPAuth | None:
    if verifier is None:
        return None
    if execution_context is None:
        raise TypeError("An execution context is required by auth_verifier")
    result = verifier.verify(execution_context)
    if inspect.isawaitable(result):
        result = await result
    if result is not None and not isinstance(result, VerifiedMCPAuth):
        raise TypeError("auth_verifier returned invalid verified OAuth data")
    return result


def _security_rejection(
    request: _WorkerRequest, options: MCPHandlerOptions
) -> tuple[int, str] | None:
    try:
        url = urlsplit(str(request.url))
        endpoint = _canonical_hostname(url.hostname or "") or ""
    except ValueError:
        return 400, "Invalid request URL"
    allowed_hosts = options.allowed_hostnames
    if allowed_hosts is None:
        if endpoint in _LOCAL_HOSTNAMES:
            allowed_hosts = tuple(_LOCAL_HOSTNAMES)
        elif endpoint.endswith(".workers.dev"):
            allowed_hosts = (endpoint,)

    if allowed_hosts is not None:
        host = _header(request, "host")
        accepted_hosts = {
            canonical
            for value in allowed_hosts
            if (canonical := _canonical_hostname(value)) is not None
        }
        if host is None or _host_authority_hostname(host) not in accepted_hosts:
            return 403, "Invalid Host header"

    allowed_origins = options.allowed_origin_hostnames
    if allowed_origins == "*":
        return None
    origin = _header(request, "origin")
    if origin is None:
        return None
    try:
        parsed_origin = urlsplit(origin)
    except ValueError:
        return 403, "Invalid Origin header"
    if parsed_origin.scheme not in {"http", "https"} or parsed_origin.hostname is None:
        return 403, "Invalid Origin header"
    origin_hostname = _canonical_hostname(parsed_origin.hostname)
    if origin_hostname is None:
        return 403, "Invalid Origin header"

    if allowed_origins is None:
        origins = set(_LOCAL_HOSTNAMES)
        if endpoint.endswith(".workers.dev"):
            origins.add(endpoint)
        cors = options.cors
        if cors is not False:
            configured = urlsplit(cors.origin)
            if configured.scheme in {"http", "https"} and configured.hostname:
                configured_hostname = _canonical_hostname(configured.hostname)
                if configured_hostname is not None:
                    origins.add(configured_hostname)
    else:
        origins = {
            canonical
            for value in allowed_origins
            if (canonical := _canonical_hostname(value)) is not None
        }
    if origin_hostname not in origins:
        return 403, "Invalid Origin header"
    return None


def _canonical_hostname(value: str) -> str | None:
    if not value or value != value.strip() or not value.isascii():
        return None
    try:
        return ipaddress.ip_address(value).compressed.lower()
    except ValueError:
        pass
    if len(value) > 253 or value.endswith("."):
        return None
    labels = value.split(".")
    if not all(_HOST_LABEL.fullmatch(label) for label in labels):
        return None
    return value.lower()


def _host_authority_hostname(authority: str) -> str | None:
    if (
        not authority
        or authority != authority.strip()
        or not authority.isascii()
        or any(character in authority for character in "/?#,@")
    ):
        return None

    if authority.startswith("["):
        closing = authority.find("]")
        if closing < 0:
            return None
        hostname = authority[1:closing]
        suffix = authority[closing + 1 :]
        try:
            canonical = ipaddress.IPv6Address(hostname).compressed.lower()
        except ValueError:
            return None
        if suffix and (not suffix.startswith(":") or not _valid_port(suffix[1:])):
            return None
        return canonical

    if "[" in authority or "]" in authority or authority.count(":") > 1:
        return None
    hostname, separator, port = authority.rpartition(":")
    if not separator:
        hostname = authority
    elif not _valid_port(port):
        return None
    return _canonical_hostname(hostname)


def _valid_port(port: str) -> bool:
    return port.isascii() and port.isdecimal() and len(port) <= 5 and int(port) <= 65535


def _header(request: _WorkerRequest, name: str) -> str | None:
    headers = request.headers
    value = headers.get(name)
    return str(value) if value is not None else None


def _method(request: _WorkerRequest) -> str:
    method = request.method
    if isinstance(method, HTTPMethod):
        return method.value
    return str(method)


def _request_aborted(request: _WorkerRequest) -> bool:
    signal = getattr(request, "signal", None)
    if signal is None:
        js_request = getattr(request, "js_object", None)
        signal = getattr(js_request, "signal", None)
    return bool(getattr(signal, "aborted", False))


def _body_rejection(
    request: _WorkerRequest, max_request_body_size: int
) -> tuple[int, str] | None:
    content_length = _header(request, "content-length")
    if content_length is not None:
        if not content_length.isascii() or not content_length.isdecimal():
            return 400, "Invalid Content-Length header"
        normalized_length = content_length.lstrip("0") or "0"
        maximum = str(max_request_body_size)
        if len(normalized_length) > len(maximum) or (
            len(normalized_length) == len(maximum) and normalized_length > maximum
        ):
            return 413, "Request body too large"

    body = getattr(request, "body", None)
    if isinstance(body, bytes | bytearray | memoryview):
        if len(body) > max_request_body_size:
            return 413, "Request body too large"
    return None


def _cors_headers(
    options: CORSOptions | Literal[False],
) -> list[tuple[bytes, bytes]]:
    if options is False:
        return []
    return [
        (b"access-control-allow-origin", options.origin.encode()),
        (b"access-control-allow-headers", options.headers.encode()),
        (b"access-control-allow-methods", options.methods.encode()),
        (b"access-control-expose-headers", options.expose_headers.encode()),
        (b"access-control-max-age", str(options.max_age).encode()),
    ]


def _response(
    body: str | None,
    status: int,
    headers: list[tuple[bytes, bytes]],
    *,
    extra_headers: Mapping[str, str] | None = None,
) -> Response:
    response_headers = {name.decode(): value.decode() for name, value in headers}
    response_headers.update(extra_headers or {})
    return Response(
        body,
        status=status,
        headers=response_headers,
    )


def _json_error(
    code: int,
    message: str,
    status: int,
    headers: list[tuple[bytes, bytes]],
) -> Response:
    response_headers = {name.decode(): value.decode() for name, value in headers}
    response_headers["content-type"] = "application/json"
    body = json.dumps(
        {
            "jsonrpc": "2.0",
            "error": {"code": code, "message": message},
            "id": None,
        },
        separators=(",", ":"),
    )
    return Response(body, status=status, headers=response_headers)


def _report_error(
    reporter: Callable[[Exception], object] | None, error: Exception
) -> None:
    if reporter is None:
        return
    try:
        reporter(error)
    except Exception:
        pass
