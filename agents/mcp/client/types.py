from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol


class MCPConnectionState(StrEnum):
    AUTHENTICATING = "authenticating"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCOVERING = "discovering"
    READY = "ready"
    FAILED = "failed"


class MCPError(Exception):
    """Base error for the provider-neutral MCP client core."""


class MCPTransportNotSupported(MCPError):
    """Signal that automatic HTTP negotiation should try SSE."""


class MCPRemoteError(MCPError):
    """A standard JSON-RPC error returned by an MCP server."""

    def __init__(self, code: int, message: str, data: object = None):
        super().__init__(message)
        self.code = code
        self.data = data


class MCPAuthorizationRequired(MCPError):
    """Signal that a transport has started an OAuth authorization flow."""

    def __init__(
        self,
        auth_url: str,
        client_id: str | None = None,
        *,
        scope_step_up: bool = False,
    ):
        super().__init__("MCP authorization is required")
        self.auth_url = auth_url
        self.client_id = client_id
        self.scope_step_up = scope_step_up


class MCPStaleSessionError(MCPError):
    """A restored HTTP session no longer exists on the server."""


class MCPIsolateLostError(MCPError):
    """An in-memory elicitation ended because its manager was disposed."""


class MCPAbortSignal(Protocol):
    @property
    def aborted(self) -> bool: ...

    async def wait(self) -> None: ...


@dataclass(frozen=True, slots=True)
class MCPServerRow:
    id: str
    name: str
    server_url: str
    callback_url: str
    client_id: str | None = None
    auth_url: str | None = None
    server_options: str | None = None


@dataclass(frozen=True, slots=True)
class MCPRetryOptions:
    max_attempts: int = 3
    base_delay_ms: int = 500
    max_delay_ms: int = 5_000


@dataclass(frozen=True, slots=True)
class MCPServerFilter:
    server_id: str | Sequence[str] | None = None
    server_name: str | Sequence[str] | None = None
    state: MCPConnectionState | Sequence[MCPConnectionState] | None = None


@dataclass(slots=True)
class MCPCatalog:
    tools: list[dict[str, Any]] = field(default_factory=list)
    prompts: list[dict[str, Any]] = field(default_factory=list)
    resources: list[dict[str, Any]] = field(default_factory=list)
    resource_templates: list[dict[str, Any]] = field(default_factory=list)
    capabilities: dict[str, Any] | None = None
    instructions: str | None = None


type ElicitationHandler = Callable[
    [Mapping[str, Any], str, MCPAbortSignal | None],
    Mapping[str, Any] | Awaitable[Mapping[str, Any]],
]


@dataclass(frozen=True, slots=True)
class MCPElicitationHandlers:
    form: ElicitationHandler | None = None
    url: ElicitationHandler | None = None


class MCPOAuthProvider(Protocol):
    server_id: str
    client_id: str | None

    async def check_state(self, state: str) -> tuple[bool, str | None]: ...

    async def consume_state(self, state: str) -> None: ...


type ElicitationDispatcher = Callable[
    [Mapping[str, Any], MCPAbortSignal | None],
    Awaitable[Mapping[str, Any]],
]
type CatalogChanged = Callable[[], Awaitable[object]]


@dataclass(frozen=True, slots=True)
class MCPTransportContext:
    server: MCPServerRow
    client_name: str
    client_version: str
    client_options: Mapping[str, Any]
    transport_options: Mapping[str, Any]
    oauth_provider: MCPOAuthProvider | None
    elicit: ElicitationDispatcher
    authorization_params: Mapping[str, str] | None = None
    catalog_changed: CatalogChanged | None = None
    discover_result: Mapping[str, Any] | None = None


class MCPTransportSession(Protocol):
    @property
    def session_id(self) -> str | None: ...

    protocol_version: str | None

    async def discover(self) -> MCPCatalog: ...

    async def call_tool(
        self,
        name: str,
        arguments: Mapping[str, Any],
        *,
        signal: MCPAbortSignal | None = None,
    ) -> Mapping[str, Any]: ...

    async def read_resource(
        self,
        params: Mapping[str, Any],
        *,
        signal: MCPAbortSignal | None = None,
    ) -> Mapping[str, Any]: ...

    async def get_prompt(
        self,
        params: Mapping[str, Any],
        *,
        signal: MCPAbortSignal | None = None,
    ) -> Mapping[str, Any]: ...

    async def close(self) -> None: ...


class MCPTransportFactory(Protocol):
    async def open(self, context: MCPTransportContext) -> MCPTransportSession: ...


type ToolInvoker = Callable[[Mapping[str, Any]], Awaitable[Mapping[str, Any]]]


@dataclass(frozen=True, slots=True)
class MCPAIToolDescriptor:
    input_schema: Mapping[str, Any]
    execute: ToolInvoker
    description: str | None = None
    title: str | None = None
    output_schema: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class MCPOAuthCallbackResult:
    auth_success: bool
    server_id: str | None = None
    auth_error: str | None = None


type OAuthCallbackHandler = Callable[[MCPOAuthCallbackResult], object]


@dataclass(frozen=True, slots=True)
class MCPOAuthCallbackPolicy:
    success_redirect: str | None = None
    error_redirect: str | None = None
    custom_handler: OAuthCallbackHandler | None = None
