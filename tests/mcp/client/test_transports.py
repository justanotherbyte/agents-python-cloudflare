from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

import pytest

from agents.mcp.client import (
    HTTPTransportAdapter,
    MCPCatalog,
    MCPServerRow,
    MCPTransportContext,
    MCPTransportNotSupported,
    RPCTransportAdapter,
)


class EmptySession:
    session_id = None
    protocol_version = None

    async def discover(self) -> MCPCatalog:
        return MCPCatalog()

    async def call_tool(
        self, name: str, arguments: Mapping[str, Any], **_: Any
    ) -> Mapping[str, Any]:
        return {}

    async def read_resource(
        self, params: Mapping[str, Any], **_: Any
    ) -> Mapping[str, Any]:
        return {}

    async def get_prompt(
        self, params: Mapping[str, Any], **_: Any
    ) -> Mapping[str, Any]:
        return {}

    async def close(self) -> None:
        pass


def context(
    *, url: str = "https://mcp.example.com", transport: Mapping[str, Any] | None = None
) -> MCPTransportContext:
    async def elicit(request: Mapping[str, Any], signal: object) -> Mapping[str, Any]:
        return {"action": "decline"}

    return MCPTransportContext(
        server=MCPServerRow("server", "Server", url, ""),
        client_name="client",
        client_version="1.0.0",
        client_options={},
        transport_options=transport or {"type": "auto"},
        oauth_provider=None,
        elicit=elicit,
    )


@pytest.mark.asyncio
async def test_http_adapter_falls_back_to_sse_only_for_unsupported_transport() -> None:
    class Connector:
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def open(self, *, transport: str, context: object) -> EmptySession:
            self.calls.append(transport)
            if transport == "streamable-http":
                raise MCPTransportNotSupported
            return EmptySession()

    connector = Connector()
    session = await HTTPTransportAdapter(connector).open(context())
    assert isinstance(session, EmptySession)
    assert connector.calls == ["streamable-http", "sse"]


class Target:
    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []

    async def handle_mcp_message(
        self, message: dict[str, Any], signal: object
    ) -> object:
        self.messages.append(message)
        method = message.get("method")
        request_id = message.get("id")
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {"tools": {}, "resources": {}},
                    "instructions": "RPC",
                },
            }
        if method == "notifications/initialized":
            return None
        if method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "tools": [{"name": "count", "inputSchema": {"type": "object"}}]
                },
            }
        if method == "resources/list":
            return {"jsonrpc": "2.0", "id": request_id, "result": {"resources": []}}
        if method == "resources/templates/list":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"resourceTemplates": []},
            }
        if method == "tools/call":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"content": [{"type": "text", "text": "1"}]},
            }
        raise AssertionError(message)


@pytest.mark.asyncio
async def test_rpc_adapter_uses_standard_json_rpc_and_discovers_catalog() -> None:
    target = Target()

    class Resolver:
        def resolve(self, binding_name: str, name: str, props: object) -> Target:
            assert (binding_name, name, props) == ("MCP", "counter", {"tenant": "a"})
            return target

    session = await RPCTransportAdapter(Resolver()).open(
        context(
            url="rpc:counter",
            transport={
                "type": "rpc",
                "bindingName": "MCP",
                "props": {"tenant": "a"},
            },
        )
    )
    catalog = await session.discover()
    result = await session.call_tool("count", {})

    assert session.protocol_version == "2025-06-18"
    assert catalog.tools[0]["name"] == "count"
    assert catalog.instructions == "RPC"
    assert result == {"content": [{"type": "text", "text": "1"}]}
    assert target.messages[0]["jsonrpc"] == "2.0"
    assert target.messages[0]["method"] == "initialize"


@pytest.mark.asyncio
async def test_rpc_adapter_uses_typescript_camel_case_handler_without_signal() -> None:
    target = Target()

    class TypeScriptTarget:
        async def handleMcpMessage(self, message: dict[str, Any]) -> object:
            return await target.handle_mcp_message(message, None)

    class Resolver:
        def resolve(self, binding_name: str, name: str, props: object) -> object:
            return TypeScriptTarget()

    session = await RPCTransportAdapter(Resolver()).open(
        context(
            url="rpc:counter",
            transport={"type": "rpc", "bindingName": "MCP"},
        )
    )

    assert session.protocol_version == "2025-06-18"
    assert [message["method"] for message in target.messages] == [
        "initialize",
        "notifications/initialized",
    ]


@pytest.mark.asyncio
async def test_rpc_adapter_forwards_request_cancellation() -> None:
    entered = asyncio.Event()

    class BlockingTarget(Target):
        async def handle_mcp_message(
            self, message: dict[str, Any], signal: object
        ) -> object:
            if message.get("method") == "tools/call":
                self.messages.append(message)
                entered.set()
                await asyncio.Event().wait()
            if message.get("method") == "notifications/cancelled":
                self.messages.append(message)
                return None
            return await super().handle_mcp_message(message, signal)

    class Resolver:
        target = BlockingTarget()

        def resolve(self, binding_name: str, name: str, props: object) -> Target:
            return self.target

    class Signal:
        def __init__(self) -> None:
            self.event = asyncio.Event()

        @property
        def aborted(self) -> bool:
            return self.event.is_set()

        async def wait(self) -> None:
            await self.event.wait()

    resolver = Resolver()
    session = await RPCTransportAdapter(resolver).open(
        context(
            url="rpc:counter",
            transport={"type": "rpc", "bindingName": "MCP"},
        )
    )
    signal = Signal()
    pending = asyncio.create_task(session.call_tool("count", {}, signal=signal))
    await entered.wait()
    signal.event.set()

    with pytest.raises(asyncio.CancelledError):
        await pending
    assert resolver.target.messages[-1]["method"] == "notifications/cancelled"
    assert resolver.target.messages[-1]["params"]["requestId"] is not None


@pytest.mark.asyncio
async def test_rpc_adapter_cleans_up_when_caller_cancels_request() -> None:
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    class BlockingTarget(Target):
        async def handle_mcp_message(
            self, message: dict[str, Any], signal: object
        ) -> object:
            if message.get("method") == "tools/call":
                self.messages.append(message)
                entered.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.set()
            if message.get("method") == "notifications/cancelled":
                self.messages.append(message)
                return None
            return await super().handle_mcp_message(message, signal)

    class Resolver:
        target = BlockingTarget()

        def resolve(self, binding_name: str, name: str, props: object) -> Target:
            return self.target

    class Signal:
        aborted = False

        async def wait(self) -> None:
            await asyncio.Event().wait()

    resolver = Resolver()
    session = await RPCTransportAdapter(resolver).open(
        context(
            url="rpc:counter",
            transport={"type": "rpc", "bindingName": "MCP"},
        )
    )
    pending = asyncio.create_task(session.call_tool("count", {}, signal=Signal()))
    await entered.wait()
    pending.cancel()

    with pytest.raises(asyncio.CancelledError):
        await pending
    assert cancelled.is_set()
    assert resolver.target.messages[-1]["method"] == "notifications/cancelled"
