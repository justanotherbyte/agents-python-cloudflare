from __future__ import annotations

import asyncio
import copy
import sqlite3
from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from agents.mcp.client import (
    CURRENT_MCP_SCHEMA_VERSION,
    MCP_SCHEMA_VERSION_KEY,
    MCPAIToolDescriptor,
    MCPClientManager,
    MCPCatalog,
    MCPConnectionState,
    MCPServerFilter,
    MCPServerRow,
    MCPServerStore,
    MCPTransportContext,
    encode_server_options,
)
from agents.lifecycle import Lifecycle
from fakes import FakeCtx


class Storage:
    def __init__(self) -> None:
        self.values: dict[str, Any] = {}

    async def get(self, key: str | Sequence[str]) -> Any:
        if isinstance(key, str):
            return copy.deepcopy(self.values.get(key))
        return {
            item: copy.deepcopy(self.values[item])
            for item in key
            if item in self.values
        }

    async def put(self, key: str | dict[str, Any], value: object = None) -> None:
        if isinstance(key, dict):
            self.values.update(copy.deepcopy(key))
        else:
            self.values[key] = copy.deepcopy(value)

    async def delete(self, key: str | Sequence[str]) -> bool | int:
        if isinstance(key, str):
            return self.values.pop(key, None) is not None
        return sum(self.values.pop(item, None) is not None for item in key)

    async def list(self, *, prefix: str = "") -> dict[str, Any]:
        return {
            key: copy.deepcopy(value)
            for key, value in self.values.items()
            if key.startswith(prefix)
        }


class Sql:
    def __init__(self) -> None:
        self.connection = sqlite3.connect(":memory:")
        self.connection.row_factory = sqlite3.Row

    def execute(self, query: str, *params: object) -> list[dict[str, Any]]:
        cursor = self.connection.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]


class Session:
    session_id = "session-1"
    protocol_version = "2025-06-18"

    def __init__(self, catalog: MCPCatalog | None = None) -> None:
        self.catalog = catalog or MCPCatalog()
        self.closed = False
        self.tool_calls: list[tuple[str, dict[str, Any]]] = []

    async def discover(self) -> MCPCatalog:
        return self.catalog

    async def call_tool(
        self, name: str, arguments: Mapping[str, Any], *, signal: object = None
    ) -> Mapping[str, Any]:
        self.tool_calls.append((name, dict(arguments)))
        return {"content": [{"type": "text", "text": "ok"}]}

    async def read_resource(
        self, params: Mapping[str, Any], *, signal: object = None
    ) -> Mapping[str, Any]:
        return {"contents": [dict(params)]}

    async def get_prompt(
        self, params: Mapping[str, Any], *, signal: object = None
    ) -> Mapping[str, Any]:
        return {"messages": [dict(params)]}

    async def close(self) -> None:
        self.closed = True


class Factory:
    def __init__(self, session: Session, *, failures: int = 0) -> None:
        self.session = session
        self.failures = failures
        self.calls = 0
        self.contexts: list[MCPTransportContext] = []
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.block = False

    async def open(self, context: MCPTransportContext) -> Session:
        self.calls += 1
        self.contexts.append(context)
        self.entered.set()
        if self.block:
            await self.release.wait()
        if self.calls <= self.failures:
            raise RuntimeError(f"attempt {self.calls}\nfailed")
        return self.session


async def make_manager(
    factory: Factory, *, storage: Storage | None = None, sql: Sql | None = None
) -> tuple[MCPClientManager, Storage, Sql]:
    storage = storage or Storage()
    sql = sql or Sql()
    manager = MCPClientManager(
        "test-client",
        "1.0.0",
        storage=storage,
        sql=sql,
        transports={"auto": factory, "rpc": factory},
        sleep=lambda _delay: asyncio.sleep(0),
    )
    await manager.on_start()
    return manager, storage, sql


@pytest.mark.asyncio
async def test_connect_deduplicates_retries_and_projects_catalog_and_tools() -> None:
    catalog = MCPCatalog(
        tools=[
            {
                "name": "search",
                "description": "Search",
                "annotations": {"title": "Web search"},
                "inputSchema": {
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                },
            }
        ],
        prompts=[{"name": "review"}],
        resources=[{"uri": "file:///readme"}],
        resource_templates=[{"uriTemplate": "file:///{path}"}],
        capabilities={"tools": {}, "prompts": {}, "resources": {}},
        instructions="Be concise",
    )
    factory = Factory(Session(catalog), failures=2)
    manager, storage, _ = await make_manager(factory)
    changes = 0

    def changed() -> None:
        nonlocal changes
        changes += 1

    manager.add_state_listener(changed)
    server_id = await manager.register_server(
        "Search Server",
        url="https://mcp.example.com",
        name="Search",
        retry={"maxAttempts": 3, "baseDelayMs": 1, "maxDelayMs": 1},
    )
    first = asyncio.create_task(manager.establish_connection(server_id))
    second = asyncio.create_task(manager.establish_connection(server_id))
    await asyncio.gather(first, second)

    assert factory.calls == 3
    assert manager.mcp_connections[server_id].state == MCPConnectionState.READY
    assert changes > 0
    assert storage.values[MCP_SCHEMA_VERSION_KEY] == CURRENT_MCP_SCHEMA_VERSION
    assert manager.list_tools() == [{**catalog.tools[0], "serverId": server_id}]
    assert manager.list_resource_templates() == [
        {**catalog.resource_templates[0], "serverId": server_id}
    ]
    assert manager.list_tools(MCPServerFilter(server_name="other")) == []

    body = manager.get_mcp_servers()
    assert body["servers"][server_id] == {
        "name": "Search",
        "server_url": "https://mcp.example.com",
        "auth_url": None,
        "state": "ready",
        "error": None,
        "instructions": "Be concise",
        "capabilities": catalog.capabilities,
    }
    assert "resourceTemplates" not in body
    assert manager.get_catalog_frame() == {
        "type": "cf_agent_mcp_servers",
        "mcp": body,
    }

    tools = manager.get_ai_tools()
    descriptor = tools[f"tool_{server_id.replace('-', '')}_search"]
    assert isinstance(descriptor, MCPAIToolDescriptor)
    assert descriptor.title == "Web search"
    assert await descriptor.execute({"q": "workers"}) == {
        "content": [{"type": "text", "text": "ok"}]
    }
    assert factory.session.tool_calls == [("search", {"q": "workers"})]

    await manager.dispose()
    assert factory.session.closed is True


@pytest.mark.asyncio
async def test_cold_start_restores_rows_session_options_and_catalog() -> None:
    storage = Storage()
    sql = Sql()
    store = MCPServerStore(sql)
    store.prepare()
    await storage.put(MCP_SCHEMA_VERSION_KEY, CURRENT_MCP_SCHEMA_VERSION)
    store.save(
        MCPServerRow(
            "saved",
            "Saved",
            "https://saved.example.com",
            "",
            server_options=encode_server_options(
                {
                    "transport": {
                        "type": "streamable-http",
                        "sessionId": "saved-session",
                        "protocolVersion": "2025-06-18",
                    },
                    "retry": {"maxAttempts": 1},
                }
            ),
        )
    )
    factory = Factory(Session(MCPCatalog(tools=[{"name": "restored"}])))

    manager, _, _ = await make_manager(factory, storage=storage, sql=sql)

    assert factory.calls == 1
    assert factory.contexts[0].transport_options["sessionId"] == "saved-session"
    assert manager.list_tools() == [{"name": "restored", "serverId": "saved"}]


@pytest.mark.asyncio
async def test_concurrent_connection_work_uses_one_transport_open() -> None:
    factory = Factory(Session())
    factory.block = True
    manager, _, _ = await make_manager(factory)
    await manager.register_server(
        "one", url="https://one.example.com", name="One", retry={"maxAttempts": 1}
    )

    first = asyncio.create_task(manager.establish_connection("one"))
    await factory.entered.wait()
    second = asyncio.create_task(manager.establish_connection("one"))
    await asyncio.sleep(0)
    assert factory.calls == 1
    factory.release.set()
    await asyncio.gather(first, second)
    assert factory.calls == 1


@pytest.mark.asyncio
async def test_manager_starts_as_a_lifecycle_capability() -> None:
    ctx = FakeCtx()
    manager = MCPClientManager("client", "1.0.0")
    lifecycle = Lifecycle(ctx, host=object())
    lifecycle.use(manager)

    await lifecycle.start()

    assert manager.capability_id == "mcp"
    assert manager.list_servers() == ()
    assert await ctx.storage.get(MCP_SCHEMA_VERSION_KEY) == CURRENT_MCP_SCHEMA_VERSION
