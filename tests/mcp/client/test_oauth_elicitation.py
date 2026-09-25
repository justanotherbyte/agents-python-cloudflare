from __future__ import annotations

import asyncio
import copy
import sqlite3
from collections.abc import Mapping, Sequence
from typing import Any

import pytest
from workers import Request

from agents.mcp.client import (
    DurableOAuthProvider,
    MCPAuthorizationRequired,
    MCPClientManager,
    MCPCatalog,
    MCPElicitationHandlers,
    MCPIsolateLostError,
    MCPOAuthCallbackPolicy,
    MCPTransportContext,
    OAUTH_STATE_TTL_MS,
    decode_server_options,
)


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
            key: value for key, value in self.values.items() if key.startswith(prefix)
        }


class Sql:
    def __init__(self) -> None:
        self.connection = sqlite3.connect(":memory:")
        self.connection.row_factory = sqlite3.Row

    def execute(self, query: str, *params: object) -> list[dict[str, Any]]:
        cursor = self.connection.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]


class Session:
    session_id = None
    protocol_version = "2025-06-18"

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


class OAuthFactory:
    def __init__(self) -> None:
        self.calls = 0
        self.context: MCPTransportContext | None = None

    async def open(self, context: MCPTransportContext) -> Session:
        self.calls += 1
        self.context = context
        if context.authorization_params is None:
            raise MCPAuthorizationRequired(
                "https://auth.example.com/authorize", "oauth-client"
            )
        assert context.authorization_params["code"] == "accepted"
        return Session()


async def manager_with(factory: object, *, runner: object = None) -> MCPClientManager:
    manager = MCPClientManager(
        "client",
        "1.0.0",
        storage=Storage(),
        sql=Sql(),
        transports={"auto": factory},
        run_in_host_context=runner,
        sleep=lambda _delay: asyncio.sleep(0),
    )
    await manager.on_start()
    return manager


@pytest.mark.asyncio
async def test_oauth_state_callback_policy_and_stable_id_migration() -> None:
    factory = OAuthFactory()
    manager = await manager_with(factory)
    await manager.register_server(
        "old",
        url="https://mcp.example.com",
        name="OAuth",
        callback_url="https://agent.example.com/oauth/callback",
        retry={"maxAttempts": 1},
    )
    await manager.establish_connection("old")
    provider = manager.mcp_connections["old"].oauth_provider
    assert isinstance(provider, DurableOAuthProvider)
    state = await provider.state()
    storage = provider.storage
    await storage.put("/client/old/custom", {"kept": True})

    request = Request(
        f"https://agent.example.com/oauth/callback?code=accepted&state={state}"
    )
    assert manager.is_callback_request(request) is True
    result = await manager.handle_callback_request(request)
    assert result.auth_success is True
    assert manager.mcp_connections["old"].state == "ready"

    await manager.migrate_server_id("old", "Stable ID")
    assert "stable-id" in manager.mcp_connections
    assert manager.mcp_connections["stable-id"].oauth_provider.server_id == "stable-id"
    assert (await storage.get("/client/stable-id/custom")) == {"kept": True}
    assert await storage.get("/client/old/custom") is None

    manager.configure_oauth_callback(MCPOAuthCallbackPolicy(error_redirect="/failed"))
    response = manager.oauth_callback_response(
        result.__class__(False, "stable-id", "bad\nvalue"),
        "https://agent.example.com/oauth/callback",
    )
    assert response.status == 302
    assert (
        response.headers["location"]
        == "https://agent.example.com/failed?error=bad+value"
    )


@pytest.mark.asyncio
async def test_oauth_state_is_one_use_and_expires() -> None:
    now = [1_000]
    storage = Storage()
    provider = DurableOAuthProvider(
        storage,
        "client",
        "https://agent.example.com/callback",
        server_id="server",
        clock_ms=lambda: now[0],
    )
    state = await provider.state()
    assert await provider.check_state(state) == (True, None)
    await provider.consume_state(state)
    assert await provider.check_state(state) == (
        False,
        "State not found or already used",
    )
    expired_state = await provider.state()
    now[0] += OAUTH_STATE_TTL_MS + 1
    assert await provider.check_state(expired_state) == (False, "State expired")


@pytest.mark.asyncio
async def test_elicitation_runs_in_host_context_and_disposal_rejects_it() -> None:
    entered = asyncio.Event()
    host_context_calls = 0

    async def runner(callback: object) -> object:
        nonlocal host_context_calls
        host_context_calls += 1
        return await callback()

    async def form(
        request: object, server_id: str, signal: object
    ) -> Mapping[str, Any]:
        assert server_id == "server"
        entered.set()
        await asyncio.Event().wait()
        return {"action": "accept"}

    async def url(request: object, server_id: str, signal: object) -> Mapping[str, Any]:
        assert server_id == "server"
        return {"action": "accept"}

    class Factory:
        context: MCPTransportContext | None = None

        async def open(self, context: MCPTransportContext) -> Session:
            self.context = context
            return Session()

    factory = Factory()
    manager = await manager_with(factory, runner=runner)
    manager.configure_elicitation_handlers(MCPElicitationHandlers(form=form, url=url))
    await manager.register_server(
        "server", url="https://mcp.example.com", name="Server"
    )
    await manager.establish_connection("server")
    assert factory.context is not None
    assert await factory.context.elicit(
        {"method": "elicitation/create", "params": {"mode": "url"}}, None
    ) == {"action": "accept"}
    persisted = decode_server_options(manager.list_servers()[0].server_options)
    assert persisted["capabilities"] == {"elicitation": {"form": {}, "url": {}}}

    pending = asyncio.create_task(
        factory.context.elicit(
            {"method": "elicitation/create", "params": {"mode": "form"}}, None
        )
    )
    await entered.wait()
    await manager.dispose()
    with pytest.raises(MCPIsolateLostError, match="retry"):
        await pending
    assert host_context_calls == 2
