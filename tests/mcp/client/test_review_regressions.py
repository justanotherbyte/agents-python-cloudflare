from __future__ import annotations

import asyncio
import copy
import sqlite3
from collections.abc import Mapping, Sequence
from typing import Any

import pytest
from workers import Request

from agents.mcp.client import (
    CURRENT_MCP_SCHEMA_VERSION,
    MCP_SCHEMA_VERSION_KEY,
    DurableOAuthProvider,
    MCPAuthorizationRequired,
    MCPClientManager,
    MCPCatalog,
    MCPConnectionState,
    MCPError,
    MCPServerRow,
    MCPServerStore,
    MCPStaleSessionError,
    MCPTransportContext,
    RPCTransportAdapter,
    encode_server_options,
)
from agents.mcp.client.oauth import OfficialOAuthTokenStorage


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
    session_id = "session"
    protocol_version = "2025-06-18"

    def __init__(self, catalog: MCPCatalog | None = None) -> None:
        self.catalog = catalog or MCPCatalog()
        self.closed = False

    async def discover(self) -> MCPCatalog:
        return self.catalog

    async def call_tool(self, *_: Any, **__: Any) -> Mapping[str, Any]:
        return {}

    async def read_resource(self, *_: Any, **__: Any) -> Mapping[str, Any]:
        return {}

    async def get_prompt(self, *_: Any, **__: Any) -> Mapping[str, Any]:
        return {}

    async def close(self) -> None:
        self.closed = True


def stored() -> tuple[Storage, Sql, MCPServerStore]:
    storage = Storage()
    sql = Sql()
    store = MCPServerStore(sql)
    store.prepare()
    storage.values[MCP_SCHEMA_VERSION_KEY] = CURRENT_MCP_SCHEMA_VERSION
    return storage, sql, store


@pytest.mark.asyncio
async def test_restore_isolates_bad_options_and_bounds_concurrency() -> None:
    storage, sql, store = stored()
    store.save(
        MCPServerRow("bad", "Bad", "https://bad.example.com", "", server_options="[")
    )
    for index in range(4):
        store.save(
            MCPServerRow(
                f"good-{index}",
                f"Good {index}",
                f"https://good-{index}.example.com",
                "",
                server_options=encode_server_options({"retry": {"maxAttempts": 1}}),
            )
        )

    class Factory:
        active = 0
        maximum = 0

        async def open(self, context: MCPTransportContext) -> Session:
            self.active += 1
            self.maximum = max(self.maximum, self.active)
            await asyncio.sleep(0.01)
            self.active -= 1
            return Session()

    factory = Factory()
    manager = MCPClientManager(
        "client",
        "1",
        storage=storage,
        sql=sql,
        transports={"auto": factory},
        restore_concurrency=2,
    )
    await manager.on_start()

    assert manager.get_connection("bad").state == MCPConnectionState.FAILED
    assert all(
        manager.get_connection(f"good-{index}").state == MCPConnectionState.READY
        for index in range(4)
    )
    assert factory.maximum == 2


@pytest.mark.asyncio
async def test_replacement_fences_a_cancelled_connection_result() -> None:
    old = Session()
    new = Session()
    entered = asyncio.Event()

    class Factory:
        async def open(self, context: MCPTransportContext) -> Session:
            if context.server.name == "Old":
                entered.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    return old
            return new

    storage, sql, _ = stored()
    manager = MCPClientManager(
        "client", "1", storage=storage, sql=sql, transports={"auto": Factory()}
    )
    await manager.on_start()
    await manager.register_server("same", url="https://old.example.com", name="Old")
    connecting = asyncio.create_task(manager.establish_connection("same"))
    await entered.wait()

    await manager.register_server("same", url="https://new.example.com", name="New")
    await connecting
    await manager.establish_connection("same")

    connection = manager.get_connection("same")
    assert connection.server.name == "New"
    assert connection.session is new
    assert connection.state == MCPConnectionState.READY
    assert old.closed is True


@pytest.mark.asyncio
async def test_stale_restored_session_is_cleared_and_reconnected_once() -> None:
    class StaleSession(Session):
        async def discover(self) -> MCPCatalog:
            raise MCPStaleSessionError("gone")

    storage, sql, store = stored()
    store.save(
        MCPServerRow(
            "saved",
            "Saved",
            "https://saved.example.com",
            "",
            server_options=encode_server_options(
                {
                    "transport": {
                        "sessionId": "old-session",
                        "protocolVersion": "2025-06-18",
                    },
                    "discoverResult": {"capabilities": {"tools": {}}},
                    "retry": {"maxAttempts": 1},
                }
            ),
        )
    )

    class Factory:
        contexts: list[MCPTransportContext] = []
        sessions = [StaleSession(), Session(MCPCatalog(tools=[{"name": "ok"}]))]

        async def open(self, context: MCPTransportContext) -> Session:
            self.contexts.append(context)
            return self.sessions[len(self.contexts) - 1]

    factory = Factory()
    manager = MCPClientManager(
        "client", "1", storage=storage, sql=sql, transports={"auto": factory}
    )
    await manager.on_start()

    assert factory.contexts[0].transport_options["sessionId"] == "old-session"
    assert "sessionId" not in factory.contexts[1].transport_options
    assert factory.sessions[0].closed is True
    assert manager.get_connection("saved").state == MCPConnectionState.READY


@pytest.mark.asyncio
async def test_stale_session_during_open_is_retried_without_the_session_id() -> None:
    storage, sql, store = stored()
    store.save(
        MCPServerRow(
            "saved",
            "Saved",
            "https://saved.example.com",
            "",
            server_options=encode_server_options(
                {
                    "transport": {
                        "sessionId": "old-session",
                        "protocolVersion": "2025-06-18",
                    },
                    "retry": {"maxAttempts": 1},
                }
            ),
        )
    )

    class Factory:
        contexts: list[MCPTransportContext] = []

        async def open(self, context: MCPTransportContext) -> Session:
            self.contexts.append(context)
            if len(self.contexts) == 1:
                raise MCPStaleSessionError("gone")
            return Session()

    factory = Factory()
    manager = MCPClientManager(
        "client", "1", storage=storage, sql=sql, transports={"auto": factory}
    )
    await manager.on_start()

    assert len(factory.contexts) == 2
    assert "sessionId" not in factory.contexts[1].transport_options
    assert manager.get_connection("saved").state == MCPConnectionState.READY


@pytest.mark.asyncio
async def test_oauth_callback_requires_registered_url_and_cleans_pkce() -> None:
    class Factory:
        async def open(self, context: MCPTransportContext) -> Session:
            return Session()

    storage, sql, _ = stored()
    manager = MCPClientManager(
        "client", "1", storage=storage, sql=sql, transports={"auto": Factory()}
    )
    await manager.on_start()
    await manager.register_server(
        "oauth",
        url="https://mcp.example.com",
        name="OAuth",
        callback_url="https://agent.example.com/callback",
    )
    provider = manager.get_connection("oauth").oauth_provider
    assert isinstance(provider, DurableOAuthProvider)
    state = await provider.state()
    await provider.save_code_verifier(state, "verifier")

    wrong = await manager.handle_callback_request(
        Request(f"https://evil.example.com/callback?code=ok&state={state}")
    )
    assert wrong.auth_success is False
    assert await provider.check_state(state) == (True, None)

    result = await manager.handle_callback_request(
        Request(f"https://agent.example.com/callback?code=ok&state={state}")
    )
    assert result.auth_success is True
    assert (await provider.check_state(state))[0] is False
    with pytest.raises(ValueError, match="verifier"):
        await provider.code_verifier(state)


@pytest.mark.asyncio
async def test_official_oauth_storage_round_trips_sdk_models() -> None:
    from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

    provider = DurableOAuthProvider(
        Storage(),
        "client",
        "https://agent.example.com/callback",
        server_id="server",
        client_id="oauth-client",
    )
    adapter = OfficialOAuthTokenStorage(provider)
    client = OAuthClientInformationFull(client_id="oauth-client")
    token = OAuthToken(access_token="secret", scope="tools")

    await adapter.set_client_info(client)
    await adapter.set_tokens(token)

    assert await adapter.get_client_info() == client
    assert await adapter.get_tokens() == token


@pytest.mark.asyncio
async def test_rpc_catalog_rejects_a_repeated_cursor() -> None:
    class Target:
        async def handle_mcp_message(
            self, message: Mapping[str, Any], signal: object
        ) -> object:
            method = message.get("method")
            request_id = message.get("id")
            if method == "initialize":
                result = {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "server", "version": "1"},
                }
            elif method == "notifications/initialized":
                return None
            elif method == "tools/list":
                result = {"tools": [], "nextCursor": "same"}
            else:
                raise AssertionError(message)
            return {"jsonrpc": "2.0", "id": request_id, "result": result}

    class Resolver:
        def resolve(self, *_: object) -> Target:
            return Target()

    async def elicit(*_: object) -> Mapping[str, Any]:
        return {"action": "decline"}

    session = await RPCTransportAdapter(Resolver()).open(
        MCPTransportContext(
            server=MCPServerRow("rpc", "RPC", "rpc:server", ""),
            client_name="client",
            client_version="1",
            client_options={"listMaxPages": 10},
            transport_options={"type": "rpc", "bindingName": "MCP"},
            oauth_provider=None,
            elicit=elicit,
        )
    )
    with pytest.raises(Exception, match="repeated pagination cursor"):
        await session.discover()


@pytest.mark.asyncio
async def test_rpc_elicitation_continuations_are_bounded() -> None:
    class Target:
        continuation = 0

        async def handle_mcp_message(
            self, message: Mapping[str, Any], signal: object
        ) -> object:
            method = message.get("method")
            request_id = message.get("id")
            if method == "initialize":
                result = {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {"tools": {}},
                }
                return {"jsonrpc": "2.0", "id": request_id, "result": result}
            if method == "notifications/initialized":
                return None
            if method == "tools/call" or "result" in message:
                self.continuation += 1
                return {
                    "jsonrpc": "2.0",
                    "id": f"elicit-{self.continuation}",
                    "method": "elicitation/create",
                    "params": {"mode": "form"},
                }
            raise AssertionError(message)

    target = Target()

    class Resolver:
        def resolve(self, *_: object) -> Target:
            return target

    async def elicit(*_: object) -> Mapping[str, Any]:
        return {"action": "decline"}

    session = await RPCTransportAdapter(Resolver()).open(
        MCPTransportContext(
            server=MCPServerRow("rpc", "RPC", "rpc:server", ""),
            client_name="client",
            client_version="1",
            client_options={},
            transport_options={"type": "rpc", "bindingName": "MCP"},
            oauth_provider=None,
            elicit=elicit,
        )
    )

    with pytest.raises(MCPError, match="continuation limit"):
        await session.call_tool("loop", {})


@pytest.mark.asyncio
async def test_rpc_cancellation_does_not_wait_for_a_stubborn_handler() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    class Target:
        async def handle_mcp_message(
            self, message: Mapping[str, Any], signal: object
        ) -> object:
            method = message.get("method")
            request_id = message.get("id")
            if method == "initialize":
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "protocolVersion": "2025-06-18",
                        "capabilities": {"tools": {}},
                    },
                }
            if method == "notifications/initialized":
                return None
            if method == "notifications/cancelled":
                return None
            if method == "tools/call":
                entered.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    await release.wait()
                return {"jsonrpc": "2.0", "id": request_id, "result": {}}
            raise AssertionError(message)

    class Resolver:
        def resolve(self, *_: object) -> Target:
            return Target()

    class Signal:
        event = asyncio.Event()

        @property
        def aborted(self) -> bool:
            return self.event.is_set()

        async def wait(self) -> None:
            await self.event.wait()

    async def elicit(*_: object) -> Mapping[str, Any]:
        return {"action": "decline"}

    session = await RPCTransportAdapter(Resolver()).open(
        MCPTransportContext(
            server=MCPServerRow("rpc", "RPC", "rpc:server", ""),
            client_name="client",
            client_version="1",
            client_options={},
            transport_options={"type": "rpc", "bindingName": "MCP"},
            oauth_provider=None,
            elicit=elicit,
        )
    )
    signal = Signal()
    pending = asyncio.create_task(session.call_tool("slow", {}, signal=signal))
    await entered.wait()
    signal.event.set()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(pending, 0.5)
    release.set()
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_connect_timeout_is_terminal_for_the_attempt() -> None:
    class Factory:
        async def open(self, context: MCPTransportContext) -> Session:
            await asyncio.Event().wait()
            return Session()

    storage, sql, _ = stored()
    manager = MCPClientManager(
        "client",
        "1",
        storage=storage,
        sql=sql,
        transports={"auto": Factory()},
        connect_timeout_ms=5,
    )
    await manager.on_start()
    await manager.register_server(
        "slow",
        url="https://slow.example.com",
        name="Slow",
        retry={"maxAttempts": 1},
    )

    await manager.establish_connection("slow")

    assert manager.get_connection("slow").state == MCPConnectionState.FAILED


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "http://169.254.169.254/mcp",
        "http://[::ffff:10.0.0.1]/mcp",
        "http://2130706433/mcp",
        "https://service.internal./mcp",
        "https://user:password@example.com/mcp",
    ],
)
async def test_registration_rejects_internal_or_ambiguous_urls(url: str) -> None:
    storage, sql, _ = stored()
    manager = MCPClientManager("client", "1", storage=storage, sql=sql)
    await manager.on_start()

    with pytest.raises(ValueError):
        await manager.register_server("blocked", url=url, name="Blocked")


@pytest.mark.asyncio
async def test_scope_throw_cleans_generated_oauth_state_and_verifier() -> None:
    class ScopeSession(Session):
        auth_url = ""

        async def call_tool(self, *_: Any, **__: Any) -> Mapping[str, Any]:
            raise MCPAuthorizationRequired(
                self.auth_url, "oauth-client", scope_step_up=True
            )

    session = ScopeSession()

    class Factory:
        async def open(self, context: MCPTransportContext) -> Session:
            return session

    storage, sql, _ = stored()
    manager = MCPClientManager(
        "client", "1", storage=storage, sql=sql, transports={"auto": Factory()}
    )
    await manager.on_start()
    await manager.register_server(
        "scope",
        url="https://mcp.example.com",
        name="Scope",
        callback_url="https://agent.example.com/callback",
        transport={"onInsufficientScope": "throw"},
    )
    await manager.establish_connection("scope")
    provider = manager.get_connection("scope").oauth_provider
    assert isinstance(provider, DurableOAuthProvider)
    provider.client_id = "oauth-client"
    state = await provider.state()
    await provider.save_code_verifier(state, "verifier")
    session.auth_url = f"https://auth.example.com/authorize?state={state}"

    with pytest.raises(MCPError, match="additional OAuth scope"):
        await manager.call_tool("scope", "tool", {})

    assert manager.get_connection("scope").state == MCPConnectionState.READY
    assert (await provider.check_state(state))[0] is False
    with pytest.raises(ValueError, match="verifier"):
        await provider.code_verifier(state)
