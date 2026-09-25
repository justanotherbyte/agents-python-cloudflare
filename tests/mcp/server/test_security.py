from __future__ import annotations

from dataclasses import dataclass

import pytest
from mcp.server import MCPServer
from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.auth.provider import AccessToken
from mcp.server.mcpserver import Context

from agents.mcp.server import (
    CORSOptions,
    MCPAuthContext,
    MCPHandlerOptions,
    VerifiedMCPAuth,
    create_mcp_handler,
    get_mcp_auth_context,
)

from .conftest import ASGITestRuntime, Request, modern_request


def factory(_context):
    return MCPServer("security")


@pytest.mark.asyncio
async def test_localhost_defaults_reject_rebound_host_and_foreign_origin():
    handler = create_mcp_handler(factory, _runtime=ASGITestRuntime())
    request = modern_request("server/discover")
    request.url = "http://localhost/mcp"
    request.headers["host"] = "evil.example"

    host = await handler.fetch(request)
    request.headers["host"] = "localhost:8787"
    request.headers["origin"] = "https://evil.example"
    origin = await handler.fetch(request)

    assert host.status == 403
    assert host.headers["content-type"] == "application/json"
    assert origin.status == 403
    assert "Invalid Host" in host.body
    assert "Invalid Origin" in origin.body


@pytest.mark.asyncio
async def test_localhost_and_workers_dev_defaults_accept_their_endpoint():
    runtime = ASGITestRuntime()
    handler = create_mcp_handler(factory, _runtime=runtime)
    local = modern_request("server/discover")
    local.url = "http://localhost/mcp"
    local.headers.update({"host": "localhost:8787", "origin": "http://localhost:3000"})
    worker = modern_request("server/discover")
    worker.url = "https://server.account.workers.dev/mcp"
    worker.headers.update(
        {
            "host": "server.account.workers.dev",
            "origin": "https://server.account.workers.dev",
        }
    )

    assert (await handler.fetch(local)).status == 200
    assert (await handler.fetch(worker)).status == 200


@pytest.mark.asyncio
async def test_bracketed_ipv6_host_is_accepted_and_malformed_authorities_are_rejected():
    handler = create_mcp_handler(factory, _runtime=ASGITestRuntime())
    request = modern_request("server/discover")
    request.url = "http://[::1]/mcp"
    request.headers["host"] = "[::1]:8787"

    assert (await handler.fetch(request)).status == 200

    for authority in (
        "::1",
        "[::1",
        "[::1]suffix",
        "[::1]:invalid",
        "[::1]:65536",
        f"[::1]:{'9' * 5000}",
        "localhost,evil.example",
        "user@localhost",
        " localhost",
    ):
        request.headers["host"] = authority
        assert (await handler.fetch(request)).status == 403


@pytest.mark.asyncio
async def test_explicit_host_and_origin_allowlists_compare_hostnames():
    runtime = ASGITestRuntime()
    handler = create_mcp_handler(
        factory,
        MCPHandlerOptions(
            allowed_hostnames=("mcp.example.com",),
            allowed_origin_hostnames=("client.example.com",),
        ),
        _runtime=runtime,
    )
    accepted = modern_request("server/discover")
    accepted.url = "https://mcp.example.com/mcp"
    accepted.headers.update(
        {
            "host": "mcp.example.com:443",
            "origin": "https://client.example.com:8443",
        }
    )

    assert (await handler.fetch(accepted)).status == 200
    accepted.headers["origin"] = "null"
    assert (await handler.fetch(accepted)).status == 403
    accepted.headers["origin"] = "https://other.example.com"
    assert (await handler.fetch(accepted)).status == 403


@pytest.mark.asyncio
async def test_non_browser_client_without_origin_is_valid():
    handler = create_mcp_handler(
        factory,
        MCPHandlerOptions(allowed_origin_hostnames=("client.example",)),
        _runtime=ASGITestRuntime(),
    )

    assert (await handler.fetch(modern_request("server/discover"))).status == 200


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "origin", ["null", "not a url", "https://[::1", "https://any.example"]
)
async def test_origin_validation_can_be_explicitly_disabled(origin):
    handler = create_mcp_handler(
        factory,
        MCPHandlerOptions(allowed_origin_hostnames="*"),
        _runtime=ASGITestRuntime(),
    )
    request = modern_request("server/discover")
    request.headers["origin"] = origin

    assert (await handler.fetch(request)).status == 200


def test_disabled_origin_validation_still_validates_host_allowlist():
    with pytest.raises(ValueError, match="hostname without a scheme or port"):
        create_mcp_handler(
            factory,
            MCPHandlerOptions(
                allowed_hostnames=("https://mcp.example.com",),
                allowed_origin_hostnames="*",
            ),
        )


@pytest.mark.asyncio
async def test_concrete_cors_origin_extends_default_origin_allowlist():
    handler = create_mcp_handler(
        factory,
        MCPHandlerOptions(cors=CORSOptions(origin="https://app.example.com")),
        _runtime=ASGITestRuntime(),
    )
    request = modern_request("server/discover")
    request.headers["origin"] = "https://app.example.com:8443"

    assert (await handler.fetch(request)).status == 200


@dataclass
class Verifier:
    result: object
    seen: object = None

    async def verify(self, execution_context):
        self.seen = execution_context
        return self.result


@pytest.mark.asyncio
async def test_verified_auth_reaches_factory_official_context_and_full_stream():
    props = {"user_id": "user-1"}
    access_token = AccessToken(
        token="secret-token",
        client_id="client-1",
        scopes=["read"],
        subject="user-1",
        claims={"iss": "https://issuer.example"},
    )
    verifier = Verifier(VerifiedMCPAuth(access_token=access_token, props=props))
    execution_context = object()
    seen = []

    def auth_factory(context):
        seen.append(("factory", context.auth_info, get_mcp_auth_context()))
        server = MCPServer("auth")

        @server.tool()
        async def inspect_auth(ctx: Context) -> str:
            seen.append(("before", get_mcp_auth_context(), get_access_token()))
            await ctx.report_progress(1, total=2)
            seen.append(("after", get_mcp_auth_context(), get_access_token()))
            request = ctx.request_context.request
            seen.append(("user", request.scope["user"].access_token))
            return "ok"

        return server

    request = modern_request(
        "tools/call",
        {
            "name": "inspect_auth",
            "arguments": {},
            "_meta": {"progressToken": "progress-1"},
        },
    )
    handler = create_mcp_handler(
        auth_factory,
        MCPHandlerOptions(auth_verifier=verifier),
        _runtime=ASGITestRuntime(),
    )

    response = await handler.fetch(request, execution_context=execution_context)

    assert response.status == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert b"notifications/progress" in response.body
    assert verifier.seen is execution_context
    assert seen[0][1] is access_token
    assert seen[0][2].props is props
    assert all(item[1].props is props for item in seen[1:3])
    assert all(item[2] is access_token for item in seen[1:3])
    assert seen[3][1] is access_token
    assert get_mcp_auth_context() is None
    assert get_access_token() is None


@pytest.mark.asyncio
async def test_static_trusted_auth_context_is_scoped_and_reset():
    props = {"tenant": "a"}
    seen = []

    def context_factory(_context):
        seen.append(get_mcp_auth_context())
        return MCPServer("context")

    handler = create_mcp_handler(
        context_factory,
        MCPHandlerOptions(auth_context=MCPAuthContext(props=props)),
        _runtime=ASGITestRuntime(),
    )

    assert (await handler.fetch(modern_request("server/discover"))).status == 200
    assert seen[0].props is props
    assert get_mcp_auth_context() is None


@pytest.mark.asyncio
async def test_invalid_verified_auth_fails_closed_before_factory():
    called = False

    def guarded_factory(_context):
        nonlocal called
        called = True
        return MCPServer("never")

    handler = create_mcp_handler(
        guarded_factory,
        MCPHandlerOptions(auth_verifier=Verifier({"token": "untrusted"})),
        _runtime=ASGITestRuntime(),
    )

    response = await handler.fetch(
        modern_request("server/discover"), execution_context=object()
    )

    assert response.status == 500
    assert called is False
    assert "untrusted" not in response.body
