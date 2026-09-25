from __future__ import annotations

import ast
import json
import sys
import types
from pathlib import Path

import pytest
from mcp.server import MCPServer
from mcp.types import Completion

from agents.mcp.server import (
    CORSOptions,
    MCPHandlerOptions,
    create_mcp_handler,
)

from .conftest import ASGITestRuntime, AbortSignal, Request, modern_request


def server_factory(_context):
    server = MCPServer("test", version="1.0.0")

    @server.tool()
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    @server.prompt()
    def welcome(name: str) -> str:
        """Welcome somebody."""
        return f"Hello {name}"

    @server.resource("note://current")
    def note() -> str:
        """Return the current note."""
        return "remember this"

    @server.completion()
    async def complete_name(ref, argument, context):
        return Completion(values=["Ada"])

    return server


@pytest.mark.asyncio
async def test_official_server_preserves_request_result_and_error_semantics():
    runtime = ASGITestRuntime()
    handler = create_mcp_handler(server_factory, _runtime=runtime)

    discovery = await handler.fetch(modern_request("server/discover"))
    result = await handler.fetch(
        modern_request("tools/call", {"name": "add", "arguments": {"a": 2, "b": 3}})
    )
    missing = await handler.fetch(
        modern_request("tools/call", {"name": "missing", "arguments": {}})
    )

    assert discovery.status == 200
    assert discovery.json()["result"]["supportedVersions"] == ["2026-07-28"]
    result_body = result.json()
    assert result_body["jsonrpc"] == "2.0"
    assert result_body["id"] == 1
    assert result_body["result"]["content"] == [{"type": "text", "text": "5"}]
    assert result_body["result"]["structuredContent"] == {"result": 5}
    assert result_body["result"]["isError"] is False
    missing_body = missing.json()
    assert missing_body["id"] == 1
    assert missing_body["result"]["isError"] is True
    assert missing_body["result"]["content"] == [
        {"type": "text", "text": "Unknown tool: missing"}
    ]


@pytest.mark.asyncio
async def test_official_server_exposes_tools_prompts_resources_and_completion():
    runtime = ASGITestRuntime()
    handler = create_mcp_handler(server_factory, _runtime=runtime)

    tools = await handler.fetch(modern_request("tools/list"))
    prompts = await handler.fetch(modern_request("prompts/list"))
    prompt = await handler.fetch(
        modern_request("prompts/get", {"name": "welcome", "arguments": {"name": "Ada"}})
    )
    resources = await handler.fetch(modern_request("resources/list"))
    resource = await handler.fetch(
        modern_request("resources/read", {"uri": "note://current"})
    )
    completion = await handler.fetch(
        modern_request(
            "completion/complete",
            {
                "ref": {"type": "ref/prompt", "name": "welcome"},
                "argument": {"name": "name", "value": "A"},
            },
        )
    )

    assert [tool["name"] for tool in tools.json()["result"]["tools"]] == ["add"]
    assert [prompt["name"] for prompt in prompts.json()["result"]["prompts"]] == [
        "welcome"
    ]
    assert prompt.json()["result"]["messages"][0]["content"]["text"] == "Hello Ada"
    assert resources.json()["result"]["resources"][0]["uri"] == "note://current"
    resource_body = resource.json()
    assert "result" in resource_body, resource_body
    assert resource_body["result"]["contents"][0]["text"] == "remember this"
    assert completion.json()["result"]["completion"]["values"] == ["Ada"]


@pytest.mark.asyncio
async def test_exact_route_matching_allows_query_but_not_aliases():
    runtime = ASGITestRuntime()
    handler = create_mcp_handler(
        server_factory,
        MCPHandlerOptions(route="/custom"),
        _runtime=runtime,
    )
    request = modern_request("server/discover")
    request.url = "https://example.com/custom?trace=1"

    matched = await handler.fetch(request)
    request.url = "https://example.com/custom/"
    trailing = await handler.fetch(request)
    request.url = "https://example.com/mcp"
    default = await handler.fetch(request)

    assert matched.status == 200
    assert trailing.status == 404
    assert default.status == 404
    assert runtime.calls == 1


@pytest.mark.asyncio
async def test_parse_and_protocol_errors_come_from_official_mcp_boundary():
    runtime = ASGITestRuntime()
    handler = create_mcp_handler(server_factory, _runtime=runtime)
    malformed = modern_request("server/discover")
    malformed.body = b"{"
    malformed_envelope = modern_request("server/discover")
    malformed_envelope.body = b"[]"

    parse_error = await handler.fetch(malformed)
    invalid_request = await handler.fetch(malformed_envelope)

    assert parse_error.status == 400
    assert parse_error.json() == {
        "jsonrpc": "2.0",
        "id": None,
        "error": {"code": -32700, "message": "Parse error"},
    }
    assert invalid_request.json()["error"]["code"] == -32600


@pytest.mark.asyncio
async def test_preflight_and_cors_cover_mcp_headers_without_constructing_server():
    calls = 0

    def factory(_context):
        nonlocal calls
        calls += 1
        return MCPServer("unused")

    runtime = ASGITestRuntime()
    handler = create_mcp_handler(
        factory,
        MCPHandlerOptions(
            cors=CORSOptions(origin="https://client.example"),
            allowed_origin_hostnames=("client.example",),
        ),
        _runtime=runtime,
    )
    response = await handler.fetch(
        Request(
            method="OPTIONS",
            headers={
                "host": "example.com",
                "origin": "https://client.example",
                "access-control-request-method": "POST",
                "access-control-request-headers": "mcp-method, mcp-name",
            },
        )
    )

    assert response.status == 200
    assert response.headers["access-control-allow-origin"] == "https://client.example"
    assert "Mcp-Method" in response.headers["access-control-allow-headers"]
    assert "Mcp-Name" in response.headers["access-control-allow-headers"]
    assert calls == 0
    assert runtime.calls == 0


@pytest.mark.asyncio
async def test_unsupported_and_preaborted_requests_stop_before_auth_and_factory():
    calls = []

    class Verifier:
        def verify(self, execution_context):
            calls.append(("auth", execution_context))

    def factory(_context):
        calls.append(("factory", None))
        return MCPServer("unused")

    runtime = ASGITestRuntime()
    handler = create_mcp_handler(
        factory,
        MCPHandlerOptions(auth_verifier=Verifier()),
        _runtime=runtime,
    )
    unsupported = modern_request("server/discover")
    unsupported.method = "PUT"
    aborted = modern_request("server/discover")
    aborted.signal = AbortSignal(aborted=True)
    disabled_preflight = modern_request("server/discover")
    disabled_preflight.method = "OPTIONS"

    unsupported_response = await handler.fetch(unsupported, execution_context=object())
    aborted_response = await handler.fetch(aborted, execution_context=object())
    disabled_handler = create_mcp_handler(
        factory,
        MCPHandlerOptions(cors=False, auth_verifier=Verifier()),
        _runtime=runtime,
    )
    disabled_response = await disabled_handler.fetch(
        disabled_preflight, execution_context=object()
    )

    assert unsupported_response.status == 405
    assert unsupported_response.headers["allow"] == "GET, POST, DELETE, OPTIONS"
    assert aborted_response.status == 499
    assert disabled_response.status == 405
    assert disabled_response.headers["allow"] == "GET, POST, DELETE"
    assert calls == []
    assert runtime.calls == 0


@pytest.mark.asyncio
async def test_declared_oversized_body_stops_at_worker_boundary():
    calls = 0

    def factory(_context):
        nonlocal calls
        calls += 1
        return MCPServer("unused")

    runtime = ASGITestRuntime()
    handler = create_mcp_handler(
        factory,
        MCPHandlerOptions(max_request_body_size=8),
        _runtime=runtime,
    )
    request = modern_request("server/discover")
    request.body = b""
    request.headers["content-length"] = "9"
    buffered = modern_request("server/discover")
    buffered.body = b"123456789"
    malformed = modern_request("server/discover")
    malformed.body = b""
    malformed.headers["content-length"] = "9, 9"
    extreme = modern_request("server/discover")
    extreme.body = b""
    extreme.headers["content-length"] = "9" * 5000

    response = await handler.fetch(request)
    buffered_response = await handler.fetch(buffered)
    malformed_response = await handler.fetch(malformed)
    extreme_response = await handler.fetch(extreme)

    assert response.status == 413
    assert json.loads(response.body)["error"]["message"] == "Request body too large"
    assert buffered_response.status == 413
    assert malformed_response.status == 400
    assert extreme_response.status == 413
    assert (
        json.loads(malformed_response.body)["error"]["message"]
        == "Invalid Content-Length header"
    )
    assert calls == 0
    assert runtime.calls == 0


@pytest.mark.asyncio
async def test_cors_false_removes_access_control_headers():
    runtime = ASGITestRuntime()
    handler = create_mcp_handler(
        server_factory,
        MCPHandlerOptions(cors=False, json_response=True),
        _runtime=runtime,
    )

    response = await handler.fetch(modern_request("server/discover"))

    assert response.status == 200
    assert all(not name.startswith("access-control-") for name in response.headers)


@pytest.mark.parametrize("route", ["mcp", "/mcp?x=1", "/mcp#fragment"])
def test_rejects_non_path_routes(route):
    with pytest.raises(ValueError, match="exact absolute pathname"):
        create_mcp_handler(server_factory, MCPHandlerOptions(route=route))


@pytest.mark.asyncio
async def test_internal_errors_are_generic_and_reporter_failure_is_ignored():
    seen = []

    def broken_factory(_context):
        raise RuntimeError("secret implementation detail")

    def reporter(error):
        seen.append(str(error))
        raise RuntimeError("reporting also failed")

    handler = create_mcp_handler(
        broken_factory,
        MCPHandlerOptions(on_error=reporter),
        _runtime=ASGITestRuntime(),
    )

    response = await handler.fetch(modern_request("server/discover"))
    body = json.loads(response.body)

    assert response.status == 500
    assert body == {
        "jsonrpc": "2.0",
        "error": {"code": -32603, "message": "Internal server error"},
        "id": None,
    }
    assert seen == ["secret implementation detail"]
    assert "secret" not in response.body


def test_server_subtree_has_no_agent_or_lifecycle_imports():
    server_root = Path(__file__).parents[3] / "agents/mcp/server"
    imported = set()
    dynamic_imports = set()
    for source in server_root.rglob("*.py"):
        tree = ast.parse(source.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module or "")
            elif (
                isinstance(node, ast.Call)
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                if isinstance(node.func, ast.Name) and node.func.id == "__import__":
                    dynamic_imports.add(node.args[0].value)
                elif (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr == "import_module"
                ):
                    dynamic_imports.add(node.args[0].value)

    assert not any(
        name == "agents"
        or name.startswith("agents.core")
        or name.startswith("agents.lifecycle")
        for name in imported | dynamic_imports
    )


@pytest.mark.asyncio
async def test_production_adapter_uses_top_level_asgi_fetch(monkeypatch):
    from agents.mcp.server._handler import _WorkersASGIRuntime

    calls = []
    expected = object()

    async def fetch(app, request, env, execution_context):
        calls.append((app, request, env, execution_context))
        return expected

    monkeypatch.setitem(sys.modules, "asgi", types.SimpleNamespace(fetch=fetch))
    runtime = _WorkersASGIRuntime()
    app = object()
    request = modern_request("server/discover")
    env = object()
    execution_context = object()

    result = await runtime.fetch(app, request, env, execution_context)

    assert result is expected
    assert len(calls) == 1
    assert calls[0][0] is not app
    assert calls[0][1:] == (request, env, execution_context)
