from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager

import pytest
from mcp.server import MCPServer
from mcp.server.mcpserver import Context, RequestStateSecurity
from mcp.types import (
    ElicitRequest,
    ElicitRequestFormParams,
    ElicitResult,
    InputRequiredResult,
)

from agents.mcp.server import (
    MCPHandlerOptions,
    StatelessElicitation,
    create_mcp_handler,
)
from agents.mcp.server._handler import _PerRequestLifespanApp

from .conftest import ASGITestRuntime, modern_request


@pytest.mark.asyncio
async def test_progress_selects_sse_and_finishes_with_one_result():
    def factory(_context):
        server = MCPServer("progress")

        @server.tool()
        async def work(ctx: Context) -> str:
            await ctx.report_progress(1, total=2, message="half")
            await ctx.report_progress(2, total=2, message="done")
            return "finished"

        return server

    request = modern_request(
        "tools/call",
        {
            "name": "work",
            "arguments": {},
            "_meta": {"progressToken": "p-1"},
        },
    )
    handler = create_mcp_handler(factory, _runtime=ASGITestRuntime())

    response = await handler.fetch(request)

    text = response.body.decode()
    assert response.headers["content-type"].startswith("text/event-stream")
    assert text.count("notifications/progress") == 2
    assert text.count('"jsonrpc":"2.0","id":1,"result":') == 1
    assert text.index('"progress":1') < text.index('"progress":2')


@pytest.mark.asyncio
async def test_json_response_mode_returns_json_and_progress_is_a_noop():
    def factory(_context):
        server = MCPServer("json")

        @server.tool()
        async def work(ctx: Context) -> str:
            await ctx.report_progress(1, total=1)
            return "finished"

        return server

    request = modern_request(
        "tools/call",
        {
            "name": "work",
            "arguments": {},
            "_meta": {"progressToken": "p-1"},
        },
        accept="application/json",
    )
    handler = create_mcp_handler(
        factory,
        MCPHandlerOptions(json_response=True),
        _runtime=ASGITestRuntime(),
    )

    response = await handler.fetch(request)

    assert response.headers["content-type"] == "application/json"
    assert response.json()["result"]["content"][0]["text"] == "finished"
    assert b"notifications/progress" not in response.body


@pytest.mark.asyncio
async def test_disconnect_cancels_handler_and_closes_lifespan():
    started = asyncio.Event()
    cleaned = asyncio.Event()
    lifespan_events = []

    @asynccontextmanager
    async def lifespan(_server):
        lifespan_events.append("start")
        try:
            yield None
        finally:
            lifespan_events.append("stop")

    def factory(_context):
        server = MCPServer("cancel", lifespan=lifespan)

        @server.tool()
        async def hang() -> str:
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaned.set()
            return "unreachable"

        return server

    request = modern_request("tools/call", {"name": "hang", "arguments": {}})
    request.disconnect = asyncio.Event()
    handler = create_mcp_handler(factory, _runtime=ASGITestRuntime())
    response_task = asyncio.create_task(handler.fetch(request))

    await asyncio.wait_for(started.wait(), timeout=1)
    request.disconnect.set()
    response = await asyncio.wait_for(response_task, timeout=1)

    assert response.status == 499
    assert cleaned.is_set()
    assert lifespan_events == ["start", "stop"]


@pytest.mark.asyncio
async def test_production_lifespan_stays_open_until_stream_finishes():
    started = asyncio.Event()
    release = asyncio.Event()
    events = []

    async def app(scope, receive, send):
        if scope["type"] == "lifespan":
            assert (await receive())["type"] == "lifespan.startup"
            events.append("start")
            await send({"type": "lifespan.startup.complete"})
            assert (await receive())["type"] == "lifespan.shutdown"
            events.append("stop")
            await send({"type": "lifespan.shutdown.complete"})
            return
        events.append("request")
        await send({"type": "http.response.start", "status": 200})
        await send({"type": "http.response.body", "body": b"first", "more_body": True})
        started.set()
        await release.wait()
        await send({"type": "http.response.body", "body": b"second"})

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(_message):
        return None

    wrapped = _PerRequestLifespanApp(app)
    task = asyncio.create_task(
        wrapped(
            {"type": "http", "state": {}},
            receive,
            send,
        )
    )

    await asyncio.wait_for(started.wait(), timeout=1)
    assert events == ["start", "request"]
    release.set()
    await asyncio.wait_for(task, timeout=1)
    assert events == ["start", "request", "stop"]


@pytest.mark.asyncio
async def test_production_lifespan_cleans_up_on_cancellation_and_startup_failure():
    started = asyncio.Event()
    stopped = asyncio.Event()

    async def cancellable_app(scope, receive, send):
        if scope["type"] == "lifespan":
            assert (await receive())["type"] == "lifespan.startup"
            await send({"type": "lifespan.startup.complete"})
            assert (await receive())["type"] == "lifespan.shutdown"
            stopped.set()
            await send({"type": "lifespan.shutdown.complete"})
            return
        started.set()
        await asyncio.Event().wait()

    async def failed_app(scope, receive, send):
        assert scope["type"] == "lifespan"
        assert (await receive())["type"] == "lifespan.startup"
        await send({"type": "lifespan.startup.failed", "message": "broken"})

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(_message):
        return None

    wrapped = _PerRequestLifespanApp(cancellable_app)
    task = asyncio.create_task(wrapped({"type": "http"}, receive, send))
    await asyncio.wait_for(started.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert stopped.is_set()

    failed = _PerRequestLifespanApp(failed_app)
    with pytest.raises(RuntimeError, match="broken"):
        await asyncio.wait_for(
            failed({"type": "http"}, receive, send),
            timeout=1,
        )


@pytest.mark.asyncio
async def test_every_success_error_and_rejection_closes_its_server_lifespan():
    active = 0
    closed = 0

    @asynccontextmanager
    async def lifespan(_server):
        nonlocal active, closed
        active += 1
        try:
            yield None
        finally:
            active -= 1
            closed += 1

    def factory(_context):
        server = MCPServer("cleanup", lifespan=lifespan)

        @server.tool()
        def fail() -> str:
            raise RuntimeError("failed")

        return server

    handler = create_mcp_handler(factory, _runtime=ASGITestRuntime())

    success = await handler.fetch(modern_request("server/discover"))
    error = await handler.fetch(
        modern_request("tools/call", {"name": "fail", "arguments": {}})
    )
    rejection = await handler.fetch(modern_request("unknown/method"))

    assert success.status == 200
    assert "error" in error.json()["result"]["content"][0]["text"].lower()
    assert "error" in rejection.json()
    assert active == 0
    assert closed == 3


@pytest.mark.asyncio
async def test_stateless_elicitation_rounds_share_official_request_state_security():
    security = RequestStateSecurity(keys=[b"s" * 32], audience="elicitation-test")
    elicitation = StatelessElicitation(request_state_security=security)
    factory_contexts = []
    active = 0
    closed = 0

    @asynccontextmanager
    async def lifespan(_server):
        nonlocal active, closed
        active += 1
        try:
            yield None
        finally:
            active -= 1
            closed += 1

    def factory(context):
        factory_contexts.append(context)
        server = MCPServer(
            "elicitation",
            lifespan=lifespan,
            request_state_security=context.elicitation.request_state_security,
        )

        @server.tool()
        async def confirm(ctx: Context) -> str | InputRequiredResult:
            answer = (ctx.input_responses or {}).get("confirmation")
            if not isinstance(answer, ElicitResult):
                return InputRequiredResult(
                    input_requests={
                        "confirmation": ElicitRequest(
                            params=ElicitRequestFormParams(
                                message="Continue?",
                                requested_schema={
                                    "type": "object",
                                    "properties": {"ok": {"type": "boolean"}},
                                    "required": ["ok"],
                                },
                            )
                        )
                    },
                    request_state="round-one",
                )
            return "accepted" if answer.content == {"ok": True} else "declined"

        return server

    handler = create_mcp_handler(
        factory,
        MCPHandlerOptions(elicitation=elicitation),
        _runtime=ASGITestRuntime(),
    )
    first = await handler.fetch(
        modern_request("tools/call", {"name": "confirm", "arguments": {}})
    )
    first_result = first.json()["result"]
    request_state = first_result["requestState"]
    second_request = modern_request(
        "tools/call",
        {
            "name": "confirm",
            "arguments": {},
            "requestState": request_state,
            "inputResponses": {
                "confirmation": {"action": "accept", "content": {"ok": True}}
            },
        },
    )
    second = await handler.fetch(second_request)

    assert first_result["resultType"] == "input_required"
    assert (
        first_result["inputRequests"]["confirmation"]["method"] == "elicitation/create"
    )
    assert request_state != "round-one"
    assert second.json()["result"]["content"][0]["text"] == "accepted"
    assert len(factory_contexts) == 2
    assert all(context.elicitation is elicitation for context in factory_contexts)
    assert active == 0
    assert closed == 2


@pytest.mark.asyncio
async def test_tampered_elicitation_state_is_rejected_without_running_tool_body():
    security = RequestStateSecurity(keys=[b"t" * 32], audience="tamper-test")
    calls = 0

    def factory(context):
        server = MCPServer(
            "elicitation",
            request_state_security=context.elicitation.request_state_security,
        )

        @server.tool()
        async def guarded(ctx: Context) -> str | InputRequiredResult:
            nonlocal calls
            calls += 1
            if ctx.input_responses is None:
                return InputRequiredResult(request_state="continue")
            return "done"

        return server

    handler = create_mcp_handler(
        factory,
        MCPHandlerOptions(
            elicitation=StatelessElicitation(request_state_security=security)
        ),
        _runtime=ASGITestRuntime(),
    )
    request = modern_request(
        "tools/call",
        {
            "name": "guarded",
            "arguments": {},
            "requestState": "tampered",
            "inputResponses": {},
        },
    )

    response = await handler.fetch(request)

    assert response.json()["error"]["code"] == -32602
    assert response.json()["error"]["message"] == "Invalid or expired requestState"
    assert calls == 0
