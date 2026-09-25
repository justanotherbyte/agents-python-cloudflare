from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AbortSignal:
    aborted: bool = False


@dataclass
class Request:
    url: str = "https://example.com/mcp"
    method: str = "POST"
    headers: dict[str, str] = field(default_factory=dict)
    body: bytes = b""
    disconnect: asyncio.Event | None = None
    signal: AbortSignal = field(default_factory=AbortSignal)

    def __post_init__(self) -> None:
        self.headers = {key.lower(): value for key, value in self.headers.items()}


@dataclass
class ASGIResponse:
    status: int
    headers: dict[str, str]
    body: bytes
    chunks: list[bytes]

    def json(self) -> object:
        return json.loads(self.body)


class ASGITestRuntime:
    def __init__(self) -> None:
        self.calls = 0

    async def fetch(self, app, request, env, execution_context):
        self.calls += 1
        shutdown, lifespan_task, state = await self._startup(app)
        response_done = asyncio.Event()
        sent_request = False
        messages: list[dict[str, Any]] = []

        async def receive() -> dict[str, Any]:
            nonlocal sent_request
            if not sent_request:
                sent_request = True
                return {
                    "type": "http.request",
                    "body": request.body,
                    "more_body": False,
                }
            if request.disconnect is None:
                await response_done.wait()
            else:
                response_wait = asyncio.create_task(response_done.wait())
                disconnect_wait = asyncio.create_task(request.disconnect.wait())
                done, pending = await asyncio.wait(
                    {response_wait, disconnect_wait},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                if disconnect_wait in done and request.disconnect.is_set():
                    return {"type": "http.disconnect"}
            return {"type": "http.disconnect"}

        async def send(message: dict[str, Any]) -> None:
            messages.append(message)
            if message["type"] == "http.response.body" and not message.get(
                "more_body", False
            ):
                response_done.set()

        from urllib.parse import unquote, urlsplit

        url = urlsplit(request.url)
        scope = {
            "asgi": {"spec_version": "2.0", "version": "3.0"},
            "headers": [
                (name.encode(), value.encode())
                for name, value in request.headers.items()
            ],
            "http_version": "1.1",
            "method": request.method,
            "scheme": url.scheme,
            "path": unquote(url.path),
            "raw_path": url.path.encode(),
            "query_string": url.query.encode(),
            "type": "http",
            "env": env,
            "state": dict(state),
        }
        try:
            await app(scope, receive, send)
        finally:
            response_done.set()
            await shutdown()
            await lifespan_task

        starts = [item for item in messages if item["type"] == "http.response.start"]
        if not starts:
            return ASGIResponse(499, {}, b"", [])
        chunks = [
            item.get("body", b"")
            for item in messages
            if item["type"] == "http.response.body"
        ]
        headers: dict[str, str] = {}
        for name, value in starts[0].get("headers", []):
            headers[name.decode().lower()] = value.decode()
        return ASGIResponse(starts[0]["status"], headers, b"".join(chunks), chunks)

    async def _startup(self, app):
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        startup = asyncio.get_running_loop().create_future()
        shutdown_complete = asyncio.get_running_loop().create_future()
        state: dict[str, object] = {}

        async def receive():
            return await queue.get()

        async def send(message):
            if message["type"] == "lifespan.startup.complete":
                startup.set_result(None)
            elif message["type"] == "lifespan.startup.failed":
                startup.set_exception(
                    RuntimeError(message.get("message", "startup failed"))
                )
            elif message["type"] == "lifespan.shutdown.complete":
                shutdown_complete.set_result(None)
            elif message["type"] == "lifespan.shutdown.failed":
                shutdown_complete.set_exception(
                    RuntimeError(message.get("message", "shutdown failed"))
                )

        task = asyncio.create_task(
            app(
                {
                    "asgi": {"spec_version": "2.0", "version": "3.0"},
                    "state": state,
                    "type": "lifespan",
                },
                receive,
                send,
            )
        )
        await queue.put({"type": "lifespan.startup"})
        await startup

        async def shutdown() -> None:
            await queue.put({"type": "lifespan.shutdown"})
            await shutdown_complete

        return shutdown, task, state


def modern_request(
    method: str,
    params: dict[str, object] | None = None,
    *,
    request_id: str | int = 1,
    accept: str = "application/json, text/event-stream",
) -> Request:
    params = dict(params or {})
    meta = dict(params.get("_meta", {}))
    meta.update(
        {
            "io.modelcontextprotocol/protocolVersion": "2026-07-28",
            "io.modelcontextprotocol/clientInfo": {
                "name": "test",
                "version": "1.0.0",
            },
            "io.modelcontextprotocol/clientCapabilities": {},
        }
    )
    params["_meta"] = meta
    name = params.get("name") or params.get("uri")
    headers = {
        "accept": accept,
        "content-type": "application/json",
        "host": "example.com",
        "mcp-protocol-version": "2026-07-28",
        "mcp-method": method,
    }
    if isinstance(name, str):
        headers["mcp-name"] = name
    return Request(
        headers=headers,
        body=json.dumps(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params,
            },
            separators=(",", ":"),
        ).encode(),
    )
