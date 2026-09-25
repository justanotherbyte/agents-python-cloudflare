from __future__ import annotations

import asyncio
import copy
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from agents.browser import BrowserRequest


class FakeSubscription:
    def __init__(self, cancel: Callable[[], None]):
        self._cancel = cancel
        self.cancelled = False

    def cancel(self) -> None:
        if self.cancelled:
            return
        self.cancelled = True
        self._cancel()


class FakeSocket:
    next_attach = 0

    def __init__(self, *, respond: bool = True):
        self.respond = respond
        self.accepted = False
        self.closed = 0
        self.sent: list[dict[str, object]] = []
        self.listeners: dict[str, list[Callable[[object], None]]] = {}

    def accept(self) -> None:
        self.accepted = True

    def subscribe(self, event: str, callback: Callable[[object], None]):
        self.listeners.setdefault(event, []).append(callback)
        return FakeSubscription(lambda: self.listeners[event].remove(callback))

    def send(self, data: str) -> None:
        message = json.loads(data)
        self.sent.append(message)
        if not self.respond:
            return
        if message["method"] == "Target.attachToTarget":
            FakeSocket.next_attach += 1
            result = {"sessionId": f"attached-{FakeSocket.next_attach}"}
        else:
            result = {"echo": message["method"]}
        asyncio.get_running_loop().call_soon(
            self.emit,
            "message",
            {"data": json.dumps({"id": message["id"], "result": result})},
        )

    def close(self, _code: int = 1000, _reason: str = "") -> None:
        self.closed += 1
        self.emit("close", {})

    def emit(self, event: str, value: object) -> None:
        for callback in tuple(self.listeners.get(event, ())):
            callback(value)


class FakeResponse:
    def __init__(
        self,
        body: object = b"",
        *,
        status: int = 200,
        headers: Mapping[str, str] | None = None,
        websocket: FakeSocket | None = None,
    ):
        if isinstance(body, bytes):
            self._body = body
        elif isinstance(body, str):
            self._body = body.encode()
        else:
            self._body = json.dumps(body, separators=(",", ":")).encode()
        self.status = status
        self.headers = dict(headers or {})
        self.websocket = websocket
        self.closed = False

    async def read(self) -> bytes:
        return self._body

    async def aclose(self) -> None:
        self.closed = True


@dataclass(frozen=True)
class SeenRequest:
    url: str
    method: str
    upgrade: bool


class FakeBrowser:
    def __init__(self):
        self.requests: list[SeenRequest] = []
        self.responses: list[FakeResponse] = []
        self.sockets: list[FakeSocket] = []
        self.quick_calls: list[tuple[str, dict[str, object]]] = []
        self.quick_handler: Callable[[str, Mapping[str, object]], FakeResponse] = (
            lambda _action, _params: FakeResponse({"success": True, "result": "ok"})
        )
        self.created = 0
        self.list_statuses: list[int] = []
        self.delete_statuses: list[int] = []
        self.live_targets = False
        self.create_gate: asyncio.Event | None = None

    async def fetch(self, request: BrowserRequest) -> FakeResponse:
        upgrade = request.headers.get("Upgrade") == "websocket"
        self.requests.append(SeenRequest(request.url, request.method, upgrade))
        if self.create_gate is not None and request.method == "POST":
            await self.create_gate.wait()
        if upgrade:
            socket = FakeSocket()
            self.sockets.append(socket)
            session_id = "session-direct"
            parts = request.url.split("/browser/", 1)
            if len(parts) == 2:
                session_id = parts[1].split("?", 1)[0]
            response = FakeResponse(
                websocket=socket,
                headers={"CF-Browser-Session-ID": session_id},
            )
        elif request.method == "POST":
            self.created += 1
            response = FakeResponse({"sessionId": f"session-{self.created}"})
        elif request.url.endswith("/json/protocol"):
            response = FakeResponse(
                {
                    "domains": [
                        {
                            "domain": "Page",
                            "commands": [{"name": "navigate"}],
                            "events": [{"name": "loadEventFired"}],
                            "types": [{"id": "FrameId"}],
                        }
                    ]
                }
            )
        elif request.url.endswith("/json/list"):
            status = self.list_statuses.pop(0) if self.list_statuses else 200
            target = {"id": "target-1", "type": "page", "url": "https://x.test"}
            if self.live_targets:
                target["devtoolsFrontendUrl"] = (
                    "https://live.browser.run/ui/view?wss=socket&jwt=token"
                )
            response = FakeResponse([target], status=status)
        elif request.method == "DELETE":
            status = self.delete_statuses.pop(0) if self.delete_statuses else 204
            response = FakeResponse(status=status)
        else:
            response = FakeResponse(status=204)
        self.responses.append(response)
        return response

    async def quick_action(
        self, action: str, params: Mapping[str, object]
    ) -> FakeResponse:
        self.quick_calls.append((action, dict(params)))
        response = self.quick_handler(action, params)
        self.responses.append(response)
        return response


class FakeStorage:
    def __init__(self):
        self.values: dict[str, object] = {}

    async def get(self, key: str) -> object | None:
        return copy.deepcopy(self.values.get(key))

    async def put(self, key: str, value: object) -> None:
        self.values[key] = copy.deepcopy(value)

    async def delete(self, key: str) -> bool:
        return self.values.pop(key, None) is not None

    async def list(self, *, prefix: str) -> Mapping[str, object]:
        return {
            key: copy.deepcopy(value)
            for key, value in self.values.items()
            if key.startswith(prefix)
        }
