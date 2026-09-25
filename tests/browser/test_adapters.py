from __future__ import annotations

from dataclasses import dataclass

import pytest

from agents.browser import BrowserRequest, WorkersBrowserBinding, WorkersBrowserStorage
from agents.browser.adapters import _WorkersRuntime


class FakeProxy:
    def __init__(self, callback):
        self.callback = callback
        self.destroyed = False

    def __call__(self, value):
        self.callback(value)

    def destroy(self):
        self.destroyed = True


class FakeObject:
    fromEntries = object()


class FakeRequest:
    seen: list[tuple[str, object]] = []

    @classmethod
    def new(cls, url, init):
        request = (url, init)
        cls.seen.append(request)
        return request


class FakeBody:
    def __init__(self):
        self.cancelled = False

    async def cancel(self):
        self.cancelled = True


class FakeHeaders(dict):
    pass


class FakeJsSocket:
    def __init__(self):
        self.accepted = False
        self.listeners = {}
        self.sent = []
        self.closed = 0

    def accept(self):
        self.accepted = True

    def addEventListener(self, event, callback):
        self.listeners.setdefault(event, []).append(callback)

    def removeEventListener(self, event, callback):
        self.listeners[event].remove(callback)

    def send(self, data):
        self.sent.append(data)

    def close(self, _code, _reason):
        self.closed += 1

    def emit(self, event, value):
        for callback in tuple(self.listeners.get(event, ())):
            callback(value)


class FakeJsResponse:
    def __init__(self, *, body=b"payload", socket=None):
        self.status = 201
        self.headers = FakeHeaders(
            {
                "content-type": "application/json",
                "cf-browser-session-id": "session-js",
            }
        )
        self.webSocket = socket
        self.body = FakeBody()
        self._bytes = body

    async def arrayBuffer(self):
        return self._bytes


@dataclass
class FakeRuntimeState:
    conversions: list[object]
    proxies: list[FakeProxy]


def runtime():
    state = FakeRuntimeState([], [])

    def to_js(value, *, dict_converter):
        assert dict_converter is FakeObject.fromEntries
        state.conversions.append(value)
        return value

    def create_proxy(callback):
        proxy = FakeProxy(callback)
        state.proxies.append(proxy)
        return proxy

    return _WorkersRuntime(FakeRequest, FakeObject, create_proxy, to_js), state


@pytest.mark.asyncio
async def test_workers_binding_uses_js_request_and_camel_case_quick_action():
    worker_runtime, state = runtime()

    class Binding:
        def __init__(self):
            self.fetch_request = None
            self.quick_call = None

        async def fetch(self, request):
            self.fetch_request = request
            return FakeJsResponse()

        async def quickAction(self, action, params):
            self.quick_call = (action, params)
            return FakeJsResponse(body=b"quick")

    raw = Binding()
    binding = WorkersBrowserBinding(raw, _runtime=worker_runtime)
    response = await binding.fetch(
        BrowserRequest("https://browser.test", method="POST", headers={"X-A": "b"})
    )
    assert raw.fetch_request == (
        "https://browser.test",
        {"method": "POST", "headers": {"X-A": "b"}},
    )
    assert response.status == 201
    assert response.headers["cf-browser-session-id"] == "session-js"
    assert await response.read() == b"payload"
    await response.aclose()

    quick = await binding.quick_action("markdown", {"url": "https://x.test"})
    assert raw.quick_call == ("markdown", {"url": "https://x.test"})
    await quick.aclose()
    assert quick._response.body.cancelled is True
    assert state.conversions == [
        {"method": "POST", "headers": {"X-A": "b"}},
        {"url": "https://x.test"},
    ]


@pytest.mark.asyncio
async def test_workers_socket_retains_and_releases_listener_proxies():
    worker_runtime, state = runtime()
    raw_socket = FakeJsSocket()
    binding = WorkersBrowserBinding(
        type(
            "Binding",
            (),
            {"fetch": lambda _self, _request: FakeJsResponse(socket=raw_socket)},
        )(),
        _runtime=worker_runtime,
    )
    response = await binding.fetch(BrowserRequest("https://browser.test"))
    socket = response.websocket
    assert socket is not None
    socket.accept()
    seen = []
    subscription = socket.subscribe("message", seen.append)
    raw_socket.emit("message", type("Event", (), {"data": "hello"})())
    assert seen == [{"data": "hello"}]
    assert state.proxies[0].destroyed is False

    subscription.cancel()
    subscription.cancel()
    assert state.proxies[0].destroyed is True
    assert raw_socket.listeners["message"] == []


@pytest.mark.asyncio
async def test_workers_storage_converts_values_and_prefix_options():
    worker_runtime, state = runtime()

    class Storage:
        def __init__(self):
            self.values = {"cdp:a": {"sessionId": "one"}}
            self.list_options = None

        async def get(self, key):
            return self.values.get(key)

        async def put(self, key, value):
            self.values[key] = value

        async def delete(self, key):
            return self.values.pop(key, None) is not None

        async def list(self, options):
            self.list_options = options
            return self.values

    raw = Storage()
    storage = WorkersBrowserStorage(raw, _runtime=worker_runtime)
    await storage.put("cdp:b", {"sessionId": "two"})
    assert await storage.get("cdp:b") == {"sessionId": "two"}
    assert await storage.list(prefix="cdp:") == raw.values
    assert raw.list_options == {"prefix": "cdp:"}
    assert await storage.delete("cdp:b") is True
    assert state.conversions == [{"sessionId": "two"}, {"prefix": "cdp:"}]
