from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from .protocols import BrowserResponse, BrowserSocket, BrowserSocketSubscription
from .types import BrowserRequest


@dataclass(frozen=True)
class _WorkersRuntime:
    request: object
    object: object
    create_proxy: Callable[[object], object]
    to_js: Callable[..., object]


def _load_runtime() -> _WorkersRuntime:
    try:
        from js import Object, Request  # ty: ignore[unresolved-import]
        from pyodide.ffi import create_proxy, to_js
    except ImportError as error:
        raise RuntimeError(
            "Workers browser adapters are available only in the Pyodide Workers runtime"
        ) from error
    return _WorkersRuntime(Request, Object, create_proxy, to_js)


async def _await(value: object) -> object:
    if inspect.isawaitable(value):
        return await value
    return value


def _python(value: object) -> object:
    convert = getattr(value, "to_py", None)
    return convert() if callable(convert) else value


class _WorkersSubscription:
    def __init__(self, socket: object, event: str, proxy: object):
        self._socket = socket
        self._event = event
        self._proxy = proxy

    def cancel(self) -> None:
        proxy, self._proxy = self._proxy, None
        if proxy is None:
            return
        try:
            getattr(self._socket, "removeEventListener")(self._event, proxy)
        finally:
            destroy = getattr(proxy, "destroy", None)
            if callable(destroy):
                destroy()


class WorkersBrowserSocket:
    """Adapt a Workers WebSocket while retaining and releasing callback proxies."""

    def __init__(self, socket: object, runtime: _WorkersRuntime | None = None):
        self._socket = socket
        self._runtime = runtime or _load_runtime()

    def accept(self) -> object:
        return getattr(self._socket, "accept")()

    def send(self, data: str) -> object:
        return getattr(self._socket, "send")(data)

    def close(self, code: int = 1000, reason: str = "") -> object:
        return getattr(self._socket, "close")(code, reason)

    def subscribe(
        self,
        event: str,
        callback: Callable[[object], None],
    ) -> BrowserSocketSubscription:
        def receive(raw: object) -> None:
            if event == "message":
                callback({"data": getattr(raw, "data", raw)})
            else:
                callback(raw)

        proxy = self._runtime.create_proxy(receive)
        getattr(self._socket, "addEventListener")(event, proxy)
        return _WorkersSubscription(self._socket, event, proxy)


class WorkersBrowserResponse:
    """Adapt a JS Response and explicitly own its unread body."""

    def __init__(self, response: object, runtime: _WorkersRuntime | None = None):
        self._response = response
        self._runtime = runtime or _load_runtime()
        headers = getattr(response, "headers", None)
        self._headers = {
            name: value
            for name in ("content-type", "cf-browser-session-id")
            if headers is not None and (value := headers.get(name)) is not None
        }
        raw_socket = getattr(response, "webSocket", None)
        self._websocket = (
            WorkersBrowserSocket(raw_socket, self._runtime)
            if raw_socket is not None
            else None
        )
        self._consumed = False
        self._closed = False

    @property
    def status(self) -> int:
        return int(getattr(self._response, "status"))

    @property
    def headers(self) -> Mapping[str, str]:
        return self._headers

    @property
    def websocket(self) -> BrowserSocket | None:
        return self._websocket

    async def read(self) -> bytes:
        if self._consumed:
            return b""
        self._consumed = True
        buffer = await _await(getattr(self._response, "arrayBuffer")())
        value = _python(buffer)
        if isinstance(value, (bytes, bytearray, memoryview)):
            return bytes(value)
        try:
            from js import Uint8Array  # ty: ignore[unresolved-import]
        except ImportError as error:
            raise RuntimeError("Workers runtime does not expose Uint8Array") from error
        converted = _python(Uint8Array.new(buffer))
        if not isinstance(converted, (bytes, bytearray, memoryview, list, tuple)):
            raise TypeError("Uint8Array did not convert to byte values")
        return bytes(converted)

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._consumed:
            return
        body = getattr(self._response, "body", None)
        cancel = getattr(body, "cancel", None)
        if callable(cancel):
            await _await(cancel())


class WorkersBrowserBinding:
    """Adapt a real Browser binding's `fetch` and camelCase `quickAction`."""

    def __init__(self, binding: object, *, _runtime: _WorkersRuntime | None = None):
        self._binding = binding
        self._runtime = _runtime or _load_runtime()

    async def fetch(self, request: BrowserRequest) -> BrowserResponse:
        init = self._runtime.to_js(
            {"method": request.method, "headers": dict(request.headers)},
            dict_converter=getattr(self._runtime.object, "fromEntries"),
        )
        js_request = getattr(self._runtime.request, "new")(request.url, init)
        response = await _await(getattr(self._binding, "fetch")(js_request))
        return WorkersBrowserResponse(response, self._runtime)

    async def quick_action(
        self,
        action: str,
        params: Mapping[str, object],
    ) -> BrowserResponse:
        js_params = self._runtime.to_js(
            dict(params), dict_converter=getattr(self._runtime.object, "fromEntries")
        )
        response = await _await(
            getattr(self._binding, "quickAction")(action, js_params)
        )
        return WorkersBrowserResponse(response, self._runtime)


class WorkersBrowserStorage:
    """Adapt Durable Object storage without leaking JS values into session logic."""

    def __init__(self, storage: object, *, _runtime: _WorkersRuntime | None = None):
        self._storage = storage
        self._runtime = _runtime or _load_runtime()

    async def get(self, key: str) -> object | None:
        return _python(await _await(getattr(self._storage, "get")(key)))

    async def put(self, key: str, value: object) -> None:
        converted = self._runtime.to_js(
            value, dict_converter=getattr(self._runtime.object, "fromEntries")
        )
        await _await(getattr(self._storage, "put")(key, converted))

    async def delete(self, key: str) -> object:
        return await _await(getattr(self._storage, "delete")(key))

    async def list(self, *, prefix: str) -> Mapping[str, object]:
        options = self._runtime.to_js(
            {"prefix": prefix},
            dict_converter=getattr(self._runtime.object, "fromEntries"),
        )
        value = _python(await _await(getattr(self._storage, "list")(options)))
        if isinstance(value, Mapping):
            return value
        raise TypeError("Durable Object storage list did not return a mapping")
