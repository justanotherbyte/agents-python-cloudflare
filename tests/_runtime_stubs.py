"""Install stand-ins for the runtime-only modules `agents` imports.

`js`, `workers`, `pyodide` and `pyodide.ffi` exist only inside Pyodide, and
`agents` imports them at module load. `install()` must run before the first
`import agents`, so both conftest and fakes call it at their own top.
"""

from __future__ import annotations

import asyncio
import sys
import types
from http import HTTPMethod
from typing import Any


def _default_wait_until(coro: Any) -> Any:
    # A coroutine handed to waitUntil with no test driving it would warn "never
    # awaited"; closing it is a quiet no-op. Tests that need the work run install
    # a WaitUntilRecorder over this.
    close = getattr(coro, "close", None)
    if close is not None:
        close()
    return coro


class _StubObject:
    # js.Object: only fromEntries (a dict_converter our to_js ignores) and keys.
    @staticmethod
    def fromEntries(entries: Any) -> Any:
        return entries

    @staticmethod
    def keys(obj: Any) -> list[Any]:
        if isinstance(obj, dict):
            return list(obj.keys())
        return list(getattr(obj, "__dict__", {}).keys())


class _StubTextEncoder:
    @classmethod
    def new(cls) -> _StubTextEncoder:
        return cls()

    @staticmethod
    def encode(value: str) -> bytes:
        return value.encode("utf-8")


class _StubStreamController:
    def __init__(self) -> None:
        self.chunks: list[bytes] = []
        self.closed = False
        self.failure: str | None = None

    def enqueue(self, chunk: bytes) -> None:
        self.chunks.append(chunk)

    def close(self) -> None:
        self.closed = True

    def error(self, error: object) -> None:
        self.failure = str(error)
        self.closed = True


class _StubReadableStream:
    def __init__(self, source: dict[str, Any]) -> None:
        self._source = source

    @classmethod
    def new(cls, source: dict[str, Any]) -> _StubReadableStream:
        return cls(source)

    async def read_all(self) -> bytes:
        controller = _StubStreamController()
        pulls = 0
        while not controller.closed:
            pulls += 1
            if pulls > 10_000:
                raise RuntimeError("stream did not close")
            result = self._source["pull"](controller)
            if asyncio.iscoroutine(result):
                await result
        if controller.failure is not None:
            raise RuntimeError(controller.failure)
        return b"".join(controller.chunks)

    async def cancel(self, reason: object = None) -> None:
        callback = self._source.get("cancel")
        if callback is None:
            return
        result = callback(reason)
        if asyncio.iscoroutine(result):
            await result


class _StubDurableObject:
    # Agent subclasses this base. Unlike AGENTS.md's no-op form, this assigns
    # ctx/env, because a real construction through Agent.__init__ reads self.ctx.
    def __init__(self, ctx: Any, env: Any) -> None:
        self.ctx = ctx
        self.env = env


class _StubWorkflowEntrypoint(_StubDurableObject):
    pass


class _StubHeaders(dict[str, str]):
    def __init__(self, headers: Any = None) -> None:
        super().__init__()
        for key, value in dict(headers or {}).items():
            self[key] = value

    def __setitem__(self, key: str, value: str) -> None:
        super().__setitem__(key.lower(), value)

    def __getitem__(self, key: str) -> str:
        return super().__getitem__(key.lower())

    def __contains__(self, key: object) -> bool:
        if isinstance(key, str):
            key = key.lower()
        return super().__contains__(key)

    def __delitem__(self, key: str) -> None:
        super().__delitem__(key.lower())

    def get(self, key: object, default: Any = None) -> Any:
        if isinstance(key, str):
            key = key.lower()
        return super().get(key, default)

    def pop(self, key: object, default: Any = None) -> Any:
        if isinstance(key, str):
            key = key.lower()
        return super().pop(key, default)

    def setdefault(self, key: str, default: str = "") -> str:
        return super().setdefault(key.lower(), default)

    def set(self, key: str, value: str) -> None:
        self[key] = value


class _StubJsResponse:
    def __init__(
        self,
        body: Any = None,
        *,
        status: int = 200,
        headers: Any = None,
        web_socket: Any = None,
    ) -> None:
        self.body = body
        self.status = status
        self.headers = _StubHeaders(headers or {})
        self.webSocket = web_socket

    @classmethod
    def new(cls, body: Any, init: Any) -> _StubJsResponse:
        return cls(
            body,
            status=init.status,
            headers=init.headers,
            web_socket=init.webSocket,
        )


class _StubResponse:
    def __init__(
        self,
        body: Any = None,
        *,
        status: int = 200,
        headers: Any = None,
        web_socket: Any = None,
    ) -> None:
        if isinstance(body, _StubJsResponse):
            self.js_object = body
        else:
            self.js_object = _StubJsResponse(
                body,
                status=status,
                headers=headers,
                web_socket=web_socket,
            )

    @property
    def body(self) -> Any:
        return self.js_object.body

    @property
    def status(self) -> int:
        return self.js_object.status

    @property
    def headers(self) -> _StubHeaders:
        return self.js_object.headers

    @property
    def web_socket(self) -> Any:
        return self.js_object.webSocket


class _StubRequest:
    def __init__(
        self,
        url: str = "https://example.com/",
        *,
        method: str | HTTPMethod = "GET",
        headers: Any = None,
        **kwargs: Any,
    ) -> None:
        self.url = url
        self.method = method.value if isinstance(method, HTTPMethod) else method
        self.headers = _StubHeaders(headers or {})
        self.js_object = self
        self.__dict__.update(kwargs)

    @classmethod
    def new(cls, url: str, init: Any) -> _StubRequest:
        options = {"body": init.body} if hasattr(init, "body") else {}
        return cls(url, method=init.method, headers=init.headers, **options)


def _to_js(data: Any, **_kwargs: Any) -> Any:
    # Identity: the fakes round-trip attachments as plain Python, and nothing in a
    # test crosses a real JS boundary.
    return data


def _create_proxy(obj: Any) -> Any:
    return obj


def _set_timeout(callback: Any, _delay: int, *args: Any) -> int:
    asyncio.get_running_loop().call_soon(callback, *args)
    return 0


def install() -> None:
    if "js" in sys.modules and getattr(sys.modules["js"], "_agents_stub", False):
        return

    js = types.ModuleType("js")
    setattr(js, "_agents_stub", True)
    setattr(js, "Object", _StubObject)
    setattr(js, "ReadableStream", _StubReadableStream)
    setattr(js, "TextEncoder", _StubTextEncoder)
    setattr(js, "Response", _StubJsResponse)
    setattr(js, "Request", _StubRequest)
    setattr(js, "WebSocketPair", types.SimpleNamespace(new=lambda: None))
    setattr(js, "setTimeout", _set_timeout)
    sys.modules["js"] = js

    workers = types.ModuleType("workers")
    setattr(workers, "_agents_stub", True)
    setattr(workers, "DurableObject", _StubDurableObject)
    setattr(workers, "WorkflowEntrypoint", _StubWorkflowEntrypoint)
    setattr(workers, "Request", _StubRequest)
    setattr(workers, "Response", _StubResponse)
    setattr(workers, "waitUntil", _default_wait_until)
    sys.modules["workers"] = workers

    pyodide = types.ModuleType("pyodide")
    setattr(pyodide, "_agents_stub", True)
    sys.modules["pyodide"] = pyodide

    ffi = types.ModuleType("pyodide.ffi")
    setattr(ffi, "_agents_stub", True)
    setattr(ffi, "to_js", _to_js)
    setattr(ffi, "create_proxy", _create_proxy)
    setattr(ffi, "create_once_callable", _create_proxy)
    setattr(ffi, "JsProxy", object)
    sys.modules["pyodide.ffi"] = ffi
    setattr(pyodide, "ffi", ffi)
