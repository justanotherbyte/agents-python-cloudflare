from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from .types import BrowserRequest


class BrowserSocketSubscription(Protocol):
    """A retained socket listener that can be released exactly once."""

    def cancel(self) -> object: ...


class BrowserSocket(Protocol):
    """Host adapter for a Browser Run WebSocket."""

    def accept(self) -> object: ...

    def send(self, data: str) -> object | Awaitable[object]: ...

    def close(
        self, code: int = 1000, reason: str = ""
    ) -> object | Awaitable[object]: ...

    def subscribe(
        self,
        event: str,
        callback: Callable[[object], None],
    ) -> BrowserSocketSubscription: ...


class BrowserResponse(Protocol):
    """Host-neutral response returned by Browser Run transports."""

    @property
    def status(self) -> int: ...

    @property
    def headers(self) -> Mapping[str, str]: ...

    @property
    def websocket(self) -> BrowserSocket | None: ...

    async def read(self) -> bytes: ...

    async def aclose(self) -> None: ...


class BrowserTransport(Protocol):
    """Injected Browser Run binding or HTTP adapter."""

    async def fetch(self, request: BrowserRequest) -> BrowserResponse: ...


class QuickActionBinding(Protocol):
    """Injected Browser Run binding exposing Quick Actions."""

    async def quick_action(
        self,
        action: str,
        params: Mapping[str, object],
    ) -> BrowserResponse: ...


class BrowserStorage(Protocol):
    """Minimal durable key-value storage used by browser sessions."""

    async def get(self, key: str) -> object | None: ...

    async def put(self, key: str, value: object) -> None: ...

    async def delete(self, key: str) -> object: ...

    async def list(self, *, prefix: str) -> Mapping[str, object]: ...
