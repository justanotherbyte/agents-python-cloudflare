from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, overload

from workers import Request

from .jobs import LifecycleJobs, LifecycleMemoryLimitContext


@dataclass(frozen=True, slots=True)
class CapabilityRequestContext:
    request: Request


@dataclass(frozen=True, slots=True)
class CapabilityWebSocketUpgradeContext:
    request: Request


@dataclass(frozen=True, slots=True)
class CapabilityWebSocketMessageContext:
    websocket: object
    message: object


@dataclass(frozen=True, slots=True)
class CapabilityWebSocketCloseContext:
    websocket: object
    code: int
    reason: str
    was_clean: bool


@dataclass(frozen=True, slots=True)
class CapabilityWebSocketErrorContext:
    websocket: object
    error: object


@dataclass(frozen=True, slots=True)
class LifecycleHostContextScope:
    request: Request | None = None
    connection: object | None = None


@dataclass(frozen=True, slots=True)
class CurrentLifecycleContext:
    host: object
    request: Request | None = None
    connection: object | None = None


@dataclass(frozen=True, slots=True)
class LifecycleEvent:
    source: str
    type: str
    payload: object


@dataclass(frozen=True, slots=True)
class LifecycleRouteAddress:
    key: str
    data: str = field(compare=False)

    def __post_init__(self) -> None:
        if type(self.key) is not str or not self.key.strip():
            raise ValueError("route address key must be a non-empty string")
        if type(self.data) is not str:
            raise ValueError("route address data must be a string")


@dataclass(frozen=True, slots=True)
class LifecycleRouteContext:
    source: LifecycleRouteAddress | None
    payload: object


@dataclass(frozen=True, slots=True)
class LifecycleRouteEnvelope:
    version: int
    source: LifecycleRouteAddress | None
    target: LifecycleRouteAddress
    capability_id: str
    payload: object


class LifecycleRouteTransport(Protocol):
    def __call__(
        self,
        envelope: LifecycleRouteEnvelope,
    ) -> object | Awaitable[object]: ...


class LifecycleStorage(Protocol):
    @overload
    async def get(self, key: str) -> Any: ...

    @overload
    async def get(self, key: Sequence[str]) -> dict[str, Any]: ...

    @overload
    async def put(self, key: str, value: object) -> None: ...

    @overload
    async def put(self, key: dict[str, Any]) -> None: ...

    @overload
    async def delete(self, key: str) -> bool: ...

    @overload
    async def delete(self, key: Sequence[str]) -> int: ...

    async def list(
        self,
        *,
        prefix: str = "",
        start: str | None = None,
        start_after: str | None = None,
        end: str | None = None,
        reverse: bool = False,
        limit: int | None = None,
    ) -> dict[str, Any]: ...

    def transaction_sync[T](self, callback: Callable[[], T]) -> T: ...


class LifecycleSql(Protocol):
    def execute(self, query: str, *params: object) -> list[dict[str, Any]]: ...


class LifecycleSockets(Protocol):
    def accept(
        self,
        websocket: object,
        *,
        tags: Sequence[str] = (),
    ) -> None: ...

    def get(self, *, tag: str | None = None) -> tuple[object, ...]: ...

    def serialize_attachment(self, websocket: object, value: object) -> None: ...

    def deserialize_attachment(self, websocket: object) -> object | None: ...


class LifecycleEvents(Protocol):
    async def emit(self, type: str, payload: object) -> None: ...


class LifecycleRoutes(Protocol):
    @property
    def source(self) -> LifecycleRouteAddress | None: ...

    async def to_root(self, payload: object) -> object: ...

    async def to(
        self,
        target: LifecycleRouteAddress,
        payload: object,
    ) -> object: ...


class LifecycleRetainedWork(Protocol):
    @property
    def available(self) -> bool: ...

    def retain(self, factory: Callable[[], Awaitable[object]]) -> None: ...


class LifecycleServices(Protocol):
    @property
    def storage(self) -> LifecycleStorage: ...

    @property
    def sql(self) -> LifecycleSql: ...

    @property
    def sockets(self) -> LifecycleSockets: ...

    @property
    def events(self) -> LifecycleEvents: ...

    @property
    def routes(self) -> LifecycleRoutes: ...

    @property
    def retained_work(self) -> LifecycleRetainedWork: ...

    def track_alarm_work(self, awaitable: Awaitable[object]) -> bool: ...

    def _alarm_work_is_tracked(self, awaitable: Awaitable[object]) -> bool: ...

    async def _notify_host_memory_limit(
        self,
        context: LifecycleMemoryLimitContext,
    ) -> None: ...

    @property
    def jobs(self) -> LifecycleJobs: ...

    @property
    def owns_physical_alarm(self) -> bool: ...

    @property
    def starting(self) -> bool: ...

    @property
    def is_ready(self) -> bool: ...

    async def ready(self) -> None: ...

    async def run_in_host_context[T](
        self,
        callback: Callable[[], T | Awaitable[T]],
        *,
        scope: LifecycleHostContextScope | None = None,
    ) -> T: ...
