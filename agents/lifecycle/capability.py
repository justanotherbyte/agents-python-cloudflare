from __future__ import annotations

from types import MemberDescriptorType
from typing import ClassVar, cast

from workers import Response

from .jobs import (
    LifecycleJobContext,
    LifecycleJobOutcome,
    LifecycleMemoryLimitContext,
)
from .types import (
    CapabilityRequestContext,
    CapabilityWebSocketCloseContext,
    CapabilityWebSocketErrorContext,
    CapabilityWebSocketMessageContext,
    CapabilityWebSocketUpgradeContext,
    LifecycleRouteContext,
    LifecycleServices,
)


_SERVICE_BINDING = "_agents_lifecycle_services"


class LifecycleCapability:
    """Experimental base for a stateful capability installed on a Lifecycle."""

    __slots__ = (_SERVICE_BINDING,)

    capability_id: ClassVar[str] = ""

    @property
    def lifecycle(self) -> LifecycleServices:
        """Bound, capability-scoped services available after `Lifecycle.use`."""
        try:
            return _SERVICE_SLOT.__get__(self, LifecycleCapability)
        except AttributeError:
            raise RuntimeError("capability is not installed on a Lifecycle")

    async def on_start(self) -> None:
        pass

    async def on_request(
        self,
        context: CapabilityRequestContext,
    ) -> Response | None:
        return None

    async def on_websocket_upgrade(
        self,
        context: CapabilityWebSocketUpgradeContext,
    ) -> Response | None:
        return None

    async def on_websocket_message(
        self,
        context: CapabilityWebSocketMessageContext,
    ) -> bool | None:
        return False

    async def on_websocket_close(
        self,
        context: CapabilityWebSocketCloseContext,
    ) -> bool | None:
        return False

    async def on_websocket_error(
        self,
        context: CapabilityWebSocketErrorContext,
    ) -> bool | None:
        return False

    async def on_route(self, context: LifecycleRouteContext) -> object:
        raise LookupError(f"capability {self.capability_id} cannot receive routes")

    async def on_job(self, context: LifecycleJobContext) -> LifecycleJobOutcome:
        return None

    async def on_job_error(
        self,
        context: LifecycleJobContext,
        error: BaseException,
    ) -> LifecycleJobOutcome:
        return None

    async def on_memory_limit(self, context: LifecycleMemoryLimitContext) -> None:
        pass

    async def on_dispose(self) -> None:
        pass


_SERVICE_SLOT = cast(
    MemberDescriptorType,
    type.__getattribute__(LifecycleCapability, "__dict__")[_SERVICE_BINDING],
)
