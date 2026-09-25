from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from .protocol import stream_resume_none_frame


@dataclass(slots=True)
class ContinuationRequest:
    connection: Any
    connection_id: str | None
    request_id: str
    body: dict[str, Any]
    client_tools: list[Any] | None
    past_coalesce: bool = False


class AutoContinuationController:
    """Coalesce tool answers and fire once the leaf tool batch is complete."""

    COALESCE_SECONDS = 0.05

    def __init__(
        self,
        *,
        generate_request_id: Callable[[], str],
        is_stream_active: Callable[[], bool],
        has_pending_interaction: Callable[[], bool],
        has_incomplete_tool_batch: Callable[[], bool],
        drain_interactions: Callable[[], Awaitable[None]],
        fire: Callable[[ContinuationRequest], Awaitable[None]],
        spawn: Callable[[Callable[[], Awaitable[None]]], None],
    ) -> None:
        self._generate_request_id = generate_request_id
        self._is_stream_active = is_stream_active
        self._has_pending_interaction = has_pending_interaction
        self._has_incomplete_tool_batch = has_incomplete_tool_batch
        self._drain_interactions = drain_interactions
        self._fire = fire
        self._spawn = spawn
        self.pending: ContinuationRequest | None = None
        self.deferred: ContinuationRequest | None = None
        self.active_request_id: str | None = None
        self.active_connection_id: str | None = None
        self.awaiting_connections: dict[str, tuple[Any, Any]] = {}
        self._timer: asyncio.TimerHandle | None = None
        self._barrier_active = False

    def schedule(
        self,
        connection: Any,
        body: dict[str, Any],
        client_tools: list[Any] | None,
    ) -> None:
        pending = self.pending
        if pending is not None and pending.past_coalesce:
            if self.deferred is not None:
                self.awaiting_connections.pop(self.deferred.connection.id, None)
            self.deferred = ContinuationRequest(
                connection,
                connection.id,
                self._generate_request_id(),
                body,
                client_tools,
            )
            self.awaiting_connections[connection.id] = (connection, None)
            return
        if pending is None:
            pending = ContinuationRequest(
                connection,
                connection.id,
                self._generate_request_id(),
                body,
                client_tools,
            )
            self.pending = pending
        else:
            pending.connection = connection
            pending.connection_id = connection.id
            pending.body = body
            pending.client_tools = client_tools
        self.awaiting_connections[connection.id] = (connection, None)
        self.arm()

    def rearm(self) -> None:
        if self.pending is not None and not self.pending.past_coalesce:
            self.arm()

    def arm(self) -> None:
        self.cancel_timer()
        loop = asyncio.get_running_loop()
        self._timer = loop.call_later(self.COALESCE_SECONDS, self._on_timer)

    def _on_timer(self) -> None:
        self._timer = None
        if self.pending is not None:
            self._spawn(self.fire_when_stable)

    async def fire_when_stable(self) -> None:
        pending = self.pending
        if pending is None or pending.past_coalesce or self._barrier_active:
            return
        if self._is_stream_active():
            return
        if self._has_pending_interaction():
            self._barrier_active = True
            try:
                await self._drain_interactions()
            finally:
                self._barrier_active = False
            pending = self.pending
            if pending is None or pending.past_coalesce or self._is_stream_active():
                return
        if self._has_incomplete_tool_batch():
            return
        self.cancel_timer()
        pending.past_coalesce = True
        await self._fire(pending)

    def activate(self, request_id: str) -> None:
        pending = self.pending
        if pending is None or pending.request_id != request_id:
            return
        self.active_request_id = request_id
        self.active_connection_id = pending.connection_id
        self.pending = None

    def finish(self, request_id: str) -> None:
        if self.active_request_id == request_id:
            self.active_request_id = None
            self.active_connection_id = None
        if self.pending is None and self.deferred is not None:
            self.pending = self.deferred
            self.deferred = None
            if self.pending.connection_id is not None:
                self.awaiting_connections[self.pending.connection_id] = (
                    self.pending.connection,
                    None,
                )
            self.rearm()

    def release_connection(self, connection_id: str) -> None:
        self.awaiting_connections.pop(connection_id, None)
        if self.pending is not None and self.pending.connection_id == connection_id:
            self.pending.connection_id = None
        if self.deferred is not None and self.deferred.connection_id == connection_id:
            self.deferred.connection_id = None
        if self.active_connection_id == connection_id:
            self.active_connection_id = None

    def settle_waiters(self) -> None:
        awaiting = tuple(self.awaiting_connections.values())
        self.awaiting_connections.clear()
        for connection, probe_id in awaiting:
            connection.send_if_open(stream_resume_none_frame(probe_id))

    def cancel_timer(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def reset(self) -> None:
        self.cancel_timer()
        self._barrier_active = False
        self.settle_waiters()
        self.pending = None
        self.deferred = None
        self.active_request_id = None
        self.active_connection_id = None
        self.awaiting_connections.clear()
