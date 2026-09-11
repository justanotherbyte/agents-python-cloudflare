from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class TurnResult[T]:
    status: Literal["completed", "stale"]
    value: T | None = None


@dataclass(frozen=True)
class TurnContext:
    request_id: str
    generation: int


class TurnReentryError(RuntimeError):
    pass


type _TurnLease = tuple[TurnQueue, TurnContext]


_executing_turn_queues: ContextVar[tuple[_TurnLease, ...]] = ContextVar(
    "executing_turn_queues",
    default=(),
)


class TurnQueue:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._generation = 0
        self._active_context: TurnContext | None = None

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def active_request_id(self) -> str | None:
        if self._active_context is None:
            return None
        return self._active_context.request_id

    def is_current(self, context: TurnContext) -> bool:
        return (
            self._active_context is context and context.generation == self._generation
        )

    async def enqueue[T](
        self,
        request_id: str,
        fn: Callable[[TurnContext], Awaitable[T]],
    ) -> TurnResult[T]:
        if any(
            queue is self and context is self._active_context
            for queue, context in _executing_turn_queues.get()
        ):
            raise TurnReentryError("chat turn queue cannot re-enter itself")

        captured = self._generation
        async with self._lock:
            if captured != self._generation:
                return TurnResult(status="stale")

            context = TurnContext(request_id, captured)
            self._active_context = context
            token = _executing_turn_queues.set(
                (*_executing_turn_queues.get(), (self, context))
            )
            try:
                return TurnResult(status="completed", value=await fn(context))
            finally:
                _executing_turn_queues.reset(token)
                self._active_context = None

    def reset(self) -> None:
        self._generation += 1
