from __future__ import annotations

import inspect
from collections.abc import Callable
from contextvars import ContextVar
from typing import Any

from .types import CurrentLifecycleContext


_CURRENT_LIFECYCLE_CONTEXT: ContextVar[CurrentLifecycleContext | None] = ContextVar(
    "agents_current_lifecycle_context",
    default=None,
)


def get_current_lifecycle_context() -> CurrentLifecycleContext | None:
    return _CURRENT_LIFECYCLE_CONTEXT.get()


async def _call_maybe_async(callback: Callable[..., Any], *args: object) -> Any:
    result = callback(*args)
    if inspect.isawaitable(result):
        return await result
    return result


async def _call_with_context(
    context: CurrentLifecycleContext | None,
    callback: Callable[..., Any],
    *args: object,
) -> Any:
    token = _CURRENT_LIFECYCLE_CONTEXT.set(context)
    try:
        return await _call_maybe_async(callback, *args)
    finally:
        _CURRENT_LIFECYCLE_CONTEXT.reset(token)
