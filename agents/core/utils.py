# TODO: move to agents/utils.py rather than agents/core/utils.py

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterable, AsyncIterator, Iterable
from typing import Any, TypeVar
from urllib.parse import urlsplit
from uuid import uuid4


T = TypeVar("T")


# Separates "no value supplied" from "an explicit null". Shared rather than per-module
# so a value can pass between layers and still compare identical.
class _Missing:
    def __repr__(self) -> str:
        return "<MISSING>"


MISSING: Any = _Missing()


def now_ms() -> int:
    # Epoch milliseconds, so a row written by either runtime reads the same.
    return int(time.time() * 1000)


def gen_id() -> str:
    return uuid4().hex


def noop() -> None:
    pass


async def maybe_aiter[T](
    values: Iterable[T] | AsyncIterable[T],
) -> AsyncIterator[T]:
    if isinstance(values, AsyncIterable):
        async for value in values:
            yield value
        return

    for value in values:
        yield value


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(value, high))


def dumps_wire(data: Any) -> str:
    # For anything leaving Python, persisted or sent. allow_nan=False keeps NaN and
    # Infinity out of it — see AGENTS.md, wire invariants.
    return json.dumps(data, allow_nan=False, separators=(",", ":"))


def url_path(url: str) -> str:
    # urlsplit, not urlparse: urlparse also strips a ";" and everything after it off the
    # last path segment, and an agent name reaches us unescaped — see AGENTS.md.
    return urlsplit(url).path


def url_with_path(url: str, path: str) -> str:
    return urlsplit(url)._replace(path=path).geturl()


def loads_or_none(text: Any) -> Any:
    # SQL NULL, unparseable text and literal "null" all collapse to None. The
    # distinction is kept only where it changes persisted state — see
    # design/PORTING_FIBERS.md.
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return None


def loads_dict_or_none(text: Any) -> dict[str, Any] | None:
    # For payloads that are contractually objects, so a scalar or array degrades to
    # absent. An empty object is valid, so this tests shape rather than truth.
    parsed = loads_or_none(text)
    return parsed if isinstance(parsed, dict) else None


def error_message(error: Any) -> str:
    # Deliberately not bare str(): a no-argument raise stringifies to "", which would
    # persist an error row with nothing in it. The reference has the same hole.
    return str(error) or "Unknown error occurred"
