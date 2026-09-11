from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator, Iterable

from ..core.utils import maybe_aiter
from .types import ChatReplyT, ChunkT

# id used for the text block _normalize opens around plain strings
_TEXT_ID = "0"


async def _chat_items(reply: ChatReplyT) -> AsyncIterator[str | ChunkT]:
    if isinstance(reply, str):
        yield reply
        return

    if not isinstance(reply, (Iterable, AsyncIterable)):
        raise TypeError(
            "on_chat_message must return a str, or an (async) iterable of str or "
            f"chunk dicts, got {type(reply).__name__}"
        )

    async for item in maybe_aiter(reply):
        yield item


async def _normalize(
    reply: ChatReplyT,
    text_id: str = _TEXT_ID,
) -> AsyncIterator[ChunkT]:
    # str items become text deltas in a lazily opened text block; dicts pass through
    # as raw chunks. Mixing is fine — the text block closes before any dict.
    text_open = False

    async for item in _chat_items(reply):
        if isinstance(item, str):
            if not text_open:
                yield {"type": "text-start", "id": text_id}
                text_open = True
            yield {"type": "text-delta", "id": text_id, "delta": item}
            continue

        if text_open:
            yield {"type": "text-end", "id": text_id}
            text_open = False
        yield item

    if text_open:
        yield {"type": "text-end", "id": text_id}
