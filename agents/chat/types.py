from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable, Iterable
from typing import Any

ChunkT = dict[str, Any]
MessageT = dict[str, Any]

# what on_chat_message is allowed to hand back
ChatReplyT = str | Iterable[str | ChunkT] | AsyncIterable[str | ChunkT]


class ChatOptions:
    def __init__(
        self,
        request_id: str,
        trigger: str,
        body: dict[str, Any],
        abort: asyncio.Event,
        *,
        continuation: bool = False,
        client_tools: list[Any] | None = None,
    ):
        self.request_id = request_id
        self.trigger = trigger
        self.body = body
        self.abort = abort
        self.continuation = continuation
        self.client_tools = client_tools

    @property
    def aborted(self) -> bool:
        return self.abort.is_set()

    def __repr__(self) -> str:
        return f"<ChatOptions request_id={self.request_id!r} trigger={self.trigger!r}>"
