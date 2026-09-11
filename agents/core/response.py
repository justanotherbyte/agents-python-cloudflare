from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

from .protocol import FrameT, rpc_chunk, rpc_error, rpc_result
from .utils import MISSING

if TYPE_CHECKING:
    from ..lifecycle.websockets import Connection


class StreamingResponse:
    def __init__(self, connection: Connection, rpc_id: str):
        self._connection = connection
        self._rpc_id = rpc_id
        self._closed = False

    @property
    def is_closed(self) -> bool:
        return self._closed

    def send(self, chunk: Any) -> bool:
        if self._closed:
            warnings.warn(
                "StreamingResponse.send() called after stream was closed - "
                "data not sent"
            )
            return False

        return self._connection.send_if_open(rpc_chunk(self._rpc_id, chunk))

    def _close_with(self, frame: FrameT) -> bool:
        # Flips closed before sending, so a send that fails still leaves the stream shut
        # and a later end() cannot append a second terminal.
        if self._closed:
            return False

        self._closed = True
        return self._connection.send_if_open(frame)

    def end(self, final_chunk: Any = MISSING) -> bool:
        return self._close_with(rpc_result(self._rpc_id, final_chunk))

    def error(self, message: str) -> bool:
        return self._close_with(rpc_error(self._rpc_id, message))
