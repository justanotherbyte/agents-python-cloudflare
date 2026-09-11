from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .protocol import stream_pending_frame, stream_resume_none_frame
from ..lifecycle.websockets import Connection


class PreStreamTurns:
    def __init__(self) -> None:
        self._accepted: dict[str, None] = {}
        self._awaiting: dict[str, tuple[Connection, Any]] = {}

    @property
    def latest_request_id(self) -> str | None:
        return next(reversed(self._accepted), None)

    def begin(self, request_id: str) -> None:
        self._accepted[request_id] = None

    def settle(self, request_id: str) -> bool:
        self._accepted.pop(request_id, None)
        return not self._accepted

    def park(self, connection: Connection, probe_id: Any = None) -> bool:
        request_id = self.latest_request_id
        if request_id is None:
            return False
        self._awaiting[connection.id] = (connection, probe_id)
        connection.send_if_open(stream_pending_frame(request_id, probe_id))
        return True

    def release(self, connection_id: str) -> None:
        self._awaiting.pop(connection_id, None)

    def flush_on_stream_start(
        self,
        notify: Callable[[Connection], None],
    ) -> None:
        awaiting = tuple(self._awaiting.values())
        self._awaiting.clear()
        for connection, _probe_id in awaiting:
            notify(connection)

    def release_awaiting(self) -> None:
        awaiting = tuple(self._awaiting.values())
        self._awaiting.clear()
        for connection, probe_id in awaiting:
            connection.send_if_open(stream_resume_none_frame(probe_id))

    def reset(self) -> None:
        self._accepted.clear()
        self._awaiting.clear()
