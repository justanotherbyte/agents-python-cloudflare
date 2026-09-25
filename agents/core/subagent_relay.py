from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, MutableMapping
from contextlib import suppress
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, NotRequired, Protocol, TypedDict

from .connection_state import (
    connection_state_flags,
    connection_user_state,
    merge_connection_state,
)
from .utils import dumps_wire


class RelayPathStep(TypedDict):
    className: str
    name: str


class RelayTarget(TypedDict):
    url: str
    headers: dict[str, str]
    path: NotRequired[list[RelayPathStep]]


class RelayLimitError(RuntimeError):
    pass


@dataclass(frozen=True)
class RelayLimits:
    max_frames: int
    max_bytes: int


@dataclass(frozen=True)
class SendOperation:
    data: str


@dataclass(frozen=True)
class StateOperation:
    data: object


@dataclass(frozen=True)
class TagsOperation:
    data: tuple[str, ...]


@dataclass(frozen=True)
class CloseOperation:
    code: int
    reason: str


type RelayOperation = SendOperation | StateOperation | TagsOperation | CloseOperation


class RelaySink(Protocol):
    def send_if_open(self, data: str) -> bool: ...

    def _set_raw_state(self, data: object) -> object: ...

    def _insert_tags(self, tags: list[str]) -> None: ...

    def close(self, code: int = 1000, reason: str = "") -> None: ...


def _operation_wire(operation: RelayOperation) -> dict[str, Any]:
    if isinstance(operation, SendOperation):
        return {"type": "send", "data": operation.data}
    if isinstance(operation, StateOperation):
        return {"type": "state", "data": operation.data}
    if isinstance(operation, TagsOperation):
        return {"type": "tags", "data": list(operation.data)}
    return {"type": "close", "code": operation.code, "reason": operation.reason}


# This virtual connection records bounded operations instead of owning a socket.
class BufferedRelayConnection:
    def __init__(
        self,
        connection_id: str,
        *,
        physical_key: str | None = None,
        state: Any,
        tags: list[str],
        max_frames: int,
        max_bytes: int,
    ) -> None:
        self.id = connection_id
        self._physical_key = physical_key
        self._state = deepcopy(state)
        self._tags = list(tags)
        self._limits = RelayLimits(max_frames, max_bytes)
        self._operation_bytes = 0
        self._closed = False
        self.operations: list[RelayOperation] = []

    def _append(self, operation: RelayOperation) -> None:
        if len(self.operations) >= self._limits.max_frames:
            raise RelayLimitError("sub-agent relay frame limit exceeded")
        operation_bytes = len(dumps_wire(_operation_wire(operation)).encode("utf-8"))
        list_bytes = 2 + self._operation_bytes + operation_bytes + len(self.operations)
        if list_bytes > self._limits.max_bytes:
            raise RelayLimitError("sub-agent relay byte limit exceeded")
        self._operation_bytes += operation_bytes
        self.operations.append(operation)

    def send(self, data: str) -> None:
        if self._closed:
            raise RuntimeError("WebSocket send() after close")
        self._append(SendOperation(data))

    def send_json(self, data: dict[str, Any]) -> None:
        self.send(dumps_wire(data))

    def send_if_open(self, data: str | dict[str, Any]) -> bool:
        if self._closed:
            return False
        self.send(data if isinstance(data, str) else dumps_wire(data))
        return True

    def set_state(self, data: object) -> object:
        if self._closed:
            raise RuntimeError("WebSocket state update after close")
        flags = connection_state_flags(self._state)
        current = connection_user_state(self._state)
        value = data(deepcopy(current)) if callable(data) else data
        updated = json.loads(dumps_wire(value))
        self._set_raw_state(merge_connection_state(updated, flags))
        return deepcopy(updated)

    def _get_raw_state(self) -> Any:
        return deepcopy(self._state)

    def _set_raw_state(self, data: object) -> object:
        if self._closed:
            raise RuntimeError("WebSocket state update after close")
        updated = json.loads(dumps_wire(data))
        self._append(StateOperation(updated))
        self._state = updated
        return deepcopy(updated)

    @property
    def state(self) -> Any:
        return deepcopy(connection_user_state(self._state))

    def _insert_tags(self, tags: list[str]) -> None:
        if self._closed:
            raise RuntimeError("WebSocket tag update after close")
        if not all(isinstance(tag, str) for tag in tags):
            raise TypeError("WebSocket tags must be strings")
        updated = list(tags)
        self._append(TagsOperation(tuple(updated)))
        self._tags = updated

    @property
    def tags(self) -> list[str]:
        return list(self._tags)

    def close(self, code: int = 1000, reason: str = "") -> None:
        if self._closed:
            return
        if isinstance(code, bool) or not isinstance(code, int):
            raise TypeError("WebSocket close code must be an integer")
        if not isinstance(reason, str):
            raise TypeError("WebSocket close reason must be a string")
        self._append(CloseOperation(code, reason))
        self._closed = True

    def operation_log(self) -> str:
        return dumps_wire([_operation_wire(operation) for operation in self.operations])


class RelaySession:
    def __init__(
        self,
        connections: MutableMapping[str, Any],
        connection: BufferedRelayConnection,
        *,
        key: str | None = None,
    ) -> None:
        self._connections = connections
        self.connection = connection
        self._key = key or connection.id

    def __enter__(self) -> BufferedRelayConnection:
        self._connections[self._key] = self.connection
        return self.connection

    def __exit__(self, exc_type, exc, traceback) -> None:
        try:
            if self._connections.get(self._key) is self.connection:
                self._connections.pop(self._key, None)
        except BaseException:
            if exc is None:
                raise


def decode_operation_log(raw: Any, limits: RelayLimits) -> list[RelayOperation]:
    if not isinstance(raw, str):
        raise TypeError("sub-agent relay returned a non-string operation log")
    if len(raw.encode("utf-8")) > limits.max_bytes:
        raise RelayLimitError("sub-agent relay byte limit exceeded")

    parsed = json.loads(raw)
    if not isinstance(parsed, list):
        raise TypeError("sub-agent relay returned an invalid operation log")
    if len(parsed) > limits.max_frames:
        raise RelayLimitError("sub-agent relay frame limit exceeded")

    operations: list[RelayOperation] = []
    closed = False
    for item in parsed:
        if closed:
            raise TypeError("sub-agent relay operation follows close")
        operation = _decode_operation(item)
        operations.append(operation)
        closed = isinstance(operation, CloseOperation)
    return operations


def _decode_operation(value: Any) -> RelayOperation:
    if not isinstance(value, dict):
        raise TypeError("sub-agent relay returned an invalid operation")
    kind = value.get("type")
    if kind == "send":
        if set(value) != {"type", "data"} or not isinstance(value["data"], str):
            raise TypeError("sub-agent relay returned an invalid send operation")
        return SendOperation(value["data"])
    if kind == "state":
        if set(value) != {"type", "data"}:
            raise TypeError("sub-agent relay returned an invalid state operation")
        return StateOperation(value["data"])
    if kind == "tags":
        data = value.get("data")
        if (
            set(value) != {"type", "data"}
            or not isinstance(data, list)
            or not all(isinstance(tag, str) for tag in data)
        ):
            raise TypeError("sub-agent relay returned an invalid tags operation")
        return TagsOperation(tuple(data))
    if kind == "close":
        code = value.get("code")
        reason = value.get("reason")
        if (
            set(value) != {"type", "code", "reason"}
            or isinstance(code, bool)
            or not isinstance(code, int)
            or not isinstance(reason, str)
        ):
            raise TypeError("sub-agent relay returned an invalid close operation")
        return CloseOperation(code, reason)
    raise TypeError("sub-agent relay returned an unknown operation")


def apply_operations(sink: RelaySink, operations: list[RelayOperation]) -> None:
    for operation in operations:
        if isinstance(operation, SendOperation):
            if not sink.send_if_open(operation.data):
                return
        elif isinstance(operation, StateOperation):
            sink._set_raw_state(operation.data)
        elif isinstance(operation, TagsOperation):
            sink._insert_tags(list(operation.data))
        else:
            sink.close(operation.code, operation.reason)
            return


async def relay_round_trip(
    forward: Callable[[], Awaitable[Any]],
    sink: RelaySink,
    limits: RelayLimits,
    timeout: float,
) -> None:
    try:
        raw = await asyncio.wait_for(forward(), timeout=timeout)
        operations = decode_operation_log(raw, limits)
        apply_operations(sink, operations)
    except BaseException:
        with suppress(BaseException):
            sink.close(1011, "Sub-agent relay failed")
        raise
