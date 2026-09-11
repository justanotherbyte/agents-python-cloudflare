from __future__ import annotations

import asyncio
import inspect
import json
import traceback
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from contextlib import asynccontextmanager, suppress
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, NamedTuple
from urllib.parse import parse_qs, urlsplit

from js import Object, WebSocketPair  # ty: ignore[unresolved-import]
from pyodide.ffi import to_js
from workers import Request, Response

from ..core.connection_state import (
    connection_state_flags,
    connection_user_state,
    merge_connection_state,
)
from ..core.error import HookError
from ..core.subagent_relay import RelayLimits, RelayTarget, relay_round_trip
from ..core.utils import gen_id
from .capability import LifecycleCapability
from .types import (
    CapabilityWebSocketCloseContext,
    CapabilityWebSocketErrorContext,
    CapabilityWebSocketMessageContext,
    CapabilityWebSocketUpgradeContext,
    LifecycleHostContextScope,
)

type JsonT = dict[str, Any]
type Attachment = dict[str, Any]

_CURRENT_WEBSOCKET_CONNECTION: ContextVar[tuple[object, object | None] | None] = (
    ContextVar("agents_current_websocket_connection", default=None)
)


def _clone_json(value: object) -> object:
    return json.loads(json.dumps(value, allow_nan=False, separators=(",", ":")))


def _current_websocket_connection(owner: object) -> object | None:
    current = _CURRENT_WEBSOCKET_CONNECTION.get()
    if current is None or current[0] is not owner:
        return None
    return current[1]


def _utf16_length(value: str) -> int:
    return sum(2 if ord(character) > 0xFFFF else 1 for character in value)


def _prepare_tags(connection_id: str, tags: Iterable[object]) -> list[str]:
    if _utf16_length(connection_id) > 256:
        raise ValueError("A connection tag must not exceed 256 characters")
    if isinstance(tags, str):
        raise ValueError("A connection tag collection must contain strings")
    prepared = [connection_id]
    for tag in tags:
        if not isinstance(tag, str):
            raise ValueError(f"A connection tag must be a string. Received: {tag}")
        if tag == connection_id:
            continue
        if not tag:
            raise ValueError("A connection tag must not be empty")
        if _utf16_length(tag) > 256:
            raise ValueError("A connection tag must not exceed 256 characters")
        prepared.append(tag)
    if len(prepared) > 10:
        raise ValueError(
            "A connection can only have 10 tags, including the default id tag"
        )
    return prepared


def _decode_relay_target(value: object) -> RelayTarget | None:
    if not isinstance(value, dict):
        return None
    url = value.get("url")
    headers = value.get("headers")
    if not isinstance(url, str) or not url or not isinstance(headers, dict):
        return None
    if not all(
        isinstance(key, str) and isinstance(item, str) for key, item in headers.items()
    ):
        return None
    return RelayTarget(url=url, headers=headers)


@dataclass(slots=True)
class _ConnectionRecord:
    id: str
    tags: list[str]
    state: object
    uri: str | None = None
    platform_extensions: dict[str, object] = field(default_factory=dict)
    extensions: dict[str, object] = field(default_factory=dict)

    def attachment(self) -> Attachment:
        platform = {
            **self.platform_extensions,
            "id": self.id,
            "tags": list(self.tags),
        }
        if self.uri is not None:
            platform["uri"] = self.uri
        return {
            **self.extensions,
            "__pk": platform,
            "__user": self.state,
        }


def _decode_attachment(value: object) -> _ConnectionRecord | None:
    if not isinstance(value, Mapping):
        return None

    raw = dict(value)
    target = raw.get("target")
    if target is not None and _decode_relay_target(target) is None:
        return None
    if "__pk" in raw:
        platform_value = raw.get("__pk")
        if not isinstance(platform_value, Mapping):
            return None
        platform = dict(platform_value)
        connection_id = platform.get("id")
        if not isinstance(connection_id, str) or not connection_id:
            return None

        stored_tags = platform.get("tags")
        if stored_tags is None:
            stored_tags = []
        if not isinstance(stored_tags, list):
            return None
        tags = _prepare_tags(connection_id, stored_tags)
        uri = platform.get("uri")
        return _ConnectionRecord(
            id=connection_id,
            tags=tags,
            state=raw.get("__user"),
            uri=uri if isinstance(uri, str) else None,
            platform_extensions={
                key: item
                for key, item in platform.items()
                if key not in {"id", "tags", "uri"}
            },
            extensions={
                key: item for key, item in raw.items() if key not in {"__pk", "__user"}
            },
        )

    if not {"id", "state", "tags"}.issubset(raw):
        return None
    connection_id = raw.get("id")
    if not isinstance(connection_id, str) or not connection_id:
        return None
    stored_tags = raw.get("tags")
    tags = stored_tags if isinstance(stored_tags, list) else []
    return _ConnectionRecord(
        id=connection_id,
        tags=_prepare_tags(connection_id, tags),
        state=raw.get("state"),
        extensions={
            key: item for key, item in raw.items() if key not in {"id", "state", "tags"}
        },
    )


def _read_record(websocket: object) -> _ConnectionRecord | None:
    deserialize = getattr(websocket, "deserializeAttachment")
    value = deserialize()
    if value is None:
        return None
    to_python = getattr(value, "to_py", None)
    if callable(to_python):
        value = to_python()
    return _decode_attachment(value)


def _write_attachment(websocket: object, attachment: Attachment) -> None:
    value = to_js(attachment, dict_converter=Object.fromEntries)
    getattr(websocket, "serializeAttachment")(value)


def attachment_id(websocket: object) -> str | None:
    try:
        record = _read_record(websocket)
    except Exception:  # noqa: BLE001
        return None
    return None if record is None else record.id


def _same_socket(left: object, right: object) -> bool:
    if left is right:
        return True
    try:
        return bool(left == right)
    except Exception:  # noqa: BLE001
        return False


def _as_exception(error: Any) -> BaseException:
    if isinstance(error, BaseException):
        return error
    return Exception(str(error))


async def _call_maybe_async(callback: Callable[..., Any], *args: Any) -> Any:
    result = callback(*args)
    if inspect.isawaitable(result):
        return await result
    return result


class ConnectionContext:
    def __init__(self, request: Request):
        self.request = request


class _SocketTurn(NamedTuple):
    ready: bool
    connection: Connection | None


class Connection:
    def __init__(
        self,
        id: str,
        server: Any,
        *,
        record: _ConnectionRecord | None = None,
        physical_key: str | None = None,
        persistent: bool = True,
        on_tags_changed: (
            Callable[[Connection, Sequence[str], Sequence[str]], None] | None
        ) = None,
    ) -> None:
        self.id = id
        self._server = server
        self._record = record
        self._physical_key = physical_key or gen_id()
        self._persistent = persistent
        self._on_tags_changed = on_tags_changed

    def _attachment_record(self) -> _ConnectionRecord:
        if self._record is None:
            self._record = _read_record(self._server)
        if self._record is None:
            raise RuntimeError("missing managed WebSocket attachment")
        return self._record

    def send(self, data: str) -> None:
        self._server.send(data)

    def send_json(self, data: JsonT) -> None:
        self.send(json.dumps(data))

    def send_if_open(self, data: str | JsonT) -> bool:
        payload = data if isinstance(data, str) else json.dumps(data)
        try:
            self.send(payload)
            return True
        except Exception as exc:
            if "WebSocket send() after close" in str(exc):
                return False
            raise

    def set_state(self, data: object) -> object:
        record = self._attachment_record()
        flags = connection_state_flags(record.state)
        current = connection_user_state(record.state)
        value = data(deepcopy(current)) if callable(data) else data
        committed = _clone_json(value)
        raw = merge_connection_state(committed, flags)
        self._set_raw_state(raw)
        return deepcopy(committed)

    def _get_raw_state(self) -> object:
        return deepcopy(self._attachment_record().state)

    def _set_raw_state(self, data: object) -> object:
        record = self._attachment_record()
        committed = _clone_json(data)
        candidate = _ConnectionRecord(
            id=record.id,
            tags=record.tags,
            state=committed,
            uri=record.uri,
            platform_extensions=record.platform_extensions,
            extensions=record.extensions,
        )
        if self._persistent:
            _write_attachment(self._server, candidate.attachment())
        self._record = candidate
        return deepcopy(committed)

    @property
    def state(self) -> object:
        return deepcopy(connection_user_state(self._attachment_record().state))

    def _insert_tags(self, tags: list[str]) -> None:
        record = self._attachment_record()
        previous_tags = record.tags
        candidate = _ConnectionRecord(
            id=record.id,
            tags=_prepare_tags(self.id, tags),
            state=record.state,
            uri=record.uri,
            platform_extensions=record.platform_extensions,
            extensions=record.extensions,
        )
        if self._persistent:
            _write_attachment(self._server, candidate.attachment())
        self._record = candidate
        if self._on_tags_changed is not None:
            self._on_tags_changed(self, previous_tags, candidate.tags)

    def _get_attachment(self) -> Attachment:
        return self._attachment_record().attachment()

    @property
    def tags(self) -> list[str]:
        return list(self._attachment_record().tags)

    @property
    def uri(self) -> str | None:
        return self._attachment_record().uri

    def __repr__(self) -> str:
        return f"<Connection id={self.id!r}>"

    def close(self, code: int = 1000, reason: str = "") -> None:
        self._server.close(code, reason)


class WebSockets(LifecycleCapability):
    capability_id = "websockets"

    def __init__(
        self,
        *,
        on_connect: Callable[[Connection, ConnectionContext], object] | None = None,
        on_message: Callable[[Connection, Any], object] | None = None,
        on_close: Callable[[Connection, int, str, bool], object] | None = None,
        on_error: Callable[[BaseException, Connection | None], object] | None = None,
        get_connection_tags: (
            Callable[
                [Connection, ConnectionContext],
                Iterable[object] | Awaitable[Iterable[object]],
            ]
            | None
        ) = None,
        socket_accept: Callable[[object, Sequence[str]], None] | None = None,
        socket_source: Callable[[str | None], Iterable[object]] | None = None,
        ensure_ready: Callable[[], Awaitable[None]] | None = None,
        pair_factory: Callable[[], object] | None = None,
        before_upgrade: (
            Callable[[Request], Response | None | Awaitable[Response | None]] | None
        ) = None,
        relay_target_for_request: Callable[[Request], RelayTarget | None] | None = None,
        relay_target_is_valid: Callable[[RelayTarget], bool] | None = None,
        relay_forward: Callable[[dict[str, Any]], Awaitable[Any]] | None = None,
        relay_max_frames: int = 1_000,
        relay_max_bytes: int = 1024 * 1024,
        relay_timeout: float = 30,
        hide_relay_connections: bool = False,
    ) -> None:
        self._connections: dict[str, Any] = {}
        self._connection_index: dict[str, tuple[str, tuple[str, ...]]] = {}
        self._id_index: dict[str, list[str]] = {}
        self._tag_index: dict[str, list[str]] = {}
        self._hydrated = False
        self._hydrated_tags: set[str] = set()
        self._on_connect = on_connect
        self._on_message = on_message
        self._on_close = on_close
        self._on_error = on_error
        self._get_connection_tags = get_connection_tags
        self._socket_accept = socket_accept
        self._socket_source = socket_source
        self._ensure_ready = ensure_ready
        self._pair_factory = pair_factory
        self._before_upgrade = before_upgrade
        self._relay_target_for_request = relay_target_for_request
        self._relay_target_is_valid = relay_target_is_valid
        self._relay_forward = relay_forward
        self._relay_limits = RelayLimits(relay_max_frames, relay_max_bytes)
        self._relay_timeout = relay_timeout
        self._hide_relay_connections = hide_relay_connections
        self._relay_locks: dict[str, asyncio.Lock] = {}

    def _services(self):
        try:
            return self.lifecycle
        except RuntimeError:
            return None

    async def _invoke(
        self,
        callback: Callable[..., Any],
        *args: Any,
        request: Request | None = None,
        connection: Connection | None = None,
    ) -> Any:
        token = _CURRENT_WEBSOCKET_CONNECTION.set((self, connection))
        try:
            services = self._services()
            if services is None:
                return await _call_maybe_async(callback, *args)

            async def invoke() -> Any:
                return await _call_maybe_async(callback, *args)

            return await services.run_in_host_context(
                invoke,
                scope=LifecycleHostContextScope(
                    request=request,
                    connection=connection,
                ),
            )
        finally:
            _CURRENT_WEBSOCKET_CONNECTION.reset(token)

    async def report_error(
        self,
        error: BaseException,
        connection: Connection | None = None,
    ) -> None:
        if self._on_error is None:
            traceback.print_exception(error)
            return
        try:
            await self._invoke(
                self._on_error,
                error,
                connection,
                connection=connection,
            )
        except BaseException as raised:  # noqa: BLE001
            if raised is not error:
                traceback.print_exception(raised)

    async def dispatch_hook(
        self,
        name: str,
        hook: Callable[..., Any],
        *args: Any,
        connection: Connection | None = None,
        request: Request | None = None,
        propagate: bool,
    ) -> Any:
        try:
            return await self._invoke(
                hook,
                *args,
                request=request,
                connection=connection,
            )
        except HookError:
            raise
        except Exception as exc:
            wrapped = HookError(name, exc)
            wrapped.__cause__ = exc
            await self.report_error(wrapped, connection)
            if propagate:
                raise wrapped from exc
            return None

    def _native_accept(self, websocket: object, tags: Sequence[str]) -> None:
        if self._socket_accept is not None:
            self._socket_accept(websocket, tags)
            return
        services = self._services()
        if services is None:
            raise RuntimeError("WebSockets is not installed on a Lifecycle")
        services.sockets.accept(websocket, tags=tags)

    def _native_sockets(self, tag: str | None = None) -> Iterable[object]:
        if self._socket_source is not None:
            return self._socket_source(tag)
        services = self._services()
        if services is None:
            return ()
        return services.sockets.get(tag=tag)

    @staticmethod
    def new_attachment(
        uri: str,
        connection_id: str | None = None,
        *,
        target: RelayTarget | None = None,
    ) -> Attachment:
        connection_id = connection_id or gen_id()
        attachment: Attachment = {
            "__pk": {
                "id": connection_id,
                "tags": [connection_id],
                "uri": uri,
            },
            "__user": None,
        }
        if target is not None:
            attachment["target"] = target
        return attachment

    def wrap(self, websocket: object, attachment: Attachment) -> Connection:
        record = _decode_attachment(attachment)
        if record is None:
            raise ValueError("invalid managed WebSocket attachment")
        _write_attachment(websocket, record.attachment())
        connection = Connection(
            record.id,
            websocket,
            record=record,
            on_tags_changed=self._replace_connection_tags,
        )
        return self.register(connection)

    def _connection_for_socket(self, websocket: object) -> Connection | None:
        for connection in self._connections.values():
            if isinstance(connection, Connection) and _same_socket(
                connection._server, websocket
            ):
                return connection
        return None

    def adopt(self, websocket: object) -> Connection | None:
        existing = self._connection_for_socket(websocket)
        if existing is not None:
            return existing
        try:
            record = _read_record(websocket)
            if record is None:
                return None
            target = _decode_relay_target(record.extensions.get("target"))
            if (
                target is not None
                and self._relay_target_is_valid is not None
                and not self._relay_target_is_valid(target)
            ):
                return None
            connection = Connection(
                record.id,
                websocket,
                record=record,
                on_tags_changed=self._replace_connection_tags,
            )
            return self.register(connection)
        except Exception:  # noqa: BLE001
            return None

    def hydrate(
        self,
        websockets: Iterable[object] | None = None,
        *,
        tag: str | None = None,
    ) -> None:
        if self._hydrated or (tag is not None and tag in self._hydrated_tags):
            return
        if websockets is None:
            try:
                websockets = self._native_sockets(tag)
            except Exception:  # noqa: BLE001
                return
        ordered_keys: list[str] = []
        for websocket in websockets:
            connection = self.adopt(websocket)
            if connection is not None:
                ordered_keys.append(connection._physical_key)
        if tag is None:
            ordered_keys.extend(
                key for key in self._connections if key not in ordered_keys
            )
            ordered = [(key, self._connections[key]) for key in ordered_keys]
            self._connections.clear()
            self._connections.update(ordered)
            self._rebuild_indexes()
            self._hydrated = True
        else:
            self._hydrated_tags.add(tag)

    def register(self, connection: Connection) -> Connection:
        existing = self._connection_for_socket(connection._server)
        if existing is not None:
            return existing
        connection._on_tags_changed = self._replace_connection_tags
        self._connections[connection._physical_key] = connection
        self._index_connection(connection)
        return connection

    @staticmethod
    def _remove_index_key(index: dict[str, list[str]], value: str, key: str) -> None:
        keys = index.get(value)
        if keys is None:
            return
        index[value] = [candidate for candidate in keys if candidate != key]
        if not index[value]:
            index.pop(value, None)

    def _drop_connection_index(self, physical_key: str) -> None:
        indexed = self._connection_index.pop(physical_key, None)
        if indexed is None:
            return
        connection_id, tags = indexed
        self._remove_index_key(self._id_index, connection_id, physical_key)
        for tag in tags:
            self._remove_index_key(self._tag_index, tag, physical_key)

    def _index_connection(self, connection: Connection) -> None:
        physical_key = connection._physical_key
        self._drop_connection_index(physical_key)
        tags = tuple(dict.fromkeys(connection.tags))
        self._connection_index[physical_key] = (connection.id, tags)
        self._id_index.setdefault(connection.id, []).append(physical_key)
        for tag in tags:
            self._tag_index.setdefault(tag, []).append(physical_key)

    def _rebuild_indexes(self) -> None:
        self._connection_index.clear()
        self._id_index.clear()
        self._tag_index.clear()
        for connection in self._connections.values():
            if isinstance(connection, Connection):
                self._index_connection(connection)

    def _replace_connection_tags(
        self,
        connection: Connection,
        previous: Sequence[str],
        current: Sequence[str],
    ) -> None:
        self._index_connection(connection)

    def discard(self, connection: object) -> None:
        physical_key = getattr(connection, "_physical_key", None)
        if isinstance(physical_key, str):
            self._relay_locks.pop(physical_key, None)
            self._drop_connection_index(physical_key)
        if self._connections.get(physical_key) is connection:
            self._connections.pop(physical_key, None)
        for key, candidate in tuple(self._connections.items()):
            if candidate is connection:
                self._drop_connection_index(key)
                self._connections.pop(key, None)

    def _unique_connections(self) -> tuple[Connection, ...]:
        seen: set[int] = set()
        connections: list[Connection] = []
        for connection in self._connections.values():
            identity = id(connection)
            if identity in seen:
                continue
            seen.add(identity)
            connections.append(connection)
        return tuple(connections)

    def _indexed_connections(self, keys: Iterable[str]) -> tuple[Connection, ...]:
        seen: set[int] = set()
        connections: list[Connection] = []
        for key in keys:
            connection = self._connections.get(key)
            if not isinstance(connection, Connection):
                continue
            identity = id(connection)
            if identity in seen:
                continue
            seen.add(identity)
            connections.append(connection)
        return tuple(connections)

    def _connections_with_tag(self, tag: str) -> tuple[Connection, ...]:
        candidates = set(self._id_index.get(tag, ()))
        candidates.update(self._tag_index.get(tag, ()))
        keys = (key for key in self._connections if key in candidates)
        indexed = list(self._indexed_connections(keys))
        if len(self._connection_index) == len(self._connections):
            return tuple(indexed)

        seen = {id(connection) for connection in indexed}
        for key, connection in self._connections.items():
            if key in self._connection_index or id(connection) in seen:
                continue
            if getattr(connection, "id", None) == tag:
                indexed.append(connection)
                seen.add(id(connection))
                continue
            try:
                tags = connection.tags
            except Exception:  # noqa: BLE001
                continue
            if tag in tags:
                indexed.append(connection)
                seen.add(id(connection))
        return tuple(indexed)

    @staticmethod
    def _relay_target(connection: Connection) -> RelayTarget | None:
        try:
            target = connection._get_attachment().get("target")
        except Exception:  # noqa: BLE001
            return None
        return _decode_relay_target(target)

    def _is_visible(self, connection: Connection) -> bool:
        return (
            not self._hide_relay_connections or self._relay_target(connection) is None
        )

    def get_connections(
        self,
        tag: str | None = None,
        *,
        hydrate: bool = True,
    ) -> tuple[Connection, ...]:
        if hydrate:
            self.hydrate(tag=tag)
            if tag is not None:
                # Native tags did not exist before this capability owned acceptance.
                self.hydrate()
        if tag is None:
            return tuple(
                connection
                for connection in self._unique_connections()
                if self._is_visible(connection)
            )

        matches = tuple(
            connection
            for connection in self._connections_with_tag(tag)
            if self._is_visible(connection)
        )
        return matches

    def get_connection(
        self,
        connection_id: str,
        *,
        hydrate: bool = True,
    ) -> Connection | None:
        if hydrate:
            self.hydrate()
        indexed_keys = self._id_index.get(connection_id, ())
        matches = [
            connection
            for connection in self._indexed_connections(indexed_keys)
            if self._is_visible(connection)
        ]
        if len(self._connection_index) != len(self._connections):
            seen = {id(connection) for connection in matches}
            matches.extend(
                connection
                for connection in self._unique_connections()
                if id(connection) not in seen
                and connection.id == connection_id
                and self._is_visible(connection)
            )
        if not matches:
            return None
        if len(matches) > 1:
            raise ValueError(
                f"More than one connection found for id {connection_id}. "
                "Use get_connections(id) instead."
            )
        return matches[0]

    def broadcast(
        self,
        data: str,
        exclude: Iterable[str] = (),
        *,
        hydrate: bool = True,
    ) -> None:
        excluded = frozenset(exclude)
        for connection in self.get_connections(hydrate=hydrate):
            if connection.id not in excluded:
                connection.send_if_open(data)

    def broadcast_json(
        self,
        data: JsonT,
        exclude: Iterable[str] = (),
        *,
        hydrate: bool = True,
    ) -> None:
        self.broadcast(json.dumps(data), exclude=exclude, hydrate=hydrate)

    async def _ready(self) -> None:
        if self._ensure_ready is not None:
            await self._ensure_ready()

    def _new_pair(self):
        return (
            self._pair_factory()
            if self._pair_factory is not None
            else WebSocketPair.new()
        )

    @staticmethod
    def _connection_id(request: Request) -> str:
        values = parse_qs(urlsplit(request.url).query, keep_blank_values=True).get(
            "_pk"
        )
        return (values[0] if values else "") or gen_id()

    async def upgrade(self, request: Request) -> Response:
        await self._ready()
        if self._before_upgrade is not None:
            rejection = await self._invoke(
                self._before_upgrade,
                request,
                request=request,
            )
            if rejection is not None:
                return rejection
        target = (
            self._relay_target_for_request(request)
            if self._relay_target_for_request is not None
            else None
        )
        if (
            target is not None
            and self._relay_target_is_valid is not None
            and not self._relay_target_is_valid(target)
        ):
            raise ValueError("invalid root WebSocket relay target")
        pair = self._new_pair()
        client, server = pair.object_values()
        attachment = self.new_attachment(
            request.url,
            self._connection_id(request),
            target=target,
        )
        record = _decode_attachment(attachment)
        if record is None:
            raise RuntimeError("failed to create managed WebSocket attachment")
        connection = self.register(
            Connection(
                record.id,
                server,
                record=record,
                persistent=False,
                on_tags_changed=self._replace_connection_tags,
            )
        )
        context = ConnectionContext(request)

        try:
            tags: Iterable[object] = ()
            if target is None and self._get_connection_tags is not None:
                result = await self.dispatch_hook(
                    "get_connection_tags",
                    self._get_connection_tags,
                    connection,
                    context,
                    connection=connection,
                    request=request,
                    propagate=True,
                )
                tags = result or ()
            prepared_tags = _prepare_tags(connection.id, tags)
            current = connection._attachment_record()
            accepted = _ConnectionRecord(
                id=current.id,
                tags=prepared_tags,
                state=current.state,
                uri=current.uri,
                platform_extensions=current.platform_extensions,
                extensions=current.extensions,
            )
            self._native_accept(server, prepared_tags)
            _write_attachment(server, accepted.attachment())
            previous_tags = connection.tags
            connection._record = accepted
            connection._persistent = True
            self._replace_connection_tags(
                connection,
                previous_tags,
                accepted.tags,
            )

            if target is not None:
                await self.dispatch_hook(
                    "on_connect",
                    self._relay_event,
                    connection,
                    "connect",
                    connection=connection,
                    request=request,
                    propagate=True,
                )
            elif self._on_connect is not None:
                await self.dispatch_hook(
                    "on_connect",
                    self._on_connect,
                    connection,
                    context,
                    connection=connection,
                    request=request,
                    propagate=True,
                )
        except BaseException:
            self.discard(connection)
            with suppress(Exception):
                server.close()
            raise

        return Response(None, status=101, web_socket=client)

    async def on_websocket_upgrade(
        self,
        context: CapabilityWebSocketUpgradeContext,
    ) -> Response | None:
        if not any(
            callback is not None
            for callback in (
                self._on_connect,
                self._on_message,
                self._on_close,
                self._on_error,
            )
        ):
            return None
        return await self.upgrade(context.request)

    @asynccontextmanager
    async def _socket_turn(self, websocket: object):
        connection = None
        ready = True
        try:
            connection = self.adopt(websocket)
            await self._ready()
        except HookError:
            ready = False
        except Exception as exc:  # noqa: BLE001
            ready = False
            await self.report_error(exc, connection)

        try:
            yield _SocketTurn(ready, connection)
        except HookError:
            pass
        except Exception as exc:  # noqa: BLE001
            await self.report_error(exc, connection)

    async def _relay_event(
        self,
        connection: Connection,
        event: str,
        **detail: Any,
    ) -> bool:
        target = self._relay_target(connection)
        if target is None:
            return False
        relay_forward = self._relay_forward
        if relay_forward is None:
            raise RuntimeError("no root WebSocket relay is configured")
        lock = self._relay_locks.setdefault(connection._physical_key, asyncio.Lock())
        async with lock:
            payload = {
                "event": event,
                "connectionId": connection.id,
                "relayId": connection._physical_key,
                "url": target["url"],
                "headers": target["headers"],
                "state": connection._get_raw_state(),
                "tags": connection.tags,
                **detail,
            }

            async def forward() -> Any:
                return await relay_forward(payload)

            await relay_round_trip(
                forward,
                connection,
                self._relay_limits,
                timeout=self._relay_timeout,
            )
        return True

    async def handle_message(self, websocket: object, message: object) -> bool:
        async with self._socket_turn(websocket) as turn:
            if not turn.ready:
                return turn.connection is not None
            connection = turn.connection
            if connection is None:
                return False
            if await self._relay_event(connection, "message", message=message):
                return True
            if self._on_message is not None:
                await self.dispatch_hook(
                    "on_message",
                    self._on_message,
                    connection,
                    message,
                    connection=connection,
                    propagate=False,
                )
            return True
        return True

    async def on_websocket_message(
        self,
        context: CapabilityWebSocketMessageContext,
    ) -> bool:
        return await self.handle_message(context.websocket, context.message)

    @staticmethod
    def _reciprocate_close(websocket: object, code: int, reason: str) -> None:
        if code in {1005, 1006, 1015}:
            return
        with suppress(Exception):
            getattr(websocket, "close")(code, reason)

    async def handle_close(
        self,
        websocket: object,
        code: int,
        reason: str,
        was_clean: bool,
    ) -> bool:
        async with self._socket_turn(websocket) as turn:
            if not turn.ready:
                if turn.connection is None:
                    return False
                self._reciprocate_close(websocket, code, reason)
                self.discard(turn.connection)
                return True
            connection = turn.connection
            if connection is None:
                return False
            try:
                relayed = await self._relay_event(
                    connection,
                    "close",
                    code=code,
                    reason=reason,
                    wasClean=was_clean,
                )
                if not relayed and self._on_close is not None:
                    await self.dispatch_hook(
                        "on_close",
                        self._on_close,
                        connection,
                        code,
                        reason,
                        was_clean,
                        connection=connection,
                        propagate=False,
                    )
            finally:
                self._reciprocate_close(websocket, code, reason)
                self.discard(connection)
            return True
        return True

    async def on_websocket_close(
        self,
        context: CapabilityWebSocketCloseContext,
    ) -> bool:
        return await self.handle_close(
            context.websocket,
            context.code,
            context.reason,
            context.was_clean,
        )

    async def handle_error(self, websocket: object, error: object) -> bool:
        async with self._socket_turn(websocket) as turn:
            if not turn.ready:
                return turn.connection is not None
            connection = turn.connection
            if connection is None:
                return False
            if await self._relay_event(connection, "error", error=str(error)):
                return True
            await self.report_error(_as_exception(error), connection)
            return True
        return True

    async def on_websocket_error(
        self,
        context: CapabilityWebSocketErrorContext,
    ) -> bool:
        return await self.handle_error(context.websocket, context.error)

    async def on_dispose(self) -> None:
        self.hydrate()
        for connection in self._unique_connections():
            with suppress(Exception):
                connection.close(1001, "Durable Object destroyed")
            self.discard(connection)
