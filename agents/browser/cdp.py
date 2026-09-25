from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime

from .protocols import BrowserSocket, BrowserSocketSubscription


@dataclass(frozen=True)
class CdpDebugEntry:
    at: str
    type: str
    data: Mapping[str, object]


class CdpSession:
    """Correlate CDP commands over one socket and own all listener cleanup."""

    def __init__(
        self,
        socket: BrowserSocket,
        *,
        default_timeout_ms: int = 10_000,
        session_id: str | None = None,
        on_close: Callable[[], object | Awaitable[object]] | None = None,
    ) -> None:
        self.session_id = session_id
        self._socket = socket
        self._default_timeout_ms = default_timeout_ms
        self._on_close = on_close
        self._next_id = 1
        self._pending: dict[int, tuple[str, asyncio.Future[object]]] = {}
        self._debug: list[CdpDebugEntry] = []
        self._subscriptions: list[BrowserSocketSubscription] = [
            socket.subscribe("message", self._handle_message),
            socket.subscribe("error", self._handle_error),
            socket.subscribe("close", self._handle_close),
        ]
        self._disconnected = False
        self._socket_close_requested = False
        self._cleanup_done = False
        self._close_lock = asyncio.Lock()
        self._close_future: asyncio.Future[None] | None = None

    async def send(
        self,
        method: str,
        params: object | None = None,
        *,
        session_id: str | None = None,
        timeout_ms: int | None = None,
    ) -> object:
        """Send one CDP command and return its correlated method result."""

        if self._disconnected:
            raise RuntimeError("CDP session is disconnected")
        if not method:
            raise ValueError("CDP method must not be empty")
        command_id = self._next_id
        self._next_id += 1
        timeout = self._default_timeout_ms if timeout_ms is None else timeout_ms
        if timeout <= 0:
            raise ValueError("CDP timeout_ms must be positive")
        loop = asyncio.get_running_loop()
        future: asyncio.Future[object] = loop.create_future()
        self._pending[command_id] = (method, future)
        self._record("send", id=command_id, method=method, session_id=session_id)
        message: dict[str, object] = {"id": command_id, "method": method}
        if params is not None:
            message["params"] = params
        if session_id:
            message["sessionId"] = session_id
        try:
            sent = self._socket.send(
                json.dumps(message, allow_nan=False, separators=(",", ":"))
            )
            if inspect.isawaitable(sent):
                await sent
            async with asyncio.timeout(timeout / 1000):
                return await future
        except TimeoutError as error:
            if not future.done():
                future.cancel()
            raise TimeoutError(
                f"CDP command timed out after {timeout}ms: {method}"
            ) from error
        finally:
            self._pending.pop(command_id, None)

    async def attach_to_target(
        self,
        target_id: str,
        *,
        timeout_ms: int | None = None,
    ) -> str:
        """Attach to a target in flattened mode and return its socket-local ID."""

        if not target_id:
            raise ValueError("attach_to_target requires a target_id")
        result = await self.send(
            "Target.attachToTarget",
            {"targetId": target_id, "flatten": True},
            timeout_ms=timeout_ms,
        )
        session_id = result.get("sessionId") if isinstance(result, Mapping) else None
        if not isinstance(session_id, str) or not session_id:
            raise RuntimeError(
                f"Target.attachToTarget did not return a sessionId for {target_id}"
            )
        self._record("attach", target_id=target_id, session_id=session_id)
        return session_id

    def debug_log(self, limit: int = 50) -> tuple[CdpDebugEntry, ...]:
        """Return the newest bounded protocol debug entries."""

        return tuple(self._debug[-max(1, limit) :])

    def clear_debug_log(self) -> None:
        self._debug.clear()

    async def disconnect(self) -> None:
        """Drop the socket without deleting its Browser Run session."""

        if not self._disconnected:
            self._disconnected = True
            self._reject_all(RuntimeError("CDP session disconnected"))
            self._release_subscriptions()
        if self._socket_close_requested:
            return
        self._socket_close_requested = True
        try:
            closed = self._socket.close(1000, "Done")
            if inspect.isawaitable(closed):
                await closed
        except BaseException:
            self._socket_close_requested = False
            raise

    async def close(self) -> None:
        """Disconnect and run the session's asynchronous owner cleanup once."""

        async with self._close_lock:
            if self._cleanup_done:
                return
            future = self._close_future
            leader = future is None
            if leader:
                future = asyncio.get_running_loop().create_future()
                self._close_future = future
        if not leader:
            await asyncio.shield(future)
            return

        try:
            await self.disconnect()
            if self._on_close is not None:
                result = self._on_close()
                if inspect.isawaitable(result):
                    await result
        except BaseException as error:
            if not future.done():
                future.set_exception(error)
            async with self._close_lock:
                if self._close_future is future:
                    self._close_future = None
            future.exception()
            raise

        self._cleanup_done = True
        if not future.done():
            future.set_result(None)
        await asyncio.shield(future)

    async def __aenter__(self) -> CdpSession:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()

    def _handle_message(self, event: object) -> None:
        data = event.get("data") if isinstance(event, Mapping) else event
        if not isinstance(data, str):
            return
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            return
        if not isinstance(payload, Mapping):
            return
        self._record(
            "receive",
            id=payload.get("id"),
            method=payload.get("method"),
            session_id=payload.get("sessionId"),
        )
        command_id = payload.get("id")
        if not isinstance(command_id, int):
            return
        pending = self._pending.get(command_id)
        if pending is None:
            return
        method, future = pending
        if future.done():
            return
        error = payload.get("error")
        if isinstance(error, Mapping):
            code = error.get("code", "unknown")
            message = error.get("message", "CDP error")
            future.set_exception(
                RuntimeError(f"CDP error {code}: {message} for {method}")
            )
            return
        future.set_result(payload.get("result"))

    def _handle_error(self, _event: object) -> None:
        self._disconnected = True
        self._reject_all(RuntimeError("CDP socket error"))
        self._release_subscriptions()

    def _handle_close(self, _event: object) -> None:
        self._disconnected = True
        self._socket_close_requested = True
        self._reject_all(RuntimeError("CDP connection closed"))
        self._release_subscriptions()

    def _release_subscriptions(self) -> None:
        subscriptions, self._subscriptions = self._subscriptions, []
        for subscription in subscriptions:
            try:
                subscription.cancel()
            except Exception:
                continue

    def _reject_all(self, error: Exception) -> None:
        for _method, future in tuple(self._pending.values()):
            if not future.done():
                future.set_exception(error)
        self._pending.clear()

    def _record(self, type: str, **data: object) -> None:
        self._debug.append(
            CdpDebugEntry(datetime.now(UTC).isoformat(), type, dict(data))
        )
        if len(self._debug) > 400:
            del self._debug[:-400]
