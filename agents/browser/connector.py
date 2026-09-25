from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import Awaitable, Callable, Mapping
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, replace
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

from .browser_run import (
    BrowserRenderingError,
    connect_browser,
    connect_browser_session,
    create_browser_session,
    delete_browser_session,
    list_browser_targets,
)
from .cdp import CdpSession
from .protocols import BrowserTransport
from .spec import CdpSpec, load_cdp_spec
from .store import (
    DEFAULT_SWEEP_IDLE_MS,
    BrowserSessionLock,
    BrowserSessionStore,
    StoredBrowserSession,
)
from .types import (
    BrowserLiveView,
    BrowserLiveViewTarget,
    BrowserLiveViewUrl,
    BrowserSessionInfo,
    BrowserSessionOptions,
    ConnectBrowserOptions,
    SweepEntry,
)

DEFAULT_EXEC_SWEEP_IDLE_MS = 24 * 60 * 60 * 1000
LIVE_VIEW_URL_TTL_MS = 5 * 60 * 1000
_EXEC_TOUCH_INTERVAL_MS = 60 * 1000
_EXEC_PREFIX = "cdp:exec:"
_REUSE_PREFIX = "cdp:reuse:"
_KITESURF_PREFIX = "kitesurf:"
_DELETE_PREFIX = "cdp:delete:"
_TARGET_PREFIX = "target:"


@dataclass
class _CachedSocket:
    session: CdpSession
    browser_session_id: str | None
    attached: dict[str, str]


class BrowserConnector:
    """Own durable Browser Run sessions while exposing programmatic CDP calls."""

    def __init__(
        self,
        browser: BrowserTransport,
        store: BrowserSessionStore,
        *,
        session: BrowserSessionOptions | None = None,
        timeout_ms: int = 10_000,
        now_ms: Callable[[], int] | None = None,
    ) -> None:
        self._browser = browser
        self._store = store
        self._options = session or BrowserSessionOptions()
        self._timeout_ms = timeout_ms
        self._now_ms = now_ms or (lambda: time.time_ns() // 1_000_000)
        self._sockets: dict[str, _CachedSocket] = {}
        self._connect_locks: dict[str, asyncio.Lock] = {}

    async def send(
        self,
        execution_id: str,
        method: str,
        params: object | None = None,
        *,
        session_id: str | None = None,
        timeout_ms: int | None = None,
    ) -> object:
        """Send a CDP command, resolving stable target handles after reconnect."""

        socket = await self._socket(execution_id)
        resolved = await self._resolve_handle(execution_id, session_id)
        return await socket.send(
            method,
            params,
            session_id=resolved,
            timeout_ms=timeout_ms,
        )

    async def attach_to_target(
        self,
        execution_id: str,
        target_id: str,
        *,
        timeout_ms: int | None = None,
    ) -> str:
        """Attach now and return a stable `target:` handle for later passes."""

        await self._attach(execution_id, target_id, timeout_ms=timeout_ms)
        return f"{_TARGET_PREFIX}{target_id}"

    async def spec(self) -> CdpSpec:
        """Load the binding's live CDP specification."""

        self._reject_kitesurf("CDP protocol discovery")
        return await load_cdp_spec(self._browser)

    async def start_session(self, execution_id: str) -> BrowserSessionInfo:
        """Ensure reuse, or atomically promote a dynamic execution session."""

        self._reject_kitesurf("durable sessions")
        if self._options.mode == "one-shot":
            raise ValueError("start_session requires reuse or dynamic mode")
        reuse_key = self._reuse_key()
        if self._options.mode == "dynamic":
            promoted = await self._promote_dynamic(execution_id)
            if promoted is not None:
                targets = await list_browser_targets(self._browser, promoted.session_id)
                return BrowserSessionInfo(promoted.session_id, targets)
        stored = await self._ensure_stored_session(reuse_key)
        return BrowserSessionInfo(
            stored.session_id,
            await list_browser_targets(self._browser, stored.session_id),
        )

    async def session_info(self) -> BrowserSessionInfo | None:
        """Return the current shared session, removing an expired record."""

        self._reject_kitesurf("durable session information")
        key = self._reuse_key()
        stored = await self._read(key)
        if stored is None:
            return None
        if stored.pending_delete:
            await self._retry_delete(key)
            return None
        try:
            targets = await list_browser_targets(self._browser, stored.session_id)
        except BrowserRenderingError as error:
            if error.status not in (404, 410):
                raise
            await self._delete_if(key, stored.session_id)
            return None
        return BrowserSessionInfo(stored.session_id, targets)

    async def reset_session(self, execution_id: str) -> BrowserSessionInfo:
        """Close the shared browser and create a fresh shared session."""

        self._reject_kitesurf("session reset")
        if self._options.mode == "one-shot":
            raise ValueError("reset_session requires reuse or dynamic mode")
        key = self._reuse_key()
        exec_key = self._exec_key(execution_id)
        pending_key: str | None = None
        expected: str | None = None
        async with self._locked(key, exec_key):
            shared = await self._store.get(key)
            if shared is not None:
                expected = shared.session_id
                pending_key = key
                await self._store.set(
                    key,
                    replace(
                        shared,
                        closed_at=shared.closed_at or self._now_ms(),
                        pending_delete=True,
                        delete_after_cleanup=True,
                    ),
                )
                execution = await self._store.get(exec_key)
                if execution is not None and execution.session_id == expected:
                    await self._store.delete(exec_key)
        if pending_key is not None and expected is not None:
            await self._drop_sockets_for(expected)
            await self._retry_delete(pending_key)
        stored = await self._ensure_stored_session(self._reuse_key())
        return BrowserSessionInfo(
            stored.session_id,
            await list_browser_targets(self._browser, stored.session_id),
        )

    async def close_session(self) -> None:
        """Idempotently close the shared session without opening a replacement."""

        self._reject_kitesurf("explicit session close")
        await self._close_stored(self._reuse_key())

    async def live_view(
        self,
        *,
        mode: str | None = None,
    ) -> BrowserLiveView | None:
        """Return fresh, ephemeral Live View URLs for the shared session."""

        self._reject_kitesurf("Live View")
        _validate_live_view_mode(mode)
        info = await self.session_info()
        if info is None:
            return None
        targets = tuple(
            BrowserLiveViewTarget(
                target.id,
                _apply_live_view_mode(target.devtools_frontend_url, mode),
                target.url,
                target.title,
                target.type,
            )
            for target in info.targets
            if target.devtools_frontend_url is not None
        )
        return BrowserLiveView(info.session_id, targets, LIVE_VIEW_URL_TTL_MS)

    async def live_view_url(
        self,
        execution_id: str,
        *,
        target_id: str | None = None,
        mode: str | None = None,
    ) -> BrowserLiveViewUrl:
        """Select one fresh Live View URL for an execution's current browser."""

        self._reject_kitesurf("Live View")
        _validate_live_view_mode(mode)
        await self._socket(execution_id)
        cached = self._sockets[execution_id]
        if cached.browser_session_id is None:
            raise RuntimeError("execution has no reconnectable Browser Run session")
        targets = await list_browser_targets(self._browser, cached.browser_session_id)
        if target_id is not None:
            target = next((item for item in targets if item.id == target_id), None)
        else:
            target = next(
                (
                    item
                    for item in targets
                    if item.type == "page" and item.devtools_frontend_url is not None
                ),
                None,
            )
            if target is None:
                target = next(
                    (
                        item
                        for item in targets
                        if item.devtools_frontend_url is not None
                    ),
                    None,
                )
        if target is None:
            if target_id is not None:
                raise LookupError(f"No target {target_id} found")
            raise LookupError("No browser tab is open")
        if target.devtools_frontend_url is None:
            raise RuntimeError(f"Target {target.id} has no Live View URL")
        return BrowserLiveViewUrl(
            _apply_live_view_mode(target.devtools_frontend_url, mode),
            target.id,
            LIVE_VIEW_URL_TTL_MS,
        )

    async def on_pass_end(self, execution_id: str, status: str) -> None:
        """Release the pass socket while retaining reconnectable session state."""

        await self._drop_socket(execution_id)
        if self._options.browser != "kitesurf" or status != "paused":
            return
        key = self._exec_key(execution_id)
        async with await self._store.acquire_lock(key):
            stored = await self._store.get(key)
            if stored is not None and stored.session_id.startswith(_KITESURF_PREFIX):
                await self._store.set(key, replace(stored, closed_at=self._now_ms()))

    async def dispose_execution(self, execution_id: str) -> None:
        """Release terminal execution ownership and its unpromoted browser."""

        await self._drop_socket(execution_id)
        self._connect_locks.pop(execution_id, None)
        key = self._exec_key(execution_id)
        if self._options.browser == "kitesurf":
            async with await self._store.acquire_lock(key):
                stored = await self._store.get(key)
                if stored is not None and stored.session_id.startswith(
                    _KITESURF_PREFIX
                ):
                    await self._store.delete(key)
                elif stored is not None:
                    await self._store.set(
                        key,
                        replace(
                            stored,
                            closed_at=stored.closed_at or self._now_ms(),
                            pending_delete=True,
                            delete_after_cleanup=True,
                        ),
                    )
            if stored is not None and not stored.session_id.startswith(
                _KITESURF_PREFIX
            ):
                with suppress(Exception):
                    await self._retry_delete(key)
            return

        to_close: str | None = None
        keys = [key]
        if self._options.mode == "dynamic":
            keys.append(self._reuse_key())
        async with self._locked(*keys):
            stored = await self._store.get(key)
            if stored is None:
                return
            shared = (
                await self._store.get(self._reuse_key())
                if self._options.mode == "dynamic"
                else None
            )
            promoted = shared is not None and shared.session_id == stored.session_id
            if stored.pending_delete:
                to_close = stored.session_id
            elif not promoted and stored.closed_at is None:
                to_close = stored.session_id
                await self._store.set(
                    key,
                    replace(
                        stored,
                        closed_at=self._now_ms(),
                        pending_delete=True,
                        delete_after_cleanup=True,
                    ),
                )
            elif promoted:
                await self._store.delete(key)
        if to_close is not None:
            with suppress(Exception):
                await self._retry_delete(key)

    async def sweep(
        self,
        *,
        max_idle_ms: int | None = None,
        max_exec_idle_ms: int = DEFAULT_EXEC_SWEEP_IDLE_MS,
    ) -> tuple[SweepEntry, ...]:
        """Reclaim stale shared/exec sessions and age execution tombstones."""

        shared_idle = (
            max_idle_ms
            if max_idle_ms is not None
            else (
                self._options.keep_alive_ms
                if self._options.keep_alive_ms is not None
                else DEFAULT_SWEEP_IDLE_MS
            )
        )
        keys = {self._reuse_key(), *(await self._store.list("cdp:")).keys()}
        swept: list[SweepEntry] = []
        for key in sorted(keys):
            is_exec = key.startswith(_EXEC_PREFIX)
            idle_ms = max_exec_idle_ms if is_exec else shared_idle
            to_close: str | None = None
            to_invalidate: str | None = None
            pending = False
            async with await self._store.acquire_lock(key):
                stored = await self._store.get(key)
                if stored is None:
                    continue
                now = self._now_ms()
                if stored.pending_delete:
                    to_close = stored.session_id
                    pending = True
                elif stored.closed_at is not None:
                    if stored.session_id in self._active_session_ids():
                        to_invalidate = stored.session_id
                    if now - stored.closed_at >= idle_ms:
                        await self._store.delete(key)
                elif stored.session_id in self._active_session_ids():
                    await self._store.set(key, replace(stored, updated_at=now))
                    continue
                elif now - stored.updated_at < idle_ms:
                    continue
                elif is_exec:
                    await self._store.set(
                        key,
                        replace(
                            stored,
                            closed_at=now,
                            pending_delete=True,
                            delete_after_cleanup=False,
                        ),
                    )
                    to_close = stored.session_id
                    pending = True
                else:
                    await self._store.set(
                        key,
                        replace(
                            stored,
                            closed_at=now,
                            pending_delete=True,
                            delete_after_cleanup=True,
                        ),
                    )
                    to_close = stored.session_id
                    pending = True
            if to_invalidate is not None:
                await self._drop_sockets_for(to_invalidate)
            if pending and to_close is not None:
                await self._drop_sockets_for(to_close)
                if to_close.startswith(_KITESURF_PREFIX):
                    await self._finish_delete(key, to_close)
                else:
                    with suppress(Exception):
                        await self._retry_delete(key)
            if to_close is not None:
                swept.append(SweepEntry(key, to_close))
        return tuple(swept)

    async def _socket(self, execution_id: str) -> CdpSession:
        if not execution_id:
            raise ValueError("execution_id must not be empty")
        cached = self._sockets.get(execution_id)
        if cached is not None:
            await self._touch_cached(execution_id, cached.browser_session_id)
            return cached.session
        lock = self._connect_locks.setdefault(execution_id, asyncio.Lock())
        async with lock:
            cached = self._sockets.get(execution_id)
            if cached is not None:
                await self._touch_cached(execution_id, cached.browser_session_id)
                return cached.session
            return await self._open_socket(execution_id)

    async def _open_socket(self, execution_id: str) -> CdpSession:
        if self._options.browser == "kitesurf":
            key = self._exec_key(execution_id)
            async with await self._store.acquire_lock(key):
                if await self._store.get(key) is not None:
                    raise RuntimeError(
                        "Kitesurf connection was closed and cannot be resumed; "
                        "start a new execution"
                    )
                now = self._now_ms()
                await self._store.set(
                    key,
                    StoredBrowserSession(f"{_KITESURF_PREFIX}{uuid4()}", now, now),
                )
            try:
                session = await connect_browser(
                    self._browser,
                    ConnectBrowserOptions(
                        timeout_ms=self._timeout_ms,
                        browser="kitesurf",
                    ),
                )
            except BaseException:
                marker = await self._read(key)
                if marker is not None:
                    await self._delete_if(key, marker.session_id)
                raise
            self._sockets[execution_id] = _CachedSocket(session, None, {})
            return session

        stored = await self._resolve_session(execution_id)
        session = await connect_browser_session(
            self._browser,
            stored.session_id,
            timeout_ms=self._timeout_ms,
        )
        self._sockets[execution_id] = _CachedSocket(session, stored.session_id, {})
        return session

    async def _resolve_session(self, execution_id: str) -> StoredBrowserSession:
        if self._options.mode == "reuse":
            return await self._ensure_stored_session(self._reuse_key())
        exec_key = self._exec_key(execution_id)
        existing = await self._read(exec_key)
        if existing is not None:
            if existing.closed_at is not None:
                if existing.pending_delete:
                    with suppress(Exception):
                        await self._retry_delete(exec_key)
                raise RuntimeError(
                    f"Browser session {existing.session_id} expired or was swept while "
                    "this execution was paused; start a new execution"
                )
            if await self._is_alive(existing):
                if self._now_ms() - existing.updated_at >= _EXEC_TOUCH_INTERVAL_MS:
                    await self._touch_if(exec_key, existing)
                return existing
            await self._tombstone(exec_key, existing.session_id)
            raise RuntimeError(
                f"Browser session {existing.session_id} expired or was swept while "
                "this execution was paused; start a new execution"
            )
        if self._options.mode == "dynamic":
            shared = await self._read(self._reuse_key())
            if shared is not None:
                if shared.pending_delete:
                    await self._retry_delete(self._reuse_key())
                elif shared.closed_at is None and await self._is_alive(shared):
                    await self._touch_if(self._reuse_key(), shared)
                    return shared
                else:
                    await self._delete_if(self._reuse_key(), shared.session_id)
        return await self._create_and_commit(exec_key)

    async def _ensure_stored_session(self, key: str) -> StoredBrowserSession:
        for _attempt in range(3):
            existing = await self._read(key)
            if existing is None:
                return await self._create_and_commit(key)
            if existing.pending_delete:
                await self._retry_delete(key)
                continue
            if existing.closed_at is not None:
                await self._delete_if(key, existing.session_id)
                continue
            alive = existing.closed_at is None and await self._is_alive(existing)
            async with await self._store.acquire_lock(key):
                current = await self._store.get(key)
                if current is None or current.session_id != existing.session_id:
                    continue
                if alive:
                    refreshed = replace(current, updated_at=self._now_ms())
                    await self._store.set(key, refreshed)
                    return refreshed
                await self._store.delete(key)
        raise RuntimeError(f"Browser session entry {key} changed concurrently")

    async def _create_and_commit(self, key: str) -> StoredBrowserSession:
        info = await create_browser_session(
            self._browser,
            keep_alive_ms=self._options.keep_alive_ms,
            recording=self._options.recording,
        )
        now = self._now_ms()
        created = StoredBrowserSession(info.session_id, now, now)
        async with await self._store.acquire_lock(key):
            winner = await self._store.get(key)
            if winner is None:
                await self._store.set(key, created)
        if winner is not None:
            cleanup_key = await self._queue_orphan_delete(created)
            with suppress(Exception):
                await self._retry_delete(cleanup_key)
            return winner
        return created

    async def _is_alive(self, stored: StoredBrowserSession) -> bool:
        try:
            await list_browser_targets(self._browser, stored.session_id)
            return True
        except BrowserRenderingError as error:
            if error.status in (404, 410):
                return False
            raise

    async def _attach(
        self,
        execution_id: str,
        target_id: str,
        *,
        timeout_ms: int | None = None,
    ) -> str:
        socket = await self._socket(execution_id)
        cached = self._sockets[execution_id]
        handle = f"{_TARGET_PREFIX}{target_id}"
        live = cached.attached.get(handle)
        if live is None:
            live = await socket.attach_to_target(target_id, timeout_ms=timeout_ms)
            cached.attached[handle] = live
        return live

    async def _resolve_handle(
        self, execution_id: str, session_id: str | None
    ) -> str | None:
        if session_id is None or not session_id.startswith(_TARGET_PREFIX):
            return session_id
        return await self._attach(execution_id, session_id[len(_TARGET_PREFIX) :])

    async def _drop_socket(self, execution_id: str) -> None:
        cached = self._sockets.pop(execution_id, None)
        if cached is not None:
            await cached.session.disconnect()

    async def _drop_sockets_for(self, session_id: str) -> None:
        executions = [
            execution_id
            for execution_id, cached in self._sockets.items()
            if cached.browser_session_id == session_id
        ]
        for execution_id in executions:
            await self._drop_socket(execution_id)

    async def _close_stored(self, key: str) -> None:
        async with await self._store.acquire_lock(key):
            stored = await self._store.get(key)
            if stored is not None:
                await self._store.set(
                    key,
                    replace(
                        stored,
                        closed_at=stored.closed_at or self._now_ms(),
                        pending_delete=True,
                        delete_after_cleanup=True,
                    ),
                )
        if stored is not None and not stored.session_id.startswith(_KITESURF_PREFIX):
            await self._drop_sockets_for(stored.session_id)
            await self._retry_delete(key)

    async def _read(self, key: str) -> StoredBrowserSession | None:
        async with await self._store.acquire_lock(key):
            return await self._store.get(key)

    async def _delete_if(self, key: str, session_id: str) -> None:
        async with await self._store.acquire_lock(key):
            current = await self._store.get(key)
            if current is not None and current.session_id == session_id:
                await self._store.delete(key)

    async def _touch_if(self, key: str, expected: StoredBrowserSession) -> None:
        async with await self._store.acquire_lock(key):
            current = await self._store.get(key)
            if current is not None and current.session_id == expected.session_id:
                await self._store.set(key, replace(current, updated_at=self._now_ms()))

    async def _touch_cached(self, execution_id: str, session_id: str | None) -> None:
        if session_id is None:
            return
        for key in (self._exec_key(execution_id), self._reuse_key()):
            stored = await self._read(key)
            if (
                stored is not None
                and stored.session_id == session_id
                and stored.closed_at is None
                and self._now_ms() - stored.updated_at >= _EXEC_TOUCH_INTERVAL_MS
            ):
                await self._touch_if(key, stored)

    async def _tombstone(self, key: str, session_id: str) -> None:
        async with await self._store.acquire_lock(key):
            current = await self._store.get(key)
            if current is not None and current.session_id == session_id:
                await self._store.set(
                    key,
                    replace(
                        current,
                        closed_at=current.closed_at or self._now_ms(),
                        pending_delete=False,
                        delete_after_cleanup=False,
                    ),
                )

    async def _queue_orphan_delete(self, session: StoredBrowserSession) -> str:
        key = self._delete_key(session.session_id)
        async with await self._store.acquire_lock(key):
            current = await self._store.get(key)
            if current is None:
                await self._store.set(
                    key,
                    replace(
                        session,
                        closed_at=session.closed_at or self._now_ms(),
                        pending_delete=True,
                        delete_after_cleanup=True,
                    ),
                )
        return key

    async def _retry_delete(self, key: str) -> None:
        stored = await self._read(key)
        if stored is None or not stored.pending_delete:
            return
        await delete_browser_session(self._browser, stored.session_id)
        await self._finish_delete(key, stored.session_id)

    async def _finish_delete(self, key: str, session_id: str) -> None:
        async with await self._store.acquire_lock(key):
            current = await self._store.get(key)
            if (
                current is None
                or current.session_id != session_id
                or not current.pending_delete
            ):
                return
            if current.delete_after_cleanup:
                await self._store.delete(key)
            else:
                await self._store.set(
                    key,
                    replace(
                        current,
                        pending_delete=False,
                        delete_after_cleanup=False,
                    ),
                )

    async def _promote_dynamic(self, execution_id: str) -> StoredBrowserSession | None:
        exec_key = self._exec_key(execution_id)
        reuse_key = self._reuse_key()
        for _attempt in range(3):
            execution = await self._read(exec_key)
            if execution is None:
                return None
            if execution.closed_at is not None or not await self._is_alive(execution):
                await self._tombstone(exec_key, execution.session_id)
                raise RuntimeError("Cannot promote an expired browser session")
            shared = await self._read(reuse_key)
            if shared is not None and shared.session_id == execution.session_id:
                await self._touch_if(reuse_key, shared)
                return execution
            if shared is not None and shared.pending_delete:
                await self._retry_delete(reuse_key)
                continue
            shared_alive = (
                shared is not None
                and shared.closed_at is None
                and await self._is_alive(shared)
            )
            async with self._locked(exec_key, reuse_key):
                current_exec = await self._store.get(exec_key)
                current_shared = await self._store.get(reuse_key)
                expected_shared_id = shared.session_id if shared is not None else None
                current_shared_id = (
                    current_shared.session_id if current_shared is not None else None
                )
                if (
                    current_exec is None
                    or current_exec.session_id != execution.session_id
                    or current_shared_id != expected_shared_id
                ):
                    continue
                if shared_alive:
                    raise RuntimeError(
                        "Cannot promote while a healthy shared browser session exists"
                    )
                if current_shared is not None:
                    await self._store.set(
                        reuse_key,
                        replace(
                            current_shared,
                            closed_at=current_shared.closed_at or self._now_ms(),
                            pending_delete=True,
                            delete_after_cleanup=True,
                        ),
                    )
                    stale_key = reuse_key
                else:
                    await self._store.set(
                        reuse_key,
                        replace(execution, updated_at=self._now_ms()),
                    )
                    return execution
            await self._retry_delete(stale_key)
        raise RuntimeError("Browser session promotion changed concurrently; retry")

    def _active_session_ids(self) -> set[str]:
        return {
            cached.browser_session_id
            for cached in self._sockets.values()
            if cached.browser_session_id is not None
        }

    @asynccontextmanager
    async def _locked(self, *keys: str):
        locks: list[BrowserSessionLock] = []
        try:
            for key in sorted(set(keys)):
                lock = await self._store.acquire_lock(key)
                locks.append(lock)
            yield
        finally:
            for lock in reversed(locks):
                await lock.release()

    def _exec_key(self, execution_id: str) -> str:
        return f"{_EXEC_PREFIX}{execution_id}"

    def _reuse_key(self) -> str:
        return f"{_REUSE_PREFIX}{self._options.key}"

    @staticmethod
    def _delete_key(session_id: str) -> str:
        digest = hashlib.sha256(session_id.encode()).hexdigest()
        return f"{_DELETE_PREFIX}{digest}"

    def _reject_kitesurf(self, feature: str) -> None:
        if self._options.browser == "kitesurf":
            raise ValueError(f"Kitesurf does not support {feature}")


def _validate_live_view_mode(mode: str | None) -> None:
    if mode not in (None, "tab", "devtools"):
        raise ValueError("Live View mode must be tab or devtools")


def _apply_live_view_mode(url: str, mode: str | None) -> str:
    if mode is None:
        return url
    parsed = urlsplit(url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query["mode"] = mode
    return urlunsplit((*parsed[:3], urlencode(query), parsed.fragment))
