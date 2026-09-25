from __future__ import annotations

import asyncio
import weakref
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from .protocols import BrowserStorage

DEFAULT_SWEEP_IDLE_MS = 10 * 60 * 1000


@dataclass(frozen=True)
class StoredBrowserSession:
    """Durable Browser Run ownership record."""

    session_id: str
    created_at: int
    updated_at: int
    closed_at: int | None = None
    pending_delete: bool = False
    delete_after_cleanup: bool = False

    def stored(self) -> dict[str, object]:
        value: dict[str, object] = {
            "sessionId": self.session_id,
            "createdAt": self.created_at,
            "updatedAt": self.updated_at,
        }
        if self.closed_at is not None:
            value["closedAt"] = self.closed_at
        if self.pending_delete:
            value["pendingDelete"] = True
        if self.delete_after_cleanup:
            value["deleteAfterCleanup"] = True
        return value


class BrowserSessionLock(Protocol):
    async def __aenter__(self) -> None: ...

    async def __aexit__(self, *_exc: object) -> None: ...

    async def release(self) -> None: ...


class BrowserSessionStore(Protocol):
    async def acquire_lock(self, key: str) -> BrowserSessionLock: ...

    async def get(self, key: str) -> StoredBrowserSession | None: ...

    async def set(self, key: str, session: StoredBrowserSession) -> None: ...

    async def delete(self, key: str) -> None: ...

    async def list(self, prefix: str) -> Mapping[str, StoredBrowserSession]: ...


@dataclass
class _Queue:
    lock: asyncio.Lock
    users: int = 0


class _Lock:
    def __init__(self, queues: dict[str, _Queue], key: str, queue: _Queue):
        self._queues = queues
        self._key = key
        self._queue = queue
        self._released = False

    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, *_exc: object) -> None:
        await self.release()

    async def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._queue.lock.release()
        self._queue.users -= 1
        if self._queue.users == 0 and self._queues.get(self._key) is self._queue:
            del self._queues[self._key]


class DurableBrowserSessionStore:
    """Store session records under `browser-session:` with same-key queues."""

    _queues: weakref.WeakKeyDictionary[object, dict[str, _Queue]] = (
        weakref.WeakKeyDictionary()
    )

    def __init__(self, storage: BrowserStorage):
        self._storage = storage
        try:
            self._storage_queues = self._queues.setdefault(storage, {})
        except TypeError as error:
            raise TypeError(
                "BrowserStorage adapters must support weak references for queue cleanup"
            ) from error

    async def acquire_lock(self, key: str) -> BrowserSessionLock:
        queue = self._storage_queues.get(key)
        if queue is None:
            queue = _Queue(asyncio.Lock())
            self._storage_queues[key] = queue
        queue.users += 1
        try:
            await queue.lock.acquire()
        except BaseException:
            queue.users -= 1
            if queue.users == 0 and self._storage_queues.get(key) is queue:
                del self._storage_queues[key]
            raise
        return _Lock(self._storage_queues, key, queue)

    async def get(self, key: str) -> StoredBrowserSession | None:
        return _decode(await self._storage.get(self._key(key)))

    async def set(self, key: str, session: StoredBrowserSession) -> None:
        await self._storage.put(self._key(key), session.stored())

    async def delete(self, key: str) -> None:
        await self._storage.delete(self._key(key))

    async def list(self, prefix: str) -> Mapping[str, StoredBrowserSession]:
        storage_prefix = self._key(prefix)
        values = await self._storage.list(prefix=storage_prefix)
        result: dict[str, StoredBrowserSession] = {}
        for key, value in values.items():
            decoded = _decode(value)
            if decoded is not None:
                result[key[len("browser-session:") :]] = decoded
        return result

    @staticmethod
    def _key(key: str) -> str:
        return f"browser-session:{key}"


def _decode(value: object) -> StoredBrowserSession | None:
    if not isinstance(value, Mapping):
        return None
    session_id = value.get("sessionId")
    created_at = value.get("createdAt")
    updated_at = value.get("updatedAt")
    closed_at = value.get("closedAt")
    pending_delete = value.get("pendingDelete", False)
    delete_after_cleanup = value.get("deleteAfterCleanup", False)
    if (
        not isinstance(session_id, str)
        or not isinstance(created_at, int)
        or not isinstance(updated_at, int)
        or (closed_at is not None and not isinstance(closed_at, int))
        or not isinstance(pending_delete, bool)
        or not isinstance(delete_after_cleanup, bool)
    ):
        return None
    return StoredBrowserSession(
        session_id,
        created_at,
        updated_at,
        closed_at,
        pending_delete,
        delete_after_cleanup,
    )
