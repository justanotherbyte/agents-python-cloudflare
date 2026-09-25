from __future__ import annotations

import asyncio

import pytest

from agents.browser import DurableBrowserSessionStore, StoredBrowserSession

from .fakes import FakeStorage


@pytest.mark.asyncio
async def test_store_uses_pinned_prefix_and_camel_case_record_shape():
    storage = FakeStorage()
    store = DurableBrowserSessionStore(storage)
    session = StoredBrowserSession("session-1", 10, 20, 30)
    await store.set("cdp:exec:x", session)

    assert storage.values == {
        "browser-session:cdp:exec:x": {
            "sessionId": "session-1",
            "createdAt": 10,
            "updatedAt": 20,
            "closedAt": 30,
        }
    }
    assert await store.get("cdp:exec:x") == session
    assert await store.list("cdp:") == {"cdp:exec:x": session}


@pytest.mark.asyncio
async def test_same_storage_same_key_locks_serialize_across_store_adapters():
    storage = FakeStorage()
    one = DurableBrowserSessionStore(storage)
    two = DurableBrowserSessionStore(storage)
    first = await one.acquire_lock("key")
    acquired = asyncio.Event()

    async def waiter():
        lock = await two.acquire_lock("key")
        acquired.set()
        await lock.release()

    task = asyncio.create_task(waiter())
    await asyncio.sleep(0)
    assert not acquired.is_set()
    await first.release()
    await task
    assert acquired.is_set()


@pytest.mark.asyncio
async def test_different_keys_do_not_block_and_release_is_idempotent():
    store = DurableBrowserSessionStore(FakeStorage())
    first = await store.acquire_lock("one")
    second = await asyncio.wait_for(store.acquire_lock("two"), timeout=0.1)
    await first.release()
    await first.release()
    await second.release()
