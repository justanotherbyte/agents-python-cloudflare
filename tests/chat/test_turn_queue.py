from __future__ import annotations

import asyncio

import pytest

from agents.chat.turn_queue import TurnContext, TurnQueue, TurnReentryError


@pytest.mark.asyncio
async def test_turn_queue_runs_work_in_fifo_order():
    queue = TurnQueue()
    release = asyncio.Event()
    seen = []

    async def first(context: TurnContext):
        assert queue.is_current(context)
        seen.append("first-start")
        await release.wait()
        seen.append("first-end")

    async def second(context: TurnContext):
        assert queue.is_current(context)
        seen.append("second")

    one = asyncio.create_task(queue.enqueue("one", first))
    await asyncio.sleep(0)
    two = asyncio.create_task(queue.enqueue("two", second))
    await asyncio.sleep(0)

    assert seen == ["first-start"]
    release.set()
    await asyncio.gather(one, two)
    assert seen == ["first-start", "first-end", "second"]


@pytest.mark.asyncio
async def test_turn_queue_reset_invalidates_queued_work():
    queue = TurnQueue()
    release = asyncio.Event()
    seen = []

    async def first(context: TurnContext):
        assert queue.is_current(context)
        await release.wait()

    async def second(context: TurnContext):
        seen.append("second")

    one = asyncio.create_task(queue.enqueue("one", first))
    await asyncio.sleep(0)
    two = asyncio.create_task(queue.enqueue("two", second))
    await asyncio.sleep(0)
    queue.reset()
    release.set()

    first_result, second_result = await asyncio.gather(one, two)
    assert first_result.status == "completed"
    assert second_result.status == "stale"
    assert seen == []


@pytest.mark.asyncio
async def test_turn_context_is_current_only_while_callback_owns_queue():
    queue = TurnQueue()
    captured = None

    async def run(context: TurnContext):
        nonlocal captured
        captured = context
        assert context.request_id == "request"
        assert queue.active_request_id == "request"
        assert queue.is_current(context)

    await queue.enqueue("request", run)

    assert captured is not None
    assert queue.active_request_id is None
    assert not queue.is_current(captured)


@pytest.mark.asyncio
async def test_reset_invalidates_active_turn_context():
    queue = TurnQueue()
    entered = asyncio.Event()
    release = asyncio.Event()
    current_after_reset = True

    async def run(context: TurnContext):
        nonlocal current_after_reset
        entered.set()
        await release.wait()
        current_after_reset = queue.is_current(context)

    task = asyncio.create_task(queue.enqueue("request", run))
    await entered.wait()
    queue.reset()
    release.set()
    await task

    assert not current_after_reset


@pytest.mark.asyncio
async def test_exception_and_cancellation_clear_active_context():
    queue = TurnQueue()

    async def fail(context: TurnContext):
        assert queue.is_current(context)
        raise ValueError("failed")

    with pytest.raises(ValueError, match="failed"):
        await queue.enqueue("failure", fail)
    assert queue.active_request_id is None

    entered = asyncio.Event()

    async def wait(context: TurnContext):
        assert queue.is_current(context)
        entered.set()
        await asyncio.Event().wait()

    task = asyncio.create_task(queue.enqueue("cancelled", wait))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert queue.active_request_id is None


@pytest.mark.asyncio
async def test_same_queue_reentry_raises_without_deadlocking():
    queue = TurnQueue()

    async def inner(context: TurnContext):
        raise AssertionError(f"inner callback ran with {context!r}")

    async def outer(context: TurnContext):
        assert queue.is_current(context)
        with pytest.raises(TurnReentryError, match="cannot re-enter"):
            await queue.enqueue("inner", inner)

    await queue.enqueue("outer", outer)


@pytest.mark.asyncio
async def test_active_turn_can_enqueue_on_different_queue():
    outer_queue = TurnQueue()
    inner_queue = TurnQueue()
    seen = []

    async def inner(context: TurnContext):
        assert inner_queue.is_current(context)
        seen.append("inner")

    async def outer(context: TurnContext):
        assert outer_queue.is_current(context)
        await inner_queue.enqueue("inner", inner)

    await outer_queue.enqueue("outer", outer)

    assert seen == ["inner"]


@pytest.mark.asyncio
async def test_spawned_task_can_enqueue_after_inherited_lease_expires():
    queue = TurnQueue()
    release = asyncio.Event()
    delayed = None

    async def later(context: TurnContext):
        assert queue.is_current(context)
        return "later"

    async def outer(context: TurnContext):
        nonlocal delayed
        assert queue.is_current(context)

        async def enqueue_later():
            await release.wait()
            return await queue.enqueue("later", later)

        delayed = asyncio.create_task(enqueue_later())

    await queue.enqueue("outer", outer)
    release.set()

    assert delayed is not None
    result = await delayed
    assert result.status == "completed"
    assert result.value == "later"
