from __future__ import annotations

import asyncio

import fakes
import pytest

from agents.lifecycle import (
    Lifecycle,
    LifecycleCapability,
    LifecycleSql,
    get_current_lifecycle_context,
)


@pytest.mark.asyncio
async def test_preparation_is_shared_retryable_and_precedes_startup():
    gate = fakes.AsyncGate()
    failure = RuntimeError("preparation failed")
    events: list[str] = []
    attempts = 0

    class PreparedCapability(LifecycleCapability):
        capability_id = "prepared"

        async def on_start(self) -> None:
            rows = self.lifecycle.sql.execute("SELECT value FROM prepared_values")
            assert rows == [{"value": "ready"}]
            events.append("capability")

    async def prepare(sql: LifecycleSql) -> None:
        nonlocal attempts
        assert get_current_lifecycle_context() is None
        attempts += 1
        events.append(f"prepare:{attempts}")
        sql.execute("CREATE TABLE IF NOT EXISTS prepared_values (value TEXT NOT NULL)")
        if attempts == 1:
            await gate.block()
            raise failure
        sql.execute("DELETE FROM prepared_values")
        sql.execute("INSERT INTO prepared_values VALUES ('ready')")

    async def host_start() -> None:
        events.append("host")

    lifecycle = Lifecycle(
        fakes.FakeCtx(),
        host=object(),
        prepare=prepare,
        on_start=host_start,
    )
    lifecycle.use(PreparedCapability())

    leader = asyncio.create_task(lifecycle.start())
    await gate.wait_until_blocked()
    follower = asyncio.create_task(lifecycle.start())
    await asyncio.sleep(0)
    gate.release()

    leader_result, follower_result = await asyncio.gather(
        leader,
        follower,
        return_exceptions=True,
    )

    assert leader_result is failure
    assert follower_result is failure
    assert events == ["prepare:1"]

    await lifecycle.start()

    assert events == ["prepare:1", "prepare:2", "capability", "host"]


def test_preparation_is_lazy_during_lifecycle_construction():
    prepared = []

    class LazyContext:
        @property
        def storage(self):
            raise RuntimeError("storage read")

    def prepare(_sql: LifecycleSql) -> None:
        prepared.append(True)

    Lifecycle(LazyContext(), host=object(), prepare=prepare)

    assert prepared == []
