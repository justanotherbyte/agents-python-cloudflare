from __future__ import annotations

import asyncio

import fakes
import pytest
from workers import DurableObject

import agents.lifecycle._runtime as lifecycle_module
from agents.lifecycle import Lifecycle, LifecycleCapability


class ServiceCapability(LifecycleCapability):
    capability_id = "services"


class ServiceHost(DurableObject):
    def __init__(self, ctx, env):
        super().__init__(ctx, env)
        self.lifecycle = Lifecycle(ctx, host=self)
        self.capability = ServiceCapability()
        self.lifecycle.use(self.capability)


def build_services():
    ctx = fakes.FakeCtx()
    host = ServiceHost(ctx, object())
    return ctx, host.lifecycle, host.capability.lifecycle


@pytest.mark.asyncio
async def test_storage_and_sql_services_are_bounded_and_operational():
    ctx, _lifecycle, services = build_services()

    assert ServiceHost.__bases__ == (DurableObject,)

    await services.storage.put({"item:b": 2, "item:a": 1, "other": 0})
    get_keys = ctx.storage.get
    delete_keys = ctx.storage.delete

    async def assert_list_get(keys):
        assert type(keys) is list
        return await get_keys(keys)

    async def assert_list_delete(keys):
        assert type(keys) is list
        return await delete_keys(keys)

    ctx.storage.get = assert_list_get
    ctx.storage.delete = assert_list_delete
    assert await services.storage.get(("item:b", "item:a")) == {
        "item:a": 1,
        "item:b": 2,
    }
    assert await services.storage.delete(("missing:a", "missing:b")) == 0
    ctx.storage.get = get_keys
    ctx.storage.delete = delete_keys

    assert await services.storage.get(["item:b", "item:a"]) == {
        "item:a": 1,
        "item:b": 2,
    }
    assert await services.storage.list(
        prefix="item:",
        start_after="item:a",
        limit=1,
    ) == {"item:b": 2}
    assert await services.storage.delete(["item:a", "missing"]) == 1

    services.sql.execute("CREATE TABLE values_table (value INTEGER NOT NULL)")
    services.sql.execute("INSERT INTO values_table VALUES (?)", 7)
    assert services.sql.execute("SELECT value FROM values_table") == [{"value": 7}]

    def rollback() -> None:
        services.sql.execute("INSERT INTO values_table VALUES (?)", 9)
        raise RuntimeError("rollback")

    with pytest.raises(RuntimeError, match="rollback"):
        services.storage.transaction_sync(rollback)
    assert services.sql.execute("SELECT value FROM values_table") == [{"value": 7}]

    for forbidden in (
        "ctx",
        "host",
        "set_alarm",
        "delete_alarm",
        "facets",
        "bindings",
    ):
        assert not hasattr(services, forbidden)
        assert not hasattr(services.storage, forbidden)


def test_socket_services_accept_filter_and_round_trip_attachments(monkeypatch):
    ctx, _lifecycle, services = build_services()
    first = fakes.FakeSocket()
    second = fakes.FakeSocket()
    converted = []

    def to_js(value, **_options):
        converted.append(value)
        return value

    monkeypatch.setattr(lifecycle_module, "to_js", to_js)

    services.sockets.accept(first, tags=("all", "room:one"))
    services.sockets.accept(second, tags=("all", "room:two"))
    services.sockets.serialize_attachment(first, {"owner": "services"})

    assert services.sockets.get() == (first, second)
    assert services.sockets.get(tag="room:one") == (first,)
    assert services.sockets.get(tag="missing") == ()
    assert services.sockets.deserialize_attachment(first) == {"owner": "services"}
    assert services.sockets.deserialize_attachment(second) is None
    assert ctx.accepted_websockets == [first, second]
    assert converted[0] == ["all", "room:one"]
    assert converted[1] == ["all", "room:two"]


@pytest.mark.asyncio
async def test_capability_ready_returns_during_own_startup_without_deadlock():
    states: list[tuple[bool, bool]] = []

    class ReadyCapability(LifecycleCapability):
        capability_id = "ready"

        async def on_start(self) -> None:
            states.append((self.lifecycle.starting, self.lifecycle.is_ready))
            await self.lifecycle.ready()
            states.append((self.lifecycle.starting, self.lifecycle.is_ready))

    lifecycle = Lifecycle(fakes.FakeCtx(), host=object())
    capability = ReadyCapability()
    lifecycle.use(capability)

    await lifecycle.start()
    await capability.lifecycle.ready()

    assert states == [(True, False), (True, False)]
    assert capability.lifecycle.starting is False
    assert capability.lifecycle.is_ready is True


@pytest.mark.asyncio
async def test_capability_ready_waits_when_called_outside_startup_owner():
    gate = fakes.AsyncGate()

    class GatedCapability(LifecycleCapability):
        capability_id = "gated"

        async def on_start(self) -> None:
            await gate.block()

    lifecycle = Lifecycle(fakes.FakeCtx(), host=object())
    capability = GatedCapability()
    lifecycle.use(capability)

    startup = asyncio.create_task(lifecycle.start())
    await gate.wait_until_blocked()
    waiter = asyncio.create_task(capability.lifecycle.ready())
    await asyncio.sleep(0)

    assert not waiter.done()

    gate.release()
    await asyncio.gather(startup, waiter)


def test_lifecycle_construction_does_not_read_storage():
    class LazyContext:
        @property
        def storage(self):
            raise RuntimeError("storage read")

    lifecycle = Lifecycle(LazyContext(), host=object())
    lifecycle.use(ServiceCapability())
