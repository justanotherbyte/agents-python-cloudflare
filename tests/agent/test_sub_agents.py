from __future__ import annotations

import asyncio
import json
import types
from collections.abc import Callable
from http import HTTPMethod
from typing import Any, cast

import fakes
import pytest
from workers import Request, Response

import agents.core.agent as agent_module
import agents.lifecycle._runtime as lifecycle_module
import agents.schedules as schedules_module
from agents import Agent
from agents.core.error import HookError, RoutingException
from agents.core.facets import _agent_route_address
from agents.core.protocol import PathStep
from agents.core.utils import dumps_wire
from agents.schedules import RetryOptions, ScheduleOptions, scheduler_callback


class RootAgent(Agent):
    pass


class ChildAgent(Agent):
    def __init__(self, ctx, env):
        super().__init__(ctx, env)
        self.started_paths: list[list[PathStep]] = []

    async def on_start(self):
        self.started_paths.append(self.parent_path)


class OtherChild(Agent):
    pass


class GrandChild(Agent):
    pass


class ScheduledChild(Agent):
    def __init__(self, ctx, env):
        super().__init__(ctx, env)
        self.schedule_calls = []
        self.retry_calls = []
        self.failures_remaining = 0
        self.cancel_then_fail_calls = 0
        self.infrastructure_error: BaseException | None = None

    @scheduler_callback()
    async def record(self, payload, schedule):
        self.schedule_calls.append((payload, schedule.id))

    @scheduler_callback()
    async def cancel_self(self, payload, schedule):
        self.schedule_calls.append((payload, schedule.id))
        await self.cancel_schedule(schedule.id)

    @scheduler_callback()
    async def flaky(self, payload, schedule):
        self.retry_calls.append((payload, schedule.id, schedule.time))
        if self.failures_remaining:
            self.failures_remaining -= 1
            raise RuntimeError("retry me")

    @scheduler_callback()
    async def cancel_then_fail(self, payload, schedule):
        self.cancel_then_fail_calls += 1
        await self.cancel_schedule(schedule.id)
        raise RuntimeError("cancelled callback failed")

    @scheduler_callback()
    async def fail_infrastructure(self, payload, schedule):
        if self.infrastructure_error is not None:
            raise self.infrastructure_error


class ScheduledGrandChild(ScheduledChild):
    pass


class SchedulingOnStartChild(ScheduledChild):
    async def on_start(self):
        await self.schedule_every(10, "record")


class RetainedProxy:
    def __init__(self, callback: Callable[[], Any]):
        self.callback = callback
        self.destroy_calls = 0

    def __call__(self) -> Any:
        return self.callback()

    def destroy(self) -> None:
        self.destroy_calls += 1


class FacetStub:
    def __init__(self, agent: Agent):
        self.agent = agent

    async def _cf_init_as_facet(self, name: str, parent_path: str) -> None:
        await self.agent._cf_init_as_facet(name, parent_path)

    async def fetch(self, request: Request) -> Any:
        response = await self.agent.fetch(request)
        return response.js_object

    async def _cf_route_lifecycle(self, envelope: str) -> str:
        return await self.agent._cf_route_lifecycle(envelope)


class FailingBootstrap:
    def __init__(self, error: BaseException):
        self.error = error

    async def _cf_init_as_facet(self, name: str, parent_path: str) -> None:
        raise self.error


class BlockingFailure:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def _cf_init_as_facet(self, name: str, parent_path: str) -> None:
        self.started.set()
        await self.release.wait()
        raise RuntimeError("bootstrap failed")


class RootNamespace:
    def __init__(self) -> None:
        self.ids: list[str] = []
        self.get_ids: list[str] = []
        self.root_stub = object()

    def idFromName(self, name: str) -> str:
        self.ids.append(name)
        return name

    def get(self, durable_id: str) -> object:
        self.get_ids.append(durable_id)
        return self.root_stub


class FacetRuntime:
    def __init__(self) -> None:
        self.exports: Any = None
        self.stubs: dict[str, Any] = {}
        self.agents: dict[str, Agent] = {}
        self.stub_overrides: dict[tuple[str, str], Any] = {}
        self.identities_by_key: dict[str, list[str]] = {}
        self.identities_by_owner_key: dict[tuple[str, str], list[str]] = {}
        self.get_calls: list[tuple[str, str, RetainedProxy]] = []
        self.abort_calls: list[tuple[str, str, str]] = []
        self.delete_calls: list[tuple[str, str]] = []
        self.proxies: list[RetainedProxy] = []
        self.delete_error: BaseException | None = None

    def create_proxy(self, callback: Callable[[], Any]) -> RetainedProxy:
        proxy = RetainedProxy(callback)
        self.proxies.append(proxy)
        return proxy

    def for_owner(self, owner: str) -> FacetFacade:
        return FacetFacade(self, owner)

    def get(self, owner: str, key: str, proxy: RetainedProxy) -> Any:
        self.get_calls.append((owner, key, proxy))
        options = proxy()
        identity = options["id"]
        owner_key = (owner, key)
        if owner_key in self.stub_overrides:
            return self.stub_overrides[owner_key]
        if identity not in self.stubs:
            cls = options["class"]
            agent = fakes.build_agent(cls=cls, name=identity)
            cast(Any, agent.ctx).exports = self.exports
            cast(Any, agent.ctx).facets = self.for_owner(identity)
            self.agents[identity] = agent
            self.stubs[identity] = FacetStub(agent)
            self.identities_by_key.setdefault(key, []).append(identity)
            self.identities_by_owner_key.setdefault(owner_key, []).append(identity)
        return self.stubs[identity]

    def agent(self, key: str, index: int = -1) -> Agent:
        identity = self.identities_by_key[key][index]
        return self.agents[identity]

    def abort(self, owner: str, key: str, reason: str) -> None:
        self.abort_calls.append((owner, key, reason))

    def delete(self, owner: str, key: str) -> None:
        self.delete_calls.append((owner, key))
        if self.delete_error is not None:
            raise self.delete_error
        if owner != "root":
            raise RuntimeError("only a root may delete facet storage")
        identities = self.identities_by_owner_key.get((owner, key), [])
        if identities:
            identity = identities.pop()
            self.stubs.pop(identity, None)
            self.agents.pop(identity, None)
            self.identities_by_key[key].remove(identity)
            if not identities:
                self.identities_by_owner_key.pop((owner, key))
            if not self.identities_by_key[key]:
                self.identities_by_key.pop(key)


class FacetFacade:
    def __init__(self, runtime: FacetRuntime, owner: str) -> None:
        self.runtime = runtime
        self.owner = owner

    def get(self, key: str, proxy: RetainedProxy) -> Any:
        return self.runtime.get(self.owner, key, proxy)

    def abort(self, key: str, reason: str) -> None:
        self.runtime.abort(self.owner, key, reason)

    def delete(self, key: str) -> None:
        self.runtime.delete(self.owner, key)


def _root(
    monkeypatch,
    root_cls: type[Agent] = RootAgent,
    *child_classes: type[Agent],
) -> tuple[Agent, FacetRuntime, RootNamespace]:
    runtime = FacetRuntime()
    namespace = RootNamespace()
    exports = types.SimpleNamespace(**{root_cls.__name__: namespace})
    for child_cls in child_classes or (ChildAgent, OtherChild, GrandChild):
        setattr(exports, child_cls.__name__, child_cls)
    runtime.exports = exports
    root = fakes.build_agent(cls=root_cls, name="root")
    cast(Any, root.ctx).exports = exports
    cast(Any, root.ctx).facets = runtime.for_owner("root")
    namespace.root_stub = FacetStub(root)
    monkeypatch.setattr(agent_module, "create_proxy", runtime.create_proxy)
    return root, runtime, namespace


def _reincarnate_root(
    root: Agent,
    runtime: FacetRuntime,
    namespace: RootNamespace,
) -> Agent:
    reincarnated = fakes.build_agent(
        cls=type(root),
        name=root.ctx.id.name,
        conn=root.ctx.conn,
    )
    cast(Any, reincarnated.ctx).exports = runtime.exports
    cast(Any, reincarnated.ctx).facets = runtime.for_owner("root")
    namespace.root_stub = FacetStub(reincarnated)
    return reincarnated


@pytest.mark.asyncio
async def test_spawn_records_ancestry_and_reuses_stub_registry_and_proxy(monkeypatch):
    root, runtime, namespace = _root(monkeypatch)
    times = iter([100, 200])
    monkeypatch.setattr(agent_module, "now_ms", lambda: next(times))

    first = await root.sub_agent(ChildAgent, "leaf")
    second = await root.sub_agent(ChildAgent, "leaf")

    key = "ChildAgent\0leaf"
    child = cast(ChildAgent, runtime.agent(key))
    assert first is second
    assert child.parent_path == [{"className": "RootAgent", "name": "root"}]
    assert child.started_paths == [[{"className": "RootAgent", "name": "root"}]]
    assert root.has_sub_agent(ChildAgent, "leaf") is True
    assert root.list_sub_agents() == [
        {"class_name": "ChildAgent", "name": "leaf", "created_at": 100}
    ]
    assert len(runtime.proxies) == 1
    assert runtime.get_calls[0][2] is runtime.get_calls[1][2]
    assert namespace.ids[0].startswith("cf-agents:v2:leaf:")


@pytest.mark.asyncio
async def test_cancelled_first_bootstrap_rolls_back_and_reuses_proxy_on_retry(
    monkeypatch,
):
    root, runtime, _ = _root(monkeypatch)
    key = "ChildAgent\0leaf"
    runtime.stub_overrides[("root", key)] = FailingBootstrap(
        asyncio.CancelledError("bootstrap failed")
    )

    with pytest.raises(asyncio.CancelledError, match="bootstrap failed"):
        await root.sub_agent(ChildAgent, "leaf")

    assert root.has_sub_agent(ChildAgent, "leaf") is False
    assert len(runtime.proxies) == 1

    runtime.stub_overrides.pop(("root", key))
    await root.sub_agent(ChildAgent, "leaf")

    assert root.has_sub_agent(ChildAgent, "leaf") is True
    assert len(runtime.proxies) == 1


@pytest.mark.asyncio
async def test_first_bootstrap_records_before_start_and_rolls_back_failure(
    monkeypatch,
):
    root, runtime, _ = _root(monkeypatch)
    failure = BlockingFailure()
    runtime.stub_overrides[("root", "ChildAgent\0leaf")] = failure

    spawn = asyncio.create_task(root.sub_agent(ChildAgent, "leaf"))
    await failure.started.wait()
    assert root.has_sub_agent(ChildAgent, "leaf") is True

    failure.release.set()
    with pytest.raises(RuntimeError, match="bootstrap failed"):
        await spawn
    assert root.has_sub_agent(ChildAgent, "leaf") is False


@pytest.mark.asyncio
async def test_concurrent_retry_uses_same_facet_without_losing_registry_row(
    monkeypatch,
):
    started = asyncio.Event()
    release = asyncio.Event()

    class FailOnceChild(Agent):
        def __init__(self, ctx, env):
            super().__init__(ctx, env)
            self.attempts = 0

        async def on_start(self):
            self.attempts += 1
            if self.attempts == 1:
                started.set()
                await release.wait()
                raise RuntimeError("first bootstrap failed")

        async def on_error(self, error, connection=None):
            pass

    root, runtime, _ = _root(monkeypatch, RootAgent, FailOnceChild)
    first = asyncio.create_task(root.sub_agent(FailOnceChild, "leaf"))
    await started.wait()
    second = asyncio.create_task(root.sub_agent(FailOnceChild, "leaf"))
    await asyncio.sleep(0)

    release.set()
    with pytest.raises(HookError, match="first bootstrap failed"):
        await first
    second_stub = await second

    child = cast(FailOnceChild, runtime.agent("FailOnceChild\0leaf"))
    assert child.attempts == 2
    assert second_stub is runtime.stubs[child.ctx.id.name]
    assert root.has_sub_agent(FailOnceChild, "leaf") is True
    assert len(runtime.proxies) == 1


@pytest.mark.asyncio
async def test_spawn_waits_for_same_facet_deletion_cleanup(monkeypatch):
    root, runtime, _ = _root(monkeypatch)
    await root.sub_agent(ChildAgent, "leaf")
    old_proxy = runtime.proxies[-1]
    cleanup_started = asyncio.Event()
    cleanup_release = asyncio.Event()

    async def block_cleanup(_path):
        cleanup_started.set()
        await cleanup_release.wait()

    monkeypatch.setattr(root, "_cleanup_facet_prefix", block_cleanup)
    deletion = asyncio.create_task(root.delete_sub_agent(ChildAgent, "leaf"))
    await cleanup_started.wait()
    spawn = asyncio.create_task(root.sub_agent(ChildAgent, "leaf"))
    await asyncio.sleep(0)

    assert not spawn.done()

    cleanup_release.set()
    await deletion
    new_stub = await spawn

    assert old_proxy.destroy_calls == 1
    assert runtime.proxies[-1].destroy_calls == 0
    assert new_stub is runtime.stubs[runtime.agent("ChildAgent\0leaf").ctx.id.name]


@pytest.mark.asyncio
async def test_failed_rebootstrap_preserves_an_existing_registry_row(monkeypatch):
    root, runtime, _ = _root(monkeypatch)
    await root.sub_agent(ChildAgent, "leaf")
    runtime.stub_overrides[("root", "ChildAgent\0leaf")] = FailingBootstrap(
        RuntimeError("failed")
    )

    with pytest.raises(RuntimeError, match="failed"):
        await root.sub_agent(ChildAgent, "leaf")

    assert root.has_sub_agent(ChildAgent, "leaf") is True


@pytest.mark.asyncio
async def test_registry_orders_filters_and_survives_abort(monkeypatch):
    root, runtime, _ = _root(monkeypatch)
    times = iter([300, 100, 200])
    monkeypatch.setattr(agent_module, "now_ms", lambda: next(times))

    await root.sub_agent(ChildAgent, "late")
    await root.sub_agent(OtherChild, "early")
    await root.sub_agent(ChildAgent, "middle")
    root.abort_sub_agent(OtherChild, "early", "pause")

    assert root.list_sub_agents() == [
        {"class_name": "OtherChild", "name": "early", "created_at": 100},
        {"class_name": "ChildAgent", "name": "middle", "created_at": 200},
        {"class_name": "ChildAgent", "name": "late", "created_at": 300},
    ]
    assert root.list_sub_agents(ChildAgent) == [
        {"class_name": "ChildAgent", "name": "middle", "created_at": 200},
        {"class_name": "ChildAgent", "name": "late", "created_at": 300},
    ]
    assert root.has_sub_agent(OtherChild, "early") is True
    assert runtime.abort_calls == [("root", "OtherChild\0early", "pause")]


@pytest.mark.asyncio
async def test_delete_releases_proxy_and_failed_delete_preserves_truth(monkeypatch):
    root, runtime, _ = _root(monkeypatch)
    await root.sub_agent(ChildAgent, "leaf")
    first_proxy = runtime.proxies[0]

    await root.delete_sub_agent(ChildAgent, "leaf")

    assert root.has_sub_agent(ChildAgent, "leaf") is False
    assert first_proxy.destroy_calls == 1

    await root.sub_agent(ChildAgent, "leaf")
    second_proxy = runtime.proxies[1]
    runtime.delete_error = RuntimeError("root-only deletion")

    with pytest.raises(RuntimeError, match="root-only deletion"):
        await root.delete_sub_agent(ChildAgent, "leaf")

    assert root.has_sub_agent(ChildAgent, "leaf") is True
    assert second_proxy.destroy_calls == 0

    runtime.delete_error = None
    await root.sub_agent(ChildAgent, "leaf")
    assert second_proxy.destroy_calls == 1
    assert root.has_sub_agent(ChildAgent, "leaf") is True

    await root.delete_sub_agent(OtherChild, "missing")
    assert root.has_sub_agent(OtherChild, "missing") is False


@pytest.mark.asyncio
async def test_delete_releases_proxy_when_registry_cleanup_fails(monkeypatch):
    root, runtime, _ = _root(monkeypatch)
    await root.sub_agent(ChildAgent, "leaf")
    proxy = runtime.proxies[0]

    def fail_cleanup(class_name: str, name: str) -> None:
        raise RuntimeError(f"cannot forget {class_name}/{name}")

    monkeypatch.setattr(root, "_forget_sub_agent", fail_cleanup)

    with pytest.raises(RuntimeError, match="cannot forget ChildAgent/leaf"):
        await root.delete_sub_agent(ChildAgent, "leaf")

    assert runtime.delete_calls == [("root", "ChildAgent\0leaf")]
    assert proxy.destroy_calls == 1


@pytest.mark.asyncio
async def test_cold_delete_with_persisted_schedule_does_not_deadlock(monkeypatch):
    monkeypatch.setattr(schedules_module, "now_ms", lambda: 1_000)
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)
    root, runtime, namespace = _root(monkeypatch, RootAgent, ScheduledChild)
    await root.sub_agent(ScheduledChild, "leaf")
    child = cast(ScheduledChild, runtime.agent("ScheduledChild\0leaf"))
    scheduled = await child.schedule_every(10, "record")
    cold_root = _reincarnate_root(root, runtime, namespace)

    await asyncio.wait_for(
        cold_root.delete_sub_agent(ScheduledChild, "leaf"),
        timeout=0.1,
    )

    assert (
        cold_root.sql("SELECT id FROM cf_agents_jobs WHERE id = ?", scheduled.id) == []
    )
    assert cold_root.has_sub_agent(ScheduledChild, "leaf") is False


@pytest.mark.asyncio
async def test_cold_delete_allows_startup_to_resolve_same_facet(monkeypatch):
    monkeypatch.setattr(schedules_module, "now_ms", lambda: 1_000)
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)
    root, runtime, namespace = _root(monkeypatch, RootAgent, ScheduledChild)
    await root.sub_agent(ScheduledChild, "leaf")
    child = cast(ScheduledChild, runtime.agent("ScheduledChild\0leaf"))
    scheduled = await child.schedule_every(10, "record")
    cold_root = _reincarnate_root(root, runtime, namespace)

    async def resolve_during_startup():
        await cold_root.sub_agent(ScheduledChild, "leaf")

    cold_root.on_start = resolve_during_startup

    await asyncio.wait_for(
        cold_root.delete_sub_agent(ScheduledChild, "leaf"),
        timeout=0.1,
    )

    assert (
        cold_root.sql("SELECT id FROM cf_agents_jobs WHERE id = ?", scheduled.id) == []
    )
    assert cold_root.has_sub_agent(ScheduledChild, "leaf") is False


@pytest.mark.asyncio
async def test_cold_root_rejects_route_that_races_facet_deletion(monkeypatch):
    root, runtime, namespace = _root(monkeypatch, RootAgent, ScheduledChild)
    await root.sub_agent(ScheduledChild, "leaf")
    child = cast(ScheduledChild, runtime.agent("ScheduledChild\0leaf"))
    await root._ensure_initialized()
    cold_root = _reincarnate_root(root, runtime, namespace)
    startup_started = asyncio.Event()
    startup_release = asyncio.Event()

    async def block_startup():
        startup_started.set()
        await startup_release.wait()

    cold_root.on_start = block_startup
    schedule = asyncio.create_task(child.schedule_every(10, "record"))
    await startup_started.wait()

    deletion = asyncio.create_task(cold_root.delete_sub_agent(ScheduledChild, "leaf"))
    startup_release.set()
    result, _ = await asyncio.gather(schedule, deletion, return_exceptions=True)

    assert not isinstance(result, BaseException) or isinstance(result, PermissionError)
    assert cold_root.sql("SELECT id FROM cf_agents_jobs") == []


@pytest.mark.asyncio
async def test_parent_agent_supports_only_a_direct_top_level_parent(monkeypatch):
    root, runtime, namespace = _root(monkeypatch)
    await root.sub_agent(ChildAgent, "leaf")
    child = cast(ChildAgent, runtime.agent("ChildAgent\0leaf"))

    assert await child.parent_agent(RootAgent) is namespace.root_stub
    assert namespace.get_ids[-1] == "root"

    with pytest.raises(RoutingException, match="has no parent"):
        await root.parent_agent(RootAgent)
    with pytest.raises(RoutingException, match="recorded parent class"):
        await child.parent_agent(OtherChild)

    await child.sub_agent(GrandChild, "deep")
    grandchild = cast(GrandChild, runtime.agent("GrandChild\0deep"))
    child.abort_sub_agent(GrandChild, "deep", "pause nested")
    assert runtime.abort_calls[-1] == (
        child.ctx.id.name,
        "GrandChild\0deep",
        "pause nested",
    )
    with pytest.raises(RuntimeError, match="only a root"):
        await child.delete_sub_agent(GrandChild, "deep")
    assert child.has_sub_agent(GrandChild, "deep") is True
    with pytest.raises(RoutingException, match="only reaches a top-level parent"):
        await grandchild.parent_agent(ChildAgent)


@pytest.mark.asyncio
async def test_same_named_descendants_under_different_parents_do_not_alias(
    monkeypatch,
):
    root, runtime, _ = _root(monkeypatch)
    await root.sub_agent(ChildAgent, "left")
    await root.sub_agent(ChildAgent, "right")
    left = cast(ChildAgent, runtime.agent("ChildAgent\0left"))
    right = cast(ChildAgent, runtime.agent("ChildAgent\0right"))

    await left.sub_agent(GrandChild, "same")
    await right.sub_agent(GrandChild, "same")

    identities = runtime.identities_by_key["GrandChild\0same"]
    assert len(identities) == 2
    assert identities[0] != identities[1]
    assert left.has_sub_agent(GrandChild, "same") is True
    assert right.has_sub_agent(GrandChild, "same") is True

    left.abort_sub_agent(GrandChild, "same", "left only")
    assert runtime.abort_calls[-1] == (
        left.ctx.id.name,
        "GrandChild\0same",
        "left only",
    )


@pytest.mark.asyncio
async def test_facet_schedules_are_root_owned_and_isolated(monkeypatch):
    monkeypatch.setattr(schedules_module, "now_ms", lambda: 1_000)
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)
    root, runtime, _ = _root(monkeypatch, RootAgent, ScheduledChild)
    await root.sub_agent(ScheduledChild, "left")
    await root.sub_agent(ScheduledChild, "right")
    left = cast(ScheduledChild, runtime.agent("ScheduledChild\0left"))
    right = cast(ScheduledChild, runtime.agent("ScheduledChild\0right"))

    left_schedule = await left.schedule_every(10, "record", {"side": "same"})
    right_schedule = await right.schedule_every(10, "record", {"side": "same"})

    assert left_schedule.id != right_schedule.id
    assert await root.list_schedules() == ()
    assert await left.list_schedules() == (left_schedule,)
    assert await right.list_schedules() == (right_schedule,)
    assert await left.get_schedule_by_id(right_schedule.id) is None
    assert await left.cancel_schedule(right_schedule.id) is False
    rows = root.sql("SELECT capability, payload FROM cf_agents_jobs ORDER BY id")
    assert len(rows) == 2
    assert {row["capability"] for row in rows} == {"scheduler"}
    owners = {json.loads(row["payload"])["owner_path_key"] for row in rows}
    assert owners == {
        "RootAgent:root/ScheduledChild:left",
        "RootAgent:root/ScheduledChild:right",
    }


@pytest.mark.asyncio
async def test_facet_cannot_cancel_a_corrupt_sibling_schedule(monkeypatch):
    monkeypatch.setattr(schedules_module, "now_ms", lambda: 1_000)
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)
    root, runtime, _ = _root(monkeypatch, RootAgent, ScheduledChild)
    await root.sub_agent(ScheduledChild, "left")
    await root.sub_agent(ScheduledChild, "right")
    left = cast(ScheduledChild, runtime.agent("ScheduledChild\0left"))
    right = cast(ScheduledChild, runtime.agent("ScheduledChild\0right"))
    scheduled = await left.schedule_every(10, "record")
    root.sql(
        "UPDATE cf_agents_jobs SET payload = '{' WHERE id = ?",
        scheduled.id,
    )

    assert await right.cancel_schedule(scheduled.id) is False
    assert root.sql("SELECT id FROM cf_agents_jobs WHERE id = ?", scheduled.id) == [
        {"id": scheduled.id}
    ]
    assert await root.cancel_schedule(scheduled.id) is True


@pytest.mark.asyncio
async def test_facet_scheduling_during_startup_fails_without_deadlock(monkeypatch):
    root, _, _ = _root(monkeypatch, RootAgent, SchedulingOnStartChild)

    with pytest.raises(HookError, match="Lifecycle cannot route"):
        await asyncio.wait_for(
            root.sub_agent(SchedulingOnStartChild, "startup"),
            timeout=0.1,
        )

    assert root.has_sub_agent(SchedulingOnStartChild, "startup") is False


@pytest.mark.asyncio
async def test_root_alarm_routes_nested_callback_and_self_cancellation(monkeypatch):
    monkeypatch.setattr(schedules_module, "now_ms", lambda: 1_000)
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)
    root, runtime, _ = _root(
        monkeypatch,
        RootAgent,
        ScheduledChild,
        ScheduledGrandChild,
    )
    await root.sub_agent(ScheduledChild, "parent")
    parent = cast(ScheduledChild, runtime.agent("ScheduledChild\0parent"))
    await parent.sub_agent(ScheduledGrandChild, "nested")
    nested = cast(
        ScheduledGrandChild,
        runtime.agent("ScheduledGrandChild\0nested"),
    )
    scheduled = await nested.schedule(0, "cancel_self", {"nested": True})

    await root._lifecycle.alarm()

    assert nested.schedule_calls == [({"nested": True}, scheduled.id)]
    assert root.sql("SELECT id FROM cf_agents_jobs") == []


@pytest.mark.asyncio
async def test_routed_recurring_self_cancel_stops_local_retries(monkeypatch):
    monkeypatch.setattr(schedules_module, "now_ms", lambda: 1_000)
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)
    root, runtime, _ = _root(monkeypatch, RootAgent, ScheduledChild)
    await root.sub_agent(ScheduledChild, "self-cancel")
    child = cast(ScheduledChild, runtime.agent("ScheduledChild\0self-cancel"))
    scheduled = await child.schedule_every(
        10,
        "cancel_then_fail",
        options=ScheduleOptions(retry=RetryOptions(max_attempts=3)),
    )
    root.sql(
        "UPDATE cf_agents_jobs SET time = ? WHERE id = ?",
        1_000,
        scheduled.id,
    )

    await root._lifecycle.alarm()

    assert child.cancel_then_fail_calls == 1
    assert root.sql("SELECT id FROM cf_agents_jobs WHERE id = ?", scheduled.id) == []


@pytest.mark.asyncio
async def test_routed_cancel_during_retry_delay_stops_next_attempt(monkeypatch):
    monkeypatch.setattr(schedules_module, "now_ms", lambda: 1_000)
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)
    retry_started = asyncio.Event()
    retry_release = asyncio.Event()

    async def block_retry(*_args):
        retry_started.set()
        await retry_release.wait()

    monkeypatch.setattr(schedules_module, "_retry_sleep", block_retry)
    root, runtime, _ = _root(monkeypatch, RootAgent, ScheduledChild)
    await root.sub_agent(ScheduledChild, "cancel-delay")
    child = cast(ScheduledChild, runtime.agent("ScheduledChild\0cancel-delay"))
    child.failures_remaining = 2
    scheduled = await child.schedule_every(
        10,
        "flaky",
        options=ScheduleOptions(retry=RetryOptions(max_attempts=3)),
    )
    root.sql(
        "UPDATE cf_agents_jobs SET time = ? WHERE id = ?",
        1_000,
        scheduled.id,
    )

    alarm = asyncio.create_task(root._lifecycle.alarm())
    await retry_started.wait()
    assert await child.cancel_schedule(scheduled.id) is True
    retry_release.set()
    await alarm

    assert len(child.retry_calls) == 1
    assert root.sql("SELECT id FROM cf_agents_jobs WHERE id = ?", scheduled.id) == []


@pytest.mark.asyncio
async def test_incomplete_owner_path_is_not_exposed_or_dispatched(monkeypatch):
    monkeypatch.setattr(schedules_module, "now_ms", lambda: 1_000)
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)
    root, runtime, _ = _root(monkeypatch, RootAgent, ScheduledChild)
    await root.sub_agent(ScheduledChild, "legacy-owner")
    child = cast(ScheduledChild, runtime.agent("ScheduledChild\0legacy-owner"))
    scheduled = await child.schedule(0, "record", {"legacy": True})
    [row] = root.sql(
        "SELECT payload FROM cf_agents_jobs WHERE id = ?",
        scheduled.id,
    )
    payload = json.loads(row["payload"])
    payload.pop("owner_path_key")
    root.sql(
        "UPDATE cf_agents_jobs SET payload = ? WHERE id = ?",
        json.dumps(payload),
        scheduled.id,
    )

    assert await child.get_schedule_by_id(scheduled.id) is None
    assert await root.get_schedule_by_id(scheduled.id) is None
    await root._lifecycle.alarm()

    assert child.schedule_calls == []
    assert root.sql("SELECT id FROM cf_agents_jobs") == []


@pytest.mark.asyncio
async def test_routed_callback_retries_and_emits_from_its_owner(monkeypatch):
    monkeypatch.setattr(schedules_module, "now_ms", lambda: 1_000)
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)

    async def no_retry_delay(*_args):
        return None

    monkeypatch.setattr(schedules_module, "_retry_sleep", no_retry_delay)
    root, runtime, _ = _root(monkeypatch, RootAgent, ScheduledChild)
    await root.sub_agent(ScheduledChild, "retry-owner")
    child = cast(ScheduledChild, runtime.agent("ScheduledChild\0retry-owner"))
    child.failures_remaining = 1
    root_events = []
    child_events = []
    root._lifecycle._event_listeners += (root_events.append,)
    child._lifecycle._event_listeners += (child_events.append,)
    scheduled = await child.schedule(
        0,
        "flaky",
        {"retry": True},
        ScheduleOptions(
            retry=RetryOptions(
                max_attempts=2,
                base_delay_ms=1,
                max_delay_ms=1,
            )
        ),
    )
    root.sql(
        "UPDATE cf_agents_jobs SET retry_options = ? WHERE id = ?",
        '{"maxAttempts":2}',
        scheduled.id,
    )

    await root._lifecycle.alarm()

    assert child.retry_calls == [
        ({"retry": True}, scheduled.id, scheduled.time),
        ({"retry": True}, scheduled.id, scheduled.time),
    ]
    assert root_events == []
    assert [event.type for event in child_events] == [
        "schedule:create",
        "schedule:execute",
        "schedule:retry",
    ]
    assert root.sql("SELECT id FROM cf_agents_jobs") == []


@pytest.mark.asyncio
async def test_routed_recurring_infrastructure_failure_keeps_durable_job(monkeypatch):
    monkeypatch.setattr(schedules_module, "now_ms", lambda: 1_000)
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)
    root, runtime, _ = _root(monkeypatch, RootAgent, ScheduledChild)
    await root.sub_agent(ScheduledChild, "retry-owner")
    child = cast(ScheduledChild, runtime.agent("ScheduledChild\0retry-owner"))
    child.infrastructure_error = RuntimeError(
        "Durable Object reset because its code was updated"
    )
    scheduled = await child.schedule_every(
        10,
        "fail_infrastructure",
        options=ScheduleOptions(retry=RetryOptions(max_attempts=1)),
    )
    root.sql(
        "UPDATE cf_agents_jobs SET time = ? WHERE id = ?",
        1_000,
        scheduled.id,
    )

    with pytest.raises(RuntimeError, match="code was updated"):
        await root._lifecycle.alarm()

    assert root.sql("SELECT id FROM cf_agents_jobs WHERE id = ?", scheduled.id) == [
        {"id": scheduled.id}
    ]


@pytest.mark.asyncio
async def test_missing_routed_callback_still_emits_owner_execute_event(monkeypatch):
    monkeypatch.setattr(schedules_module, "now_ms", lambda: 1_000)
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)
    root, runtime, _ = _root(monkeypatch, RootAgent, ScheduledChild)
    await root.sub_agent(ScheduledChild, "updated")
    child = cast(ScheduledChild, runtime.agent("ScheduledChild\0updated"))
    events = []
    child._lifecycle._event_listeners += (events.append,)
    scheduled = await child.schedule(0, "record")
    child.scheduler._callbacks.pop("record")

    await root._lifecycle.alarm()

    assert [event.type for event in events] == ["schedule:create", "schedule:execute"]
    assert root.sql("SELECT id FROM cf_agents_jobs WHERE id = ?", scheduled.id) == []


@pytest.mark.asyncio
async def test_missing_facet_route_prunes_its_schedules(monkeypatch):
    monkeypatch.setattr(schedules_module, "now_ms", lambda: 1_000)
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)
    root, runtime, _ = _root(monkeypatch, RootAgent, ScheduledChild)
    await root.sub_agent(ScheduledChild, "gone")
    child = cast(ScheduledChild, runtime.agent("ScheduledChild\0gone"))
    scheduled = await child.schedule(0, "record")
    root._forget_sub_agent("ScheduledChild", "gone")

    await root._lifecycle.alarm()

    assert child.schedule_calls == []
    assert root.sql("SELECT id FROM cf_agents_jobs WHERE id = ?", scheduled.id) == []


@pytest.mark.asyncio
async def test_stale_route_response_cannot_clean_a_sibling(monkeypatch):
    monkeypatch.setattr(schedules_module, "now_ms", lambda: 1_000)
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)
    root, runtime, _ = _root(monkeypatch, RootAgent, ScheduledChild)
    await root.sub_agent(ScheduledChild, "target")
    await root.sub_agent(ScheduledChild, "victim")
    target = cast(ScheduledChild, runtime.agent("ScheduledChild\0target"))
    victim = cast(ScheduledChild, runtime.agent("ScheduledChild\0victim"))
    target_schedule = await target.schedule(0, "record")
    victim_schedule = await victim.schedule_every(10, "record")
    [target_identity] = runtime.identities_by_owner_key[
        ("root", "ScheduledChild\0target")
    ]
    target_stub = runtime.stubs[target_identity]

    async def malicious_stale_response(_envelope):
        return dumps_wire({"type": "stale", "path": victim.self_path})

    target_stub._cf_route_lifecycle = malicious_stale_response

    await root._lifecycle.alarm()

    assert root.sql(
        "SELECT id FROM cf_agents_jobs WHERE id IN (?, ?) ORDER BY id",
        target_schedule.id,
        victim_schedule.id,
    ) == sorted(
        [{"id": target_schedule.id}, {"id": victim_schedule.id}],
        key=lambda row: row["id"],
    )


@pytest.mark.asyncio
async def test_stale_route_response_cannot_clean_the_root(monkeypatch):
    monkeypatch.setattr(schedules_module, "now_ms", lambda: 1_000)
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)
    root, runtime, _ = _root(monkeypatch, RootAgent, ScheduledChild)
    await root.sub_agent(ScheduledChild, "target")
    target = cast(ScheduledChild, runtime.agent("ScheduledChild\0target"))
    scheduled = await target.schedule(0, "record")
    [target_identity] = runtime.identities_by_owner_key[
        ("root", "ScheduledChild\0target")
    ]

    async def malicious_stale_response(_envelope):
        return dumps_wire({"type": "stale", "path": root.self_path})

    runtime.stubs[target_identity]._cf_route_lifecycle = malicious_stale_response

    await root._lifecycle.alarm()

    assert root.sql("SELECT id FROM cf_agents_jobs WHERE id = ?", scheduled.id) == [
        {"id": scheduled.id}
    ]


@pytest.mark.asyncio
async def test_delete_cleans_encoded_facet_schedule_subtree(monkeypatch):
    monkeypatch.setattr(schedules_module, "now_ms", lambda: 1_000)
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)
    root, runtime, _ = _root(
        monkeypatch,
        RootAgent,
        ScheduledChild,
        ScheduledGrandChild,
    )
    await root.sub_agent(ScheduledChild, "a/b%*")
    child = cast(ScheduledChild, runtime.agent("ScheduledChild\0a/b%*"))
    child_schedule = await child.schedule_every(10, "record")
    [child_row] = root.sql(
        "SELECT payload FROM cf_agents_jobs WHERE id = ?",
        child_schedule.id,
    )
    child_payload = json.loads(child_row["payload"])
    child_payload.pop("owner_path_key")
    child_payload["intervalSeconds"] = 0
    root.sql(
        "UPDATE cf_agents_jobs SET payload = ? WHERE id = ?",
        json.dumps(child_payload),
        child_schedule.id,
    )
    await child.sub_agent(ScheduledGrandChild, "nested")
    nested = cast(
        ScheduledGrandChild,
        runtime.agent("ScheduledGrandChild\0nested"),
    )
    nested_schedule = await nested.schedule_every(10, "record")
    await root.sub_agent(ScheduledChild, "a/b%*x")
    sibling = cast(ScheduledChild, runtime.agent("ScheduledChild\0a/b%*x"))
    sibling_schedule = await sibling.schedule_every(10, "record")
    child_address = _agent_route_address(child.self_path)
    nested_address = _agent_route_address(nested.self_path)
    sibling_address = _agent_route_address(sibling.self_path)
    await root.tasks._sync_root_wake(child_address, "child-task", 10_000)
    await root.tasks._sync_root_wake(nested_address, "nested-task", 10_000)
    await root.tasks._sync_root_wake(sibling_address, "sibling-task", 10_000)
    child_task_job = f"task:{child_address.key}:child-task"
    nested_task_job = f"task:{nested_address.key}:nested-task"
    sibling_task_job = f"task:{sibling_address.key}:sibling-task"
    root.sql(
        "INSERT INTO cf_agents_facet_runs "
        "(owner_path, owner_path_key, run_id, created_at) VALUES (?, ?, ?, ?)",
        json.dumps(nested.self_path),
        "corrupt-owner-key",
        "run",
        1_000,
    )

    await root.delete_sub_agent(ScheduledChild, "a/b%*")

    assert (
        root.sql(
            "SELECT id FROM cf_agents_jobs WHERE id IN (?, ?, ?, ?)",
            child_schedule.id,
            nested_schedule.id,
            child_task_job,
            nested_task_job,
        )
        == []
    )
    assert root.sql("SELECT run_id FROM cf_agents_facet_runs") == []
    assert await sibling.get_schedule_by_id(sibling_schedule.id) == sibling_schedule
    assert root.sql("SELECT id FROM cf_agents_jobs WHERE id = ?", sibling_task_job) == [
        {"id": sibling_task_job}
    ]


@pytest.mark.asyncio
async def test_failed_facet_delete_preserves_routed_schedules(monkeypatch):
    monkeypatch.setattr(schedules_module, "now_ms", lambda: 1_000)
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)
    root, runtime, _ = _root(monkeypatch, RootAgent, ScheduledChild)
    await root.sub_agent(ScheduledChild, "live")
    child = cast(ScheduledChild, runtime.agent("ScheduledChild\0live"))
    scheduled = await child.schedule_every(10, "record")
    child_address = _agent_route_address(child.self_path)
    await root.tasks._sync_root_wake(child_address, "task-run", 10_000)
    task_job = f"task:{child_address.key}:task-run"
    runtime.delete_error = RuntimeError("storage remains live")

    with pytest.raises(RuntimeError, match="storage remains live"):
        await root.delete_sub_agent(ScheduledChild, "live")

    assert root.has_sub_agent(ScheduledChild, "live") is True
    assert await child.get_schedule_by_id(scheduled.id) == scheduled
    assert root.sql("SELECT id FROM cf_agents_jobs WHERE id = ?", task_job) == [
        {"id": task_job}
    ]

    runtime.delete_error = None
    await root._retry_pending_facet_deletions()

    assert root.has_sub_agent(ScheduledChild, "live") is False
    assert root.sql("SELECT id FROM cf_agents_jobs WHERE id = ?", scheduled.id) == []
    assert root.sql("SELECT id FROM cf_agents_jobs WHERE id = ?", task_job) == []


@pytest.mark.asyncio
async def test_deleted_facet_cannot_enqueue_new_root_schedule(monkeypatch):
    monkeypatch.setattr(schedules_module, "now_ms", lambda: 1_000)
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)
    root, runtime, _ = _root(monkeypatch, RootAgent, ScheduledChild)
    await root.sub_agent(ScheduledChild, "deleted")
    child = cast(ScheduledChild, runtime.agent("ScheduledChild\0deleted"))
    await child.schedule_every(10, "record")

    await root.delete_sub_agent(ScheduledChild, "deleted")

    with pytest.raises(PermissionError, match="no longer registered"):
        await child.schedule_every(10, "record")
    assert root.sql("SELECT id FROM cf_agents_jobs") == []


@pytest.mark.asyncio
async def test_cleanup_failure_does_not_leave_deleted_facet_registered(monkeypatch):
    monkeypatch.setattr(schedules_module, "now_ms", lambda: 1_000)
    monkeypatch.setattr(lifecycle_module, "now_ms", lambda: 1_000)
    root, runtime, _ = _root(monkeypatch, RootAgent, ScheduledChild)
    await root.sub_agent(ScheduledChild, "deleted")
    child = cast(ScheduledChild, runtime.agent("ScheduledChild\0deleted"))
    scheduled = await child.schedule_every(10, "record")
    proxy = runtime.proxies[-1]
    cleanup = root._cleanup_facet_prefix

    async def fail_cleanup(_path):
        raise RuntimeError("cleanup unavailable")

    monkeypatch.setattr(root, "_cleanup_facet_prefix", fail_cleanup)

    with pytest.raises(RuntimeError, match="cleanup unavailable"):
        await root.delete_sub_agent(ScheduledChild, "deleted")

    assert root.has_sub_agent(ScheduledChild, "deleted") is False
    assert proxy.destroy_calls == 1
    assert root._has_pending_facet_deletion(ScheduledChild.__name__, "deleted")
    assert root.sql("SELECT id FROM cf_agents_jobs WHERE id = ?", scheduled.id) == [
        {"id": scheduled.id}
    ]

    monkeypatch.setattr(root, "_cleanup_facet_prefix", cleanup)
    await root._retry_pending_facet_deletions()

    assert not root._has_pending_facet_deletion(ScheduledChild.__name__, "deleted")
    assert root.sql("SELECT id FROM cf_agents_jobs WHERE id = ?", scheduled.id) == []


@pytest.mark.asyncio
async def test_nested_http_runs_each_gate_and_preserves_replacement_request(
    monkeypatch,
):
    gates: list[tuple[str, dict[str, str], str]] = []
    leaf_requests: list[Request] = []

    class GatedRoot(Agent):
        async def on_before_sub_agent(self, request, child):
            gates.append((self.name, dict(child), request.url))
            return Request(
                request.url.replace("?q=1", "?replacement=1"),
                method=HTTPMethod.PUT,
                headers={"X-Gate": "root"},
                body="replacement payload",
            )

    class GatedChild(Agent):
        async def on_before_sub_agent(self, request, child):
            gates.append((self.name, dict(child), request.url))

    class LeafAgent(Agent):
        async def on_request(self, request):
            leaf_requests.append(request)
            return Response("leaf", status=207, headers={"X-Leaf": "yes"})

    root, _, _ = _root(monkeypatch, GatedRoot, GatedChild, LeafAgent)
    request = Request(
        "https://example.com/agents/gated-root/root/sub/gated-child/one/"
        "sub/leaf-agent/two/resource?q=1",
        method=HTTPMethod.POST,
        headers={"X-Original": "yes"},
        body="payload",
    )

    response = await root.fetch(request)

    assert response.status == 207
    assert response.body == "leaf"
    assert dict(response.headers) == {"x-leaf": "yes"}
    assert gates == [
        (
            "root",
            {"class_name": "GatedChild", "name": "one"},
            request.url,
        ),
        (
            "one",
            {"class_name": "LeafAgent", "name": "two"},
            "https://example.com/sub/leaf-agent/two/resource?replacement=1",
        ),
    ]
    assert len(leaf_requests) == 1
    forwarded = leaf_requests[0]
    assert forwarded.url == "https://example.com/resource?replacement=1"
    assert forwarded.method == HTTPMethod.PUT
    assert dict(forwarded.headers) == {"x-gate": "root"}
    assert forwarded.body == "replacement payload"


@pytest.mark.asyncio
@pytest.mark.parametrize("stop_at", ["root", "child"])
async def test_http_gate_can_short_circuit_before_the_next_child_wakes(
    monkeypatch,
    stop_at,
):
    wakes: list[str] = []

    class GatedRoot(Agent):
        async def on_before_sub_agent(self, request, child):
            if stop_at == "root":
                return Response("root blocked", status=401)

    class GatedChild(Agent):
        async def on_start(self):
            wakes.append("child")

        async def on_before_sub_agent(self, request, child):
            if stop_at == "child":
                return Response("child blocked", status=403)

    class LeafAgent(Agent):
        async def on_start(self):
            wakes.append("leaf")

    root, _, _ = _root(monkeypatch, GatedRoot, GatedChild, LeafAgent)
    response = await root.fetch(
        Request(
            "https://example.com/agents/gated-root/root/sub/gated-child/one/"
            "sub/leaf-agent/two"
        )
    )

    if stop_at == "root":
        assert response.status == 401
        assert wakes == []
    else:
        assert response.status == 403
        assert wakes == ["child"]
