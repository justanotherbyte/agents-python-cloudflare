from __future__ import annotations

import pytest

from agents.workflows import (
    AgentWorkflowFacetOrigin,
    AgentWorkflowPathStep,
    AgentWorkflowRootOrigin,
    RpcOnlyAgentStub,
    decode_workflow_origin,
    encode_workflow_origin,
    resolve_workflow_agent,
    _WorkersWorkflowAgentResolver,
)


class Resolver:
    def __init__(self) -> None:
        self.root = object()
        self.resolved: list[tuple[str, str]] = []
        self.invocations: list[
            tuple[tuple[AgentWorkflowPathStep, ...], str, tuple]
        ] = []
        self.released: list[object] = []

    async def resolve_root(self, binding: str, name: str) -> object:
        self.resolved.append((binding, name))
        return self.root

    async def invoke_agent_path(
        self,
        root: object,
        path: tuple[AgentWorkflowPathStep, ...],
        method: str,
        args: tuple[object, ...],
    ) -> object:
        assert root is self.root
        self.invocations.append((path, method, args))
        return {"method": method, "args": list(args)}

    async def release(self, agent: object) -> None:
        self.released.append(agent)


def test_root_origin_round_trips_the_persisted_version_one_shape():
    origin = AgentWorkflowRootOrigin(binding="RootAgents", name="tenant-7")

    wire = encode_workflow_origin(origin)

    assert wire == {
        "kind": "agent",
        "version": 1,
        "binding": "RootAgents",
        "name": "tenant-7",
    }
    assert decode_workflow_origin(wire) == origin


def test_facet_origin_round_trips_a_root_first_path():
    origin = AgentWorkflowFacetOrigin(
        root_binding="RootAgents",
        path=(
            AgentWorkflowPathStep(class_name="RootAgent", name="root"),
            AgentWorkflowPathStep(class_name="ChildAgent", name="child"),
        ),
    )

    wire = encode_workflow_origin(origin)

    assert wire == {
        "kind": "facet",
        "version": 1,
        "rootBinding": "RootAgents",
        "path": [
            {"className": "RootAgent", "name": "root"},
            {"className": "ChildAgent", "name": "child"},
        ],
    }
    assert decode_workflow_origin(wire) == origin


@pytest.mark.parametrize(
    "wire",
    [
        {"kind": "agent", "version": 2, "binding": "RootAgents", "name": "x"},
        {"kind": "future", "version": 1},
        {"kind": "facet", "version": 1, "rootBinding": "RootAgents", "path": []},
    ],
)
def test_unknown_or_malformed_origins_fail_loudly(wire):
    with pytest.raises(ValueError):
        decode_workflow_origin(wire)


@pytest.mark.asyncio
async def test_root_origin_resolves_the_named_agent_directly():
    resolver = Resolver()

    agent = await resolve_workflow_agent(
        AgentWorkflowRootOrigin(binding="RootAgents", name="root"), resolver
    )

    assert agent is resolver.root
    assert resolver.resolved == [("RootAgents", "root")]


@pytest.mark.asyncio
async def test_facet_origin_returns_an_rpc_only_path_stub():
    resolver = Resolver()
    path = (
        AgentWorkflowPathStep(class_name="RootAgent", name="root"),
        AgentWorkflowPathStep(class_name="ChildAgent", name="child"),
    )

    agent = await resolve_workflow_agent(
        AgentWorkflowFacetOrigin(root_binding="RootAgents", path=path), resolver
    )

    assert isinstance(agent, RpcOnlyAgentStub)
    assert await agent.record_result("task-1", {"ok": True}) == {
        "method": "record_result",
        "args": ["task-1", {"ok": True}],
    }
    assert resolver.resolved == [("RootAgents", "root")]
    assert resolver.invocations == [(path, "record_result", ("task-1", {"ok": True}))]
    with pytest.raises(RuntimeError, match="RPC-only stub"):
        agent.fetch("https://example.com")


@pytest.mark.asyncio
async def test_workers_resolver_initializes_root_stub_before_returning_it():
    calls = []

    class Stub: ...

    stub = Stub()

    async def initialize():
        calls.append("initialized")

    setattr(stub, "__unsafe_ensureInitialized", initialize)

    class Namespace:
        def idFromName(self, name):
            return f"id:{name}"

        def get(self, durable_id):
            calls.append(durable_id)
            return stub

    env = type("Env", (), {"ROOT_AGENTS": Namespace()})()

    resolved = await _WorkersWorkflowAgentResolver(env).resolve_root(
        "ROOT_AGENTS", "tenant-7"
    )

    assert resolved is stub
    assert calls == ["id:tenant-7", "initialized"]
