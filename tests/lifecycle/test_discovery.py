from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any, cast

import fakes
import pytest

from agents import Agent, rpc_callable
from agents.core._discovery import static_mro_members
from agents.lifecycle.websockets import Connection


def test_static_mro_members_use_python_precedence_without_descriptors():
    evaluated = []

    class Root:
        shared = "root"
        root = "root"

        @property
        def dangerous(self):
            evaluated.append(True)
            raise RuntimeError("descriptor evaluated")

    class Left(Root):
        shared = "left"
        left = "left"

    class Right(Root):
        shared = "right"
        right = "right"

    class Leaf(Left, Right):
        leaf = "leaf"

    members = static_mro_members(Leaf())

    assert members["shared"] == "left"
    assert members["root"] == "root"
    assert members["left"] == "left"
    assert members["right"] == "right"
    assert members["leaf"] == "leaf"
    assert members["dangerous"] is Root.__dict__["dangerous"]
    assert evaluated == []


def test_static_mro_members_returns_a_fresh_map():
    class Example:
        value = 1

    instance = Example()
    first = static_mro_members(instance)
    second = static_mro_members(instance)
    first["value"] = 2

    assert second["value"] == 1


def test_static_mro_members_bypass_hostile_metaclass_lookup():
    intercepted = []

    class HostileMeta(type):
        def __getattribute__(cls, name):
            if name in {"__mro__", "__dict__"}:
                intercepted.append(name)
                raise RuntimeError("metaclass lookup")
            return super().__getattribute__(name)

    class Example(metaclass=HostileMeta):
        value = 1

    members = static_mro_members(Example())

    assert members["value"] == 1
    assert intercepted == []


def test_static_mro_members_bypass_hostile_metaclass_data_descriptors():
    evaluated = []

    def read_mro(_cls):
        evaluated.append("mro")
        raise RuntimeError("metaclass MRO descriptor")

    def read_dict(_cls):
        evaluated.append("dict")
        raise RuntimeError("metaclass namespace descriptor")

    hostile_meta = type(
        "HostileMeta",
        (type,),
        {"__mro__": property(read_mro), "__dict__": property(read_dict)},
    )
    example_type = hostile_meta("Example", (), {"value": 1})

    members = static_mro_members(example_type())

    assert members["value"] == 1
    assert evaluated == []


@pytest.mark.asyncio
async def test_rpc_discovery_honors_shadowing_and_does_not_evaluate_properties():
    evaluated = []

    class BaseAgent(Agent):
        @rpc_callable()
        def inherited(self):
            return "base"

        setattr(inherited, "__get__", lambda *_args: evaluated.append("function-get"))

        @rpc_callable()
        def hidden(self):
            return "should not run"

        @rpc_callable()
        def replaced(self):
            return "base"

    class DiscoveryAgent(BaseAgent):
        def hidden(self):
            return "hidden"

        @rpc_callable()
        def replaced(self):
            return "child"

        @property
        def dangerous(self):
            evaluated.append(True)
            raise RuntimeError("descriptor evaluated")

    agent = cast(DiscoveryAgent, fakes.build_agent(cls=DiscoveryAgent))
    connection = fakes.FakeConnection()

    assert agent.inherited() == "base"
    assert agent.replaced() == "child"
    assert callable(DiscoveryAgent.__dict__["replaced"])
    assert evaluated == []

    await agent._handle_rpc(
        cast(Connection, connection),
        {"id": "1", "method": "inherited", "args": []},
    )
    await agent._handle_rpc(
        cast(Connection, connection),
        {"id": "2", "method": "hidden", "args": []},
    )
    await agent._handle_rpc(
        cast(Connection, connection),
        {"id": "3", "method": "replaced", "args": []},
    )
    await agent._handle_rpc(
        cast(Connection, connection),
        {"id": "4", "method": "dangerous", "args": []},
    )

    assert connection.frames == [
        {"type": "rpc", "id": "1", "success": True, "done": True, "result": "base"},
        {
            "type": "rpc",
            "id": "2",
            "success": False,
            "error": "Method hidden is not callable",
        },
        {"type": "rpc", "id": "3", "success": True, "done": True, "result": "child"},
        {
            "type": "rpc",
            "id": "4",
            "success": False,
            "error": "Method dangerous does not exist",
        },
    ]
    assert evaluated == []


@pytest.mark.asyncio
async def test_rpc_discovery_supports_class_and_static_method_decorator_orders():
    class MethodKindsAgent(Agent):
        @rpc_callable()
        @classmethod
        def outer_class(cls):
            return cls.__name__

        @classmethod
        @rpc_callable()
        def inner_class(cls):
            return cls.__name__

        @rpc_callable()
        @staticmethod
        def outer_static():
            return "outer-static"

        @staticmethod
        @rpc_callable()
        def inner_static():
            return "inner-static"

    agent = cast(MethodKindsAgent, fakes.build_agent(cls=MethodKindsAgent))
    connection = fakes.FakeConnection()

    assert agent.outer_class() == "MethodKindsAgent"
    assert agent.inner_class() == "MethodKindsAgent"
    assert agent.outer_static() == "outer-static"
    assert agent.inner_static() == "inner-static"

    for rpc_id, method in enumerate(
        ("outer_class", "inner_class", "outer_static", "inner_static"),
        start=1,
    ):
        await agent._handle_rpc(
            cast(Connection, connection),
            {"id": str(rpc_id), "method": method, "args": []},
        )

    assert [frame["result"] for frame in connection.frames] == [
        "MethodKindsAgent",
        "MethodKindsAgent",
        "outer-static",
        "inner-static",
    ]


def test_rpc_callable_rejects_arbitrary_callable_objects():
    class CallableObject:
        def __call__(self):
            return None

    with pytest.raises(TypeError, match="only decorate methods"):
        rpc_callable()(cast(Any, CallableObject()))


def test_rpc_callable_preserves_identity_and_attaches_frozen_metadata():
    def method():
        return None

    decorated = rpc_callable()(method)
    metadata = method.__dict__["__agents_rpc_metadata__"]

    assert decorated is method
    assert metadata.streaming is False
    with pytest.raises(FrozenInstanceError):
        setattr(metadata, "streaming", True)


def test_rpc_callable_rejects_class_descriptors_that_do_not_wrap_functions():
    descriptor = classmethod(cast(Any, property(lambda _self: "unsafe")))

    with pytest.raises(TypeError, match="only decorate methods"):
        rpc_callable()(descriptor)


@pytest.mark.parametrize("descriptor_type", [classmethod, staticmethod])
def test_rpc_discovery_ignores_hostile_descriptor_subclasses(descriptor_type):
    evaluated = []

    class HostileDescriptor(descriptor_type):
        def __getattribute__(self, name):
            if name in {"__dict__", "__func__", "__get__"}:
                evaluated.append(name)
                raise RuntimeError("descriptor evaluated")
            return super().__getattribute__(name)

    class HostileAgent(Agent):
        hostile = HostileDescriptor(lambda *_args: None)

    fakes.build_agent(cls=HostileAgent)

    assert evaluated == []


def test_rpc_discovery_does_not_inspect_hostile_arbitrary_members():
    evaluated = []

    class HostileMemberType(type):
        def __eq__(self, _other):
            evaluated.append("type equality")
            raise RuntimeError("type equality evaluated")

    class HostileMember(metaclass=HostileMemberType):
        pass

    class HostileAgent(Agent):
        hostile = HostileMember()

    fakes.build_agent(cls=HostileAgent)

    assert evaluated == []


def test_rpc_discovery_ignores_hostile_noncanonical_member_names():
    evaluated = []

    class HostileName(str):
        def __hash__(self):
            evaluated.append("hash")
            return str.__hash__(self)

        def __eq__(self, _other):
            evaluated.append("equality")
            raise RuntimeError("name equality evaluated")

    hostile_name = HostileName("hostile")
    hostile_agent = type(
        "HostileAgent",
        (Agent,),
        {cast(str, hostile_name): lambda _self: None},
    )
    evaluated.clear()

    fakes.build_agent(cls=hostile_agent)

    assert evaluated == []


def test_rpc_discovery_does_not_inspect_hostile_metadata():
    evaluated = []

    class HostileMetadata:
        @property
        def __class__(self):
            evaluated.append("class")
            raise RuntimeError("class evaluated")

    def method(_self):
        return None

    method.__dict__["__agents_rpc_metadata__"] = HostileMetadata()
    hostile_agent = type("HostileAgent", (Agent,), {"method": method})

    fakes.build_agent(cls=hostile_agent)

    assert evaluated == []


def test_rpc_discovery_does_not_compare_hostile_metadata_keys():
    evaluated = []

    class HostileKey:
        def __hash__(self):
            return hash("__agents_rpc_metadata__")

        def __eq__(self, _other):
            evaluated.append("equality")
            raise RuntimeError("metadata key equality evaluated")

    def method(_self):
        return None

    cast(dict[Any, Any], method.__dict__)[HostileKey()] = object()
    hostile_agent = type("HostileAgent", (Agent,), {"method": method})

    fakes.build_agent(cls=hostile_agent)

    assert evaluated == []


def test_rpc_maps_and_bound_entries_are_fresh_per_agent_instance():
    class RpcAgent(Agent):
        @rpc_callable()
        def ping(self):
            return "pong"

    first = fakes.build_agent(cls=RpcAgent)
    second = fakes.build_agent(cls=RpcAgent)
    first_map = object.__getattribute__(first, "_Agent__rpc_meths")
    second_map = object.__getattribute__(second, "_Agent__rpc_meths")

    assert first_map is not second_map
    assert first_map["ping"] is not second_map["ping"]
