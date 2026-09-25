from __future__ import annotations

import types
from collections.abc import Mapping

import fakes
import pytest

from agents import Agent
from agents.mcp.client import MCPClientManager, RPCTransportAdapter
from agents.workflows import WorkflowProgressCallback


class AgentNamespace:
    def idFromName(self, name: str) -> str:
        return name


class EnvWrapper:
    def __init__(self, env: object) -> None:
        self._env = env

    def __getattr__(self, name: str) -> object:
        return getattr(self._env, name)


class MCPNamespace:
    def __init__(self) -> None:
        self.stub = types.SimpleNamespace(initializations=[])

        async def initialize(props: object = None) -> None:
            self.stub.initializations.append(props)

        setattr(self.stub, "__unsafe_ensureInitialized", initialize)
        self.calls: list[object] = []

    def idFromName(self, name: str) -> str:
        return f"id:{name}"

    def get(self, durable_id: object) -> object:
        self.calls.append(durable_id)
        return self.stub


class WorkflowInstance:
    async def status(self) -> Mapping[str, object]:
        return {"status": "running"}


class WorkflowBinding:
    def __init__(self) -> None:
        self.created: list[dict[str, object]] = []
        self.instance = WorkflowInstance()

    async def create(self, options: Mapping[str, object]) -> Mapping[str, object]:
        self.created.append(dict(options))
        return {"id": options["id"]}

    async def get(self, workflow_id: str) -> WorkflowInstance:
        return self.instance


class IntegratedAgent(Agent):
    def initial_state(self) -> dict[str, object]:
        return {"initial": True}

    async def workflow_target(self, value: str) -> str:
        return f"handled:{value}"

    async def on_workflow_progress(
        self,
        workflow_name: str,
        workflow_id: str,
        progress: object,
    ) -> None:
        self.progress = (workflow_name, workflow_id, progress)


@pytest.mark.asyncio
async def test_agent_installs_mcp_and_composes_workflow_operations() -> None:
    reports = WorkflowBinding()
    env = types.SimpleNamespace(
        INTEGRATED_AGENT=AgentNamespace(),
        REPORTS=reports,
    )
    agent = IntegratedAgent(fakes.FakeCtx(name="tenant-7"), env)

    assert isinstance(agent.mcp, MCPClientManager)

    await agent._ensure_initialized()
    workflow_id = await agent.run_workflow(
        "REPORTS",
        {"report": 42},
    )

    assert workflow_id.startswith("wf_")
    assert reports.created[0]["params"] == {
        "report": 42,
        "__agentName": "tenant-7",
        "__agentBinding": "INTEGRATED_AGENT",
        "__workflowName": "REPORTS",
        "__agentOrigin": {
            "kind": "agent",
            "version": 1,
            "binding": "INTEGRATED_AGENT",
            "name": "tenant-7",
        },
    }
    assert agent.get_workflow(workflow_id).workflow_name == "REPORTS"


@pytest.mark.asyncio
async def test_agent_detects_binding_through_workers_env_wrapper() -> None:
    reports = WorkflowBinding()
    env = EnvWrapper(
        types.SimpleNamespace(
            INTEGRATED_AGENT=AgentNamespace(),
            REPORTS=reports,
        )
    )
    agent = IntegratedAgent(fakes.FakeCtx(name="tenant-7"), env)

    await agent._ensure_initialized()
    await agent.run_workflow("REPORTS", {})

    assert reports.created[0]["params"]["__agentBinding"] == "INTEGRATED_AGENT"


@pytest.mark.asyncio
async def test_agent_installs_rpc_mcp_transport_with_env_binding_resolver() -> None:
    namespace = MCPNamespace()
    agent = fakes.build_agent(env=types.SimpleNamespace(MCP_SERVER=namespace))

    transport = agent.mcp._transports["rpc"]
    assert isinstance(transport, RPCTransportAdapter)
    assert (
        await transport._resolver.resolve("MCP_SERVER", "docs", {"tenant": "acme"})
        is namespace.stub
    )
    assert namespace.calls == ["id:rpc:docs"]
    assert namespace.stub.initializations == [{"tenant": "acme"}]


def test_workflow_origin_descriptor_is_not_evaluated_during_construction() -> None:
    class HostileWorkflowOrigin:
        def __get__(self, instance: object, owner: object) -> object:
            raise AssertionError("workflow origin descriptor evaluated")

    class DescriptorAgent(Agent):
        _workflow_origin = HostileWorkflowOrigin()

    fakes.build_agent(cls=DescriptorAgent)


@pytest.mark.parametrize(
    "name",
    [
        "_cf_invokeAgentPath",
        "_cf_broadcastAgentPath",
        "_workflow_handleCallback",
        "_workflow_broadcast",
        "_workflow_updateState",
    ],
)
def test_workflow_rpc_apertures_are_reserved(name: str) -> None:
    with pytest.raises(TypeError, match=name):
        type("InvalidAgent", (Agent,), {name: lambda self: None})


@pytest.mark.asyncio
async def test_workflow_rpc_apertures_update_ledger_state_and_user_hooks() -> None:
    agent = fakes.build_agent(cls=IntegratedAgent)
    await agent._ensure_initialized()
    agent._workflows._ledger.track("wf-1", "REPORTS")

    await agent._workflow_handleCallback(
        {
            "type": "progress",
            "workflowName": "REPORTS",
            "workflowId": "wf-1",
            "timestamp": 10,
            "progress": {"percent": 50},
        }
    )

    assert agent.get_workflow("wf-1").status == "running"
    assert agent.progress == ("REPORTS", "wf-1", {"percent": 50})

    await agent._workflow_updateState("set", {"count": 1})
    await agent._workflow_updateState("merge", {"ready": True})
    assert agent.state == {"count": 1, "ready": True}
    await agent._workflow_updateState("reset")
    assert agent.state == {"initial": True}


@pytest.mark.asyncio
async def test_workflow_path_rpc_only_invokes_methods_on_the_addressed_agent() -> None:
    agent = fakes.build_agent(cls=IntegratedAgent, name="tenant-7")
    await agent._ensure_initialized()

    assert (
        await agent._cf_invokeAgentPath(
            agent.self_path,
            "workflow_target",
            ["value"],
        )
        == "handled:value"
    )
    with pytest.raises(ValueError, match="does not descend"):
        await agent._cf_invokeAgentPath(
            [{"className": "OtherAgent", "name": "other"}],
            "workflow_target",
            [],
        )
    with pytest.raises(ValueError, match="not callable"):
        await agent._cf_invokeAgentPath(agent.self_path, "__str__", [])


@pytest.mark.asyncio
async def test_mcp_state_changes_publish_the_flattened_catalog_frame() -> None:
    agent = fakes.build_agent()
    connection = fakes.FakeConnection()
    agent._connections[connection.id] = connection
    await agent._ensure_initialized()
    assert len(connection.frames) == 1
    connection.sent.clear()

    agent.mcp._fire_state_changed()

    assert connection.frames == [
        {
            "type": "cf_agent_mcp_servers",
            "mcp": {
                "servers": {},
                "tools": [],
                "prompts": [],
                "resources": [],
            },
        }
    ]


@pytest.mark.asyncio
async def test_agent_mcp_delegates_register_connect_and_remove() -> None:
    agent = fakes.build_agent()
    await agent._ensure_initialized()
    calls: list[tuple[str, object]] = []

    async def register(server_id: str, **options: object) -> str:
        calls.append(("register", (server_id, options)))
        return server_id

    async def connect(server_id: str) -> dict[str, object]:
        calls.append(("connect", server_id))
        return {"state": "ready"}

    async def remove(server_id: str) -> None:
        calls.append(("remove", server_id))

    agent.mcp.register_server = register
    agent.mcp.connect_to_server = connect
    agent.mcp.remove_server = remove

    result = await agent.add_mcp_server(
        "docs",
        url="https://mcp.example.com",
        name="Docs",
    )
    await agent.remove_mcp_server("docs")

    assert result == {"id": "docs", "state": "ready"}
    assert calls == [
        (
            "register",
            (
                "docs",
                {
                    "url": "https://mcp.example.com",
                    "name": "Docs",
                    "callback_url": "",
                    "client_id": None,
                    "client": None,
                    "transport": None,
                    "retry": None,
                    "binding_name": None,
                    "props": None,
                },
            ),
        ),
        ("connect", "docs"),
        ("remove", "docs"),
    ]


def test_callback_type_remains_the_concern_module_type() -> None:
    callback = WorkflowProgressCallback(
        workflow_name="REPORTS",
        workflow_id="wf-1",
        timestamp=1,
        progress=None,
    )
    assert callback.workflow_id == "wf-1"
