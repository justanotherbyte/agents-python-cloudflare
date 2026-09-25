from importlib.util import find_spec
from pathlib import Path
import subprocess
import sys

import agents
import agents.chat as chat_module
import agents.chat.agent as chat_agent_module
import agents.chat.folding as folding_module
import agents.chat.protocol as chat_protocol_module
import agents.chat.types as chat_types_module
import agents.core.agent as agent_module
import agents.core.agent_tools as agent_tools_module
import agents.core.response as response_module
import agents.core.routing as routing_module
import agents.core.rpc as rpc_module
import agents.lifecycle as lifecycle_module
import agents.lifecycle.fiber as fiber_module


ROOT_EXPORTS = {
    "AIChatAgent": chat_agent_module.AIChatAgent,
    "Agent": agent_module.Agent,
    "AgentToolResult": agent_tools_module.AgentToolResult,
    "AgentToolStatus": agent_tools_module.AgentToolStatus,
    "ChatMessageType": chat_protocol_module.ChatMessageType,
    "ChatOptions": chat_types_module.ChatOptions,
    "FiberContext": fiber_module.FiberContext,
    "FiberInspection": fiber_module.FiberInspection,
    "FiberRecoveryContext": fiber_module.FiberRecoveryContext,
    "FiberRecoveryResult": fiber_module.FiberRecoveryResult,
    "FiberSignal": fiber_module.FiberSignal,
    "StartFiberResult": fiber_module.StartFiberResult,
    "StreamingResponse": response_module.StreamingResponse,
    "apply_chunk_to_parts": folding_module.apply_chunk_to_parts,
    "route_agent_request": routing_module.route_agent_request,
    "rpc_callable": rpc_module.rpc_callable,
}

REMOVED_ROOT_EXPORTS = {
    "PartyServer",
    "route_party_request",
    "url_path",
}

CHAT_EXPORTS = {
    "AIChatAgent",
    "ChatMessageType",
    "ChatOptions",
    "apply_chunk_to_parts",
}

LIFECYCLE_EXPORTS = {
    "CapabilityRequestContext",
    "CapabilityWebSocketCloseContext",
    "CapabilityWebSocketErrorContext",
    "CapabilityWebSocketMessageContext",
    "CapabilityWebSocketUpgradeContext",
    "CurrentLifecycleContext",
    "Lifecycle",
    "LifecycleCapability",
    "LifecycleEvent",
    "LifecycleEvents",
    "LifecycleHostContextScope",
    "LifecycleJob",
    "LifecycleJobContext",
    "LifecycleJobOutcome",
    "LifecycleJobPushOptions",
    "LifecycleJobReschedule",
    "LifecycleJobs",
    "LifecycleMemoryLimitContext",
    "LifecycleRetainedWork",
    "LifecycleRouteAddress",
    "LifecycleRouteContext",
    "LifecycleRouteEnvelope",
    "LifecycleRouteTransport",
    "LifecycleRoutes",
    "LifecycleServices",
    "LifecycleSockets",
    "LifecycleSql",
    "LifecycleStorage",
    "get_current_lifecycle_context",
}

MOVED_FLAT_MODULES = {
    "agents._discovery",
    "agents._lifecycle_job_driver",
    "agents._wire",
    "agents.agent",
    "agents.agent_tools",
    "agents.chat_folding",
    "agents.connection_state",
    "agents.core_schema",
    "agents.error",
    "agents.fiber",
    "agents.fiber_schema",
    "agents.lifecycle_jobs",
    "agents.pre_stream",
    "agents.protocol",
    "agents.response",
    "agents.resumable_stream",
    "agents.subagent_relay",
    "agents.turn_queue",
    "agents.utils",
    "agents.websockets",
}


def test_package_exports_match_the_stage_three_snapshot():
    assert set(agents.__all__) == set(ROOT_EXPORTS)
    assert all(getattr(agents, name) is owner for name, owner in ROOT_EXPORTS.items())
    assert all(not hasattr(agents, name) for name in REMOVED_ROOT_EXPORTS)


def test_subpackage_exports_are_exact():
    assert set(chat_module.__all__) == CHAT_EXPORTS
    assert set(lifecycle_module.__all__) == LIFECYCLE_EXPORTS


def test_removed_compatibility_modules_and_classes_are_absent():
    assert find_spec("agents.party") is None
    assert all(find_spec(module_name) is None for module_name in MOVED_FLAT_MODULES)
    assert not hasattr(fiber_module, "FiberSupport")
    assert not hasattr(agent_tools_module, "AgentToolSupport")


def test_mcp_server_import_does_not_load_agent_or_client_modules():
    root = Path(__file__).parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import _runtime_stubs; _runtime_stubs.install(); "
            "import agents.mcp.server; import sys; "
            "assert 'agents.core.agent' not in sys.modules; "
            "assert 'agents.mcp.client' not in sys.modules",
        ],
        cwd=root,
        env={"PYTHONPATH": str(root / "tests")},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
