from __future__ import annotations

from enum import StrEnum
from typing import Any, TypedDict

from .utils import MISSING

FrameT = dict[str, Any]


class MessageType(StrEnum):
    CF_AGENT_MCP_SERVERS = "cf_agent_mcp_servers"
    CF_MCP_AGENT_EVENT = "cf_mcp_agent_event"
    CF_AGENT_STATE = "cf_agent_state"
    CF_AGENT_STATE_ERROR = "cf_agent_state_error"
    CF_AGENT_IDENTITY = "cf_agent_identity"
    CF_AGENT_SESSION = "cf_agent_session"
    CF_AGENT_SESSION_ERROR = "cf_agent_session_error"
    RPC = "rpc"


# Record shapes that cross a boundary. Declared rather than left as dict[str, Any] so a
# missing or misspelled key is a checker error at the site that builds it, rather than a
# hole the other runtime discovers.


class McpServers(TypedDict):
    servers: dict[str, Any]
    tools: list[Any]
    prompts: list[Any]
    resources: list[Any]


class PathStep(TypedDict):
    # camelCase because the digest is computed over these keys in this order and the TS
    # runtime has to reproduce it byte for byte.
    className: str
    name: str


class SubAgentRef(TypedDict):
    class_name: str
    name: str


class SubAgentRecord(SubAgentRef):
    created_at: int


# Frame builders. The client branches on key *presence*, so which keys are omitted is
# as load-bearing as their values — keeping that in one place per frame family means a
# new call site cannot get it subtly wrong. See AGENTS.md, wire invariants.


def identity_frame(name: str, agent: str) -> FrameT:
    return {"type": MessageType.CF_AGENT_IDENTITY, "name": name, "agent": agent}


def state_frame(state: Any) -> FrameT:
    return {"type": MessageType.CF_AGENT_STATE, "state": state}


def state_error_frame(error: str) -> FrameT:
    return {"type": MessageType.CF_AGENT_STATE_ERROR, "error": error}


def mcp_servers_frame(mcp: Any) -> FrameT:
    return {"type": MessageType.CF_AGENT_MCP_SERVERS, "mcp": mcp}


def rpc_chunk(rpc_id: str, result: Any) -> FrameT:
    # done must be present and false: its presence is what marks this a stream rather
    # than the final answer.
    return {
        "type": MessageType.RPC,
        "id": rpc_id,
        "success": True,
        "result": result,
        "done": False,
    }


def rpc_result(rpc_id: str, result: Any = MISSING) -> FrameT:
    # An absent result is omitted rather than nulled, because the client hands it
    # straight to resolve() with no undefined guard, so null is a different answer.
    frame: FrameT = {
        "type": MessageType.RPC,
        "id": rpc_id,
        "success": True,
        "done": True,
    }
    if result is not MISSING:
        frame["result"] = result
    return frame


def rpc_error(rpc_id: str, error: str) -> FrameT:
    # Carries neither done nor result; that absence is how the client tells an error
    # frame from a terminal one.
    return {"type": MessageType.RPC, "id": rpc_id, "success": False, "error": error}
