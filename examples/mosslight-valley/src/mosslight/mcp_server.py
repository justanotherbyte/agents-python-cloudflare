from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from workers import DurableObject

from .world import act_on_farm, inspect_farm

__all__ = (
    "MCP_PROTOCOL_VERSION",
    "MCP_SERVER_ID",
    "TOOLS",
    "FarmTools",
)

MCP_SERVER_ID = "mosslight-tools"
MCP_PROTOCOL_VERSION = "2025-06-18"


def _response(request_id: object, result: Mapping[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": dict(result)}


def _error(
    request_id: object,
    message: str,
    *,
    code: int = -32602,
) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": code, "message": message},
    }


TOOLS = [
    {
        "name": "inspect_farm",
        "title": "Inspect the farm",
        "description": "Read the current farm condition before choosing an action.",
        "inputSchema": {
            "type": "object",
            "properties": {"state": {"type": "object"}},
            "required": ["state"],
        },
    },
    {
        "name": "act_on_farm",
        "title": "Act in Mosslight Valley",
        "description": (
            "Apply one validated farm action and return the next world state."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "state": {"type": "object"},
                "action": {
                    "type": "string",
                    "enum": [
                        "plant",
                        "water",
                        "harvest",
                        "forage",
                        "fish",
                        "talk",
                        "sell",
                        "rest",
                    ],
                },
                "target": {"type": "string"},
                "actor": {
                    "type": "string",
                    "enum": ["wisp", "mira", "bramble", "nori", "tansy"],
                },
            },
            "required": ["state", "action", "actor"],
        },
    },
]


class FarmTools(DurableObject):
    async def ensure_initialized(self, _props=None):
        return None

    async def destroy(self):
        return None

    async def handleMcpMessage(self, message):
        return await self.handle_mcp_message(message)

    async def handle_mcp_message(self, raw_message, _signal=None):
        message = (
            raw_message.to_py()
            if callable(getattr(raw_message, "to_py", None))
            else raw_message
        )
        if not isinstance(message, Mapping):
            return _error(None, "MCP message must be an object")

        request_id = message.get("id")
        method = message.get("method")
        if message.get("jsonrpc") != "2.0" or not isinstance(method, str):
            return _error(request_id, "Invalid JSON-RPC request", code=-32600)
        if "id" not in message:
            return None
        if method == "initialize":
            return _response(
                request_id,
                {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "mosslight-farm-tools", "version": "1.0.0"},
                    "instructions": (
                        "Use these tools to inspect and change the farm world."
                    ),
                },
            )
        if method == "tools/list":
            return _response(request_id, {"tools": TOOLS})
        if method != "tools/call":
            return _error(
                request_id,
                f"Unsupported MCP method: {method}",
                code=-32601,
            )

        params = message.get("params")
        if not isinstance(params, Mapping):
            return _error(request_id, "Tool call params must be an object")
        arguments = params.get("arguments")
        if not isinstance(arguments, Mapping):
            return _error(request_id, "Tool arguments must be an object")
        state = arguments.get("state")
        if not isinstance(state, Mapping):
            return _error(request_id, "Tool state must be an object")

        name = params.get("name")
        if name == "inspect_farm":
            result = inspect_farm(deepcopy(dict(state)))
        elif name == "act_on_farm":
            action = arguments.get("action")
            target = arguments.get("target", "")
            actor = arguments.get("actor")
            if (
                not isinstance(action, str)
                or not isinstance(target, str)
                or not isinstance(actor, str)
            ):
                return _error(request_id, "Action, target, and actor must be strings")
            result = act_on_farm(state, action, target, actor)
        else:
            return _error(request_id, f"Unknown farm tool: {name}")
        return _response(request_id, result)


setattr(FarmTools, "__unsafe_ensureInitialized", FarmTools.ensure_initialized)
