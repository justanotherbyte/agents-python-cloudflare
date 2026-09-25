from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .chat.agent import AIChatAgent
    from .chat.folding import apply_chunk_to_parts
    from .chat.protocol import ChatMessageType
    from .chat.types import ChatOptions
    from .core.agent import Agent
    from .core.agent_tools import AgentToolResult, AgentToolStatus
    from .core.response import StreamingResponse
    from .core.routing import route_agent_request
    from .core.rpc import rpc_callable
    from .lifecycle.fiber import (
        FiberContext,
        FiberInspection,
        FiberRecoveryContext,
        FiberRecoveryResult,
        FiberSignal,
        StartFiberResult,
    )

__all__ = [
    "AIChatAgent",
    "Agent",
    "AgentToolResult",
    "AgentToolStatus",
    "ChatMessageType",
    "ChatOptions",
    "FiberContext",
    "FiberInspection",
    "FiberRecoveryContext",
    "FiberRecoveryResult",
    "FiberSignal",
    "StartFiberResult",
    "StreamingResponse",
    "apply_chunk_to_parts",
    "route_agent_request",
    "rpc_callable",
]

_EXPORTS = {
    "AIChatAgent": (".chat.agent", "AIChatAgent"),
    "Agent": (".core.agent", "Agent"),
    "AgentToolResult": (".core.agent_tools", "AgentToolResult"),
    "AgentToolStatus": (".core.agent_tools", "AgentToolStatus"),
    "ChatMessageType": (".chat.protocol", "ChatMessageType"),
    "ChatOptions": (".chat.types", "ChatOptions"),
    "FiberContext": (".lifecycle.fiber", "FiberContext"),
    "FiberInspection": (".lifecycle.fiber", "FiberInspection"),
    "FiberRecoveryContext": (".lifecycle.fiber", "FiberRecoveryContext"),
    "FiberRecoveryResult": (".lifecycle.fiber", "FiberRecoveryResult"),
    "FiberSignal": (".lifecycle.fiber", "FiberSignal"),
    "StartFiberResult": (".lifecycle.fiber", "StartFiberResult"),
    "StreamingResponse": (".core.response", "StreamingResponse"),
    "apply_chunk_to_parts": (".chat.folding", "apply_chunk_to_parts"),
    "route_agent_request": (".core.routing", "route_agent_request"),
    "rpc_callable": (".core.rpc", "rpc_callable"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, member_name = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(import_module(module_name, __name__), member_name)
    globals()[name] = value
    return value
