from typing import Any, Protocol

from .protocol import FrameT
from .utils import MISSING


# Agent Tool frames sit outside MessageType and the cf_agent_* namespace. The client
# deduplicates on (parentToolCallId, runId, sequence), so optional keys are omitted
# rather than sent as null.
AGENT_TOOL_EVENT = "agent-tool-event"
CHILD_AGENT_TOOL_CHUNK_PAGE_SIZE = 100


class ChildAgentToolStub(Protocol):
    async def _cf_start_agent_tool_run(self, input_json: str, run_id: str) -> str: ...

    async def _cf_inspect_agent_tool_run(self, run_id: str) -> str | None: ...

    async def _cf_get_agent_tool_chunks(
        self,
        run_id: str,
        after_sequence: int = -1,
        limit: int = CHILD_AGENT_TOOL_CHUNK_PAGE_SIZE,
    ) -> str: ...

    async def _cf_cancel_agent_tool_run(
        self, run_id: str, reason: str | None = None
    ) -> None: ...


def agent_tool_event_frame(
    event: FrameT,
    sequence: int,
    *,
    parent_tool_call_id: str | None = None,
    replay: bool = False,
) -> FrameT:
    # parentToolCallId and replay are omitted unless they carry information.
    frame: FrameT = {"type": AGENT_TOOL_EVENT, "sequence": sequence, "event": event}
    if parent_tool_call_id is not None:
        frame["parentToolCallId"] = parent_tool_call_id
    if replay:
        frame["replay"] = True
    return frame


def agent_tool_started_event(
    run_id: str,
    agent_type: str,
    order: int,
    *,
    input_preview: Any = MISSING,
    display: Any = MISSING,
) -> FrameT:
    event: FrameT = {
        "kind": "started",
        "runId": run_id,
        "agentType": agent_type,
        "order": order,
    }
    if input_preview is not MISSING:
        event["inputPreview"] = input_preview
    if display is not MISSING:
        event["display"] = display
    return event


def agent_tool_chunk_event(run_id: str, body: str) -> FrameT:
    return {"kind": "chunk", "runId": run_id, "body": body}


def agent_tool_finished_event(run_id: str, summary: str) -> FrameT:
    return {"kind": "finished", "runId": run_id, "summary": summary}


def agent_tool_error_event(run_id: str, error: str) -> FrameT:
    return {"kind": "error", "runId": run_id, "error": error}


def agent_tool_aborted_event(run_id: str, reason: str | None = None) -> FrameT:
    event: FrameT = {"kind": "aborted", "runId": run_id}
    if reason is not None:
        event["reason"] = reason
    return event


def agent_tool_interrupted_event(
    run_id: str,
    error: str,
    *,
    reason: str | None = None,
    child_still_running: bool | None = None,
) -> FrameT:
    event: FrameT = {"kind": "interrupted", "runId": run_id, "error": error}
    if reason is not None:
        event["reason"] = reason
    if child_still_running is not None:
        event["childStillRunning"] = child_still_running
    return event
