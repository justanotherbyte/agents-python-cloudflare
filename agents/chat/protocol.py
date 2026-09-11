from enum import StrEnum
from typing import Any

from ..core.protocol import FrameT


class ChatMessageType(StrEnum):
    CHAT_MESSAGES = "cf_agent_chat_messages"
    USE_CHAT_REQUEST = "cf_agent_use_chat_request"
    USE_CHAT_RESPONSE = "cf_agent_use_chat_response"
    CHAT_CLEAR = "cf_agent_chat_clear"
    CHAT_REQUEST_CANCEL = "cf_agent_chat_request_cancel"
    STREAM_RESUMING = "cf_agent_stream_resuming"
    STREAM_RESUME_ACK = "cf_agent_stream_resume_ack"
    STREAM_RESUME_REQUEST = "cf_agent_stream_resume_request"
    STREAM_RESUME_NONE = "cf_agent_stream_resume_none"
    STREAM_PENDING = "cf_agent_stream_pending"
    TOOL_RESULT = "cf_agent_tool_result"
    TOOL_APPROVAL = "cf_agent_tool_approval"
    MESSAGE_UPDATED = "cf_agent_message_updated"
    CHAT_RECOVERING = "cf_agent_chat_recovering"


def chat_response(
    response_type: str,
    request_id: str,
    body: str = "",
    *,
    done: bool,
    replay: bool = False,
    replay_complete: bool = False,
) -> FrameT:
    # replay and replayComplete are omitted rather than sent false, matching what a live
    # turn puts on the wire.
    frame: FrameT = {
        "type": response_type,
        "id": request_id,
        "body": body,
        "done": done,
    }
    if replay:
        frame["replay"] = True
    if replay_complete:
        frame["replayComplete"] = True
    return frame


def chat_messages_frame(messages: Any) -> FrameT:
    return {"type": ChatMessageType.CHAT_MESSAGES, "messages": messages}


def chat_clear_frame() -> FrameT:
    return {"type": ChatMessageType.CHAT_CLEAR}


def message_updated_frame(message: Any) -> FrameT:
    # Carries the whole message, not a patch, and nothing else: the client looks the id
    # up in its own list and replaces it, and it will not append one it cannot find.
    return {"type": ChatMessageType.MESSAGE_UPDATED, "message": message}


def stream_resuming_frame(request_id: str, probe_id: Any = None) -> FrameT:
    # probeId is echoed only when the offer answers a probe, and the client keys its
    # pending probe on it; omitting it there leaves the hook waiting forever.
    frame: FrameT = {"type": ChatMessageType.STREAM_RESUMING, "id": request_id}
    if probe_id is not None:
        frame["probeId"] = probe_id
    return frame


def stream_resume_none_frame(probe_id: Any = None) -> FrameT:
    # "idle" tells the client no stream exists, so it clears its loading state.
    frame: FrameT = {"type": ChatMessageType.STREAM_RESUME_NONE, "reason": "idle"}
    if probe_id is not None:
        frame["probeId"] = probe_id
    return frame


def stream_pending_frame(request_id: str | None = None, probe_id: Any = None) -> FrameT:
    frame: FrameT = {"type": ChatMessageType.STREAM_PENDING}
    if request_id is not None:
        frame["id"] = request_id
    if probe_id is not None:
        frame["probeId"] = probe_id
    return frame
