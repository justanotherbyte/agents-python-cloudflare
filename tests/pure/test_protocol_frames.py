"""Frame builders in the canonical protocol modules.

The client branches on key *presence*, so these lock which keys are omitted as
tightly as which are set — an extra or missing key changes client behaviour with
no error anywhere (see AGENTS.md, wire invariants).
"""

from __future__ import annotations

from agents.chat.protocol import (
    chat_response,
    stream_pending_frame,
    stream_resume_none_frame,
    stream_resuming_frame,
)
from agents.core.agent_tool_protocol import (
    agent_tool_aborted_event,
    agent_tool_event_frame,
    agent_tool_interrupted_event,
    agent_tool_started_event,
)
from agents.core.protocol import (
    rpc_chunk,
    rpc_error,
    rpc_result,
)


def test_rpc_chunk_marks_stream_with_done_false():
    frame = rpc_chunk("id-1", {"n": 1})
    # done present and false is what tells the client this is a stream, not the answer.
    assert frame["done"] is False
    assert frame["success"] is True
    assert frame["result"] == {"n": 1}


def test_rpc_result_omits_result_when_missing():
    frame = rpc_result("id-1")
    assert "result" not in frame
    assert frame["done"] is True
    assert frame["success"] is True


def test_rpc_result_includes_explicit_none():
    # An omitted result and an explicit null are different answers to the client.
    frame = rpc_result("id-1", None)
    assert "result" in frame
    assert frame["result"] is None
    assert frame["done"] is True


def test_rpc_error_carries_neither_done_nor_result():
    frame = rpc_error("id-1", "boom")
    assert "done" not in frame
    assert "result" not in frame
    assert frame["success"] is False
    assert frame["error"] == "boom"


def test_chat_response_omits_replay_flags_by_default():
    frame = chat_response("t", "id-1", "body", done=True)
    assert "replay" not in frame
    assert "replayComplete" not in frame
    assert frame["done"] is True
    assert frame["body"] == "body"


def test_chat_response_sets_replay_flags_only_as_true():
    frame = chat_response("t", "id-1", done=True, replay=True, replay_complete=True)
    assert frame["replay"] is True
    assert frame["replayComplete"] is True


def test_stream_resume_none_always_carries_idle_reason():
    # The client clears its spinner on reason "idle"; omitting it hangs the mount.
    frame = stream_resume_none_frame()
    assert frame["reason"] == "idle"
    assert "probeId" not in frame


def test_stream_resume_none_echoes_probe_id_when_given():
    frame = stream_resume_none_frame("probe-9")
    assert frame["reason"] == "idle"
    assert frame["probeId"] == "probe-9"


def test_stream_resuming_omits_probe_id_when_none():
    frame = stream_resuming_frame("req-1")
    assert frame["id"] == "req-1"
    assert "probeId" not in frame


def test_stream_resuming_echoes_probe_id_when_given():
    frame = stream_resuming_frame("req-1", "probe-9")
    assert frame["probeId"] == "probe-9"


def test_stream_pending_includes_only_supplied_identifiers():
    assert stream_pending_frame() == {"type": "cf_agent_stream_pending"}
    assert stream_pending_frame("req-1", "probe-9") == {
        "type": "cf_agent_stream_pending",
        "id": "req-1",
        "probeId": "probe-9",
    }


def test_agent_tool_event_frame_omits_optionals_by_default():
    frame = agent_tool_event_frame({"kind": "chunk"}, 3)
    assert frame["sequence"] == 3
    assert frame["event"] == {"kind": "chunk"}
    assert "parentToolCallId" not in frame
    assert "replay" not in frame


def test_agent_tool_event_frame_includes_optionals_when_supplied():
    frame = agent_tool_event_frame(
        {"kind": "chunk"}, 3, parent_tool_call_id="call-1", replay=True
    )
    assert frame["parentToolCallId"] == "call-1"
    assert frame["replay"] is True


def test_agent_tool_started_event_omits_input_preview_and_display():
    event = agent_tool_started_event("run-1", "worker", 0)
    assert "inputPreview" not in event
    assert "display" not in event


def test_agent_tool_started_event_includes_optionals_when_supplied():
    event = agent_tool_started_event(
        "run-1", "worker", 0, input_preview="peek", display={"label": "x"}
    )
    assert event["inputPreview"] == "peek"
    assert event["display"] == {"label": "x"}


def test_agent_tool_started_event_includes_explicit_none_preview():
    # MISSING is the omit sentinel, so an explicit None is still a present key.
    event = agent_tool_started_event("run-1", "worker", 0, input_preview=None)
    assert "inputPreview" in event
    assert event["inputPreview"] is None


def test_agent_tool_aborted_event_omits_reason_by_default():
    event = agent_tool_aborted_event("run-1")
    assert "reason" not in event


def test_agent_tool_aborted_event_includes_reason_when_given():
    event = agent_tool_aborted_event("run-1", "cancelled")
    assert event["reason"] == "cancelled"


def test_agent_tool_interrupted_event_omits_optionals_by_default():
    event = agent_tool_interrupted_event("run-1", "oops")
    assert event["error"] == "oops"
    assert "reason" not in event
    assert "childStillRunning" not in event


def test_agent_tool_interrupted_event_includes_optionals_when_given():
    event = agent_tool_interrupted_event(
        "run-1", "oops", reason="evicted", child_still_running=True
    )
    assert event["reason"] == "evicted"
    assert event["childStillRunning"] is True
