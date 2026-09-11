"""Chunk folding in `agents.chat`.

`apply_chunk_to_parts` mutates a parts list in place and returns whether it
recognised the chunk. `_normalize` is an async generator that turns a string
reply into a lazily opened text block. The wire invariants under test include
exact part shapes, static and dynamic tools, settled-state replay suppression,
and closing a text block before a following raw dict chunk.
"""

from __future__ import annotations

import pytest

from agents import apply_chunk_to_parts
from agents.chat.folding import (
    MessageAccumulator,
    _find_tool_part,
    _normalize_tool_input,
)
from agents.chat.normalize import _normalize


def test_tool_input_available_appends_typed_part_without_dynamic():
    parts: list[dict] = []
    handled = apply_chunk_to_parts(
        parts,
        {
            "type": "tool-input-available",
            "toolCallId": "c1",
            "toolName": "search",
            "input": {"q": 1},
        },
    )
    assert handled is True
    part = parts[-1]
    assert part["type"] == "tool-search"
    assert part["state"] == "input-available"
    assert part["input"] == {"q": 1}
    # The client's reducer branches on the "dynamic" flag, so leaving it unset is
    # what keeps a reloaded part rendering as tool-{name}.
    assert "dynamic" not in part


def test_second_tool_input_available_does_not_overwrite_input():
    parts: list[dict] = []
    apply_chunk_to_parts(
        parts,
        {
            "type": "tool-input-available",
            "toolCallId": "c1",
            "toolName": "search",
            "input": {"q": 1},
        },
    )
    # The part is now "input-available", not "input-streaming", so a replayed call
    # under the same id is a no-op on the input — first write wins.
    handled = apply_chunk_to_parts(
        parts,
        {
            "type": "tool-input-available",
            "toolCallId": "c1",
            "toolName": "search",
            "input": {"q": 999},
        },
    )
    assert handled is True
    assert _find_tool_part(parts, "c1")["input"] == {"q": 1}


def test_tool_output_available_sets_state_and_output():
    parts: list[dict] = []
    apply_chunk_to_parts(
        parts,
        {
            "type": "tool-input-available",
            "toolCallId": "c1",
            "toolName": "search",
            "input": {},
        },
    )
    handled = apply_chunk_to_parts(
        parts,
        {"type": "tool-output-available", "toolCallId": "c1", "output": "OUT"},
    )
    assert handled is True
    part = _find_tool_part(parts, "c1")
    assert part["state"] == "output-available"
    assert part["output"] == "OUT"


def test_settled_part_is_not_reopened_by_a_denied_chunk():
    # The settled-state guard lives on the approval/denied folds: once a part has
    # resolved, a tool-output-denied replay leaves it untouched.
    parts: list[dict] = []
    apply_chunk_to_parts(
        parts,
        {
            "type": "tool-input-available",
            "toolCallId": "c1",
            "toolName": "search",
            "input": {},
        },
    )
    apply_chunk_to_parts(
        parts,
        {"type": "tool-output-available", "toolCallId": "c1", "output": "OUT"},
    )
    handled = apply_chunk_to_parts(
        parts, {"type": "tool-output-denied", "toolCallId": "c1"}
    )
    assert handled is True
    part = _find_tool_part(parts, "c1")
    assert part["state"] == "output-available"
    assert part["output"] == "OUT"


def test_approval_request_folds_onto_an_unsettled_part():
    parts: list[dict] = []
    apply_chunk_to_parts(
        parts,
        {
            "type": "tool-input-available",
            "toolCallId": "c1",
            "toolName": "search",
            "input": {},
        },
    )
    handled = apply_chunk_to_parts(
        parts,
        {
            "type": "tool-approval-request",
            "toolCallId": "c1",
            "approvalId": "a1",
            "approvalDescriptor": {"why": "risky"},
        },
    )
    assert handled is True
    part = _find_tool_part(parts, "c1")
    assert part["state"] == "approval-requested"
    assert part["approval"] == {"id": "a1", "descriptor": {"why": "risky"}}


def test_output_denied_folds_onto_an_unsettled_part():
    parts: list[dict] = []
    apply_chunk_to_parts(
        parts,
        {
            "type": "tool-input-available",
            "toolCallId": "c1",
            "toolName": "search",
            "input": {},
        },
    )
    handled = apply_chunk_to_parts(
        parts, {"type": "tool-output-denied", "toolCallId": "c1"}
    )
    assert handled is True
    part = _find_tool_part(parts, "c1")
    assert part["state"] == "output-denied"
    # Only the state moves: a denial carries no output to record.
    assert "output" not in part


@pytest.mark.parametrize("chunk_type", ["tool-approval-request", "tool-output-denied"])
def test_approval_and_denied_chunks_naming_no_part_are_swallowed(chunk_type):
    # Neither fold ever creates a part, so a chunk for a call the tool-input chunk
    # never introduced is recognised and dropped.
    parts: list[dict] = []
    assert apply_chunk_to_parts(parts, {"type": chunk_type, "toolCallId": "x"}) is True
    assert parts == []


def test_normalize_tool_input_degrades_non_objects_to_empty():
    assert _normalize_tool_input({"a": 1}) == {"a": 1}
    assert _normalize_tool_input('{"a":1}') == {"a": 1}
    assert _normalize_tool_input("nope") == {}
    assert _normalize_tool_input(5) == {}


def test_text_chunks_fold_into_a_text_part():
    parts: list[dict] = []
    assert apply_chunk_to_parts(parts, {"type": "text-start", "id": "0"}) is True
    assert (
        apply_chunk_to_parts(parts, {"type": "text-delta", "id": "0", "delta": "hi"})
        is True
    )
    assert apply_chunk_to_parts(parts, {"type": "text-end", "id": "0"}) is True
    assert parts == [{"type": "text", "text": "hi", "state": "done"}]


def test_unknown_block_family_returns_false():
    # A -start whose family is not in the allowed set is not folded.
    assert apply_chunk_to_parts([], {"type": "foo-start"}) is False


@pytest.mark.asyncio
async def test_normalize_wraps_a_string_in_a_text_block():
    chunks = [chunk async for chunk in _normalize("hi")]
    assert chunks == [
        {"type": "text-start", "id": "0"},
        {"type": "text-delta", "id": "0", "delta": "hi"},
        {"type": "text-end", "id": "0"},
    ]


@pytest.mark.asyncio
async def test_normalize_closes_text_block_before_a_raw_dict():
    chunks = [chunk async for chunk in _normalize(["hi", {"type": "data-x"}])]
    # The text-end must land before the raw dict so the client sees a closed block.
    assert chunks == [
        {"type": "text-start", "id": "0"},
        {"type": "text-delta", "id": "0", "delta": "hi"},
        {"type": "text-end", "id": "0"},
        {"type": "data-x"},
    ]


@pytest.mark.asyncio
async def test_normalize_accepts_an_async_iterable():
    async def reply():
        yield "hi"

    chunks = [chunk async for chunk in _normalize(reply())]

    assert [chunk["type"] for chunk in chunks] == [
        "text-start",
        "text-delta",
        "text-end",
    ]


def test_reasoning_chunks_merge_provider_metadata():
    parts: list[dict] = []
    apply_chunk_to_parts(parts, {"type": "reasoning-start"})
    apply_chunk_to_parts(
        parts,
        {
            "type": "reasoning-delta",
            "delta": "because",
            "providerMetadata": {"anthropic": {"redacted": True}},
        },
    )
    apply_chunk_to_parts(
        parts,
        {
            "type": "reasoning-end",
            "providerMetadata": {"signature": "signed"},
        },
    )

    assert parts == [
        {
            "type": "reasoning",
            "text": "because",
            "state": "done",
            "providerMetadata": {
                "anthropic": {"redacted": True},
                "signature": "signed",
            },
        }
    ]


@pytest.mark.parametrize(
    ("chunk", "expected"),
    [
        (
            {
                "type": "file",
                "mediaType": "image/png",
                "url": "data:image",
                "filename": "image.png",
                "providerMetadata": {"source": "model"},
            },
            {
                "type": "file",
                "mediaType": "image/png",
                "url": "data:image",
                "filename": "image.png",
                "providerMetadata": {"source": "model"},
            },
        ),
        (
            {"type": "source-url", "sourceId": "s1", "url": "https://x"},
            {"type": "source-url", "sourceId": "s1", "url": "https://x"},
        ),
        (
            {
                "type": "source-document",
                "sourceId": "s2",
                "mediaType": "text/plain",
                "title": "Doc",
            },
            {
                "type": "source-document",
                "sourceId": "s2",
                "mediaType": "text/plain",
                "title": "Doc",
            },
        ),
    ],
)
def test_media_and_sources_have_exact_persisted_shapes(chunk, expected):
    parts: list[dict] = []
    assert apply_chunk_to_parts(parts, chunk) is True
    assert parts == [expected]


def test_dynamic_tool_stream_builds_input_and_metadata():
    parts: list[dict] = []
    apply_chunk_to_parts(
        parts,
        {
            "type": "tool-input-start",
            "toolCallId": "c1",
            "toolName": "search",
            "dynamic": True,
            "providerExecuted": True,
            "providerMetadata": {"p": 1},
            "title": "Search",
        },
    )
    apply_chunk_to_parts(
        parts,
        {"type": "tool-input-delta", "toolCallId": "c1", "inputTextDelta": '{"q"'},
    )
    apply_chunk_to_parts(
        parts,
        {"type": "tool-input-delta", "toolCallId": "c1", "inputTextDelta": ":1}"},
    )
    apply_chunk_to_parts(
        parts,
        {
            "type": "tool-input-available",
            "toolCallId": "c1",
            "toolName": "search",
            "dynamic": True,
            "input": '{"q":1}',
            "providerExecuted": True,
            "providerMetadata": {"p": 2},
            "title": "Search now",
        },
    )

    assert parts == [
        {
            "type": "dynamic-tool",
            "toolCallId": "c1",
            "toolName": "search",
            "state": "input-available",
            "input": {"q": 1},
            "providerExecuted": True,
            "callProviderMetadata": {"p": 2},
            "title": "Search now",
        }
    ]


def test_tool_input_error_can_create_a_complete_failed_part():
    parts: list[dict] = []
    apply_chunk_to_parts(
        parts,
        {
            "type": "tool-input-error",
            "toolCallId": "c1",
            "toolName": "search",
            "input": None,
            "errorText": "bad input",
        },
    )
    assert parts == [
        {
            "type": "tool-search",
            "toolCallId": "c1",
            "toolName": "search",
            "state": "output-error",
            "input": {},
            "errorText": "bad input",
        }
    ]


def test_tool_input_error_preserves_existing_dynamic_type():
    parts: list[dict] = []
    apply_chunk_to_parts(
        parts,
        {
            "type": "tool-input-start",
            "toolCallId": "c1",
            "toolName": "search",
            "dynamic": True,
        },
    )
    apply_chunk_to_parts(
        parts,
        {
            "type": "tool-input-error",
            "toolCallId": "c1",
            "toolName": "search",
            "errorText": "bad input",
        },
    )
    assert parts[0]["type"] == "dynamic-tool"


def test_final_tool_output_replaces_preliminary_but_not_another_final():
    parts: list[dict] = []
    apply_chunk_to_parts(
        parts,
        {
            "type": "tool-input-available",
            "toolCallId": "c1",
            "toolName": "search",
            "input": {},
        },
    )
    apply_chunk_to_parts(
        parts,
        {
            "type": "tool-output-available",
            "toolCallId": "c1",
            "output": "draft",
            "preliminary": True,
        },
    )
    apply_chunk_to_parts(
        parts,
        {
            "type": "tool-output-available",
            "toolCallId": "c1",
            "output": "final",
            "preliminary": False,
        },
    )
    apply_chunk_to_parts(
        parts,
        {
            "type": "tool-output-available",
            "toolCallId": "c1",
            "output": "replayed",
        },
    )
    assert parts[0]["output"] == "final"
    assert parts[0]["preliminary"] is False


def test_final_tool_output_without_preliminary_omits_the_old_flag():
    parts = [
        {
            "type": "tool-search",
            "toolCallId": "c1",
            "toolName": "search",
            "state": "output-available",
            "output": "draft",
            "preliminary": True,
        }
    ]
    apply_chunk_to_parts(
        parts,
        {"type": "tool-output-available", "toolCallId": "c1", "output": "final"},
    )
    assert parts[0]["output"] == "final"
    assert "preliminary" not in parts[0]


def test_data_parts_reconcile_and_transient_data_is_not_persisted():
    parts: list[dict] = []
    apply_chunk_to_parts(parts, {"type": "data-status", "id": "x", "data": 1})
    apply_chunk_to_parts(parts, {"type": "data-status", "id": "x", "data": 2})
    apply_chunk_to_parts(
        parts,
        {"type": "data-status", "id": "y", "data": 3, "transient": True},
    )
    assert parts == [{"type": "data-status", "id": "x", "data": 2}]


def test_step_aliases_fold_to_the_same_shape():
    parts: list[dict] = []
    apply_chunk_to_parts(parts, {"type": "start-step"})
    apply_chunk_to_parts(parts, {"type": "step-start"})
    assert parts == [{"type": "step-start"}, {"type": "step-start"}]


@pytest.mark.parametrize("chunk_type", ["finish", "finish-step"])
def test_control_chunks_are_recognized_without_creating_parts(chunk_type):
    parts: list[dict] = []

    assert apply_chunk_to_parts(parts, {"type": chunk_type}) is True
    assert parts == []


def test_accumulator_updates_message_metadata_and_reports_stream_error():
    message = {"id": "local", "role": "assistant", "parts": []}
    accumulator = MessageAccumulator(message["parts"], message)
    accumulator.apply(
        {
            "type": "start",
            "messageId": "server",
            "messageMetadata": {"model": "one"},
        }
    )
    accumulator.apply({"type": "message-metadata", "messageMetadata": {"usage": 2}})
    effect = accumulator.apply({"type": "error", "errorText": "provider failed"})

    assert message == {
        "id": "server",
        "role": "assistant",
        "parts": [],
        "metadata": {"model": "one", "usage": 2},
    }
    assert effect.terminal_error == "provider failed"


@pytest.mark.parametrize(
    "chunk_type",
    [
        "tool-input-start",
        "tool-input-delta",
        "tool-input-available",
        "tool-input-error",
        "tool-approval-request",
        "tool-output-available",
        "tool-output-error",
        "tool-output-denied",
    ],
)
def test_accumulator_suppresses_provider_replay_before_emission(chunk_type):
    parts = [
        {
            "type": "tool-search",
            "toolCallId": "c1",
            "toolName": "search",
            "state": "output-available",
            "output": "done",
        }
    ]
    accumulator = MessageAccumulator(parts)
    assert accumulator.should_suppress({"type": chunk_type, "toolCallId": "c1"}) is True


@pytest.mark.parametrize("chunk_type", ["tool-output-available", "tool-output-error"])
def test_accumulator_allows_final_output_to_replace_preliminary(chunk_type):
    parts = [
        {
            "type": "tool-search",
            "toolCallId": "c1",
            "toolName": "search",
            "state": "output-available",
            "output": "draft",
            "preliminary": True,
        }
    ]
    accumulator = MessageAccumulator(parts)
    assert (
        accumulator.should_suppress({"type": chunk_type, "toolCallId": "c1"}) is False
    )
