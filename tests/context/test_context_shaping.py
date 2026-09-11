from __future__ import annotations

import copy

import pytest

from agents.context import ContextBlocks, ContextConfig


class StaticProvider:
    async def get(self) -> str:
        return "system guidance"


def _message(message_id: str, part: dict[str, object]):
    return {"id": message_id, "role": "user", "parts": [part]}


@pytest.mark.asyncio
async def test_assemble_shapes_only_old_history_without_mutating_storage_values():
    old_tool = _message(
        "old-tool",
        {
            "type": "tool-fetch",
            "toolCallId": "call",
            "output": {
                "title": "kept",
                "body": "x" * 2_000,
                "items": ["y" * 500 for _ in range(5)],
            },
        },
    )
    old_text = _message("old-text", {"type": "text", "text": "z" * 12_000})
    recent = [
        _message(f"recent-{index}", {"type": "text", "text": "r" * 12_000})
        for index in range(4)
    ]
    messages = [old_tool, old_text, *recent]
    original = copy.deepcopy(messages)
    blocks = ContextBlocks([ContextConfig(label="soul", provider=StaticProvider())])

    assembled = await blocks.assemble(messages)

    assert assembled["system"].endswith("system guidance")
    shaped_tool = assembled["messages"][0]["parts"][0]["output"]
    assert isinstance(shaped_tool, dict)
    assert shaped_tool != old_tool["parts"][0]["output"]
    assert assembled["messages"][1]["parts"][0]["text"] == (
        "z" * 10_000 + "... [truncated 12000 chars]"
    )
    assert assembled["messages"][-4:] == recent
    assert messages == original


@pytest.mark.asyncio
async def test_assemble_preserves_tool_result_fields_and_customizes_read_limits():
    messages = [
        _message(
            "result",
            {"type": "tool-run", "result": "x" * 100, "output": "y" * 100},
        ),
        _message("text", {"type": "text", "text": "abcdef"}),
        _message("recent", {"type": "text", "text": "unchanged"}),
    ]
    blocks = ContextBlocks([])

    assembled = await blocks.assemble(
        messages,
        keep_recent=1,
        max_tool_output_chars=30,
        max_text_chars=3,
    )

    tool = assembled["messages"][0]["parts"][0]
    assert tool["result"] == "x" * 100
    assert tool["output"] == "yyyyy... [truncated 100 chars]"
    assert assembled["messages"][1]["parts"][0]["text"] == (
        "abc... [truncated 6 chars]"
    )
    assert assembled["messages"][2] is messages[2]


@pytest.mark.asyncio
async def test_structured_truncation_keeps_array_shape_and_marks_deep_values():
    nested: object = "bottom" * 100
    for _ in range(10):
        nested = {"child": nested}
    messages = [
        _message("deep", {"type": "dynamic-tool", "output": [nested, nested]}),
        _message("recent", {"type": "text", "text": "safe"}),
    ]

    assembled = await ContextBlocks([]).assemble(
        messages,
        keep_recent=1,
        max_tool_output_chars=200,
    )

    output = assembled["messages"][0]["parts"][0]["output"]
    assert isinstance(output, list)
    assert output != messages[0]["parts"][0]["output"]


@pytest.mark.asyncio
async def test_text_truncation_counts_and_slices_javascript_utf16_units():
    text = "😀" * 5
    messages = [
        _message("old", {"type": "text", "text": text}),
        _message("recent", {"type": "text", "text": "safe"}),
    ]

    assembled = await ContextBlocks([]).assemble(
        messages,
        keep_recent=1,
        max_text_chars=3,
    )

    assert assembled["messages"][0]["parts"][0]["text"] == (
        "😀\ud83d... [truncated 10 chars]"
    )
    assert messages[0]["parts"][0]["text"] == text
