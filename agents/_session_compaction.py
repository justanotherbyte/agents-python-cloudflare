from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from ._session_json import dumps_session_json


_COMPACTION_PREFIX = "compaction_"
_PROTECT_HEAD = 3
_MIN_TAIL_MESSAGES = 2


@dataclass(frozen=True)
class _StoredCompaction:
    id: str
    summary: str
    from_message_id: str
    to_message_id: str
    created_at_ms: int


@dataclass(frozen=True)
class _OverlaySpan:
    start_index: int
    end_index: int
    compaction: _StoredCompaction


@dataclass(frozen=True)
class _CompactionInput:
    messages: list[Mapping[str, object]]
    previous_summary: str | None
    budget: int


def _plan_overlays(
    path_ids: Sequence[str],
    compactions: Iterable[_StoredCompaction],
) -> list[_OverlaySpan]:
    indexes = {message_id: index for index, message_id in enumerate(path_ids)}
    latest: dict[str, _StoredCompaction] = {}
    for compaction in compactions:
        start = indexes.get(compaction.from_message_id)
        end = indexes.get(compaction.to_message_id)
        if start is not None and end is not None and end >= start:
            latest[compaction.from_message_id] = compaction
    spans = []
    index = 0
    while index < len(path_ids):
        compaction = latest.get(path_ids[index])
        if compaction is not None:
            end_index = indexes[compaction.to_message_id]
            spans.append(_OverlaySpan(index, end_index, compaction))
            index = end_index + 1
        else:
            index += 1
    return spans


def _overlay_message(compaction: _StoredCompaction) -> dict[str, object]:
    return {
        "id": f"{_COMPACTION_PREFIX}{compaction.id}",
        "role": "assistant",
        "parts": [{"type": "text", "text": compaction.summary}],
        "createdAt": datetime.now(UTC),
    }


def _prepare_compaction_input(
    messages: Sequence[Mapping[str, object]],
    keep_recent_tokens: int | float,
    estimate_tokens: Callable[[Mapping[str, object]], int],
) -> _CompactionInput | None:
    if len(messages) <= _PROTECT_HEAD + _MIN_TAIL_MESSAGES:
        return None
    compress_start = _align_boundary_forward(messages, _PROTECT_HEAD)
    compress_end = _find_tail_cut(
        messages,
        compress_start,
        keep_recent_tokens,
        estimate_tokens,
    )
    if compress_end <= compress_start:
        return None
    middle = [
        message
        for message in messages[compress_start:compress_end]
        if not _is_compaction_message(message)
    ]
    if not middle:
        return None
    existing = next(
        (message for message in messages if _is_compaction_message(message)),
        None,
    )
    previous_summary = None if existing is None else _message_text(existing)
    tokens = sum(estimate_tokens(message) for message in middle)
    return _CompactionInput(
        middle, previous_summary, max(100, math.floor(tokens * 0.2))
    )


def _build_summary_prompt(compaction: _CompactionInput) -> str:
    content = "\n\n---\n\n".join(
        _format_message(message) for message in compaction.messages
    )
    if compaction.previous_summary:
        return (
            "You are updating a conversation summary. A previous summary exists "
            "below. New conversation turns have occurred since then and need to be "
            "incorporated.\n\n"
            f"PREVIOUS SUMMARY:\n{compaction.previous_summary}\n\n"
            f"NEW TURNS TO INCORPORATE:\n{content}\n\n"
            "Update the summary. PRESERVE existing information that is still "
            "relevant. ADD new information. Remove information only if it is "
            "clearly obsolete.\n\n"
            "## Topic\n[What the conversation is about]\n\n"
            "## Key Points\n[Important information, decisions, and conclusions "
            "from the conversation]\n\n"
            "## Current State\n[Where things stand now — what has been done, what is "
            "in progress]\n\n"
            "## Open Items\n[Unresolved questions, pending tasks, or next steps "
            "discussed]\n\n"
            f"Target ~{compaction.budget} tokens. Be factual — only include "
            "information that was explicitly discussed in the conversation. Do NOT "
            "invent file paths, commands, or details that were not mentioned. Write "
            "only the summary body."
        )
    return (
        "Create a concise summary of this conversation that preserves the important "
        "information for future context.\n\n"
        f"CONVERSATION TO SUMMARIZE:\n{content}\n\n"
        "Use this structure:\n\n"
        "## Topic\n[What the conversation is about]\n\n"
        "## Key Points\n[Important information, decisions, and conclusions from "
        "the conversation]\n\n"
        "## Current State\n[Where things stand now — what has been done, what is in "
        "progress]\n\n"
        "## Open Items\n[Unresolved questions, pending tasks, or next steps "
        "discussed]\n\n"
        f"Target ~{compaction.budget} tokens. Be factual — only include information "
        "that was explicitly discussed in the conversation. Do NOT invent file "
        "paths, commands, or details that were not mentioned. Write only the summary "
        "body."
    )


def _is_compaction_message(message: Mapping[str, object]) -> bool:
    message_id = message.get("id")
    return isinstance(message_id, str) and message_id.startswith(_COMPACTION_PREFIX)


def _align_boundary_forward(
    messages: Sequence[Mapping[str, object]],
    index: int,
) -> int:
    if index <= 0 or index >= len(messages):
        return index
    previous = messages[index - 1]
    if previous.get("role") != "assistant" or not _has_tool_calls(previous):
        return index
    call_ids = _tool_call_ids(previous)
    while index < len(messages) and _is_tool_result_for(messages[index], call_ids):
        index += 1
    return index


def _align_boundary_backward(
    messages: Sequence[Mapping[str, object]],
    index: int,
) -> int:
    if index <= 0 or index >= len(messages):
        return index
    for call_index in range(index - 1, -1, -1):
        candidate = messages[call_index]
        if candidate.get("role") != "assistant" or not _has_tool_calls(candidate):
            continue
        call_ids = _tool_call_ids(candidate)
        if all(
            _is_tool_result_for(messages[result_index], call_ids)
            for result_index in range(call_index + 1, index + 1)
        ):
            return call_index
        return index
    return index


def _find_tail_cut(
    messages: Sequence[Mapping[str, object]],
    head_end: int,
    token_budget: int | float,
    estimate_tokens: Callable[[Mapping[str, object]], int],
) -> int:
    accumulated = 0
    token_cut = len(messages)
    for index in range(len(messages) - 1, head_end - 1, -1):
        tokens = estimate_tokens(messages[index])
        if accumulated + tokens > token_budget and token_cut < len(messages):
            break
        accumulated += tokens
        token_cut = index
    minimum_cut = len(messages) - _MIN_TAIL_MESSAGES
    cut = min(token_cut, minimum_cut) if minimum_cut >= head_end else token_cut
    return _align_boundary_backward(messages, cut)


def _has_tool_calls(message: Mapping[str, object]) -> bool:
    return any(_is_tool_part(part) for part in _parts(message))


def _tool_call_ids(message: Mapping[str, object]) -> set[str]:
    return {
        call_id
        for part in _parts(message)
        if _is_tool_part(part) and isinstance((call_id := part.get("toolCallId")), str)
    }


def _is_tool_result_for(message: Mapping[str, object], call_ids: set[str]) -> bool:
    return any(
        _is_tool_part(part) and part.get("toolCallId") in call_ids
        for part in _parts(message)
    )


def _is_tool_part(part: Mapping[str, object]) -> bool:
    part_type = part.get("type")
    return isinstance(part_type, str) and (
        part_type.startswith("tool-") or part_type == "dynamic-tool"
    )


def _parts(message: Mapping[str, object]) -> list[Mapping[str, object]]:
    parts = message.get("parts")
    if not isinstance(parts, list):
        return []
    return [part for part in parts if isinstance(part, Mapping)]


def _message_text(message: Mapping[str, object]) -> str:
    texts = []
    for part in _parts(message):
        if part.get("type") != "text":
            continue
        text = part.get("text")
        texts.append("" if text is None else _js_string(text))
    return "\n".join(texts)


def _format_message(message: Mapping[str, object]) -> str:
    text = _message_text(message)
    tools = "\n".join(
        _format_tool(part) for part in _parts(message) if _is_tool_part(part)
    )
    suffix = f"\n{tools}" if tools else ""
    return f"[{message.get('role')}]\n{text}{suffix}"


def _format_tool(part: Mapping[str, object]) -> str:
    tool_name = part.get("toolName")
    lines = [f"[Tool: {'unknown' if tool_name is None else _js_string(tool_name)}]"]
    if _js_truthy(part.get("input")):
        lines.append(f"Input: {_slice_utf16(dumps_session_json(part['input']), 500)}")
    if _js_truthy(part.get("output")):
        lines.append(f"Output: {_slice_utf16(_js_string(part['output']), 500)}")
    return "\n".join(lines)


def _js_truthy(value: object) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, (int, float)) and value == 0:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    return not isinstance(value, str) or bool(value)


def _js_string(value: object) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return dumps_session_json(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return ",".join("" if item is None else _js_string(item) for item in value)
    return "[object Object]"


def _slice_utf16(value: str, units: int) -> str:
    encoded = value.encode("utf-16-le", errors="surrogatepass")
    return encoded[: units * 2].decode("utf-16-le", errors="surrogatepass")


__all__: list[str] = []
