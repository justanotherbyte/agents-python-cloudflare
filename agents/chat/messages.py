from __future__ import annotations

import copy
import math
from datetime import UTC, datetime
from typing import Any

from ..core.utils import dumps_wire
from ..sessions import _sanitize_parts
from .folding import _normalize_tool_input
from .types import ChunkT, MessageT


def _transform_message(value: Any, index: int = 0) -> MessageT | None:
    if isinstance(value, dict) and isinstance(value.get("parts"), list):
        message = copy.deepcopy(value)
        parts = message["parts"]
        if not _valid_message(message):
            return None
        for part in parts:
            part_type = part.get("type")
            if isinstance(part_type, str) and (
                part_type.startswith("tool-") or part_type == "dynamic-tool"
            ):
                part["input"] = _normalize_tool_input(part.get("input"))
        return message

    source = value if isinstance(value, dict) else {}
    parts: list[ChunkT] = []
    reasoning = source.get("reasoning")
    if reasoning:
        parts.append({"type": "reasoning", "text": reasoning})

    invocations = source.get("toolInvocations")
    if isinstance(invocations, list):
        state_map = {
            "partial-call": "input-streaming",
            "call": "input-available",
            "result": "output-available",
            "error": "output-error",
        }
        for invocation in invocations:
            if not isinstance(invocation, dict) or "toolName" not in invocation:
                continue
            parts.append(
                {
                    "type": f"tool-{invocation['toolName']}",
                    "toolCallId": invocation.get("toolCallId"),
                    "state": state_map.get(invocation.get("state"), "input-available"),
                    "input": _normalize_tool_input(invocation.get("args")),
                    "output": invocation.get("result"),
                }
            )

    legacy_parts = source.get("parts")
    if isinstance(legacy_parts, list):
        for part in legacy_parts:
            if not isinstance(part, dict) or part.get("type") != "file":
                continue
            url = part.get("url")
            if not url and part.get("data") is not None:
                media_type = part.get("mimeType") or part.get("mediaType")
                url = f"data:{media_type};base64,{part['data']}"
            parts.append(
                {
                    "type": "file",
                    "url": url,
                    "mediaType": part.get("mediaType") or part.get("mimeType"),
                    "filename": part.get("filename"),
                }
            )

    content = source.get("content")
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(
                    {
                        "type": item.get("type") or "text",
                        "text": item.get("text") or "",
                    }
                )
    elif not parts and "content" in source:
        text = content if isinstance(content, str) else dumps_wire(content)
        parts.append({"type": "text", "text": text})

    if not parts:
        text = value if isinstance(value, str) else dumps_wire(value)
        parts.append({"type": "text", "text": text})

    role = source.get("role")
    message = {
        "id": source.get("id") or f"msg-{index}",
        "role": "system" if role == "data" else role or "user",
        "parts": parts,
    }
    return message if _valid_message(message) else None


def _valid_message(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and isinstance(value.get("id"), str)
        and bool(value["id"])
        and value.get("role") in ("user", "assistant", "system")
        and isinstance(value.get("parts"), list)
        and all(
            isinstance(part, dict) and isinstance(part.get("type"), str)
            for part in value["parts"]
        )
    )


def _sanitize_message(message: MessageT) -> MessageT:
    return {**message, "parts": _sanitize_parts(message["parts"])}


def _assistant_content_key(message: MessageT) -> str | None:
    if message["role"] != "assistant" or any(
        "toolCallId" in part for part in message["parts"]
    ):
        return None
    return dumps_wire(message["parts"])


def _legacy_created_at_ms(value: Any, fallback: int) -> int:
    if type(value) in (int, float) and math.isfinite(value) and value >= 0:
        return int(value)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            timestamp = int(parsed.timestamp() * 1000)
            if timestamp >= 0:
                return timestamp
        except ValueError:
            pass
    return fallback
