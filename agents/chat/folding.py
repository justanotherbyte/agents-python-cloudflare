from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

ChunkT = dict[str, Any]
MessageT = dict[str, Any]

TOOL_RESOLVED_STATES = ("output-available", "output-error", "output-denied")
_TOOL_SETTLED_STATES = TOOL_RESOLVED_STATES + ("approval-responded",)


@dataclass(frozen=True, slots=True)
class FoldEffect:
    handled: bool
    changed: bool
    persist_now: bool = False
    terminal_error: str | None = None


def _find_last_part(parts: list[ChunkT], part_type: str) -> ChunkT | None:
    for part in reversed(parts):
        if part.get("type") == part_type:
            return part
    return None


def _find_tool_part(parts: list[ChunkT], tool_call_id: str | None) -> ChunkT | None:
    if not tool_call_id:
        return None
    for part in reversed(parts):
        if part.get("toolCallId") == tool_call_id:
            return part
    return None


def _normalize_tool_input(raw: Any) -> Any:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip().startswith("{"):
        try:
            parsed = json.loads(raw)
        except ValueError:
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def _approval_record(chunk: ChunkT) -> ChunkT:
    record: ChunkT = {}
    if chunk.get("approvalId") is not None:
        record["id"] = chunk["approvalId"]
    if chunk.get("approvalDescriptor") is not None:
        record["descriptor"] = chunk["approvalDescriptor"]
    return record


def _merge_metadata(target: ChunkT, key: str, value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    current = target.get(key)
    merged = {**current, **value} if isinstance(current, dict) else dict(value)
    if current == merged:
        return False
    target[key] = merged
    return True


def _tool_type(chunk: ChunkT) -> str:
    if chunk.get("dynamic") is True:
        return "dynamic-tool"
    return f"tool-{chunk.get('toolName')}"


def _tool_part(chunk: ChunkT, state: str, *, include_input: bool) -> ChunkT:
    part: ChunkT = {
        "type": _tool_type(chunk),
        "toolCallId": chunk.get("toolCallId"),
        "toolName": chunk.get("toolName"),
        "state": state,
    }
    if include_input:
        part["input"] = _normalize_tool_input(chunk.get("input"))
    if chunk.get("providerExecuted") is not None:
        part["providerExecuted"] = chunk["providerExecuted"]
    if chunk.get("providerMetadata") is not None:
        part["callProviderMetadata"] = chunk["providerMetadata"]
    if chunk.get("title") is not None:
        part["title"] = chunk["title"]
    return part


class MessageAccumulator:
    def __init__(
        self,
        parts: list[ChunkT],
        message: MessageT | None = None,
    ) -> None:
        self.parts = parts
        self.message = message

    def should_suppress(self, chunk: ChunkT) -> bool:
        chunk_type = chunk.get("type")
        tool_call_id = chunk.get("toolCallId")
        if not isinstance(tool_call_id, str):
            return False
        part = _find_tool_part(self.parts, tool_call_id)
        if part is None:
            return False
        state = part.get("state")
        if chunk_type == "tool-input-start":
            return True
        if chunk_type in ("tool-input-delta", "tool-input-available"):
            return state != "input-streaming"
        if chunk_type == "tool-input-error":
            return state in _TOOL_SETTLED_STATES
        if chunk_type in ("tool-output-available", "tool-output-error"):
            return state in ("output-error", "output-denied") or (
                state == "output-available" and part.get("preliminary") is not True
            )
        if chunk_type in ("tool-output-denied", "tool-approval-request"):
            return state in _TOOL_SETTLED_STATES
        return False

    def apply(self, chunk: ChunkT) -> FoldEffect:
        chunk_type = chunk.get("type")
        if not isinstance(chunk_type, str):
            return FoldEffect(False, False)

        handler = _CHUNK_HANDLERS.get(chunk_type)
        if handler is not None:
            return handler(self, chunk)
        if chunk_type.startswith("data-"):
            return self._apply_data(chunk)
        return FoldEffect(False, False)

    def _apply_text_start(self, chunk: ChunkT) -> FoldEffect:
        name = chunk["type"].rpartition("-")[0]
        self.parts.append({"type": name, "text": "", "state": "streaming"})
        return FoldEffect(True, True)

    def _apply_text_delta(self, chunk: ChunkT) -> FoldEffect:
        name = chunk["type"].rpartition("-")[0]
        part = _find_last_part(self.parts, name)
        if part is None:
            part = {
                "type": name,
                "text": chunk.get("delta") or "",
                "state": "streaming",
            }
            self.parts.append(part)
        else:
            part["text"] = str(part.get("text") or "") + str(chunk.get("delta") or "")
        if name == "reasoning":
            _merge_metadata(part, "providerMetadata", chunk.get("providerMetadata"))
        return FoldEffect(True, True)

    def _apply_text_end(self, chunk: ChunkT) -> FoldEffect:
        name = chunk["type"].rpartition("-")[0]
        part = _find_last_part(self.parts, name)
        if part is None:
            return FoldEffect(True, False)
        changed = part.get("state") != "done"
        part["state"] = "done"
        if name == "reasoning":
            changed |= _merge_metadata(
                part, "providerMetadata", chunk.get("providerMetadata")
            )
        return FoldEffect(True, changed)

    def _apply_file(self, chunk: ChunkT) -> FoldEffect:
        part: ChunkT = {
            "type": "file",
            "mediaType": chunk.get("mediaType"),
            "url": chunk.get("url"),
        }
        for key in ("filename", "providerMetadata"):
            if chunk.get(key) is not None:
                part[key] = chunk[key]
        self.parts.append(part)
        return FoldEffect(True, True)

    def _apply_source(self, chunk: ChunkT) -> FoldEffect:
        chunk_type = chunk["type"]
        part: ChunkT = {"type": chunk_type, "sourceId": chunk.get("sourceId")}
        required = ("url",) if chunk_type == "source-url" else ("mediaType", "title")
        for key in required:
            part[key] = chunk.get(key)
        optional = (
            ("title", "providerMetadata")
            if chunk_type == "source-url"
            else ("filename", "providerMetadata")
        )
        for key in optional:
            if chunk.get(key) is not None:
                part[key] = chunk[key]
        self.parts.append(part)
        return FoldEffect(True, True)

    def _apply_tool_input_start(self, chunk: ChunkT) -> FoldEffect:
        if _find_tool_part(self.parts, chunk.get("toolCallId")) is not None:
            return FoldEffect(True, False)
        self.parts.append(_tool_part(chunk, "input-streaming", include_input=False))
        return FoldEffect(True, True)

    def _apply_tool_input_delta(self, chunk: ChunkT) -> FoldEffect:
        part = _find_tool_part(self.parts, chunk.get("toolCallId"))
        if part is None or part.get("state") != "input-streaming":
            return FoldEffect(True, False)
        if "inputTextDelta" in chunk:
            previous = part.get("input")
            prefix = previous if isinstance(previous, str) else ""
            part["input"] = prefix + str(chunk.get("inputTextDelta") or "")
        elif "input" in chunk:
            part["input"] = chunk["input"]
        return FoldEffect(True, True)

    def _apply_tool_input_available(self, chunk: ChunkT) -> FoldEffect:
        part = _find_tool_part(self.parts, chunk.get("toolCallId"))
        if part is None:
            self.parts.append(_tool_part(chunk, "input-available", include_input=True))
            return FoldEffect(True, True)
        if part.get("state") != "input-streaming":
            return FoldEffect(True, False)
        updated = _tool_part(chunk, "input-available", include_input=True)
        part.update(updated)
        return FoldEffect(True, True)

    def _apply_tool_input_error(self, chunk: ChunkT) -> FoldEffect:
        part = _find_tool_part(self.parts, chunk.get("toolCallId"))
        if part is not None and part.get("state") in TOOL_RESOLVED_STATES:
            return FoldEffect(True, False)
        if part is None:
            part = _tool_part(chunk, "output-error", include_input=True)
            self.parts.append(part)
        else:
            part["state"] = "output-error"
            part["input"] = _normalize_tool_input(chunk.get("input"))
            if chunk.get("providerExecuted") is not None:
                part["providerExecuted"] = chunk["providerExecuted"]
            if chunk.get("providerMetadata") is not None:
                part["callProviderMetadata"] = chunk["providerMetadata"]
        part["errorText"] = chunk.get("errorText")
        return FoldEffect(True, True)

    def _apply_tool_approval_request(self, chunk: ChunkT) -> FoldEffect:
        part = _find_tool_part(self.parts, chunk.get("toolCallId"))
        if part is None or part.get("state") in _TOOL_SETTLED_STATES:
            return FoldEffect(True, False)
        part["state"] = "approval-requested"
        part["approval"] = _approval_record(chunk)
        return FoldEffect(True, True, persist_now=True)

    def _apply_tool_output_denied(self, chunk: ChunkT) -> FoldEffect:
        part = _find_tool_part(self.parts, chunk.get("toolCallId"))
        if part is None or part.get("state") in _TOOL_SETTLED_STATES:
            return FoldEffect(True, False)
        part["state"] = "output-denied"
        return FoldEffect(True, True)

    def _tool_output_part(self, chunk: ChunkT) -> ChunkT | None:
        part = _find_tool_part(self.parts, chunk.get("toolCallId"))
        if part is None:
            return None
        state = part.get("state")
        if state in ("output-error", "output-denied"):
            return None
        if state == "output-available" and part.get("preliminary") is not True:
            return None
        return part

    def _apply_tool_output_available(self, chunk: ChunkT) -> FoldEffect:
        part = self._tool_output_part(chunk)
        if part is None:
            return FoldEffect(True, False)
        part["state"] = "output-available"
        part["output"] = chunk.get("output")
        part.pop("errorText", None)
        if "preliminary" in chunk:
            part["preliminary"] = chunk["preliminary"]
        elif part.get("preliminary") is True:
            part.pop("preliminary", None)
        return FoldEffect(True, True)

    def _apply_tool_output_error(self, chunk: ChunkT) -> FoldEffect:
        part = self._tool_output_part(chunk)
        if part is None:
            return FoldEffect(True, False)
        part["state"] = "output-error"
        part["errorText"] = chunk.get("errorText")
        part.pop("output", None)
        part.pop("preliminary", None)
        return FoldEffect(True, True)

    def _apply_step_start(self, _chunk: ChunkT) -> FoldEffect:
        self.parts.append({"type": "step-start"})
        return FoldEffect(True, True)

    def _apply_data(self, chunk: ChunkT) -> FoldEffect:
        chunk_type = chunk["type"]
        if chunk.get("transient") is True:
            return FoldEffect(True, False)
        data_id = chunk.get("id")
        if data_id is not None:
            for part in reversed(self.parts):
                if part.get("type") == chunk_type and part.get("id") == data_id:
                    changed = part.get("data") != chunk.get("data")
                    part["data"] = chunk.get("data")
                    return FoldEffect(True, changed)
        part = {"type": chunk_type, "data": chunk.get("data")}
        if data_id is not None:
            part["id"] = data_id
        self.parts.append(part)
        return FoldEffect(True, True)

    def _apply_message_chunk(self, chunk: ChunkT) -> FoldEffect:
        if self.message is None:
            return FoldEffect(True, False)
        chunk_type = chunk["type"]
        changed = False
        message_id = chunk.get("messageId")
        if chunk_type == "start" and isinstance(message_id, str):
            changed = self.message.get("id") != message_id
            self.message["id"] = message_id
        changed |= _merge_metadata(
            self.message, "metadata", chunk.get("messageMetadata")
        )
        return FoldEffect(True, changed)

    def _apply_finish_step(self, _chunk: ChunkT) -> FoldEffect:
        return FoldEffect(True, False)

    def _apply_error(self, chunk: ChunkT) -> FoldEffect:
        error = chunk.get("errorText")
        if not isinstance(error, str):
            error = json.dumps({"type": "error"}, separators=(",", ":"))
        return FoldEffect(True, False, terminal_error=error)


_ChunkHandler = Callable[[MessageAccumulator, ChunkT], FoldEffect]

_CHUNK_HANDLERS: Mapping[str, _ChunkHandler] = MappingProxyType(
    {
        "text-start": MessageAccumulator._apply_text_start,
        "text-delta": MessageAccumulator._apply_text_delta,
        "text-end": MessageAccumulator._apply_text_end,
        "reasoning-start": MessageAccumulator._apply_text_start,
        "reasoning-delta": MessageAccumulator._apply_text_delta,
        "reasoning-end": MessageAccumulator._apply_text_end,
        "file": MessageAccumulator._apply_file,
        "source-url": MessageAccumulator._apply_source,
        "source-document": MessageAccumulator._apply_source,
        "tool-input-start": MessageAccumulator._apply_tool_input_start,
        "tool-input-delta": MessageAccumulator._apply_tool_input_delta,
        "tool-input-available": MessageAccumulator._apply_tool_input_available,
        "tool-input-error": MessageAccumulator._apply_tool_input_error,
        "tool-approval-request": MessageAccumulator._apply_tool_approval_request,
        "tool-output-denied": MessageAccumulator._apply_tool_output_denied,
        "tool-output-available": MessageAccumulator._apply_tool_output_available,
        "tool-output-error": MessageAccumulator._apply_tool_output_error,
        "step-start": MessageAccumulator._apply_step_start,
        "start-step": MessageAccumulator._apply_step_start,
        "start": MessageAccumulator._apply_message_chunk,
        "finish": MessageAccumulator._apply_message_chunk,
        "message-metadata": MessageAccumulator._apply_message_chunk,
        "finish-step": MessageAccumulator._apply_finish_step,
        "error": MessageAccumulator._apply_error,
    }
)


def apply_chunk_to_parts(parts: list[ChunkT], chunk: ChunkT) -> bool:
    return MessageAccumulator(parts).apply(chunk).handled
