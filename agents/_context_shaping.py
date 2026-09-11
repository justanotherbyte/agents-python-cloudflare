from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from ._session_json import dumps_session_json


_MAX_DEPTH = 8
_TRUNCATED_FLAG = "__truncated"
_TRUNCATED_CHARS = "__truncatedChars"


def _shape_messages(
    messages: Sequence[dict[str, object]],
    *,
    keep_recent: int = 4,
    max_tool_output_chars: int = 500,
    max_text_chars: int = 10_000,
) -> list[dict[str, object]]:
    if len(messages) <= keep_recent:
        return list(messages)

    cutoff = len(messages) - keep_recent
    shaped = []
    for index, message in enumerate(messages):
        if index >= cutoff:
            shaped.append(message)
            continue
        updated = _shape_message(
            message,
            max_tool_output_chars=max_tool_output_chars,
            max_text_chars=max_text_chars,
        )
        shaped.append(message if updated is None else updated)
    return shaped


def _shape_message(
    message: dict[str, object],
    *,
    max_tool_output_chars: int,
    max_text_chars: int,
) -> dict[str, object] | None:
    parts = message.get("parts")
    if not isinstance(parts, list):
        return None
    changed = False
    shaped_parts = []
    for part in parts:
        if not isinstance(part, dict):
            shaped_parts.append(part)
            continue
        shaped = _shape_part(part, max_tool_output_chars, max_text_chars)
        if shaped is not part:
            changed = True
        shaped_parts.append(shaped)
    if not changed:
        return None
    return {**message, "parts": shaped_parts}


def _shape_part(
    part: dict[str, object],
    max_tool_output_chars: int,
    max_text_chars: int,
) -> dict[str, object]:
    part_type = part.get("type")
    if isinstance(part_type, str) and (
        part_type.startswith("tool-") or part_type == "dynamic-tool"
    ):
        if "output" in part:
            output, truncated = _truncate_tool_output(
                part["output"],
                max_tool_output_chars,
            )
            if truncated:
                return {**part, "output": output}

    if part_type == "text" and "text" in part:
        text = part["text"]
        if isinstance(text, str) and _utf16_length(text) > max_text_chars:
            return {
                **part,
                "text": (
                    f"{_slice_utf16(text, max_text_chars)}"
                    f"... [truncated {_utf16_length(text)} chars]"
                ),
            }
    return part


def _truncate_tool_output(value: object, max_chars: int) -> tuple[object, bool]:
    original_length = _value_length(value)
    if original_length <= max_chars:
        return value, False
    return _truncate_value(value, max_chars, original_length, 0), True


def _truncate_value(
    value: object,
    max_chars: int,
    original_length: int,
    depth: int,
) -> object:
    if isinstance(value, str):
        return _truncate_string(value, max_chars, _utf16_length(value))
    if value is None or not isinstance(value, (Mapping, Sequence)):
        return value
    if isinstance(value, (bytes, bytearray)):
        return value
    if depth >= _MAX_DEPTH:
        return f"[Nested output omitted {_truncated_suffix(original_length)}]"
    if isinstance(value, Sequence) and not isinstance(value, str):
        return _truncate_array(value, max_chars, original_length, depth)
    return _truncate_object(value, max_chars, original_length, depth)


def _truncate_array(
    value: Sequence[object],
    max_chars: int,
    original_length: int,
    depth: int,
) -> list[object]:
    child_budget = _child_max_chars(max_chars, len(value))
    result = [
        _truncate_value(item, child_budget, _value_length(item), depth + 1)
        for item in value
    ]
    if len(result) > 1 and _value_length(result) > max_chars:
        result = result[: _retained_array_items(result, max_chars)]
    if len(result) < len(value):
        result.append(f"Array output truncated {_truncated_suffix(original_length)}")
    if _value_length(result) > max_chars:
        return _compact_array_marker(max_chars, original_length)
    return result


def _retained_array_items(value: list[object], max_chars: int) -> int:
    low = 1
    high = len(value) - 1
    retained = 1
    while low <= high:
        middle = (low + high) // 2
        if _value_length(value[:middle]) <= max_chars:
            retained = middle
            low = middle + 1
        else:
            high = middle - 1
    return retained


def _truncate_object(
    value: Mapping[object, object],
    max_chars: int,
    original_length: int,
    depth: int,
) -> dict[str, object]:
    entries = [(str(key), item) for key, item in value.items()]
    child_budget = _child_max_chars(max_chars, len(entries))
    result = {
        key: _truncate_value(item, child_budget, _value_length(item), depth + 1)
        for key, item in entries
    }
    result_length = _value_length(result)
    if result_length <= max_chars:
        return result
    _shrink_string_fields(result, max_chars)
    if _value_length(result) > max_chars:
        result[_TRUNCATED_FLAG] = True
        result[_TRUNCATED_CHARS] = result_length
    if _value_length(result) > max_chars:
        return _compact_object_marker(max_chars, original_length)
    return result


def _shrink_string_fields(value: dict[str, object], max_chars: int) -> None:
    strings = sorted(
        ((key, item) for key, item in value.items() if isinstance(item, str)),
        key=lambda entry: _utf16_length(entry[1]),
        reverse=True,
    )
    current_length = _value_length(value)
    for key, text in strings:
        if current_length <= max_chars:
            return
        replacement = _truncate_string(
            text,
            max(0, math.floor(max_chars / 4)),
            _utf16_length(text),
        )
        value[key] = replacement
        current_length += _json_string_length(replacement) - _json_string_length(text)


def _truncate_string(value: str, max_chars: int, original_length: int) -> str:
    if _utf16_length(value) <= max_chars:
        return value
    suffix = _truncated_suffix(original_length)
    if max_chars <= _utf16_length(suffix):
        return _slice_utf16(suffix, max_chars)
    return f"{_slice_utf16(value, max_chars - _utf16_length(suffix))}{suffix}"


def _compact_object_marker(max_chars: int, original_length: int) -> dict[str, object]:
    marker: dict[str, object] = {
        _TRUNCATED_FLAG: True,
        _TRUNCATED_CHARS: original_length,
        "note": (
            "Tool output omitted because it was too large to preserve structurally."
        ),
    }
    if _value_length(marker) <= max_chars:
        return marker
    return {_TRUNCATED_FLAG: True, _TRUNCATED_CHARS: original_length}


def _compact_array_marker(max_chars: int, original_length: int) -> list[object]:
    structured = _compact_object_marker(max_chars, original_length)
    if _value_length([structured]) <= max_chars:
        return [structured]
    marker = [
        "Array output omitted because it was too large to preserve structurally "
        f"{_truncated_suffix(original_length)}"
    ]
    if _value_length(marker) <= max_chars:
        return marker
    return [_slice_utf16(_truncated_suffix(original_length), max_chars)]


def _child_max_chars(max_chars: int, child_count: int) -> int:
    if child_count <= 1:
        return max_chars
    return max(80, math.floor(max_chars / min(child_count, 10)))


def _truncated_suffix(original_length: int) -> str:
    return f"... [truncated {original_length} chars]"


def _value_length(value: object) -> int:
    if isinstance(value, str):
        return _utf16_length(value)
    try:
        rendered = dumps_session_json(value)
    except (TypeError, ValueError):
        rendered = _js_string(value)
    return _utf16_length(rendered)


def _json_string_length(value: str) -> int:
    return _utf16_length(dumps_session_json(value))


def _js_string(value: object) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, (list, tuple)):
        return ",".join(_js_string(item) for item in value)
    if isinstance(value, Mapping):
        return "[object Object]"
    return str(value)


def _utf16_length(value: str) -> int:
    return len(value.encode("utf-16-le", errors="surrogatepass")) // 2


def _slice_utf16(value: str, units: int) -> str:
    if units <= 0:
        return ""
    encoded = value.encode("utf-16-le", errors="surrogatepass")
    return encoded[: units * 2].decode("utf-16-le", errors="surrogatepass")
