from __future__ import annotations

import io
import json
import math
from collections.abc import Mapping, Sequence
from typing import cast

from .core._wire import strict_json_loads

_MAX_EXACT_INTEGER = 2**53
_MAX_ARRAY_INDEX = 2**32 - 2


def dumps_session_json(value: object) -> str:
    """Serialize JSON with JavaScript-compatible strings and numbers."""
    return _encode_json(value, set())


def clone_session_json(value: object) -> object:
    return strict_json_loads(dumps_session_json(value), "Session JSON")


def parse_session_message(content: object) -> dict[str, object] | None:
    if not isinstance(content, str):
        return None
    try:
        message = strict_json_loads(content, "Session message")
    except ValueError:
        return None
    return session_message_from_value(message)


def session_message_from_value(value: object) -> dict[str, object] | None:
    if (
        not isinstance(value, dict)
        or type(value.get("id")) is not str
        or type(value.get("role")) is not str
        or not isinstance(value.get("parts"), list)
    ):
        return None
    for part in value["parts"]:
        if not isinstance(part, dict) or type(part.get("type")) is not str:
            return None
    return cast(dict[str, object], value)


def _encode_json(value: object, seen: set[int]) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str):
        return _encode_string(value)
    if isinstance(value, int):
        if -_MAX_EXACT_INTEGER <= value <= _MAX_EXACT_INTEGER:
            return str(value)
        try:
            return _encode_number(float(value))
        except OverflowError as error:
            raise ValueError("Session JSON integer exceeds JavaScript range") from error
    if isinstance(value, float):
        return _encode_number(value)
    if isinstance(value, Mapping):
        return _encode_mapping(value, seen)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return _encode_sequence(value, seen)
    raise TypeError(f"unsupported Session JSON value: {type(value).__name__}")


def _encode_mapping(value: Mapping[object, object], seen: set[int]) -> str:
    identity = id(value)
    if identity in seen:
        raise ValueError("Session JSON cannot contain cycles")
    seen.add(identity)
    try:
        fields = []
        for key, item in _ordered_items(value):
            encoded_key = _encode_string(key)
            fields.append(f"{encoded_key}:{_encode_json(item, seen)}")
        return "{" + ",".join(fields) + "}"
    finally:
        seen.remove(identity)


def _encode_sequence(value: Sequence[object], seen: set[int]) -> str:
    identity = id(value)
    if identity in seen:
        raise ValueError("Session JSON cannot contain cycles")
    seen.add(identity)
    try:
        return "[" + ",".join(_encode_json(item, seen) for item in value) + "]"
    finally:
        seen.remove(identity)


def _encode_number(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError("Session JSON numbers must be finite")
    if value == 0:
        return "0"
    source = repr(value).lower()
    if "e" not in source:
        return source.removesuffix(".0")
    mantissa, raw_exponent = source.split("e")
    exponent = int(raw_exponent)
    absolute = abs(value)
    if 1e-6 <= absolute < 1e21:
        unsigned = mantissa.removeprefix("-")
        sign = "-" if mantissa.startswith("-") else ""
        digits = unsigned.replace(".", "")
        point = unsigned.find(".")
        integer_digits = point if point >= 0 else len(digits)
        position = integer_digits + exponent
        if position <= 0:
            return sign + "0." + "0" * -position + digits
        if position >= len(digits):
            return sign + digits + "0" * (position - len(digits))
        return sign + digits[:position] + "." + digits[position:]
    normalized = mantissa.removesuffix(".0")
    exponent_sign = "+" if exponent >= 0 else ""
    return f"{normalized}e{exponent_sign}{exponent}"


def _encode_string(value: str) -> str:
    if not any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        return json.dumps(value, ensure_ascii=False)

    encoded = io.StringIO()
    encoded.write('"')
    index = 0
    run_start = 0
    while index < len(value):
        code = ord(value[index])
        if not 0xD800 <= code <= 0xDFFF:
            index += 1
            continue
        if run_start < index:
            encoded.write(json.dumps(value[run_start:index], ensure_ascii=False)[1:-1])
        if 0xD800 <= code <= 0xDBFF and index + 1 < len(value):
            low = ord(value[index + 1])
            if 0xDC00 <= low <= 0xDFFF:
                encoded.write(chr(0x10000 + ((code - 0xD800) << 10) + low - 0xDC00))
                index += 2
                run_start = index
                continue
        encoded.write(f"\\u{code:04x}")
        index += 1
        run_start = index
    if run_start < len(value):
        encoded.write(json.dumps(value[run_start:], ensure_ascii=False)[1:-1])
    encoded.write('"')
    return encoded.getvalue()


def _ordered_items(value: Mapping[object, object]) -> list[tuple[str, object]]:
    indexed: list[tuple[int, str, object]] = []
    ordinary: list[tuple[str, object]] = []
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError("Session JSON object keys must be strings")
        index = _array_index(key)
        if index is None:
            ordinary.append((key, item))
        else:
            indexed.append((index, key, item))
    indexed.sort(key=lambda entry: entry[0])
    return [(key, item) for _, key, item in indexed] + ordinary


def _array_index(key: str) -> int | None:
    if key == "0":
        return 0
    if not key or key[0] == "0" or not key.isascii() or not key.isdigit():
        return None
    value = int(key)
    return value if value <= _MAX_ARRAY_INDEX else None
