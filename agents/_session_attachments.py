from __future__ import annotations

import base64
import binascii
import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass


_ATTACHMENT_PREFIX = "attachment:sha256:"
_MAX_WALK_DEPTH = 8
_BASE64 = re.compile(r"[A-Za-z0-9+/]*={0,2}")
_BASE64_WHITESPACE = re.compile(r"[\t\n\f\r ]")
_HASH = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class _PendingAttachment:
    hash: str
    media_type: str
    data: bytes


@dataclass(frozen=True)
class _StoredAttachment:
    media_type: str
    data: bytes


def _extract_attachments(
    message: dict[str, object],
) -> tuple[dict[str, object], tuple[_PendingAttachment, ...]]:
    attachments: list[_PendingAttachment] = []
    seen: set[str] = set()

    def walk(value: object, depth: int) -> tuple[object, bool]:
        if depth > _MAX_WALK_DEPTH:
            return value, False
        if isinstance(value, list):
            changed = False
            entries = []
            for entry in value:
                walked, entry_changed = walk(entry, depth + 1)
                changed = changed or entry_changed
                entries.append(walked)
            return (entries, True) if changed else (value, False)
        if not isinstance(value, dict):
            return value, False

        media = _inline_media(value)
        if media is not None:
            field, media_type, data = media
            digest = hashlib.sha256(data).hexdigest()
            if digest not in seen:
                seen.add(digest)
                attachments.append(_PendingAttachment(digest, media_type, data))
            replaced = dict(value)
            replaced["mediaType"] = media_type
            replaced[field] = f"{_ATTACHMENT_PREFIX}{digest}"
            return replaced, True

        changed = False
        replaced = {}
        for key, entry in value.items():
            walked, entry_changed = walk(entry, depth + 1)
            changed = changed or entry_changed
            replaced[key] = walked
        return (replaced, True) if changed else (value, False)

    parts, changed = walk(message["parts"], 0)
    if not changed:
        return message, tuple(attachments)
    stored = dict(message)
    stored["parts"] = parts
    return stored, tuple(attachments)


def _resolve_attachments(
    message: dict[str, object],
    load: Callable[[str], _StoredAttachment | None],
) -> dict[str, object]:
    def walk(value: object, depth: int) -> tuple[object, bool]:
        if depth > _MAX_WALK_DEPTH:
            return value, False
        if isinstance(value, list):
            changed = False
            entries = []
            for entry in value:
                walked, entry_changed = walk(entry, depth + 1)
                changed = changed or entry_changed
                entries.append(walked)
            return (entries, True) if changed else (value, False)
        if not isinstance(value, dict):
            return value, False

        for field in ("url", "data"):
            digest = _attachment_hash(value.get(field))
            if digest is None:
                continue
            attachment = load(digest)
            if attachment is None:
                return value, False
            replaced = dict(value)
            encoded = base64.b64encode(attachment.data).decode("ascii")
            replaced[field] = (
                f"data:{attachment.media_type};base64,{encoded}"
                if field == "url"
                else encoded
            )
            return replaced, True

        changed = False
        replaced = {}
        for key, entry in value.items():
            walked, entry_changed = walk(entry, depth + 1)
            changed = changed or entry_changed
            replaced[key] = walked
        return (replaced, True) if changed else (value, False)

    parts, changed = walk(message["parts"], 0)
    if not changed:
        return message
    resolved = dict(message)
    resolved["parts"] = parts
    return resolved


def _attachment_hashes(message: dict[str, object]) -> set[str]:
    hashes: set[str] = set()

    def walk(value: object, depth: int) -> None:
        if depth > _MAX_WALK_DEPTH:
            return
        if isinstance(value, list):
            for entry in value:
                walk(entry, depth + 1)
            return
        if not isinstance(value, dict):
            return
        for field in ("url", "data"):
            digest = _attachment_hash(value.get(field))
            if digest is not None:
                hashes.add(digest)
                return
        for entry in value.values():
            walk(entry, depth + 1)

    walk(message["parts"], 0)
    return hashes


def _inline_media(
    value: dict[object, object],
) -> tuple[str, str, bytes] | None:
    declared = value.get("mediaType")
    declared_type = declared if isinstance(declared, str) else None
    url = value.get("url")
    if isinstance(url, str):
        parsed = _parse_data_url(url)
        if parsed is None:
            return None
        inferred_type, payload = parsed
        media_type = declared_type if declared_type is not None else inferred_type
        if media_type.startswith("text/"):
            return None
        data = _decode_base64(payload)
        return None if data is None else ("url", media_type, data)
    data_value = value.get("data")
    if declared_type and isinstance(data_value, str):
        if declared_type.startswith("text/"):
            return None
        data = _decode_base64(data_value)
        return None if data is None else ("data", declared_type, data)
    return None


def _parse_data_url(value: str) -> tuple[str, str] | None:
    if not value.startswith("data:"):
        return None
    comma = value.find(",")
    if comma < 0:
        return None
    header = value[5:comma]
    if not header.endswith(";base64"):
        return None
    return header[: -len(";base64")] or "application/octet-stream", value[comma + 1 :]


def _decode_base64(value: str) -> bytes | None:
    compact = _BASE64_WHITESPACE.sub("", value)
    if len(compact) % 4 == 1 or _BASE64.fullmatch(compact) is None:
        return None
    if "=" in compact and len(compact) % 4 != 0:
        return None
    padded = compact + "=" * (-len(compact) % 4)
    try:
        return base64.b64decode(padded, validate=True)
    except (binascii.Error, ValueError):
        return None


def _attachment_hash(value: object) -> str | None:
    if not isinstance(value, str) or not value.startswith(_ATTACHMENT_PREFIX):
        return None
    digest = value[len(_ATTACHMENT_PREFIX) :]
    return digest if _HASH.fullmatch(digest) is not None else None
