from __future__ import annotations

from typing import Any

CF_READONLY_KEY = "_cf_readonly"
CF_NO_PROTOCOL_KEY = "_cf_no_protocol"

_INTERNAL_KEYS = frozenset(
    {
        CF_READONLY_KEY,
        CF_NO_PROTOCOL_KEY,
        "_cf_voiceInCall",
        "_cf_subAgentOuterUrl",
        "_cf_subAgentTags",
    }
)

_FLAGGED_STATE_ERROR = (
    "connection state must be an object or null when internal flags are set"
)


def connection_state_flags(state: object) -> dict[str, Any]:
    if not isinstance(state, dict):
        return {}
    return {key: state[key] for key in state if key in _INTERNAL_KEYS}


def connection_user_state(state: object) -> object:
    if not isinstance(state, dict) or not any(key in state for key in _INTERNAL_KEYS):
        return state
    user_state = {
        key: value for key, value in state.items() if key not in _INTERNAL_KEYS
    }
    return user_state or None


def merge_connection_state(user_state: object, flags: dict[str, Any]) -> object:
    if not flags:
        return user_state
    if user_state is not None and not isinstance(user_state, dict):
        raise TypeError(_FLAGGED_STATE_ERROR)
    visible = user_state or {}
    return {**visible, **flags}


def set_connection_state_flag(
    state: object,
    key: str,
    value: object,
) -> object:
    if state is not None and not isinstance(state, dict):
        if value is None:
            return state
        raise TypeError(_FLAGGED_STATE_ERROR)
    raw = dict(state) if isinstance(state, dict) else {}
    if value is None:
        raw.pop(key, None)
    else:
        raw[key] = value
    return raw or None
