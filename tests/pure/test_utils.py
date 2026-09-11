"""Wire and parsing helpers in `agents.core.utils`.

These guard the invariants that fail silently on the wire: compact JSON with no
NaN/Infinity, `urlsplit` (never `urlparse`) so a `;` in an agent name survives,
and the scalar/array-degrades-to-None shape checks.
"""

from __future__ import annotations

import pytest

from agents.core.utils import (
    clamp,
    dumps_wire,
    error_message,
    loads_dict_or_none,
    loads_or_none,
    url_path,
    url_with_path,
)


def test_dumps_wire_uses_compact_separators():
    assert dumps_wire({"a": 1}) == '{"a":1}'
    assert dumps_wire([1, 2]) == "[1,2]"


def test_dumps_wire_rejects_nan():
    # JSON.parse on the other runtime rejects bare NaN, so it must never be written.
    with pytest.raises(ValueError):
        dumps_wire(float("nan"))


def test_dumps_wire_rejects_infinity():
    with pytest.raises(ValueError):
        dumps_wire(float("inf"))


def test_url_path_preserves_semicolon_in_last_segment():
    # urlparse would strip ";c/d" into a params field; urlsplit keeps it, so two
    # names differing only after a ";" stay distinct agents.
    assert url_path("https://x/a/b;c/d?q=1") == "/a/b;c/d"


def test_url_with_path_replaces_only_the_path():
    replaced = url_with_path("https://x/a?q=1", "/z")
    assert url_path(replaced) == "/z"
    # netloc and query ride along untouched.
    assert replaced == "https://x/z?q=1"


def test_loads_dict_or_none_accepts_objects():
    assert loads_dict_or_none("{}") == {}
    assert loads_dict_or_none('{"a":1}') == {"a": 1}


def test_loads_dict_or_none_rejects_non_objects():
    assert loads_dict_or_none("5") is None
    assert loads_dict_or_none("[1]") is None
    assert loads_dict_or_none("null") is None
    assert loads_dict_or_none("not json") is None
    assert loads_dict_or_none(None) is None


def test_loads_or_none_collapses_null_and_invalid():
    assert loads_or_none("null") is None
    assert loads_or_none("not json") is None


def test_loads_or_none_keeps_valid_scalars_and_objects():
    assert loads_or_none("5") == 5
    assert loads_or_none('{"a":1}') == {"a": 1}


def test_error_message_falls_back_on_empty():
    # A no-arg raise stringifies to "", which would persist an empty error row.
    assert error_message(Exception()) == "Unknown error occurred"


def test_error_message_uses_the_message():
    assert error_message(Exception("boom")) == "boom"


def test_clamp_bounds():
    assert clamp(5, 0, 10) == 5
    assert clamp(-1, 0, 10) == 0
    assert clamp(99, 0, 10) == 10
