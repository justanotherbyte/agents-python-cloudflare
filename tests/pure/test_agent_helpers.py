"""Module-level helpers in the canonical core modules.

Facet identity is a byte-for-byte wire contract with the TS runtime, so it is
checked both by recomputing it independently and against one frozen golden
string. Routing is positional, so an instance named `sub` must not be mistaken
for the hop marker.
"""

from __future__ import annotations

import hashlib
import json
from urllib.parse import quote

from agents.core.agent import _parse_version
from agents.core.facets import (
    FACET_ID_PREFIX,
    _facet_identity,
    _facet_logical_name,
    _next_sub_hop,
    _parse_parent_path,
)
from agents.core.protocol import PathStep
from agents.core.routing import (
    camel_to_kebab,
    kebab_to_screaming,
)

# encodeURIComponent's unescaped set, reproduced so the recompute matches the source.
_URI_SAFE = "-_.!~*'()"


def test_parse_version_treats_garbage_and_negatives_as_zero():
    assert _parse_version("3") == 3
    assert _parse_version(3) == 3
    assert _parse_version(None) == 0
    assert _parse_version("abc") == 0
    assert _parse_version(-5) == 0
    assert _parse_version("-2") == 0


def test_camel_to_kebab_pascal_case():
    assert camel_to_kebab("MyAgent") == "my-agent"


def test_camel_to_kebab_consecutive_capitals():
    # Each capital is its own boundary, so an acronym splits letter by letter.
    assert camel_to_kebab("AIChatAgent") == "a-i-chat-agent"


def test_camel_to_kebab_screaming_snake():
    assert camel_to_kebab("SCREAMING_CASE") == "screaming-case"


def test_camel_to_kebab_trailing_capital():
    assert camel_to_kebab("AgentX") == "agent-x"


def test_kebab_to_screaming():
    assert kebab_to_screaming("my-agent") == "MY_AGENT"


def test_facet_identity_matches_independent_recompute():
    path: list[PathStep] = [
        PathStep(className="Root", name="a"),
        PathStep(className="Child", name="b"),
    ]
    payload = json.dumps(path, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode()).hexdigest()
    expected = f"{FACET_ID_PREFIX}{quote('b', safe=_URI_SAFE)}:{digest}"
    assert _facet_identity(path, "b") == expected


def test_facet_identity_golden_string():
    # Frozen to lock the exact byte format the TS runtime reproduces; recomputed
    # once and hardcoded so a change to the digest recipe is caught here.
    path: list[PathStep] = [PathStep(className="Root", name="a")]
    assert _facet_identity(path, "a") == (
        "cf-agents:v2:a:"
        "926c2d1587281b409342d788ac19368fde6dbfb0de20934c4d01a1d6075d0904"
    )


def test_facet_logical_name_round_trips_a_valid_id():
    path: list[PathStep] = [PathStep(className="Root", name="a")]
    routed = _facet_identity(path, "a")
    assert _facet_logical_name(routed) == "a"


def test_facet_logical_name_round_trips_a_reserved_char():
    # encodeURIComponent escapes ":", so a name carrying one cannot add a colon to
    # the id and it still splits into exactly four parts, decoding back cleanly.
    name = "a:b"
    routed = _facet_identity([PathStep(className="C", name=name)], name)
    assert _facet_logical_name(routed) == name


def test_facet_logical_name_rejects_wrong_prefix():
    assert _facet_logical_name("not-a-facet-id") is None


def test_facet_logical_name_rejects_malformed_colon_count():
    # Three parts, not four: not a v2 id.
    assert _facet_logical_name("cf-agents:v2:a") is None


def test_facet_logical_name_rejects_non_str():
    assert _facet_logical_name(123) is None
    assert _facet_logical_name(None) is None


def test_next_sub_hop_splits_a_top_level_path():
    hop = _next_sub_hop(
        "/agents/my-agent/inst/sub/child-class/child-name/tail",
        is_child=False,
    )
    assert hop == ("child-class", "child-name", "/tail")


def test_next_sub_hop_ignores_an_instance_named_sub():
    # The marker is read at a fixed offset, not searched for, so an instance
    # literally named "sub" is not treated as one.
    assert _next_sub_hop("/agents/my-agent/sub", is_child=False) is None


def test_next_sub_hop_returns_none_without_a_hop():
    assert _next_sub_hop("/agents/my-agent/inst", is_child=False) is None


def test_parse_parent_path_valid_list():
    steps = _parse_parent_path('[{"className":"A","name":"x"}]')
    assert steps == [PathStep(className="A", name="x")]


def test_parse_parent_path_non_list_degrades_to_empty():
    assert _parse_parent_path('{"a":1}') == []


def test_parse_parent_path_malformed_entry_degrades_to_empty():
    # A previous life wrote this, so a bad entry degrades rather than raising.
    assert _parse_parent_path('[{"className":"A"}]') == []
