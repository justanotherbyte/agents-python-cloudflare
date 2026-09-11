from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from typing import Any, Literal, TypedDict, cast
from urllib.parse import quote, unquote

from ..lifecycle import LifecycleRouteAddress, LifecycleRouteEnvelope
from ._wire import strict_json_loads
from .protocol import PathStep
from .utils import dumps_wire, loads_or_none


class _FacetOperationGate:
    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.users = 0


# A child's routing id carries this prefix, so an object recognises itself as a
# child without reading storage.
FACET_ID_PREFIX = "cf-agents:v2:"

# The fixed URL segment between one parent-to-child hop and the next. Not
# configurable: the client builds the same literal.
SUB_PREFIX = "sub"

# Separates a child's class from its name in the runtime's facet key. Reserved, so
# a name carrying one is rejected rather than silently addressing a different child.
_FACET_KEY_SEP = "\0"

# encodeURIComponent leaves exactly these unescaped and escapes ":", which is what
# lets a routing id be split on colons to recover the name.
_URI_SAFE = "-_.!~*'()"


def _facet_logical_name(routed_name: Any) -> str | None:
    # The routing id is itself the marker, so nothing is written on spawn and read back
    # on wake — the socket map is built before any table is readable. Malformed reads
    # as "not a child" because __init__ re-runs on every wake with no way to repair.
    if not isinstance(routed_name, str):
        return None
    if not routed_name.startswith(FACET_ID_PREFIX):
        return None

    # The name cannot contribute a colon of its own, because it is percent-encoded.
    parts = routed_name.split(":")
    if len(parts) != 4:
        return None

    return unquote(parts[2])


def _next_sub_hop(path: str, *, is_child: bool) -> tuple[str, str, str] | None:
    """Split off the next parent-to-child hop, or None if the path ends here.

    Returns the child's kebab-case class, its decoded name, and the path the
    child should see.
    """
    parts = [part for part in path.split("/") if part]

    # The marker's position is known rather than searched for, so a parent instance
    # named "sub" cannot be mistaken for one. Only correct while the router keeps its
    # prefix to a single segment.
    parts = parts if is_child else parts[3:]

    if len(parts) < 3 or parts[0] != SUB_PREFIX:
        return None

    tail = parts[3:]
    remaining = "/" + "/".join(tail) if tail else "/"
    return parts[1], unquote(parts[2]), remaining


def _facet_identity(path: list[PathStep], name: str) -> str:
    # Every routing id is minted from the root's namespace, so the ancestor path is
    # folded in to keep same-named children under different parents distinct. The
    # digest must match the other runtime's bytes exactly — see AGENTS.md.
    payload = dumps_wire(path)
    digest = hashlib.sha256(payload.encode()).hexdigest()
    return f"{FACET_ID_PREFIX}{quote(name, safe=_URI_SAFE)}:{digest}"


def _agent_path_key(path: list[PathStep]) -> str:
    return "/".join(
        f"{quote(step['className'], safe=_URI_SAFE)}:"
        f"{quote(step['name'], safe=_URI_SAFE)}"
        for step in path
    )


def _agent_route_address(path: list[PathStep]) -> LifecycleRouteAddress:
    return LifecycleRouteAddress(_agent_path_key(path), dumps_wire(path))


@dataclass(frozen=True, slots=True)
class _StaleLifecycleRoute:
    path: list[PathStep]


class _LifecycleRouteAddressWire(TypedDict):
    key: str
    data: str


class _LifecycleRouteEnvelopeWire(TypedDict):
    version: int
    source: _LifecycleRouteAddressWire | None
    target: _LifecycleRouteAddressWire
    capability_id: str
    payload: object


class _LifecycleRouteValueWire(TypedDict):
    type: Literal["value"]
    value: object


class _LifecycleRouteStaleWire(TypedDict):
    type: Literal["stale"]
    path: list[PathStep]


def _route_address_path(address: LifecycleRouteAddress) -> list[PathStep]:
    try:
        parsed = strict_json_loads(address.data, "Lifecycle route address")
    except ValueError as error:
        raise ValueError("Lifecycle route address is not a valid Agent path") from error
    path = _agent_path_from_value(parsed)
    if _agent_path_key(path) != address.key:
        raise ValueError("Lifecycle route address key does not match its Agent path")
    return path


def _agent_path_from_value(value: object) -> list[PathStep]:
    if not isinstance(value, list) or not value:
        raise ValueError("Lifecycle route address is not a valid Agent path")
    path = []
    for entry in value:
        if not isinstance(entry, dict):
            raise ValueError("Lifecycle route address is not a valid Agent path")
        class_name = entry.get("className")
        name = entry.get("name")
        if not isinstance(class_name, str) or not isinstance(name, str):
            raise ValueError("Lifecycle route address is not a valid Agent path")
        path.append(_path_step(class_name, name))
    return path


def _route_address_to_wire(
    address: LifecycleRouteAddress | None,
) -> _LifecycleRouteAddressWire | None:
    if address is None:
        return None
    return _LifecycleRouteAddressWire(key=address.key, data=address.data)


def _route_address_from_wire(value: object) -> LifecycleRouteAddress | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("Lifecycle route address must be an object")
    address = LifecycleRouteAddress(
        cast(str, value.get("key")),
        cast(str, value.get("data")),
    )
    _route_address_path(address)
    return address


def _route_envelope_to_wire(envelope: LifecycleRouteEnvelope) -> str:
    target = _route_address_to_wire(envelope.target)
    assert target is not None
    return dumps_wire(
        _LifecycleRouteEnvelopeWire(
            version=envelope.version,
            source=_route_address_to_wire(envelope.source),
            target=target,
            capability_id=envelope.capability_id,
            payload=envelope.payload,
        )
    )


def _route_envelope_from_wire(raw: str) -> LifecycleRouteEnvelope:
    try:
        value = strict_json_loads(raw, "Lifecycle route envelope")
    except ValueError as error:
        raise ValueError("invalid Lifecycle route envelope") from error
    if not isinstance(value, dict):
        raise ValueError("Lifecycle route envelope must be an object")
    target = _route_address_from_wire(value.get("target"))
    if target is None:
        raise ValueError("Lifecycle route envelope requires a target")
    capability_id = value.get("capability_id")
    if not isinstance(capability_id, str) or not capability_id:
        raise ValueError("Lifecycle route envelope requires a capability")
    return LifecycleRouteEnvelope(
        version=cast(int, value.get("version")),
        source=_route_address_from_wire(value.get("source")),
        target=target,
        capability_id=capability_id,
        payload=value.get("payload"),
    )


def _route_rpc_result(value: object) -> object | _StaleLifecycleRoute:
    value = _rpc_string(value, "Lifecycle route RPC")
    try:
        result = strict_json_loads(value, "Lifecycle route RPC result")
    except ValueError as error:
        raise ValueError("Lifecycle route RPC returned invalid JSON") from error
    if not isinstance(result, dict) or result.get("type") not in ("value", "stale"):
        raise ValueError("Lifecycle route RPC returned an invalid envelope")
    if result["type"] == "value" and set(result) == {"type", "value"}:
        return result["value"]
    if result["type"] == "stale" and set(result) == {"type", "path"}:
        return _StaleLifecycleRoute(_agent_path_from_value(result["path"]))
    raise ValueError("Lifecycle route RPC returned an invalid envelope")


def _rpc_string(value: object, label: str) -> str:
    to_py = getattr(value, "to_py", None)
    if callable(to_py):
        value = to_py()
    if not isinstance(value, str):
        raise TypeError(f"{label} returned a non-string result")
    return value


def _facet_key(class_name: str, name: str) -> str:
    return f"{class_name}{_FACET_KEY_SEP}{name}"


def _path_step(class_name: str, name: str) -> PathStep:
    return PathStep(className=class_name, name=name)


def _parse_parent_path(raw: Any) -> list[PathStep]:
    # A previous life wrote this, so a malformed row degrades to "no known ancestors"
    # rather than raising on a path that runs during startup.
    parsed = loads_or_none(raw)
    if not isinstance(parsed, list):
        return []

    steps = []
    for entry in parsed:
        if not isinstance(entry, dict):
            return []
        class_name = entry.get("className")
        name = entry.get("name")
        if not isinstance(class_name, str) or not isinstance(name, str):
            return []
        steps.append(_path_step(class_name, name))

    return steps
