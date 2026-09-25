from __future__ import annotations

import json
import time
import weakref
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass

from .browser_run import create_browser_session, delete_browser_session
from .protocols import BrowserTransport
from .types import BrowserRequest


@dataclass(frozen=True)
class CdpSpecItem:
    name: str
    qualified_name: str
    description: str | None = None


@dataclass(frozen=True)
class CdpSpecDomain:
    name: str
    commands: tuple[CdpSpecItem, ...]
    events: tuple[CdpSpecItem, ...]
    types: tuple[CdpSpecItem, ...]
    description: str | None = None


@dataclass(frozen=True)
class CdpSpec:
    domains: tuple[CdpSpecDomain, ...]


_cache: weakref.WeakKeyDictionary[object, tuple[float, CdpSpec]] = (
    weakref.WeakKeyDictionary()
)


def _items(domain: str, values: object, *, id_key: str) -> tuple[CdpSpecItem, ...]:
    if not isinstance(values, list):
        return ()
    result: list[CdpSpecItem] = []
    for value in values:
        if not isinstance(value, Mapping) or not isinstance(value.get(id_key), str):
            continue
        name = value[id_key]
        description = value.get("description")
        result.append(
            CdpSpecItem(
                name,
                f"{domain}.{name}",
                description if isinstance(description, str) else None,
            )
        )
    return tuple(result)


def normalize_cdp_spec(payload: object) -> CdpSpec:
    """Normalize a live `/json/protocol` response into a searchable shape."""

    raw_domains = payload.get("domains") if isinstance(payload, Mapping) else None
    if not isinstance(raw_domains, list):
        return CdpSpec(())
    domains: list[CdpSpecDomain] = []
    for value in raw_domains:
        if not isinstance(value, Mapping) or not isinstance(value.get("domain"), str):
            continue
        name = value["domain"]
        description = value.get("description")
        domains.append(
            CdpSpecDomain(
                name=name,
                commands=_items(name, value.get("commands"), id_key="name"),
                events=_items(name, value.get("events"), id_key="name"),
                types=_items(name, value.get("types"), id_key="id"),
                description=description if isinstance(description, str) else None,
            )
        )
    return CdpSpec(tuple(domains))


async def load_cdp_spec(
    browser: BrowserTransport,
    *,
    cache_ttl_ms: int = 300_000,
) -> CdpSpec:
    """Load and cache the live CDP protocol separately for each binding."""

    now = time.monotonic()
    cached = _cache.get(browser)
    if cached is not None and now - cached[0] < cache_ttl_ms / 1000:
        return cached[1]
    session = await create_browser_session(browser)
    try:
        response = await browser.fetch(
            BrowserRequest(
                f"https://localhost/v1/devtools/browser/{session.session_id}/json/protocol"
            )
        )
        status = response.status
        try:
            data = await response.read()
        finally:
            await response.aclose()
        if not 200 <= status < 300:
            raise RuntimeError(f"Failed to fetch live CDP spec: {status}")
        spec = normalize_cdp_spec(json.loads(data))
        _cache[browser] = (time.monotonic(), spec)
        return spec
    finally:
        # Protocol discovery cleanup must not replace its result or original error.
        with suppress(Exception):
            await delete_browser_session(browser, session.session_id)
