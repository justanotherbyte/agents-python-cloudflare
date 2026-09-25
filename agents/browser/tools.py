from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Literal

from .protocols import QuickActionBinding
from .quick_actions import (
    QuickActionExtractInput,
    QuickActionOptions,
    QuickActionPage,
    QuickActionScrapeInput,
    browser_content,
    browser_extract,
    browser_links,
    browser_markdown,
    browser_scrape,
)

type QuickActionToolName = Literal["markdown", "extract", "links", "scrape", "content"]


@dataclass(frozen=True)
class ModelToolDescriptor:
    """Provider-neutral JSON Schema plus its host-side async executor."""

    name: str
    description: str
    input_schema: Mapping[str, object]
    execute: Callable[[Mapping[str, object]], Awaitable[object]]


_PAGE_PROPERTIES: dict[str, object] = {
    "url": {"type": "string", "format": "uri"},
    "html": {"type": "string"},
}
_PAGE_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": _PAGE_PROPERTIES,
    "oneOf": [
        {"required": ["url"], "not": {"required": ["html"]}},
        {"required": ["html"], "not": {"required": ["url"]}},
    ],
    "additionalProperties": False,
}


def _page(input: Mapping[str, object], options: QuickActionOptions) -> QuickActionPage:
    url = input.get("url")
    html = input.get("html")
    if url is not None and not isinstance(url, str):
        raise ValueError("url must be a string")
    if html is not None and not isinstance(html, str):
        raise ValueError("html must be a string")
    return QuickActionPage(url=url, html=html, options=options)


def _bound(value: object, max_chars: int) -> object:
    if max_chars <= 0:
        return value
    if isinstance(value, str):
        if _serialized_length(value) <= max_chars:
            return value
        return _bound_string(value, max_chars)
    try:
        serialized = json.dumps(value, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError):
        value = str(value)
        serialized = json.dumps(value, allow_nan=False, separators=(",", ":"))
    if len(serialized) <= max_chars:
        return value
    if isinstance(value, (list, tuple)):
        kept: list[object] = []
        for item in value:
            candidate = [*kept, item]
            if _serialized_length(candidate) > max_chars:
                break
            kept.append(item)
        if kept:
            return tuple(kept) if isinstance(value, tuple) else kept
    summary: object = {
        "truncated": True,
        "note": (
            f"Result is too large ({len(serialized)} characters); narrow the request."
        ),
        "preview": "",
    }
    if _serialized_length(summary) > max_chars:
        summary = {"truncated": True}
    if _serialized_length(summary) > max_chars:
        return _bound_string("[truncated]", max_chars)
    if not isinstance(summary, dict) or "preview" not in summary:
        return summary
    low = 0
    high = len(serialized)
    while low < high:
        middle = (low + high + 1) // 2
        candidate = {**summary, "preview": serialized[:middle]}
        if _serialized_length(candidate) <= max_chars:
            low = middle
        else:
            high = middle - 1
    return {**summary, "preview": serialized[:low]}


def _serialized_length(value: object) -> int:
    return len(json.dumps(value, allow_nan=False, separators=(",", ":")))


def _bound_string(value: str, max_chars: int) -> object:
    if max_chars < 2:
        return 0
    low = 0
    high = len(value)
    result = ""
    while low <= high:
        middle = (low + high) // 2
        omitted = len(value) - middle
        suffix = f"\n\n[truncated {omitted} characters]" if omitted else ""
        candidate = f"{value[:middle]}{suffix}"
        if _serialized_length(candidate) <= max_chars:
            result = candidate
            low = middle + 1
        else:
            high = middle - 1
    if result:
        return result
    low = 0
    high = len(value)
    while low < high:
        middle = (low + high + 1) // 2
        if _serialized_length(value[:middle]) <= max_chars:
            low = middle
        else:
            high = middle - 1
    return value[:low]


def create_quick_action_tools(
    browser: QuickActionBinding,
    *,
    actions: Sequence[QuickActionToolName] | None = None,
    options: QuickActionOptions | None = None,
    max_chars: int = 50_000,
) -> dict[str, ModelToolDescriptor]:
    """Build bounded model tools without exposing host options or raw execution."""

    enabled = set(
        ("markdown", "extract", "links", "scrape") if actions is None else actions
    )
    request_options = options or QuickActionOptions()
    tools: dict[str, ModelToolDescriptor] = {}

    async def markdown(input: Mapping[str, object]) -> object:
        return _bound(
            await browser_markdown(browser, _page(input, request_options)), max_chars
        )

    async def links(input: Mapping[str, object]) -> object:
        return _bound(
            list(await browser_links(browser, _page(input, request_options))), max_chars
        )

    async def content(input: Mapping[str, object]) -> object:
        return _bound(
            await browser_content(browser, _page(input, request_options)), max_chars
        )

    async def extract(input: Mapping[str, object]) -> object:
        prompt = input.get("prompt")
        schema = input.get("schema")
        if prompt is not None and not isinstance(prompt, str):
            raise ValueError("prompt must be a string")
        if schema is not None and not isinstance(schema, Mapping):
            raise ValueError("schema must be a JSON object")
        request = QuickActionExtractInput(
            _page(input, request_options), prompt=prompt, schema=schema
        )
        return _bound(await browser_extract(browser, request), max_chars)

    async def scrape(input: Mapping[str, object]) -> object:
        selectors = input.get("selectors")
        if not isinstance(selectors, list) or not all(
            isinstance(selector, str) for selector in selectors
        ):
            raise ValueError("selectors must be an array of strings")
        request = QuickActionScrapeInput(_page(input, request_options), selectors)
        result = await browser_scrape(browser, request)
        return _bound([asdict(item) for item in result], max_chars)

    definitions = {
        "markdown": (
            "browser_markdown",
            "Render a URL or raw HTML and return Markdown.",
            _PAGE_SCHEMA,
            markdown,
        ),
        "extract": (
            "browser_extract",
            "Extract structured data using a prompt, JSON Schema, or both.",
            {
                **_PAGE_SCHEMA,
                "properties": {
                    **_PAGE_PROPERTIES,
                    "prompt": {"type": "string"},
                    "schema": {"type": "object"},
                },
                "anyOf": [{"required": ["prompt"]}, {"required": ["schema"]}],
            },
            extract,
        ),
        "links": (
            "browser_links",
            "Return links found in a URL or raw HTML.",
            _PAGE_SCHEMA,
            links,
        ),
        "scrape": (
            "browser_scrape",
            "Scrape selected elements from a URL or raw HTML.",
            {
                **_PAGE_SCHEMA,
                "properties": {
                    **_PAGE_PROPERTIES,
                    "selectors": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                    },
                },
                "required": ["selectors"],
            },
            scrape,
        ),
        "content": (
            "browser_content",
            "Return fully rendered HTML; prefer Markdown unless raw HTML is needed.",
            _PAGE_SCHEMA,
            content,
        ),
    }
    for action in enabled:
        if action not in definitions:
            raise ValueError(f"unsupported Quick Action model tool: {action}")
        name, description, schema, execute = definitions[action]
        tools[name] = ModelToolDescriptor(name, description, schema, execute)
    return tools
