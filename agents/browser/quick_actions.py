"""Typed Browser Run Quick Actions.

Workers require compatibility date 2026-03-24 or later. Local `wrangler dev`
also requires `remote: true` on the Browser binding.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from .browser_run import BrowserRenderingError
from .protocols import BrowserResponse, QuickActionBinding

QUICK_ACTION_REQUIREMENTS = (
    "Browser Run Quick Actions require compatibility_date 2026-03-24 or later; "
    "local wrangler dev also requires remote: true on the browser binding"
)
type QuickAction = Literal[
    "content",
    "screenshot",
    "pdf",
    "markdown",
    "snapshot",
    "scrape",
    "json",
    "links",
]


@dataclass(frozen=True)
class GotoOptions:
    wait_until: str | None = None
    timeout_ms: int | None = None

    def params(self) -> dict[str, object]:
        result: dict[str, object] = {}
        if self.wait_until is not None:
            result["waitUntil"] = self.wait_until
        if self.timeout_ms is not None:
            result["timeout"] = self.timeout_ms
        return result


@dataclass(frozen=True)
class Viewport:
    width: int
    height: int
    device_scale_factor: float | None = None

    def params(self) -> dict[str, object]:
        result: dict[str, object] = {"width": self.width, "height": self.height}
        if self.device_scale_factor is not None:
            result["deviceScaleFactor"] = self.device_scale_factor
        return result


@dataclass(frozen=True)
class Authentication:
    username: str
    password: str


@dataclass(frozen=True)
class QuickActionOptions:
    """Host-controlled navigation and request policy for Quick Actions."""

    goto: GotoOptions | None = None
    viewport: Viewport | None = None
    cookies: Sequence[Mapping[str, object]] = ()
    authentication: Authentication | None = None
    headers: Mapping[str, str] = field(default_factory=dict)
    user_agent: str | None = None
    reject_request_patterns: Sequence[str] = ()
    reject_resource_types: Sequence[str] = ()
    allow_request_patterns: Sequence[str] = ()
    allow_resource_types: Sequence[str] = ()

    def params(self) -> dict[str, object]:
        result: dict[str, object] = {}
        if self.goto is not None:
            result["gotoOptions"] = self.goto.params()
        if self.viewport is not None:
            result["viewport"] = self.viewport.params()
        if self.cookies:
            result["cookies"] = [dict(cookie) for cookie in self.cookies]
        if self.authentication is not None:
            result["authenticate"] = {
                "username": self.authentication.username,
                "password": self.authentication.password,
            }
        if self.headers:
            result["setExtraHTTPHeaders"] = dict(self.headers)
        if self.user_agent is not None:
            result["userAgent"] = self.user_agent
        if self.reject_request_patterns:
            result["rejectRequestPattern"] = list(self.reject_request_patterns)
        if self.reject_resource_types:
            result["rejectResourceTypes"] = list(self.reject_resource_types)
        if self.allow_request_patterns:
            result["allowRequestPattern"] = list(self.allow_request_patterns)
        if self.allow_resource_types:
            result["allowResourceTypes"] = list(self.allow_resource_types)
        return result


@dataclass(frozen=True)
class QuickActionPage:
    """A Quick Action page with exactly one URL or raw HTML source."""

    url: str | None = None
    html: str | None = None
    options: QuickActionOptions = field(default_factory=QuickActionOptions)

    def __post_init__(self) -> None:
        if (self.url is None) == (self.html is None):
            raise ValueError("Quick Actions require exactly one of url or html")
        if self.url == "" or self.html == "":
            raise ValueError("Quick Action url or html must not be empty")

    def params(self) -> dict[str, object]:
        result = self.options.params()
        if self.url is not None:
            result["url"] = self.url
        else:
            result["html"] = self.html
        return result


@dataclass(frozen=True)
class QuickActionExtractInput:
    page: QuickActionPage
    prompt: str | None = None
    schema: Mapping[str, object] | None = None
    custom_ai: Sequence[Mapping[str, str]] = ()

    def __post_init__(self) -> None:
        if self.prompt is None and self.schema is None:
            raise ValueError("browser_extract requires a prompt, a schema, or both")

    def params(self) -> dict[str, object]:
        result = self.page.params()
        if self.prompt is not None:
            result["prompt"] = self.prompt
        if self.schema is not None:
            result["response_format"] = {
                "type": "json_schema",
                "json_schema": dict(self.schema),
            }
        if self.custom_ai:
            result["custom_ai"] = [dict(model) for model in self.custom_ai]
        return result


@dataclass(frozen=True)
class QuickActionScrapeInput:
    page: QuickActionPage
    selectors: Sequence[str]

    def __post_init__(self) -> None:
        if not self.selectors:
            raise ValueError("browser_scrape requires at least one selector")

    def params(self) -> dict[str, object]:
        return {
            **self.page.params(),
            "elements": [{"selector": selector} for selector in self.selectors],
        }


@dataclass(frozen=True)
class QuickActionScreenshotInput:
    page: QuickActionPage
    screenshot_options: Mapping[str, object] = field(default_factory=dict)
    selector: str | None = None

    def params(self) -> dict[str, object]:
        result = self.page.params()
        if self.screenshot_options:
            result["screenshotOptions"] = dict(self.screenshot_options)
        if self.selector is not None:
            result["selector"] = self.selector
        return result


@dataclass(frozen=True)
class QuickActionBinary:
    data: bytes
    content_type: str


@dataclass(frozen=True)
class QuickActionSnapshot:
    content: str
    screenshot: str


@dataclass(frozen=True)
class ScrapeAttribute:
    name: str
    value: str


@dataclass(frozen=True)
class ScrapeElement:
    html: str
    text: str
    width: float
    height: float
    top: float
    left: float
    attributes: tuple[ScrapeAttribute, ...]


@dataclass(frozen=True)
class QuickActionScrapeResult:
    selector: str
    results: tuple[ScrapeElement, ...]


type QuickActionInput = (
    QuickActionPage
    | QuickActionExtractInput
    | QuickActionScrapeInput
    | QuickActionScreenshotInput
)


async def _body(response: BrowserResponse) -> bytes:
    try:
        return await response.read()
    finally:
        await response.aclose()


def _content_type(response: BrowserResponse) -> str:
    for key, value in response.headers.items():
        if key.lower() == "content-type":
            return value
    return "application/octet-stream"


async def run_quick_action(
    browser: QuickActionBinding,
    action: QuickAction,
    input: QuickActionInput,
) -> BrowserResponse:
    """Start a typed Quick Action; the caller owns the returned response body."""

    return await _run_action(browser, action, input.params())


def _error_detail(data: bytes) -> str:
    text = data.decode("utf-8", errors="replace")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return text[:200]
    if isinstance(payload, Mapping):
        errors = payload.get("errors")
        if isinstance(errors, list):
            messages = [
                item.get("message")
                for item in errors
                if isinstance(item, Mapping) and isinstance(item.get("message"), str)
            ]
            if messages:
                return "; ".join(messages)
        raw = payload.get("rawAiResponse")
        if isinstance(raw, str):
            return raw
    return text[:200]


async def _json_action(
    browser: QuickActionBinding,
    action: str,
    params: Mapping[str, object],
) -> object:
    response = await _run_action(browser, action, params)
    status = response.status
    data = await _body(response)
    if not 200 <= status < 300:
        detail = _error_detail(data)
        suffix = f": {detail}" if detail else ""
        raise BrowserRenderingError(
            f"Browser Run {action} failed ({status}){suffix}. "
            f"{QUICK_ACTION_REQUIREMENTS}",
            status,
        )
    try:
        payload = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BrowserRenderingError(
            f"Browser Run {action} returned invalid JSON. {QUICK_ACTION_REQUIREMENTS}",
            status,
        ) from error
    if not isinstance(payload, Mapping) or payload.get("success") is False:
        raise BrowserRenderingError(
            f"Browser Run {action} returned no result: {_error_detail(data)}",
            status,
        )
    if "result" not in payload:
        raise BrowserRenderingError(f"Browser Run {action} returned no result", status)
    return payload["result"]


async def _binary_action(
    browser: QuickActionBinding,
    action: str,
    params: Mapping[str, object],
) -> QuickActionBinary:
    response = await _run_action(browser, action, params)
    status = response.status
    content_type = _content_type(response)
    data = await _body(response)
    if not 200 <= status < 300:
        detail = _error_detail(data)
        raise BrowserRenderingError(
            f"Browser Run {action} failed ({status}): {detail}. "
            f"{QUICK_ACTION_REQUIREMENTS}",
            status,
        )
    return QuickActionBinary(data, content_type)


async def _run_action(
    browser: QuickActionBinding,
    action: str,
    params: Mapping[str, object],
) -> BrowserResponse:
    try:
        return await browser.quick_action(action, params)
    except Exception as error:
        raise RuntimeError(
            f"Browser Run {action} could not start. {QUICK_ACTION_REQUIREMENTS}"
        ) from error


async def browser_content(browser: QuickActionBinding, page: QuickActionPage) -> str:
    result = await _json_action(browser, "content", page.params())
    if not isinstance(result, str):
        raise TypeError("Browser Run content result was not a string")
    return result


async def browser_markdown(browser: QuickActionBinding, page: QuickActionPage) -> str:
    result = await _json_action(browser, "markdown", page.params())
    if not isinstance(result, str):
        raise TypeError("Browser Run markdown result was not a string")
    return result


async def browser_extract(
    browser: QuickActionBinding, input: QuickActionExtractInput
) -> object:
    return await _json_action(browser, "json", input.params())


async def browser_links(
    browser: QuickActionBinding, page: QuickActionPage
) -> tuple[str, ...]:
    result = await _json_action(browser, "links", page.params())
    if not isinstance(result, list) or not all(
        isinstance(item, str) for item in result
    ):
        raise TypeError("Browser Run links result was not a string array")
    return tuple(result)


async def browser_scrape(
    browser: QuickActionBinding, input: QuickActionScrapeInput
) -> tuple[QuickActionScrapeResult, ...]:
    result = await _json_action(browser, "scrape", input.params())
    if not isinstance(result, list):
        raise TypeError("Browser Run scrape result was not an array")
    return tuple(_scrape_result(item) for item in result)


async def browser_snapshot(
    browser: QuickActionBinding, page: QuickActionPage
) -> QuickActionSnapshot:
    result = await _json_action(browser, "snapshot", page.params())
    if not isinstance(result, Mapping):
        raise TypeError("Browser Run snapshot result was not an object")
    content = result.get("content")
    screenshot = result.get("screenshot")
    if not isinstance(content, str) or not isinstance(screenshot, str):
        raise TypeError("Browser Run snapshot result omitted content or screenshot")
    return QuickActionSnapshot(content, screenshot)


async def browser_screenshot(
    browser: QuickActionBinding, input: QuickActionScreenshotInput
) -> QuickActionBinary:
    return await _binary_action(browser, "screenshot", input.params())


async def browser_pdf(
    browser: QuickActionBinding, page: QuickActionPage
) -> QuickActionBinary:
    return await _binary_action(browser, "pdf", page.params())


def _scrape_result(value: object) -> QuickActionScrapeResult:
    if not isinstance(value, Mapping) or not isinstance(value.get("selector"), str):
        raise TypeError("Browser Run scrape entry omitted its selector")
    raw_results = value.get("results")
    if not isinstance(raw_results, list):
        raise TypeError("Browser Run scrape entry omitted its results")
    return QuickActionScrapeResult(
        value["selector"], tuple(_scrape_element(item) for item in raw_results)
    )


def _scrape_element(value: object) -> ScrapeElement:
    if not isinstance(value, Mapping):
        raise TypeError("Browser Run scrape element was not an object")
    required = ("html", "text", "width", "height", "top", "left")
    if not all(key in value for key in required):
        raise TypeError("Browser Run scrape element omitted geometry or content")
    html = value["html"]
    text = value["text"]
    geometry = tuple(value[key] for key in required[2:])
    if (
        not isinstance(html, str)
        or not isinstance(text, str)
        or not all(isinstance(item, (int, float)) for item in geometry)
    ):
        raise TypeError("Browser Run scrape element has invalid content or geometry")
    raw_attributes = value.get("attributes", [])
    if not isinstance(raw_attributes, list):
        raise TypeError("Browser Run scrape attributes were not an array")
    attributes: list[ScrapeAttribute] = []
    for item in raw_attributes:
        if (
            not isinstance(item, Mapping)
            or not isinstance(item.get("name"), str)
            or not isinstance(item.get("value"), str)
        ):
            raise TypeError("Browser Run scrape attribute was invalid")
        attributes.append(ScrapeAttribute(item["name"], item["value"]))
    return ScrapeElement(
        html,
        text,
        float(geometry[0]),
        float(geometry[1]),
        float(geometry[2]),
        float(geometry[3]),
        tuple(attributes),
    )
