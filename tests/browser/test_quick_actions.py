from __future__ import annotations

import json

import pytest

from agents.browser import (
    Authentication,
    BrowserRenderingError,
    GotoOptions,
    QuickActionExtractInput,
    QuickActionOptions,
    QuickActionPage,
    QuickActionScrapeInput,
    QuickActionScreenshotInput,
    Viewport,
    browser_extract,
    browser_screenshot,
    browser_scrape,
    create_quick_action_tools,
)

from .fakes import FakeBrowser, FakeResponse


def test_quick_action_page_requires_exactly_one_source():
    with pytest.raises(ValueError, match="exactly one"):
        QuickActionPage()
    with pytest.raises(ValueError, match="exactly one"):
        QuickActionPage(url="https://x.test", html="<p>x</p>")
    assert QuickActionPage(html="<p>x</p>").params() == {"html": "<p>x</p>"}


@pytest.mark.asyncio
async def test_typed_extract_maps_all_host_navigation_and_request_options():
    browser = FakeBrowser()
    browser.quick_handler = lambda _action, _params: FakeResponse(
        {"success": True, "result": {"title": "x"}}
    )
    options = QuickActionOptions(
        goto=GotoOptions("networkidle0", 4_500),
        viewport=Viewport(1280, 720, 2),
        authentication=Authentication("u", "p"),
        headers={"Authorization": "Bearer token"},
        user_agent="agent",
        reject_request_patterns=(r"\.css$",),
        reject_resource_types=("image",),
        allow_request_patterns=("x.test",),
        allow_resource_types=("document",),
    )
    result = await browser_extract(
        browser,
        QuickActionExtractInput(
            QuickActionPage(url="https://x.test", options=options),
            prompt="title",
            schema={"type": "object"},
        ),
    )
    assert result == {"title": "x"}
    action, params = browser.quick_calls[0]
    assert action == "json"
    assert params["gotoOptions"] == {"waitUntil": "networkidle0", "timeout": 4500}
    assert params["authenticate"] == {"username": "u", "password": "p"}
    assert params["rejectResourceTypes"] == ["image"]


@pytest.mark.asyncio
async def test_screenshot_returns_exact_bytes_content_type_and_closes_body():
    browser = FakeBrowser()
    browser.quick_handler = lambda _action, _params: FakeResponse(
        b"\x89PNG", headers={"Content-Type": "image/png"}
    )
    result = await browser_screenshot(
        browser,
        QuickActionScreenshotInput(QuickActionPage(html="<h1>x</h1>")),
    )
    assert result.data == b"\x89PNG"
    assert result.content_type == "image/png"
    assert browser.responses[0].closed


@pytest.mark.asyncio
async def test_quick_action_error_names_compatibility_and_remote_requirements():
    browser = FakeBrowser()
    browser.quick_handler = lambda _action, _params: FakeResponse(
        {"errors": [{"message": "binding unavailable"}]}, status=500
    )
    with pytest.raises(BrowserRenderingError) as raised:
        await browser_screenshot(
            browser,
            QuickActionScreenshotInput(QuickActionPage(url="https://x.test")),
        )
    message = str(raised.value)
    assert "2026-03-24" in message
    assert "remote: true" in message
    assert "binding unavailable" in message


@pytest.mark.asyncio
async def test_binding_rpc_failure_names_compatibility_and_remote_requirements():
    browser = FakeBrowser()

    def fail(_action, _params):
        raise RuntimeError('RPC receiver does not implement "quickAction"')

    browser.quick_handler = fail
    with pytest.raises(RuntimeError) as raised:
        await browser_screenshot(
            browser,
            QuickActionScreenshotInput(QuickActionPage(url="https://x.test")),
        )
    assert "2026-03-24" in str(raised.value)
    assert "remote: true" in str(raised.value)


@pytest.mark.asyncio
async def test_model_tools_are_provider_neutral_bounded_and_hide_host_options():
    browser = FakeBrowser()
    browser.quick_handler = lambda _action, _params: FakeResponse(
        {"success": True, "result": "x" * 100}
    )
    tools = create_quick_action_tools(
        browser,
        options=QuickActionOptions(headers={"X-Host": "secret"}),
        max_chars=10,
    )
    assert set(tools) == {
        "browser_markdown",
        "browser_extract",
        "browser_links",
        "browser_scrape",
    }
    assert "browser_execute" not in tools
    assert "X-Host" not in str(tools["browser_markdown"].input_schema)
    result = await tools["browser_markdown"].execute({"url": "https://x.test"})
    assert result.startswith("x")
    assert len(json.dumps(result, separators=(",", ":"))) <= 10
    assert browser.quick_calls[0][1]["setExtraHTTPHeaders"] == {"X-Host": "secret"}


@pytest.mark.asyncio
async def test_raw_content_is_opt_in_and_model_source_validation_is_exact():
    browser = FakeBrowser()
    tools = create_quick_action_tools(browser, actions=("content",))
    assert set(tools) == {"browser_content"}
    schema = tools["browser_content"].input_schema
    assert "oneOf" in schema
    with pytest.raises(ValueError, match="exactly one"):
        await tools["browser_content"].execute({"url": "https://x.test", "html": "x"})
    assert create_quick_action_tools(browser, actions=()) == {}


@pytest.mark.asyncio
async def test_oversized_arrays_preserve_shape_and_objects_get_explicit_preview():
    browser = FakeBrowser()
    values = [f"https://x.test/{index}" for index in range(20)]
    browser.quick_handler = lambda _action, _params: FakeResponse(
        {"success": True, "result": values}
    )
    links = create_quick_action_tools(browser, max_chars=60)["browser_links"]
    result = await links.execute({"url": "https://x.test"})
    assert isinstance(result, list)
    assert 0 < len(result) < len(values)

    browser.quick_handler = lambda _action, _params: FakeResponse(
        {"success": True, "result": {"blob": "y" * 200}}
    )
    extract = create_quick_action_tools(browser, max_chars=100)["browser_extract"]
    result = await extract.execute({"url": "https://x.test", "prompt": "blob"})
    assert result["truncated"] is True
    assert "preview" in result
    assert len(json.dumps(result, separators=(",", ":"))) <= 100


@pytest.mark.asyncio
async def test_model_bound_counts_json_escaping_and_truncation_suffix():
    browser = FakeBrowser()
    value = '"\\\n' * 100
    browser.quick_handler = lambda _action, _params: FakeResponse(
        {"success": True, "result": value}
    )
    tool = create_quick_action_tools(browser, max_chars=80)["browser_markdown"]
    result = await tool.execute({"url": "https://x.test"})
    assert len(json.dumps(result, separators=(",", ":"))) <= 80


@pytest.mark.asyncio
async def test_model_bound_handles_small_limits_and_non_json_numbers():
    browser = FakeBrowser()
    browser.quick_handler = lambda _action, _params: FakeResponse(
        {"success": True, "result": float("nan")}
    )
    tool = create_quick_action_tools(browser, max_chars=1)["browser_extract"]
    result = await tool.execute({"url": "https://x.test", "prompt": "number"})
    assert len(json.dumps(result, allow_nan=False, separators=(",", ":"))) <= 1


@pytest.mark.asyncio
async def test_scrape_results_are_typed_with_geometry_and_attributes():
    browser = FakeBrowser()
    browser.quick_handler = lambda _action, _params: FakeResponse(
        {
            "success": True,
            "result": [
                {
                    "selector": "h1",
                    "results": [
                        {
                            "html": "<h1>x</h1>",
                            "text": "x",
                            "width": 100,
                            "height": 20,
                            "top": 1,
                            "left": 2,
                            "attributes": [{"name": "class", "value": "title"}],
                        }
                    ],
                }
            ],
        }
    )
    result = await browser_scrape(
        browser,
        QuickActionScrapeInput(QuickActionPage(url="https://x.test"), ("h1",)),
    )
    assert result[0].results[0].width == 100
    assert result[0].results[0].attributes[0].name == "class"
