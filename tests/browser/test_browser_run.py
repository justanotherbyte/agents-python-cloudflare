from __future__ import annotations

import asyncio
import json

import pytest

from agents.browser import (
    BrowserRenderingError,
    CdpSession,
    ConnectBrowserOptions,
    connect_browser,
    create_browser_session,
    get_browser_recording,
    list_browser_targets,
    load_cdp_spec,
)

from .fakes import FakeBrowser, FakeResponse, FakeSocket


@pytest.mark.asyncio
async def test_session_requests_are_typed_and_close_every_response():
    browser = FakeBrowser()
    info = await create_browser_session(
        browser, keep_alive_ms=60_000, include_targets=True, recording=True
    )
    targets = await list_browser_targets(browser, info.session_id)

    assert info.session_id == "session-1"
    assert targets[0].id == "target-1"
    assert browser.requests[0].url.endswith(
        "?keep_alive=60000&targets=true&recording=true"
    )
    assert all(response.closed for response in browser.responses)


@pytest.mark.asyncio
async def test_direct_connection_correlates_commands_and_deletes_on_async_close():
    browser = FakeBrowser()
    session = await connect_browser(browser)
    assert await session.send("Browser.getVersion") == {"echo": "Browser.getVersion"}

    await session.close()
    await session.close()

    assert browser.sockets[0].accepted
    deletes = [request for request in browser.requests if request.method == "DELETE"]
    assert len(deletes) == 1
    assert all(not listeners for listeners in browser.sockets[0].listeners.values())


@pytest.mark.asyncio
async def test_close_coalesces_concurrent_callers_and_retries_failed_cleanup():
    socket = FakeSocket()
    gate = asyncio.Event()
    started = asyncio.Event()
    attempts = 0

    async def cleanup():
        nonlocal attempts
        attempts += 1
        started.set()
        await gate.wait()
        if attempts == 1:
            raise RuntimeError("delete failed")

    session = CdpSession(socket, on_close=cleanup)
    first = asyncio.create_task(session.close())
    second = asyncio.create_task(session.close())
    await started.wait()
    gate.set()
    results = await asyncio.gather(first, second, return_exceptions=True)
    assert attempts == 1
    assert all(isinstance(result, RuntimeError) for result in results)

    await session.close()
    assert attempts == 2
    assert socket.closed == 1


@pytest.mark.asyncio
async def test_disconnect_releases_remaining_subscriptions_after_cancel_failure():
    cancelled = []

    class Subscription:
        def __init__(self, event):
            self.event = event

        def cancel(self):
            cancelled.append(self.event)
            if self.event == "message":
                raise RuntimeError("listener removal failed")

    class Socket(FakeSocket):
        def subscribe(self, event, _callback):
            return Subscription(event)

    session = CdpSession(Socket())
    await session.disconnect()
    assert cancelled == ["message", "error", "close"]


@pytest.mark.asyncio
async def test_cdp_attach_error_timeout_and_context_cleanup():
    socket = FakeSocket(respond=False)
    deleted = 0

    async def cleanup():
        nonlocal deleted
        deleted += 1

    with pytest.raises(TimeoutError, match="Page.navigate"):
        async with CdpSession(
            socket, default_timeout_ms=5, on_close=cleanup
        ) as session:
            await session.send("Page.navigate")
    assert deleted == 1
    assert socket.closed == 1
    assert all(not listeners for listeners in socket.listeners.values())


@pytest.mark.asyncio
async def test_cdp_socket_errors_reject_all_correlated_commands():
    socket = FakeSocket(respond=False)
    session = CdpSession(socket)
    pending = asyncio.create_task(session.send("Runtime.evaluate"))
    await asyncio.sleep(0)
    socket.emit("error", {})
    with pytest.raises(RuntimeError, match="socket error"):
        await pending
    assert all(not listeners for listeners in socket.listeners.values())
    await session.disconnect()


@pytest.mark.asyncio
async def test_cdp_cancellation_leaves_no_pending_listener_or_socket_resource():
    socket = FakeSocket(respond=False)
    session = CdpSession(socket)
    pending = asyncio.create_task(session.send("Runtime.evaluate"))
    await asyncio.sleep(0)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    await session.close()
    assert socket.closed == 1
    assert all(not listeners for listeners in socket.listeners.values())


@pytest.mark.asyncio
async def test_live_spec_is_normalized_cached_per_binding_and_session_is_deleted():
    one = FakeBrowser()
    two = FakeBrowser()
    first = await load_cdp_spec(one)
    again = await load_cdp_spec(one)
    other = await load_cdp_spec(two)

    assert again is first
    assert first.domains[0].commands[0].qualified_name == "Page.navigate"
    assert first.domains[0].events[0].qualified_name == "Page.loadEventFired"
    assert other == first
    assert one.created == 1
    assert two.created == 1
    assert len([r for r in one.requests if r.method == "DELETE"]) == 1


@pytest.mark.asyncio
async def test_recording_preserves_metadata_events_and_exact_bytes():
    browser = FakeBrowser()
    raw = json.dumps(
        {
            "success": True,
            "result": {
                "sessionId": "recorded",
                "duration": 12,
                "events": {"target-1": [{"type": 4}]},
            },
        },
        separators=(",", ":"),
    ).encode()

    async def recording_fetch(request):
        browser.requests.append(request)
        return FakeResponse(raw, headers={"content-type": "application/json"})

    browser.fetch = recording_fetch
    recording = await get_browser_recording(
        browser, account_id="account", api_token="secret", session_id="recorded"
    )
    assert recording.session_id == "recorded"
    assert recording.duration_ms == 12
    assert recording.events["target-1"] == ({"type": 4},)
    assert recording.data == raw


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "options",
    [
        ConnectBrowserOptions(browser="kitesurf", keep_alive_ms=1),
        ConnectBrowserOptions(browser="kitesurf", include_targets=True),
        ConnectBrowserOptions(browser="kitesurf", recording=True),
    ],
)
async def test_kitesurf_direct_connection_rejects_chromium_options_before_fetch(
    options,
):
    browser = FakeBrowser()
    with pytest.raises(ValueError, match="Kitesurf does not support"):
        await connect_browser(browser, options)
    assert browser.requests == []


@pytest.mark.asyncio
async def test_session_errors_keep_status_and_close_response():
    browser = FakeBrowser()

    async def failed(_request):
        response = FakeResponse("no", status=429)
        browser.responses.append(response)
        return response

    browser.fetch = failed
    with pytest.raises(BrowserRenderingError) as raised:
        await create_browser_session(browser)
    assert raised.value.status == 429
    assert browser.responses[0].closed
