from __future__ import annotations

import json
import inspect
from collections.abc import Mapping
from urllib.parse import urlencode

from .cdp import CdpSession
from .protocols import BrowserResponse, BrowserTransport
from .types import (
    BrowserRecording,
    BrowserRequest,
    BrowserSessionInfo,
    BrowserTargetInfo,
    ConnectBrowserOptions,
)

_BASE = "https://localhost/v1/devtools/browser"


class BrowserRenderingError(RuntimeError):
    """A Browser Run request failed with an HTTP status."""

    def __init__(self, message: str, status: int):
        super().__init__(message)
        self.status = status


def _endpoint(
    session_id: str | None = None,
    *,
    keep_alive_ms: int | None = None,
    include_targets: bool = False,
    recording: bool = False,
    browser: str | None = None,
) -> str:
    url = _BASE if session_id is None else f"{_BASE}/{session_id}"
    query: list[tuple[str, str]] = []
    if keep_alive_ms is not None:
        query.append(("keep_alive", str(keep_alive_ms)))
    if include_targets:
        query.append(("targets", "true"))
    if recording:
        query.append(("recording", "true"))
    if browser is not None:
        query.append(("browser", browser))
    return f"{url}?{urlencode(query)}" if query else url


def _header(headers: Mapping[str, str], name: str) -> str | None:
    wanted = name.lower()
    for key, value in headers.items():
        if key.lower() == wanted:
            return value
    return None


async def _read_response(response: BrowserResponse) -> bytes:
    try:
        return await response.read()
    finally:
        await response.aclose()


async def _json_response(response: BrowserResponse) -> object:
    data = await _read_response(response)
    try:
        return json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Browser Run returned invalid JSON") from error


def _target(value: object) -> BrowserTargetInfo | None:
    if not isinstance(value, Mapping) or not isinstance(value.get("id"), str):
        return None
    return BrowserTargetInfo(
        id=value["id"],
        type=value.get("type") if isinstance(value.get("type"), str) else None,
        url=value.get("url") if isinstance(value.get("url"), str) else None,
        title=value.get("title") if isinstance(value.get("title"), str) else None,
        description=(
            value.get("description")
            if isinstance(value.get("description"), str)
            else None
        ),
        devtools_frontend_url=(
            value.get("devtoolsFrontendUrl")
            if isinstance(value.get("devtoolsFrontendUrl"), str)
            else None
        ),
        websocket_debugger_url=(
            value.get("webSocketDebuggerUrl")
            if isinstance(value.get("webSocketDebuggerUrl"), str)
            else None
        ),
    )


def _session_info(payload: object) -> BrowserSessionInfo:
    if not isinstance(payload, Mapping) or not isinstance(
        payload.get("sessionId"), str
    ):
        raise ValueError("Browser Run response did not include a sessionId")
    session_id = payload["sessionId"]
    if not session_id:
        raise ValueError("Browser Run response did not include a sessionId")
    raw_targets = payload.get("targets")
    targets = ()
    if isinstance(raw_targets, list):
        targets = tuple(target for item in raw_targets if (target := _target(item)))
    websocket_url = payload.get("webSocketDebuggerUrl")
    return BrowserSessionInfo(
        session_id=session_id,
        targets=targets,
        websocket_debugger_url=websocket_url
        if isinstance(websocket_url, str)
        else None,
    )


async def create_browser_session(
    browser: BrowserTransport,
    *,
    keep_alive_ms: int | None = None,
    include_targets: bool = False,
    recording: bool = False,
) -> BrowserSessionInfo:
    """Create a reconnectable Chromium Browser Run session."""

    response = await browser.fetch(
        BrowserRequest(
            _endpoint(
                keep_alive_ms=keep_alive_ms,
                include_targets=include_targets,
                recording=recording,
            ),
            method="POST",
        )
    )
    if not 200 <= response.status < 300:
        status = response.status
        await _read_response(response)
        raise BrowserRenderingError(
            f"Failed to create Browser Run session: {status}", status
        )
    return _session_info(await _json_response(response))


async def list_browser_targets(
    browser: BrowserTransport, session_id: str
) -> tuple[BrowserTargetInfo, ...]:
    """List the current targets for a Browser Run session."""

    response = await browser.fetch(BrowserRequest(f"{_BASE}/{session_id}/json/list"))
    if not 200 <= response.status < 300:
        status = response.status
        await _read_response(response)
        raise BrowserRenderingError(
            f"Failed to list Browser Run targets for {session_id}: {status}", status
        )
    payload = await _json_response(response)
    if not isinstance(payload, list):
        return ()
    return tuple(target for item in payload if (target := _target(item)))


async def delete_browser_session(browser: BrowserTransport, session_id: str) -> None:
    """Delete a Browser Run session; an already-missing session is successful."""

    response = await browser.fetch(
        BrowserRequest(f"{_BASE}/{session_id}", method="DELETE")
    )
    status = response.status
    await _read_response(response)
    if not 200 <= status < 300 and status != 404:
        raise BrowserRenderingError(
            f"Failed to delete Browser Run session {session_id}: {status}", status
        )


async def _upgrade(
    browser: BrowserTransport,
    url: str,
    *,
    timeout_ms: int,
    session_id: str | None,
    on_close=None,
) -> CdpSession:
    response = await browser.fetch(
        BrowserRequest(url, headers={"Upgrade": "websocket"})
    )
    socket = response.websocket
    status = response.status
    await response.aclose()
    if socket is None:
        raise BrowserRenderingError(
            "Browser Run binding did not return a WebSocket; "
            "configure a browser binding",
            status,
        )
    socket.accept()
    return CdpSession(
        socket,
        default_timeout_ms=timeout_ms,
        session_id=session_id,
        on_close=on_close,
    )


async def connect_browser_session(
    browser: BrowserTransport,
    session_id: str,
    *,
    timeout_ms: int = 10_000,
) -> CdpSession:
    """Reconnect to an existing Browser Run session without owning it."""

    return await _upgrade(
        browser,
        _endpoint(session_id),
        timeout_ms=timeout_ms,
        session_id=session_id,
    )


async def connect_browser(
    browser: BrowserTransport,
    options: ConnectBrowserOptions | None = None,
) -> CdpSession:
    """Open a one-shot Browser Run connection and clean it up on close."""

    options = options or ConnectBrowserOptions()
    is_kitesurf = options.browser == "kitesurf"
    if is_kitesurf and (
        options.keep_alive_ms not in (None, 0)
        or options.include_targets
        or options.recording
    ):
        raise ValueError(
            "Kitesurf does not support keep_alive_ms, include_targets, or recording"
        )
    url = _endpoint(
        keep_alive_ms=None if is_kitesurf else options.keep_alive_ms,
        include_targets=False if is_kitesurf else options.include_targets,
        recording=False if is_kitesurf else options.recording,
        browser="kitesurf" if is_kitesurf else None,
    )
    response = await browser.fetch(
        BrowserRequest(url, headers={"Upgrade": "websocket"})
    )
    socket = response.websocket
    status = response.status
    session_id = _header(response.headers, "cf-browser-session-id")
    await response.aclose()
    if socket is None:
        raise BrowserRenderingError(
            "Browser Run binding did not return a WebSocket; "
            "configure a browser binding",
            status,
        )
    if not is_kitesurf and not session_id:
        closed = socket.close(1000, "Missing session id")
        if inspect.isawaitable(closed):
            await closed
        raise ValueError("Browser Run did not include cf-browser-session-id")
    socket.accept()

    async def dispose() -> None:
        if session_id is not None:
            await delete_browser_session(browser, session_id)

    return CdpSession(
        socket,
        default_timeout_ms=options.timeout_ms,
        session_id=None if is_kitesurf else session_id,
        on_close=None if is_kitesurf else dispose,
    )


async def get_browser_recording(
    transport: BrowserTransport,
    *,
    account_id: str,
    api_token: str,
    session_id: str,
) -> BrowserRecording:
    """Retrieve a finalized recording through Cloudflare's account API."""

    endpoint = (
        "https://api.cloudflare.com/client/v4/accounts/"
        f"{account_id}/browser-rendering/recording/{session_id}"
    )
    response = await transport.fetch(
        BrowserRequest(endpoint, headers={"Authorization": f"Bearer {api_token}"})
    )
    status = response.status
    content_type = _header(response.headers, "content-type") or "application/json"
    data = await _read_response(response)
    if not 200 <= status < 300:
        raise BrowserRenderingError(
            f"Failed to fetch Browser Run recording for {session_id}: {status}",
            status,
        )
    try:
        payload = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Browser Run recording response was not valid JSON") from error
    if isinstance(payload, Mapping) and isinstance(payload.get("result"), Mapping):
        payload = payload["result"]
    if not isinstance(payload, Mapping):
        raise ValueError("Browser Run recording response was not an object")
    returned_id = payload.get("sessionId")
    duration = payload.get("duration")
    raw_events = payload.get("events")
    if not isinstance(returned_id, str) or not isinstance(duration, int):
        raise ValueError("Browser Run recording response omitted metadata")
    events: dict[str, tuple[object, ...]] = {}
    if isinstance(raw_events, Mapping):
        for target_id, values in raw_events.items():
            if isinstance(target_id, str) and isinstance(values, list):
                events[target_id] = tuple(values)
    return BrowserRecording(returned_id, duration, events, data, content_type)
