from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class BrowserRequest:
    """A request for an injected Browser Run transport."""

    url: str
    method: str = "GET"
    headers: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class BrowserTargetInfo:
    """A debuggable browser target returned by Browser Run."""

    id: str
    type: str | None = None
    url: str | None = None
    title: str | None = None
    description: str | None = None
    devtools_frontend_url: str | None = None
    websocket_debugger_url: str | None = None


@dataclass(frozen=True)
class BrowserSessionInfo:
    """A Browser Run session and any targets included in its response."""

    session_id: str
    targets: tuple[BrowserTargetInfo, ...] = ()
    websocket_debugger_url: str | None = None


@dataclass(frozen=True)
class BrowserRecording:
    """A finalized rrweb recording with its exact response bytes."""

    session_id: str
    duration_ms: int
    events: Mapping[str, tuple[object, ...]]
    data: bytes
    content_type: str


@dataclass(frozen=True)
class ConnectBrowserOptions:
    """Options for a direct Browser Run CDP connection."""

    timeout_ms: int = 10_000
    keep_alive_ms: int | None = None
    include_targets: bool = False
    recording: bool = False
    browser: Literal["chromium", "kitesurf"] = "chromium"

    def __post_init__(self) -> None:
        if self.timeout_ms <= 0:
            raise ValueError("timeout_ms must be positive")
        if self.keep_alive_ms is not None and self.keep_alive_ms < 0:
            raise ValueError("keep_alive_ms must not be negative")
        if self.browser not in ("chromium", "kitesurf"):
            raise ValueError("browser must be chromium or kitesurf")


@dataclass(frozen=True)
class BrowserSessionOptions:
    """Durable browser lifecycle configuration."""

    mode: Literal["one-shot", "reuse", "dynamic"] = "one-shot"
    key: str = "default"
    keep_alive_ms: int | None = None
    recording: bool = False
    browser: Literal["chromium", "kitesurf"] = "chromium"

    def __post_init__(self) -> None:
        if self.mode not in ("one-shot", "reuse", "dynamic"):
            raise ValueError("browser session mode must be one-shot, reuse, or dynamic")
        if self.browser not in ("chromium", "kitesurf"):
            raise ValueError("browser must be chromium or kitesurf")
        if not self.key:
            raise ValueError("browser session key must not be empty")
        if self.keep_alive_ms is not None and self.keep_alive_ms < 0:
            raise ValueError("keep_alive_ms must not be negative")
        if self.browser != "kitesurf":
            return
        if self.mode != "one-shot":
            raise ValueError('Kitesurf only supports session mode "one-shot"')
        if self.keep_alive_ms not in (None, 0):
            raise ValueError("Kitesurf does not support keep_alive_ms")
        if self.recording:
            raise ValueError("Kitesurf does not support session recording")


@dataclass(frozen=True)
class BrowserLiveViewTarget:
    target_id: str
    url: str
    page_url: str | None = None
    title: str | None = None
    type: str | None = None


@dataclass(frozen=True)
class BrowserLiveView:
    session_id: str
    targets: tuple[BrowserLiveViewTarget, ...]
    expires_in_ms: int


@dataclass(frozen=True)
class BrowserLiveViewUrl:
    url: str
    target_id: str
    expires_in_ms: int


@dataclass(frozen=True)
class SweepEntry:
    key: str
    session_id: str
