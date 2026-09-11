from __future__ import annotations

from typing import Any, cast

import fakes
import pytest
from workers import DurableObject, Request, Response

from agents.lifecycle import (
    CapabilityRequestContext,
    CapabilityWebSocketCloseContext,
    CapabilityWebSocketErrorContext,
    CapabilityWebSocketMessageContext,
    CapabilityWebSocketUpgradeContext,
    Lifecycle,
    LifecycleCapability,
)


class RecordingCapability(LifecycleCapability):
    capability_id = "recording"

    def __init__(
        self,
        name: str,
        events: list[str],
        results: dict[str, object],
    ):
        self.name = name
        self.events = events
        self.results = results
        self.received: list[tuple[str, object]] = []

    def _result(self, phase: str, context: object) -> Any:
        self.events.append(f"{self.name}:{phase}")
        self.received.append((phase, context))
        result = self.results.get(phase)
        if isinstance(result, BaseException):
            raise result
        return result

    async def on_start(self) -> None:
        self.events.append(f"{self.name}:start")
        result = self.results.get("start")
        if isinstance(result, BaseException):
            raise result

    async def on_request(self, context: object) -> Response | None:
        return self._result("request", context)

    async def on_websocket_upgrade(self, context: object) -> Response | None:
        return self._result("upgrade", context)

    async def on_websocket_message(self, context: object) -> bool:
        return self._result("message", context)

    async def on_websocket_close(self, context: object) -> bool:
        return self._result("close", context)

    async def on_websocket_error(self, context: object) -> bool:
        return self._result("error", context)


def recording_capability(
    capability_id: str,
    events: list[str],
    results: dict[str, object],
) -> RecordingCapability:
    capability_type = type(
        f"{capability_id.title()}Capability",
        (RecordingCapability,),
        {"capability_id": capability_id},
    )
    return capability_type(capability_id, events, results)


def phase_arguments(phase: str) -> tuple[object, ...]:
    if phase in {"request", "upgrade"}:
        return (Request("https://example.com/"),)
    if phase == "message":
        return (object(), "message")
    if phase == "close":
        return (object(), 1000, "done", True)
    return (object(), RuntimeError("socket failed"))


def expected_context(phase: str, arguments: tuple[object, ...]) -> object:
    if phase == "request":
        return CapabilityRequestContext(cast(Request, arguments[0]))
    if phase == "upgrade":
        return CapabilityWebSocketUpgradeContext(cast(Request, arguments[0]))
    if phase == "message":
        return CapabilityWebSocketMessageContext(arguments[0], arguments[1])
    if phase == "close":
        return CapabilityWebSocketCloseContext(
            arguments[0],
            cast(int, arguments[1]),
            cast(str, arguments[2]),
            cast(bool, arguments[3]),
        )
    return CapabilityWebSocketErrorContext(arguments[0], arguments[1])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method_name", "phase"),
    [("fetch", "request"), ("websocket_upgrade", "upgrade")],
)
async def test_response_phases_return_first_response_unchanged(
    method_name: str,
    phase: str,
):
    events: list[str] = []
    claimed = Response("claimed", status=209)
    fallback_response = Response("fallback")

    async def host(_context: object) -> Response:
        events.append("host")
        return Response("host")

    lifecycle = Lifecycle(
        fakes.FakeCtx(),
        host=object(),
        on_request=host,
        on_websocket_upgrade=host,
    )
    first = recording_capability("first", events, {phase: object()})
    lifecycle.use(first)
    lifecycle.use(recording_capability("second", events, {phase: claimed}))
    lifecycle.use(
        recording_capability("fallback", events, {phase: fallback_response}),
        fallback=True,
    )

    arguments = phase_arguments(phase)
    result = await getattr(lifecycle, method_name)(*arguments)

    assert result is claimed
    assert first.received == [(phase, expected_context(phase, arguments))]
    assert events == [
        "first:start",
        "second:start",
        "fallback:start",
        "first:" + phase,
        "second:" + phase,
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method_name", "phase"),
    [("fetch", "request"), ("websocket_upgrade", "upgrade")],
)
async def test_response_phases_forward_to_host_when_unclaimed(
    method_name: str,
    phase: str,
):
    events: list[str] = []
    host_response = Response("host")

    async def host(_context: object) -> Response:
        events.append("host")
        return host_response

    lifecycle = Lifecycle(
        fakes.FakeCtx(),
        host=object(),
        on_request=host,
        on_websocket_upgrade=host,
    )
    lifecycle.use(recording_capability("normal", events, {phase: None}))
    lifecycle.use(
        recording_capability("fallback", events, {phase: object()}),
        fallback=True,
    )

    result = await getattr(lifecycle, method_name)(*phase_arguments(phase))

    assert result is host_response
    assert events[-3:] == ["normal:" + phase, "fallback:" + phase, "host"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method_name", "phase"),
    [("fetch", "request"), ("websocket_upgrade", "upgrade")],
)
async def test_response_phases_allow_fallback_to_claim(
    method_name: str,
    phase: str,
):
    events: list[str] = []
    claimed = Response("fallback")

    async def host(_context: object) -> Response:
        events.append("host")
        return Response("host")

    lifecycle = Lifecycle(
        fakes.FakeCtx(),
        host=object(),
        on_request=host,
        on_websocket_upgrade=host,
    )
    lifecycle.use(recording_capability("normal", events, {phase: object()}))
    lifecycle.use(
        recording_capability("fallback", events, {phase: claimed}),
        fallback=True,
    )
    lifecycle.use(
        recording_capability("later", events, {phase: Response("later")}),
        fallback=True,
    )

    result = await getattr(lifecycle, method_name)(*phase_arguments(phase))

    assert result is claimed
    assert events[-2:] == ["normal:" + phase, "fallback:" + phase]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method_name", "phase"),
    [
        ("websocket_message", "message"),
        ("websocket_close", "close"),
        ("websocket_error", "error"),
    ],
)
async def test_socket_phases_stop_only_for_literal_true(
    method_name: str,
    phase: str,
):
    events: list[str] = []

    async def host(_context: object) -> bool:
        events.append("host")
        return True

    lifecycle = Lifecycle(
        fakes.FakeCtx(),
        host=object(),
        on_websocket_message=host,
        on_websocket_close=host,
        on_websocket_error=host,
    )
    lifecycle.use(recording_capability("none", events, {phase: None}))
    lifecycle.use(recording_capability("false", events, {phase: False}))
    lifecycle.use(recording_capability("truthy", events, {phase: 1}))
    lifecycle.use(
        recording_capability("consumer", events, {phase: True}),
        fallback=True,
    )
    lifecycle.use(
        recording_capability("later", events, {phase: True}),
        fallback=True,
    )

    result = await getattr(lifecycle, method_name)(*phase_arguments(phase))

    assert result is True
    assert events[-4:] == [
        "none:" + phase,
        "false:" + phase,
        "truthy:" + phase,
        "consumer:" + phase,
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method_name", "phase"),
    [
        ("fetch", "request"),
        ("websocket_upgrade", "upgrade"),
        ("websocket_message", "message"),
        ("websocket_close", "close"),
        ("websocket_error", "error"),
    ],
)
async def test_phase_error_stops_dispatch(method_name: str, phase: str):
    events: list[str] = []
    failure = RuntimeError("phase failed")

    async def response_host(_context: object) -> Response:
        events.append("host")
        return Response("host")

    async def socket_host(_context: object) -> bool:
        events.append("host")
        return False

    lifecycle = Lifecycle(
        fakes.FakeCtx(),
        host=object(),
        on_request=response_host,
        on_websocket_upgrade=response_host,
        on_websocket_message=socket_host,
        on_websocket_close=socket_host,
        on_websocket_error=socket_host,
    )
    lifecycle.use(recording_capability("throwing", events, {phase: failure}))
    lifecycle.use(recording_capability("later", events, {phase: True}))

    with pytest.raises(RuntimeError, match="phase failed"):
        await getattr(lifecycle, method_name)(*phase_arguments(phase))

    assert events[-1] == "throwing:" + phase


class PlainPhaseHost(DurableObject):
    def __init__(self, ctx: Any, env: Any):
        super().__init__(ctx, env)
        self.events: list[str] = []
        self.lifecycle = Lifecycle(
            ctx,
            host=self,
            on_start=self._start,
            on_request=self._response,
            on_websocket_upgrade=self._response,
            on_websocket_message=self._socket,
            on_websocket_close=self._socket,
            on_websocket_error=self._socket,
        )
        self.capability = recording_capability("feature", self.events, {})
        self.lifecycle.use(self.capability)

    async def _start(self) -> None:
        self.events.append("host:start")

    async def _response(self, _context: object) -> Response:
        self.events.append("host:response")
        return Response("host")

    async def _socket(self, _context: object) -> bool:
        self.events.append("host:socket")
        return False

    async def fetch(self, request: Request) -> Response:
        return await self.lifecycle.fetch(request)

    async def webSocketMessage(self, websocket: object, message: object) -> bool:
        return await self.lifecycle.websocket_message(websocket, message)

    async def webSocketClose(
        self,
        websocket: object,
        code: int,
        reason: str,
        was_clean: bool,
    ) -> bool:
        return await self.lifecycle.websocket_close(websocket, code, reason, was_clean)

    async def webSocketError(self, websocket: object, error: object) -> bool:
        return await self.lifecycle.websocket_error(websocket, error)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method_name", "phase", "host_event", "arguments"),
    [
        (
            "fetch",
            "request",
            "host:response",
            (Request("https://example.com/"),),
        ),
        (
            "fetch",
            "upgrade",
            "host:response",
            (Request("https://example.com/", headers={"Upgrade": "websocket"}),),
        ),
        ("webSocketMessage", "message", "host:socket", (object(), "message")),
        (
            "webSocketClose",
            "close",
            "host:socket",
            (object(), 1000, "done", True),
        ),
        (
            "webSocketError",
            "error",
            "host:socket",
            (object(), RuntimeError("socket failed")),
        ),
    ],
)
async def test_plain_durable_object_forwards_every_i2_entry(
    method_name: str,
    phase: str,
    host_event: str,
    arguments: tuple[object, ...],
):
    host = PlainPhaseHost(fakes.FakeCtx(), object())

    await getattr(host, method_name)(*arguments)

    assert PlainPhaseHost.__bases__ == (DurableObject,)
    assert host.capability.received == [(phase, expected_context(phase, arguments))]
    assert host.events == [
        "feature:start",
        "host:start",
        f"feature:{phase}",
        host_event,
    ]
