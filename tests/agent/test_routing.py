from __future__ import annotations

import types
from http import HTTPMethod
from typing import Any

import pytest
from workers import Request, Response

from agents import route_agent_request
from agents.core.error import RoutingException


DEFAULT_CORS_HEADERS = {
    "access-control-allow-origin": "*",
    "access-control-allow-methods": "GET, POST, HEAD, OPTIONS",
    "access-control-allow-headers": "*",
    "access-control-max-age": "86400",
}


class Stub:
    def __init__(
        self,
        response: Response | None = None,
        error: BaseException | None = None,
    ) -> None:
        self.response = response or Response("ok")
        self.error = error
        self.requests: list[Request] = []

    async def fetch(self, request: Request) -> Response:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return self.response


class Namespace:
    def __init__(self, stub: Stub) -> None:
        self.stub = stub
        self.names: list[str] = []
        self.ids: list[str] = []

    def idFromName(self, name: str) -> str:
        self.names.append(name)
        return f"id:{name}"

    def get(self, durable_id: str) -> Stub:
        self.ids.append(durable_id)
        return self.stub


class NoBindings:
    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"binding lookup was not expected: {name}")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path",
    ["/", "/agents", "/agents/my-agent", "/agents/my-agent/", "/other/x/y"],
)
async def test_unmatched_routes_fall_through_without_binding_lookup(path: str):
    request = Request(f"https://example.com{path}")

    assert await route_agent_request(request, NoBindings()) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("binding", ["MY_AGENT", "my-agent"])
async def test_route_resolves_supported_binding_names(binding: str):
    stub = Stub(Response("routed", status=202))
    namespace = Namespace(stub)
    env = types.SimpleNamespace()
    setattr(env, binding, namespace)
    request = Request("https://example.com/agents/my-agent/a;b/tail?q=1")

    response = await route_agent_request(request, env)

    assert response is not None
    assert response.status == 202
    assert namespace.names == ["a;b"]
    assert namespace.ids == ["id:a;b"]
    assert [forwarded.url for forwarded in stub.requests] == [request.url]


@pytest.mark.asyncio
async def test_route_supports_a_custom_single_segment_prefix():
    stub = Stub()
    namespace = Namespace(stub)
    env = types.SimpleNamespace(MY_AGENT=namespace)

    response = await route_agent_request(
        Request("https://example.com/custom/my-agent/one"),
        env,
        prefix="custom",
    )

    assert response is not None
    assert namespace.names == ["one"]


@pytest.mark.asyncio
async def test_route_preserves_request_method_headers_and_body():
    stub = Stub()
    namespace = Namespace(stub)
    request = Request(
        "https://example.com/agents/my-agent/one/tail?q=1",
        method=HTTPMethod.POST,
        headers={"X-Request": "yes"},
        body="payload",
    )

    await route_agent_request(
        request,
        types.SimpleNamespace(MY_AGENT=namespace),
    )

    forwarded = stub.requests[0]
    assert forwarded.url == request.url
    assert forwarded.method == HTTPMethod.POST
    assert dict(forwarded.headers) == {"x-request": "yes"}
    assert forwarded.body == "payload"


@pytest.mark.asyncio
async def test_route_rejects_a_multi_segment_prefix():
    with pytest.raises(ValueError, match="single path segment"):
        await route_agent_request(
            Request("https://example.com/agents/my-agent/one"),
            NoBindings(),
            prefix="api/agents",
        )


@pytest.mark.asyncio
async def test_matched_route_requires_a_namespace():
    with pytest.raises(RoutingException, match="no namespace found"):
        await route_agent_request(
            Request("https://example.com/agents/my-agent/one"),
            types.SimpleNamespace(),
        )


@pytest.mark.asyncio
async def test_routing_propagates_namespace_and_stub_failures():
    class BrokenNamespace:
        def idFromName(self, name: str) -> str:
            raise RuntimeError(f"cannot resolve {name}")

    request = Request("https://example.com/agents/my-agent/one")
    with pytest.raises(RuntimeError, match="cannot resolve one"):
        await route_agent_request(
            request,
            types.SimpleNamespace(MY_AGENT=BrokenNamespace()),
        )

    error = RuntimeError("stub failed")
    namespace = Namespace(Stub(error=error))
    with pytest.raises(RuntimeError, match="stub failed") as caught:
        await route_agent_request(
            request,
            types.SimpleNamespace(MY_AGENT=namespace),
        )
    assert caught.value is error


@pytest.mark.asyncio
@pytest.mark.parametrize("cors", [None, False, {}])
async def test_disabled_cors_preserves_the_upstream_response(cors):
    upstream = Response("body", status=206, headers={"X-Upstream": "yes"})
    namespace = Namespace(Stub(upstream))

    response = await route_agent_request(
        Request("https://example.com/agents/my-agent/one"),
        types.SimpleNamespace(MY_AGENT=namespace),
        cors=cors,
    )

    assert response is not None
    assert response.status == 206
    assert response.body == "body"
    assert dict(response.headers) == {"x-upstream": "yes"}


@pytest.mark.asyncio
async def test_default_cors_augments_an_http_response():
    upstream = Response(None, status=206, headers={"X-Upstream": "yes"})
    body = object()
    setattr(upstream.js_object, "body", body)
    namespace = Namespace(Stub(upstream))

    response = await route_agent_request(
        Request("https://example.com/agents/my-agent/one"),
        types.SimpleNamespace(MY_AGENT=namespace),
        cors=True,
    )

    assert response is not None
    assert response.status == 206
    assert response.body is body
    assert dict(response.headers) == {
        "x-upstream": "yes",
        **DEFAULT_CORS_HEADERS,
    }


@pytest.mark.asyncio
async def test_custom_cors_replaces_defaults_and_overwrites_matching_headers():
    upstream = Response(
        "body",
        headers={"aCcEsS-CoNtRoL-AlLoW-OrIgIn": "old", "X-Upstream": "yes"},
    )
    namespace = Namespace(Stub(upstream))

    response = await route_agent_request(
        Request("https://example.com/agents/my-agent/one"),
        types.SimpleNamespace(MY_AGENT=namespace),
        cors={
            "Access-Control-Allow-Origin": "https://client.example",
            "X-Custom": "set",
        },
    )

    assert response is not None
    assert dict(response.headers) == {
        "access-control-allow-origin": "https://client.example",
        "x-upstream": "yes",
        "x-custom": "set",
    }
    assert response.headers["ACCESS-Control-Allow-Origin"] == ("https://client.example")
    assert response.headers.get("X-CUSTOM") == "set"


@pytest.mark.asyncio
async def test_default_cors_preflight_resolves_namespace_without_waking_object():
    class SleepingNamespace:
        def idFromName(self, name: str) -> str:
            raise AssertionError("preflight must not resolve an object id")

    response = await route_agent_request(
        Request(
            "https://example.com/agents/my-agent/one",
            method=HTTPMethod.OPTIONS,
        ),
        types.SimpleNamespace(MY_AGENT=SleepingNamespace()),
        cors=True,
    )

    assert response is not None
    assert response.status == 200
    assert response.body is None
    assert dict(response.headers) == DEFAULT_CORS_HEADERS

    with pytest.raises(RoutingException, match="no namespace found"):
        await route_agent_request(
            Request(
                "https://example.com/agents/missing/one",
                method=HTTPMethod.OPTIONS,
            ),
            types.SimpleNamespace(),
            cors=True,
        )


@pytest.mark.asyncio
async def test_custom_cors_preflight_returns_only_configured_headers():
    class SleepingNamespace:
        def idFromName(self, name: str) -> str:
            raise AssertionError("preflight must not resolve an object id")

    response = await route_agent_request(
        Request(
            "https://example.com/agents/my-agent/one",
            method=HTTPMethod.OPTIONS,
        ),
        types.SimpleNamespace(MY_AGENT=SleepingNamespace()),
        cors={"Access-Control-Allow-Origin": "https://client.example"},
    )

    assert response is not None
    assert response.status == 200
    assert dict(response.headers) == {
        "access-control-allow-origin": "https://client.example"
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("cors", [None, False, {}])
async def test_options_reaches_the_object_when_cors_is_disabled(cors):
    stub = Stub(Response("handled", status=209))
    namespace = Namespace(stub)

    response = await route_agent_request(
        Request(
            "https://example.com/agents/my-agent/one",
            method=HTTPMethod.OPTIONS,
        ),
        types.SimpleNamespace(MY_AGENT=namespace),
        cors=cors,
    )

    assert response is not None
    assert response.status == 209
    assert len(stub.requests) == 1


@pytest.mark.asyncio
async def test_websocket_routing_preserves_the_upgrade_response():
    socket = object()
    upstream = Response(
        None,
        status=101,
        headers={"X-Upstream": "yes"},
        web_socket=socket,
    )
    namespace = Namespace(Stub(upstream))

    response = await route_agent_request(
        Request(
            "https://example.com/agents/my-agent/one",
            headers={"Connection": "UpGrAdE", "Upgrade": "WebSocket"},
        ),
        types.SimpleNamespace(MY_AGENT=namespace),
        cors=True,
    )

    assert response is not None
    assert response.status == 101
    assert getattr(response.js_object, "webSocket") is socket
    assert dict(response.headers) == {"x-upstream": "yes"}


@pytest.mark.asyncio
async def test_cors_never_rebuilds_a_101_response():
    socket = object()
    upstream = Response(None, status=101, web_socket=socket)
    namespace = Namespace(Stub(upstream))

    response = await route_agent_request(
        Request("https://example.com/agents/my-agent/one"),
        types.SimpleNamespace(MY_AGENT=namespace),
        cors=True,
    )

    assert response is not None
    assert response.status == 101
    assert getattr(response.js_object, "webSocket") is socket
