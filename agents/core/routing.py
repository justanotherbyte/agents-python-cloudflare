from __future__ import annotations

import re

from js import Response as JsResponse  # ty: ignore[unresolved-import]
from workers import Request, Response

from .error import RoutingException
from .utils import url_path

CorsT = bool | dict[str, str] | None

_CAMEL_BOUNDARY_REGEX = re.compile(r"[A-Z]")
# Trailing segments remain in request.url for the Agent's own request routing.
_ROUTE_REGEX = re.compile(r"^/([^/]+)/([^/]+)/([^/]+)(?:/.*)?$")

# Mirrors the CORS defaults the client library ships with.
_DEFAULT_CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, HEAD, OPTIONS",
    "Access-Control-Allow-Headers": "*",
    "Access-Control-Max-Age": "86400",
}


def _is_ws_req(headers: dict[str, str]) -> bool:
    upgrade = headers.get("upgrade")
    connection = headers.get("connection")
    return bool(
        upgrade
        and connection
        and connection.lower() == "upgrade"
        and upgrade.lower() == "websocket"
    )


def _resolve_cors_headers(cors: CorsT) -> dict[str, str] | None:
    if cors is True:
        return dict(_DEFAULT_CORS_HEADERS)
    if isinstance(cors, dict):
        return dict(cors)
    return None


def _with_cors_headers(response: Response, headers: dict[str, str]) -> Response:
    js_resp = response.js_object
    # Rebuilding a 101 drops its socket handle, and Response rejects that status.
    if js_resp.status == 101:
        return response

    # Subrequest response headers are immutable, so adding CORS requires a copy.
    copied = JsResponse.new(js_resp.body, js_resp)
    for key, value in headers.items():
        copied.headers.set(key, value)
    return Response(copied)


def camel_to_kebab(string: str) -> str:
    # All-caps binding names and class names share this conversion contract.
    if string == string.upper() and string != string.lower():
        return string.lower().replace("_", "-")

    kebabified = _CAMEL_BOUNDARY_REGEX.sub(lambda m: f"-{m.group(0).lower()}", string)
    kebabified = kebabified.removeprefix("-")
    return re.sub(r"-$", "", kebabified.replace("_", "-"))


def kebab_to_screaming(string: str) -> str:
    return string.replace("-", "_").upper()


async def route_agent_request(
    request: Request,
    env,
    /,
    prefix: str = "agents",
    cors: CorsT = None,
) -> Response | None:
    """Route a request to the agent it addresses, or return None if it addresses none.

    The default prefix is what the TypeScript client uses, so changing it means
    passing the same `prefix` to `useAgent` on the client. Returning None lets an
    unmatched path fall through to the rest of your Worker.
    """
    if "/" in prefix:
        raise ValueError(f"prefix must be a single path segment, got {prefix!r}")

    match = _ROUTE_REGEX.match(url_path(request.url))
    if not match:
        return None

    path_prefix, agent, name = match.groups()
    if path_prefix != prefix:
        return None

    binding = kebab_to_screaming(agent)
    namespace = getattr(env, binding, None) or getattr(env, agent, None)
    if namespace is None:
        raise RoutingException("no namespace found")

    cors_headers = _resolve_cors_headers(cors)
    # Preflight should not instantiate the Durable Object once its binding is known.
    if cors_headers and request.method == "OPTIONS":
        return Response(None, headers=cors_headers)

    durable_id = namespace.idFromName(name)
    response = await namespace.get(durable_id).fetch(request)
    if not cors_headers or _is_ws_req(dict(request.headers)):
        return response
    return _with_cors_headers(response, cors_headers)
