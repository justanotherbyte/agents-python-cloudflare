from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, cast, overload

from .types import MCPRetryOptions, MCPServerRow

MCP_SCHEMA_VERSION_KEY = "cf_agents:mcp_schema_version"
CURRENT_MCP_SCHEMA_VERSION = 1

_CLIENT_KEYS = (
    "capabilities",
    "supportedProtocolVersions",
    "enforceStrictCapabilities",
    "debouncedNotificationMethods",
    "versionNegotiation",
    "inputRequired",
    "listMaxPages",
    "cachePartition",
    "defaultCacheTtlMs",
)
_TRANSPORT_KEYS = (
    "type",
    "headers",
    "requestInit",
    "reconnectionOptions",
    "skipIssuerMetadataValidation",
    "onInsufficientScope",
    "maxStepUpRetries",
    "sessionId",
    "protocolVersion",
)
_OPTION_KEYS = (
    "client",
    "transport",
    "discoverResult",
    "retry",
    "bindingName",
    "props",
    "capabilities",
)


class MCPKeyValueStorage(Protocol):
    @overload
    async def get(self, key: str) -> Any: ...

    @overload
    async def get(self, key: Sequence[str]) -> dict[str, Any]: ...

    @overload
    async def put(self, key: str, value: object) -> None: ...

    @overload
    async def put(self, key: dict[str, Any]) -> None: ...

    async def delete(self, key: str | Sequence[str]) -> bool | int: ...

    async def list(self, *, prefix: str = "") -> dict[str, Any]: ...


class MCPSql(Protocol):
    def execute(self, query: str, *params: object) -> list[dict[str, Any]]: ...


def ensure_mcp_server_table(sql: MCPSql) -> None:
    sql.execute(
        """
        CREATE TABLE IF NOT EXISTS cf_agents_mcp_servers (
          id TEXT PRIMARY KEY NOT NULL,
          name TEXT NOT NULL,
          server_url TEXT NOT NULL,
          callback_url TEXT NOT NULL,
          client_id TEXT,
          auth_url TEXT,
          server_options TEXT
        )
        """
    )


def _pick(value: object, keys: Sequence[str]) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    return {key: value[key] for key in keys if key in value}


def encode_server_options(options: Mapping[str, Any] | None) -> str:
    """Encode only the durable version-1 option subset using wire key names."""
    options = options or {}
    persisted: dict[str, Any] = {}
    for key in _OPTION_KEYS:
        if key not in options:
            continue
        value = options[key]
        if key == "client":
            value = _pick(value, _CLIENT_KEYS)
        elif key == "transport":
            value = _pick(value, _TRANSPORT_KEYS)
        if value is not None:
            persisted[key] = value
    return json.dumps(persisted, allow_nan=False, separators=(",", ":"))


def decode_server_options(value: str | None) -> dict[str, Any]:
    """Decode and normalize a persisted version-1 option record."""
    if not value:
        return {}
    parsed = json.loads(value, parse_constant=_reject_constant)
    if not isinstance(parsed, dict):
        raise ValueError("MCP server options must be a JSON object")
    result = {key: parsed[key] for key in _OPTION_KEYS if key in parsed}
    client = _pick(result.get("client"), _CLIENT_KEYS)
    transport = _pick(result.get("transport"), _TRANSPORT_KEYS)
    if client is not None:
        result["client"] = client
    else:
        result.pop("client", None)
    if transport is not None:
        stateless_without_prior = (
            transport.get("protocolVersion") == "2026-07-28"
            and "discoverResult" not in result
        )
        if transport.get("sessionId") and (
            not transport.get("protocolVersion") or stateless_without_prior
        ):
            transport.pop("sessionId", None)
            transport.pop("protocolVersion", None)
            result.pop("discoverResult", None)
        result["transport"] = transport
    else:
        result.pop("transport", None)
    return result


def retry_options(options: Mapping[str, Any]) -> MCPRetryOptions:
    value = options.get("retry")
    if not isinstance(value, Mapping):
        return MCPRetryOptions()
    return MCPRetryOptions(
        max_attempts=_bounded_int(value.get("maxAttempts"), 3, 1, 10),
        base_delay_ms=_bounded_int(value.get("baseDelayMs"), 500, 0, 60_000),
        max_delay_ms=_bounded_int(value.get("maxDelayMs"), 5_000, 0, 60_000),
    )


def with_session(
    options: Mapping[str, Any],
    session_id: str | None,
    protocol_version: str | None,
    discover_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result = dict(options)
    transport = dict(cast(Mapping[str, Any], result.get("transport") or {}))
    if session_id and protocol_version:
        transport["sessionId"] = session_id
        transport["protocolVersion"] = protocol_version
        if discover_result is not None:
            result["discoverResult"] = dict(discover_result)
        else:
            result.pop("discoverResult", None)
    else:
        transport.pop("sessionId", None)
        transport.pop("protocolVersion", None)
        result.pop("discoverResult", None)
    result["transport"] = transport
    return result


class MCPServerStore:
    """Own the shared MCP registration table's version-1 row codec."""

    def __init__(self, sql: MCPSql):
        self._sql = sql

    def prepare(self) -> None:
        ensure_mcp_server_table(self._sql)

    def list(self) -> tuple[MCPServerRow, ...]:
        rows = self._sql.execute(
            "SELECT id, name, server_url, callback_url, client_id, auth_url, "
            "server_options FROM cf_agents_mcp_servers ORDER BY rowid"
        )
        return tuple(_decode_row(row) for row in rows)

    def get(self, server_id: str) -> MCPServerRow | None:
        rows = self._sql.execute(
            "SELECT id, name, server_url, callback_url, client_id, auth_url, "
            "server_options FROM cf_agents_mcp_servers WHERE id = ?",
            server_id,
        )
        return _decode_row(rows[0]) if rows else None

    def save(self, row: MCPServerRow) -> None:
        self._sql.execute(
            "INSERT OR REPLACE INTO cf_agents_mcp_servers "
            "(id, name, server_url, callback_url, client_id, auth_url, "
            "server_options) VALUES (?, ?, ?, ?, ?, ?, ?)",
            row.id,
            row.name,
            row.server_url,
            row.callback_url,
            row.client_id,
            row.auth_url,
            row.server_options,
        )

    def remove(self, server_id: str) -> None:
        self._sql.execute("DELETE FROM cf_agents_mcp_servers WHERE id = ?", server_id)

    def migrate_id(self, old_id: str, new_id: str) -> bool:
        if old_id == new_id:
            return self.get(old_id) is not None
        if self.get(new_id) is not None:
            raise ValueError(f'MCP server id "{new_id}" is already in use')
        if self.get(old_id) is None:
            return False
        self._sql.execute(
            "UPDATE cf_agents_mcp_servers SET id = ? WHERE id = ?", new_id, old_id
        )
        return True


def _decode_row(row: Mapping[str, Any]) -> MCPServerRow:
    return MCPServerRow(
        id=str(row["id"]),
        name=str(row["name"]),
        server_url=str(row["server_url"]),
        callback_url=str(row["callback_url"]),
        client_id=None if row.get("client_id") is None else str(row["client_id"]),
        auth_url=None if row.get("auth_url") is None else str(row["auth_url"]),
        server_options=(
            None if row.get("server_options") is None else str(row["server_options"])
        ),
    )


def _bounded_int(value: object, default: int, minimum: int, maximum: int) -> int:
    if type(value) is not int:
        return default
    return max(minimum, min(maximum, value))


def _reject_constant(value: str) -> object:
    raise ValueError(f"invalid JSON constant: {value}")
