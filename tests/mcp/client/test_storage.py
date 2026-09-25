from __future__ import annotations

import json
import sqlite3
from typing import Any

import pytest

from agents.mcp.client import (
    MCPServerRow,
    MCPServerStore,
    decode_server_options,
    encode_server_options,
    normalize_server_id,
    with_session,
)


class Sql:
    def __init__(self) -> None:
        self.connection = sqlite3.connect(":memory:")
        self.connection.row_factory = sqlite3.Row

    def execute(self, query: str, *params: object) -> list[dict[str, Any]]:
        cursor = self.connection.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]


def test_version_one_codec_persists_only_supported_options() -> None:
    encoded = encode_server_options(
        {
            "client": {
                "capabilities": {"elicitation": {"form": {}}},
                "supportedProtocolVersions": ["2025-06-18"],
                "jsonSchemaValidator": object(),
            },
            "transport": {
                "type": "streamable-http",
                "headers": {"Authorization": "Bearer x"},
                "fetch": object(),
            },
            "retry": {"maxAttempts": 4},
            "unknown": object(),
        }
    )

    assert json.loads(encoded) == {
        "client": {
            "capabilities": {"elicitation": {"form": {}}},
            "supportedProtocolVersions": ["2025-06-18"],
        },
        "transport": {
            "type": "streamable-http",
            "headers": {"Authorization": "Bearer x"},
        },
        "retry": {"maxAttempts": 4},
    }
    assert decode_server_options(encoded) == json.loads(encoded)


def test_codec_drops_an_unsafe_restored_session_and_rejects_non_finite_json() -> None:
    decoded = decode_server_options(
        '{"transport":{"sessionId":"old"},"discoverResult":{"tools":[]}}'
    )
    assert decoded == {"transport": {}}

    with pytest.raises(ValueError, match="JSON"):
        encode_server_options({"props": {"value": float("nan")}})


def test_session_projection_and_row_store_round_trip() -> None:
    options = with_session(
        {"transport": {"type": "streamable-http"}},
        "session-1",
        "2025-06-18",
        {"capabilities": {"tools": {}}},
    )
    sql = Sql()
    store = MCPServerStore(sql)
    store.prepare()
    row = MCPServerRow(
        "github",
        "GitHub",
        "https://mcp.example.com",
        "https://agent.example.com/callback",
        server_options=encode_server_options(options),
    )
    store.save(row)

    assert store.get("github") == row
    assert store.list() == (row,)
    assert decode_server_options(row.server_options)["transport"] == {
        "type": "streamable-http",
        "sessionId": "session-1",
        "protocolVersion": "2025-06-18",
    }

    assert store.migrate_id("github", "github-v2") is True
    assert store.get("github") is None
    assert store.get("github-v2") == MCPServerRow(
        "github-v2",
        "GitHub",
        "https://mcp.example.com",
        "https://agent.example.com/callback",
        server_options=row.server_options,
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("GitHub MCP!", "github-mcp"),
        ("42-things", "id-42-things"),
        ("!!!", "id"),
        ("__slack__", "slack"),
        ("foo---bar", "foo-bar"),
    ],
)
def test_normalize_server_id(value: str, expected: str) -> None:
    assert normalize_server_id(value) == expected
