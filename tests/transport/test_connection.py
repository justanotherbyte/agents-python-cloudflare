"""Transport surface: Connection.send_if_open, broadcast, StreamingResponse,
attachment_id, and Agent.sql.

These exercise a real Connection over a FakeSocket, and a real Agent over a
FakeCtx-backed sqlite, so the swallow-only-"after close"
rule, the broadcast exclude semantics, the single-terminal StreamingResponse
contract, and the positional sql() pass-through are checked against the source
rather than a mock that agrees with it by construction.
"""

from __future__ import annotations

import json

import fakes
import pytest

from agents import Agent, StreamingResponse
from agents.lifecycle.websockets import Connection, attachment_id

# ── Connection.send_if_open ──────────────────────────────────────────────────


def test_send_if_open_delivers_dict_and_str(fake_socket):
    sock = fake_socket()
    conn = Connection("id1", sock)

    assert conn.send_if_open({"a": 1}) is True
    assert sock.sent == [json.dumps({"a": 1})]

    assert conn.send_if_open("raw") is True
    assert sock.sent[-1] == "raw"


def test_send_if_open_swallows_after_close(fake_socket):
    sock = fake_socket()
    conn = Connection("id1", sock)

    sock.close()
    assert conn.send_if_open({"b": 2}) is False
    assert sock.sent == []


def test_send_if_open_propagates_other_errors():
    class RaisingSocket:
        def send(self, data: str) -> None:
            raise ValueError("boom")

    conn = Connection("id1", RaisingSocket())

    with pytest.raises(ValueError, match="boom"):
        conn.send_if_open("hello")


# ── broadcast / broadcast_json ───────────────────────────────────────────────


def _agent() -> Agent:
    return fakes.build_agent()


def test_broadcast_reaches_all_and_skips_closed():
    server = _agent()
    closed = fakes.FakeConnection(id="closed", open=False)
    c1 = fakes.FakeConnection(id="c1")
    c2 = fakes.FakeConnection(id="c2")
    for conn in (closed, c1, c2):
        server._connections[conn.id] = conn

    server.broadcast_json({"hello": "world"})

    assert c1.frames == [{"hello": "world"}]
    assert c2.frames == [{"hello": "world"}]
    assert closed.frames == []


def test_broadcast_honours_exclude():
    server = _agent()
    c1 = fakes.FakeConnection(id="c1")
    c2 = fakes.FakeConnection(id="c2")
    server._connections[c1.id] = c1
    server._connections[c2.id] = c2

    server.broadcast_json({"n": 1}, exclude=(c1.id,))

    assert c1.frames == []
    assert c2.frames == [{"n": 1}]


def test_broadcast_materializes_generator_exclude():
    server = _agent()
    c1 = fakes.FakeConnection(id="c1")
    c2 = fakes.FakeConnection(id="c2")
    server._connections[c1.id] = c1
    server._connections[c2.id] = c2

    server.broadcast_json({"n": 2}, exclude=(x for x in [c1.id]))

    assert c1.frames == []
    assert c2.frames == [{"n": 2}]


# ── StreamingResponse ────────────────────────────────────────────────────────


def test_streaming_response_send_chunk(fake_connection):
    sr = StreamingResponse(fake_connection, "rpc-1")

    assert sr.send({"x": 1}) is True
    frame = fake_connection.frames[-1]
    assert frame["type"] == "rpc"
    assert frame["id"] == "rpc-1"
    assert frame["success"] is True
    assert frame["result"] == {"x": 1}
    assert frame["done"] is False


def test_streaming_response_single_terminal(fake_connection):
    sr = StreamingResponse(fake_connection, "rpc-1")

    assert sr.end({"final": True}) is True
    frame = fake_connection.frames[-1]
    assert frame["done"] is True
    assert frame["result"] == {"final": True}

    sent_after_first = len(fake_connection.sent)
    assert sr.end("again") is False
    assert len(fake_connection.sent) == sent_after_first


def test_streaming_response_error_on_fresh_stream(fake_connection):
    sr = StreamingResponse(fake_connection, "rpc-1")

    assert sr.error("nope") is True
    frame = fake_connection.frames[-1]
    assert frame["type"] == "rpc"
    assert frame["success"] is False
    assert frame["error"] == "nope"
    assert "done" not in frame
    assert "result" not in frame


def test_streaming_response_send_after_close_warns(fake_connection):
    sr = StreamingResponse(fake_connection, "rpc-1")
    sr.end()
    sent_after_end = len(fake_connection.sent)

    with pytest.warns(UserWarning):
        result = sr.send({"late": True})

    assert result is False
    assert len(fake_connection.sent) == sent_after_end


# ── attachment_id ────────────────────────────────────────────────────────────


def test_attachment_id_none_when_unset(fake_socket):
    assert attachment_id(fake_socket(attachment=None)) is None


def test_attachment_id_reads_string_id(fake_socket):
    sock = fake_socket(attachment={"id": "abc", "state": {}, "tags": []})
    assert attachment_id(sock) == "abc"


def test_attachment_id_none_for_non_string_id(fake_socket):
    assert attachment_id(fake_socket(attachment={"id": 123})) is None


def test_set_state_preserves_optional_attachment_metadata(fake_socket):
    socket = fake_socket(
        attachment={
            "id": "abc",
            "state": {},
            "tags": [],
            "target": {"url": "https://example.com/sub", "headers": {}},
        }
    )
    connection = Connection("abc", socket)

    connection.set_state({"ready": True})

    attachment = socket.deserializeAttachment().to_py()
    assert attachment["__user"] == {"ready": True}
    assert attachment["__pk"] == {"id": "abc", "tags": ["abc"]}
    assert attachment["target"]["url"].endswith("/sub")


# ── Agent.sql ────────────────────────────────────────────────────────────────


def test_sql_roundtrips_and_passes_params_positionally():
    server = _agent()

    server.sql("CREATE TABLE t (a TEXT, b INTEGER)")
    server.sql("INSERT INTO t VALUES (?, ?)", "x", 1)

    assert server.sql("SELECT a, b FROM t") == [{"a": "x", "b": 1}]
    assert server.sql("SELECT ? AS v", 5) == [{"v": 5}]
