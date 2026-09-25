"""ResumableStream, driven with the sqlite-backed `sql` fixture.

These exercise the REAL buffer against a real in-memory database, so the
oversized-chunk skip, the live-vs-orphan replay terminals, and the
completed-vs-errored offer are checked against the source rather than a mock.
FakeConnection is the sink: it records the frames handed to send_if_open.
"""

from __future__ import annotations

import json

import fakes
import pytest

from agents import ChatMessageType
from agents.chat.resumable_stream import (
    ResumableStream,
    StreamStatus,
    StreamStorageUnavailable,
)


def _stream(sql) -> ResumableStream:
    stream = ResumableStream(sql, ChatMessageType.USE_CHAT_RESPONSE)
    stream.prepare()
    return stream


def _replay_chunks(frames):
    # The chunk frames, dropping the trailing replayComplete/terminal marker.
    return [f for f in frames if not f.get("replayComplete") and f.get("body")]


def test_start_marks_stream_active(sql):
    rs = _stream(sql)

    sid = rs.start("req-1", "msg-1")

    assert sid
    assert rs.has_active_stream() is True
    assert rs.active_request_id == "req-1"


def test_store_chunk_skips_oversized_without_advancing(sql):
    rs = _stream(sql)
    sid = rs.start("req-1", "msg-1")

    assert rs.store_chunk(sid, "chunk-a") is True
    assert rs.store_chunk(sid, "x" * 2_000_000) is False
    assert rs.store_chunk(sid, "chunk-b") is True

    conn = fakes.FakeConnection()
    rs.replay_active_chunks(conn, "req-1")

    chunks = _replay_chunks(conn.frames)
    assert [f["body"] for f in chunks] == ["chunk-a", "chunk-b"]


def test_live_replay_ends_with_replay_complete_not_done(sql):
    rs = _stream(sql)
    sid = rs.start("req-1", "msg-1")
    rs.store_chunk(sid, "a")
    rs.store_chunk(sid, "b")

    conn = fakes.FakeConnection()
    rs.replay_active_chunks(conn, "req-1")

    frames = conn.frames
    assert all(f["replay"] is True for f in frames)
    assert all(f["done"] is False for f in frames)
    assert frames[-1].get("replayComplete") is True
    # A live stream never gets a terminal done — the client rejoins the broadcast.
    assert not any(f["done"] is True for f in frames)
    # The stream stays active so the live tail keeps flowing.
    assert rs.has_active_stream() is True


def test_orphan_replay_finalizes_with_done(sql):
    rs1 = _stream(sql)
    sid = rs1.start("req-1", "msg-1")
    rs1.store_chunk(sid, "a")
    rs1.store_chunk(sid, "b")
    # Deliberately not completed: the row stays 'streaming' for rs2 to adopt.

    rs2 = _stream(sql)
    assert rs2.has_active_stream() is True
    assert rs2.active_request_id == "req-1"

    conn = fakes.FakeConnection()
    rs2.replay_active_chunks(conn, "req-1")

    frames = conn.frames
    assert [f["body"] for f in _replay_chunks(frames)] == ["a", "b"]

    terminal = frames[-1]
    assert terminal["done"] is True
    assert terminal["replay"] is True
    assert "replayComplete" not in terminal

    # The orphan is finalized, not left dangling.
    assert rs2.has_active_stream() is False


def test_completed_stream_replays_on_ack(sql):
    rs = _stream(sql)
    sid = rs.start("req-1", "msg-1")
    rs.store_chunk(sid, "a")
    rs.store_chunk(sid, "b")
    rs.complete(sid)

    conn = fakes.FakeConnection()
    assert rs.replay_completed_chunks(conn, "req-1") is True

    frames = conn.frames
    assert [f["body"] for f in _replay_chunks(frames)] == ["a", "b"]
    assert frames[-1]["done"] is True
    assert frames[-1]["replay"] is True


def test_errored_stream_not_offered(sql):
    rs = _stream(sql)
    sid = rs.start("req-2", "msg-2")
    rs.store_chunk(sid, "a")
    rs.mark_error(sid)

    conn = fakes.FakeConnection()
    assert rs.replay_completed_chunks(conn, "req-2") is False
    assert conn.frames == []


def test_clear_all_empties_tables_and_resets_active(sql):
    rs = _stream(sql)
    sid = rs.start("req-1", "msg-1")
    rs.store_chunk(sid, "a")
    rs.store_chunk(sid, "b")

    rs.clear_all()

    assert sql("SELECT COUNT(*) AS c FROM cf_ai_chat_stream_chunks")[0]["c"] == 0
    assert sql("SELECT COUNT(*) AS c FROM cf_ai_chat_stream_metadata")[0]["c"] == 0
    assert rs.has_active_stream() is False


def test_legacy_metadata_table_reconciles_missing_columns(sql):
    sql("""
    CREATE TABLE cf_ai_chat_stream_metadata (
        id TEXT PRIMARY KEY,
        request_id TEXT NOT NULL,
        status TEXT NOT NULL,
        created_at INTEGER NOT NULL,
        completed_at INTEGER
    )
    """)

    _stream(sql)

    columns = {
        row["name"] for row in sql("PRAGMA table_info(cf_ai_chat_stream_metadata)")
    }
    assert {"message_id", "is_continuation"} <= columns


def test_typescript_packed_chunk_rows_are_unpacked(sql):
    rs = _stream(sql)
    sid = rs.start("req-1", "msg-1")
    sql(
        "INSERT INTO cf_ai_chat_stream_chunks "
        "(id, stream_id, body, chunk_index, created_at) VALUES (?, ?, ?, ?, ?)",
        "packed",
        sid,
        json.dumps(["chunk-a", "chunk-b"]),
        0,
        1,
    )

    conn = fakes.FakeConnection()
    rs.replay_active_chunks(conn, "req-1")

    assert [frame["body"] for frame in _replay_chunks(conn.frames)] == [
        "chunk-a",
        "chunk-b",
    ]


def test_mixed_packed_and_legacy_rows_replay_in_logical_order(sql):
    rs = _stream(sql)
    stream_id = rs.start("request", "message")
    sql(
        "INSERT INTO cf_ai_chat_stream_chunks "
        "(id, stream_id, body, chunk_index, created_at) VALUES "
        "('legacy', ?, 'first', 0, 1), (?, ?, ?, 1, 2)",
        stream_id,
        "packed",
        stream_id,
        json.dumps(["second", "third"]),
    )

    connection = fakes.FakeConnection()
    rs.replay_active_chunks(connection, "request")

    assert [frame["body"] for frame in _replay_chunks(connection.frames)] == [
        "first",
        "second",
        "third",
    ]


def test_latest_stream_for_request_uses_rowid_to_break_timestamp_ties(sql):
    rs = _stream(sql)
    for stream_id, status in (("older", "completed"), ("newer", "streaming")):
        sql(
            "INSERT INTO cf_ai_chat_stream_metadata "
            "(id, request_id, status, created_at, message_id, is_continuation) "
            "VALUES (?, 'request', ?, 10, ?, 0)",
            stream_id,
            status,
            f"message-{stream_id}",
        )

    stream = rs.latest_for_request("request")

    assert stream is not None
    assert stream.stream_id == "newer"
    assert stream.status == "streaming"
    assert stream.message_id == "message-newer"


def test_latest_stream_for_request_returns_none_when_absent(sql):
    assert _stream(sql).latest_for_request("missing") is None


def test_latest_stream_for_request_rejects_unknown_status(sql):
    rs = _stream(sql)
    sql(
        "INSERT INTO cf_ai_chat_stream_metadata "
        "(id, request_id, status, created_at, is_continuation) "
        "VALUES ('bad-status', 'request', 'unknown', 1, 0)"
    )

    with pytest.raises(ValueError, match="unknown stream status"):
        rs.latest_for_request("request")


def test_strict_stream_read_reports_storage_failure(sql):
    failing = False

    def unstable_sql(query, *params):
        if failing:
            raise RuntimeError("storage offline")
        return sql(query, *params)

    rs = _stream(unstable_sql)
    failing = True

    with pytest.raises(StreamStorageUnavailable, match="stream storage read failed"):
        rs.latest_for_request("request")


def test_mark_orphaned_only_transitions_streaming_rows(sql):
    rs = _stream(sql)
    streaming_id = rs.start("streaming", "message-streaming")
    completed_id = rs.start("completed", "message-completed")
    rs.complete(completed_id)

    assert rs.mark_orphaned(streaming_id) is True
    assert rs.mark_orphaned(completed_id) is False
    statuses = {
        row["id"]: row["status"]
        for row in sql("SELECT id, status FROM cf_ai_chat_stream_metadata")
    }
    assert statuses == {streaming_id: "error", completed_id: "completed"}


def test_recovery_snapshot_has_exact_identity_and_ordered_decoded_bodies(sql):
    rs = _stream(sql)
    stream_id = rs.start("request", "target-message")
    sql(
        "INSERT INTO cf_ai_chat_stream_chunks "
        "(id, stream_id, body, chunk_index, created_at) VALUES (?, ?, ?, ?, ?)",
        "later-row",
        stream_id,
        json.dumps(["second", "third"]),
        1,
        2,
    )
    sql(
        "INSERT INTO cf_ai_chat_stream_chunks "
        "(id, stream_id, body, chunk_index, created_at) VALUES (?, ?, ?, ?, ?)",
        "earlier-row",
        stream_id,
        "first",
        0,
        1,
    )

    snapshot = rs.recovery_snapshot("request", "target-message")

    assert snapshot is not None
    assert snapshot.stream.stream_id == stream_id
    assert snapshot.stream.request_id == "request"
    assert snapshot.stream.message_id == "target-message"
    assert snapshot.bodies == ("first", "second", "third")


def test_completed_bodies_support_after_sequence(sql):
    rs = _stream(sql)
    stream_id = rs.start("request", "message")
    rs.store_chunk(stream_id, "zero")
    sql(
        "INSERT INTO cf_ai_chat_stream_chunks "
        "(id, stream_id, body, chunk_index, created_at) VALUES (?, ?, ?, ?, ?)",
        "packed",
        stream_id,
        json.dumps(["one", "two"]),
        1,
        2,
    )
    rs.complete(stream_id)

    assert rs.completed_bodies_for_request("request", after_sequence=0) == (
        "one",
        "two",
    )


def test_stream_status_is_a_string_enum(sql):
    rs = _stream(sql)
    rs.start("request", "message")
    stream = rs.latest_for_request("request")
    assert stream is not None
    assert stream.status is StreamStatus.STREAMING
    assert stream.status == "streaming"


def test_writes_ten_chunks_as_one_durable_segment(sql):
    rs = _stream(sql)
    stream_id = rs.start("request", "message")
    for index in range(11):
        rs.store_chunk(stream_id, f"chunk-{index}")

    rows = sql(
        "SELECT body FROM cf_ai_chat_stream_chunks "
        "WHERE stream_id = ? ORDER BY chunk_index",
        stream_id,
    )
    assert len(rows) == 2
    assert json.loads(rows[0]["body"]) == [f"chunk-{index}" for index in range(10)]
    assert rows[1]["body"] == "chunk-10"

    reincarnated = _stream(sql)
    connection = fakes.FakeConnection()
    reincarnated.replay_active_chunks(connection, "request")
    assert [frame["body"] for frame in _replay_chunks(connection.frames)] == [
        f"chunk-{index}" for index in range(11)
    ]


def test_segment_byte_limit_starts_a_new_row(sql):
    rs = _stream(sql)
    stream_id = rs.start("request", "message")
    rs.store_chunk(stream_id, "a" * 300_000)
    rs.store_chunk(stream_id, "b" * 300_000)

    rows = sql(
        "SELECT body FROM cf_ai_chat_stream_chunks "
        "WHERE stream_id = ? ORDER BY chunk_index",
        stream_id,
    )
    assert [len(row["body"]) for row in rows] == [300_000, 300_000]


def test_segment_byte_limit_accounts_for_json_escaping(sql):
    rs = _stream(sql)
    stream_id = rs.start("request", "message")
    quoted = '"' * 260_000
    rs.store_chunk(stream_id, quoted)
    rs.store_chunk(stream_id, quoted)

    rows = sql(
        "SELECT body FROM cf_ai_chat_stream_chunks "
        "WHERE stream_id = ? ORDER BY chunk_index",
        stream_id,
    )
    assert len(rows) == 2


def test_replay_keyset_paginates_duplicate_chunk_indexes(sql):
    rs = _stream(sql)
    stream_id = rs.start("request", "message")
    for index in range(205):
        sql(
            "INSERT INTO cf_ai_chat_stream_chunks "
            "(id, stream_id, body, chunk_index, created_at) VALUES (?, ?, ?, 0, ?)",
            f"row-{index}",
            stream_id,
            f"body-{index}",
            index,
        )

    connection = fakes.FakeConnection()
    rs.replay_active_chunks(connection, "request")
    assert [frame["body"] for frame in _replay_chunks(connection.frames)] == [
        f"body-{index}" for index in range(205)
    ]


def test_recovery_snapshot_stops_at_chunk_limit(sql):
    rs = _stream(sql)
    stream_id = rs.start("request", "message")
    for body in ("one", "two", "three"):
        rs.store_chunk(stream_id, body)

    snapshot = rs.recovery_snapshot(
        "request",
        "message",
        max_chunks=2,
        max_bytes=100,
    )
    assert snapshot is not None
    assert snapshot.limit_exceeded is True
    assert snapshot.bodies == ("one", "two")


def test_recovery_snapshot_stops_at_utf8_byte_limit(sql):
    rs = _stream(sql)
    stream_id = rs.start("request", "message")
    rs.store_chunk(stream_id, "££")

    snapshot = rs.recovery_snapshot(
        "request",
        "message",
        max_chunks=10,
        max_bytes=3,
    )
    assert snapshot is not None
    assert snapshot.limit_exceeded is True


def test_latest_terminal_error_is_single_sql_slot(sql):
    rs = _stream(sql)
    rs.record_terminal_error("first", "one")
    rs.record_terminal_error("second", "two")

    terminal = _stream(sql).latest_terminal_error()
    assert terminal is not None
    assert terminal.request_id == "second"
    assert terminal.body == "two"

    rs.clear_terminal_error()
    assert rs.latest_terminal_error() is None


def test_errored_stream_replays_chunks_without_synthesizing_terminal(sql):
    rs = _stream(sql)
    stream_id = rs.start("request", "message")
    rs.store_chunk(stream_id, "partial")
    rs.mark_error(stream_id)

    connection = fakes.FakeConnection()
    assert rs.replay_error_chunks(connection, "request") is True
    assert [frame["body"] for frame in connection.frames] == ["partial"]
    assert all(frame["done"] is False for frame in connection.frames)


def test_replay_read_failure_sends_an_error_terminal(sql):
    fail_chunk_reads = False

    def unstable_sql(query, *params):
        if fail_chunk_reads and "SELECT rowid AS row_id" in query:
            raise RuntimeError("storage offline")
        return sql(query, *params)

    rs = _stream(unstable_sql)
    stream_id = rs.start("request", "message")
    rs.store_chunk(stream_id, "chunk")
    rs.complete(stream_id)
    fail_chunk_reads = True

    connection = fakes.FakeConnection()
    assert rs.replay_completed_chunks(connection, "request") is True
    assert connection.frames == [
        {
            "type": ChatMessageType.USE_CHAT_RESPONSE,
            "id": "request",
            "body": "Unable to replay the retained chat stream",
            "done": True,
            "error": True,
        }
    ]


def test_live_replay_read_failure_leaves_the_producer_registered(sql):
    fail_chunk_reads = False

    def unstable_sql(query, *params):
        if fail_chunk_reads and "SELECT rowid AS row_id" in query:
            raise RuntimeError("storage offline")
        return sql(query, *params)

    rs = _stream(unstable_sql)
    stream_id = rs.start("request", "message")
    rs.store_chunk(stream_id, "chunk")
    fail_chunk_reads = True

    connection = fakes.FakeConnection()
    rs.replay_active_chunks(connection, "request")

    assert rs.has_active_stream() is True
    assert rs.active_request_id == "request"
    assert connection.frames[-1]["error"] is True


def test_adopted_replay_read_failure_stops_offering_the_stream(sql):
    fail_chunk_reads = False

    def unstable_sql(query, *params):
        if fail_chunk_reads and "SELECT rowid AS row_id" in query:
            raise RuntimeError("storage offline")
        return sql(query, *params)

    stream_id = _stream(unstable_sql).start("request", "message")
    fail_chunk_reads = True
    reincarnated = _stream(unstable_sql)

    connection = fakes.FakeConnection()
    reincarnated.replay_active_chunks(connection, "request")

    assert reincarnated.has_active_stream() is False
    assert sql(
        "SELECT status FROM cf_ai_chat_stream_metadata WHERE id = ?", stream_id
    ) == [{"status": "error"}]
    assert connection.frames[-1]["error"] is True


def test_cleanup_deadline_and_sweep_cover_completed_and_abandoned(sql):
    rs = _stream(sql)
    sql(
        "INSERT INTO cf_ai_chat_stream_metadata "
        "(id, request_id, status, created_at, completed_at) "
        "VALUES ('completed', 'done', 'completed', 1, 100), "
        "('abandoned', 'old', 'streaming', 200, NULL)"
    )
    sql(
        "INSERT INTO cf_ai_chat_stream_chunks "
        "(id, stream_id, body, chunk_index, created_at) "
        "VALUES ('completed-chunk', 'completed', 'done', 0, 100), "
        "('abandoned-chunk', 'abandoned', 'old', 0, 200)"
    )

    assert rs.next_cleanup_deadline() == 100 + 10 * 60 * 1000
    rs.cleanup(200 + 60 * 60 * 1000 + 1)

    assert sql("SELECT id FROM cf_ai_chat_stream_metadata") == []
    assert sql("SELECT id FROM cf_ai_chat_stream_chunks") == []


def test_cleanup_never_reaps_the_live_active_stream(sql):
    rs = _stream(sql)
    stream_id = rs.start("request", "message")
    sql(
        "UPDATE cf_ai_chat_stream_metadata SET created_at = 1 WHERE id = ?",
        stream_id,
    )

    rs.cleanup(1 + 2 * 60 * 60 * 1000)

    assert sql(
        "SELECT id FROM cf_ai_chat_stream_metadata WHERE id = ?",
        stream_id,
    ) == [{"id": stream_id}]


def test_continuation_marker_survives_reincarnation_and_replay(sql):
    stream = _stream(sql)
    stream_id = stream.start("request", "message", continuation=True)
    stream.store_chunk(stream_id, "partial")

    reincarnated = _stream(sql)
    connection = fakes.FakeConnection()
    reincarnated.replay_active_chunks(connection, "request")

    assert connection.frames
    assert all(frame["continuation"] is True for frame in connection.frames)


def test_terminal_error_retains_continuation_marker(sql):
    stream = _stream(sql)
    stream.record_terminal_error("request", "failed", continuation=True)

    terminal = _stream(sql).latest_terminal_error()

    assert terminal is not None
    assert terminal.is_continuation is True
