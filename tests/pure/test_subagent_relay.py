"""Pure operation-log validation, accounting, application, and cleanup tests."""

from __future__ import annotations

import asyncio

import pytest

from agents.core.subagent_relay import (
    BufferedRelayConnection,
    CloseOperation,
    RelayLimitError,
    RelayLimits,
    RelaySession,
    SendOperation,
    StateOperation,
    TagsOperation,
    apply_operations,
    decode_operation_log,
    relay_round_trip,
)
from agents.core.utils import dumps_wire


class Sink:
    def __init__(self, *, send_succeeds: bool = True):
        self.send_succeeds = send_succeeds
        self.events = []

    def send_if_open(self, data):
        self.events.append(("send", data))
        return self.send_succeeds

    def _set_raw_state(self, data):
        self.events.append(("state", data))

    def _insert_tags(self, tags):
        self.events.append(("tags", tags))

    def close(self, code=1000, reason=""):
        self.events.append(("close", code, reason))


def test_typed_operations_round_trip_through_wire_log():
    raw = dumps_wire(
        [
            {"type": "send", "data": "hello"},
            {"type": "state", "data": {"step": 1}},
            {"type": "tags", "data": ["one", "two"]},
            {"type": "close", "code": 1000, "reason": "done"},
        ]
    )

    operations = decode_operation_log(raw, RelayLimits(4, len(raw.encode("utf-8"))))

    assert operations == [
        SendOperation("hello"),
        StateOperation({"step": 1}),
        TagsOperation(("one", "two")),
        CloseOperation(1000, "done"),
    ]


@pytest.mark.parametrize(
    "operation",
    [
        {"type": "unknown"},
        {"type": "send", "data": 1},
        {"type": "state"},
        {"type": "tags", "data": ["valid", 1]},
        {"type": "close", "code": True, "reason": "bad"},
        {"type": "close", "code": 1000},
    ],
)
def test_invalid_operation_shapes_are_rejected(operation):
    with pytest.raises(TypeError):
        decode_operation_log(dumps_wire([operation]), RelayLimits(10, 10_000))


def test_operation_after_close_is_rejected():
    raw = dumps_wire(
        [
            {"type": "close", "code": 1000, "reason": "done"},
            {"type": "send", "data": "late"},
        ]
    )

    with pytest.raises(TypeError, match="follows close"):
        decode_operation_log(raw, RelayLimits(10, 10_000))


def test_utf8_byte_limit_counts_the_complete_serialized_list():
    raw = dumps_wire([{"type": "send", "data": "snowman: \u2603"}])
    exact_bytes = len(raw.encode("utf-8"))

    assert decode_operation_log(raw, RelayLimits(1, exact_bytes))
    with pytest.raises(RelayLimitError, match="byte limit"):
        decode_operation_log(raw, RelayLimits(1, exact_bytes - 1))

    connection = BufferedRelayConnection(
        "connection",
        state={},
        tags=[],
        max_frames=1,
        max_bytes=exact_bytes,
    )
    connection.send("snowman: \u2603")
    assert connection.operation_log() == raw


def test_decoder_enforces_exact_frame_limit():
    raw = dumps_wire(
        [
            {"type": "send", "data": "one"},
            {"type": "send", "data": "two"},
        ]
    )

    assert len(decode_operation_log(raw, RelayLimits(2, len(raw)))) == 2
    with pytest.raises(RelayLimitError, match="frame limit"):
        decode_operation_log(raw, RelayLimits(1, len(raw)))


def test_producer_rejects_invalid_tags_and_close_values():
    connection = BufferedRelayConnection(
        "connection", state={}, tags=[], max_frames=10, max_bytes=10_000
    )

    with pytest.raises(TypeError, match="tags must be strings"):
        connection._insert_tags([1])
    with pytest.raises(TypeError, match="close code must be an integer"):
        connection.close(True)


def test_producer_copies_recorded_state():
    connection = BufferedRelayConnection(
        "connection", state={}, tags=[], max_frames=10, max_bytes=10_000
    )
    state = {"step": 1}

    connection.set_state(state)
    state["step"] = 2

    assert connection.state == {"step": 1}
    assert connection.operation_log() == dumps_wire(
        [{"type": "state", "data": {"step": 1}}]
    )


def test_producer_preserves_internal_flags_in_raw_state_only():
    connection = BufferedRelayConnection(
        "connection",
        state={"_cf_readonly": True},
        tags=[],
        max_frames=10,
        max_bytes=10_000,
    )

    assert connection.state is None
    assert connection.set_state({"device": "sensor-1"}) == {"device": "sensor-1"}
    assert connection.state == {"device": "sensor-1"}
    assert connection.operation_log() == dumps_wire(
        [
            {
                "type": "state",
                "data": {"device": "sensor-1", "_cf_readonly": True},
            }
        ]
    )


@pytest.mark.parametrize("state", ["ready", 3, ["one", "two"]])
def test_producer_rejects_non_object_state_when_internal_flags_are_set(state):
    connection = BufferedRelayConnection(
        "connection",
        state={"_cf_readonly": True},
        tags=[],
        max_frames=10,
        max_bytes=10_000,
    )

    with pytest.raises(TypeError, match="must be an object or null"):
        connection.set_state(state)

    assert connection._get_raw_state() == {"_cf_readonly": True}
    assert connection.operation_log() == "[]"


@pytest.mark.parametrize("state", [None, "ready", 3, ["one", "two"]])
def test_producer_preserves_generic_json_state(state):
    connection = BufferedRelayConnection(
        "connection", state={}, tags=[], max_frames=10, max_bytes=10_000
    )

    assert connection.set_state(state) == state
    assert connection.state == state
    assert connection.operation_log() == dumps_wire([{"type": "state", "data": state}])


def test_producer_state_updater_receives_and_returns_snapshots():
    connection = BufferedRelayConnection(
        "connection", state=["before"], tags=[], max_frames=10, max_bytes=10_000
    )

    committed = connection.set_state(lambda current: [*current, "after"])
    committed.append("caller")

    assert connection.state == ["before", "after"]


def test_failed_recording_rolls_back_state_tags_and_close():
    state_connection = BufferedRelayConnection(
        "state", state={"old": True}, tags=["old"], max_frames=1, max_bytes=2
    )
    with pytest.raises(RelayLimitError):
        state_connection.set_state({"new": True})
    assert state_connection.state == {"old": True}

    tags_connection = BufferedRelayConnection(
        "tags", state={}, tags=["old"], max_frames=1, max_bytes=2
    )
    with pytest.raises(RelayLimitError):
        tags_connection._insert_tags(["new"])
    assert tags_connection.tags == ["old"]

    close_connection = BufferedRelayConnection(
        "close", state={}, tags=[], max_frames=1, max_bytes=2
    )
    with pytest.raises(RelayLimitError):
        close_connection.close()
    assert close_connection._closed is False


def test_close_is_idempotent_and_rejects_later_mutation():
    connection = BufferedRelayConnection(
        "connection", state={}, tags=[], max_frames=10, max_bytes=10_000
    )
    connection.close(1000, "done")
    connection.close(1001, "ignored")

    assert connection.operation_log() == dumps_wire(
        [{"type": "close", "code": 1000, "reason": "done"}]
    )
    assert connection.send_if_open("late") is False
    with pytest.raises(RuntimeError, match="after close"):
        connection.set_state({"late": True})
    with pytest.raises(RuntimeError, match="after close"):
        connection._insert_tags(["late"])


def test_send_failure_stops_later_operations():
    sink = Sink(send_succeeds=False)

    apply_operations(
        sink,
        [SendOperation("lost"), StateOperation({"must": "not apply"})],
    )

    assert sink.events == [("send", "lost")]


def test_session_cleanup_removes_only_the_registered_instance():
    connections = {}
    first = BufferedRelayConnection(
        "same", state={}, tags=[], max_frames=10, max_bytes=10_000
    )
    replacement = BufferedRelayConnection(
        "same", state={}, tags=[], max_frames=10, max_bytes=10_000
    )

    with RelaySession(connections, first):
        connections["same"] = replacement

    assert connections == {"same": replacement}


def test_sessions_with_duplicate_public_ids_use_distinct_relay_keys():
    connections = {}
    first = BufferedRelayConnection(
        "same", state={}, tags=[], max_frames=10, max_bytes=10_000
    )
    second = BufferedRelayConnection(
        "same", state={}, tags=[], max_frames=10, max_bytes=10_000
    )

    with RelaySession(connections, first, key="physical-1"):
        with RelaySession(connections, second, key="physical-2"):
            assert connections == {"physical-1": first, "physical-2": second}
        assert connections == {"physical-1": first}

    assert connections == {}


def test_session_cleanup_failure_does_not_mask_handler_failure():
    class BrokenConnections(dict):
        def pop(self, key, default=None):
            raise RuntimeError("cleanup failed")

    connections = BrokenConnections()
    connection = BufferedRelayConnection(
        "connection", state={}, tags=[], max_frames=10, max_bytes=10_000
    )

    with pytest.raises(ValueError, match="handler failed"):
        with RelaySession(connections, connection):
            raise ValueError("handler failed")


@pytest.mark.asyncio
async def test_round_trip_validates_whole_log_before_application():
    sink = Sink()
    raw = dumps_wire(
        [
            {"type": "send", "data": "must not send"},
            {"type": "unknown"},
        ]
    )

    async def forward():
        return raw

    with pytest.raises(TypeError):
        await relay_round_trip(forward, sink, RelayLimits(10, 10_000), timeout=1)

    assert sink.events == [("close", 1011, "Sub-agent relay failed")]


@pytest.mark.asyncio
async def test_round_trip_preserves_timeout_when_failure_close_raises():
    class BrokenCloseSink(Sink):
        def close(self, code=1000, reason=""):
            raise RuntimeError("close failed")

    async def forward():
        await asyncio.Event().wait()

    with pytest.raises(TimeoutError):
        await relay_round_trip(
            forward,
            BrokenCloseSink(),
            RelayLimits(10, 10_000),
            timeout=0.001,
        )
