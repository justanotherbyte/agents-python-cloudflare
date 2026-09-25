"""The chat turn: _handle_use_chat_request, _run_turn, _emit, the terminals.

Live chunks go out as broadcast envelopes shaped
``{"type": USE_CHAT_RESPONSE, "id": <request_id>, "body": <json string>, "done": ...}``,
so a chunk is inspected by decoding the envelope's body. Only connections in
``agent._connections`` receive a broadcast, so the requester is registered there.
"""

from __future__ import annotations

import asyncio
import json

import fakes
import pytest
from workers import Request, Response

from agents import AIChatAgent, ChatMessageType

_MESSAGES = [{"id": "m1", "role": "user", "parts": [{"type": "text", "text": "hi"}]}]


def _use_chat_request(request_id: str, body: str) -> dict:
    return {"id": request_id, "init": {"method": "POST", "body": body}}


def _valid_body() -> str:
    return json.dumps({"messages": _MESSAGES})


def _responses(conn, request_id: str) -> list[dict]:
    return [
        f
        for f in conn.frames
        if f.get("type") == ChatMessageType.USE_CHAT_RESPONSE
        and f.get("id") == request_id
    ]


def _inner_chunks(conn, request_id: str) -> list[dict]:
    return [
        json.loads(f["body"])
        for f in _responses(conn, request_id)
        if f["done"] is False
    ]


def _terminals(conn, request_id: str) -> list[dict]:
    return [f for f in _responses(conn, request_id) if f["done"] is True]


@pytest.mark.asyncio
async def test_successful_turn_emits_start_chunks_and_clean_terminal():
    agent = fakes.build_chat_agent(lambda o: "hello")
    conn = fakes.FakeConnection(id="req")
    agent._connections[conn.id] = conn

    await agent._handle_use_chat_request(conn, _use_chat_request("r1", _valid_body()))

    chunks = _inner_chunks(conn, "r1")
    assert any(c["type"] == "start" for c in chunks)
    assert any(c["type"].startswith("text") for c in chunks)

    terminals = _terminals(conn, "r1")
    assert len(terminals) == 1
    assert terminals[0]["body"] == ""
    assert "error" not in terminals[0]

    assistant = next(m for m in agent.messages if m.get("role") == "assistant")
    texts = [
        p.get("text") for p in assistant.get("parts", []) if p.get("type") == "text"
    ]
    assert "hello" in texts


@pytest.mark.asyncio
async def test_cancel_emits_exactly_one_terminal():
    started = asyncio.Event()
    release = asyncio.Event()

    async def provider(_options):
        async def chunks():
            yield "partial"
            started.set()
            await release.wait()

        return chunks()

    agent = fakes.build_chat_agent(provider)
    connection = fakes.FakeConnection()
    agent._connections[connection.id] = connection
    request = asyncio.create_task(
        agent._handle_use_chat_request(
            connection,
            _use_chat_request("cancelled", _valid_body()),
        )
    )
    await started.wait()

    await agent._handle_cancel({"id": "cancelled"})
    release.set()
    await asyncio.gather(request, return_exceptions=True)

    assert len(_terminals(connection, "cancelled")) == 1
    assert agent._resumable.has_active_stream() is False


@pytest.mark.asyncio
async def test_opt_in_provider_stall_watchdog_terminalizes_the_turn():
    async def provider(_options):
        async def chunks():
            await asyncio.Event().wait()
            yield "unreachable"

        return chunks()

    agent = fakes.build_chat_agent(provider)
    agent.chat_provider_stall_timeout_ms = 1
    connection = fakes.FakeConnection()
    agent._connections[connection.id] = connection

    await agent._handle_use_chat_request(
        connection,
        _use_chat_request("stalled", _valid_body()),
    )

    assert len(_terminals(connection, "stalled")) == 1
    assert _terminals(connection, "stalled")[0]["body"] == (
        "Chat provider stream stalled"
    )


@pytest.mark.asyncio
async def test_error_mid_turn_broadcasts_error_terminal_and_does_not_offer_resume():
    def boom(_options):
        raise RuntimeError("boom")

    agent = fakes.build_chat_agent(boom)
    conn = fakes.FakeConnection(id="req")
    agent._connections[conn.id] = conn

    # Must not raise out: on_error re-raises the same object, which _report_error
    # swallows once it has already gone through the notification channel.
    await agent._handle_use_chat_request(conn, _use_chat_request("r1", _valid_body()))

    terminals = _terminals(conn, "r1")
    assert len(terminals) == 1
    assert terminals[0]["error"] is True
    assert terminals[0]["body"] == "boom"

    # Marking the stream errored keeps it from being offered for resume.
    assert agent._resumable.has_active_stream() is False


@pytest.mark.asyncio
async def test_streamed_error_chunk_terminalizes_without_successful_finish():
    agent = fakes.build_chat_agent(
        lambda _options: [
            {"type": "text-delta", "id": "text", "delta": "partial"},
            {"type": "error", "errorText": "provider failed"},
            {"type": "text-delta", "id": "text", "delta": "ignored"},
        ]
    )
    conn = fakes.FakeConnection(id="req")
    agent._connections[conn.id] = conn

    await agent._handle_use_chat_request(conn, _use_chat_request("r1", _valid_body()))

    chunks = _inner_chunks(conn, "r1")
    assert [
        chunk.get("delta") for chunk in chunks if chunk["type"] == "text-delta"
    ] == ["partial"]
    assert not any(chunk["type"] == "finish" for chunk in chunks)
    assert _terminals(conn, "r1") == [
        {
            "type": ChatMessageType.USE_CHAT_RESPONSE,
            "id": "r1",
            "body": "provider failed",
            "done": True,
            "error": True,
        }
    ]


@pytest.mark.asyncio
async def test_unstorable_chunk_terminalizes_before_broadcast():
    agent = fakes.build_chat_agent(
        lambda _options: [
            {
                "type": "file",
                "mediaType": "text/plain",
                "url": "x" * 2_000_000,
            }
        ]
    )
    connection = fakes.FakeConnection(id="requester")
    agent._connections[connection.id] = connection

    await agent._handle_use_chat_request(
        connection,
        _use_chat_request("r1", _valid_body()),
    )

    assert not any(chunk["type"] == "file" for chunk in _inner_chunks(connection, "r1"))
    terminal = _terminals(connection, "r1")[-1]
    assert terminal["error"] is True
    assert terminal["body"] == "stream chunk could not be persisted"


@pytest.mark.asyncio
async def test_provider_replay_is_neither_buffered_nor_broadcast():
    agent = fakes.build_chat_agent(
        lambda _options: [
            {
                "type": "tool-input-available",
                "toolCallId": "call",
                "toolName": "search",
                "input": {"query": "one"},
            },
            {
                "type": "tool-output-available",
                "toolCallId": "call",
                "output": "result",
            },
            {
                "type": "tool-input-start",
                "toolCallId": "call",
                "toolName": "search",
            },
        ]
    )
    conn = fakes.FakeConnection(id="req")
    agent._connections[conn.id] = conn

    await agent._handle_use_chat_request(conn, _use_chat_request("r1", _valid_body()))

    chunks = _inner_chunks(conn, "r1")
    assert [chunk["type"] for chunk in chunks].count("tool-input-start") == 0

    replay = fakes.FakeConnection()
    assert agent._resumable.replay_completed_chunks(replay, "r1") is True
    replayed = [
        json.loads(frame["body"])
        for frame in replay.frames
        if frame.get("body") and frame.get("done") is False
    ]
    assert [chunk["type"] for chunk in replayed].count("tool-input-start") == 0


@pytest.mark.asyncio
async def test_resume_probe_waits_for_an_accepted_pre_stream_turn():
    agent = fakes.build_chat_agent(lambda _options: "hello")
    submitter = fakes.FakeConnection(id="submitter")
    reconnecting = fakes.FakeConnection(id="reconnecting")
    agent._connections[submitter.id] = submitter
    agent._connections[reconnecting.id] = reconnecting
    accepted = asyncio.Event()
    release = asyncio.Event()
    enqueue = agent._turn_queue.enqueue

    async def delayed_enqueue(request_id, run):
        accepted.set()
        await release.wait()
        return await enqueue(request_id, run)

    agent._turn_queue.enqueue = delayed_enqueue
    turn = asyncio.create_task(
        agent._handle_use_chat_request(
            submitter,
            _use_chat_request("r1", _valid_body()),
        )
    )
    await accepted.wait()

    agent._handle_resume_request(reconnecting, {"probeId": "probe"})
    assert reconnecting.frames == [
        {
            "type": ChatMessageType.STREAM_PENDING,
            "id": "r1",
            "probeId": "probe",
        }
    ]

    release.set()
    await turn
    resuming = next(
        frame
        for frame in reconnecting.frames
        if frame.get("type") == ChatMessageType.STREAM_RESUMING
    )
    assert resuming == {
        "type": ChatMessageType.STREAM_RESUMING,
        "id": "r1",
    }
    assert reconnecting.id in agent._pending_resume_connections


@pytest.mark.asyncio
async def test_terminal_error_is_offered_and_replayed_to_multiple_tabs():
    def boom(_options):
        raise RuntimeError("boom")

    agent = fakes.build_chat_agent(boom)
    submitter = fakes.FakeConnection(id="submitter")
    agent._connections[submitter.id] = submitter
    await agent._handle_use_chat_request(
        submitter,
        _use_chat_request("r1", _valid_body()),
    )

    for connection_id in ("one", "two"):
        reconnecting = fakes.FakeConnection(id=connection_id)
        agent._handle_resume_request(reconnecting, {"probeId": connection_id})
        assert reconnecting.frames == [
            {
                "type": ChatMessageType.STREAM_RESUMING,
                "id": "r1",
                "probeId": connection_id,
            }
        ]

        agent._handle_resume_ack(reconnecting, {"id": "r1"})
        terminal = reconnecting.frames[-1]
        assert terminal == {
            "type": ChatMessageType.USE_CHAT_RESPONSE,
            "id": "r1",
            "body": "boom",
            "done": True,
            "error": True,
        }


@pytest.mark.asyncio
async def test_parked_probe_transitions_to_terminal_replay_on_setup_failure():
    agent = fakes.build_chat_agent(lambda _options: "unused")
    submitter = fakes.FakeConnection(id="submitter")
    reconnecting = fakes.FakeConnection(id="reconnecting")
    agent._connections[submitter.id] = submitter
    agent._connections[reconnecting.id] = reconnecting
    accepted = asyncio.Event()
    release = asyncio.Event()
    enqueue = agent._turn_queue.enqueue

    async def delayed_enqueue(request_id, run):
        accepted.set()
        await release.wait()
        return await enqueue(request_id, run)

    agent._turn_queue.enqueue = delayed_enqueue
    turn = asyncio.create_task(
        agent._handle_use_chat_request(
            submitter,
            _use_chat_request("r1", "not json"),
        )
    )
    await accepted.wait()
    agent._handle_resume_request(reconnecting, {"probeId": "probe"})
    assert reconnecting.frames[-1]["type"] == ChatMessageType.STREAM_PENDING

    release.set()
    await turn
    assert reconnecting.frames[-1] == {
        "type": ChatMessageType.STREAM_RESUMING,
        "id": "r1",
    }


@pytest.mark.asyncio
async def test_chat_clear_releases_parked_probe_with_its_probe_id():
    agent = fakes.build_chat_agent(lambda _options: "unused")
    submitter = fakes.FakeConnection(id="submitter")
    reconnecting = fakes.FakeConnection(id="reconnecting")
    agent._connections[submitter.id] = submitter
    agent._connections[reconnecting.id] = reconnecting
    accepted = asyncio.Event()
    release = asyncio.Event()
    enqueue = agent._turn_queue.enqueue

    async def delayed_enqueue(request_id, run):
        accepted.set()
        await release.wait()
        return await enqueue(request_id, run)

    agent._turn_queue.enqueue = delayed_enqueue
    turn = asyncio.create_task(
        agent._handle_use_chat_request(
            submitter,
            _use_chat_request("r1", _valid_body()),
        )
    )
    await accepted.wait()
    agent._handle_resume_request(reconnecting, {"probeId": "probe"})

    await agent._handle_chat_clear(submitter)
    assert reconnecting.frames[-2:] == [
        {
            "type": ChatMessageType.STREAM_RESUME_NONE,
            "reason": "idle",
            "probeId": "probe",
        },
        {"type": ChatMessageType.CHAT_CLEAR},
    ]

    release.set()
    await turn


@pytest.mark.asyncio
async def test_new_accepted_turn_clears_previous_terminal_error():
    responses = iter((RuntimeError("first failed"), "recovered"))

    def reply(_options):
        value = next(responses)
        if isinstance(value, Exception):
            raise value
        return value

    agent = fakes.build_chat_agent(reply)
    connection = fakes.FakeConnection(id="requester")
    agent._connections[connection.id] = connection
    await agent._handle_use_chat_request(
        connection,
        _use_chat_request("first", _valid_body()),
    )
    assert agent._resumable.latest_terminal_error() is not None

    await agent._handle_use_chat_request(
        connection,
        _use_chat_request("second", _valid_body()),
    )
    assert agent._resumable.latest_terminal_error() is None


@pytest.mark.asyncio
async def test_queued_success_clears_an_older_turn_error_recorded_after_accept():
    first_started = asyncio.Event()
    release_first = asyncio.Event()

    async def reply(options):
        if options.request_id == "first":
            first_started.set()
            await release_first.wait()
            raise RuntimeError("first failed")
        return "second succeeded"

    agent = fakes.build_chat_agent(reply)
    connection = fakes.FakeConnection(id="requester")
    agent._connections[connection.id] = connection
    first = asyncio.create_task(
        agent._handle_use_chat_request(
            connection,
            _use_chat_request("first", _valid_body()),
        )
    )
    await first_started.wait()
    second = asyncio.create_task(
        agent._handle_use_chat_request(
            connection,
            _use_chat_request("second", _valid_body()),
        )
    )
    await asyncio.sleep(0)

    release_first.set()
    await first
    await second

    assert agent._resumable.latest_terminal_error() is None


@pytest.mark.asyncio
async def test_chat_stream_cleanup_uses_the_shared_alarm_slot():
    agent = fakes.build_chat_agent(lambda _options: "hello")
    stream_id = agent._resumable.start("request", "message")
    agent._resumable.store_chunk(stream_id, "chunk")
    agent._resumable.complete(stream_id)

    deadline = agent._resumable.next_cleanup_deadline()
    await agent._schedule_next_alarm()
    assert deadline is not None
    assert agent.ctx.storage.alarm_time_ms == deadline

    agent.sql(
        "UPDATE cf_ai_chat_stream_metadata SET completed_at = 1 WHERE id = ?",
        stream_id,
    )
    await agent._alarm_housekeeping()
    assert (
        agent.sql(
            "SELECT id FROM cf_ai_chat_stream_metadata WHERE id = ?",
            stream_id,
        )
        == []
    )
    assert (
        agent.sql(
            "SELECT id FROM cf_ai_chat_stream_chunks WHERE stream_id = ?",
            stream_id,
        )
        == []
    )


@pytest.mark.asyncio
async def test_pre_stream_failure_sends_unicast_terminal():
    agent = fakes.build_chat_agent(lambda o: "hi")
    conn = fakes.FakeConnection(id="req")
    agent._connections[conn.id] = conn

    # A body that fails json parsing raises inside _start_turn, before the stream
    # is registered, so the terminal is unicast to the requester.
    await agent._handle_use_chat_request(conn, _use_chat_request("r1", "not json"))

    terminals = _terminals(conn, "r1")
    assert len(terminals) == 1
    assert terminals[0]["error"] is True
    assert isinstance(terminals[0]["body"], str) and terminals[0]["body"]
    # Nothing was broadcast: the failure predates the stream, so no chunk exists.
    assert agent._resumable.has_active_stream() is False


@pytest.mark.asyncio
async def test_durable_turn_settles_when_stream_metadata_cannot_be_written():
    class DurableChat(AIChatAgent):
        durable_chat_recovery = True

        async def on_chat_message(self, options):
            return "unused"

    agent = fakes.build_chat_agent(cls=DurableChat)
    connection = fakes.FakeConnection(id="requester")
    agent._connections[connection.id] = connection
    sql = agent._resumable._sql

    def unstable_sql(query, *params):
        if query.startswith("INSERT INTO cf_ai_chat_stream_metadata"):
            raise RuntimeError("metadata unavailable")
        return sql(query, *params)

    agent._resumable._sql = unstable_sql

    await agent._handle_use_chat_request(
        connection,
        _use_chat_request("r1", _valid_body()),
    )

    terminal = _terminals(connection, "r1")[-1]
    assert terminal["error"] is True
    assert terminal["body"] == "stream metadata could not be persisted"


@pytest.mark.asyncio
async def test_clear_invalidates_an_active_turn_after_provider_await():
    release = asyncio.Event()

    async def reply(_options):
        await release.wait()
        return "late answer"

    agent = fakes.build_chat_agent(reply)
    conn = fakes.FakeConnection(id="req")
    agent._connections[conn.id] = conn
    turn = asyncio.create_task(
        agent._handle_use_chat_request(conn, _use_chat_request("r1", _valid_body()))
    )
    await asyncio.sleep(0)

    await agent._handle_chat_clear(conn)
    release.set()
    await turn

    assert agent.messages == []
    assert agent._resumable.has_active_stream() is False
    assert agent.sql("SELECT * FROM cf_ai_chat_stream_chunks") == []
    assert len(_terminals(conn, "r1")) == 1


@pytest.mark.asyncio
async def test_clear_while_async_reply_finishes_does_not_resurrect_transcript():
    waiting = asyncio.Event()
    release = asyncio.Event()

    async def reply(_options):
        yield {"type": "text-start", "id": "text"}
        waiting.set()
        await release.wait()

    agent = fakes.build_chat_agent(reply)
    conn = fakes.FakeConnection(id="req")
    agent._connections[conn.id] = conn
    turn = asyncio.create_task(
        agent._handle_use_chat_request(conn, _use_chat_request("r1", _valid_body()))
    )
    await waiting.wait()

    await agent._handle_chat_clear(conn)
    release.set()
    await turn

    assert agent.messages == []
    assert agent.sql("SELECT * FROM cf_agents_session_messages") == []
    assert agent.sql("SELECT * FROM cf_ai_chat_stream_chunks") == []
    assert len(_terminals(conn, "r1")) == 1


@pytest.mark.asyncio
async def test_emit_excludes_pending_resume_and_stores_before_broadcast():
    agent = fakes.build_chat_agent(lambda o: "hi")
    conn_a = fakes.FakeConnection(id="A")
    conn_b = fakes.FakeConnection(id="B")
    agent._connections[conn_a.id] = conn_a
    agent._connections[conn_b.id] = conn_b

    # B is mid-resume, so it must be held out of the live broadcast until it ACKs.
    agent._pending_resume_connections.add(conn_b.id)

    await agent._handle_use_chat_request(conn_a, _use_chat_request("r1", _valid_body()))

    assert _responses(conn_a, "r1")  # the requester saw the live chunks
    assert _responses(conn_b, "r1") == []  # the pending-resume one did not

    # The buffered prefix is real: a fresh connection can replay the completed
    # stream, which only succeeds if the chunks were stored before broadcast.
    conn_c = fakes.FakeConnection(id="C")
    assert agent._resumable.replay_completed_chunks(conn_c, "r1") is True


@pytest.mark.asyncio
async def test_new_stream_rejoins_connection_that_never_acked_previous_offer():
    agent = fakes.build_chat_agent(lambda _options: "hi")
    submitter = fakes.FakeConnection(id="submitter")
    reconnecting = fakes.FakeConnection(id="reconnecting")
    agent._connections[submitter.id] = submitter
    agent._connections[reconnecting.id] = reconnecting
    agent._pending_resume_connections.add(reconnecting.id)
    agent._pending_resume_requests[reconnecting.id] = "old-request"

    await agent._handle_use_chat_request(
        submitter,
        _use_chat_request("new-request", _valid_body()),
    )

    assert reconnecting.id not in agent._pending_resume_connections
    assert reconnecting.id not in agent._pending_resume_requests
    assert _responses(reconnecting, "new-request")


@pytest.mark.asyncio
async def test_resume_ack_replays_a_completed_stream_while_another_is_active():
    agent = fakes.build_chat_agent(lambda o: "hi")
    conn = fakes.FakeConnection(id="C")

    # A completed, retained stream the client is resuming, and a different turn that
    # has since become the active one. complete() clears active, start() re-sets it.
    rs = agent._resumable
    sid_a = rs.start("rA", "mA")
    rs.store_chunk(sid_a, "a1")
    rs.complete(sid_a)
    rs.start("rB", "mB")
    assert rs.active_request_id == "rB"

    agent._pending_resume_connections.add(conn.id)
    agent._handle_resume_ack(conn, {"id": "rA"})

    # A different active stream must not swallow the ACK: rA's buffer is replayed and
    # terminated so the client settles instead of waiting out its probe timeout.
    replayed = _responses(conn, "rA")
    assert [f["body"] for f in replayed if f["done"] is False] == ["a1"]
    terminals = _terminals(conn, "rA")
    assert len(terminals) == 1
    assert terminals[0]["replay"] is True
    assert conn.id not in agent._pending_resume_connections


@pytest.mark.asyncio
async def test_request_override_cannot_skip_messages_endpoint():
    class UserChat(AIChatAgent):
        def __init__(self, ctx, env):
            super().__init__(ctx, env)
            self.requests = []

        async def on_chat_message(self, options):
            return ""

        def on_request(self, request):
            self.requests.append(request.url)
            return Response("user", status=201)

    agent = fakes.build_chat_agent(cls=UserChat)

    messages = await agent._dispatch_request(
        Request("https://example.com/agents/test/get-messages")
    )
    user = await agent._dispatch_request(Request("https://example.com/custom"))

    assert messages.status == 200
    assert agent.requests == ["https://example.com/custom"]
    assert user.status == 201


@pytest.mark.asyncio
async def test_close_override_cannot_skip_resume_cleanup():
    class UserChat(AIChatAgent):
        def __init__(self, ctx, env):
            super().__init__(ctx, env)
            self.closed = []

        async def on_chat_message(self, options):
            return ""

        def on_close(self, connection, code, reason, was_clean):
            self.closed.append(connection.id)

    agent = fakes.build_chat_agent(cls=UserChat)
    connection = fakes.FakeConnection()
    agent._pending_resume_connections.add(connection.id)

    await agent._dispatch_close(connection, 1000, "done", True)

    assert connection.id not in agent._pending_resume_connections
    assert agent.closed == [connection.id]
