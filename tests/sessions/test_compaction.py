from __future__ import annotations

import asyncio
import re
from datetime import UTC, datetime

import fakes
import pytest

import agents
from agents.lifecycle import Lifecycle, LifecycleEvent
from agents.sessions import (
    AppendOptions,
    CompactOptions,
    CompactResult,
    HistoryBatchReadOptions,
    HistoryReadOptions,
    SearchResult,
    Sessions,
    StoredCompaction,
    create_compact_function,
)


async def _started(*, events=()):
    ctx = fakes.FakeCtx()
    sessions = Sessions()
    lifecycle = Lifecycle(ctx, host=object(), event_listeners=events)
    lifecycle.use(sessions)
    await lifecycle.start()
    return sessions, ctx


def _text(message_id: str, text: str, *, role: str = "user"):
    return {
        "id": message_id,
        "role": role,
        "parts": [{"type": "text", "text": text}],
    }


async def _append_conversation(session, count: int):
    for index in range(count):
        await session.append_message(
            _text(
                f"m{index}",
                f"turn {index} " + "body " * 20,
                role="user" if index % 2 == 0 else "assistant",
            )
        )


@pytest.mark.asyncio
async def test_compaction_surface_builders_storage_and_events_are_exact():
    assert not hasattr(agents, "create_compact_function")
    lifecycle_events: list[LifecycleEvent] = []
    sessions, ctx = await _started(events=(lifecycle_events.append,))
    session = sessions.session("one")
    other = sessions.session("two")
    changes = []
    sessions.subscribe(changes.append)

    def first(_messages):
        return None

    async def second(_messages):
        return None

    assert session.on_compaction(first) is session
    assert session.on_compaction(second) is session
    assert session.compact_after(100) is session
    first_row = await session.add_compaction("summary", "a", "b")
    second_row = await session.add_compaction("summary", "a", "b")
    other_row = await other.add_compaction("other", "x", "y")

    assert isinstance(first_row, StoredCompaction)
    assert re.fullmatch(
        r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}",
        first_row.id,
    )
    assert datetime.fromisoformat(first_row.created_at.replace("Z", "+00:00")).tzinfo
    assert [row.id for row in await session.get_compactions()] == [
        first_row.id,
        second_row.id,
    ]
    assert await other.get_compactions() == [other_row]
    assert ctx.storage.sql.exec(
        "SELECT session_id, seq FROM cf_agents_session_compactions "
        "ORDER BY session_id, seq"
    ).toArray() == [
        {"session_id": "one", "seq": 1},
        {"session_id": "one", "seq": 2},
        {"session_id": "two", "seq": 1},
    ]
    assert changes == [
        {"type": "compact", "sessionId": "one"},
        {"type": "compact", "sessionId": "one"},
        {"type": "compact", "sessionId": "two"},
    ]
    assert [event.type for event in lifecycle_events] == ["session:compacted"] * 3
    assert lifecycle_events[0].payload == {
        "sessionId": "one",
        "compactionId": first_row.id,
    }


@pytest.mark.asyncio
async def test_overlays_are_non_destructive_branch_local_and_planned_by_sequence():
    sessions, ctx = await _started()
    session = sessions.session()
    for message_id in ("root", "a", "b", "c", "leaf"):
        await session.append_message(_text(message_id, message_id))
    await session.append_message(
        _text("branch", "branch"),
        AppendOptions(parent_id="a"),
    )

    old = await session.add_compaction("old long", "a", "c")
    newest = await session.add_compaction("new short", "a", "b")
    await session.add_compaction("invalid", "a", "missing")
    single = await session.add_compaction("single", "c", "c")
    branch = await session.add_compaction("branch summary", "a", "branch")

    main = await session.get_history(HistoryReadOptions(leaf_id="leaf"))
    assert [message["id"] for message in main] == [
        "root",
        f"compaction_{newest.id}",
        f"compaction_{single.id}",
        "leaf",
    ]
    assert main[1]["parts"][0]["text"] == "new short"
    assert isinstance(main[1]["createdAt"], datetime)
    assert main[1]["createdAt"].tzinfo == UTC

    branch_history = await session.get_history(HistoryReadOptions(leaf_id="branch"))
    assert [message["id"] for message in branch_history] == [
        "root",
        f"compaction_{branch.id}",
    ]
    assert ctx.storage.sql.exec(
        "SELECT id FROM cf_agents_session_messages ORDER BY seq"
    ).toArray() == [
        {"id": "root"},
        {"id": "a"},
        {"id": "b"},
        {"id": "c"},
        {"id": "leaf"},
        {"id": "branch"},
    ]
    assert old in await session.get_compactions()


@pytest.mark.asyncio
async def test_overlay_reads_skip_covered_hydration_and_keep_raw_stats_and_budgets(
    monkeypatch,
):
    sessions, ctx = await _started()
    session = sessions.session()
    for message_id in ("m0", "m1", "m2", "m3", "m4"):
        await session.append_message(_text(message_id, message_id * 10))
    compaction = await session.add_compaction("middle summary", "m1", "m3")
    execute = ctx.storage.sql.exec
    hydration_ids = []
    statements = []

    def record(query, *params):
        statements.append(query)
        if "id IN (SELECT value FROM json_each(?))" in query and params:
            hydration_ids.append(params[-1])
        return execute(query, *params)

    monkeypatch.setattr(ctx.storage.sql, "exec", record)
    history = await session.get_history()
    assert [message["id"] for message in history] == [
        "m0",
        f"compaction_{compaction.id}",
        "m4",
    ]
    assert all(
        all(message_id not in encoded for message_id in ("m1", "m2", "m3"))
        for encoded in hydration_ids
    )
    assert not any(
        "SELECT id, summary" in query and "ORDER BY seq" in query
        for query in statements
    )

    stats = await session.get_history_row_stats()
    assert [stat.id for stat in stats] == ["m0", "m1", "m2", "m3", "m4"]
    batches = [
        batch
        async for batch in session.history_batches(
            HistoryBatchReadOptions(batch_size=2, max_batch_bytes=10_000)
        )
    ]
    assert [[message["id"] for message in batch] for batch in batches] == [
        ["m0", f"compaction_{compaction.id}"],
        ["m4"],
    ]

    recent_budget = sum(stat.bytes for stat in stats[-3:])
    recent = await session.get_recent_history(recent_budget)
    assert [message["id"] for message in recent.messages] == ["m2", "m3", "m4"]
    assert recent.total_content_bytes == sum(stat.bytes for stat in stats)
    assert recent.truncated is True


@pytest.mark.asyncio
async def test_manual_compact_handles_callbacks_validation_iteration_and_failures():
    lifecycle_events: list[LifecycleEvent] = []
    sessions, ctx = await _started(events=(lifecycle_events.append,))
    session = sessions.session()
    await _append_conversation(session, 10)

    with pytest.raises(RuntimeError, match="Call on_compaction"):
        await session.compact()

    session.on_compaction(
        lambda _messages: CompactResult(
            from_message_id="m3",
            to_message_id="missing",
            summary="ignored",
        )
    )
    assert await session.compact() is None
    assert await session.get_compactions() == []

    session.on_compaction(
        lambda _messages: CompactResult(
            from_message_id="m7",
            to_message_id="m3",
            summary="reversed",
        )
    )
    assert await session.compact() is None
    assert await session.get_compactions() == []

    seen = []

    async def first(messages):
        seen.append([message["id"] for message in messages])
        return CompactResult("m3", "m6", "first summary")

    session.on_compaction(first)
    assert await session.compact() == CompactResult("m3", "m6", "first summary")
    stored = await session.get_compactions()
    assert len(stored) == 1
    assert [message["id"] for message in await session.get_history()][3] == (
        f"compaction_{stored[0].id}"
    )

    def iterative(messages):
        assert any(message["id"].startswith("compaction_") for message in messages)
        return CompactResult("not-used", "m7", "second summary")

    session.on_compaction(iterative)
    assert await session.compact() == CompactResult("m3", "m7", "second summary")
    assert (await session.get_compactions())[-1].from_message_id == "m3"

    def fail(_messages):
        raise RuntimeError("model failed")

    session.on_compaction(fail)
    assert await session.compact() is None
    assert len(await session.get_compactions()) == 2
    assert lifecycle_events[-1].type == "session:error"
    assert lifecycle_events[-1].payload == {
        "sessionId": "",
        "error": "model failed",
    }
    assert ctx.storage.sql.exec(
        "SELECT COUNT(*) AS count FROM cf_agents_session_messages"
    ).toArray() == [{"count": 10}]


@pytest.mark.asyncio
async def test_reference_compactor_protects_boundaries_and_builds_exact_prompts():
    messages = [
        _text(f"m{index}", f"turn {index} " + "body " * 20) for index in range(12)
    ]
    prompts = []

    def summarize(prompt):
        prompts.append(prompt)
        return "  summary body  "

    compact = create_compact_function(
        CompactOptions(summarize=summarize, keep_recent_tokens=1)
    )
    result = await compact(messages)
    assert result == CompactResult("m3", "m9", "  summary body  ")
    assert prompts[0].startswith(
        "Create a concise summary of this conversation that preserves"
    )
    assert "CONVERSATION TO SUMMARIZE:\n[user]\nturn 3" in prompts[0]
    assert "turn 9" in prompts[0]
    assert "turn 10" not in prompts[0]
    assert "Target ~100 tokens." in prompts[0]

    called = False

    async def must_not_run(_prompt):
        nonlocal called
        called = True
        return "summary"

    too_short = create_compact_function(CompactOptions(summarize=must_not_run))
    assert await too_short(messages[:5]) is None
    assert called is False

    blank = create_compact_function(
        CompactOptions(summarize=lambda _prompt: "   ", keep_recent_tokens=1)
    )
    assert await blank(messages) is None


@pytest.mark.asyncio
async def test_reference_compactor_keeps_tool_groups_and_folds_previous_summary():
    messages = [_text(f"m{index}", f"turn {index}") for index in range(10)]
    messages[2] = {
        "id": "call-head",
        "role": "assistant",
        "parts": [
            {
                "type": "tool-read",
                "toolCallId": "head",
                "toolName": "read",
                "input": {},
            }
        ],
    }
    messages[3] = {
        "id": "result-head-1",
        "role": "tool",
        "parts": [{"type": "tool-read", "toolCallId": "head", "output": []}],
    }
    messages[4] = {
        "id": "result-head-2",
        "role": "tool",
        "parts": [{"type": "tool-read", "toolCallId": "head", "output": "ok"}],
    }
    prompts = []
    compact = create_compact_function(
        CompactOptions(
            summarize=lambda prompt: prompts.append(prompt) or "summary",
            keep_recent_tokens=0,
        )
    )
    result = await compact(messages)
    assert result is not None
    assert result.from_message_id == "m5"

    tail = [_text(f"t{index}", f"tail {index}") for index in range(10)]
    tail[6] = {
        "id": "call-tail",
        "role": "assistant",
        "parts": [{"type": "dynamic-tool", "toolCallId": "tail"}],
    }
    tail[7] = {
        "id": "result-tail-1",
        "role": "tool",
        "parts": [{"type": "dynamic-tool", "toolCallId": "tail"}],
    }
    tail[8] = {
        "id": "result-tail-2",
        "role": "tool",
        "parts": [{"type": "dynamic-tool", "toolCallId": "tail"}],
    }
    result = await compact(tail)
    assert result is not None
    assert result.to_message_id == "t5"

    previous = _text("compaction_previous", "earlier summary", role="assistant")
    iterative = [*tail[:4], previous, *tail[4:]]
    result = await compact(iterative)
    assert result is not None
    assert not result.from_message_id.startswith("compaction_")
    assert "PREVIOUS SUMMARY:\nearlier summary" in prompts[-1]


@pytest.mark.asyncio
async def test_reference_prompt_formats_tool_values_with_javascript_semantics():
    messages = [_text(f"m{index}", f"turn {index}") for index in range(9)]
    messages[3] = {
        "id": "tool",
        "role": "assistant",
        "parts": [
            {"type": "text"},
            {"type": "text", "text": None},
            {"type": "text", "text": "kept"},
            {
                "type": "tool-inspect",
                "toolCallId": "call",
                "input": {},
                "output": {"answer": 42},
            },
            {
                "type": "tool-list",
                "toolCallId": "list",
                "toolName": "list",
                "input": [],
                "output": [1, None, 3],
            },
        ],
    }
    prompts = []
    compact = create_compact_function(
        CompactOptions(
            summarize=lambda prompt: prompts.append(prompt) or "summary",
            keep_recent_tokens=0,
        )
    )

    assert await compact(messages) is not None
    assert "[Tool: unknown]\nInput: {}\nOutput: [object Object]" in prompts[0]
    assert "[Tool: list]\nInput: []\nOutput: 1,,3" in prompts[0]
    assert "[assistant]\n\n\nkept\n[Tool: unknown]" in prompts[0]


@pytest.mark.asyncio
async def test_reference_budget_uses_nullish_reasoning_and_result_fallbacks():
    messages = [_text(f"m{index}", f"turn {index}") for index in range(9)]
    messages[3] = {
        "id": "nullable",
        "role": "assistant",
        "parts": [
            {"type": "reasoning", "text": None, "reasoning": "x" * 4_000},
            {
                "type": "tool-work",
                "output": None,
                "result": "y" * 4_000,
            },
        ],
    }
    prompts = []
    compact = create_compact_function(
        CompactOptions(
            summarize=lambda prompt: prompts.append(prompt) or "summary",
            keep_recent_tokens=0,
        )
    )

    assert await compact(messages) is not None
    assert "Target ~405 tokens." in prompts[0]


@pytest.mark.asyncio
async def test_compaction_callbacks_are_serialized_against_stale_overlays():
    sessions, _ = await _started()
    session = sessions.session()
    await _append_conversation(session, 7)
    gate = fakes.AsyncGate()
    calls = 0

    async def compact(messages):
        nonlocal calls
        calls += 1
        if calls == 1:
            await gate.block()
            return CompactResult("m1", "m3", "first")
        assert any(message["id"].startswith("compaction_") for message in messages)
        return CompactResult("ignored", "m5", "second")

    session.on_compaction(compact)
    first = asyncio.create_task(session.compact())
    await gate.wait_until_blocked()
    second = asyncio.create_task(session.compact())
    await asyncio.sleep(0)
    gate.release()

    assert await first == CompactResult("m1", "m3", "first")
    assert await second == CompactResult("m1", "m5", "second")
    history = await session.get_history()
    assert history[1]["parts"][0]["text"] == "second"


@pytest.mark.asyncio
async def test_message_mutation_fences_an_inflight_summary():
    sessions, _ = await _started()
    session = sessions.session()
    await _append_conversation(session, 7)
    gate = fakes.AsyncGate()

    async def compact(_messages):
        await gate.block()
        return CompactResult("m1", "m4", "stale summary")

    session.on_compaction(compact)
    pending = asyncio.create_task(session.compact())
    await gate.wait_until_blocked()
    await session.update_message(_text("m2", "changed while summarizing"))
    gate.release()

    assert await pending is None
    assert await session.get_compactions() == []
    assert (await session.get_message("m2"))["parts"][0]["text"] == (
        "changed while summarizing"
    )


@pytest.mark.asyncio
async def test_malformed_compactions_are_skipped_without_hiding_raw_history():
    sessions, ctx = await _started()
    session = sessions.session()
    await session.append_message(_text("m0", "zero"))
    await session.append_message(_text("m1", "one"))
    ctx.storage.sql.exec(
        "INSERT INTO cf_agents_session_compactions "
        "(session_id, id, seq, summary, from_message_id, to_message_id, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        "",
        "malformed",
        1,
        b"not text",
        "m0",
        "m1",
        "not an integer",
    )
    ctx.storage.sql.exec(
        "INSERT INTO cf_agents_session_compactions "
        "(session_id, id, seq, summary, from_message_id, to_message_id, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        "",
        "extreme-time",
        2,
        "summary",
        "m0",
        "m1",
        9_223_372_036_854_775_807,
    )

    assert await session.get_compactions() == []
    assert [message["id"] for message in await session.get_history()] == ["m0", "m1"]


@pytest.mark.asyncio
async def test_auto_compaction_runs_after_append_once_and_contains_failures():
    lifecycle_events: list[LifecycleEvent] = []
    sessions, _ = await _started(events=(lifecycle_events.append,))
    session = sessions.session()
    await _append_conversation(session, 5)
    order = []

    async def listener(event):
        order.append(event["type"])

    sessions.subscribe(listener)
    calls = 0

    async def compact(messages):
        nonlocal calls
        calls += 1
        order.append("callback")
        return CompactResult("m1", messages[-2]["id"], "automatic")

    session.on_compaction(compact).compact_after(0)
    inserted = await session.append_message(_text("m5", "new turn"))
    assert inserted.inserted is True
    assert calls == 1
    assert order == ["append", "callback", "compact"]
    assert [event.type for event in lifecycle_events[-2:]] == [
        "session:message:appended",
        "session:compacted",
    ]

    await session.append_message(_text("m5", "duplicate"))
    assert calls == 1

    async def fail(_messages):
        raise RuntimeError("automatic failure")

    session.on_compaction(fail)
    result = await session.append_message(_text("m6", "another turn"))
    assert result.inserted is True
    assert lifecycle_events[-1].type == "session:error"
    assert lifecycle_events[-1].payload == {
        "sessionId": "",
        "error": "automatic failure",
    }


@pytest.mark.asyncio
async def test_queued_auto_compaction_rechecks_the_reduced_token_estimate():
    sessions, _ = await _started()
    session = sessions.session()
    for index in range(6):
        await session.append_message(_text(f"m{index}", "short body"))
    gate = fakes.AsyncGate()
    calls = 0

    async def compact(_messages):
        nonlocal calls
        calls += 1
        await gate.block()
        return CompactResult("m1", "m4", "x")

    session.on_compaction(compact).compact_after(20)
    first = asyncio.create_task(session._maybe_auto_compact())
    await gate.wait_until_blocked()
    second = asyncio.create_task(session._maybe_auto_compact())
    await asyncio.sleep(0)
    gate.release()
    await asyncio.gather(first, second)

    assert calls == 1
    assert len(await session.get_compactions()) == 1


@pytest.mark.asyncio
async def test_auto_compaction_estimation_failures_do_not_reject_committed_appends(
    monkeypatch,
):
    lifecycle_events: list[LifecycleEvent] = []
    sessions, _ = await _started(events=(lifecycle_events.append,))
    session = sessions.session()
    session.on_compaction(lambda _messages: None).compact_after(0)

    def fail_estimate():
        raise RuntimeError("estimate failed")

    monkeypatch.setattr(session, "_active_token_estimate", fail_estimate)
    result = await session.append_message(_text("committed", "body"))

    assert result.inserted is True
    assert await session.get_message("committed") == _text("committed", "body")
    assert lifecycle_events[-1].type == "session:error"
    assert lifecycle_events[-1].payload == {
        "sessionId": "",
        "error": "estimate failed",
    }


@pytest.mark.asyncio
async def test_clear_removes_compactions_and_search_remains_independent():
    sessions, ctx = await _started()
    session = sessions.session()
    await session.append_message(_text("message", "searchable text"))
    await session.add_compaction("summary", "message", "message")
    assert await session.search("searchable") == [
        SearchResult("message", "user", "searchable text")
    ]

    await session.clear_messages()

    assert await session.get_compactions() == []
    assert (
        ctx.storage.sql.exec("SELECT id FROM cf_agents_session_compactions").toArray()
        == []
    )
