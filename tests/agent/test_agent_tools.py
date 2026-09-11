from __future__ import annotations

import asyncio
import json
import types

import fakes
import pytest

from agents import AIChatAgent, Agent
from agents.chat.resumable_stream import StreamStorageUnavailable
from agents.core.agent_tools import AgentToolRuns
from agents.core.schema import prepare_core_schema


def _events(connection) -> list[dict]:
    return [
        frame for frame in connection.frames if frame.get("type") == "agent-tool-event"
    ]


def _child(cls: type[AIChatAgent]) -> AIChatAgent:
    return fakes.build_chat_agent(cls=cls, name="child")


def _attach_child(parent, child) -> None:
    async def resolve(_class_name, _name):
        return child

    parent._resolve_sub_agent = resolve


@pytest.mark.asyncio
async def test_agent_tool_runs_operates_through_narrow_live_adapters():
    sql = fakes.make_sql()
    sql_calls = []

    def tracked_sql(query, *params):
        sql_calls.append(query)
        return sql(query, *params)

    events = []
    resolved = []
    settings = {"max_concurrent": 0, "recovery_grace_ms": 60_000}
    child = fakes.FakeChildAgentToolStub(
        start={
            "runId": "standalone",
            "status": "completed",
            "output": "answer",
            "summary": "answer",
        }
    )

    async def resolve(class_name, name):
        resolved.append((class_name, name))
        return child

    runs = AgentToolRuns(
        sql=tracked_sql,
        publish=events.append,
        resolve_sub_agent=resolve,
        max_concurrent=lambda: settings["max_concurrent"],
        recovery_grace_ms=lambda: settings["recovery_grace_ms"],
    )

    assert sql_calls == []
    prepare_core_schema(tracked_sql)
    runs.prepare()
    settings["max_concurrent"] = 1

    class StandaloneChild:
        pass

    result = await runs.run_agent_tool(
        StandaloneChild,
        input="question",
        run_id="standalone",
    )

    assert result.output == "answer"
    assert resolved == [("StandaloneChild", "standalone")]
    assert [frame["event"]["kind"] for frame in events] == ["started", "finished"]


@pytest.mark.asyncio
async def test_awaited_agent_tool_returns_output_and_emits_ordered_events():
    class ResearchAgent(AIChatAgent):
        async def on_chat_message(self, options):
            assert options.body["agentToolInput"] == "question"
            return "answer"

    parent = fakes.build_agent()
    child = _child(ResearchAgent)
    _attach_child(parent, child)
    connection = fakes.FakeConnection()
    parent._connections[connection.id] = connection

    result = await parent.run_agent_tool(
        ResearchAgent,
        input="question",
        run_id="research-1",
        parent_tool_call_id="tool-call-1",
    )

    assert result.run_id == "research-1"
    assert result.agent_type == "ResearchAgent"
    assert result.status == "completed"
    assert result.output == "answer"
    assert result.summary == "answer"

    events = _events(connection)
    assert [frame["sequence"] for frame in events] == list(range(len(events)))
    assert events[0]["event"]["kind"] == "started"
    assert events[0]["parentToolCallId"] == "tool-call-1"
    assert events[-1]["event"] == {
        "kind": "finished",
        "runId": "research-1",
        "summary": "answer",
    }
    assert all("replay" not in frame for frame in events)

    chunks = [frame for frame in events if frame["event"]["kind"] == "chunk"]
    bodies = [json.loads(frame["event"]["body"]) for frame in chunks]
    assert bodies[0]["type"] == "start"
    assert any(body.get("delta") == "answer" for body in bodies)
    assert bodies[-1]["type"] == "finish"

    row = parent.sql(
        "SELECT status, output_json, summary FROM cf_agent_tool_runs "
        "WHERE run_id = 'research-1'"
    )[0]
    assert row == {
        "status": "completed",
        "output_json": '"answer"',
        "summary": "answer",
    }


@pytest.mark.asyncio
async def test_cold_agent_tool_delegate_awaits_lifecycle_readiness():
    class ResearchAgent(AIChatAgent):
        async def on_chat_message(self, options):
            return "answer"

    parent = Agent(fakes.FakeCtx(), types.SimpleNamespace())
    child = _child(ResearchAgent)
    _attach_child(parent, child)

    result = await parent.run_agent_tool(
        ResearchAgent,
        input="question",
        run_id="cold-run",
    )

    assert result.output == "answer"
    assert parent._lifecycle._ready is True


@pytest.mark.asyncio
async def test_agent_tool_run_id_is_idempotent():
    class CountingAgent(AIChatAgent):
        calls = 0

        async def on_chat_message(self, options):
            type(self).calls += 1
            return "once"

    parent = fakes.build_agent()
    child = _child(CountingAgent)
    _attach_child(parent, child)
    connection = fakes.FakeConnection()
    parent._connections[connection.id] = connection

    first = await parent.run_agent_tool(CountingAgent, input="go", run_id="stable")
    event_count = len(_events(connection))
    second = await parent.run_agent_tool(
        CountingAgent, input="ignored", run_id="stable"
    )

    assert first == second
    assert CountingAgent.calls == 1
    assert len(_events(connection)) == event_count


@pytest.mark.asyncio
async def test_new_run_resolves_supplied_agent_class_name():
    class NamedAgent(AIChatAgent):
        async def on_chat_message(self, options):
            return "named"

    parent = fakes.build_agent()
    child = _child(NamedAgent)
    resolved = []

    async def resolve(class_name, name):
        resolved.append((class_name, name))
        return child

    parent._resolve_sub_agent = resolve

    result = await parent.run_agent_tool(NamedAgent, input="x", run_id="named-run")

    assert result.status == "completed"
    assert resolved == [("NamedAgent", "named-run")]


@pytest.mark.asyncio
async def test_retry_resolves_persisted_agent_type_instead_of_supplied_class():
    class OriginalAgent(AIChatAgent):
        pass

    class ReplacementAgent(AIChatAgent):
        pass

    parent = fakes.build_agent()
    parent.sql(
        "INSERT INTO cf_agent_tool_runs "
        "(run_id, agent_type, status, display_order, started_at) "
        "VALUES ('stable-type', 'OriginalAgent', 'interrupted', 0, 1)"
    )
    resolved = []
    child = fakes.FakeChildAgentToolStub(
        inspection={
            "status": "completed",
            "output": "original",
            "summary": "original",
        }
    )

    async def resolve(class_name, name):
        resolved.append((class_name, name))
        return child

    parent._resolve_sub_agent = resolve

    result = await parent.run_agent_tool(
        ReplacementAgent,
        input="ignored",
        run_id="stable-type",
    )

    assert result.status == "completed"
    assert result.agent_type == "OriginalAgent"
    assert result.output == "original"
    assert resolved == [("OriginalAgent", "stable-type")]


@pytest.mark.asyncio
async def test_persisted_agent_type_resolution_never_falls_back_to_supplied_class():
    class ReplacementAgent(AIChatAgent):
        pass

    parent = fakes.build_agent()
    parent.sql(
        "INSERT INTO cf_agent_tool_runs "
        "(run_id, agent_type, status, display_order, started_at) "
        "VALUES ('missing-type', 'OriginalAgent', 'interrupted', 0, 1)"
    )
    resolved = []

    async def resolve(class_name, name):
        resolved.append((class_name, name))
        raise RuntimeError("class unavailable")

    parent._resolve_sub_agent = resolve

    result = await parent.run_agent_tool(
        ReplacementAgent,
        input="ignored",
        run_id="missing-type",
    )

    assert result.status == "interrupted"
    assert result.reason == "inspect-failed"
    assert result.agent_type == "OriginalAgent"
    assert resolved == [("OriginalAgent", "missing-type")]


@pytest.mark.asyncio
async def test_agent_tool_failure_settles_parent_and_replays():
    class BrokenAgent(AIChatAgent):
        async def on_chat_message(self, options):
            raise RuntimeError("child failed")

    parent = fakes.build_agent()
    child = _child(BrokenAgent)
    _attach_child(parent, child)
    live = fakes.FakeConnection()
    parent._connections[live.id] = live

    result = await parent.run_agent_tool(BrokenAgent, input="go", run_id="broken")

    assert result.status == "error"
    assert result.error == "child failed"
    assert _events(live)[-1]["event"]["kind"] == "error"

    replay = fakes.FakeConnection(id="replay")
    parent._replay_agent_tool_runs(replay)
    replayed = _events(replay)
    assert [frame["sequence"] for frame in replayed] == [
        frame["sequence"] for frame in _events(live)
    ]
    assert all(frame["replay"] is True for frame in replayed)
    assert replayed[-1]["event"] == {
        "kind": "error",
        "runId": "broken",
        "error": "child failed",
    }


def test_connect_replay_interrupts_an_abandoned_run_after_grace():
    parent = fakes.build_agent()
    parent.agent_tool_recovery_grace_ms = 0
    parent.sql(
        "INSERT INTO cf_agent_tool_runs "
        "(run_id, agent_type, status, display_order, started_at) "
        "VALUES ('stale', 'Child', 'running', 0, 0)"
    )

    connection = fakes.FakeConnection()
    parent._replay_agent_tool_runs(connection)

    row = parent.sql(
        "SELECT status, interrupted_reason, child_still_running "
        "FROM cf_agent_tool_runs WHERE run_id = 'stale'"
    )[0]
    assert row == {
        "status": "interrupted",
        "interrupted_reason": "recovery-deadline",
        "child_still_running": None,
    }
    assert _events(connection)[-1]["event"]["kind"] == "interrupted"


@pytest.mark.asyncio
async def test_reissue_repairs_soft_interruption_with_higher_sequences():
    class RecoveredAgent(AIChatAgent):
        async def on_chat_message(self, options):
            return "recovered"

    child = _child(RecoveredAgent)
    await child._cf_start_agent_tool_run('"input"', "recoverable")

    parent = fakes.build_agent()
    _attach_child(parent, child)
    parent.agent_tool_recovery_grace_ms = 0
    parent.sql(
        "INSERT INTO cf_agent_tool_runs "
        "(run_id, agent_type, status, display_order, started_at) "
        "VALUES ('recoverable', 'RecoveredAgent', 'running', 0, 0)"
    )
    connection = fakes.FakeConnection()
    parent._connections[connection.id] = connection
    parent._replay_agent_tool_runs(connection)
    interrupted_sequence = _events(connection)[-1]["sequence"]
    replay_event_count = len(_events(connection))

    result = await parent.run_agent_tool(
        RecoveredAgent, input="input", run_id="recoverable"
    )

    assert result.status == "completed"
    assert result.output == "recovered"
    repaired = _events(connection)[replay_event_count:]
    assert repaired
    assert all(frame["sequence"] > interrupted_sequence for frame in repaired)
    assert repaired[-1]["event"]["kind"] == "finished"

    reconnect = fakes.FakeConnection(id="reconnect")
    parent._replay_agent_tool_runs(reconnect)
    replayed = _events(reconnect)
    assert replayed[-1]["event"]["kind"] == "finished"
    assert replayed[-1]["sequence"] == repaired[-1]["sequence"]


@pytest.mark.asyncio
async def test_repair_does_not_remirror_gapped_or_duplicate_child_chunks():
    class ChildAgent(AIChatAgent):
        pass

    parent = fakes.build_agent()
    child = fakes.FakeChildAgentToolStub(
        inspection={"status": "completed", "output": "done", "summary": "done"},
        chunks=[
            {"sequence": 0, "body": "zero"},
            {"sequence": 2, "body": "two"},
            {"sequence": 2, "body": "duplicate-two"},
        ],
    )
    _attach_child(parent, child)
    parent.sql(
        "INSERT INTO cf_agent_tool_runs "
        "(run_id, agent_type, status, display_order, started_at) "
        "VALUES ('gapped', 'ChildAgent', 'interrupted', 0, 1)"
    )
    parent.sql(
        "INSERT INTO cf_agent_tool_chunks (run_id, sequence, body) "
        "VALUES ('gapped', 1, 'zero'), ('gapped', 2, 'two')"
    )
    connection = fakes.FakeConnection()
    parent._connections[connection.id] = connection

    result = await parent.run_agent_tool(ChildAgent, input="x", run_id="gapped")

    assert result.status == "completed"
    assert parent.sql(
        "SELECT sequence, body FROM cf_agent_tool_chunks "
        "WHERE run_id = 'gapped' ORDER BY sequence"
    ) == [{"sequence": 1, "body": "zero"}, {"sequence": 2, "body": "two"}]
    assert [
        frame for frame in _events(connection) if frame["event"]["kind"] == "chunk"
    ] == []


@pytest.mark.asyncio
async def test_child_chunks_are_mirrored_through_bounded_pages():
    class ChildAgent(AIChatAgent):
        pass

    parent = fakes.build_agent()
    child = fakes.FakeChildAgentToolStub(
        inspection={"status": "completed", "output": "done", "summary": "done"},
        chunks=[
            {"sequence": sequence, "body": f"chunk-{sequence}"}
            for sequence in range(205)
        ],
    )
    _attach_child(parent, child)
    parent.sql(
        "INSERT INTO cf_agent_tool_runs "
        "(run_id, agent_type, status, display_order, started_at) "
        "VALUES ('paged', 'ChildAgent', 'running', 0, 1)"
    )

    result = await parent.run_agent_tool(ChildAgent, input="x", run_id="paged")

    assert result.status == "completed"
    assert child.chunk_calls == [("paged", -1), ("paged", 99), ("paged", 199)]
    assert (
        parent.sql(
            "SELECT COUNT(*) AS count FROM cf_agent_tool_chunks WHERE run_id = 'paged'"
        )[0]["count"]
        == 205
    )


@pytest.mark.asyncio
async def test_concurrent_reissue_inspects_and_finishes_only_once():
    class ChildAgent(AIChatAgent):
        pass

    entered = asyncio.Event()
    release = asyncio.Event()

    async def inspect_hook(run_id):
        entered.set()
        await release.wait()

    parent = fakes.build_agent()
    child = fakes.FakeChildAgentToolStub(
        inspection={"status": "completed", "output": "done", "summary": "done"},
        inspect_hook=inspect_hook,
    )
    _attach_child(parent, child)
    parent.sql(
        "INSERT INTO cf_agent_tool_runs "
        "(run_id, agent_type, status, display_order, started_at) "
        "VALUES ('shared', 'ChildAgent', 'running', 0, 1)"
    )

    first = asyncio.create_task(
        parent.run_agent_tool(ChildAgent, input="x", run_id="shared")
    )
    await entered.wait()
    second = asyncio.create_task(
        parent.run_agent_tool(ChildAgent, input="x", run_id="shared")
    )
    await asyncio.sleep(0)
    release.set()

    first_result, second_result = await asyncio.gather(first, second)
    assert first_result == second_result
    assert first_result.status == "completed"
    assert child.inspect_calls == ["shared"]


@pytest.mark.asyncio
async def test_completed_run_survives_parent_reincarnation():
    class DurableChild(AIChatAgent):
        async def on_chat_message(self, options):
            return "durable"

    class DifferentChild(AIChatAgent):
        pass

    parent_db = fakes.new_sqlite()
    parent = fakes.build_agent(conn=parent_db)
    child = _child(DurableChild)
    _attach_child(parent, child)
    expected = await parent.run_agent_tool(
        DurableChild, input="x", run_id="durable-run"
    )

    reincarnated = fakes.build_agent(conn=parent_db)

    async def fail_if_resolved(_cls, _name):
        raise AssertionError("a hard-terminal retry must not wake the child")

    reincarnated._resolve_sub_agent = fail_if_resolved
    actual = await reincarnated.run_agent_tool(
        DifferentChild, input="ignored", run_id="durable-run"
    )

    assert actual == expected
    connection = fakes.FakeConnection()
    reincarnated._replay_agent_tool_runs(connection)
    assert _events(connection)[-1]["event"]["kind"] == "finished"


@pytest.mark.asyncio
async def test_stale_rows_do_not_exhaust_the_concurrency_limit():
    class FreshChild(AIChatAgent):
        async def on_chat_message(self, options):
            return "fresh"

    parent = fakes.build_agent()
    parent.agent_tool_recovery_grace_ms = 0
    for index in range(parent.max_concurrent_agent_tools):
        parent.sql(
            "INSERT INTO cf_agent_tool_runs "
            "(run_id, agent_type, status, display_order, started_at) "
            "VALUES (?, 'OldChild', 'running', 0, 0)",
            f"old-{index}",
        )
    _attach_child(parent, _child(FreshChild))

    result = await parent.run_agent_tool(FreshChild, input="x", run_id="new-run")

    assert result.status == "completed"
    stale = parent.sql(
        "SELECT status FROM cf_agent_tool_runs WHERE run_id LIKE 'old-%'"
    )
    assert {row["status"] for row in stale} == {"interrupted"}


def test_replay_advances_terminal_past_partially_mirrored_repair():
    parent = fakes.build_agent()
    parent.sql(
        "INSERT INTO cf_agent_tool_runs "
        "(run_id, agent_type, status, display_order, started_at) "
        "VALUES ('partial', 'Child', 'interrupted', 0, 0)"
    )
    parent.sql(
        "INSERT INTO cf_agent_tool_terminals (run_id, sequence) VALUES ('partial', 1)"
    )
    parent.sql(
        "INSERT INTO cf_agent_tool_chunks (run_id, sequence, body) "
        "VALUES ('partial', 2, '{}'), ('partial', 3, '{}')"
    )

    connection = fakes.FakeConnection()
    parent._replay_agent_tool_runs(connection)

    assert [frame["sequence"] for frame in _events(connection)] == [0, 2, 3, 4]
    terminal = parent.sql(
        "SELECT sequence FROM cf_agent_tool_terminals WHERE run_id = 'partial'"
    )[0]
    assert terminal["sequence"] == 4


@pytest.mark.asyncio
async def test_child_inspection_accepts_completed_null_output():
    child = fakes.build_chat_agent(lambda options: "unused")
    child.sql(
        "INSERT INTO cf_ai_chat_agent_tool_runs "
        "(run_id, request_id, status, input_json, started_at, completed_at) "
        "VALUES ('null-output', NULL, 'completed', NULL, 1, 2)"
    )

    inspection_json = await child._cf_inspect_agent_tool_run("null-output")
    inspection = json.loads(inspection_json)

    assert inspection["status"] == "completed"
    assert "output" not in inspection


@pytest.mark.asyncio
async def test_explicit_empty_run_id_is_rejected():
    parent = fakes.build_agent()
    with pytest.raises(ValueError, match="run_id must not be blank"):
        await parent.run_agent_tool(AIChatAgent, input="x", run_id="")


@pytest.mark.asyncio
async def test_lost_start_response_recovers_child_terminal():
    class DurableChild(AIChatAgent):
        async def on_chat_message(self, options):
            return "committed"

    real_child = _child(DurableChild)

    parent = fakes.build_agent()
    _attach_child(
        parent,
        fakes.FakeChildAgentToolStub(
            delegate=real_child,
            start_error=RuntimeError("response lost"),
        ),
    )

    result = await parent.run_agent_tool(
        DurableChild, input="x", run_id="lost-response"
    )

    assert result.status == "completed"
    assert result.output == "committed"
    assert (
        parent.sql(
            "SELECT status FROM cf_agent_tool_runs WHERE run_id = 'lost-response'"
        )[0]["status"]
        == "completed"
    )


@pytest.mark.asyncio
async def test_inspection_failure_persists_retryable_interruption():
    class ChildAgent(AIChatAgent):
        pass

    parent = fakes.build_agent()
    _attach_child(
        parent,
        fakes.FakeChildAgentToolStub(inspect_error=RuntimeError("offline")),
    )
    parent.sql(
        "INSERT INTO cf_agent_tool_runs "
        "(run_id, agent_type, status, display_order, started_at) "
        "VALUES ('inspect-failed', 'ChildAgent', 'running', 0, 1)"
    )

    result = await parent.run_agent_tool(ChildAgent, input="x", run_id="inspect-failed")

    assert result.status == "interrupted"
    assert result.reason == "inspect-failed"
    row = parent.sql(
        "SELECT status, interrupted_reason FROM cf_agent_tool_runs "
        "WHERE run_id = 'inspect-failed'"
    )[0]
    assert row == {
        "status": "interrupted",
        "interrupted_reason": "inspect-failed",
    }


@pytest.mark.asyncio
async def test_child_reconciles_running_row_from_completed_stream():
    child = fakes.build_chat_agent(lambda options: "settled")
    await child._cf_start_agent_tool_run('"x"', "stranded")
    child.sql(
        "UPDATE cf_ai_chat_agent_tool_runs SET status = 'running', "
        "output_json = NULL, summary = NULL, completed_at = NULL "
        "WHERE run_id = 'stranded'"
    )

    inspection_json = await child._cf_inspect_agent_tool_run("stranded")
    inspection = json.loads(inspection_json)

    assert inspection["status"] == "completed"
    assert inspection["output"] == "settled"


@pytest.mark.asyncio
async def test_child_reads_packed_typescript_stream_chunks():
    child = fakes.build_chat_agent(lambda options: "unused")
    child.sql(
        "INSERT INTO cf_ai_chat_agent_tool_runs "
        "(run_id, request_id, status, input_json, started_at, completed_at) "
        "VALUES ('typescript', 'request-ts', 'completed', NULL, 1, 2)"
    )
    child.sql(
        "INSERT INTO cf_ai_chat_stream_metadata "
        "(id, request_id, status, created_at) "
        "VALUES ('stream-ts', 'request-ts', 'completed', 1)"
    )
    packed = json.dumps(['{"type":"start"}', '{"type":"finish"}'])
    child.sql(
        "INSERT INTO cf_ai_chat_stream_chunks "
        "(id, stream_id, body, chunk_index, created_at) "
        "VALUES ('segment', 'stream-ts', ?, 0, 1)",
        packed,
    )

    chunks_json = await child._cf_get_agent_tool_chunks("typescript", -1)

    assert json.loads(chunks_json) == [
        {"sequence": 0, "body": '{"type":"start"}'},
        {"sequence": 1, "body": '{"type":"finish"}'},
    ]


@pytest.mark.asyncio
async def test_unserializable_input_does_not_resolve_child():
    class ChildAgent(AIChatAgent):
        pass

    parent = fakes.build_agent()
    resolved = False

    async def resolve(_cls, _name):
        nonlocal resolved
        resolved = True
        raise AssertionError("child must not be resolved")

    parent._resolve_sub_agent = resolve

    with pytest.raises(ValueError):
        await parent.run_agent_tool(
            ChildAgent,
            input={"invalid": float("nan")},
            run_id="invalid-input",
        )

    assert resolved is False
    assert (
        parent.sql(
            "SELECT run_id FROM cf_agent_tool_runs WHERE run_id = 'invalid-input'"
        )
        == []
    )


@pytest.mark.asyncio
async def test_retry_restarts_missing_child_run():
    class ChildAgent(AIChatAgent):
        pass

    child = fakes.FakeChildAgentToolStub(
        start={
            "runId": "missing-child",
            "status": "completed",
            "output": "restarted",
            "summary": "restarted",
        }
    )
    parent = fakes.build_agent()
    _attach_child(parent, child)
    parent.sql(
        "INSERT INTO cf_agent_tool_runs "
        "(run_id, agent_type, status, display_order, started_at) "
        "VALUES ('missing-child', 'ChildAgent', 'interrupted', 0, 1)"
    )

    result = await parent.run_agent_tool(
        ChildAgent, input={"task": "resume"}, run_id="missing-child"
    )

    assert result.status == "completed"
    assert result.output == "restarted"
    assert child.start_calls == [('{"task":"resume"}', "missing-child")]


@pytest.mark.asyncio
async def test_completed_repair_backfills_parent_for_later_retries():
    class ChildAgent(AIChatAgent):
        pass

    child = fakes.FakeChildAgentToolStub(
        inspection={
            "status": "completed",
            "output": "recovered",
            "summary": "recovered",
        }
    )
    parent = fakes.build_agent()
    _attach_child(parent, child)
    parent.sql(
        "INSERT INTO cf_agent_tool_runs "
        "(run_id, agent_type, status, display_order, started_at, completed_at) "
        "VALUES ('completed-repair', 'ChildAgent', 'completed', 0, 1, 2)"
    )

    first = await parent.run_agent_tool(
        ChildAgent, input="ignored", run_id="completed-repair"
    )
    second = await parent.run_agent_tool(
        ChildAgent, input="ignored", run_id="completed-repair"
    )

    assert first == second
    assert first.output == "recovered"
    assert child.inspect_calls == ["completed-repair"]
    assert parent.sql(
        "SELECT output_json, summary FROM cf_agent_tool_runs "
        "WHERE run_id = 'completed-repair'"
    )[0] == {"output_json": '"recovered"', "summary": "recovered"}


@pytest.mark.asyncio
async def test_retry_chunk_failure_persists_interruption():
    class ChildAgent(AIChatAgent):
        pass

    parent = fakes.build_agent()
    _attach_child(
        parent,
        fakes.FakeChildAgentToolStub(
            inspection={"status": "completed", "output": "done", "summary": "done"},
            chunks_error=RuntimeError("chunk transport failed"),
        ),
    )
    parent.sql(
        "INSERT INTO cf_agent_tool_runs "
        "(run_id, agent_type, status, display_order, started_at) "
        "VALUES ('broken-chunks', 'ChildAgent', 'running', 0, 1)"
    )

    result = await parent.run_agent_tool(ChildAgent, input="x", run_id="broken-chunks")

    assert result.status == "interrupted"
    assert result.reason == "child-rpc-failed"
    assert result.error == "chunk transport failed"


@pytest.mark.asyncio
async def test_preset_retry_abort_does_not_resolve_child():
    class ChildAgent(AIChatAgent):
        pass

    parent = fakes.build_agent()
    parent.sql(
        "INSERT INTO cf_agent_tool_runs "
        "(run_id, agent_type, status, display_order, started_at) "
        "VALUES ('abort-retry', 'ChildAgent', 'interrupted', 0, 1)"
    )
    resolved = False

    async def resolve(_cls, _name):
        nonlocal resolved
        resolved = True
        raise AssertionError("aborted retry must not resolve its child")

    parent._resolve_sub_agent = resolve
    abort = asyncio.Event()
    abort.set()

    result = await parent.run_agent_tool(
        ChildAgent, input="x", run_id="abort-retry", abort=abort
    )

    assert result.status == "aborted"
    assert resolved is False


@pytest.mark.asyncio
async def test_lost_response_recovery_honors_abort():
    class ChildAgent(AIChatAgent):
        async def on_chat_message(self, options):
            return "completed in child"

    real_child = _child(ChildAgent)
    abort = asyncio.Event()

    parent = fakes.build_agent()
    _attach_child(
        parent,
        fakes.FakeChildAgentToolStub(
            delegate=real_child,
            start_hook=lambda *_args: abort.set(),
            start_error=RuntimeError("response lost"),
        ),
    )

    result = await parent.run_agent_tool(
        ChildAgent, input="x", run_id="abort-lost-response", abort=abort
    )

    assert result.status == "aborted"
    assert (
        parent.sql(
            "SELECT status FROM cf_agent_tool_runs WHERE run_id = 'abort-lost-response'"
        )[0]["status"]
        == "aborted"
    )


@pytest.mark.asyncio
async def test_concurrent_retry_repairs_after_owner_task_is_cancelled():
    class ChildAgent(AIChatAgent):
        pass

    start_entered = asyncio.Event()

    async def start_hook(input_json, run_id):
        start_entered.set()
        await asyncio.Event().wait()

    parent = fakes.build_agent()
    child = fakes.FakeChildAgentToolStub(
        inspection={
            "status": "completed",
            "output": "repaired",
            "summary": "repaired",
        },
        start_hook=start_hook,
    )
    _attach_child(parent, child)
    owner = asyncio.create_task(
        parent.run_agent_tool(ChildAgent, input="x", run_id="cancelled-owner")
    )
    await start_entered.wait()
    retry_one = asyncio.create_task(
        parent.run_agent_tool(ChildAgent, input="x", run_id="cancelled-owner")
    )
    retry_two = asyncio.create_task(
        parent.run_agent_tool(ChildAgent, input="x", run_id="cancelled-owner")
    )
    await asyncio.sleep(0)

    owner.cancel()
    with pytest.raises(asyncio.CancelledError):
        await owner
    first_result, second_result = await asyncio.gather(retry_one, retry_two)

    assert first_result == second_result
    assert first_result.status == "completed"
    assert first_result.output == "repaired"
    assert child.start_calls == [('"x"', "cancelled-owner")]
    assert child.inspect_calls == ["cancelled-owner"]


@pytest.mark.asyncio
async def test_cancelled_parent_task_can_retry_real_child_run():
    class ChildAgent(AIChatAgent):
        calls = 0
        first_started = asyncio.Event()

        async def on_chat_message(self, options):
            type(self).calls += 1
            if type(self).calls == 1:
                type(self).first_started.set()
                await asyncio.Event().wait()
            return "recovered"

    child = _child(ChildAgent)
    parent = fakes.build_agent()
    _attach_child(parent, child)
    owner = asyncio.create_task(
        parent.run_agent_tool(ChildAgent, input="x", run_id="cancelled-real-child")
    )
    await ChildAgent.first_started.wait()

    owner.cancel()
    with pytest.raises(asyncio.CancelledError):
        await owner

    assert child.sql(
        "SELECT status FROM cf_ai_chat_agent_tool_runs "
        "WHERE run_id = 'cancelled-real-child'"
    ) == [{"status": "running"}]

    result = await parent.run_agent_tool(
        ChildAgent, input="x", run_id="cancelled-real-child"
    )

    assert result.status == "completed"
    assert result.output == "recovered"
    assert ChildAgent.calls == 2


@pytest.mark.asyncio
async def test_child_run_schema_reconciles_legacy_columns():
    conn = fakes.new_sqlite()
    conn.execute(
        "CREATE TABLE cf_ai_chat_agent_tool_runs ("
        "run_id TEXT PRIMARY KEY, status TEXT NOT NULL)"
    )
    child = fakes.build_chat_agent(lambda options: "unused", conn=conn)

    assert await child._cf_inspect_agent_tool_run("missing") is None
    columns = {
        row["name"]
        for row in child.sql("PRAGMA table_info(cf_ai_chat_agent_tool_runs)")
    }

    assert {
        "request_id",
        "input_json",
        "output_json",
        "summary",
        "error_message",
        "started_at",
        "completed_at",
    } <= columns


@pytest.mark.asyncio
async def test_retry_inspection_failure_honors_abort():
    class ChildAgent(AIChatAgent):
        pass

    abort = asyncio.Event()

    parent = fakes.build_agent()
    _attach_child(
        parent,
        fakes.FakeChildAgentToolStub(
            inspect_hook=lambda _run_id: abort.set(),
            inspect_error=RuntimeError("offline"),
        ),
    )
    parent.sql(
        "INSERT INTO cf_agent_tool_runs "
        "(run_id, agent_type, status, display_order, started_at) "
        "VALUES ('aborted-inspection', 'ChildAgent', 'interrupted', 0, 1)"
    )

    result = await parent.run_agent_tool(
        ChildAgent, input="x", run_id="aborted-inspection", abort=abort
    )

    assert result.status == "aborted"


@pytest.mark.asyncio
async def test_orphaned_child_turn_is_restartable_after_reincarnation():
    class ChildAgent(AIChatAgent):
        async def on_chat_message(self, options):
            return "restarted"

    conn = fakes.new_sqlite()
    child = fakes.build_chat_agent(cls=ChildAgent, conn=conn)
    child.sql(
        "INSERT INTO cf_ai_chat_agent_tool_runs "
        "(run_id, request_id, status, input_json, started_at) "
        "VALUES ('orphan', 'agent-tool-orphan', 'running', '\"x\"', 1)"
    )
    child.sql(
        "INSERT INTO cf_ai_chat_stream_metadata "
        "(id, request_id, status, created_at) "
        "VALUES ('orphan-stream', 'agent-tool-orphan', 'streaming', 1)"
    )

    reincarnated = fakes.build_chat_agent(cls=ChildAgent, conn=conn)
    assert await reincarnated._cf_inspect_agent_tool_run("orphan") is None

    inspection = json.loads(
        await reincarnated._cf_start_agent_tool_run('"x"', "orphan")
    )

    assert inspection["status"] == "completed"
    assert inspection["output"] == "restarted"
    assert (
        reincarnated.sql(
            "SELECT status FROM cf_ai_chat_stream_metadata WHERE id = 'orphan-stream'"
        )[0]["status"]
        == "error"
    )


@pytest.mark.asyncio
async def test_child_repair_does_not_restart_when_stream_storage_is_unavailable():
    class ChildAgent(AIChatAgent):
        provider_calls = 0

        async def on_chat_message(self, options):
            type(self).provider_calls += 1
            return "must not run"

    child = fakes.build_chat_agent(cls=ChildAgent)
    child.sql(
        "INSERT INTO cf_ai_chat_agent_tool_runs "
        "(run_id, request_id, status, input_json, started_at) "
        "VALUES ('uncertain', 'agent-tool-uncertain', 'running', '\"x\"', 1)"
    )

    def unavailable_sql(query, *params):
        raise RuntimeError("storage offline")

    child._resumable._sql = unavailable_sql

    with pytest.raises(StreamStorageUnavailable):
        await child._cf_inspect_agent_tool_run("uncertain")

    assert ChildAgent.provider_calls == 0
    assert child.sql(
        "SELECT status FROM cf_ai_chat_agent_tool_runs WHERE run_id = 'uncertain'"
    ) == [{"status": "running"}]


@pytest.mark.asyncio
async def test_waiter_finishes_hydration_after_owner_is_cancelled():
    class ChildAgent(AIChatAgent):
        pass

    entered = asyncio.Event()

    async def inspect_hook(run_id):
        if not entered.is_set():
            entered.set()
            await asyncio.Event().wait()

    parent = fakes.build_agent()
    child = fakes.FakeChildAgentToolStub(
        inspection={
            "status": "completed",
            "output": "hydrated",
            "summary": "hydrated",
        },
        inspect_hook=inspect_hook,
    )
    _attach_child(parent, child)
    parent.sql(
        "INSERT INTO cf_agent_tool_runs "
        "(run_id, agent_type, status, display_order, started_at, completed_at) "
        "VALUES ('cancelled-hydration', 'ChildAgent', 'completed', 0, 1, 2)"
    )
    owner = asyncio.create_task(
        parent.run_agent_tool(ChildAgent, input="ignored", run_id="cancelled-hydration")
    )
    await entered.wait()
    waiter = asyncio.create_task(
        parent.run_agent_tool(ChildAgent, input="ignored", run_id="cancelled-hydration")
    )

    owner.cancel()
    with pytest.raises(asyncio.CancelledError):
        await owner
    result = await waiter

    assert result.status == "completed"
    assert result.output == "hydrated"
    assert child.inspect_calls == ["cancelled-hydration", "cancelled-hydration"]
    assert (
        parent.sql(
            "SELECT chunks_mirrored FROM cf_agent_tool_runs "
            "WHERE run_id = 'cancelled-hydration'"
        )[0]["chunks_mirrored"]
        == 1
    )
