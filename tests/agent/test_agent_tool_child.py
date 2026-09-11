from __future__ import annotations

import asyncio
import json

import fakes
import pytest

from agents import AIChatAgent
from agents.chat.agent_tools import ChildAgentToolRuns
from agents.chat.resumable_stream import ResumableStream
from agents.chat.turn_queue import TurnQueue


@pytest.mark.asyncio
async def test_child_cancellation_wins_race_with_completion():
    class ChildAgent(AIChatAgent):
        started = asyncio.Event()
        release = asyncio.Event()

        async def on_chat_message(self, options):
            type(self).started.set()
            await type(self).release.wait()
            return "late result"

    child = fakes.build_chat_agent(cls=ChildAgent)
    task = asyncio.create_task(child._cf_start_agent_tool_run('"input"', "run"))
    await ChildAgent.started.wait()

    await child._cf_cancel_agent_tool_run("run", "cancelled by parent")
    ChildAgent.release.set()
    inspection = json.loads(await task)

    assert inspection["status"] == "aborted"
    assert inspection["error"] == "cancelled by parent"
    assert "output" not in inspection


@pytest.mark.asyncio
async def test_child_chunks_preserve_sparse_durable_sequences():
    child = fakes.build_chat_agent(lambda options: "unused")
    child.sql(
        "INSERT INTO cf_ai_chat_agent_tool_runs "
        "(run_id, request_id, status, started_at) "
        "VALUES ('sparse', NULL, 'completed', 1)"
    )
    child.sql(
        "INSERT INTO cf_ai_chat_agent_tool_chunks (run_id, sequence, body) "
        "VALUES ('sparse', 0, 'zero'), ('sparse', 2, 'two')"
    )

    chunks = json.loads(await child._cf_get_agent_tool_chunks("sparse", 0))

    assert chunks == [{"sequence": 2, "body": "two"}]


@pytest.mark.asyncio
async def test_exact_ledger_page_does_not_fall_back_to_stream_chunks():
    child = fakes.build_chat_agent(lambda options: "unused")
    stream_id = child._resumable.start("request", "message")
    child._resumable.store_chunk(stream_id, "wrong-source")
    child._resumable.complete(stream_id)
    child.sql(
        "INSERT INTO cf_ai_chat_agent_tool_runs "
        "(run_id, request_id, status, started_at) "
        "VALUES ('paged', 'request', 'completed', 1)"
    )
    for sequence in range(100):
        child.sql(
            "INSERT INTO cf_ai_chat_agent_tool_chunks (run_id, sequence, body) "
            "VALUES ('paged', ?, ?)",
            sequence,
            f"chunk-{sequence}",
        )

    first = json.loads(await child._cf_get_agent_tool_chunks("paged", -1, 100))
    exhausted = json.loads(await child._cf_get_agent_tool_chunks("paged", 99, 100))

    assert len(first) == 100
    assert exhausted == []


@pytest.mark.asyncio
async def test_child_owner_retries_schema_after_constructor_failure():
    chat = fakes.build_chat_agent(lambda options: "unused")
    sql = chat.sql
    sql("DROP TABLE cf_ai_chat_agent_tool_chunks")
    sql("DROP TABLE cf_ai_chat_agent_tool_runs")
    streams = chat._resumable
    unavailable = True

    def unstable_sql(query, *params):
        if unavailable:
            raise RuntimeError("schema unavailable")
        return sql(query, *params)

    class Adapter:
        def prepare_agent_tool_turn(self, tool_input, run_id):
            return frozenset()

        async def run_agent_tool_turn(
            self, run_id, tool_input, request_id, abort, context
        ):
            return None

        async def collect_agent_tool_result(
            self,
            run_id,
            tool_input,
            *,
            previous_assistant_ids=None,
            message_id=None,
        ):
            return None, ""

    adapter = Adapter()
    owner = ChildAgentToolRuns(
        unstable_sql,
        streams,
        TurnQueue(),
        adapter.prepare_agent_tool_turn,
        adapter.run_agent_tool_turn,
        adapter.collect_agent_tool_result,
    )
    unavailable = False

    assert await owner.inspect("missing") is None
    tables = {
        row["name"]
        for row in sql(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name LIKE 'cf_ai_chat_agent_tool_%'"
        )
    }
    assert tables == {
        "cf_ai_chat_agent_tool_runs",
        "cf_ai_chat_agent_tool_chunks",
    }


@pytest.mark.asyncio
async def test_child_recovery_collects_without_preparing_or_executing_again():
    sql = fakes.make_sql()
    streams = ResumableStream(sql, "response")
    streams.prepare()
    calls = []

    def prepare(tool_input, run_id):
        raise AssertionError("recovery must not prepare a new provider turn")

    async def execute(run_id, tool_input, request_id, abort, context):
        raise AssertionError("recovery must not execute the provider again")

    async def collect(
        run_id,
        tool_input,
        *,
        previous_assistant_ids=None,
        message_id=None,
    ):
        calls.append((run_id, tool_input, message_id))
        return "recovered", "recovered"

    owner = ChildAgentToolRuns(
        sql,
        streams,
        TurnQueue(),
        prepare,
        execute,
        collect,
    )
    owner.prepare()
    sql(
        "INSERT INTO cf_ai_chat_agent_tool_runs "
        "(run_id, request_id, status, input_json, started_at) "
        "VALUES ('recover', 'request', 'running', '\"input\"', 1)"
    )
    stream_id = streams.start("request", "assistant")
    streams.complete(stream_id)

    inspection = await owner.inspect("recover")

    assert inspection is not None
    assert inspection.status == "completed"
    assert inspection.output == "recovered"
    assert calls == [("recover", "input", "assistant")]
