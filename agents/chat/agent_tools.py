from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from ..core.agent_tool_protocol import CHILD_AGENT_TOOL_CHUNK_PAGE_SIZE
from ..core.utils import MISSING, dumps_wire, error_message, now_ms
from .resumable_stream import ResumableStream, SqlFn
from .turn_queue import TurnContext, TurnQueue

ChildRunStatus = Literal["running", "completed", "error", "aborted"]
ChildTerminalStatus = Literal["completed", "error", "aborted"]


@dataclass(frozen=True)
class ChildRunInspection:
    run_id: str
    request_id: str | None
    status: ChildRunStatus
    started_at: int
    completed_at: int | None = None
    output: Any = MISSING
    summary: str | None = None
    error: str | None = None

    def to_wire(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "runId": self.run_id,
            "requestId": self.request_id,
            "status": self.status,
            "startedAt": self.started_at,
        }
        if self.completed_at is not None:
            result["completedAt"] = self.completed_at
        if self.status == "completed":
            if self.output is not MISSING:
                result["output"] = self.output
            result["summary"] = self.summary or ""
        elif self.error is not None:
            result["error"] = self.error
        return result


@dataclass(frozen=True)
class ChildChunk:
    sequence: int
    body: str

    def to_wire(self) -> dict[str, Any]:
        return {"sequence": self.sequence, "body": self.body}


class ChildTurnPrepare(Protocol):
    def __call__(
        self,
        input: Any,
        run_id: str,
    ) -> frozenset[str] | Awaitable[frozenset[str]]: ...


class ChildTurnExecute(Protocol):
    async def __call__(
        self,
        run_id: str,
        input: Any,
        request_id: str,
        abort: asyncio.Event,
        context: TurnContext,
    ) -> str | None: ...


class ChildTurnCollect(Protocol):
    async def __call__(
        self,
        run_id: str,
        input: Any,
        *,
        previous_assistant_ids: frozenset[str] | None = None,
        message_id: str | None = None,
    ) -> tuple[Any, str]: ...


class ChildAgentToolRuns:
    def __init__(
        self,
        sql: SqlFn,
        streams: ResumableStream,
        queue: TurnQueue,
        prepare_turn: ChildTurnPrepare,
        execute_turn: ChildTurnExecute,
        collect_result: ChildTurnCollect,
    ) -> None:
        self._sql = sql
        self._streams = streams
        self._queue = queue
        self._prepare_turn = prepare_turn
        self._execute_turn = execute_turn
        self._collect_result = collect_result
        self._aborts: dict[str, asyncio.Event] = {}
        self._run_ids_by_request: dict[str, str] = {}
        self._chunk_sequences: dict[str, int] = {}
        self._turn_errors: dict[str, str] = {}
        self._active_run_ids: set[str] = set()

    def prepare(self) -> None:
        """Prepare child-owned tables during host startup."""
        self._ensure_tables()

    def abort_active(self) -> None:
        for abort in self._aborts.values():
            abort.set()

    def capture_chunk(self, request_id: str, body: str) -> None:
        run_id = self._run_ids_by_request.get(request_id)
        if run_id is None:
            return
        sequence = self._chunk_sequences.get(run_id, 0)
        self._sql(
            "INSERT INTO cf_ai_chat_agent_tool_chunks (run_id, sequence, body) "
            "VALUES (?, ?, ?)",
            run_id,
            sequence,
            body,
        )
        self._chunk_sequences[run_id] = sequence + 1

    def capture_error(self, request_id: str, error: str) -> None:
        run_id = self._run_ids_by_request.get(request_id)
        if run_id is not None:
            self._turn_errors[run_id] = error

    async def start(self, input_json: str, run_id: str) -> ChildRunInspection:
        self._ensure_tables()
        existing = await self._reconcile(run_id)
        if existing is not None:
            return existing

        input = json.loads(input_json)
        request_id = f"agent-tool-{run_id}"
        self._sql(
            "INSERT INTO cf_ai_chat_agent_tool_runs "
            "(run_id, request_id, status, input_json, started_at) "
            "VALUES (?, ?, 'running', ?, ?)",
            run_id,
            request_id,
            input_json,
            now_ms(),
        )

        abort = asyncio.Event()
        self._aborts[request_id] = abort
        try:

            async def run(context: TurnContext) -> None:
                await self._execute(run_id, input, request_id, abort, context)

            result = await self._queue.enqueue(request_id, run)
            if result.status == "stale":
                self._settle(
                    run_id,
                    "aborted",
                    error="chat history was cleared before the turn started",
                )
        except asyncio.CancelledError:
            if abort.is_set():
                self._settle(run_id, "aborted", error="cancelled")
            raise
        except Exception as exc:  # noqa: BLE001
            self._settle(run_id, "error", error=error_message(exc))
        finally:
            self._aborts.pop(request_id, None)

        inspection = self._inspection(run_id)
        if inspection is None:
            raise RuntimeError(f"agent tool run {run_id!r} disappeared")
        return inspection

    async def cancel(self, run_id: str, reason: str | None = None) -> None:
        self._ensure_tables()
        rows = self._sql(
            "SELECT request_id FROM cf_ai_chat_agent_tool_runs WHERE run_id = ?",
            run_id,
        )
        if not rows:
            return
        abort = self._aborts.get(rows[0]["request_id"])
        if abort is not None:
            abort.set()
        self._settle(run_id, "aborted", error=reason)

    async def inspect(self, run_id: str) -> ChildRunInspection | None:
        self._ensure_tables()
        return await self._reconcile(run_id)

    def chunks(
        self,
        run_id: str,
        after_sequence: int = -1,
        limit: int = CHILD_AGENT_TOOL_CHUNK_PAGE_SIZE,
    ) -> list[ChildChunk]:
        self._ensure_tables()
        has_ledger_chunks = self._sql(
            "SELECT 1 FROM cf_ai_chat_agent_tool_chunks WHERE run_id = ? LIMIT 1",
            run_id,
        )
        if has_ledger_chunks:
            rows = self._sql(
                "SELECT sequence, body FROM cf_ai_chat_agent_tool_chunks "
                "WHERE run_id = ? AND sequence > ? ORDER BY sequence ASC LIMIT ?",
                run_id,
                after_sequence,
                limit,
            )
            return [ChildChunk(row["sequence"], row["body"]) for row in rows]

        runs = self._sql(
            "SELECT request_id FROM cf_ai_chat_agent_tool_runs WHERE run_id = ?",
            run_id,
        )
        if not runs or runs[0]["request_id"] is None:
            return []
        bodies = self._streams.completed_bodies_for_request(
            runs[0]["request_id"],
            after_sequence=after_sequence,
            limit=limit,
        )
        return [
            ChildChunk(after_sequence + offset + 1, body)
            for offset, body in enumerate(bodies)
        ]

    async def _execute(
        self,
        run_id: str,
        input: Any,
        request_id: str,
        abort: asyncio.Event,
        context: TurnContext,
    ) -> None:
        if abort.is_set():
            self._settle(run_id, "aborted", error="cancelled")
            return

        self._run_ids_by_request[request_id] = run_id
        self._chunk_sequences[run_id] = 0
        self._active_run_ids.add(run_id)
        try:
            previous_ids = self._prepare_turn(input, run_id)
            if inspect.isawaitable(previous_ids):
                previous_ids = await previous_ids
            turn_error = await self._execute_turn(
                run_id,
                input,
                request_id,
                abort,
                context,
            )
            turn_error = turn_error or self._turn_errors.get(run_id)
            if abort.is_set():
                self._settle(run_id, "aborted", error="cancelled")
            elif turn_error is not None:
                self._settle(run_id, "error", error=turn_error)
            else:
                output, summary = await self._collect_result(
                    run_id,
                    input,
                    previous_assistant_ids=previous_ids,
                )
                self._settle(run_id, "completed", output=output, summary=summary)
        except Exception as exc:  # noqa: BLE001
            self._settle(run_id, "error", error=error_message(exc))
        finally:
            self._active_run_ids.discard(run_id)
            self._run_ids_by_request.pop(request_id, None)
            self._chunk_sequences.pop(run_id, None)
            self._turn_errors.pop(run_id, None)

    async def _reconcile(self, run_id: str) -> ChildRunInspection | None:
        inspection = self._inspection(run_id)
        if inspection is None or inspection.status != "running":
            return inspection
        if run_id in self._active_run_ids:
            return inspection

        rows = self._sql(
            "SELECT request_id, input_json FROM cf_ai_chat_agent_tool_runs "
            "WHERE run_id = ?",
            run_id,
        )
        request_id = rows[0]["request_id"]
        if request_id is None:
            return inspection

        stream = self._streams.latest_for_request(request_id)
        restart = stream is None
        if stream is not None and stream.status == "streaming":
            restart = self._streams.mark_orphaned(stream.stream_id)
            if not restart:
                stream = self._streams.latest_for_request(request_id)
                restart = stream is None

        if restart:
            self._sql(
                "DELETE FROM cf_ai_chat_agent_tool_chunks WHERE run_id = ?", run_id
            )
            self._sql(
                "DELETE FROM cf_ai_chat_agent_tool_runs WHERE run_id = ? "
                "AND status = 'running'",
                run_id,
            )
            return None

        if stream is not None and stream.status == "error":
            self._settle(
                run_id,
                "error",
                error="Agent tool turn failed before its child row settled",
            )
            return self._inspection(run_id)

        try:
            input_json = rows[0]["input_json"]
            input = None if input_json is None else json.loads(input_json)
            output, summary = await self._collect_result(
                run_id,
                input,
                message_id=None if stream is None else stream.message_id,
            )
            self._settle(run_id, "completed", output=output, summary=summary)
        except Exception as exc:  # noqa: BLE001
            self._settle(run_id, "error", error=error_message(exc))
        return self._inspection(run_id)

    def _settle(
        self,
        run_id: str,
        status: ChildTerminalStatus,
        *,
        output: Any = MISSING,
        summary: str | None = None,
        error: str | None = None,
    ) -> None:
        output_json = (
            dumps_wire(output)
            if status == "completed" and output is not MISSING
            else None
        )
        self._sql(
            "UPDATE cf_ai_chat_agent_tool_runs SET status = ?, output_json = ?, "
            "summary = ?, error_message = ?, completed_at = ? WHERE run_id = ? "
            "AND status = 'running'",
            status,
            output_json,
            summary,
            error,
            now_ms(),
            run_id,
        )

    def _inspection(self, run_id: str) -> ChildRunInspection | None:
        rows = self._sql(
            "SELECT run_id, request_id, status, output_json, summary, "
            "error_message, started_at, completed_at "
            "FROM cf_ai_chat_agent_tool_runs WHERE run_id = ?",
            run_id,
        )
        if not rows:
            return None
        row = rows[0]
        status = row["status"]
        if status not in ("running", "completed", "error", "aborted"):
            raise ValueError(f"unknown child agent-tool status: {status!r}")
        output = MISSING
        if status == "completed" and row["output_json"] is not None:
            output = json.loads(row["output_json"])
        return ChildRunInspection(
            run_id=row["run_id"],
            request_id=row["request_id"],
            status=status,
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            output=output,
            summary=row["summary"],
            error=row["error_message"],
        )

    def _ensure_tables(self) -> None:
        self._sql("""
        CREATE TABLE IF NOT EXISTS cf_ai_chat_agent_tool_runs (
            run_id TEXT PRIMARY KEY,
            request_id TEXT,
            status TEXT NOT NULL,
            input_json TEXT,
            output_json TEXT,
            summary TEXT,
            error_message TEXT,
            started_at INTEGER NOT NULL,
            completed_at INTEGER
        )
        """)
        self._reconcile_columns()
        self._sql(
            "CREATE INDEX IF NOT EXISTS idx_ai_chat_agent_tool_request_id "
            "ON cf_ai_chat_agent_tool_runs(request_id)"
        )
        self._sql("""
        CREATE TABLE IF NOT EXISTS cf_ai_chat_agent_tool_chunks (
            run_id TEXT NOT NULL,
            sequence INTEGER NOT NULL,
            body TEXT NOT NULL,
            PRIMARY KEY (run_id, sequence)
        )
        """)

    def _reconcile_columns(self) -> None:
        columns = {
            row["name"]
            for row in self._sql("PRAGMA table_info(cf_ai_chat_agent_tool_runs)")
        }
        additions = (
            (
                "request_id",
                "ALTER TABLE cf_ai_chat_agent_tool_runs ADD COLUMN request_id TEXT",
            ),
            (
                "input_json",
                "ALTER TABLE cf_ai_chat_agent_tool_runs ADD COLUMN input_json TEXT",
            ),
            (
                "output_json",
                "ALTER TABLE cf_ai_chat_agent_tool_runs ADD COLUMN output_json TEXT",
            ),
            (
                "summary",
                "ALTER TABLE cf_ai_chat_agent_tool_runs ADD COLUMN summary TEXT",
            ),
            (
                "error_message",
                "ALTER TABLE cf_ai_chat_agent_tool_runs ADD COLUMN error_message TEXT",
            ),
            (
                "started_at",
                (
                    "ALTER TABLE cf_ai_chat_agent_tool_runs ADD COLUMN started_at "
                    "INTEGER NOT NULL DEFAULT 0"
                ),
            ),
            (
                "completed_at",
                (
                    "ALTER TABLE cf_ai_chat_agent_tool_runs ADD COLUMN completed_at "
                    "INTEGER"
                ),
            ),
        )
        for name, query in additions:
            if name not in columns:
                self._sql(query)
