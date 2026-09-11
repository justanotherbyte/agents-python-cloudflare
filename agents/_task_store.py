from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

from .lifecycle import LifecycleSql

type _TaskRunState = Literal[
    "pending", "running", "waiting", "completed", "failed", "cancelled"
]
type _TerminalTaskRunState = Literal["completed", "failed", "cancelled"]
type _TaskStepState = Literal["running", "waiting", "completed", "failed"]
type _TaskWaitReason = Literal["sleep", "retry", "memory"]
_RUN_COLUMN_NAMES = (
    "run_id",
    "definition",
    "input",
    "state",
    "result",
    "error_name",
    "error_message",
    "status_message",
    "metadata",
    "idempotency_key",
    "retain",
    "attempt",
    "generation",
    "next_at",
    "wait_reason",
    "cancel_requested",
    "cancel_reason",
    "created_at",
    "started_at",
    "updated_at",
    "settled_at",
)
_STEP_COLUMN_NAMES = (
    "run_id",
    "step_name",
    "kind",
    "state",
    "result",
    "error_name",
    "error_message",
    "attempt",
    "next_at",
    "created_at",
    "started_at",
    "updated_at",
    "completed_at",
)


@dataclass(frozen=True, slots=True)
class _TaskRunRow:
    run_id: str
    definition: str
    input: str | None
    state: _TaskRunState
    result: str | None
    error_name: str | None
    error_message: str | None
    status_message: str | None
    metadata: str | None
    idempotency_key: str | None
    retain: int
    attempt: int
    generation: str | None
    next_at: int | None
    wait_reason: _TaskWaitReason | None
    cancel_requested: int
    cancel_reason: str | None
    created_at: int
    started_at: int | None
    updated_at: int
    settled_at: int | None


@dataclass(frozen=True, slots=True)
class _TaskStepRow:
    run_id: str
    step_name: str
    kind: Literal["do", "sleep"]
    state: _TaskStepState
    result: str | None
    error_name: str | None
    error_message: str | None
    attempt: int
    next_at: int | None
    created_at: int
    started_at: int | None
    updated_at: int
    completed_at: int | None


class _StoreFenceRejected(Exception):
    pass


class _TaskStore:
    def __init__(
        self,
        sql: LifecycleSql,
        transaction_sync: Callable[[Callable[[], object]], object],
    ) -> None:
        self._sql = sql
        self._transaction_sync = transaction_sync

    def prepare(self) -> None:
        self._sql.execute("""
        CREATE TABLE IF NOT EXISTS cf_agents_task_runs (
          run_id TEXT PRIMARY KEY,
          definition TEXT NOT NULL,
          input TEXT,
          state TEXT NOT NULL CHECK (state IN (
            'pending', 'running', 'waiting', 'completed', 'failed', 'cancelled'
          )),
          result TEXT,
          error_name TEXT,
          error_message TEXT,
          status_message TEXT,
          metadata TEXT,
          idempotency_key TEXT UNIQUE,
          retain INTEGER NOT NULL DEFAULT 1,
          attempt INTEGER NOT NULL DEFAULT 0,
          generation TEXT,
          next_at INTEGER,
          wait_reason TEXT,
          cancel_requested INTEGER NOT NULL DEFAULT 0,
          cancel_reason TEXT,
          created_at INTEGER NOT NULL,
          started_at INTEGER,
          updated_at INTEGER NOT NULL,
          settled_at INTEGER
        ) WITHOUT ROWID
        """)

        self._sql.execute("""
        CREATE INDEX IF NOT EXISTS cf_agents_task_runs_definition
        ON cf_agents_task_runs (definition, created_at)
        """)
        self._sql.execute("""
        CREATE TABLE IF NOT EXISTS cf_agents_task_steps (
          run_id TEXT NOT NULL,
          step_name TEXT NOT NULL,
          kind TEXT NOT NULL CHECK (kind IN ('do', 'sleep')),
          state TEXT NOT NULL CHECK (state IN (
            'running', 'waiting', 'completed', 'failed'
          )),
          result TEXT,
          error_name TEXT,
          error_message TEXT,
          attempt INTEGER NOT NULL DEFAULT 0,
          next_at INTEGER,
          created_at INTEGER NOT NULL,
          started_at INTEGER,
          updated_at INTEGER NOT NULL,
          completed_at INTEGER,
          PRIMARY KEY (run_id, step_name)
        ) WITHOUT ROWID
        """)

    def schema_is_compatible(self) -> bool:
        for table, expected in (
            ("cf_agents_task_runs", _RUN_COLUMN_NAMES),
            ("cf_agents_task_steps", _STEP_COLUMN_NAMES),
        ):
            columns = self._sql.execute(f"PRAGMA table_info({table})")
            if columns and tuple(row["name"] for row in columns) != expected:
                return False
        return True

    def get(self, run_id: str) -> _TaskRunRow | None:
        rows = self._sql.execute(
            "SELECT * FROM cf_agents_task_runs WHERE run_id = ?",
            run_id,
        )
        return _row(rows[0]) if rows else None

    def get_by_idempotency_key(self, key: str) -> _TaskRunRow | None:
        rows = self._sql.execute(
            "SELECT * FROM cf_agents_task_runs WHERE idempotency_key = ?",
            key,
        )
        return _row(rows[0]) if rows else None

    def insert_pending(
        self,
        *,
        run_id: str,
        definition: str,
        input_json: str | None,
        metadata_json: str | None,
        idempotency_key: str | None,
        retain: bool,
        created_at: int,
    ) -> None:
        self._sql.execute(
            """
            INSERT INTO cf_agents_task_runs
              (run_id, definition, input, state, metadata, idempotency_key,
               retain, attempt, next_at, cancel_requested, created_at, updated_at)
            VALUES (?, ?, ?, 'pending', ?, ?, ?, 0, ?, 0, ?, ?)
            """,
            run_id,
            definition,
            input_json,
            metadata_json,
            idempotency_key,
            int(retain),
            created_at,
            created_at,
            created_at,
        )

    def reconcile_deadlines(self, current_time: int) -> None:
        self._sql.execute(
            """
            UPDATE cf_agents_task_runs
            SET next_at = ?, updated_at = ?
            WHERE state = 'running'
              AND (generation IS NOT NULL OR wait_reason IS NOT 'memory')
            """,
            current_time,
            current_time,
        )
        self._sql.execute(
            """
            UPDATE cf_agents_task_runs
            SET next_at = ?, updated_at = ?
            WHERE state IN ('pending', 'waiting') AND next_at IS NULL
            """,
            current_time,
            current_time,
        )

    def non_terminal_deadlines(self) -> tuple[tuple[str, int], ...]:
        rows = self._sql.execute(
            """
            SELECT run_id, next_at FROM cf_agents_task_runs
            WHERE state IN ('pending', 'waiting', 'running') AND next_at IS NOT NULL
            ORDER BY run_id
            """
        )
        return tuple(
            (cast(str, row["run_id"]), cast(int, row["next_at"])) for row in rows
        )

    def authoritative_deadline(self, run_id: str) -> int | None:
        rows = self._sql.execute(
            """
            SELECT next_at FROM cf_agents_task_runs
            WHERE run_id = ? AND state IN ('pending', 'waiting', 'running')
            """,
            run_id,
        )
        return cast(int | None, rows[0]["next_at"]) if rows else None

    def claim_run(
        self,
        run_id: str,
        generation: str,
        claimed_at: int,
        claim_deadline: int,
    ) -> _TaskRunRow | None:
        self._sql.execute(
            """
            UPDATE cf_agents_task_runs
            SET state = 'running', attempt = attempt + 1, generation = ?,
                started_at = coalesce(started_at, ?), next_at = ?,
                wait_reason = NULL, updated_at = ?
            WHERE run_id = ?
              AND state IN ('pending', 'waiting', 'running')
              AND cancel_requested = 0
              AND (next_at IS NULL OR next_at <= ?)
            """,
            generation,
            claimed_at,
            claim_deadline,
            claimed_at,
            run_id,
            claimed_at,
        )
        if _changes(self._sql) == 0:
            return None
        return self.get(run_id)

    def refresh_claim(
        self,
        run_id: str,
        generation: str,
        next_at: int,
        updated_at: int,
    ) -> bool:
        self._sql.execute(
            """
            UPDATE cf_agents_task_runs
            SET next_at = ?, updated_at = ?
            WHERE run_id = ? AND generation = ? AND state = 'running'
              AND cancel_requested = 0
            """,
            next_at,
            updated_at,
            run_id,
            generation,
        )
        return _changes(self._sql) > 0

    def extend_active_claim(
        self,
        run_id: str,
        generation: str,
        next_at: int,
        updated_at: int,
    ) -> bool:
        self._sql.execute(
            """
            UPDATE cf_agents_task_runs
            SET next_at = ?, updated_at = ?
            WHERE run_id = ? AND generation = ? AND state = 'running'
            """,
            next_at,
            updated_at,
            run_id,
            generation,
        )
        return _changes(self._sql) > 0

    def newest_running_step(self, run_id: str) -> _TaskStepRow | None:
        rows = self._sql.execute(
            """
            SELECT * FROM cf_agents_task_steps
            WHERE run_id = ? AND state = 'running'
            ORDER BY started_at DESC, step_name DESC
            LIMIT 1
            """,
            run_id,
        )
        return _step_row(rows[0]) if rows else None

    def get_step(self, run_id: str, step_name: str) -> _TaskStepRow | None:
        rows = self._sql.execute(
            """
            SELECT * FROM cf_agents_task_steps
            WHERE run_id = ? AND step_name = ?
            """,
            run_id,
            step_name,
        )
        return _step_row(rows[0]) if rows else None

    def count_steps(self, run_id: str) -> int:
        rows = self._sql.execute(
            "SELECT count(*) AS count FROM cf_agents_task_steps WHERE run_id = ?",
            run_id,
        )
        return cast(int, rows[0]["count"])

    def insert_step(
        self,
        run_id: str,
        generation: str,
        step_name: str,
        started_at: int,
    ) -> bool:
        self._sql.execute(
            """
            INSERT OR IGNORE INTO cf_agents_task_steps
              (run_id, step_name, kind, state, attempt, created_at, started_at,
               updated_at)
            SELECT ?, ?, 'do', 'running', 1, ?, ?, ?
            WHERE EXISTS (
              SELECT 1 FROM cf_agents_task_runs
              WHERE run_id = ? AND generation = ? AND state = 'running'
                AND cancel_requested = 0
            )
            """,
            run_id,
            step_name,
            started_at,
            started_at,
            started_at,
            run_id,
            generation,
        )
        return _changes(self._sql) > 0

    def insert_completed_sleep(
        self,
        run_id: str,
        generation: str,
        step_name: str,
        completed_at: int,
    ) -> bool:
        self._sql.execute(
            """
            INSERT OR IGNORE INTO cf_agents_task_steps
              (run_id, step_name, kind, state, attempt, created_at,
               updated_at, completed_at)
            SELECT ?, ?, 'sleep', 'completed', 0, ?, ?, ?
            WHERE EXISTS (
              SELECT 1 FROM cf_agents_task_runs
              WHERE run_id = ? AND generation = ? AND state = 'running'
                AND cancel_requested = 0
            )
            """,
            run_id,
            step_name,
            completed_at,
            completed_at,
            completed_at,
            run_id,
            generation,
        )
        return _changes(self._sql) > 0

    def park_new_sleep(
        self,
        run_id: str,
        generation: str,
        step_name: str,
        wake_at: int,
        updated_at: int,
    ) -> bool:
        # TODO: is this worth being a functools.partial?
        def park() -> None:
            self._sql.execute(
                """
                INSERT OR IGNORE INTO cf_agents_task_steps
                  (run_id, step_name, kind, state, attempt, next_at,
                   created_at, updated_at)
                SELECT ?, ?, 'sleep', 'waiting', 0, ?, ?, ?
                WHERE EXISTS (
                  SELECT 1 FROM cf_agents_task_runs
              WHERE run_id = ? AND generation = ? AND state = 'running'
                AND cancel_requested = 0
                )
                """,
                run_id,
                step_name,
                wake_at,
                updated_at,
                updated_at,
                run_id,
                generation,
            )
            if _changes(self._sql) == 0:
                raise _StoreFenceRejected
            self._park_run(run_id, generation, "sleep", wake_at, updated_at)

        try:
            self._transaction_sync(park)
        except _StoreFenceRejected:
            return False
        return True

    def park_existing_sleep(
        self,
        run_id: str,
        generation: str,
        wake_at: int,
        updated_at: int,
    ) -> bool:
        try:
            self._transaction_sync(
                lambda: self._park_run(
                    run_id,
                    generation,
                    "sleep",
                    wake_at,
                    updated_at,
                )
            )
        except _StoreFenceRejected:
            return False
        return True

    def complete_sleep(
        self,
        run_id: str,
        generation: str,
        step_name: str,
        completed_at: int,
    ) -> bool:
        self._sql.execute(
            """
            UPDATE cf_agents_task_steps
            SET state = 'completed', next_at = NULL, updated_at = ?,
                completed_at = ?
            WHERE run_id = ? AND step_name = ? AND kind = 'sleep'
              AND state IN ('running', 'waiting')
              AND EXISTS (
                SELECT 1 FROM cf_agents_task_runs
                WHERE run_id = ? AND generation = ? AND state = 'running'
                  AND cancel_requested = 0
              )
            """,
            completed_at,
            completed_at,
            run_id,
            step_name,
            run_id,
            generation,
        )
        return _changes(self._sql) > 0

    def restart_step(
        self,
        run_id: str,
        generation: str,
        step_name: str,
        started_at: int,
    ) -> _TaskStepRow | None:
        self._sql.execute(
            """
            UPDATE cf_agents_task_steps
            SET state = 'running', attempt = attempt + 1, next_at = NULL,
                started_at = ?, updated_at = ?
            WHERE run_id = ? AND step_name = ?
              AND (
                state = 'running'
                OR (state = 'waiting' AND (next_at IS NULL OR next_at <= ?))
              )
              AND EXISTS (
                SELECT 1 FROM cf_agents_task_runs
                WHERE run_id = ? AND generation = ? AND state = 'running'
                  AND cancel_requested = 0
              )
            """,
            started_at,
            started_at,
            run_id,
            step_name,
            started_at,
            run_id,
            generation,
        )
        if _changes(self._sql) == 0:
            return None
        return self.get_step(run_id, step_name)

    def complete_step(
        self,
        run_id: str,
        generation: str,
        step_name: str,
        attempt: int,
        result: str | None,
        completed_at: int,
    ) -> bool:
        self._sql.execute(
            """
            UPDATE cf_agents_task_steps
            SET state = 'completed', result = ?, next_at = NULL,
                updated_at = ?, completed_at = ?
            WHERE run_id = ? AND step_name = ?
              AND state = 'running' AND attempt = ?
              AND EXISTS (
                SELECT 1 FROM cf_agents_task_runs
                WHERE run_id = ? AND generation = ? AND state = 'running'
                  AND cancel_requested = 0
              )
            """,
            result,
            completed_at,
            completed_at,
            run_id,
            step_name,
            attempt,
            run_id,
            generation,
        )
        return _changes(self._sql) > 0

    def fail_step(
        self,
        run_id: str,
        generation: str,
        step_name: str,
        attempt: int,
        error_name: str,
        error_message: str,
        failed_at: int,
    ) -> bool:
        self._sql.execute(
            """
            UPDATE cf_agents_task_steps
            SET state = 'failed', error_name = ?, error_message = ?,
                next_at = NULL, updated_at = ?
            WHERE run_id = ? AND step_name = ?
              AND state = 'running' AND attempt = ?
              AND EXISTS (
                SELECT 1 FROM cf_agents_task_runs
                WHERE run_id = ? AND generation = ? AND state = 'running'
                  AND cancel_requested = 0
              )
            """,
            error_name,
            error_message,
            failed_at,
            run_id,
            step_name,
            attempt,
            run_id,
            generation,
        )
        return _changes(self._sql) > 0

    def park_step_retry(
        self,
        run_id: str,
        generation: str,
        step_name: str,
        attempt: int,
        wake_at: int,
        updated_at: int,
    ) -> bool:
        def park() -> None:
            self._sql.execute(
                """
                UPDATE cf_agents_task_steps
                SET state = 'waiting', next_at = ?, updated_at = ?
                WHERE run_id = ? AND step_name = ?
                  AND state = 'running' AND attempt = ?
                  AND EXISTS (
                    SELECT 1 FROM cf_agents_task_runs
                    WHERE run_id = ? AND generation = ? AND state = 'running'
                      AND cancel_requested = 0
                  )
                """,
                wake_at,
                updated_at,
                run_id,
                step_name,
                attempt,
                run_id,
                generation,
            )
            if _changes(self._sql) == 0:
                raise _StoreFenceRejected
            self._park_run(run_id, generation, "retry", wake_at, updated_at)

        try:
            self._transaction_sync(park)
        except _StoreFenceRejected:
            return False
        return True

    def _park_run(
        self,
        run_id: str,
        generation: str,
        reason: Literal["sleep", "retry"],
        wake_at: int,
        updated_at: int,
    ) -> None:
        self._sql.execute(
            """
            UPDATE cf_agents_task_runs
            SET state = 'waiting', generation = NULL, next_at = ?,
                wait_reason = ?, updated_at = ?
            WHERE run_id = ? AND generation = ? AND state = 'running'
              AND cancel_requested = 0
            """,
            wake_at,
            reason,
            updated_at,
            run_id,
            generation,
        )
        if _changes(self._sql) == 0:
            raise _StoreFenceRejected

    def update_status(
        self,
        run_id: str,
        generation: str,
        message: str,
        updated_at: int,
    ) -> bool:
        self._sql.execute(
            """
            UPDATE cf_agents_task_runs
            SET status_message = ?, updated_at = ?
            WHERE run_id = ? AND generation = ? AND state = 'running'
              AND cancel_requested = 0
            """,
            message,
            updated_at,
            run_id,
            generation,
        )
        return _changes(self._sql) > 0

    def settle_completed(
        self,
        run_id: str,
        generation: str,
        result: str | None,
        settled_at: int,
    ) -> bool:
        self._sql.execute(
            """
            UPDATE cf_agents_task_runs
            SET state = 'completed', result = ?, generation = NULL,
                next_at = NULL, settled_at = ?, updated_at = ?
            WHERE run_id = ? AND generation = ? AND state = 'running'
              AND cancel_requested = 0
            """,
            result,
            settled_at,
            settled_at,
            run_id,
            generation,
        )
        return _changes(self._sql) > 0

    def settle_failed(
        self,
        run_id: str,
        generation: str,
        error_name: str,
        error_message: str,
        settled_at: int,
    ) -> bool:
        self._sql.execute(
            """
            UPDATE cf_agents_task_runs
            SET state = 'failed', error_name = ?, error_message = ?,
                generation = NULL, next_at = NULL, settled_at = ?, updated_at = ?
            WHERE run_id = ? AND generation = ? AND state = 'running'
              AND cancel_requested = 0
            """,
            error_name,
            error_message,
            settled_at,
            settled_at,
            run_id,
            generation,
        )
        return _changes(self._sql) > 0

    def request_cancel(
        self,
        run_id: str,
        generation: str,
        reason: str | None,
        requested_at: int,
    ) -> bool:
        self._sql.execute(
            """
            UPDATE cf_agents_task_runs
            SET cancel_requested = 1, cancel_reason = ?, next_at = ?, updated_at = ?
            WHERE run_id = ? AND generation = ? AND state = 'running'
            """,
            reason,
            requested_at,
            requested_at,
            run_id,
            generation,
        )
        return _changes(self._sql) > 0

    def settle_cancelled_claim(
        self,
        run_id: str,
        generation: str,
        reason: str | None,
        settled_at: int,
    ) -> bool:
        self._sql.execute(
            """
            UPDATE cf_agents_task_runs
            SET state = 'cancelled', cancel_requested = 1, cancel_reason = ?,
                generation = NULL, next_at = NULL, settled_at = ?, updated_at = ?
            WHERE run_id = ? AND generation = ? AND state = 'running'
            """,
            reason,
            settled_at,
            settled_at,
            run_id,
            generation,
        )
        return _changes(self._sql) > 0

    def backoff_memory_claim(
        self,
        run_id: str,
        generation: str,
        next_at: int,
        updated_at: int,
    ) -> bool:
        self._sql.execute(
            """
            UPDATE cf_agents_task_runs
            SET generation = NULL, next_at = ?, wait_reason = 'memory', updated_at = ?
            WHERE run_id = ? AND generation = ? AND state = 'running'
              AND cancel_requested = 0
            """,
            next_at,
            updated_at,
            run_id,
            generation,
        )
        return _changes(self._sql) > 0

    def settle_failed_unclaimed(
        self,
        run_id: str,
        error_name: str,
        error_message: str,
        settled_at: int,
    ) -> bool:
        self._sql.execute(
            """
            UPDATE cf_agents_task_runs
            SET state = 'failed', error_name = ?, error_message = ?,
                generation = NULL, next_at = NULL, settled_at = ?, updated_at = ?
            WHERE run_id = ? AND state IN ('pending', 'waiting', 'running')
              AND generation IS NULL AND cancel_requested = 0
            """,
            error_name,
            error_message,
            settled_at,
            settled_at,
            run_id,
        )
        return _changes(self._sql) > 0

    def list(
        self,
        *,
        definition: str | None,
        states: Sequence[_TaskRunState],
        limit: int,
    ) -> tuple[_TaskRunRow, ...]:
        query = "SELECT * FROM cf_agents_task_runs WHERE 1 = 1"
        params: list[object] = []
        if definition is not None:
            query += " AND definition = ?"
            params.append(definition)
        if states:
            query += " AND state IN (" + ", ".join("?" for _ in states) + ")"
            params.extend(states)
        query += " ORDER BY created_at DESC, run_id DESC LIMIT ?"
        params.append(limit)
        return tuple(_row(row) for row in self._sql.execute(query, *params))

    def unretained_terminal_runs(self) -> tuple[_TaskRunRow, ...]:
        rows = self._sql.execute(
            """
            SELECT * FROM cf_agents_task_runs
            WHERE retain = 0 AND state IN ('completed', 'failed', 'cancelled')
            ORDER BY settled_at ASC, run_id ASC
            """
        )
        return tuple(_row(row) for row in rows)

    def cancel_parked(self, run_id: str, reason: str | None, settled_at: int) -> bool:
        self._sql.execute(
            """
            UPDATE cf_agents_task_runs
            SET state = 'cancelled', cancel_requested = 1, cancel_reason = ?,
                generation = NULL, next_at = NULL, settled_at = ?, updated_at = ?
            WHERE run_id = ?
              AND state IN ('pending', 'waiting', 'running')
              AND generation IS NULL
            """,
            reason,
            settled_at,
            settled_at,
            run_id,
        )
        return _changes(self._sql) > 0

    def delete_unretained_terminal(self, run_id: str) -> bool:
        def delete_rows() -> None:
            rows = self._sql.execute(
                """
                SELECT 1 FROM cf_agents_task_runs
                WHERE run_id = ? AND retain = 0
                  AND state IN ('completed', 'failed', 'cancelled')
                """,
                run_id,
            )
            if not rows:
                raise _StoreFenceRejected
            self._sql.execute(
                "DELETE FROM cf_agents_task_steps WHERE run_id = ?",
                run_id,
            )
            self._sql.execute(
                "DELETE FROM cf_agents_task_runs WHERE run_id = ? AND retain = 0",
                run_id,
            )

        try:
            self._transaction_sync(delete_rows)
        except _StoreFenceRejected:
            return False
        return True

    def delete_unaccepted(self, run_id: str) -> bool:
        self._sql.execute(
            """
            DELETE FROM cf_agents_task_runs
            WHERE run_id = ? AND state = 'pending' AND attempt = 0
              AND generation IS NULL
            """,
            run_id,
        )
        return _changes(self._sql) > 0

    def delete_terminal(
        self,
        *,
        states: Sequence[_TerminalTaskRunState],
        settled_before: int | None,
        limit: int,
    ) -> tuple[tuple[str, str], ...]:
        if not states:
            return ()
        query = (
            "SELECT run_id, definition FROM cf_agents_task_runs WHERE state IN ("
            + ", ".join("?" for _ in states)
            + ")"
        )
        params: list[object] = list(states)
        if settled_before is not None:
            query += " AND settled_at < ?"
            params.append(settled_before)
        query += " ORDER BY settled_at ASC, run_id ASC LIMIT ?"
        params.append(limit)
        selected = tuple(
            (cast(str, row["run_id"]), cast(str, row["definition"]))
            for row in self._sql.execute(query, *params)
        )
        if not selected:
            return ()

        def delete_rows() -> None:
            for run_id, _ in selected:
                self._sql.execute(
                    "DELETE FROM cf_agents_task_steps WHERE run_id = ?",
                    run_id,
                )
                self._sql.execute(
                    "DELETE FROM cf_agents_task_runs WHERE run_id = ?",
                    run_id,
                )

        self._transaction_sync(delete_rows)
        return selected


def _changes(sql: LifecycleSql) -> int:
    rows = sql.execute("SELECT changes() AS count")
    return cast(int, rows[0]["count"])


def _row(value: dict[str, Any]) -> _TaskRunRow:
    return _TaskRunRow(
        run_id=cast(str, value["run_id"]),
        definition=cast(str, value["definition"]),
        input=cast(str | None, value["input"]),
        state=cast(_TaskRunState, value["state"]),
        result=cast(str | None, value["result"]),
        error_name=cast(str | None, value["error_name"]),
        error_message=cast(str | None, value["error_message"]),
        status_message=cast(str | None, value["status_message"]),
        metadata=cast(str | None, value["metadata"]),
        idempotency_key=cast(str | None, value["idempotency_key"]),
        retain=cast(int, value["retain"]),
        attempt=cast(int, value["attempt"]),
        generation=cast(str | None, value["generation"]),
        next_at=cast(int | None, value["next_at"]),
        wait_reason=cast(_TaskWaitReason | None, value["wait_reason"]),
        cancel_requested=cast(int, value["cancel_requested"]),
        cancel_reason=cast(str | None, value["cancel_reason"]),
        created_at=cast(int, value["created_at"]),
        started_at=cast(int | None, value["started_at"]),
        updated_at=cast(int, value["updated_at"]),
        settled_at=cast(int | None, value["settled_at"]),
    )


def _step_row(value: dict[str, Any]) -> _TaskStepRow:
    return _TaskStepRow(
        run_id=cast(str, value["run_id"]),
        step_name=cast(str, value["step_name"]),
        kind=cast(Literal["do", "sleep"], value["kind"]),
        state=cast(_TaskStepState, value["state"]),
        result=cast(str | None, value["result"]),
        error_name=cast(str | None, value["error_name"]),
        error_message=cast(str | None, value["error_message"]),
        attempt=cast(int, value["attempt"]),
        next_at=cast(int | None, value["next_at"]),
        created_at=cast(int, value["created_at"]),
        started_at=cast(int | None, value["started_at"]),
        updated_at=cast(int, value["updated_at"]),
        completed_at=cast(int | None, value["completed_at"]),
    )
