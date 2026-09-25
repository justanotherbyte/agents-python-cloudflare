from __future__ import annotations

import builtins
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from ..core._wire import strict_json_loads
from ..core.utils import MISSING, dumps_wire, gen_id

type _SqlExecutor = Callable[..., list[dict[str, Any]]]
type LifecycleJobOutcome = LifecycleJobReschedule | Literal["yield"] | None

_DEFAULT_RETRY = {
    "maxAttempts": 3,
    "baseDelayMs": 100,
    "maxDelayMs": 3_000,
}
_DEFAULT_HUNG_TIMEOUT_SECONDS = 30
_MAX_SQLITE_INTEGER = 2**63 - 1
_MAX_HUNG_TIMEOUT_SECONDS = _MAX_SQLITE_INTEGER // 1_000


@dataclass(frozen=True, slots=True)
class LifecycleJobPushOptions:
    fn: str
    time: int
    payload: object = MISSING
    id: str | None = None
    retry: dict[str, object] | None = None
    singleflight: bool = False
    hung_timeout_seconds: int | None = None
    exclusive: bool = False
    recovery_loop: bool = False


@dataclass(frozen=True, slots=True)
class LifecycleJob:
    id: str
    capability: str
    fn: str
    time: int
    payload: object | None
    payload_present: bool
    retry: dict[str, object] | None
    singleflight: bool
    exclusive: bool
    recovery_loop: bool
    created_at: int


@dataclass(frozen=True, slots=True)
class LifecycleJobContext:
    job: LifecycleJob
    attempt: int


@dataclass(frozen=True, slots=True)
class LifecycleMemoryLimitContext:
    sealed: bool
    next_time: int | None
    executing: LifecycleJob | None
    purged_recovery_loop_jobs: tuple[LifecycleJob, ...]


@dataclass(frozen=True, slots=True)
class LifecycleJobReschedule:
    reschedule_at: int


class LifecycleJobs(Protocol):
    async def push(self, options: LifecycleJobPushOptions) -> LifecycleJob: ...

    async def _push_unvalidated_retry(
        self,
        options: LifecycleJobPushOptions,
    ) -> LifecycleJob: ...

    async def cancel(self, id: str) -> bool: ...

    async def reschedule(self, id: str, time: int) -> bool: ...

    async def get(self, id: str) -> LifecycleJob | None: ...

    async def _get_unvalidated_retry(self, id: str) -> LifecycleJob | None: ...

    async def list(
        self,
        *,
        skip_invalid: bool = False,
    ) -> tuple[LifecycleJob, ...]: ...

    async def _list_unvalidated_retry(self) -> tuple[LifecycleJob, ...]: ...

    async def rearm(self) -> None: ...


class _LifecycleJobQueue:
    __slots__ = ("_sql",)

    def __init__(self, sql: _SqlExecutor):
        self._sql = sql

    def prepare(self) -> None:
        self._sql("""
        CREATE TABLE IF NOT EXISTS cf_agents_jobs (
          id TEXT PRIMARY KEY NOT NULL,
          capability TEXT NOT NULL,
          fn TEXT NOT NULL,
          time INTEGER NOT NULL,
          payload TEXT,
          retry_options TEXT,
          singleflight INTEGER NOT NULL DEFAULT 0,
          hung_timeout_seconds INTEGER,
          exclusive INTEGER NOT NULL DEFAULT 0,
          recovery_loop INTEGER NOT NULL DEFAULT 0,
          running INTEGER NOT NULL DEFAULT 0,
          execution_started_at INTEGER,
          created_at INTEGER NOT NULL DEFAULT (unixepoch())
        ) WITHOUT ROWID
        """)

    def push(
        self,
        capability: str,
        options: LifecycleJobPushOptions,
        *,
        validate_retry: bool = True,
    ) -> LifecycleJob:
        due = _validate_time(options.time)
        if type(options.fn) is not str or not options.fn.strip():
            raise ValueError("jobs require a non-empty fn")
        job_id = options.id if options.id is not None else gen_id()
        if type(job_id) is not str or not job_id:
            raise ValueError("job id must be a non-empty string")
        payload = None if options.payload is MISSING else dumps_wire(options.payload)
        retry = _encode_retry(options.retry, validate=validate_retry)
        _validate_job_flags(options)
        self._sql(
            """
            INSERT INTO cf_agents_jobs
              (id, capability, fn, time, payload, retry_options, singleflight,
               hung_timeout_seconds, exclusive, recovery_loop, running,
               execution_started_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL)
            ON CONFLICT(id) DO UPDATE SET
              fn = excluded.fn,
              time = excluded.time,
              payload = excluded.payload,
              retry_options = excluded.retry_options,
              singleflight = excluded.singleflight,
              hung_timeout_seconds = excluded.hung_timeout_seconds,
              exclusive = excluded.exclusive,
              recovery_loop = excluded.recovery_loop,
              running = 0,
              execution_started_at = NULL
            WHERE cf_agents_jobs.capability = excluded.capability
            """,
            job_id,
            capability,
            options.fn,
            due,
            payload,
            retry,
            int(options.singleflight),
            options.hung_timeout_seconds,
            int(options.exclusive),
            int(options.recovery_loop),
        )
        job = self.get(capability, job_id, validate_retry=validate_retry)
        if job is not None:
            return job
        rows = self._sql(
            "SELECT capability FROM cf_agents_jobs WHERE id = ?",
            job_id,
        )
        if rows:
            owner = rows[0]["capability"]
            raise RuntimeError(
                f"job id {job_id!r} already belongs to {owner!r}; "
                "job ids are scoped to their owner"
            )
        raise RuntimeError(f"failed to persist job {job_id}")

    def cancel(self, capability: str, job_id: str) -> bool:
        if not self._is_owned(capability, job_id):
            return False
        self._sql(
            "DELETE FROM cf_agents_jobs WHERE id = ? AND capability = ?",
            job_id,
            capability,
        )
        return True

    def reschedule(self, capability: str, job_id: str, time: int) -> bool:
        due = _validate_time(time)
        if not self._is_owned(capability, job_id):
            return False
        self._sql(
            """
            UPDATE cf_agents_jobs
            SET time = ?, running = 0, execution_started_at = NULL
            WHERE id = ? AND capability = ?
            """,
            due,
            job_id,
            capability,
        )
        return True

    def _is_owned(self, capability: str, job_id: str) -> bool:
        return bool(
            self._sql(
                "SELECT id FROM cf_agents_jobs WHERE id = ? AND capability = ?",
                job_id,
                capability,
            )
        )

    def get(
        self,
        capability: str,
        job_id: str,
        *,
        validate_retry: bool = True,
    ) -> LifecycleJob | None:
        rows = self._sql(
            "SELECT * FROM cf_agents_jobs WHERE id = ? AND capability = ?",
            job_id,
            capability,
        )
        return (
            self.job_from_row(rows[0], validate_retry=validate_retry) if rows else None
        )

    def list(
        self,
        capability: str,
        *,
        skip_invalid: bool = False,
        validate_retry: bool = True,
    ) -> tuple[LifecycleJob, ...]:
        rows = self._sql(
            "SELECT * FROM cf_agents_jobs WHERE capability = ? ORDER BY time ASC",
            capability,
        )
        if not skip_invalid:
            return tuple(
                self.job_from_row(row, validate_retry=validate_retry) for row in rows
            )
        jobs = []
        for row in rows:
            try:
                jobs.append(self.job_from_row(row, validate_retry=validate_retry))
            except (TypeError, ValueError):
                continue
        return tuple(jobs)

    def has_exclusive(self) -> bool:
        return self._valid_exclusive_time() is not None

    def _valid_exclusive_time(self) -> int | None:
        rows = self._sql(
            "SELECT * FROM cf_agents_jobs WHERE exclusive = 1 ORDER BY time ASC"
        )
        for row in rows:
            try:
                self.job_from_row(row)
            except (TypeError, ValueError):
                continue
            return int(row["time"])
        return None

    def due(
        self,
        time: int,
        *,
        exclusive_only: bool = False,
    ) -> builtins.list[dict[str, Any]]:
        exclusive = " AND exclusive = 1" if exclusive_only else ""
        return self._sql(
            f"SELECT * FROM cf_agents_jobs WHERE time <= ?{exclusive} "
            "ORDER BY time ASC",
            time,
        )

    def due_row(
        self,
        job_id: str,
        time: int,
        *,
        exclusive_only: bool = False,
    ) -> dict[str, Any] | None:
        exclusive = " AND exclusive = 1" if exclusive_only else ""
        rows = self._sql(
            f"SELECT * FROM cf_agents_jobs WHERE id = ? AND time <= ?{exclusive}",
            job_id,
            time,
        )
        return rows[0] if rows else None

    def mark_running(self, job_id: str, time: int) -> None:
        self._sql(
            """
            UPDATE cf_agents_jobs
            SET running = 1, execution_started_at = ?
            WHERE id = ?
            """,
            time,
            job_id,
        )

    @staticmethod
    def is_hung(row: dict[str, Any], time: int) -> bool:
        started = row["execution_started_at"] or 0
        return time - started >= _hung_timeout_ms(row)

    def clear_running(self, job_id: str) -> None:
        self._sql(
            """
            UPDATE cf_agents_jobs
            SET running = 0, execution_started_at = NULL
            WHERE id = ? AND running = 1
            """,
            job_id,
        )

    def clear_intent(self, row: dict[str, Any]) -> None:
        where, values = _intent_guard(row, require_running=True)
        self._sql(
            f"""
            UPDATE cf_agents_jobs
            SET running = 0, execution_started_at = NULL
            WHERE {where}
            """,
            *values,
        )

    def delete(self, job_id: str) -> None:
        self._sql("DELETE FROM cf_agents_jobs WHERE id = ?", job_id)

    def delete_intent(
        self,
        row: dict[str, Any],
        *,
        require_running: bool,
    ) -> None:
        where, values = _intent_guard(row, require_running=require_running)
        self._sql(f"DELETE FROM cf_agents_jobs WHERE {where}", *values)

    def retime_intent(
        self,
        row: dict[str, Any],
        time: int,
        *,
        require_running: bool,
    ) -> None:
        where, values = _intent_guard(row, require_running=require_running)
        self._sql(
            f"""
            UPDATE cf_agents_jobs
            SET time = ?, running = 0, execution_started_at = NULL
            WHERE {where}
            """,
            _validate_time(time),
            *values,
        )

    def delay_recovery_loop_jobs(self, time: int) -> None:
        due = _validate_time(time)
        self._sql(
            """
            UPDATE cf_agents_jobs
            SET time = ?, running = 0, execution_started_at = NULL
            WHERE recovery_loop = 1 AND time <= ?
            """,
            due,
            due,
        )

    def recovery_loop_jobs(self) -> tuple[LifecycleJob, ...]:
        rows = self._sql(
            "SELECT * FROM cf_agents_jobs WHERE recovery_loop = 1 ORDER BY time ASC"
        )
        jobs: list[LifecycleJob] = []
        for row in rows:
            try:
                jobs.append(self.job_from_row(row))
            except (TypeError, ValueError):
                continue
        return tuple(jobs)

    def purge_recovery_loop_jobs(self) -> None:
        self._sql("DELETE FROM cf_agents_jobs WHERE recovery_loop = 1")

    def apply_outcome(
        self,
        row: dict[str, Any],
        outcome: LifecycleJobOutcome,
    ) -> None:
        where, values = _intent_guard(row, require_running=True)
        if outcome is None:
            self._sql(
                f"DELETE FROM cf_agents_jobs WHERE {where}",
                *values,
            )
            return
        if outcome == "yield":
            self.clear_intent(row)
            return
        if type(outcome) is LifecycleJobReschedule:
            self._sql(
                f"""
                UPDATE cf_agents_jobs
                SET time = ?, running = 0, execution_started_at = NULL
                WHERE {where}
                """,
                _validate_time(outcome.reschedule_at),
                *values,
            )
            return
        raise ValueError(f"invalid job outcome for {row['id']}")

    def next_alarm_time(self, time: int) -> int | None:
        # SqlStorage rejects the SQLite maximum when FFI converts its bind to BigInt.
        time_upper_bound = _MAX_SQLITE_INTEGER
        exclusive_time = self._valid_exclusive_time()
        if exclusive_time is not None:
            ready = self._sql(
                f"""
                SELECT MIN(time) AS time FROM cf_agents_jobs
                WHERE exclusive = 1
                  AND (singleflight = 0
                    OR running = 0
                    OR coalesce(execution_started_at, 0)
                       + coalesce(hung_timeout_seconds, ?) * 1000 <= ?)
                   AND typeof(time) IN ('integer', 'real')
                   AND time BETWEEN 0 AND {time_upper_bound}
                """,
                _DEFAULT_HUNG_TIMEOUT_SECONDS,
                time,
            )
            ready_time = ready[0]["time"] if ready else None
            candidate = int(ready_time) if ready_time is not None else None
            in_flight = self._sql(
                """
                SELECT MIN(
                    coalesce(execution_started_at, 0)
                    + coalesce(hung_timeout_seconds, ?) * 1000
                ) AS recheck
                FROM cf_agents_jobs
                WHERE exclusive = 1
                  AND singleflight = 1
                  AND running = 1
                  AND coalesce(execution_started_at, 0)
                      + coalesce(hung_timeout_seconds, ?) * 1000 > ?
                """,
                _DEFAULT_HUNG_TIMEOUT_SECONDS,
                _DEFAULT_HUNG_TIMEOUT_SECONDS,
                time,
            )
            recheck = in_flight[0]["recheck"] if in_flight else None
            if recheck is not None:
                return (
                    int(recheck) if candidate is None else min(candidate, int(recheck))
                )
            return candidate

        ready = self._sql(
            f"""
            SELECT MIN(time) AS time FROM cf_agents_jobs
            WHERE (singleflight = 0
               OR running = 0
               OR coalesce(execution_started_at, 0)
                  + coalesce(hung_timeout_seconds, ?) * 1000 <= ?)
              AND typeof(time) IN ('integer', 'real')
              AND time BETWEEN 0 AND {time_upper_bound}
            """,
            _DEFAULT_HUNG_TIMEOUT_SECONDS,
            time,
        )
        ready_time = ready[0]["time"] if ready else None
        candidate = max(int(ready_time), time + 1) if ready_time is not None else None

        in_flight = self._sql(
            """
            SELECT MIN(
                coalesce(execution_started_at, 0)
                + coalesce(hung_timeout_seconds, ?) * 1000
            ) AS recheck
            FROM cf_agents_jobs
            WHERE singleflight = 1
              AND running = 1
              AND coalesce(execution_started_at, 0)
                  + coalesce(hung_timeout_seconds, ?) * 1000 > ?
            """,
            _DEFAULT_HUNG_TIMEOUT_SECONDS,
            _DEFAULT_HUNG_TIMEOUT_SECONDS,
            time,
        )
        recheck = in_flight[0]["recheck"] if in_flight else None
        if recheck is not None:
            candidate = (
                int(recheck) if candidate is None else min(candidate, int(recheck))
            )
        return candidate

    @staticmethod
    def job_from_row(
        row: dict[str, Any],
        *,
        validate_retry: bool = True,
    ) -> LifecycleJob:
        return LifecycleJob(
            id=row["id"],
            capability=row["capability"],
            fn=row["fn"],
            time=_validate_time(row["time"]),
            payload=(
                strict_json_loads(row["payload"], "Lifecycle job payload")
                if row["payload"] is not None
                else None
            ),
            payload_present=row["payload"] is not None,
            retry=_decode_retry(row["retry_options"], validate=validate_retry),
            singleflight=row["singleflight"] == 1,
            exclusive=row["exclusive"] == 1,
            recovery_loop=row["recovery_loop"] == 1,
            created_at=row["created_at"],
        )


def _validate_time(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"invalid job time: {value!r}")
    if not _number_is_finite(value) or value < 0 or value > _MAX_SQLITE_INTEGER:
        raise ValueError(f"invalid job time: {value!r}")
    return math.floor(value)


def _decode_retry(
    raw: object,
    *,
    validate: bool = True,
) -> dict[str, object] | None:
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise ValueError("job retry options must be JSON text")
    parsed = strict_json_loads(raw, "Lifecycle job retry")
    if not isinstance(parsed, dict):
        raise ValueError("job retry options must be a JSON object")
    if validate:
        _validate_retry(parsed)
    return parsed


def _encode_retry(
    options: dict[str, object] | None,
    *,
    validate: bool = True,
) -> str | None:
    if options is None:
        return None
    if not isinstance(options, dict):
        raise ValueError("job retry options must be a mapping")
    if validate:
        _validate_retry(options)
    return dumps_wire(options)


def _validate_retry(options: dict[str, object]) -> None:
    unknown = set(options) - set(_DEFAULT_RETRY)
    if unknown:
        raise ValueError(f"unknown retry option: {sorted(unknown)[0]}")

    max_attempts = options.get("maxAttempts", _DEFAULT_RETRY["maxAttempts"])
    if (
        type(max_attempts) is not int
        or not _number_is_finite(max_attempts)
        or max_attempts < 1
    ):
        raise ValueError("retry.maxAttempts must be an integer >= 1")

    base_delay = _positive_retry_number(
        options.get("baseDelayMs", _DEFAULT_RETRY["baseDelayMs"]),
        "baseDelayMs",
    )
    max_delay = _positive_retry_number(
        options.get("maxDelayMs", _DEFAULT_RETRY["maxDelayMs"]),
        "maxDelayMs",
    )
    if base_delay > max_delay:
        raise ValueError("retry.baseDelayMs must be <= retry.maxDelayMs")


def _positive_retry_number(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not _number_is_finite(value)
        or value <= 0
    ):
        raise ValueError(f"retry.{name} must be > 0")
    return value


def _number_is_finite(value: int | float) -> bool:
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _validate_job_flags(options: LifecycleJobPushOptions) -> None:
    for name in ("singleflight", "exclusive", "recovery_loop"):
        if type(getattr(options, name)) is not bool:
            raise ValueError(f"{name} must be a boolean")
    timeout = options.hung_timeout_seconds
    if timeout is not None and (
        type(timeout) is not int or timeout < 1 or timeout > _MAX_HUNG_TIMEOUT_SECONDS
    ):
        raise ValueError("hung_timeout_seconds must be an integer >= 1")


def _hung_timeout_ms(row: dict[str, Any]) -> int:
    seconds = row["hung_timeout_seconds"]
    timeout = seconds if seconds is not None else _DEFAULT_HUNG_TIMEOUT_SECONDS
    return int(timeout * 1_000)


def _intent_guard(
    row: dict[str, Any],
    *,
    require_running: bool,
) -> tuple[str, tuple[object, ...]]:
    columns = (
        "id",
        "capability",
        "fn",
        "time",
        "payload",
        "retry_options",
        "singleflight",
        "hung_timeout_seconds",
        "exclusive",
        "recovery_loop",
    )
    if require_running:
        columns += ("execution_started_at",)
    clauses = [f"{column} IS ?" for column in columns]
    if require_running:
        clauses.append("running = 1")
    return " AND ".join(clauses), tuple(row[column] for column in columns)
