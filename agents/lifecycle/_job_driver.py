from __future__ import annotations

import asyncio
import inspect
import random
import re
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, cast

from .jobs import (
    LifecycleJob,
    LifecycleJobContext,
    LifecycleJobOutcome,
    LifecycleMemoryLimitContext,
    _LifecycleJobQueue,
    _validate_retry,
)


_DEADMAN_DELAY_MS = 30_000
_DEFAULT_RETRY = {
    "maxAttempts": 3,
    "baseDelayMs": 100,
    "maxDelayMs": 3_000,
}
_OOM_ALARM_STRIKES_KEY = "cf_agents:oom_alarm_strikes"
_CODE_UPDATE = re.compile(
    r"reset because its code was updated|this script has been upgraded",
    re.IGNORECASE,
)
_CONNECTION_LOST = re.compile(r"network connection lost", re.IGNORECASE)
_STORAGE_RESET = re.compile(
    r"Internal error in Durable Object storage caused object to be reset",
    re.IGNORECASE,
)
_MEMORY_LIMIT = re.compile(r"exceeded its memory limit", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class _JobDispatch:
    on_job: Callable[[LifecycleJobContext], Awaitable[LifecycleJobOutcome]]
    on_job_error: (
        Callable[
            [LifecycleJobContext, BaseException],
            Awaitable[LifecycleJobOutcome],
        ]
        | None
    ) = None


@dataclass(slots=True)
class _AlarmScope:
    memory_generation: int
    executing_row: dict[str, Any] | None = None
    executing_version: int = 0
    tracked: set[int] = field(default_factory=set)
    finished: bool = False
    tracked_failed: bool = False
    memory_struck: bool = False


@dataclass(frozen=True, slots=True)
class _TrackedWork:
    scope: _AlarmScope
    row: dict[str, Any]
    version: int
    awaitable: Awaitable[object]


_CURRENT_ALARM_SCOPE: ContextVar[tuple[object, _AlarmScope] | None] = ContextVar(
    "lifecycle_alarm_scope",
    default=None,
)


class _LifecycleJobDriver:
    def __init__(
        self,
        *,
        queue: _LifecycleJobQueue,
        storage: Callable[[], object],
        clock: Callable[[], int],
        disabled: Callable[[], bool],
        resolve_dispatch: Callable[[str], _JobDispatch | None],
        run_host_alarm: Callable[[], Awaitable[None]],
        arm_at: Callable[[int], Awaitable[None]],
        rearm: Callable[[], Awaitable[None]],
        report_error: Callable[[BaseException], Awaitable[None]],
        memory_limit: Callable[[LifecycleMemoryLimitContext], Awaitable[None]],
        max_memory_limit_strikes: Callable[[], int],
        reset: Callable[[str], Awaitable[None]],
        schedule_background: Callable[[Awaitable[object]], bool],
    ) -> None:
        self._queue = queue
        self._storage = storage
        self._clock = clock
        self._disabled = disabled
        self._resolve_dispatch = resolve_dispatch
        self._run_host_alarm = run_host_alarm
        self._arm_at = arm_at
        self._rearm = rearm
        self._report_error = report_error
        self._memory_limit = memory_limit
        self._max_memory_limit_strikes = max_memory_limit_strikes
        self._reset = reset
        self._schedule_background = schedule_background
        self._tracked: dict[int, _TrackedWork] = {}
        self._seen: dict[int, Awaitable[object]] = {}
        self._versions: dict[str, int] = {}
        self._mutation_sequence = 0
        self._active_claims: dict[str, int] = {}
        self._active_alarms = 0
        self._strike_lock = asyncio.Lock()
        self._last_claim_started_at = 0
        self._memory_generation = 0

    async def run_alarm(self, initialize: Callable[[], Awaitable[None]]) -> None:
        scope = _AlarmScope(self._memory_generation)
        token = _CURRENT_ALARM_SCOPE.set((self, scope))
        self._active_alarms += 1
        clean = False
        try:
            await initialize()
            await self._drive_due_jobs(scope)
            await self._run_host_alarm()
            scope.finished = True
            clean = True
        except asyncio.CancelledError:
            await self._rearm_after_failure()
            raise
        except BaseException as error:
            if _is_memory_limit_reset(error):
                await self._handle_memory_limit(
                    scope.executing_row,
                    expected_version=scope.executing_version,
                    incident=scope,
                )
                scope.finished = True
                return
            await self._rearm_after_failure()
            raise
        finally:
            self._active_alarms -= 1
            _CURRENT_ALARM_SCOPE.reset(token)
            if scope.executing_row is not None:
                self._prune_version(scope.executing_row["id"])
            self._prune_seen_if_quiescent()

        if clean and not scope.tracked_failed:
            try:
                await self._clear_strikes_if_quiescent(scope.memory_generation)
            except asyncio.CancelledError:
                await self._rearm_after_failure()
                raise
        await self._rearm()

    def note_mutation(self, job_id: str) -> None:
        if job_id not in self._active_claims and not self._has_tracked_job(job_id):
            return
        self._mutation_sequence += 1
        self._versions[job_id] = self._mutation_sequence

    def track(self, awaitable: Awaitable[object]) -> bool:
        current = _CURRENT_ALARM_SCOPE.get()
        if current is None or current[0] is not self:
            return False
        scope = current[1]
        row = scope.executing_row
        if row is None:
            return False

        key = id(awaitable)
        if self._seen.get(key) is awaitable:
            return True
        existing = self._tracked.get(key)
        if existing is not None and existing.awaitable is awaitable:
            self._seen[key] = awaitable
            return True
        work = _TrackedWork(
            scope,
            dict(row),
            scope.executing_version,
            awaitable,
        )
        scope.tracked.add(key)
        self._tracked[key] = work
        observer = self._observe_tracked(key, work, awaitable)
        try:
            accepted = self._schedule_background(observer)
        except BaseException:
            self._tracked.pop(key, None)
            scope.tracked.discard(key)
            observer.close()
            raise
        if not accepted:
            self._tracked.pop(key, None)
            scope.tracked.discard(key)
            observer.close()
            return False
        self._seen[key] = awaitable
        return True

    def is_tracked(self, awaitable: Awaitable[object]) -> bool:
        tracked = self._tracked.get(id(awaitable))
        return tracked is not None and tracked.awaitable is awaitable

    async def _drive_due_jobs(self, scope: _AlarmScope) -> None:
        invocation_time = self._clock()
        exclusive_only = self._queue.has_exclusive()
        due = self._queue.due(invocation_time, exclusive_only=exclusive_only)
        if not due:
            return
        if not self._disabled():
            await self._arm_at(invocation_time + _DEADMAN_DELAY_MS)

        for stale in due:
            if self._disabled():
                return
            row = self._queue.due_row(
                stale["id"],
                invocation_time,
                exclusive_only=self._queue.has_exclusive(),
            )
            if row is None:
                continue
            if (
                row["singleflight"] == 1
                and row["running"] == 1
                and not self._queue.is_hung(row, invocation_time)
            ):
                continue
            claim_started_at = max(invocation_time, self._last_claim_started_at + 1)
            self._last_claim_started_at = claim_started_at
            self._queue.mark_running(row["id"], claim_started_at)
            claimed = dict(row)
            claimed["running"] = 1
            claimed["execution_started_at"] = claim_started_at
            scope.executing_row = claimed
            scope.executing_version = self._begin_claim(claimed["id"])
            try:
                await self._drive_job(claimed, scope.executing_version)
            except BaseException:
                self._end_claim(claimed["id"], preserve_version=True)
                raise
            else:
                self._end_claim(claimed["id"])
            scope.executing_row = None

    async def _drive_job(self, row: dict[str, Any], claim_version: int) -> None:
        try:
            job = self._queue.job_from_row(row, validate_retry=False)
        except Exception as error:
            try:
                self._queue.delete(row["id"])
            except BaseException as cleanup_error:
                await self._report_error(cleanup_error)
            await self._report_error(error)
            return

        dispatch = self._resolve_dispatch(row["capability"])
        if dispatch is None:
            self._queue.delete(row["id"])
            await self._report_error(
                LookupError(
                    f"no installed capability or host handler for job {row['id']}"
                )
            )
            return

        retry = {**_DEFAULT_RETRY, **(job.retry or {})}
        max_attempts = cast(int, retry["maxAttempts"])
        outcome: LifecycleJobOutcome = None
        try:
            _validate_retry(retry)
            for attempt in range(1, max_attempts + 1):
                try:
                    outcome = await dispatch.on_job(
                        LifecycleJobContext(job, attempt=attempt)
                    )
                    break
                except asyncio.CancelledError:
                    raise
                except BaseException as error:
                    if _is_code_update_reset(error):
                        raise
                    if attempt == max_attempts:
                        raise
                    await _retry_sleep(
                        attempt,
                        cast(int | float, retry["baseDelayMs"]),
                        cast(int | float, retry["maxDelayMs"]),
                    )
        except asyncio.CancelledError:
            await self._clear_running_best_effort(row, claim_version)
            raise
        except BaseException as error:
            if _is_platform_failure(error):
                if not _is_memory_limit_reset(error):
                    await self._clear_running_best_effort(row, claim_version)
                raise
            if dispatch.on_job_error is not None:
                try:
                    outcome = await dispatch.on_job_error(
                        LifecycleJobContext(job, attempt=max_attempts),
                        error,
                    )
                except asyncio.CancelledError:
                    await self._clear_running_best_effort(row, claim_version)
                    raise
                except BaseException as hook_error:
                    if _is_platform_failure(hook_error):
                        if not _is_memory_limit_reset(hook_error):
                            await self._clear_running_best_effort(row, claim_version)
                        raise
                    try:
                        await self._report_error(hook_error)
                    except asyncio.CancelledError:
                        await self._clear_running_best_effort(row, claim_version)
                        raise
                    outcome = None

        if not self._disabled() and self._version_matches(row["id"], claim_version):
            self._queue.apply_outcome(row, outcome)

    async def _clear_running_best_effort(
        self,
        row: dict[str, Any],
        claim_version: int,
    ) -> None:
        try:
            if self._version_matches(row["id"], claim_version):
                self._queue.clear_intent(row)
        except BaseException as error:
            await self._report_error(error)

    def _begin_claim(self, job_id: str) -> int:
        self._mutation_sequence += 1
        self._versions[job_id] = self._mutation_sequence
        self._active_claims[job_id] = self._active_claims.get(job_id, 0) + 1
        return self._mutation_sequence

    def _end_claim(self, job_id: str, *, preserve_version: bool = False) -> None:
        remaining = self._active_claims[job_id] - 1
        if remaining:
            self._active_claims[job_id] = remaining
        else:
            self._active_claims.pop(job_id, None)
            if not preserve_version:
                self._prune_version(job_id)

    def _version_matches(self, job_id: str, expected: int) -> bool:
        return self._versions.get(job_id, 0) == expected

    def _has_tracked_job(self, job_id: str) -> bool:
        return any(work.row["id"] == job_id for work in self._tracked.values())

    def _prune_version(self, job_id: str) -> None:
        if job_id in self._active_claims or self._has_tracked_job(job_id):
            return
        self._versions.pop(job_id, None)

    async def _rearm_after_failure(self) -> None:
        try:
            await self._rearm()
        except asyncio.CancelledError:
            raise
        except BaseException as rearm_error:
            await self._report_error(rearm_error)

    async def _observe_tracked(
        self,
        key: int,
        work: _TrackedWork,
        awaitable: Awaitable[object],
    ) -> None:
        error: BaseException | None = None
        try:
            await awaitable
        except BaseException as tracked_error:
            error = tracked_error
        await self._settle_tracked(key, work, error)

    async def _settle_tracked(
        self,
        key: int,
        work: _TrackedWork,
        error: BaseException | None,
    ) -> None:
        if error is not None:
            work.scope.tracked_failed = True
        if error is not None and _is_memory_limit_reset(error):
            try:
                await self._handle_memory_limit(
                    work.row,
                    expected_version=work.version,
                    require_running=False,
                    incident=work.scope,
                )
            finally:
                self._tracked.pop(key, None)
                work.scope.tracked.discard(key)
                self._prune_version(work.row["id"])
                self._prune_seen_if_quiescent()
            return
        if error is not None and not isinstance(error, asyncio.CancelledError):
            try:
                await self._report_error(error)
            finally:
                self._tracked.pop(key, None)
                work.scope.tracked.discard(key)
                self._prune_version(work.row["id"])
                self._prune_seen_if_quiescent()
            return

        self._tracked.pop(key, None)
        work.scope.tracked.discard(key)
        if (
            work.scope.finished
            and not work.scope.tracked
            and not work.scope.tracked_failed
        ):
            await self._clear_strikes_if_quiescent(work.scope.memory_generation)
        self._prune_version(work.row["id"])
        self._prune_seen_if_quiescent()

    def _prune_seen_if_quiescent(self) -> None:
        if not self._active_alarms and not self._tracked:
            self._seen.clear()

    async def _handle_memory_limit(
        self,
        executing_row: dict[str, Any] | None,
        *,
        expected_version: int | None = None,
        require_running: bool = True,
        incident: _AlarmScope | None = None,
    ) -> None:
        acquire = asyncio.create_task(self._strike_lock.acquire())
        cancelled: asyncio.CancelledError | None = None
        while not acquire.done():
            try:
                await asyncio.shield(acquire)
            except asyncio.CancelledError as cancellation:
                cancelled = cancellation
        try:
            acquire.result()
            await self._handle_memory_limit_locked(
                executing_row,
                expected_version=expected_version,
                require_running=require_running,
                incident=incident,
                cancelled=cancelled,
            )
        finally:
            self._strike_lock.release()

    async def _handle_memory_limit_locked(
        self,
        executing_row: dict[str, Any] | None,
        *,
        expected_version: int | None,
        require_running: bool,
        incident: _AlarmScope | None,
        cancelled: asyncio.CancelledError | None,
    ) -> None:
        if incident is not None and incident.memory_struck:
            if cancelled is not None:
                raise cancelled
            return
        if incident is not None:
            incident.memory_struck = True
        self._memory_generation += 1
        await self._apply_memory_limit_policy(
            executing_row,
            expected_version=expected_version,
            require_running=require_running,
            cancelled=cancelled,
        )

    async def _apply_memory_limit_policy(
        self,
        executing_row: dict[str, Any] | None,
        *,
        expected_version: int | None,
        require_running: bool,
        cancelled: asyncio.CancelledError | None,
    ) -> None:
        strikes = 1
        storage: object | None = None
        try:
            storage = self._storage()
            strike_write = asyncio.create_task(_record_memory_limit_strike(storage))
            while not strike_write.done():
                try:
                    strikes = await asyncio.shield(strike_write)
                except asyncio.CancelledError as error:
                    cancelled = error
            if not strike_write.cancelled():
                strikes = strike_write.result()
        except BaseException:
            pass

        limit = self._max_memory_limit_strikes()
        sealed = strikes >= limit
        next_time = None if sealed else self._clock() + min(300, 30 * strikes) * 1_000
        purged: tuple[LifecycleJob, ...] = ()
        if sealed:
            try:
                purged = self._queue.recovery_loop_jobs()
            except asyncio.CancelledError as error:
                cancelled = error
            except BaseException:
                pass

        try:
            current_intent = executing_row is not None and (
                expected_version is None
                or self._versions.get(executing_row["id"], 0) == expected_version
            )
            if current_intent and executing_row is not None:
                if sealed:
                    self._queue.delete_intent(
                        executing_row,
                        require_running=require_running,
                    )
                elif next_time is not None:
                    self._queue.retime_intent(
                        executing_row,
                        next_time,
                        require_running=require_running,
                    )
            if sealed:
                self._queue.purge_recovery_loop_jobs()
            elif next_time is not None:
                self._queue.delay_recovery_loop_jobs(next_time)
        except asyncio.CancelledError as error:
            cancelled = error
        except BaseException:
            pass

        context = LifecycleMemoryLimitContext(
            sealed=sealed,
            next_time=next_time,
            executing=(
                self._queue.job_from_row(executing_row)
                if executing_row is not None
                else None
            ),
            purged_recovery_loop_jobs=purged,
        )
        try:
            await self._memory_limit(context)
        except asyncio.CancelledError as error:
            cancelled = error
        except BaseException:
            pass

        if sealed:
            try:
                await _call_storage(
                    storage,
                    "delete",
                    _OOM_ALARM_STRIKES_KEY,
                )
            except asyncio.CancelledError as error:
                cancelled = error
            except BaseException:
                pass
        try:
            await self._rearm()
        except asyncio.CancelledError as error:
            cancelled = error
        except BaseException:
            pass
        try:
            sync = getattr(storage, "sync", None) if storage is not None else None
            if callable(sync):
                await _maybe_await(sync())
        except asyncio.CancelledError as error:
            cancelled = error
        except BaseException:
            pass
        try:
            await self._reset(
                f"alarm memory-limit strike {strikes}/{limit}"
                f"{' (sealed)' if sealed else ''}"
            )
        except asyncio.CancelledError as error:
            cancelled = error
        except BaseException:
            pass
        if cancelled is not None:
            raise cancelled

    async def _clear_strikes_if_quiescent(self, expected_generation: int) -> None:
        async with self._strike_lock:
            if (
                self._active_alarms
                or self._tracked
                or self._memory_generation != expected_generation
            ):
                return
            try:
                storage = self._storage()
                prior = await _call_storage(storage, "get", _OOM_ALARM_STRIKES_KEY)
                if type(prior) is int and prior > 0:
                    await _call_storage(
                        storage,
                        "delete",
                        _OOM_ALARM_STRIKES_KEY,
                    )
            except asyncio.CancelledError:
                raise
            except BaseException:
                pass


async def _retry_sleep(attempt: int, base_delay_ms: float, max_delay_ms: float) -> None:
    upper_bound = min((2**attempt) * base_delay_ms, max_delay_ms)
    await asyncio.sleep(random.random() * upper_bound / 1_000)


async def _record_memory_limit_strike(storage: object) -> int:
    prior = await _call_storage(storage, "get", _OOM_ALARM_STRIKES_KEY)
    strikes = (prior if type(prior) is int else 0) + 1
    await _call_storage(storage, "put", _OOM_ALARM_STRIKES_KEY, strikes)
    return strikes


def _error_chain(error: BaseException):
    current: object | None = error
    seen: set[int] = set()
    for _ in range(8):
        if current is None or id(current) in seen:
            return
        seen.add(id(current))
        yield current
        current = getattr(current, "__cause__", None) or getattr(
            current,
            "cause",
            None,
        )


def _message(error: object) -> str:
    return str(error) if isinstance(error, (BaseException, str)) else ""


def _is_code_update_reset(error: BaseException) -> bool:
    return any(_CODE_UPDATE.search(_message(item)) for item in _error_chain(error))


def _is_memory_limit_reset(error: BaseException) -> bool:
    return any(_MEMORY_LIMIT.search(_message(item)) for item in _error_chain(error))


def _is_platform_failure(error: BaseException) -> bool:
    if _is_code_update_reset(error) or _is_memory_limit_reset(error):
        return True
    for item in _error_chain(error):
        message = _message(item)
        if _CONNECTION_LOST.search(message) or _STORAGE_RESET.search(message):
            return True
        if (
            bool(getattr(item, "retryable", False))
            and not bool(getattr(item, "overloaded", False))
            and "Durable Object is overloaded" not in message
        ):
            return True
    return False


async def _call_storage(storage: object, name: str, *args: object) -> object:
    return await _maybe_await(getattr(storage, name)(*args))


async def _maybe_await(value: object) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value
