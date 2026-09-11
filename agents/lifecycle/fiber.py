from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Collection
from contextlib import suppress
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Literal

from ..core.schema import CORE_SCHEMA_VERSION, read_core_schema_version
from ..core.utils import (
    MISSING,
    clamp,
    dumps_wire,
    error_message,
    gen_id,
    loads_dict_or_none,
    loads_or_none,
    now_ms,
)
from ._job_driver import _OOM_ALARM_STRIKES_KEY, _is_platform_failure
from .capability import LifecycleCapability
from .fiber_schema import prepare_fiber_schema
from .jobs import (
    LifecycleJobContext,
    LifecycleJobPushOptions,
    LifecycleJobReschedule,
    LifecycleMemoryLimitContext,
)

# Terminal statuses. `interrupted` counts as terminal: resolve_fiber may move it
# on later, but until then the scan and the waiters treat it as done.
_TERMINAL_STATUSES = frozenset({"completed", "aborted", "interrupted", "error"})

# Framework fibers this port does not drive. Recovery leaves them for a runtime
# that owns them rather than handing them to the user hook.
INTERNAL_FIBER_PREFIX = "__cf_internal_"

# Kept in one place so the ledger reads cannot drift apart.
_FIBER_COLUMNS = (
    "fiber_id, idempotency_key, name, status, snapshot, metadata_json, "
    "error_message, created_at, started_at, completed_at"
)
_FIBER_SELECT = "SELECT " + _FIBER_COLUMNS + " FROM cf_agents_fibers"

# Managed fibers whose run row never reached storage. Both queries bind their status
# values so the recovery scan and pending check cannot drift into different rules.
_LEDGER_ONLY_ROWS_QUERY = (
    "SELECT f.fiber_id, f.idempotency_key, f.name, f.status, f.snapshot, "
    "f.metadata_json, f.error_message, f.created_at, f.started_at, "
    "f.completed_at FROM cf_agents_fibers f LEFT JOIN cf_agents_runs r "
    "ON r.id = f.fiber_id WHERE f.status IN (?, ?) AND r.id IS NULL"
)
_LEDGER_ONLY_COUNT_QUERY = (
    "SELECT COUNT(*) AS count FROM cf_agents_fibers f "
    "LEFT JOIN cf_agents_runs r ON r.id = f.fiber_id "
    "WHERE f.status IN (?, ?) AND r.id IS NULL"
)

_MAINTENANCE_JOB_ID = "__cf_internal_fibers_maintenance"
_MAINTENANCE_JOB_FN = "__cf_internal_fibers_maintenance"
_MAINTENANCE_DISABLED_ERROR = "fiber maintenance is unavailable for this Agent"
_MEMORY_LIMIT_ERROR = "fiber recovery disabled after repeated memory-limit resets"
_DISPOSED_ERROR = "fiber capability disposed"


def _placeholders(values: Collection[Any]) -> str:
    return ", ".join("?" for _ in values)


def _stringify_snapshot(value: Any) -> str | None:
    # The one place missing-vs-null changes persisted state: absent binds NULL so
    # COALESCE keeps the prior snapshot, explicit None writes "null" and overwrites.
    if value is MISSING:
        return None
    return dumps_wire(value)


# Dataclasses, not TypedDicts: user code reads these by attribute (ctx.snapshot,
# result.accepted), FiberContext carries a bound method over a captured closure, and
# FiberRecoveryContext is mutated in place while a recovery runs.
@dataclass
class FiberContext:
    """Handed to a runFiber callback: identity, cancellation, checkpointing."""

    id: str
    signal: FiberSignal
    _stash: Callable[[Any], None]
    # A recovered snapshot arrives through FiberRecoveryContext, never here.
    snapshot: Any = None

    def stash(self, data: Any) -> None:
        self._stash(data)


@dataclass
class FiberRecoveryContext:
    """Handed to on_fiber_recovered when an interrupted fiber is detected."""

    id: str
    name: str
    snapshot: Any
    created_at: int
    recovery_reason: str = "interrupted"
    status: str | None = None
    idempotency_key: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class FiberInspection:
    """Read-only view of a managed fiber's ledger row."""

    fiber_id: str
    name: str
    status: str
    created_at: int
    idempotency_key: str | None = None
    snapshot: Any = None
    error: str | None = None
    metadata: dict[str, Any] | None = None
    started_at: int | None = None
    settled_at: int | None = None


@dataclass
class StartFiberResult(FiberInspection):
    """A start_fiber outcome: the captured row plus whether it was newly run."""

    accepted: bool = False


@dataclass
class FiberRecoveryResult:
    """Returned from on_fiber_recovered to settle a managed fiber's ledger."""

    status: str
    snapshot: Any = MISSING
    error: Any = MISSING
    reason: str | None = None
    metadata: dict[str, Any] | None = None


class FiberSignal:
    """Cooperative cancellation. Never Task.cancel(): a fiber checks `aborted`
    or awaits `wait()` and stops on its own, so cancel_fiber does not have to
    wait for uncooperative code to notice."""

    def __init__(self) -> None:
        self._event = asyncio.Event()
        self.reason: str | None = None

    @property
    def aborted(self) -> bool:
        return self._event.is_set()

    def abort(self, reason: str | None = None) -> None:
        if not self._event.is_set():
            self.reason = reason
            self._event.set()

    async def wait(self) -> None:
        await self._event.wait()


# Identifies the fiber currently on the stack, so a bare stash() finds the right
# one even with concurrent fibers under asyncio.gather.
_fiber_ctx: ContextVar[FiberContext | None] = ContextVar("fiber_ctx", default=None)


def _completed_future() -> asyncio.Future[None]:
    fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    fut.set_result(None)
    return fut


def _recovery_error_message(result: FiberRecoveryResult) -> str | None:
    if result.status == "error":
        if result.error is MISSING:
            return None
        return error_message(result.error)
    if result.status in ("aborted", "interrupted"):
        return result.reason
    return None


def _validate_recovery_status(result: FiberRecoveryResult) -> None:
    if result.status not in _TERMINAL_STATUSES:
        allowed = ", ".join(sorted(_TERMINAL_STATUSES))
        raise ValueError(
            f"invalid fiber recovery status {result.status!r}; "
            f"expected one of {allowed}"
        )


class FiberCapability(LifecycleCapability):
    """Legacy durable fibers hosted by Lifecycle."""

    capability_id = "fibers"

    keep_alive_interval_ms: int = 30_000
    detached_fibers_enabled = False
    fiber_recovery_scan_deadline_ms: int = 10_000
    fiber_recovery_max_age_ms: int = 24 * 60 * 60 * 1000
    fiber_recovery_hook_timeout_ms: int = 10_000

    # Base recovery-alarm backoff caps at ~5 min so a poison hook stops waking the DO
    # every heartbeat, while a draining multi-pass recovery stays prompt (streak 0).
    fiber_recovery_max_backoff_ms: int = 5 * 60 * 1000

    def __init__(
        self,
        *,
        on_fiber_recovered: (
            Callable[[FiberRecoveryContext], Awaitable[FiberRecoveryResult | None]]
            | None
        ) = None,
        on_internal_fiber_recovery: (
            Callable[[FiberRecoveryContext], Awaitable[bool]] | None
        ) = None,
        keep_alive_interval_ms: Callable[[], int] | None = None,
        detached_fibers_enabled: Callable[[], bool] | None = None,
        recovery_scan_deadline_ms: Callable[[], int] | None = None,
        recovery_max_age_ms: Callable[[], int] | None = None,
        recovery_hook_timeout_ms: Callable[[], int] | None = None,
        recovery_max_backoff_ms: Callable[[], int] | None = None,
        maintenance_enabled: Callable[[], bool] | None = None,
        defer_startup_recovery_to_alarm: Callable[[], bool] | None = None,
        defer_internal_recovery: Callable[[FiberRecoveryContext], bool] | None = None,
        shared_schema: bool = False,
    ) -> None:
        self._fiber_active_ids: set[str] = set()
        self._managed_abort: dict[str, FiberSignal] = {}
        self._pending_detached_leases: dict[str, int] = {}
        self._managed_terminal_waiters: dict[str, set[asyncio.Future[None]]] = {}
        self._fiber_recovery_in_progress = False
        self._fiber_recovery_lock = asyncio.Lock()
        self._fiber_recovery_owner: asyncio.Task[Any] | None = None
        self._queued_recovery_filters: list[Callable[[FiberRecoveryContext], bool]] = []
        self._recovery_no_progress_scans = 0
        self._active_lease_generations: set[int] = set()
        self._next_lease_generation = 0
        self._lease_acquisition_lock = asyncio.Lock()
        self._keep_alive_refs = 0
        self._prepared = False
        self._activation_reconciled = False
        self._maintenance_sealed = False
        self._disposed = False
        self._shared_schema = shared_schema
        self._on_recovered = on_fiber_recovered
        self._on_internal_recovery = on_internal_fiber_recovery
        self._get_keep_alive_interval_ms = keep_alive_interval_ms or (
            lambda: self.keep_alive_interval_ms
        )
        self._get_detached_fibers_enabled = detached_fibers_enabled or (
            lambda: self.detached_fibers_enabled
        )
        self._get_recovery_scan_deadline_ms = recovery_scan_deadline_ms or (
            lambda: self.fiber_recovery_scan_deadline_ms
        )
        self._get_recovery_max_age_ms = recovery_max_age_ms or (
            lambda: self.fiber_recovery_max_age_ms
        )
        self._get_recovery_hook_timeout_ms = recovery_hook_timeout_ms or (
            lambda: self.fiber_recovery_hook_timeout_ms
        )
        self._get_recovery_max_backoff_ms = recovery_max_backoff_ms or (
            lambda: self.fiber_recovery_max_backoff_ms
        )
        self._get_maintenance_enabled = maintenance_enabled or (lambda: True)
        self._defer_startup_recovery_to_alarm = defer_startup_recovery_to_alarm or (
            lambda: False
        )
        self._defer_internal_recovery = defer_internal_recovery or (lambda _ctx: False)

    def _sql(self, query: str, *params: object) -> list[dict[str, Any]]:
        return self.lifecycle.sql.execute(query, *params)

    def _prepare(self) -> None:
        if self._prepared:
            return
        if self._shared_schema:
            tables = self._sql(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table' AND name = 'cf_agents_state'"
            )
            if tables:
                version = read_core_schema_version(self._sql)
                if version >= CORE_SCHEMA_VERSION:
                    self._prepared = True
                    return
        prepare_fiber_schema(self._sql)
        self._prepared = True

    async def _reconcile_activation(self) -> None:
        if self._activation_reconciled or self._fiber_recovery_in_progress:
            return
        self._prepare()
        if not self._maintenance_available():
            self._activation_reconciled = True
            return
        if self._defer_startup_recovery_to_alarm():
            if await self._canonicalize_maintenance_job(due_only=False):
                self._activation_reconciled = True
                return
            if self._has_pending_fiber_recovery():
                await self.lifecycle.jobs.push(self._maintenance_job(now_ms()))
            else:
                await self._synchronize_maintenance_intent()
            self._activation_reconciled = True
            return
        if await self._memory_breaker_deadline() is not None:
            self._activation_reconciled = True
            return
        await self._canonicalize_maintenance_job(due_only=True)
        await self._synchronize_maintenance_intent()
        await self._scan_run_fibers()
        await self._synchronize_maintenance_intent()
        self._activation_reconciled = True

    async def _canonicalize_maintenance_job(self, *, due_only: bool) -> bool:
        exists, persisted_time = self._maintenance_job_time()
        if not exists:
            return False
        current_time = now_ms()
        if persisted_time is not None and due_only and persisted_time > current_time:
            return False
        deadline = persisted_time
        if deadline is None:
            deadline = (
                current_time
                if not due_only
                else current_time + self._get_keep_alive_interval_ms()
            )
        await self.lifecycle.jobs.push(self._maintenance_job(deadline))
        return not due_only or persisted_time is not None

    def _maintenance_job_time(self) -> tuple[bool, int | None]:
        rows = self._sql(
            "SELECT time FROM cf_agents_jobs WHERE id = ? AND capability = ? LIMIT 1",
            _MAINTENANCE_JOB_ID,
            self.capability_id,
        )
        if not rows:
            return False, None
        persisted_time = rows[0]["time"]
        if type(persisted_time) is not int or persisted_time < 0:
            return True, None
        return True, persisted_time

    def _has_future_maintenance_wake(self) -> bool:
        exists, persisted_time = self._maintenance_job_time()
        return exists and type(persisted_time) is int and persisted_time > now_ms()

    async def _memory_breaker_deadline(self) -> int | None:
        exists, persisted_time = self._maintenance_job_time()
        if not exists or persisted_time is None or persisted_time <= now_ms():
            return None
        strikes = await self.lifecycle.storage.get(_OOM_ALARM_STRIKES_KEY)
        if type(strikes) is int and strikes > 0:
            return persisted_time
        return None

    @staticmethod
    def _maintenance_job(time: int) -> LifecycleJobPushOptions:
        return LifecycleJobPushOptions(
            id=_MAINTENANCE_JOB_ID,
            fn=_MAINTENANCE_JOB_FN,
            time=time,
            retry={"maxAttempts": 1},
            recovery_loop=True,
        )

    async def _run_operation(self, callback: Callable[[], Any]) -> Any:
        await self.lifecycle.ready()

        async def run() -> Any:
            await self._reconcile_activation()
            result = callback()
            if inspect.isawaitable(result):
                return await result
            return result

        return await self.lifecycle.run_in_host_context(run)

    async def on_start(self) -> None:
        try:
            await self._reconcile_activation()
        except Exception as error:
            self._activation_reconciled = False
            if _is_platform_failure(error):
                raise
            self._prepared = False

    async def keep_alive(self) -> Callable[[], None]:
        if self._maintenance_available() and not self.lifecycle.retained_work.available:
            raise RuntimeError("retained work is not available")
        generation = await self._run_operation(
            lambda: self._acquire_keep_alive(
                require_maintenance=True,
                require_retained_work=True,
            )
        )

        disposed = False

        def dispose() -> None:
            nonlocal disposed
            if disposed:
                return
            disposed = True
            self._release_keep_alive_retained(generation)

        return dispose

    async def _acquire_keep_alive(
        self,
        *,
        require_maintenance: bool = False,
        require_retained_work: bool = False,
    ) -> int | None:
        if require_maintenance and not self._maintenance_available():
            raise RuntimeError(_MAINTENANCE_DISABLED_ERROR)
        if require_retained_work and not self.lifecycle.retained_work.available:
            raise RuntimeError("retained work is not available")
        if not self._maintenance_available():
            return None

        async with self._lease_acquisition_lock:
            if require_maintenance and not self._maintenance_available():
                raise RuntimeError(_MAINTENANCE_DISABLED_ERROR)
            if require_retained_work and not self.lifecycle.retained_work.available:
                raise RuntimeError("retained work is not available")
            if not self._maintenance_available():
                return None
            self._next_lease_generation += 1
            generation = self._next_lease_generation
            self._active_lease_generations.add(generation)
            self._keep_alive_refs = len(self._active_lease_generations)
            if self._keep_alive_refs != 1:
                return generation
            try:
                await self._synchronize_maintenance_intent()
            except BaseException:
                self._active_lease_generations.discard(generation)
                self._keep_alive_refs = len(self._active_lease_generations)
                with suppress(Exception):
                    await self._synchronize_maintenance_intent()
                raise
            return generation

    def _release_keep_alive_retained(self, generation: int | None) -> None:
        if self._disposed or generation not in self._active_lease_generations:
            return
        self._active_lease_generations.discard(generation)
        self._keep_alive_refs = len(self._active_lease_generations)
        if self._keep_alive_refs == 0 and self.lifecycle.retained_work.available:
            self.lifecycle.retained_work.retain(
                lambda: self._run_retained_maintenance_sync()
            )

    async def _run_retained_maintenance_sync(self) -> None:
        if self._disposed:
            return
        try:
            await self.lifecycle.run_in_host_context(
                self._synchronize_maintenance_intent
            )
        except RuntimeError as error:
            if str(error) != "Lifecycle is disposed":
                raise

    async def _release_keep_alive(self, generation: int | None) -> None:
        if generation is None:
            return
        async with self._lease_acquisition_lock:
            if generation not in self._active_lease_generations:
                return
            self._active_lease_generations.discard(generation)
            self._keep_alive_refs = len(self._active_lease_generations)
            if self._keep_alive_refs == 0:
                await self._synchronize_maintenance_intent()

    async def keep_alive_while(self, fn: Callable[[], Awaitable[Any]]) -> Any:
        async def run() -> Any:
            generation = await self._acquire_keep_alive(require_maintenance=True)
            try:
                return await fn()
            finally:
                await self._release_keep_alive(generation)

        return await self._run_operation(run)

    def _spawn_background(self, factory: Callable[[], Awaitable[Any]]) -> None:
        self.lifecycle.retained_work.retain(factory)

    def _maintenance_available(self) -> bool:
        return (
            not self._disposed
            and not self._maintenance_sealed
            and self._get_maintenance_enabled()
        )

    async def _synchronize_maintenance_intent(self) -> None:
        if not self._maintenance_available():
            return
        pending = self._prepared and self._has_pending_fiber_recovery()
        if self._keep_alive_refs == 0 and not pending:
            await self.lifecycle.jobs.cancel(_MAINTENANCE_JOB_ID)
            return
        deadline = self._maintenance_deadline(pending)
        _, persisted_time = self._maintenance_job_time()
        breaker_deadline = await self._memory_breaker_deadline()
        if breaker_deadline is not None:
            deadline = max(deadline, breaker_deadline)
        elif persisted_time is not None:
            deadline = min(deadline, persisted_time)
        await self.lifecycle.jobs.push(self._maintenance_job(deadline))

    def _maintenance_deadline(self, pending: bool) -> int:
        deadline: int | None = None
        if self._keep_alive_refs > 0:
            deadline = now_ms() + self._get_keep_alive_interval_ms()
        if pending:
            recovery = now_ms() + self._recovery_backoff_ms()
            deadline = recovery if deadline is None else min(deadline, recovery)
        if deadline is None:
            raise RuntimeError("fiber maintenance requires live work")
        return deadline

    def _recovery_backoff_ms(self) -> int:
        # Base heartbeat interval doubled per no-progress scan, capped. Streak 0 (a scan
        # that drained work) keeps the next pass at the base interval so legitimate
        # multi-pass recovery stays prompt; only a stuck scan backs off.
        exp = min(self._recovery_no_progress_scans, 30)
        delay = self._get_keep_alive_interval_ms() * (2**exp)
        return min(delay, self._get_recovery_max_backoff_ms())

    async def on_job(
        self, context: LifecycleJobContext
    ) -> LifecycleJobReschedule | None:
        if (
            context.job.id != _MAINTENANCE_JOB_ID
            or context.job.fn != _MAINTENANCE_JOB_FN
        ):
            return None
        if not self._maintenance_available():
            return None
        self._prepare()
        await self._scan_run_fibers()
        pending = self._has_pending_fiber_recovery()
        if self._keep_alive_refs == 0 and not pending:
            return None
        return LifecycleJobReschedule(self._maintenance_deadline(pending))

    async def on_job_error(
        self,
        context: LifecycleJobContext,
        error: BaseException,
    ) -> LifecycleJobReschedule | None:
        if context.job.id != _MAINTENANCE_JOB_ID:
            return None
        if not self._maintenance_available():
            return None
        return LifecycleJobReschedule(now_ms() + self._recovery_backoff_ms())

    async def on_memory_limit(self, context: LifecycleMemoryLimitContext) -> None:
        if not context.sealed or not self._maintenance_job_involved(context):
            return
        self._maintenance_sealed = True
        self._terminalize_local_work(_MEMORY_LIMIT_ERROR, status="error")

    async def on_dispose(self) -> None:
        maintenance_enabled = self._get_maintenance_enabled()
        self._disposed = True
        self._maintenance_sealed = True
        self._prepare()
        self._terminalize_local_work(_DISPOSED_ERROR, status="aborted")
        if maintenance_enabled:
            await self.lifecycle.jobs.cancel(_MAINTENANCE_JOB_ID)

    def _maintenance_job_involved(
        self,
        context: LifecycleMemoryLimitContext,
    ) -> bool:
        jobs = context.purged_recovery_loop_jobs
        if context.executing is not None:
            jobs = (*jobs, context.executing)
        return any(
            job.id == _MAINTENANCE_JOB_ID
            and job.capability == self.capability_id
            and job.fn == _MAINTENANCE_JOB_FN
            for job in jobs
        )

    def _terminalize_local_work(
        self,
        reason: str,
        *,
        status: str,
    ) -> None:
        for signal in self._managed_abort.values():
            signal.abort(reason)
        if self._prepared:
            self._sql(
                "UPDATE cf_agents_fibers SET status = ?, error_message = ?, "
                "completed_at = ? WHERE status IN ('pending', 'running')",
                status,
                reason,
                now_ms(),
            )
            self._sql("DELETE FROM cf_agents_runs")
        for waiters in self._managed_terminal_waiters.values():
            for waiter in waiters:
                if not waiter.done():
                    waiter.set_result(None)
        self._managed_terminal_waiters.clear()
        self._managed_abort.clear()
        self._pending_detached_leases.clear()
        self._fiber_active_ids.clear()
        self._active_lease_generations.clear()
        self._keep_alive_refs = 0

    # ── public API ───────────────────────────────────────────────────────

    async def run_fiber(
        self, name: str, fn: Callable[[FiberContext], Awaitable[Any]]
    ) -> Any:
        return await self._run_operation(
            lambda: self._run_fiber_internal(gen_id(), name, fn)
        )

    def stash(self, data: Any) -> None:
        ctx = _fiber_ctx.get()
        if ctx is None:
            raise RuntimeError("stash() called outside a fiber")

        ctx.stash(data)

    async def start_fiber(
        self,
        name: str,
        fn: Callable[[FiberContext], Awaitable[Any]],
        *,
        fiber_id: str | None = None,
        idempotency_key: str | None = None,
        metadata: dict[str, Any] | None = None,
        wait_for_completion: bool = False,
    ) -> StartFiberResult:
        return await self._run_operation(
            lambda: self._start_fiber(
                name,
                fn,
                fiber_id=fiber_id,
                idempotency_key=idempotency_key,
                metadata=metadata,
                wait_for_completion=wait_for_completion,
            )
        )

    async def _start_fiber(
        self,
        name: str,
        fn: Callable[[FiberContext], Awaitable[Any]],
        *,
        fiber_id: str | None,
        idempotency_key: str | None,
        metadata: dict[str, Any] | None,
        wait_for_completion: bool,
    ) -> StartFiberResult:
        if fiber_id is not None and fiber_id.strip() == "":
            raise ValueError("fiber_id must not be blank")
        if idempotency_key is not None and idempotency_key.strip() == "":
            raise ValueError("idempotency_key must not be blank")
        if not wait_for_completion and not self._maintenance_available():
            raise RuntimeError(_MAINTENANCE_DISABLED_ERROR)
        if not wait_for_completion and not self.lifecycle.retained_work.available:
            raise RuntimeError("retained work is not available")
        if not wait_for_completion and not self._get_detached_fibers_enabled():
            raise RuntimeError(
                "detached fibers are disabled pending deployed runtime verification"
            )

        resolved_id = fiber_id if fiber_id is not None else gen_id()
        existing_by_id = self._read_fiber(resolved_id)
        existing_by_key = (
            self._read_fiber_by_key(idempotency_key) if idempotency_key else None
        )

        self._reject_conflicting_identity(
            existing_by_id, existing_by_key, fiber_id, resolved_id
        )

        existing = existing_by_id or existing_by_key
        if existing is not None:
            return await self._accept_existing(existing, wait_for_completion)

        self._sql(
            "INSERT INTO cf_agents_fibers (fiber_id, idempotency_key, name, "
            "status, snapshot, metadata_json, error_message, created_at, "
            "started_at, completed_at) VALUES (?, ?, ?, 'pending', NULL, ?, "
            "NULL, ?, NULL, NULL)",
            resolved_id,
            idempotency_key,
            name,
            dumps_wire(metadata) if metadata is not None else None,
            now_ms(),
        )
        row = self._read_fiber(resolved_id)
        if row is None:
            raise RuntimeError(f"Failed to create fiber {resolved_id}")

        if wait_for_completion:
            await self._execute_managed_fiber(resolved_id, name, fn)
            completed = await self._wait_for_managed_fiber(resolved_id)
            if completed is None:
                raise RuntimeError(f"Fiber {resolved_id} no longer exists")
            return _with_accepted(completed, True)

        # Claim it in memory before scheduling: waitUntil defers the coroutine to a
        # later loop turn, and a recovery scan running in that gap would see a pending
        # ledger row with no live execution and interrupt a fiber about to start.
        self._fiber_active_ids.add(resolved_id)
        try:
            lease_generation = await self._acquire_keep_alive(
                require_maintenance=True,
                require_retained_work=True,
            )
            if lease_generation is None:
                raise RuntimeError(_MAINTENANCE_DISABLED_ERROR)
            self._pending_detached_leases[resolved_id] = lease_generation
        except asyncio.CancelledError:
            self._fiber_active_ids.discard(resolved_id)
            with suppress(Exception):
                await self._synchronize_maintenance_intent()
            raise
        except BaseException:
            self._fiber_active_ids.discard(resolved_id)
            self._sql(
                "DELETE FROM cf_agents_fibers WHERE fiber_id = ? AND status = "
                "'pending'",
                resolved_id,
            )
            with suppress(Exception):
                await self._synchronize_maintenance_intent()
            raise
        current = self._read_fiber(resolved_id)
        if current is None or current["status"] != "pending":
            self._pending_detached_leases.pop(resolved_id, None)
            self._fiber_active_ids.discard(resolved_id)
            await self._release_keep_alive(lease_generation)
            if current is None:
                raise RuntimeError(f"Fiber {resolved_id} no longer exists")
            return _with_accepted(self._fiber_inspection_from_row(current), True)
        try:
            self._spawn_background(
                lambda: self._run_retained_managed_fiber(
                    resolved_id,
                    name,
                    fn,
                    lease_generation,
                )
            )
        except BaseException:
            self._pending_detached_leases.pop(resolved_id, None)
            self._fiber_active_ids.discard(resolved_id)
            self._sql(
                "DELETE FROM cf_agents_fibers WHERE fiber_id = ? AND status = "
                "'pending'",
                resolved_id,
            )
            await self._release_keep_alive(lease_generation)
            raise
        return _with_accepted(self._fiber_inspection_from_row(row), True)

    async def inspect_fiber(self, fiber_id: str) -> FiberInspection | None:
        return await self._run_operation(lambda: self._inspect_fiber(fiber_id))

    def _inspect_fiber(self, fiber_id: str) -> FiberInspection | None:
        row = self._read_fiber(fiber_id)
        return self._fiber_inspection_from_row(row) if row else None

    async def inspect_fiber_by_key(
        self, idempotency_key: str
    ) -> FiberInspection | None:
        return await self._run_operation(
            lambda: self._inspect_fiber_by_key(idempotency_key)
        )

    def _inspect_fiber_by_key(self, idempotency_key: str) -> FiberInspection | None:
        row = self._read_fiber_by_key(idempotency_key)
        return self._fiber_inspection_from_row(row) if row else None

    async def list_fibers(
        self,
        *,
        status: str | list[str] | None = None,
        name: str | None = None,
        limit: int | None = None,
    ) -> list[FiberInspection]:
        return await self._run_operation(
            lambda: self._list_fibers(status=status, name=name, limit=limit)
        )

    def _list_fibers(
        self,
        *,
        status: str | list[str] | None,
        name: str | None,
        limit: int | None,
    ) -> list[FiberInspection]:
        where: list[str] = []
        params: list[Any] = []

        statuses = self._normalize_status_filter(status)
        if statuses:
            where.append("status IN (" + _placeholders(statuses) + ")")
            params.extend(sorted(statuses))
        if name:
            where.append("name = ?")
            params.append(name)

        rows = self._select_fibers(
            where,
            "created",
            params,
            clamp(limit if limit is not None else 50, 1, 100),
        )
        return [self._fiber_inspection_from_row(row) for row in rows]

    async def cancel_fiber(self, fiber_id: str, reason: str | None = None) -> bool:
        return await self._run_operation(lambda: self._cancel_fiber(fiber_id, reason))

    async def _cancel_fiber(self, fiber_id: str, reason: str | None = None) -> bool:
        row = self._read_fiber(fiber_id)
        if _settled(row):
            return False
        self._sql(
            "UPDATE cf_agents_fibers SET status = 'aborted', error_message = ?, "
            "completed_at = ? WHERE fiber_id = ? AND status IN "
            "('pending', 'running')",
            reason,
            now_ms(),
            fiber_id,
        )
        signal = self._managed_abort.get(fiber_id)
        if signal is not None:
            signal.abort(reason)
        lease_generation = self._pending_detached_leases.pop(fiber_id, None)
        if lease_generation is not None:
            self._fiber_active_ids.discard(fiber_id)
            await self._release_keep_alive(lease_generation)
        self._notify_managed_fiber_terminal(fiber_id)
        return True

    async def cancel_fiber_by_key(
        self, idempotency_key: str, reason: str | None = None
    ) -> bool:
        async def cancel() -> bool:
            row = self._read_fiber_by_key(idempotency_key)
            return await self._cancel_fiber(row["fiber_id"], reason) if row else False

        return await self._run_operation(cancel)

    async def resolve_fiber(self, fiber_id: str, result: FiberRecoveryResult) -> bool:
        return await self._run_operation(lambda: self._resolve_fiber(fiber_id, result))

    def _resolve_fiber(self, fiber_id: str, result: FiberRecoveryResult) -> bool:
        _validate_recovery_status(result)
        row = self._read_fiber(fiber_id)
        if row is None or row["status"] != "interrupted":
            return False
        self._apply_managed_fiber_recovery_result(fiber_id, result)
        return True

    async def delete_fibers(
        self,
        *,
        status: str | list[str] | None = None,
        settled_before: int | None = None,
        limit: int | None = None,
    ) -> int:
        return await self._run_operation(
            lambda: self._delete_fibers(
                status=status,
                settled_before=settled_before,
                limit=limit,
            )
        )

    def _delete_fibers(
        self,
        *,
        status: str | list[str] | None,
        settled_before: int | None,
        limit: int | None,
    ) -> int:
        statuses = self._normalize_status_filter(status) or {
            "completed",
            "aborted",
            "error",
        }
        terminal = sorted(s for s in statuses if _is_terminal(s))
        if not terminal:
            return 0

        where = ["status IN (" + _placeholders(terminal) + ")"]
        params: list[Any] = list(terminal)
        if settled_before is not None:
            # IS NOT NULL rides along with the cutoff, so a terminal row that never
            # recorded a settle time is out of range rather than swept as ancient.
            where += ["completed_at IS NOT NULL", "completed_at < ?"]
            params.append(settled_before)

        rows = self._select_fibers(
            where,
            "settled",
            params,
            clamp(limit if limit is not None else 100, 1, 500),
        )
        for r in rows:
            self._sql(
                "DELETE FROM cf_agents_fibers WHERE fiber_id = ? AND status IN "
                "('completed', 'aborted', 'interrupted', 'error')",
                r["fiber_id"],
            )
        return len(rows)

    def _delete_settled_fiber(self, fiber_id: str) -> None:
        self._sql(
            "DELETE FROM cf_agents_fibers WHERE fiber_id = ? AND status IN "
            "('completed', 'aborted', 'interrupted', 'error')",
            fiber_id,
        )

    # ── internal execution ────────────────────────────────────────────────

    async def _run_fiber_internal(
        self,
        id: str,
        name: str,
        fn: Callable[[FiberContext], Awaitable[Any]],
        *,
        signal: FiberSignal | None = None,
        managed: bool = False,
        before_run_cleanup: Callable[..., None] | None = None,
        lease_generation: int | None = None,
    ) -> Any:
        if signal is None:
            signal = FiberSignal()

        # Outside the try that owns the finally-delete: on a collision with an existing
        # row the finally must not fire and delete that other row.
        self._sql(
            "INSERT INTO cf_agents_runs (id, name, snapshot, created_at) "
            "VALUES (?, ?, NULL, ?)",
            id,
            name,
            now_ms(),
        )
        self._fiber_active_ids.add(id)

        def write_snapshot(data: Any) -> None:
            snapshot = dumps_wire(data)
            self._sql(
                "UPDATE cf_agents_runs SET snapshot = ? WHERE id = ?", snapshot, id
            )
            if managed:
                self._sql(
                    "UPDATE cf_agents_fibers SET snapshot = ? WHERE fiber_id = ?",
                    snapshot,
                    id,
                )

        preserve = False
        try:
            if lease_generation is None:
                lease_generation = await self._acquire_keep_alive()

            ctx = FiberContext(id=id, signal=signal, _stash=write_snapshot)
            token = _fiber_ctx.set(ctx)
            try:
                result = await fn(ctx)
                if before_run_cleanup is not None:
                    before_run_cleanup(True)
                return result
            except asyncio.CancelledError:
                raise
            except BaseException as error:
                if before_run_cleanup is not None and not _is_platform_failure(error):
                    before_run_cleanup(False, error)
                raise
            finally:
                _fiber_ctx.reset(token)
        except asyncio.CancelledError:
            # Cancellation can arrive while the first maintenance alarm is arming,
            # before the callback starts. The run row remains the recovery handoff in
            # either case.
            preserve = True
            raise
        except BaseException as error:
            preserve = _is_platform_failure(error)
            raise
        finally:
            self._fiber_active_ids.discard(id)
            if not preserve:
                self._sql("DELETE FROM cf_agents_runs WHERE id = ?", id)
            if lease_generation is not None:
                await self._release_keep_alive(lease_generation)
            elif preserve:
                with suppress(Exception):
                    await self._synchronize_maintenance_intent()

    async def _run_retained_managed_fiber(
        self,
        fiber_id: str,
        name: str,
        fn: Callable[[FiberContext], Awaitable[Any]],
        lease_generation: int,
    ) -> None:
        if self._disposed:
            self._pending_detached_leases.pop(fiber_id, None)
            self._fiber_active_ids.discard(fiber_id)
            await self._release_keep_alive(lease_generation)
            return

        async def run() -> None:
            try:
                await self._execute_managed_fiber(
                    fiber_id,
                    name,
                    fn,
                    lease_generation=lease_generation,
                )
            finally:
                self._pending_detached_leases.pop(fiber_id, None)
                self._fiber_active_ids.discard(fiber_id)
                await self._release_keep_alive(lease_generation)

        try:
            await self._run_operation(run)
        except RuntimeError as error:
            if str(error) != "Lifecycle is disposed":
                raise

    async def _execute_managed_fiber(
        self,
        fiber_id: str,
        name: str,
        fn: Callable[[FiberContext], Awaitable[Any]],
        *,
        lease_generation: int | None = None,
    ) -> None:
        row = self._read_fiber(fiber_id)
        if row is None or row["status"] != "pending":
            # Release the start-time claim: this path never reaches the runner that
            # would otherwise clear it.
            self._fiber_active_ids.discard(fiber_id)
            return

        signal = FiberSignal()
        self._managed_abort[fiber_id] = signal
        self._pending_detached_leases.pop(fiber_id, None)
        self._sql(
            "UPDATE cf_agents_fibers SET status = 'running', started_at = ? "
            "WHERE fiber_id = ? AND status = 'pending'",
            now_ms(),
            fiber_id,
        )
        updated = self._read_fiber(fiber_id)
        if updated is None or updated["status"] != "running":
            # Moved out of pending between the read and the write, so there is nothing
            # to run.
            self._managed_abort.pop(fiber_id, None)
            self._fiber_active_ids.discard(fiber_id)
            return

        settled = False

        def before_run_cleanup(ok: bool, error: Any = None) -> None:
            nonlocal settled
            settled = True
            self._settle_managed_fiber_execution(fiber_id, ok, error, signal)

        try:
            await self._run_fiber_internal(
                fiber_id,
                name,
                fn,
                signal=signal,
                managed=True,
                before_run_cleanup=before_run_cleanup,
                lease_generation=lease_generation,
            )
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            # Reached only for a failure outside fn, where the cleanup never fired.
            if _is_platform_failure(error):
                raise
            if not settled:
                self._settle_managed_fiber_execution(fiber_id, False, error, signal)
        finally:
            self._managed_abort.pop(fiber_id, None)

    def _settle_managed_fiber_execution(
        self, fiber_id: str, ok: bool, error: Any, signal: FiberSignal
    ) -> None:
        now = now_ms()
        if ok:
            self._sql(
                "UPDATE cf_agents_fibers SET status = 'completed', "
                "completed_at = ? WHERE fiber_id = ? AND status = 'running'",
                now,
                fiber_id,
            )
            self._notify_managed_fiber_terminal(fiber_id)
            return

        # A callback that ignored cancellation still settles as aborted, so the
        # conditional WHERE cannot regress an already-aborted row.
        status = "aborted" if signal.aborted else "error"
        self._sql(
            "UPDATE cf_agents_fibers SET status = ?, error_message = ?, "
            "completed_at = ? WHERE fiber_id = ? AND status = 'running'",
            status,
            error_message(error),
            now,
            fiber_id,
        )
        self._notify_managed_fiber_terminal(fiber_id)

    def _apply_managed_fiber_recovery_result(
        self, fiber_id: str, result: FiberRecoveryResult
    ) -> None:
        now = now_ms()
        snapshot = _stringify_snapshot(result.snapshot)
        # Named to avoid shadowing the error_message helper imported above.
        message = _recovery_error_message(result)
        include_metadata = result.status == "completed" and result.metadata is not None

        # metadata_json takes the same COALESCE treatment as snapshot, so binding NULL
        # leaves the stored value alone and the two branches collapse to one statement.
        self._sql(
            "UPDATE cf_agents_fibers SET status = ?, "
            "snapshot = COALESCE(?, snapshot), "
            "metadata_json = COALESCE(?, metadata_json), error_message = ?, "
            "completed_at = ? WHERE fiber_id = ? AND status = 'interrupted'",
            result.status,
            snapshot,
            dumps_wire(result.metadata) if include_metadata else None,
            message,
            now,
            fiber_id,
        )
        self._notify_managed_fiber_terminal(fiber_id)

    # ── waiters ─────────────────────────────────────────────────────────

    def _notify_managed_fiber_terminal(self, fiber_id: str) -> None:
        row = self._read_fiber(fiber_id)
        if not _settled(row):
            return
        waiters = self._managed_terminal_waiters.pop(fiber_id, None)
        if not waiters:
            return
        for fut in waiters:
            if not fut.done():
                fut.set_result(None)

    def _wait_for_managed_fiber_terminal(self, fiber_id: str) -> Awaitable[None]:
        row = self._read_fiber(fiber_id)
        if _settled(row):
            return _completed_future()

        fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._managed_terminal_waiters.setdefault(fiber_id, set()).add(fut)
        return fut

    async def _wait_for_managed_fiber(self, fiber_id: str) -> FiberInspection | None:
        row = self._read_fiber(fiber_id)
        if _settled(row):
            return self._fiber_inspection_from_row(row) if row else None

        if fiber_id not in self._managed_abort:
            # No in-memory runner owns it, so an eviction left it behind. The scan
            # finalizes ledger-only non-terminal rows, which resolves the wait.
            if not self._get_maintenance_enabled():
                raise RuntimeError(
                    "cannot wait for an unmanaged fiber without routed recovery"
                )
            if self._fiber_recovery_in_progress:
                raise RuntimeError(
                    "cannot wait for an unmanaged fiber during fiber recovery"
                )
            if await self._memory_breaker_deadline() is not None:
                raise RuntimeError(
                    "cannot wait for an unmanaged fiber during memory backoff"
                )
            await self._scan_run_fibers()

        await self._wait_for_managed_fiber_terminal(fiber_id)
        return await self.inspect_fiber(fiber_id)

    # ── recovery scan ───────────────────────────────────────────────────

    async def _check_run_fibers(self) -> None:
        await self._run_operation(self._scan_run_fibers)

    async def resume_deferred_recovery(
        self,
        recovery_filter: Callable[[FiberRecoveryContext], bool],
    ) -> None:
        if not self._maintenance_available():
            return
        if self._fiber_recovery_owner is asyncio.current_task():
            self._queued_recovery_filters.append(recovery_filter)
            return
        self._prepare()
        await self._scan_run_fibers(recovery_filter=recovery_filter, wait=True)
        await self._synchronize_maintenance_intent()

    async def _scan_run_fibers(
        self,
        recovery_filter: Callable[[FiberRecoveryContext], bool] | None = None,
        *,
        wait: bool = False,
    ) -> None:
        if (
            self._fiber_recovery_in_progress or self._fiber_recovery_lock.locked()
        ) and not wait:
            return
        async with self._fiber_recovery_lock:
            if await self._memory_breaker_deadline() is not None:
                return
            self._fiber_recovery_owner = asyncio.current_task()
            self._fiber_recovery_in_progress = True
            scan_started_at = now_ms()
            deadline_ms = self._get_recovery_scan_deadline_ms()
            max_age_ms = self._get_recovery_max_age_ms()
            made_progress = False

            try:
                made_progress = await self._recover_orphan_run_rows(
                    scan_started_at,
                    deadline_ms,
                    max_age_ms,
                    recovery_filter,
                )
                if await self._recover_ledger_only_rows(
                    scan_started_at,
                    deadline_ms,
                    recovery_filter,
                ):
                    made_progress = True
                if recovery_filter is None:
                    pending = self._has_pending_fiber_recovery()
                    if made_progress or not pending:
                        self._recovery_no_progress_scans = 0
                    else:
                        self._recovery_no_progress_scans += 1
            finally:
                self._fiber_recovery_in_progress = False
                self._fiber_recovery_owner = None
        if self._queued_recovery_filters:
            queued = tuple(self._queued_recovery_filters)
            self._queued_recovery_filters.clear()
            await self._scan_run_fibers(
                recovery_filter=lambda ctx: any(check(ctx) for check in queued),
                wait=True,
            )

    async def _recover_orphan_run_rows(
        self,
        scan_started_at: int,
        deadline_ms: int,
        max_age_ms: int,
        recovery_filter: Callable[[FiberRecoveryContext], bool] | None = None,
    ) -> bool:
        made_progress = False
        rows = self._sql("SELECT id, name, snapshot, created_at FROM cf_agents_runs")
        for row in rows:
            if _deadline_passed(scan_started_at, deadline_ms):
                break
            if row["id"] in self._fiber_active_ids:
                continue

            ctx = FiberRecoveryContext(
                id=row["id"],
                name=row["name"],
                snapshot=loads_or_none(row["snapshot"]),
                created_at=row["created_at"],
            )
            if recovery_filter is not None and not recovery_filter(ctx):
                continue
            managed_row = self._read_fiber(row["id"])
            if self._defer_internal_recovery(ctx):
                continue

            if managed_row is not None:
                if _is_terminal(managed_row["status"]):
                    # The ledger already holds the terminal status and this run row is
                    # just stale.
                    self._sql("DELETE FROM cf_agents_runs WHERE id = ?", row["id"])
                    self._notify_managed_fiber_terminal(row["id"])
                    made_progress = True
                    continue
                self._mark_interrupted(row["id"], row["snapshot"])
                ctx.idempotency_key = managed_row["idempotency_key"]
                ctx.metadata = loads_dict_or_none(managed_row["metadata_json"])
                ctx.status = "interrupted"

            recovered = await self._run_fiber_recovery_hook(ctx, managed_row)
            # Managed rows always clean up. An unmanaged row is retained for a later
            # retry, but only until it ages out — a hook that keeps throwing would loop
            # forever.
            too_old = max_age_ms > 0 and now_ms() - row["created_at"] > max_age_ms
            if recovered or managed_row is not None or too_old:
                self._sql("DELETE FROM cf_agents_runs WHERE id = ?", row["id"])
                made_progress = True
            if managed_row is not None:
                self._notify_managed_fiber_terminal(row["id"])

        return made_progress

    async def _recover_ledger_only_rows(
        self,
        scan_started_at: int,
        deadline_ms: int,
        recovery_filter: Callable[[FiberRecoveryContext], bool] | None = None,
    ) -> bool:
        # Managed fibers stuck with no live run row: the process died before ever
        # writing one, so finalize them here.
        made_progress = False
        rows = self._sql(_LEDGER_ONLY_ROWS_QUERY, "pending", "running")
        for row in rows:
            if _deadline_passed(scan_started_at, deadline_ms):
                break
            if row["fiber_id"] in self._fiber_active_ids:
                continue

            # TODO: FiberRecoveryContext.from_row maybe?
            ctx = FiberRecoveryContext(
                id=row["fiber_id"],
                name=row["name"],
                snapshot=loads_or_none(row["snapshot"]),
                created_at=row["created_at"],
                idempotency_key=row["idempotency_key"],
                metadata=loads_dict_or_none(row["metadata_json"]),
                status="interrupted",
            )
            if recovery_filter is not None and not recovery_filter(ctx):
                continue
            if self._defer_internal_recovery(ctx):
                continue
            self._mark_interrupted(row["fiber_id"])
            await self._run_fiber_recovery_hook(ctx, row)
            # Finalized this pass whatever the hook returned, which counts as progress.
            self._notify_managed_fiber_terminal(row["fiber_id"])
            made_progress = True

        return made_progress

    def _mark_interrupted(self, fiber_id: str, snapshot: str | None = None) -> None:
        # COALESCE, not a plain set: the ledger-only caller has no snapshot to offer,
        # and a managed stash writes the run row and the ledger together, so the orphan
        # caller's is never staler than what is already stored.
        self._sql(
            "UPDATE cf_agents_fibers SET status = 'interrupted', "
            "snapshot = COALESCE(?, snapshot), completed_at = ? WHERE fiber_id = ? "
            "AND status IN ('pending', 'running')",
            snapshot,
            now_ms(),
            fiber_id,
        )

    def _restore_recoverable_fiber(
        self,
        fiber_id: str,
        managed_row: dict[str, Any],
    ) -> None:
        prior_status = managed_row["status"]
        if prior_status not in {"pending", "running"}:
            return
        self._sql(
            "UPDATE cf_agents_fibers SET status = ?, completed_at = ? "
            "WHERE fiber_id = ? AND status = 'interrupted'",
            prior_status,
            managed_row["completed_at"],
            fiber_id,
        )

    async def _run_fiber_recovery_hook(
        self, ctx: FiberRecoveryContext, managed_row: dict[str, Any] | None
    ) -> bool:
        try:

            async def recover() -> bool:
                handled = False
                if self._on_internal_recovery is not None:
                    internal = self._on_internal_recovery(ctx)
                    timeout_ms = self._get_recovery_hook_timeout_ms()
                    if timeout_ms > 0:
                        handled = await asyncio.wait_for(
                            internal,
                            timeout=timeout_ms / 1000,
                        )
                    else:
                        handled = await internal
                if handled:
                    return True
                if ctx.name.startswith(INTERNAL_FIBER_PREFIX):
                    return False
                recovery_result = None
                if self._on_recovered is not None:
                    recovery_result = await self._on_recovered(ctx)
                if recovery_result is not None:
                    _validate_recovery_status(recovery_result)
                    if managed_row is not None:
                        self._apply_managed_fiber_recovery_result(
                            ctx.id, recovery_result
                        )
                return True

            async def recover_in_host_context() -> bool:
                result = await self.lifecycle.run_in_host_context(recover)
                if type(result) is not bool:
                    raise TypeError("fiber recovery callback must return a boolean")
                return result

            handled = await recover_in_host_context()
            return bool(handled)
        except Exception as exc:
            if _is_platform_failure(exc):
                if managed_row is not None:
                    self._restore_recoverable_fiber(ctx.id, managed_row)
                raise
            self._record_fiber_recovery_failure(ctx, managed_row, exc)
            return False

    def _record_fiber_recovery_failure(
        self, ctx: FiberRecoveryContext, managed_row: dict[str, Any] | None, error: Any
    ) -> None:
        if managed_row is None:
            return
        self._sql(
            "UPDATE cf_agents_fibers SET status = 'error', error_message = ?, "
            "completed_at = ? WHERE fiber_id = ? AND status = 'interrupted'",
            error_message(error),
            now_ms(),
            ctx.id,
        )
        self._notify_managed_fiber_terminal(ctx.id)

    def _has_pending_fiber_recovery(self) -> bool:
        for row in self._sql("SELECT id FROM cf_agents_runs"):
            if row["id"] not in self._fiber_active_ids:
                return True
        ledger_only = self._sql(_LEDGER_ONLY_COUNT_QUERY, "pending", "running")
        count = ledger_only[0]["count"] if ledger_only else 0
        return count > 0

    # ── ledger reads and mapping ──────────────────────────────────────────

    def _read_fiber(self, fiber_id: str) -> dict[str, Any] | None:
        rows = self._sql(
            "SELECT fiber_id, idempotency_key, name, status, snapshot, "
            "metadata_json, error_message, created_at, started_at, completed_at "
            "FROM cf_agents_fibers WHERE fiber_id = ? LIMIT 1",
            fiber_id,
        )
        return rows[0] if rows else None

    def _read_fiber_by_key(self, idempotency_key: str) -> dict[str, Any] | None:
        rows = self._sql(
            "SELECT fiber_id, idempotency_key, name, status, snapshot, "
            "metadata_json, error_message, created_at, started_at, completed_at "
            "FROM cf_agents_fibers "
            "WHERE idempotency_key = ? LIMIT 1",
            idempotency_key,
        )
        return rows[0] if rows else None

    def _reject_conflicting_identity(
        self,
        existing_by_id: dict[str, Any] | None,
        existing_by_key: dict[str, Any] | None,
        fiber_id: str | None,
        resolved_id: str,
    ) -> None:
        # Nothing resolved by key means nothing to disagree with the id about.
        if existing_by_key is None:
            return

        key_id = existing_by_key["fiber_id"]
        if (existing_by_id is not None and existing_by_id["fiber_id"] != key_id) or (
            fiber_id and key_id != resolved_id
        ):
            raise ValueError("fiberId and idempotencyKey refer to different fibers")

    async def _accept_existing(
        self, existing: dict[str, Any], wait_for_completion: bool
    ) -> StartFiberResult:
        if wait_for_completion and not _is_terminal(existing["status"]):
            waited = await self._wait_for_managed_fiber(existing["fiber_id"])
            if waited is not None:
                return _with_accepted(waited, False)
            raise RuntimeError(f"Fiber {existing['fiber_id']} no longer exists")
        return _with_accepted(self._fiber_inspection_from_row(existing), False)

    # TODO: this could be FiberInspection.from_row once construction is shared.
    def _fiber_inspection_from_row(self, row: dict[str, Any]) -> FiberInspection:
        return FiberInspection(
            fiber_id=row["fiber_id"],
            name=row["name"],
            status=row["status"],
            created_at=row["created_at"],
            idempotency_key=row["idempotency_key"],
            snapshot=loads_or_none(row["snapshot"]),
            error=row["error_message"],
            metadata=loads_dict_or_none(row["metadata_json"]),
            started_at=row["started_at"],
            settled_at=row["completed_at"],
        )

    def _normalize_status_filter(
        self, status: str | list[str] | None
    ) -> set[str] | None:
        if not status:
            return None
        if isinstance(status, str):
            return {status}
        return set(status)

    def _select_fibers(
        self,
        where: list[str],
        order: Literal["created", "settled"],
        params: list[Any],
        limit: int,
    ) -> list[dict[str, Any]]:
        # Predicates are interpolated and values are always bound. A status or name
        # can reach here from a caller's own RPC surface, so neither is ever a
        # fragment.
        query = _FIBER_SELECT
        if where:
            query += " WHERE " + " AND ".join(where)
        if order == "created":
            query += " ORDER BY created_at DESC, fiber_id DESC LIMIT ?"
        else:
            query += " ORDER BY completed_at ASC, created_at ASC LIMIT ?"
        return self._sql(query, *params, limit)


def _is_terminal(status: str) -> bool:
    return status in _TERMINAL_STATUSES


def _settled(row: dict[str, Any] | None) -> bool:
    # A missing row counts as settled: the ledger is the only record of a managed fiber,
    # so there is nothing left to cancel or wait on.
    return row is None or _is_terminal(row["status"])


def _deadline_passed(scan_started_at: int, deadline_ms: int) -> bool:
    return deadline_ms > 0 and now_ms() - scan_started_at > deadline_ms


def _with_accepted(inspection: FiberInspection, accepted: bool) -> StartFiberResult:
    # vars(), not asdict(): StartFiberResult subclasses FiberInspection, so the fields
    # line up, and asdict would deep-copy snapshot and metadata instead of passing the
    # same objects the caller inspected.
    return StartFiberResult(**vars(inspection), accepted=accepted)
