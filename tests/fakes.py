"""Deterministic fakes for the runtime surface `agents` depends on.

Storage is real in-memory sqlite so migrations and queries run for real; the
socket, alarm, and waitUntil surfaces are small recorders.
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import json
import sqlite3
import types
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import _runtime_stubs

_runtime_stubs.install()

from agents import AIChatAgent, Agent, ChatOptions  # noqa: E402

# ── storage ──────────────────────────────────────────────────────────────

_MISSING = object()


class AsyncGate:
    def __init__(self) -> None:
        self._blocked = asyncio.Event()
        self._released = asyncio.Event()

    async def block(self) -> None:
        self._blocked.set()
        await self._released.wait()

    async def wait_until_blocked(self) -> None:
        await self._blocked.wait()

    def release(self) -> None:
        self._released.set()


class FakeCursor:
    def __init__(self, rows: list[dict[str, Any]]):
        self._rows = rows

    def toArray(self) -> list[dict[str, Any]]:
        return self._rows


class FakeSql:
    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def exec(self, query: str, *params: Any) -> FakeCursor:
        cur = self._conn.execute(query, params)
        rows = [dict(row) for row in cur.fetchall()]
        return FakeCursor(rows)


class _FakeDurableState:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self.kv: dict[str, Any] = {}
        self.alarm_time_ms: int | None = None
        self.alarm_history_ms: list[tuple[str, int | None]] = []
        self.storage_sync_calls = 0
        self.websockets: list[Any] = []
        self.websocket_tags: dict[int, tuple[str, ...]] = {}


class FakeStorage:
    def __init__(
        self,
        conn: sqlite3.Connection,
        state: _FakeDurableState | None = None,
    ):
        self._conn = conn
        self._state = state or _FakeDurableState(conn)
        self.sql = FakeSql(conn)
        self._next_alarm_gate: AsyncGate | None = None

    @property
    def alarm_time_ms(self) -> int | None:
        return self._state.alarm_time_ms

    @property
    def alarm_history_ms(self) -> Sequence[tuple[str, int | None]]:
        return tuple(self._state.alarm_history_ms)

    async def get(self, key: str | Sequence[str]) -> Any:
        if isinstance(key, str):
            return copy.deepcopy(self._state.kv.get(key))
        return {
            item: copy.deepcopy(self._state.kv[item])
            for item in sorted(key)
            if item in self._state.kv
        }

    async def put(self, key: str | dict[str, Any], value: Any = _MISSING) -> None:
        if isinstance(key, dict):
            self._state.kv.update(copy.deepcopy(key))
            return
        if value is _MISSING:
            raise TypeError("put requires a value")
        self._state.kv[key] = copy.deepcopy(value)

    async def delete(self, key: str | Sequence[str]) -> bool | int:
        if isinstance(key, str):
            return self._state.kv.pop(key, _MISSING) is not _MISSING
        deleted = 0
        for item in key:
            if self._state.kv.pop(item, _MISSING) is not _MISSING:
                deleted += 1
        return deleted

    async def list(
        self,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        options = options or {}
        prefix = options.get("prefix", "")
        start = options.get("start")
        start_after = options.get("startAfter")
        if start is not None and start_after is not None:
            raise TypeError("start and startAfter are mutually exclusive")
        end = options.get("end")
        reverse = options.get("reverse", False)
        limit = options.get("limit")
        keys = [
            key
            for key in sorted(self._state.kv, reverse=reverse)
            if key.startswith(prefix)
            and (start is None or key >= start)
            and (start_after is None or key > start_after)
            and (end is None or key < end)
        ]
        if limit is not None:
            keys = keys[:limit]
        return {key: copy.deepcopy(self._state.kv[key]) for key in keys}

    def transactionSync[T](self, callback: Callable[[], T]) -> T:
        self._conn.execute("SAVEPOINT fake_storage")
        try:
            result = callback()
            if inspect.isawaitable(result):
                close = getattr(result, "close", None)
                if close is not None:
                    close()
                raise TypeError("transactionSync requires a synchronous callback")
        except BaseException:
            self._conn.execute("ROLLBACK TO SAVEPOINT fake_storage")
            self._conn.execute("RELEASE SAVEPOINT fake_storage")
            raise
        self._conn.execute("RELEASE SAVEPOINT fake_storage")
        return result

    def pause_next_alarm(self, gate: AsyncGate) -> None:
        self._next_alarm_gate = gate

    async def setAlarm(self, when_ms: int) -> None:
        gate = self._next_alarm_gate
        self._next_alarm_gate = None
        if gate is not None:
            await gate.block()
        self._state.alarm_time_ms = when_ms
        self._state.alarm_history_ms.append(("set", when_ms))

    async def getAlarm(self) -> int | None:
        return self._state.alarm_time_ms

    async def deleteAlarm(self) -> None:
        self._state.alarm_time_ms = None
        self._state.alarm_history_ms.append(("delete", None))

    @property
    def sync_calls(self) -> int:
        return self._state.storage_sync_calls

    async def sync(self) -> None:
        self._state.storage_sync_calls += 1


class FakeId:
    def __init__(self, name: str):
        self.name = name


class FakeCtx:
    def __init__(
        self,
        name: str = "test-agent",
        *,
        websockets: list[Any] | None = None,
        conn: sqlite3.Connection | None = None,
        runtime: FakeDurableObjectRuntime | None = None,
    ):
        self._runtime = runtime
        if runtime is None:
            self.conn = conn or new_sqlite()
            self._durable_state = _FakeDurableState(self.conn)
            self._durable_state.websockets.extend(websockets or [])
        else:
            self.conn = runtime._state.conn
            self._durable_state = runtime._state
        self.storage = FakeStorage(self.conn, self._durable_state)
        self.id = FakeId(name)
        self.accepted_websockets: list[Any] = []
        self.abort_calls: list[tuple[str, object | None]] = []
        self.exports: Any = None
        self.facets: Any = None

    def getWebSockets(self, tag: str | None = None) -> list[Any]:
        if tag is None:
            return [
                websocket
                for websocket in self._durable_state.websockets
                if not getattr(websocket, "closed", False)
            ]
        return [
            websocket
            for websocket in self._durable_state.websockets
            if not getattr(websocket, "closed", False)
            if tag in self._durable_state.websocket_tags.get(id(websocket), ())
        ]

    def acceptWebSocket(
        self,
        websocket: Any,
        tags: Sequence[str] = (),
    ) -> None:
        self.accepted_websockets.append(websocket)
        if websocket not in self._durable_state.websockets:
            self._durable_state.websockets.append(websocket)
        self._durable_state.websocket_tags[id(websocket)] = tuple(tags)

    async def blockConcurrencyWhile(self, fn: Callable[[], Any]) -> Any:
        return await fn()

    def abort(self, reason: str, options: object | None = None) -> None:
        self.abort_calls.append((reason, options))


class FakeDurableObjectRuntime:
    def __init__(self, name: str = "test-agent"):
        self.name = name
        self._state = _FakeDurableState(new_sqlite())
        self.evictions = 0

    def new_context(self) -> FakeCtx:
        return FakeCtx(name=self.name, runtime=self)

    def evict(self) -> FakeCtx:
        """Return a new activation; callers must stop using the previous context."""
        self.evictions += 1
        return self.new_context()


class FakeRouteTransport:
    def __init__(self, owner: str, *, max_payload_bytes: int):
        self.owner = owner
        self.max_payload_bytes = max_payload_bytes
        self._handlers: dict[tuple[str, str], Callable[[Any], Any]] = {}
        self._calls: list[dict[str, Any]] = []

    @property
    def calls(self) -> tuple[dict[str, Any], ...]:
        return tuple(copy.deepcopy(self._calls))

    def register(
        self,
        target: str,
        capability_id: str,
        handler: Callable[[Any], Any],
    ) -> None:
        self._handlers[(target, capability_id)] = handler

    async def route(
        self,
        *,
        version: int,
        source: str,
        target: str,
        capability_id: str,
        payload: Any,
    ) -> Any:
        if version != 1:
            raise ValueError(f"unknown route version: {version}")
        if not self._owns(source) or not self._owns(target):
            raise PermissionError("route address is outside the transport owner")
        handler = self._handlers.get((target, capability_id))
        if handler is None:
            raise LookupError(f"unknown route capability: {capability_id}")
        payload_bytes = json.dumps(
            payload,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        if len(payload_bytes) > self.max_payload_bytes:
            raise ValueError("route payload exceeds the configured byte limit")
        envelope = {
            "version": version,
            "source": source,
            "target": target,
            "capability_id": capability_id,
            "payload": copy.deepcopy(payload),
        }
        self._calls.append(envelope)
        result = handler(copy.deepcopy(payload))
        if inspect.isawaitable(result):
            return await result
        return result

    def _owns(self, address: str) -> bool:
        return address == self.owner or address.startswith(f"{self.owner}/")


class FakeResourceTracker:
    def __init__(self) -> None:
        self._active: dict[tuple[str, str], str] = {}
        self._events: list[tuple[str, str, str, str]] = []

    @property
    def active(self) -> dict[tuple[str, str], str]:
        return dict(self._active)

    @property
    def events(self) -> tuple[tuple[str, str, str, str], ...]:
        return tuple(self._events)

    def claim(self, kind: str, key: str, owner: str) -> Callable[[], None]:
        resource = (kind, key)
        if resource in self._active:
            raise RuntimeError(f"resource already claimed: {kind}/{key}")
        self._active[resource] = owner
        self._events.append(("claim", kind, key, owner))
        released = False

        def release() -> None:
            nonlocal released
            if released:
                return
            released = True
            self._active.pop(resource)
            self._events.append(("release", kind, key, owner))

        return release


def new_sqlite() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.isolation_level = None  # autocommit, matching the runtime's per-statement sql
    return conn


def make_sql(conn: sqlite3.Connection | None = None) -> Callable[..., list[dict]]:
    """A standalone `sql(query, *params) -> list[dict]`, matching Agent.sql."""
    conn = conn or new_sqlite()

    def sql(query: str, *params: Any) -> list[dict[str, Any]]:
        cur = conn.execute(query, params)
        return [dict(row) for row in cur.fetchall()]

    return sql


# ── sockets and connections ────────────────────────────────────────────────


class FakeAttachment:
    # Stands in for the JsProxy deserializeAttachment hands back; .to_py() unwraps.
    def __init__(self, value: Any):
        self._value = value

    def to_py(self) -> Any:
        return self._value


class FakeSocket:
    """The hibernatable socket proxy the runtime accepts and hydrates."""

    def __init__(self, attachment: Any = None):
        self.sent: list[str] = []
        self._attachment = attachment
        self.closed = False

    def send(self, data: str) -> None:
        if self.closed:
            raise RuntimeError("WebSocket send() after close")
        self.sent.append(data)

    def close(self, code: int = 1000, reason: str = "") -> None:
        self.closed = True

    def serializeAttachment(self, value: Any) -> None:
        self._attachment = value

    def deserializeAttachment(self) -> Any:
        if self._attachment is None:
            return None
        return FakeAttachment(self._attachment)


class FakeConnection:
    """A connection that records the frames handed to it, so protocol tests
    assert on the objects sent rather than re-parsing JSON."""

    def __init__(self, id: str = "conn-1", *, open: bool = True):
        self.id = id
        self.open = open
        self.sent: list[Any] = []
        self._raw_state: object = None

    def send(self, data: Any) -> None:
        if not self.open:
            raise RuntimeError("WebSocket send() after close")
        self.sent.append(data)

    def send_json(self, data: dict[str, Any]) -> None:
        self.send(data)

    def send_if_open(self, data: Any) -> bool:
        if not self.open:
            return False
        self.sent.append(data)
        return True

    def _get_raw_state(self) -> object:
        return copy.deepcopy(self._raw_state)

    def _set_raw_state(self, state: object) -> object:
        self._raw_state = copy.deepcopy(state)
        return copy.deepcopy(state)

    def close(self) -> None:
        self.open = False

    @property
    def frames(self) -> list[Any]:
        # A uniform dict view: broadcast_json sends JSON strings while a unicast
        # send_json/send_if_open sends the dict, so tests read one shape either way.
        out: list[Any] = []
        for item in self.sent:
            if isinstance(item, str):
                try:
                    out.append(json.loads(item))
                except ValueError:
                    out.append(item)
            else:
                out.append(item)
        return out


class FakeChildAgentToolStub:
    """Protocol-complete fake for the hidden child agent-tool RPC surface."""

    def __init__(
        self,
        *,
        start: dict[str, Any] | None = None,
        inspection: dict[str, Any] | None = None,
        chunks: list[dict[str, Any]] | None = None,
        delegate: Any = None,
        start_hook: Callable[..., Any] | None = None,
        inspect_hook: Callable[..., Any] | None = None,
        start_error: Exception | None = None,
        inspect_error: Exception | None = None,
        chunks_error: Exception | None = None,
    ) -> None:
        self.start_result = start
        self.inspection_result = inspection
        self.chunk_results = list(chunks or [])
        self.delegate = delegate
        self.start_hook = start_hook
        self.inspect_hook = inspect_hook
        self.start_error = start_error
        self.inspect_error = inspect_error
        self.chunks_error = chunks_error
        self.start_calls: list[tuple[str, str]] = []
        self.inspect_calls: list[str] = []
        self.chunk_calls: list[tuple[str, int]] = []
        self.cancel_calls: list[tuple[str, str | None]] = []

    async def _cf_start_agent_tool_run(self, input_json: str, run_id: str) -> str:
        self.start_calls.append((input_json, run_id))
        if self.delegate is not None:
            result = await self.delegate._cf_start_agent_tool_run(input_json, run_id)
        else:
            payload = self.start_result or {
                "runId": run_id,
                "status": "running",
                "startedAt": 0,
            }
            result = json.dumps(payload)
        await self._run_hook(self.start_hook, input_json, run_id)
        if self.start_error is not None:
            raise self.start_error
        return result

    async def _cf_inspect_agent_tool_run(self, run_id: str) -> str | None:
        self.inspect_calls.append(run_id)
        if self.delegate is not None:
            result = await self.delegate._cf_inspect_agent_tool_run(run_id)
        elif self.inspection_result is None:
            result = None
        else:
            result = json.dumps({"runId": run_id, **self.inspection_result})
        await self._run_hook(self.inspect_hook, run_id)
        if self.inspect_error is not None:
            raise self.inspect_error
        return result

    async def _cf_get_agent_tool_chunks(
        self,
        run_id: str,
        after_sequence: int = -1,
        limit: int = 100,
    ) -> str:
        self.chunk_calls.append((run_id, after_sequence))
        if self.delegate is not None:
            result = await self.delegate._cf_get_agent_tool_chunks(
                run_id, after_sequence, limit
            )
        else:
            result = json.dumps(
                [
                    chunk
                    for chunk in self.chunk_results
                    if chunk["sequence"] > after_sequence
                ][:limit]
            )
        if self.chunks_error is not None:
            raise self.chunks_error
        return result

    async def _cf_cancel_agent_tool_run(
        self, run_id: str, reason: str | None = None
    ) -> None:
        self.cancel_calls.append((run_id, reason))
        if self.delegate is not None:
            await self.delegate._cf_cancel_agent_tool_run(run_id, reason)

    @staticmethod
    async def _run_hook(hook: Callable[..., Any] | None, *args: Any) -> None:
        if hook is None:
            return
        result = hook(*args)
        if inspect.isawaitable(result):
            await result


# ── agent builders ─────────────────────────────────────────────────────────


def _env() -> Any:
    return types.SimpleNamespace()


def build_agent(
    cls: type[Agent] = Agent,
    *,
    name: str = "test-agent",
    websockets: list[Any] | None = None,
    conn: sqlite3.Connection | None = None,
    env: Any = None,
) -> Agent:
    ctx = FakeCtx(name=name, websockets=websockets, conn=conn)
    agent = cls(ctx, env if env is not None else _env())
    if isinstance(agent, Agent):
        agent._run_schema_migration()
        agent._hydrate_state()
    return agent


ChatReplyFn = Callable[[ChatOptions], Any]


def build_chat_agent(
    reply: ChatReplyFn | None = None,
    *,
    cls: type[AIChatAgent] | None = None,
    name: str = "test-agent",
    conn: sqlite3.Connection | None = None,
    env: Any = None,
) -> AIChatAgent:
    if cls is None:

        class _Chat(AIChatAgent):
            async def on_chat_message(self, options: ChatOptions) -> Any:
                if reply is None:
                    return ""
                result = reply(options)
                if inspect.isawaitable(result):
                    result = await result
                return result

        cls = _Chat

    ctx = FakeCtx(name=name, conn=conn)
    agent = cls(ctx, env if env is not None else _env())
    agent._run_schema_migration()
    agent._hydrate_state()
    agent._prepare_chat_storage()
    return agent


# ── waitUntil ───────────────────────────────────────────────────────────────


@dataclass
class _WaitUntilItem:
    awaitable: Any
    owner: str | None
    context: str | None
    status: Literal["pending", "running", "completed", "failed", "cancelled"] = (
        "pending"
    )
    result: Any = None
    error: BaseException | None = None
    runner: asyncio.Task[Any] | None = None

    def record(self) -> dict[str, Any]:
        return {
            "owner": self.owner,
            "context": self.context,
            "status": self.status,
            "result": self.result,
            "error": self.error,
        }


class WaitUntilRecorder:
    """Captures coroutines handed to workers.waitUntil so a test drives them.

    Never detaches a task: undrained coroutines are closed on teardown, which
    keeps a background fiber from running unless a test explicitly drains it.
    Owner metadata is configured on the recorder so the runtime call remains a
    one-argument `waitUntil(coro)`.
    """

    def __init__(
        self,
        *,
        owner: str | None = None,
        context: str | None = None,
    ) -> None:
        self.owner = owner
        self.context = context
        self._items: list[_WaitUntilItem] = []

    @property
    def coros(self) -> tuple[Any, ...]:
        return tuple(item.awaitable for item in self._items if item.status == "pending")

    @property
    def records(self) -> list[dict[str, Any]]:
        return [item.record() for item in self._items]

    def __call__(self, coro: Any) -> Any:
        self._items.append(
            _WaitUntilItem(
                awaitable=coro,
                owner=self.owner,
                context=self.context,
            )
        )
        return coro

    async def drain_next(self) -> Any:
        item = next(item for item in self._items if item.status == "pending")
        item.status = "running"
        item.runner = asyncio.current_task()
        try:
            result = await item.awaitable
        except asyncio.CancelledError as error:
            item.status = "cancelled"
            item.error = error
            raise
        except BaseException as error:
            item.status = "failed"
            item.error = error
            raise
        finally:
            item.runner = None
        item.status = "completed"
        item.result = result
        return result

    async def drain(self) -> None:
        while self.coros:
            await self.drain_next()

    def cancel_all(self) -> None:
        for item in self._items:
            if item.status == "pending":
                self._cancel_awaitable(item.awaitable)
                item.status = "cancelled"
            elif item.status == "running" and item.runner is not None:
                item.runner.cancel()

    @staticmethod
    def _cancel_awaitable(awaitable: Any) -> None:
        cancel = getattr(awaitable, "cancel", None)
        if cancel is not None:
            cancel()
            return
        close = getattr(awaitable, "close", None)
        if close is not None:
            close()

    def close(self) -> None:
        self.cancel_all()
