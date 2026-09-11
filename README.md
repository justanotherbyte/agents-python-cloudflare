# agents-py

A server-side Python port of Cloudflare's Agents SDK, for Python Workers.

An agent is a Durable Object with a WebSocket protocol on top: persistent state
that syncs to every connected browser, methods the client can call directly, chat
turns that survive a refresh, and durable execution that survives eviction.

The one constraint that shapes everything: this speaks to the **unmodified
TypeScript client**. `agents/react`, `useAgent`, `useAgentChat` and `agent.stub.*`
all work against a Python Worker with no client changes, because the JSON wire
format is reproduced exactly.

```python
from agents import Agent, rpc_callable, route_agent_request
from workers import Response, WorkerEntrypoint


class Counter(Agent):
    def initial_state(self):
        return {"count": 0}

    @rpc_callable()
    def increment(self, n: int = 1):
        state = self.state
        state["count"] += n
        self.set_state(state)
        return state["count"]


class Default(WorkerEntrypoint):
    async def fetch(self, request):
        return await route_agent_request(request, self.env, cors=True) or Response(
            "Not Found", status=404
        )
```

`route_agent_request` returns `None` for a path that addresses no agent, so the
rest of your Worker can handle it.

```tsx
const agent = useAgent({ agent: "counter", name: "main" });
await agent.stub.increment(5); // every connected client re-renders
```

Note the name mapping: the class `Counter` is addressed as `counter`, and
`DungeonMaster` as `dungeon-master`. The client derives the same kebab-case from
the class name, so the two always agree.

## Status

Working today: HTTP and WebSocket routing with hibernation, the identity frame,
bidirectional state sync, RPC including streaming, chat turns with persistence
and resumable streaming, durable fibers with a recovery scan, cross-runtime schema
versioning, sub-agents, awaited agent tools, and persistent scheduling through
`Agent.schedule()` or the experimental `agents.schedules.Scheduler` capability.
Facet schedules live in the root job queue while callbacks execute on their
owning facet.

Not implemented: task queues, MCP, workflows, detached agent tools,
email, and observability. `design/PORT_TODO.md` tracks all of it feature by feature —
check there before reaching for something.

## Install

There is no PyPI release. Build the wheel and depend on it by path:

```bash
cd agents-py && uv build
```

```toml
# pyproject.toml
dependencies = [
    "agents-py @ file:///abs/path/to/agents-py/dist/agents_py-0.1.0-py3-none-any.whl",
]
```

Because it is a built wheel rather than an editable install, an edit to the SDK
does not reach your Worker until you rebuild and clear the resolved environment:

```bash
uv build                            # in agents-py
rm -rf .venv .venv-workers pylock.toml uv.lock   # in your Worker
```

`pywrangler dev` does **not** hot-reload the vendored wheel. Restart it after a
rebuild or you will be testing the old code.

## Configuration

```jsonc
{
  "main": "src/entry.py",
  "compatibility_flags": ["python_workers"],
  "durable_objects": {
    "bindings": [{ "class_name": "Counter", "name": "COUNTER" }]
  },
  "migrations": [{ "tag": "v1", "new_sqlite_classes": ["Counter"] }],
  "assets": {
    "directory": "./frontend/build/client",
    "not_found_handling": "single-page-application",
    // Without this, the SPA fallback answers the WebSocket upgrade with index.html.
    "run_worker_first": ["/agents/*"]
  }
}
```

Every agent class needs a binding and a `new_sqlite_classes` migration — except a
sub-agent class, which needs neither.

## State

`state` is a dict persisted to SQLite and mirrored to every connected client.
`set_state` writes and broadcasts in one step; `initial_state` supplies the
starting value on first use.

```python
class Room(Agent):
    def initial_state(self):
        return {"players": [], "phase": "lobby"}

    @rpc_callable()
    def join(self, who: str):
        state = self.state
        state["players"] = [*state["players"], who]
        self.set_state(state)
```

Read-modify-write the whole dict as above. `self.state` hands back the stored
value, so mutating it in place without calling `set_state` persists nothing and
tells no one.

A client's own `setState` is applied optimistically before it reaches the server,
so the update is broadcast to everyone *except* its sender.

## RPC

`@rpc_callable()` exposes a method to `agent.stub.<name>()`. Arguments and return
values cross as JSON. Sync and async methods both work.

```python
@rpc_callable()
async def summarise(self, doc_id: str) -> str:
    rows = self.sql("SELECT body FROM docs WHERE id = ?", doc_id)
    return rows[0]["body"][:200]
```

For a long answer, take a stream and push to it. A streaming method receives a
`StreamingResponse` as its first argument, before the client's own arguments:

```python
@rpc_callable(streaming=True)
async def tail(self, stream, n: int):
    for row in self.sql("SELECT line FROM log LIMIT ?", n):
        stream.send(row["line"])
    stream.end()
```

The stream owns the call's terminal frame: `end()` (or `error()`) must be reached
on every path, because the client's promise settles only when a terminal arrives.

The decorator only attaches immutable discovery metadata. The method remains an
ordinary Python method and can be called directly from your own code.

## SQL

`self.sql(query, *params)` returns a list of dicts. Parameters are bound, never
interpolated.

```python
self.sql("CREATE TABLE IF NOT EXISTS docs (id TEXT PRIMARY KEY, body TEXT)")
self.sql("INSERT INTO docs VALUES (?, ?)", doc_id, body)
rows = self.sql("SELECT body FROM docs WHERE id = ?", doc_id)
```

Tables prefixed `cf_agents_`, `cf_ai_chat_`, and `cf_agent_tool_` belong to the
framework. Create your own in `on_start`.

## Chat

`AIChatAgent` handles the chat protocol; you implement one method. Return a
string, or yield to stream. Message history is persisted and available as
`self.messages`.

```python
from agents import AIChatAgent


class Assistant(AIChatAgent):
    async def on_chat_message(self, options):
        for message in history_from(self.messages):
            if options.aborted:  # the user pressed stop
                return
            yield message
```

`options` carries `request_id`, `trigger`, the raw `body`, and `aborted`.
Yielding a `str` becomes a text delta; yielding a dict is passed through as a raw
UI message chunk, which is how tool calls are rendered. Set
`max_persisted_messages` on the class to cap rows kept in SQLite.

A client that drops mid-answer reconnects and picks the same turn back up — the
chunks are buffered in SQLite, replayed on reconnect, then joined to the live
stream. That works with no code from you.

## Fibers

A fiber is durable execution: checkpointed work that resumes after the Durable
Object is evicted mid-flight.

```python
from agents import FiberRecoveryResult


class Job(Agent):
    @rpc_callable()
    async def start(self):
        result = await self.start_fiber(
            "import", self._import, wait_for_completion=True
        )
        return result.fiber_id

    async def _import(self, ctx):
        for i, batch in enumerate(batches()):
            await push(batch)
            ctx.stash({"done": i})  # checkpoint

    async def on_fiber_recovered(self, ctx):
        done = (ctx.snapshot or {}).get("done", 0)
        for i, batch in enumerate(batches()):
            if i > done:
                await push(batch)
        return FiberRecoveryResult(status="completed")
```

If the object dies mid-run, the next activation's recovery scan hands your last
checkpoint to `on_fiber_recovered`, which resumes rather than restarting.
`inspect_fiber`, `list_fibers`, `cancel_fiber` and `delete_fibers` read and drive
the ledger. A recovery result must use one of `completed`, `error`, `aborted`, or
`interrupted`; other values are rejected before they can corrupt the ledger.

Fibers currently run only on awaited paths, so pass
`wait_for_completion=True` — a detached background fiber escapes the Durable
Object's I/O context and is disabled by default pending deployed S1 evidence.
The isolated runtime probe opts in through `detached_fibers_enabled`; application
agents should not enable it until that probe passes for the deployed runtime.

## Sub-agents

An agent can spawn named children that run on its own machine, each with its own
isolated SQLite. A child class needs **no binding and no migration** — it only has
to be exported from the entry module.

```python
class Project(Agent):
    @rpc_callable()
    async def add_task(self, name: str, title: str):
        task = await self.sub_agent(Task, name)  # gets or creates
        return await task.set_title(title)


class Task(Agent):
    def set_title(self, title: str):  # plain method, not @rpc_callable
        self.sql("INSERT OR REPLACE INTO meta VALUES ('title', ?)", title)
        return {"title": title}
```

Methods reached through a sub-agent stub remain ordinary callable methods.
`@rpc_callable` attaches protocol metadata without replacing the function, so a
method may be exposed to both a sub-agent stub and the Agent protocol.

Children are reachable over HTTP by walking the path, one `/sub/{class}/{name}`
hop per level:

```
/agents/project/roadmap/sub/task/design/status
```

`on_before_sub_agent(request, child)` gates each hop. Return `None` to forward, or
a `Response` to answer without waking the child. It runs only in `fetch`, so it
gates HTTP and not RPC, and each hop is checked by its own parent — a nested child
needs the hook on the intermediate class too.

```python
async def on_before_sub_agent(self, request, child):
    if not allowed(child["name"]):
        return Response("Forbidden", status=403)
    return None
```

Lifecycle: `abort_sub_agent` stops a child and keeps its storage;
`await delete_sub_agent(...)` destroys it, transitively. `has_sub_agent` and
`list_sub_agents` read the parent's registry, and `parent_agent(cls)` reaches back
up one level.

Two limits to design around:

- **`delete_sub_agent` only works from the top-level agent.** The runtime refuses
  to let a sub-agent destroy its own children. Use `abort_sub_agent` at depth, or
  delete an ancestor — deletion is transitive.
- **Child WebSockets are buffered through the root.** The root owns the physical
  socket and forwards each event to the addressed child, including nested paths.
  A handler must finish within 30 seconds and return at most 1,000 frames or 1 MiB
  per event. Live cross-facet token delivery remains disabled pending deployed
  runtime evidence. Buffered operations are collected only during the addressed
  connection's current event, so broadcasts cannot reach an idle child socket.

A child's `on_start` must not call back into its parent: the parent is awaiting the
child's startup from inside `blockConcurrencyWhile`, so it deadlocks.

## Agent tools

`run_agent_tool` delegates one awaited task to a chat-capable child. The child
class is the first argument; each `run_id` names its durable sub-agent, so retrying
the same ID returns or repairs the same execution rather than running it twice.

```python
class ResearchAgent(AIChatAgent):
    async def on_chat_message(self, options):
        question = options.body["agentToolInput"]
        return await research(question)


class Coordinator(Agent):
    async def investigate(self, question: str):
        result = await self.run_agent_tool(
            ResearchAgent,
            input=question,
            run_id=f"research:{stable_id(question)}",
            parent_tool_call_id="tool-call-1",
        )
        if result.status != "completed":
            raise RuntimeError(result.error or result.status)
        return result.output
```

The result mirrors the TypeScript SDK's awaited shape: `run_id`, `agent_type`,
`status`, `output`, `summary`, `error`, `reason`, and `child_still_running`.
`format_agent_tool_input`, `get_agent_tool_output`, and
`get_agent_tool_summary` can be overridden on the child; by default the input is
stored as a user message and the final assistant text becomes the output and
summary.

By default, a string input or serialized value is truncated to 500 characters
and sent as `inputPreview` to every parent connection. Pass `input_preview=None`
for sensitive inputs. `display` supplies optional UI metadata and
`display_order` defaults to `0`. An `abort` event is checked before and after the
awaited child RPC; reliable mid-flight cross-facet cancellation is not enabled.

At most four agent tools run concurrently by default. Override
`max_concurrent_agent_tools` on the parent to change that limit; rejection
returns an `error` result rather than raising. Soft runs become repairable
`interrupted` rows after `agent_tool_recovery_grace_ms`, which defaults to five
minutes.

The parent persists every `agent-tool-event` before broadcasting it and replays
the same sequence on reconnect. The current implementation awaits the child's
active RPC and then forwards its buffered chunks as one ordered batch. Live
cross-facet tailing, mid-flight cancellation, progress milestones, and detached
runs remain gated on deployed runtime probes.

## Durable chat recovery

Set `durable_chat_recovery = True` on an `AIChatAgent` to persist immutable turn
context in a managed awaited fiber. If the object is evicted mid-turn, startup
recovery folds a bounded prefix of stored stream chunks into a partial assistant
message, emits the normal terminal frame, and cleans up the internal fiber. It
never calls `on_chat_message` or the model during startup, so recovery cannot
duplicate provider work. The feature is off by default; ordinary orphan streams
still replay and settle without being added to message history.

## Lifecycle hooks

| Hook | When |
|---|---|
| `on_start` | Once per activation, before anything else is served |
| `on_connect(connection, ctx)` | A client socket opened |
| `on_message(connection, message)` | A frame the protocol did not claim |
| `on_close(connection, code, reason, was_clean)` | A socket closed |
| `on_request(request)` | A plain HTTP request to this agent |
| `on_error(error, connection=None)` | Any hook or plumbing failure |
| `on_before_sub_agent(request, child)` | A request is about to hop into a child |

`on_start` is the place for schema setup. It runs inside the init barrier, so it
completes before any request is served.

Lifecycle hooks may be synchronous or asynchronous. Framework handshakes,
internal routes, and cleanup run in protected dispatch methods before the public
hook, so an override does not call `super()` to preserve SDK behavior. Methods
whose names begin with `_dispatch_` are internal and should not be overridden.

Broadcasting outside a state update:

```python
self.broadcast_json({"type": "toast", "text": "done"})  # optional exclude=[id]
```

## Gotchas

- **Nothing in `__init__` may raise.** The runtime re-runs the constructor on
  every wake with no way in to repair, so a raise there bricks the object
  permanently. Do setup in `on_start`.
- **Restart `pywrangler dev` after rebuilding the wheel.** It will not pick up SDK
  changes on its own.
- **State must be a JSON object.** A scalar poisons every later read, including
  the one in the connection handshake.
- Deployed behaviour differs from local in one place that matters: `getWebSockets()`
  returns `[]` under miniflare, so socket-recovery bugs are invisible until
  deployed.

## Further reading

- `design/PORT_TODO.md` — per-feature parity against the reference, and what is
  missing.
- `design/PROTOCOL.md` — the wire protocol, frame by frame.
- `AGENTS.md` — internal contract and house rules. Read this before changing the
  SDK itself.
- `design/PORTING_FIBERS.md` — the durable-execution design in full.
