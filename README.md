# Cloudflare Agents SDK for Python

This Python implementation of the Cloudflare Agents SDK lets you build stateful
Agents in Python Workers and connect to them with the existing TypeScript and
React clients. The examples below use `agents` 0.22.0 and
`@cloudflare/ai-chat` 0.10.1.

```python
# src/entry.py
from agents import Agent, rpc_callable, route_agent_request
from workers import Response, WorkerEntrypoint


class Counter(Agent):
    def initial_state(self):
        return {"count": 0}

    @rpc_callable()
    def increment(self, amount: int = 1):
        state = self.state
        state["count"] += amount
        self.set_state(state)
        return state["count"]


class Default(WorkerEntrypoint):
    async def fetch(self, request):
        return await route_agent_request(request, self.env) or Response(
            "Not Found", status=404
        )
```

Use the same Agent from a TypeScript client:

```tsx
import { useState } from "react";
import { useAgent } from "agents/react";

type CounterState = { count: number };

export function CounterButton() {
  const [error, setError] = useState<string | null>(null);
  const agent = useAgent<CounterState>({
    agent: "Counter",
    name: "main"
  });

  async function increment() {
    try {
      setError(null);
      await agent.stub.increment(5);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Increment failed");
    }
  }

  return (
    <>
      <button onClick={() => void increment()}>
        Count: {agent.state?.count ?? 0}
      </button>
      {error && <p role="alert">{error}</p>}
    </>
  );
}
```

The `Counter` class is available at `/agents/counter/main`. Class names are
converted to kebab-case on both sides, so `DungeonMaster` becomes
`dungeon-master`.

## Install

Python Workers use a built wheel. From this repository, build it with Python
3.12 and `uv`:

```bash
uv build
```

Add the wheel to the Worker project's `pyproject.toml`:

```toml
[project]
requires-python = ">=3.12"
dependencies = [
    "agents-py @ file:///absolute/path/to/agents-py/dist/agents_py-0.1.0-py3-none-any.whl",
]
```

Install the Python Worker tooling and resolve the environment:

```bash
uv add --dev workers-py
uv sync
```

Install the browser client in the frontend project:

```bash
npm install agents@0.22.0 react@19 react-dom@19
```

For chat applications, also install the AI Chat client:

```bash
npm install @cloudflare/ai-chat@0.10.1 @ai-sdk/react@3 ai@6 zod@4
```

## Configure

Register each top-level Agent class as a SQLite Durable Object:

```jsonc
{
  "$schema": "node_modules/wrangler/config-schema.json",
  "name": "python-agent",
  "main": "src/entry.py",
  "compatibility_date": "2026-07-20",
  "compatibility_flags": ["python_workers"],
  "durable_objects": {
    "bindings": [{ "class_name": "Counter", "name": "COUNTER" }]
  },
  "migrations": [{ "tag": "v1", "new_sqlite_classes": ["Counter"] }]
}
```

The binding name is the screaming-snake form of the class name: `Counter`
becomes `COUNTER`, while `DungeonMaster` becomes `DUNGEON_MASTER`.

If the Worker also serves a single-page application, route Agent requests to the
Worker before the asset fallback:

```jsonc
{
  "assets": {
    "directory": "./frontend/build/client",
    "not_found_handling": "single-page-application",
    "run_worker_first": ["/agents/*"]
  }
}
```

Without `run_worker_first`, the SPA can answer WebSocket upgrades and chat
history requests with `index.html`.

`route_agent_request()` performs routing, not authentication. For cross-origin
HTTP, let credentialless `OPTIONS` requests reach the router with an explicit
CORS header dictionary before authenticating other methods. For WebSocket
upgrades, validate both credentials and `Origin` before routing. `cors=True`
allows every HTTP origin and does not protect WebSocket upgrades.

## Run and deploy

Run these commands from the Worker project:

```bash
uv run pywrangler dev
uv run pywrangler deploy
```

Agent instances are addressed as:

```text
/agents/{kebab-case-class-name}/{instance-name}
```

`route_agent_request()` returns `None` when a path does not address an Agent, so
the rest of the Worker can handle that request.

## State

Agent state is a JSON dictionary stored in the Durable Object's SQLite database
and synchronized with every connected client.

```python
class Room(Agent):
    def initial_state(self):
        return {"players": [], "phase": "lobby"}

    @rpc_callable()
    def join(self, player: str):
        state = self.state
        state["players"] = [*state["players"], player]
        self.set_state(state)
        return state
```

Always call `set_state()` after changing state. Mutating the value returned by
`self.state` does not persist or broadcast the update by itself. State values
must be JSON-compatible dictionaries with finite numbers.

A state update sent by a client is broadcast to the other clients. The sender
already applied its update optimistically, so it does not receive an echo.

## Callable methods

Decorate a method with `@rpc_callable()` to expose it through
`agent.stub.<method>()`. Synchronous and asynchronous methods are supported, and
arguments and return values cross the connection as JSON.

```python
@rpc_callable()
async def summarize(self, document_id: str) -> str:
    rows = self.sql("SELECT body FROM documents WHERE id = ?", document_id)
    if not rows:
        return ""
    return rows[0]["body"][:200]
```

The decorator adds RPC metadata without replacing the method, so it remains a
normal Python method when called from server code.

### Streaming RPC

A streaming RPC receives a `StreamingResponse` before the client's arguments:

```python
from agents import StreamingResponse, rpc_callable


@rpc_callable(streaming=True)
async def tail(self, stream: StreamingResponse, limit: int):
    limit = max(1, min(int(limit), 100))
    for row in self.sql("SELECT line FROM logs ORDER BY id DESC LIMIT ?", limit):
        stream.send(row["line"])
    stream.end()
```

Call `stream.end()` or `stream.error("reason")` on every handled path. The
client's promise settles when it receives that terminal frame.

## SQL

Use `self.sql(query, *params)` for application data. It returns a list of
dictionaries and binds parameters positionally.

```python
class Documents(Agent):
    async def on_start(self):
        self.sql(
            "CREATE TABLE IF NOT EXISTS documents ("
            "id TEXT PRIMARY KEY, body TEXT NOT NULL)"
        )

    @rpc_callable()
    def save(self, document_id: str, body: str):
        self.sql(
            "INSERT OR REPLACE INTO documents (id, body) VALUES (?, ?)",
            document_id,
            body,
        )
```

Create application tables in `on_start()`, which runs before the Agent serves
requests. Keep `__init__()` limited to in-memory setup. Do not use table names
beginning with `cf_agents_`, `cf_ai_chat_`, or `cf_agent_tool_`; those prefixes
belong to the SDK.

## Chat Agents

Subclass `AIChatAgent` and implement `on_chat_message()`. Return a string or
yield strings to stream a response. Persisted conversation history is available
as `self.messages`.

```python
from agents import AIChatAgent, ChatOptions


class Assistant(AIChatAgent):
    max_persisted_messages = 200

    async def on_chat_message(self, options: ChatOptions):
        if options.aborted:
            return

        yield "Hello"
        yield " from Python."
```

`ChatOptions` provides `request_id`, `trigger`, `body`, `abort`, and `aborted`.
Yielded strings become text deltas; yielded dictionaries pass through as raw UI
message chunks for tools and custom parts.

Connect with `useAgentChat`:

```tsx
import { useAgent } from "agents/react";
import { useAgentChat } from "@cloudflare/ai-chat/react";

function Chat() {
  const agent = useAgent({ agent: "Assistant", name: "main" });
  const {
    messages,
    sendMessage,
    clearHistory,
    status,
    error,
    connectionError
  } = useAgentChat({ agent });

  return (
    <form
      onSubmit={async (event) => {
        event.preventDefault();
        try {
          await sendMessage({ text: "Hello" });
        } catch (cause) {
          console.error("Could not send message", cause);
        }
      }}
    >
      <button type="submit" disabled={status !== "ready"}>
        Send
      </button>
      <button type="button" onClick={() => clearHistory()}>
        Clear
      </button>
      <pre>{JSON.stringify(messages, null, 2)}</pre>
      {(error || connectionError) && (
        <p role="alert">{(error || connectionError)?.message}</p>
      )}
    </form>
  );
}
```

Chat chunks are stored before they are broadcast. If the connection drops while
an answer is streaming, the client reconnects, replays the stored prefix, and
continues with the live response without application code.

## Scheduling

Use the Scheduler for callbacks that must run at a particular time or interval.
Decorated callbacks receive the stored payload and `Schedule` record.

```python
from agents import Agent, rpc_callable
from agents.schedules import Schedule, scheduler_callback


class Reminders(Agent):
    @scheduler_callback()
    async def deliver(self, payload: object, schedule: Schedule):
        self.broadcast_json(
            {
                "type": "reminder",
                "scheduleId": schedule.id,
                "payload": payload,
            }
        )

    @rpc_callable()
    async def remind_in_one_minute(self, message: str) -> str:
        schedule = await self.schedule(60, "deliver", {"message": message})
        return schedule.id
```

The first argument to `schedule()` can be a delay in seconds, a `datetime`, or a
cron expression. Naive datetimes are interpreted as UTC. Cron expressions use
five or six fields and run in UTC. Use `schedule_every()` for recurring
intervals:

```python
schedule = await self.schedule_every(300, "deliver", {"type": "heartbeat"})
await self.cancel_schedule(schedule.id)
```

Other inspection methods are `get_schedule_by_id()` and `list_schedules()`.

## Replayable Tasks

Tasks persist each run and step so completed steps are reused when execution
restarts. Put side effects behind `step.do()` and give every step a stable name.

```python
from agents import Agent, rpc_callable
from agents.tasks import TaskStep, task_definition


class Reports(Agent):
    @task_definition()
    async def build_report(self, input: dict[str, int], step: TaskStep):
        @step.do("calculate")
        async def calculate():
            return input["value"] * 2

        result = await calculate()
        return {"result": result}

    @rpc_callable()
    async def start_report(self, job_id: str, value: int) -> str:
        receipt = await self.tasks.run(
            "build_report",
            {"value": value},
            idempotency_key=f"report:{job_id}",
        )
        return receipt.run_id
```

`tasks.run()` returns after durable acceptance. Inspect a run with
`self.tasks.get(run_id)`, cancel it with `self.tasks.cancel(run_id)`, or list runs
with `self.tasks.list()`.

Steps can retry, time out, sleep, and expose an idempotency key for external
operations:

```python
from agents.tasks import TaskStepAttempt, TaskStepConfig, TaskStepRetryOptions


async def call_service(attempt: TaskStepAttempt):
    attempt.signal.throw_if_aborted()
    return {
        "requestId": attempt.idempotency_key,
        "status": "submitted",
    }


result = await step.do(
    "call-service",
    TaskStepConfig(
        retries=TaskStepRetryOptions(
            limit=3,
            delay="1 second",
            backoff="exponential",
        ),
        timeout="30 seconds",
    ),
    call_service,
)

await step.sleep("wait-for-indexing", "10 seconds")
```

Task inputs, metadata, step results, and final results must be strict JSON values.
Use Tasks for replayable orchestration; use Scheduler when the primary concern is
when a callback should run.

## Sessions

`AIChatAgent` installs `Sessions` automatically and uses the default session for
its canonical message history. Named sessions are useful for side conversations,
audit trails, and application-specific message trees.

```python
from agents import AIChatAgent


class Assistant(AIChatAgent):
    async def on_chat_message(self, options):
        audit = self.sessions.session("audit")
        await audit.append_message(
            {
                "id": options.request_id,
                "role": "user",
                "parts": [{"type": "text", "text": "Turn accepted"}],
            }
        )
        return "Accepted"
```

A session stores branches rather than only a flat list. Omitting `parent_id`
appends to the latest leaf, `parent_id=None` starts a new root, and an explicit
message ID creates a branch from that parent.

```python
session = self.sessions.session("research")
history = await session.get_history()
latest = await session.get_latest_leaf()
matches = await session.search("deployment notes")
await session.delete_messages(["message-id"])
await session.clear_messages()
```

Use `history_batches()` for large histories, `search()` for message text, and
`on_compaction()` with `compact()` to summarize older history while keeping the
stored messages intact.

## Context

`ContextBlocks` combines named context providers into a system prompt and can
shape older messages without changing stored Session history.

```python
from agents import AIChatAgent
from agents.context import AgentContextProvider, ContextBlocks, ContextConfig


class Assistant(AIChatAgent):
    async def on_start(self):
        self.context = ContextBlocks(
            [
                ContextConfig(
                    label="memory",
                    description="Facts remembered about the user",
                    max_tokens=1_000,
                    provider=AgentContextProvider(self),
                )
            ],
            prompt_store=AgentContextProvider(self, "system-prompt"),
        )
        await self.context.load()

    async def on_chat_message(self, options):
        model_input = await self.context.assemble(self.messages)
        return await call_your_model(model_input)
```

Use `await self.context.set_block(...)` or
`await self.context.append_to_block(...)` to update writable context. Use
`AgentSearchProvider` for searchable key/value context, and
`await self.context.tools()` to expose `set_context` and `search_context` tools
based on provider capabilities.
`assemble()` returns a system prompt and a shaped message list to pass to your
model provider; it does not modify the stored Session history.

## Sub-agents

Sub-agents are named child Agents with isolated SQLite storage. The child class
must be exported from the Worker entry module, but it does not need its own
Durable Object binding or migration.

```python
from agents import Agent, rpc_callable


class ProjectItem(Agent):
    async def on_start(self):
        self.sql(
            "CREATE TABLE IF NOT EXISTS metadata ("
            "key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )

    def set_title(self, title: str):
        self.sql(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES ('title', ?)",
            title,
        )
        return title


class Project(Agent):
    @rpc_callable()
    async def add_item(self, name: str, title: str):
        item = await self.sub_agent(ProjectItem, name)
        return await item.set_title(title)
```

Only `Project` is added to `durable_objects.bindings` and
`new_sqlite_classes`. Use `has_sub_agent()` and `list_sub_agents()` for
inspection, `abort_sub_agent()` to stop a child while preserving storage, and
`delete_sub_agent()` to destroy it.

Only a top-level Agent can delete sub-agents. A child's `on_start()` must not
call its waiting parent, because the parent is waiting for child startup to
finish.

Nested HTTP and WebSocket routes add one `/sub/{class}/{name}` hop per child:

```text
/agents/project/roadmap/sub/project-item/design/status
```

Override `on_before_sub_agent()` to authorize each HTTP or WebSocket hop before
the child wakes.

## Agent Tools

`run_agent_tool()` delegates an awaited unit of work to a chat-capable child
Agent. A stable `run_id` makes retries join or repair the same durable execution.

```python
from agents import AIChatAgent, Agent, rpc_callable


class ResearchAgent(AIChatAgent):
    async def on_chat_message(self, options):
        question = options.body["agentToolInput"]
        return f"Research result for: {question}"


class Coordinator(Agent):
    @rpc_callable()
    async def investigate(self, job_id: str, question: str):
        result = await self.run_agent_tool(
            ResearchAgent,
            input=question,
            run_id=f"research:{job_id}",
            parent_tool_call_id=job_id,
        )
        if result.status != "completed":
            raise RuntimeError(result.error or result.status)
        return result.output
```

The child class must be exported, but it needs no binding or migration. Pass
`input_preview=None` when the input should not be shown in parent connection
events. Override `max_concurrent_agent_tools` on the parent to change the default
concurrency limit of four.

## Connections and lifecycle

Use lifecycle hooks to initialize resources and handle application traffic:

| Hook | Called when |
| --- | --- |
| `on_start()` | The Agent activation is ready for application setup |
| `on_connect(connection, context)` | A WebSocket connection opens |
| `on_message(connection, message)` | A frame is not claimed by the Agent protocol |
| `on_close(connection, code, reason, was_clean)` | A connection closes |
| `on_request(request)` | The Agent receives an ordinary HTTP request |
| `on_error(error, connection=None)` | A hook or runtime operation fails |
| `on_before_sub_agent(request, child)` | A request is about to enter a child Agent |

Hooks may be synchronous or asynchronous. Framework protocol handling and
cleanup run before the public hook, so overrides do not need to call `super()`.

Broadcast an application frame to every open connection:

```python
self.broadcast_json({"type": "notice", "text": "Build finished"})
```

Use `get_connection(id)` and `get_connections(tag=None)` to inspect connections.
Override `get_connection_tags()` to assign tags during connection setup.

## API map

| Goal | Import or API |
| --- | --- |
| [Define an Agent](#state) | `from agents import Agent` |
| [Define a chat Agent](#chat-agents) | `from agents import AIChatAgent` |
| [Route requests](#configure) | `route_agent_request(request, env, cors=...)` |
| [Expose RPC](#callable-methods) | `@rpc_callable()` |
| [Stream RPC results](#streaming-rpc) | `StreamingResponse` and `@rpc_callable(streaming=True)` |
| [Read and update state](#state) | `self.state`, `self.set_state(...)` |
| [Query SQLite](#sql) | `self.sql(query, *params)` |
| [Schedule callbacks](#scheduling) | `agents.schedules`, `self.schedule(...)`, `self.scheduler` |
| [Run replayable work](#replayable-tasks) | `agents.tasks`, `@task_definition()`, `self.tasks` |
| [Store message trees](#sessions) | `agents.sessions`, `self.sessions` on `AIChatAgent` |
| [Assemble model context](#context) | `agents.context.ContextBlocks` |
| [Create child Agents](#sub-agents) | `self.sub_agent(...)` |
| [Delegate to an Agent Tool](#agent-tools) | `self.run_agent_tool(...)` |
| [Broadcast to clients](#connections-and-lifecycle) | `self.broadcast(...)`, `self.broadcast_json(...)` |
| [Inspect connections](#connections-and-lifecycle) | `self.get_connection(...)`, `self.get_connections(...)` |

## Troubleshooting

**The Worker still uses an older SDK build.** Rebuild the wheel, run
`uv sync --reinstall-package agents-py`, remove `.venv-workers`, then restart
`pywrangler`. A running development process does not hot-reload a vendored wheel.

**Routing raises `no namespace found`.** Check that the class has a matching
binding and that the binding uses the screaming-snake class name. A missing
SQLite migration normally causes deployment to fail.

**A normal GET to an Agent returns 404.** This is the default until the Agent
implements `on_request()`. The WebSocket client can still connect at the same
Agent route.

**A WebSocket request returns the frontend HTML.** Add `/agents/*` to
`assets.run_worker_first` so the Worker sees Agent routes before the SPA fallback.

**State changes do not reach clients.** State must be a dictionary, and every
server-side mutation must finish with `set_state()`.

**A streaming RPC never resolves.** End every successful stream with `end()` and
every handled failure with `error("reason")`.

**Construction fails after a wake.** Keep application setup and table creation
in `on_start()` rather than `__init__()`.

**A sub-agent cannot start.** Export the child class from the Worker entry
module. Configure a binding and migration for the root Agent only.

## Client documentation

- [Cloudflare Agents documentation](https://developers.cloudflare.com/agents/)
- [Agents client API](https://developers.cloudflare.com/agents/api-reference/client-sdk/)
- [Chat Agents](https://developers.cloudflare.com/agents/api-reference/chat-agents/)
- [Callable methods](https://developers.cloudflare.com/agents/api-reference/callable-methods/)
