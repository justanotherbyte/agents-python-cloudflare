# AGENTS.md

This repository is a server-side Python port of Cloudflare's Agents SDK for
Python Workers. It speaks to the unmodified TypeScript client and can reopen
shared Durable Object storage written by the TypeScript SDK. Wire bytes,
key-presence semantics, durable table shapes, and migration behavior are hard
compatibility constraints; the internal Python architecture is not.

## Read first

- `design/FINAL_PORT.md` is the normative architecture, scope, and delivery
  authority.
- `design/PROTOCOL.md` records observable routing, WebSocket, state, RPC, chat,
  Agent Tool, and sub-agent frames. Historical server-topology descriptions in
  it are not architecture authority.
- `design/PORT_EVIDENCE_LEDGER.md` records what is locally, mixed-runtime, and
  deployment proven. It reports evidence, not desired behavior.
- `design/PORT_BEHAVIOR_MANIFEST.md` classifies pre-port behavior as preserve,
  replace, or retire.
- `README.md` is the user-facing introduction and examples.

When these sources disagree, use `FINAL_PORT.md` for intended behavior and
resource ownership, `PROTOCOL.md` for observable wire behavior, and the evidence
ledger for the strength of the current proof. Do not infer completion from code
existing.

The pinned TypeScript target is commit
`ec93caf6ec1efebb521aa9ab30c0a8cb2b4d50d5`, materialized in the clean detached
worktree `../agents-pinned-ec93caf6`. The moving `../agents` checkout may contain
later behavior that this port deliberately excludes.

Use the pinned target only to resolve behavior, wire, and shared-storage
questions:

- `../agents-pinned-ec93caf6/packages/agents/src/` contains Agent, RPC, state,
  routing, and the client.
- `../agents-pinned-ec93caf6/packages/ai-chat/src/` contains AIChatAgent behavior.

Read the reference, then implement native Python. Production comments must
describe the invariant, not cite TypeScript files or line numbers.

## Current port checkpoint

Checkpoint `cbf3565` implements Stage 1 through Stage 4 and Stage 5 increments
1 through 9. Its validation passed on Python 3.12.13 with 1,017 tests, production
Ruff format and lint plus `ty` with unfinished `agents/harness/` excluded, source
and wheel builds, wheel inspection, and `git diff --check`.

This is an implementation checkpoint, not a release declaration. Many ledger
rows still require an unmodified-client, mixed-runtime, hibernation, or deployed
runtime result before they are Covered.

### Stage 1: behavior lock

- Added deterministic runtime fakes for storage, alarms, readiness, retained
  work, eviction, routing, and resource ownership.
- Characterized routing, CORS, exports, connection policy, socket failures,
  sub-agent identity and deletion, chat tool answers, constructor reachability,
  and fiber keep-alive behavior.
- Recorded preserve, replace, and retire decisions instead of freezing obsolete
  implementation details.

### Stage 2: Lifecycle and storage

- Added the host-neutral `Lifecycle` capability kernel with static registration,
  one-host installation, ordered claims, fallback dispatch, and fresh
  per-instance maps.
- Added shared retryable readiness across requests, upgrades, socket wakes,
  alarms, routes, and native RPC.
- Added bounded storage, SQL, socket, event, route, retained-work, host-context,
  and reverse-disposal services.
- Added guarded, retryable preparation for the shared core schema at version 11.
- Added owner-scoped Lifecycle jobs and the sole physical-alarm driver.
- Added deadman pre-arming, retries, exclusivity, single-flight reclaim,
  generation fencing, tracked handoff, and memory-limit backoff and sealing.

### Stage 3: composition cutover

- Moved hibernating WebSocket ownership, attachments, hydration, lookup,
  callbacks, broadcast, and bounded facet relay into the `WebSockets` capability.
- Preserved identity, state, MCP placeholder, Agent Tool replay, readonly, and
  protocol-suppression handshake behavior through composed dispatch.
- Moved fiber ledgers, recovery, leases, inspection, cancellation, and
  housekeeping into `FiberCapability`.
- Moved Agent Tool execution, publication, replay, repair, joining, and
  cancellation into `AgentToolRuns` plus the chat child adapter.
- Changed `Agent` to inherit directly from `workers.DurableObject`.
- Removed `PartyServer`, `FiberSupport`, `AgentToolSupport`, party routing, and
  their public exports. Do not restore compatibility shims for these symbols.

### Stage 4: durable orchestration

- Added `Scheduler` with delay, date, cron, and interval scheduling, callback
  discovery, retries, events, legacy row migration, and root-owned facet routing.
- Added replayable `Tasks` definitions, handles, snapshots, lookup, pagination,
  cancellation, deletion, and idempotency keys.
- Added durable `TaskStep.do`, receipts, retries, strict JSON results and errors,
  first-write-wins settlement, `sleep`, and `sleep_until`.
- Added authoritative deadlines, exactly one root or routed facet wake mirror,
  generation-fenced claims, duplicate-alarm handling, retention, tracked
  handoff, cancellation races, and memory-limit policy.
- Kept legacy Fiber APIs independently operable; Fibers are compatibility
  execution, not an alias for Tasks.

### Stage 5: conversations through I9

- Added `Sessions` message trees, named and cached handles, branches, streamed
  history, bounded hydration, listeners, updates, splice deletion, and clearing.
- Added UTF-8 continuation rows under a 1.5 MiB row budget.
- Added content-addressed attachment chunks, references, hydration, global
  deduplication, and last-reference cleanup.
- Added lazy FTS5 creation, backfill, scoped search, and mutation maintenance.
- Added non-destructive compaction overlays and reconstructed history.
- Added verified, retryable legacy Sessions lifts that retain source tables
  until every copied destination is proven.
- Added `ContextBlocks` and readonly, writable, searchable, Agent SQL, and search
  providers as prompt-shaping composition outside Sessions.
- Passed the Stage 5/I8 TypeScript -> Python -> TypeScript one-way Sessions
  canary. Authority commits, deployed version IDs, artifact hashes, and outputs
  are in the evidence ledger and
  `../agents-py-testing/sessions-canary/evidence/`. That evidence directory is
  not yet committed and its manifest still needs the testing-Worker source
  commit before it satisfies final pinned-evidence policy.
- Composed `Sessions` into `AIChatAgent`, moved settled message persistence and
  complete `/get-messages` history to it, and added bounded legacy chat import.
- Reorganized implementation files into `agents/core/`, `agents/lifecycle/`, and
  `agents/chat/` without changing the root package exports.

## Package ownership

The package is organized by owner rather than by historical inheritance layer:

- `agents/core/` owns `Agent`, routing, RPC, protocol frames, state, schema,
  sub-agent routing and relay, core Agent Tool runs, response helpers, and wire
  utilities.
- `agents/lifecycle/` owns capability registration and dispatch, readiness,
  services, jobs, alarms, WebSockets, Fibers, routes, retained work, host context,
  and disposal.
- `agents/chat/` owns `AIChatAgent`, turn serialization, resumable streams,
  folding, chat Agent Tools, message normalization, and chat protocol types.
- `agents/schedules.py`, `agents/tasks.py`, and `agents/sessions.py` are narrow
  Lifecycle capabilities with their own public concern-module APIs.
- `agents/context.py` is ordinary prompt-shaping composition over provider and
  SQL seams; it is not a Lifecycle capability.

`Agent` installs fresh Scheduler, Tasks, WebSockets, FiberCapability, and
AgentToolRuns state for each Durable Object instance. `AIChatAgent` additionally
installs Sessions and its chat-owned stream and child-tool stores.

The root `agents` exports are deliberately small and locked by
`tests/pure/test_package_exports.py`. Concern-specific APIs remain in
`agents.schedules`, `agents.tasks`, `agents.sessions`, `agents.context`, and
`agents.lifecycle`. Old flat implementation imports such as `agents.utils` are
gone; use their canonical owner, such as `agents.core.utils`.

## Runtime ownership

There must be one live owner for each physical or durable resource:

- Lifecycle owns startup, capability dispatch, disposal, retained work, jobs,
  and the root physical alarm.
- Root WebSockets owns physical sockets. Facets return bounded ordered relay
  operations through the root rather than adopting parent sockets.
- Scheduler and Tasks own their records but route facet wake intent through the
  root Lifecycle job queue.
- FiberCapability owns the fiber ledger and fiber housekeeping.
- AgentToolRuns owns parent run publication and settlement; the chat adapter owns
  child turn execution and chunk collection.
- Each capability-owned table has one module that owns its DDL, parsing,
  mutation, repair, and migration. Shared core DDL is the deliberate exception:
  `agents.core.schema` coordinates its shape while the owning capability handles
  focused compatibility reconciliation and mutations.

Do not add a second alarm loop, socket registry, startup coordinator, or table
writer. Ownership transfers must be atomic: a reviewable commit cannot have two
owners or no owner.

## Constructor and startup rules

Nothing in `__init__` may perform fallible I/O, call user hooks, or start work.
The runtime reconstructs a Durable Object on every wake, so a deterministic
constructor failure can make the object permanently unreachable.

- Constructors allocate memory and compose capabilities only.
- Fallible SQL preparation runs behind Lifecycle's guarded preparation boundary.
- Capability startup and public `on_start` run only after preparation succeeds.
- Initialization is shared, locked, retryable, and awaited by every runtime
  entry point.
- Discovery reads MRO class dictionaries statically. Never evaluate descriptors
  to discover RPC methods, Scheduler callbacks, or Task definitions.
- User subclasses cannot override reserved runtime entries such as `alarm`,
  `webSocketMessage`, `webSocketClose`, `webSocketError`, or internal route RPC.
  Use the public hooks and capability APIs.

Public hooks may be synchronous or asynchronous. Protected framework dispatch
runs before the public hook, so a user override does not call `super()` to keep
identity, state, chat, routing, or cleanup behavior alive.

## Wire and JSON invariants

The unmodified client often branches on key presence rather than truthiness.
Extra keys, missing keys, and explicit `null` values can change behavior without
raising an error.

- RPC streaming chunks carry `done: false`. RPC errors carry neither `done` nor
  `result`. `StreamingResponse` owns the terminal frame for a streaming RPC.
- Every RPC and chat failure path sends a terminal frame; otherwise the client
  promise remains pending.
- Chat error frames use `error: true` and place the message in `body`.
- `stream_resume_none` echoes `probeId` and uses `reason: "idle"` when idle.
- State is always a JSON object. Reject scalar client state before persistence,
  because it would break future hydration and handshakes.
- Client-originated state updates are broadcast to everyone except the sender,
  whose local state is already optimistic.
- Persisted tool parts use `tool-{name}` and do not set `dynamic` unless the part
  is truly dynamic.
- Persisted cross-runtime JSON uses strict serialization with `allow_nan=False`.
  JavaScript cannot parse Python's default `NaN` or infinity tokens.
- Parse routing URLs with `urlsplit`; `urlparse` strips semicolon parameters from
  the final path segment and can alias two Agent names.

Update `design/PROTOCOL.md` and mixed-runtime fixtures whenever an intentional
wire change is approved. Do not silently make the Python client contract differ.

## Storage and migrations

`agents.core.schema.prepare_core_schema` owns shared core schema version 11. The
marker is `cf_schema_version` in `cf_agents_state`; the old private Python marker
is inert.

- Bootstrap the state table before reading the marker.
- Reconcile the whole current schema and stamp last.
- Missing, malformed, or negative markers behave as version zero.
- Future versions are not downgraded or rewritten.
- A partial or interrupted migration retains its old marker and retries later.
- Scheduler, Tasks, Sessions, streams, and other capability stores own their
  focused DDL and markers outside the shared core stamp.
- SQL is parameterized. Transactions follow durable invariants, not convenient
  function boundaries.
- Validate wire, persisted, JavaScript, and user data at their seams; do not
  repeatedly validate trusted internal structures.

Any shared-storage change needs TypeScript -> Python -> TypeScript reopening
evidence against exact built artifacts. A Python-only round trip is insufficient.

## Lifecycle jobs, Scheduler, and Tasks

Lifecycle jobs use globally unique physical IDs with semantic capability
ownership. Outcomes are fenced by the claimed intent and generation, so an old
completion cannot mutate a replacement or a reclaimed run. Same-owner newer
pushes win.

The root owns the physical alarm. Facet Scheduler callbacks and Task wakes are
represented as owner-keyed root jobs and delivered through addressed Lifecycle
routes. Facets must not set or rearm a physical alarm directly.

Task execution is replayed from durable evidence:

- Agent method definitions are discovered statically and must be explicitly
  decorated. A standalone `Tasks(definitions=...)` registry accepts ordinary
  callbacks supplied in its mapping.
- The two-table journal stores run rows and step rows. Attempts, receipts,
  sleeps, retries, and terminal state are durable fields or projections of those
  rows, not additional tables.
- `TaskStep.do` returns a completed stored result instead of rerunning the step.
- Settlement is first-write-wins and claim-generation fenced.
- `sleep` and `sleep_until` use authoritative Task deadlines plus one wake mirror.
- Cancellation, duplicate alarms, cold repair, terminal cleanup, and memory
  backoff must preserve newer intent and unrelated jobs.

Never use detached `asyncio.create_task` for durable work. Submit retained work
through Lifecycle or model replayable work as a Task.

## Sessions, Context, and chat

Sessions owns canonical settled conversation history. AIChatAgent's `messages`
is a bounded hydrated view, not a second durable source of truth.

- Message parent links form branches; reads select an active root-to-leaf path.
- Continuation rows and attachment chunks are storage details and must hydrate
  back to the exact logical message.
- FTS is lazy and disposable. Canonical message rows remain authoritative.
- Compaction writes overlays and never destroys raw messages.
- Legacy lifts retain source tables until copied rows and payloads verify.
- Context may shape or truncate a model request, but must not mutate Sessions to
  make the prompt fit.

Resumable chat stores chunks before broadcasting them. A reconnecting connection
is excluded from live delivery until it acknowledges replay, which preserves one
ordered buffered prefix followed by one live tail. Keep the replay/ACK transition
synchronous; inserting an `await` creates a duplicate-or-gap race.

TurnQueue serializes accepted chat turns. Tool approval and client-side results
resolve the in-flight message first, then persisted Sessions state, with
first-write-wins settlement. Approval prompts are persisted silently so a reload
does not lose the decision point.

## Sub-agents and Agent Tools

Sub-agents are same-machine Durable Object facets with isolated storage. A child
class needs an exported class but no binding or migration; only the root Agent
class has a Durable Object binding.

- Identity includes the full ancestor path. Do not simplify it to a bare child
  name or two children under different parents will collide.
- `on_before_sub_agent` gates each HTTP or WebSocket hop independently.
- A child `on_start` must not call its waiting parent; that deadlocks startup.
- `parent_agent` reaches one level only.
- `abort_sub_agent` preserves storage. `delete_sub_agent` destroys storage and is
  supported only from the top-level Agent; root deletion is transitive.
- Keep the registry row and retained JS proxy until runtime deletion succeeds.
- A buffered facet WebSocket event is limited to 30 seconds, 1,000 frames, and
  1 MiB. Exceeding a limit closes the physical socket with 1011.

Awaited Agent Tools are durable and idempotent by `run_id`. The parent persists
and replays every event, limits concurrency, and repairs stale soft runs. The
child formats input, executes one chat turn, and derives output and summary.
Sensitive callers can suppress `inputPreview`.

Live cross-facet tailing, reliable mid-flight cancellation, progress milestones,
and detached Agent Tool execution remain runtime-gated. Current awaited runs
forward a completed ordered chunk batch.

## Fibers

Fibers remain for compatibility while new replayable orchestration uses Tasks.
FiberCapability owns checkpoints, recovery, inspection, cancellation, deletion,
keep-alive leases, and one root maintenance job.

Detached fibers are disabled by default because work outside a retained Durable
Object I/O context is unsafe. Application code should use
`wait_for_completion=True` unless the deployed runtime evidence and authority
explicitly enable another mode.

Current opt-in durable chat recovery still uses managed awaited Fibers. Stage
5/I10 replaces accepted chat turns and recovery with reserved Tasks; do not build
new behavior around the temporary fiber reconstruction seam.

## Error boundaries

Lifecycle hooks are wrapped, reported once, and either propagated or swallowed
by the owning boundary:

- Fetch and upgrade failures propagate so callers receive a failure rather than
  a half-open resource.
- Socket wake callbacks report and swallow user failures at the outer runtime
  boundary so one bad message does not tear down dispatch.
- Alarm failures propagate after rearming logic so the runtime can retry.
- RPC, state, Tasks, Fibers, and chat use their own wire or ledger-visible error
  channels and must not emit duplicate unrelated reports.

Use `Connection.send_if_open` when a send can race a normal close. A raw send in
long-lived work can turn a routine disconnect into an error and abort later
broadcast recipients.

## Python Workers and FFI

The target is Python 3.12 (`.python-version`, `requires-python >=3.12`, and the
deployed Pyodide runtime). Use `.venv/bin/python`; do not verify with the system
Python or a newer interpreter that may accept syntax or typing behavior 3.12
rejects.

`js`, `workers`, and `pyodide.ffi` exist only in the Worker runtime. Tests install
deterministic substitutes from `tests/fakes.py` and `tests/_runtime_stubs.py`.
Use those fixtures rather than making production imports conditional.

- Worker storage and fetch wrappers often already return Python values; do not
  add `.to_py()` without checking the exact seam.
- Deserialized WebSocket attachments and bare facet RPC proxies may require
  conversion.
- Crossing into JavaScript dictionaries uses
  `to_js(..., dict_converter=Object.fromEntries)`.
- A JS callback used after the current call must have a retained, cached proxy
  and an explicit terminal release path.
- Do not create one proxy per facet lookup; reuse it by facet identity.

## Verification

Run focused tests while editing, then the complete Python 3.12 gate before a
reviewable implementation increment:

```bash
uvx ruff format --check agents/
uvx ruff check --select E,W --line-length 88 agents/
uvx ty check agents/
.venv/bin/python -m pytest
uv build
```

Use `uvx ruff format agents/` to apply formatting. The line length is 88.
Suppressions use `# ty: ignore[rule-name]`, must name the rule, and should exist
only for runtime behavior the checker cannot see. The final gate permits no
unfinished-directory exclusions.

Inspect the built wheel. Source imports do not prove the packaged artifact has
the right modules and exports. Runtime-risk changes also require a paired
testing-Worker fixture or commit and exact source, target, wheel, Worker, and
deployment hashes in the evidence ledger.

Tests should exercise the owning module interface. Use pure tests for algorithms,
SQLite/runtime-adapter tests for durability, and deployed probes for facts that
depend on workerd, Pyodide, hibernation, alarms, retained work, bindings, or JS
proxy lifetime. Fakes improve diagnosis but do not satisfy a deployed gate.

## Wheel delivery

There is no PyPI release. `../agents-py-testing` installs this package as a built
wheel through an absolute `file://` dependency, not an editable install.

- Build this repository before resolving the testing Worker.
- Run `../agents-py-testing/reinstall.sh` after SDK edits.
- Restart `npm run dev` in the testing project; its `uv run pywrangler dev`
  process does not hot-reload a vendored wheel.
- Do not claim deployed evidence unless the installed wheel hash matches the
  recorded Python source checkpoint.

## Remaining roadmap

The following work is not implemented or not release-proven:

- Stage 5/I10: move accepted chat turns and recovery to two reserved Tasks while
  preserving TurnQueue, resumable streams, approvals, client tools, and Agent
  Tool collection, then add continuation, `autoContinue`, and the parallel-tool
  barrier required by the authority.
- Stage 6: Workflow support and the stateless MCP server plus client manager,
  transports, OAuth, elicitation, restoration, and Agent integration.
- Stage 7: Browser Run, Quick Actions, model tools, durable sessions, reconnect,
  cleanup, Live View, recording, and Kitesurf scope.
- Stage 8: integrate exports, dependencies, compatibility dates, shared schema,
  and the flattened handshake; install the exact wheel into the testing Worker;
  run local, mixed-runtime, unmodified-client, Workflow, MCP, Browser,
  hibernation, cleanup, and deployed-runtime matrices; then inspect imports,
  startup, migrations, release docs, and the final wheel before aggregate review.
- Deployed evidence still needed for hibernating sockets, retained I/O,
  autonomous root alarms, facet-routed wakes, facet relay re-entry, cleanup,
  mixed shared tables, and the unmodified client across exact artifacts.

The old queue API, email routing, Skills engine, and full observability are
separate work unless a release explicitly adds them. Do not add speculative
stubs, aliases, dependencies, or compatibility layers for deferred scope.

## Engineering rules

- Make the smallest correct change and keep one clear owner for each invariant.
- Prefer small named functions, explicit control flow, and narrow protocols over
  generic adapters or inheritance stacks.
- Do not add backward compatibility without a concrete persisted, shipped, or
  external consumer.
- Do not combine the port with broad style cleanup.
- Add concise docstrings to public interfaces. Comments explain only
  load-bearing reasons such as wire omission, claim fencing, migration order, or
  proxy lifetime.
- Name time units in variables and convert at the seam.
- Land characterization before replacing behavior, then delete the obsolete
  implementation rather than retaining parallel paths.
- For a new increment, identify the focused test and expected initial failure,
  make the smallest production change, then simplify while the focused test and
  complete gate stay green.

## Project workflow

Issues and specs live in GitHub Issues for `justanotherbyte/agents-python`; see
`docs/agents/issue-tracker.md`. Triage uses the five-role vocabulary in
`docs/agents/triage-labels.md`.

This is a single-context repository. If present, read root `CONTEXT.md` and
relevant records under `docs/adr/`; both are created lazily and may be absent.
See `docs/agents/domain.md`.
