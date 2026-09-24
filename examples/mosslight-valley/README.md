# Mosslight Valley

Mosslight Valley is a small pixel-art farming adventure built around a shared
agentic world. The player moves Wisp directly with WASD, explores an 18 by 12
valley, and talks to Mira, Bramble, Nori, and Tansy through independent durable
`AIChatAgent`. Characters inspect the farm and perform requested work through
real MCP tools over same-Worker Durable Object RPC.

The example runs locally without an API key. Its deliberately small game-master
policy makes tool selection deterministic so the MCP and Agent mechanics are
easy to inspect. Replace `conversation.choose_action()` with a model call to let
an LLM select from the same discovered MCP catalog.

## Source layout

- `src/entry.py` composes and exports the Worker runtime classes.
- `src/mosslight/world.py` owns world geometry, movement, upgrades, and actions.
- `src/mosslight/mcp_server.py` exposes the farm actions as MCP tools.
- `src/mosslight/conversation.py` owns message, receipt, and reply helpers.
- `src/mosslight/valley_agents.py` keeps the game and four character agents together.

## What it demonstrates

- Five `AIChatAgent` characters with separate persistent conversations
- Server-authoritative WASD movement, collisions, proximity, and interactions
- Four routed character sub-agents under the authoritative `FarmGame`
- An MCP client registered and restored by the Agent lifecycle
- A separate `FarmTools` Durable Object exposing standard JSON-RPC MCP methods
- MCP tool discovery, selection, invocation, and structured tool results
- Durable world state synchronized to every connected browser
- A camera-following React pixel-art world with a live mini-map

## Run locally

Install the Worker and frontend dependencies:

```bash
uv sync --python 3.12
npm --prefix frontend ci
```

Run the Worker and frontend in separate terminals:

```bash
uv run pywrangler dev
npm --prefix frontend run dev
```

Open `http://localhost:5173`. Try messages such as:

- `Plant a turnip seed`
- `Water every thirsty crop`
- `Go forage in the forest`
- `Talk to Bramble`
- `Rest until the next day`

Use WASD, the arrow keys, or the touch D-pad to move. Walk beside a glowing
character and press E or use the Talk prompt to open their conversation. Every
character can answer questions about the current world or act on requests such
as `Please water the crops`; their MCP actions update the same world every
browser sees.

The frontend's MCP badge shows when the Agent has connected to the farm tool
server and discovered its catalog.

## Build and deploy

```bash
npm --prefix frontend run build
uv run pywrangler deploy
```

The `FarmGame` Agent and `FarmTools` MCP target each have their own Durable
Object binding. Keep both bindings and both classes in future migration tags.
The four character agents are sub-agent facets, so they are exported from the
Worker but deliberately need no additional binding or migration entry.
