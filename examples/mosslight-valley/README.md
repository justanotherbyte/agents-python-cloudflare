# Mosslight Valley

Mosslight Valley is a small pixel-art farming adventure built around a shared
agentic world. The player moves Wisp directly with WASD, explores an 18 by 12
valley, and talks to Mira, Bramble, Nori, and Tansy through independent durable
`AIChatAgent`. Workers AI gives each character a distinct conversational voice,
while characters inspect the farm and perform requested work through real MCP
tools over same-Worker Durable Object RPC.

The example runs locally without an API key. Character dialogue uses
`@cf/zai-org/glm-4.7-flash` through the account's `default` AI Gateway, with
caching bypassed because replies depend on private conversation and world state.
Tool selection stays deterministic, so model output cannot mutate the shared
world or bypass MCP validation. If inference fails, the character replies with
the plain farm report, so completed actions stay visible and the failure is
logged rather than disguised as dialogue.

## Source layout

- `src/entry.py` composes and exports the Worker runtime classes.
- `src/mosslight/world.py` owns world geometry, movement, and actions.
- `src/mosslight/mcp_server.py` exposes the farm actions as MCP tools.
- `src/mosslight/conversation.py` owns message, receipt, and reply helpers.
- `src/mosslight/valley_agents.py` keeps the game and four character agents together.
- `frontend/src/assets/pixel/` holds the 16×16 pixel art as palette-and-grid
  constants; `frontend/src/pixelArt.ts` renders them to cached images.
- `frontend/src/TitleScreen.tsx` is the title menu where you name your farm,
  read the instructions, and press Play.

## What it demonstrates

- Five `AIChatAgent` characters with separate persistent conversations
- Four Workers AI personas routed through AI Gateway logging and analytics
- Server-authoritative WASD movement, collisions, proximity, and interactions
- Four routed character sub-agents under the authoritative `FarmGame`
- An MCP client registered and restored by the Agent lifecycle
- A separate `FarmTools` Durable Object exposing standard JSON-RPC MCP methods
- MCP tool discovery, selection, invocation, and structured tool results
- One `FarmGame` Durable Object per farm name, each with its own world
- Durable world state synchronized to every browser on the same farm
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

Open `http://localhost:5173`, type a farm name, and press Play. Each name is its
own `FarmGame` Durable Object, so a new name starts a fresh farm and anyone who
types the same name joins yours. The name works like a shared join code, not a
private account. Try messages such as:

- `Plant a turnip seed`
- `Water every thirsty crop`
- `Go forage in the forest`
- `Talk to Bramble`
- `Rest until the next day`

Use WASD, the arrow keys, or the touch D-pad to move. Walk beside a glowing
character and press E or use the Talk prompt to open their conversation. Every
character can answer questions about the current world and handles their own
trade: Mira tends crops, Bramble forages, Nori fishes, and Tansy sells the
harvest. Anyone can rest to start the next day. Requests outside a villager's
trade are declined by the MCP tool, and every completed action updates the world
for every browser on that farm.

The frontend's MCP badge shows when the Agent has connected to the farm tool
server and discovered its catalog.

Workers AI uses the Cloudflare account associated with Wrangler. The `default`
AI Gateway is created automatically on its first authenticated binding request
if it does not already exist.

## Build and deploy

```bash
npm --prefix frontend run build
uv run pywrangler deploy
```

The `FarmGame` Agent and `FarmTools` MCP target each have their own Durable
Object binding. Keep both bindings and both classes in future migration tags.
The four character agents are sub-agent facets, so they are exported from the
Worker but deliberately need no additional binding or migration entry.
