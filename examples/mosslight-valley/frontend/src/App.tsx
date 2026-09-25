import { useEffect, useLayoutEffect, useRef, useState } from "react";
import type { CSSProperties, FormEvent } from "react";
import { useAgentChat } from "@cloudflare/ai-chat/react";
import { useAgent } from "agents/react";
import {
  BERRIES,
  BRAMBLE,
  BRAMBLE_PORTRAIT,
  DIRT_PATH,
  FARMHOUSE,
  FISH_ITEM,
  GRASS_TILES,
  LANTERN_SPIRIT_PORTRAIT,
  MARKET,
  MIRA,
  MIRA_PORTRAIT,
  NORI,
  NORI_PORTRAIT,
  SEEDS,
  SOIL_DRY,
  SOIL_WET,
  TANSY,
  TANSY_PORTRAIT,
  TREE,
  TURNIPS,
  TURNIP_STAGES,
  WATER_FRAMES,
  WISP,
  WOOD_ITEM,
} from "./assets/pixel";
import type { Art } from "./assets/pixel";
import { artUrl } from "./pixelArt";
import { TitleScreen } from "./TitleScreen";

type Plot = {
  crop: string | null;
  stage: number;
  watered: boolean;
};

type CharacterId = "mira" | "bramble" | "nori" | "tansy";

type CharacterState = {
  name: string;
  x: number;
  y: number;
  energy: number;
  maxEnergy: number;
  activity: string;
};

type GameState = {
  revision: number;
  day: number;
  season: string;
  weather: string;
  energy: number;
  maxEnergy: number;
  coins: number;
  player: { x: number; y: number; facing: string };
  characters: Record<CharacterId, CharacterState>;
  inventory: {
    seeds: number;
    turnips: number;
    berries: number;
    fish: number;
    wood: number;
  };
  plots: Plot[];
  friendship: Record<CharacterId, number>;
  lastAction: string;
  lastTool: string;
};

type McpCatalog = {
  servers: Record<string, { state: string; error?: string | null }>;
  tools: Array<{ name?: string }>;
};

const GRID_WIDTH = 18;
const GRID_HEIGHT = 12;
const ART_SIZE = 16;

const PLOTS = [
  [3, 5],
  [4, 5],
  [5, 5],
  [6, 5],
  [3, 6],
  [4, 6],
  [5, 6],
  [6, 6],
] as const;

function regionKeys(xStart: number, xEnd: number, yStart: number, yEnd: number) {
  const keys = [];
  for (let y = yStart; y <= yEnd; y += 1) {
    for (let x = xStart; x <= xEnd; x += 1) keys.push(`${x}:${y}`);
  }
  return keys;
}

const TREES = new Set([
  ...regionKeys(16, 18, 1, 4),
  ...regionKeys(15, 18, 9, 12),
]);
TREES.delete("16:2");
TREES.delete("16:3");
const POND = new Set(regionKeys(9, 11, 2, 4));
const PATH = new Set([
  "4:1", "4:2", "4:3", "5:3", "6:3", "7:3", "8:3",
  "7:2", "7:4", "7:5", "7:6", "7:7", "8:7", "9:7",
  "10:7", "11:7", "12:7", "13:7", "14:7", "14:6", "14:5",
  "14:4", "14:3", "14:2", "15:3", "15:4", "15:5", "15:6",
  "15:7", "15:8",
]);

const QUICK_ACTIONS = [
  ["Plant", "Plant a turnip seed"],
  ["Water", "Water every thirsty crop"],
  ["Harvest", "Harvest every ripe crop"],
  ["Forage", "Forage in the forest"],
  ["Fish", "Go fishing at the pond"],
  ["Inspect", "Inspect the farm"],
  ["Rest", "Rest until the next day"],
] as const;

const INVENTORY_ITEMS = [
  ["seeds", "Seeds", SEEDS],
  ["turnips", "Turnips", TURNIPS],
  ["berries", "Berries", BERRIES],
  ["fish", "Fish", FISH_ITEM],
  ["wood", "Wood", WOOD_ITEM],
] as const;

const CHARACTERS = {
  mira: {
    id: "mira",
    name: "Mira",
    kind: "mira",
    agent: "MiraAgent",
    route: "mira-agent",
    request: "water the crops",
    sprite: MIRA,
    portrait: MIRA_PORTRAIT,
    role: "Seed keeper and garden agent",
  },
  bramble: {
    id: "bramble",
    name: "Bramble",
    kind: "bramble",
    agent: "BrambleAgent",
    route: "bramble-agent",
    request: "forage in Fernwood",
    sprite: BRAMBLE,
    portrait: BRAMBLE_PORTRAIT,
    role: "Forager and woodland agent",
  },
  nori: {
    id: "nori",
    name: "Nori",
    kind: "nori",
    agent: "NoriAgent",
    route: "nori-agent",
    request: "go fishing",
    sprite: NORI,
    portrait: NORI_PORTRAIT,
    role: "Fisher and pond-watching agent",
  },
  tansy: {
    id: "tansy",
    name: "Tansy",
    kind: "tansy",
    agent: "TansyAgent",
    route: "tansy-agent",
    request: "sell the harvest",
    sprite: TANSY,
    portrait: TANSY_PORTRAIT,
    role: "Merchant and market agent",
  },
} as const;

type CharacterDefinition = (typeof CHARACTERS)[CharacterId];

function isNearby(
  player: GameState["player"],
  character: CharacterState,
) {
  return Math.abs(player.x - character.x) + Math.abs(player.y - character.y) <= 1;
}

function tileKind(x: number, y: number) {
  const key = `${x}:${y}`;
  if (POND.has(key)) return "water";
  if (TREES.has(key)) return "forest";
  if (PATH.has(key)) return "path";
  if (x <= 3 && y <= 2) return "house";
  if (PLOTS.some(([plotX, plotY]) => plotX === x && plotY === y)) return "soil";
  if (x === 15 && y === 2) return "market";
  return "grass";
}

function tileArt(
  kind: ReturnType<typeof tileKind>,
  x: number,
  y: number,
  plot: Plot | null,
) {
  if (kind === "water") return WATER_FRAMES;
  if (kind === "forest") return TREE;
  if (kind === "path") return DIRT_PATH;
  if (kind === "market") return MARKET;
  if (kind === "soil") return plot?.watered ? SOIL_WET : SOIL_DRY;
  return GRASS_TILES[tileHash(x, y) % GRASS_TILES.length];
}

// Stable per-tile noise so grass variants scatter instead of forming rows or diagonals.
function tileHash(x: number, y: number) {
  let hash = Math.imul(x, 374761393) + Math.imul(y, 668265263);
  hash = Math.imul(hash ^ (hash >>> 13), 1274126177);
  return (hash ^ (hash >>> 16)) >>> 0;
}

function wispArt(facing: string) {
  if (facing === "north") return WISP.north;
  if (facing === "west" || facing === "east") return WISP.west;
  return WISP.south;
}

function Sprite({
  kind,
  name,
  art,
  x,
  y,
  flipped = false,
  nearby = false,
  description,
  onInteract,
}: {
  kind: "player" | CharacterDefinition["kind"];
  name: string;
  art: Art;
  x: number;
  y: number;
  flipped?: boolean;
  nearby?: boolean;
  description?: string;
  onInteract?: () => void;
}) {
  const contents = (
    <>
      <span
        className={`sprite-art pixel ${flipped ? "is-flipped" : ""}`}
        style={{ backgroundImage: artUrl(art) }}
      />
      <b>{name}</b>
      {nearby && kind !== "player" && (
        <span className="talk-tooltip" aria-hidden="true">
          <strong>{name}</strong>
          <small>{description}</small>
          <em>Press E or tap to talk</em>
        </span>
      )}
    </>
  );
  const props = {
    className: [
      "sprite",
      `sprite-${kind}`,
      nearby ? "is-nearby" : "",
      y <= 2 ? "tooltip-below" : "",
      x <= 2 ? "tooltip-from-left" : "",
      x >= 11 ? "tooltip-from-right" : "",
    ].filter(Boolean).join(" "),
    style: { gridColumn: x, gridRow: y } as CSSProperties,
  };

  if (onInteract) {
    return (
      <button
        type="button"
        {...props}
        onClick={onInteract}
        aria-label={
          nearby
            ? `${name}, ${description}. Nearby: press E or tap to talk.`
            : `${name}, ${description}, at map position ${x}, ${y}. Walk closer to talk.`
        }
        title={nearby ? `Talk to ${name}` : `Walk closer to ${name}`}
        disabled={!nearby}
      >
        {contents}
      </button>
    );
  }

  return (
    <div
      {...props}
      aria-label={`${name} at map position ${x}, ${y}`}
    >
      {contents}
    </div>
  );
}

function FarmMap({
  state,
  onTalk,
}: {
  state: GameState;
  onTalk: (character: CharacterId) => void;
}) {
  const viewportRef = useRef<HTMLDivElement>(null);
  const [camera, setCamera] = useState({
    width: 0,
    height: 0,
    x: 0,
    y: 0,
  });
  const tiles = [];
  const planted = state.plots.filter((plot) => plot.crop).length;
  const ready = state.plots.filter((plot) => plot.crop && plot.stage >= 3).length;
  for (let y = 1; y <= GRID_HEIGHT; y += 1) {
    for (let x = 1; x <= GRID_WIDTH; x += 1) {
      const plotIndex = PLOTS.findIndex(
        ([plotX, plotY]) => plotX === x && plotY === y,
      );
      const plot = plotIndex >= 0 ? state.plots[plotIndex] : null;
      const kind = tileKind(x, y);
      tiles.push(
        <div
          className={`tile pixel tile-${kind}`}
          key={`${x}:${y}`}
          style={{
            gridColumn: x,
            gridRow: y,
            backgroundImage: artUrl(tileArt(kind, x, y, plot)),
          }}
          aria-hidden="true"
        >
          {plot?.crop && (
            <span
              className="crop pixel"
              style={{
                backgroundImage: artUrl(
                  TURNIP_STAGES[Math.min(plot.stage, TURNIP_STAGES.length - 1)],
                ),
              }}
            />
          )}
        </div>,
      );
    }
  }

  useLayoutEffect(() => {
    const viewport = viewportRef.current;
    if (!viewport) return;

    const updateCamera = () => {
      const bounds = viewport.getBoundingClientRect();
      const portrait = bounds.width <= 760 && bounds.height > bounds.width;
      const landscapePhone = bounds.width <= 760 && bounds.height <= bounds.width;
      const shortMobile = landscapePhone && bounds.height <= 600;
      const fittedTileSize = portrait
        ? Math.max(bounds.width / 5, bounds.height / 8)
        : landscapePhone
          ? Math.max(bounds.width / 8, bounds.height / 5)
          : Math.max(bounds.width / 10, bounds.height / 6);
      // Whole screen pixels per art pixel keep 16px sprites crisp and even.
      const tileSize = Math.max(ART_SIZE, Math.round(fittedTileSize / ART_SIZE) * ART_SIZE);
      const width = tileSize * GRID_WIDTH;
      const height = tileSize * GRID_HEIGHT;
      const playerX = (state.player.x - 0.5) * tileSize;
      const playerY = (state.player.y - 0.5) * tileSize;
      let targetLeft = playerX - tileSize * 0.35;
      let targetRight = playerX + tileSize * 0.35;
      let targetTop = playerY - tileSize * 0.5;
      let targetBottom = playerY + tileSize * 0.5;

      for (const character of Object.values(state.characters)) {
        if (!isNearby(state.player, character)) continue;
        const characterX = (character.x - 0.5) * tileSize;
        const characterY = (character.y - 0.5) * tileSize;
        targetLeft = Math.min(targetLeft, characterX - 100);
        targetRight = Math.max(targetRight, characterX + 100);
        if (character.y <= 2 || shortMobile) {
          targetBottom = Math.max(targetBottom, characterY + tileSize * 0.5 + 100);
        } else {
          targetTop = Math.min(targetTop, characterY - tileSize * 0.5 - 100);
        }
      }

      let playLeft = 0;
      let playTop = 90;
      let playRight = bounds.width - (bounds.width <= 1080 ? 354 : 414);
      let playBottom = bounds.height;

      if (portrait) {
        playTop = 136;
        playRight = bounds.width;
        playBottom = bounds.height - Math.max(bounds.height * 0.42, 320) - 16;
      } else if (bounds.width <= 760) {
        playTop = 0;
        playRight = bounds.width * 0.55;
      }

      const avoidOverlay = (overlay: HTMLElement | null) => {
        if (!overlay) return;
        const overlayBounds = overlay.getBoundingClientRect();
        const overlayLeft = overlayBounds.left - bounds.left;
        const overlayTop = overlayBounds.top - bounds.top;
        const overlayRight = overlayBounds.right - bounds.left;
        const overlayBottom = overlayBounds.bottom - bounds.top;
        const intersects =
          overlayLeft < playRight &&
          overlayRight > playLeft &&
          overlayTop < playBottom &&
          overlayBottom > playTop;
        if (!intersects) return;

        const gap = 8;
        const candidates = [
          { left: playLeft, top: playTop, right: playRight, bottom: overlayTop - gap },
          { left: overlayRight + gap, top: playTop, right: playRight, bottom: playBottom },
          { left: playLeft, top: playTop, right: overlayLeft - gap, bottom: playBottom },
          { left: playLeft, top: overlayBottom + gap, right: playRight, bottom: playBottom },
        ].filter(
          (candidate) =>
            candidate.right - candidate.left >= tileSize * 1.2 &&
            candidate.bottom - candidate.top >= tileSize * 1.2,
        );
        const largest = candidates.sort(
          (left, right) =>
            (right.right - right.left) * (right.bottom - right.top) -
            (left.right - left.left) * (left.bottom - left.top),
        )[0];
        if (largest) {
          ({ left: playLeft, top: playTop, right: playRight, bottom: playBottom } = largest);
        }
      };

      avoidOverlay(document.querySelector<HTMLElement>(".movement-bar"));
      avoidOverlay(document.querySelector<HTMLElement>(".mini-map"));
      avoidOverlay(document.querySelector<HTMLElement>(".inventory-hotbar"));
      avoidOverlay(document.querySelector<HTMLElement>(".event-log"));
      avoidOverlay(document.querySelector<HTMLElement>(".quick-actions"));
      const instructions = document.querySelector<HTMLDetailsElement>(
        ".game-instructions",
      );
      if (instructions?.open) avoidOverlay(instructions);

      if (targetRight - targetLeft > playRight - playLeft) {
        targetLeft = playerX - tileSize * 0.35;
        targetRight = playerX + tileSize * 0.35;
      }
      if (targetBottom - targetTop > playBottom - playTop) {
        targetTop = playerY - tileSize * 0.5;
        targetBottom = playerY + tileSize * 0.5;
      }

      const focusX = (playLeft + playRight) / 2;
      const focusY = (playTop + playBottom) / 2;
      const targetX = (targetLeft + targetRight) / 2;
      const targetY = (targetTop + targetBottom) / 2;
      const desiredX = focusX - targetX;
      const desiredY = focusY - targetY;
      const x = Math.min(playLeft, Math.max(playRight - width, desiredX));
      const y = Math.min(playTop, Math.max(playBottom - height, desiredY));

      setCamera({ width, height, x: Math.round(x), y: Math.round(y) });
    };

    updateCamera();
    const observer = new ResizeObserver(updateCamera);
    observer.observe(viewport);
    const movementBar = document.querySelector<HTMLElement>(".movement-bar");
    if (movementBar) observer.observe(movementBar);
    const miniMap = document.querySelector<HTMLElement>(".mini-map");
    if (miniMap) observer.observe(miniMap);
    const inventoryHotbar =
      document.querySelector<HTMLElement>(".inventory-hotbar");
    if (inventoryHotbar) observer.observe(inventoryHotbar);
    const eventLog = document.querySelector<HTMLElement>(".event-log");
    if (eventLog) observer.observe(eventLog);
    const quickActions = document.querySelector<HTMLElement>(".quick-actions");
    if (quickActions) observer.observe(quickActions);
    const instructions = document.querySelector<HTMLElement>(".game-instructions");
    if (instructions) observer.observe(instructions);
    return () => observer.disconnect();
  }, [
    state.player.x,
    state.player.y,
    state.characters.mira.x,
    state.characters.mira.y,
    state.characters.bramble.x,
    state.characters.bramble.y,
    state.characters.nori.x,
    state.characters.nori.y,
    state.characters.tansy.x,
    state.characters.tansy.y,
  ]);

  return (
    <div className="map-wrap" ref={viewportRef}>
      <div
        className="farm-map"
        style={camera.width ? {
          width: `${camera.width}px`,
          height: `${camera.height}px`,
          transform: `translate3d(${camera.x}px, ${camera.y}px, 0)`,
          visibility: "visible",
        } : { visibility: "hidden" }}
        role="group"
        aria-label={`Mosslight Valley farm map. Wisp is at ${state.player.x}, ${state.player.y}. ${planted} plots are planted and ${ready} are ready to harvest. ${Object.values(state.characters).map((character) => `${character.name} is at ${character.x}, ${character.y}`).join(". ")}.`}
      >
        {tiles}
        <div
          className="farmhouse pixel"
          style={{
            gridColumn: "1 / 4",
            gridRow: "1 / 3",
            backgroundImage: artUrl(FARMHOUSE),
          }}
          aria-hidden="true"
        />
        {(Object.keys(CHARACTERS) as CharacterId[]).map((id) => {
          const definition = CHARACTERS[id];
          const character = state.characters[id];
          return (
            <Sprite
              key={id}
              kind={definition.kind}
              name={definition.name}
              art={definition.sprite}
              x={character.x}
              y={character.y}
              nearby={isNearby(state.player, character)}
              description={definition.role}
              onInteract={() => onTalk(id)}
            />
          );
        })}
        <Sprite
          kind="player"
          name="Wisp"
          art={wispArt(state.player.facing)}
          flipped={state.player.facing === "east"}
          x={state.player.x}
          y={state.player.y}
        />
      </div>
      <div className="map-legend" aria-hidden="true">
        <span>Old cottage</span>
        <span>Moon-turnip field</span>
        <span>Fernwood</span>
      </div>
    </div>
  );
}

function MiniMap({ state }: { state: GameState }) {
  const tiles = [];
  for (let y = 1; y <= GRID_HEIGHT; y += 1) {
    for (let x = 1; x <= GRID_WIDTH; x += 1) {
      tiles.push(
        <i
          className={`mini-tile mini-tile-${tileKind(x, y)}`}
          key={`${x}:${y}`}
          style={{ gridColumn: x, gridRow: y }}
        />,
      );
    }
  }

  return (
    <aside className="mini-map" aria-label="Mini-map of Mosslight Valley">
      <header>
        <strong>Valley map</strong>
        <span>{GRID_WIDTH} × {GRID_HEIGHT}</span>
      </header>
      <div className="mini-map-grid">
        {tiles}
        {(Object.keys(CHARACTERS) as CharacterId[]).map((id) => {
          const character = state.characters[id];
          return (
            <span
              className={`mini-marker mini-marker-${id}`}
              key={id}
              style={{ gridColumn: character.x, gridRow: character.y }}
              role="img"
              aria-label={`${character.name} at column ${character.x}, row ${character.y}`}
            />
          );
        })}
        <span
          className="mini-marker mini-marker-player"
          style={{ gridColumn: state.player.x, gridRow: state.player.y }}
          role="img"
          aria-label={`Wisp at column ${state.player.x}, row ${state.player.y}`}
        />
      </div>
      <div className="mini-map-key" aria-hidden="true">
        <span><i className="mini-marker-player" />Wisp</span>
        {(Object.keys(CHARACTERS) as CharacterId[]).map((id) => (
          <span key={id}><i className={`mini-marker-${id}`} />{CHARACTERS[id].name}</span>
        ))}
      </div>
    </aside>
  );
}

function InventoryHotbar({ inventory }: { inventory: GameState["inventory"] }) {
  return (
    <aside className="inventory-hotbar" aria-label="Inventory hotbar">
      <strong>Pack</strong>
      <ul>
        {INVENTORY_ITEMS.map(([item, label, art]) => (
          <li
            className={inventory[item] === 0 ? "is-empty" : ""}
            key={item}
            aria-label={`${label}: ${inventory[item]}`}
          >
            <i
              className="item-icon pixel"
              style={{ backgroundImage: artUrl(art) }}
              aria-hidden="true"
            />
            <span>{label}</span>
            <b>{inventory[item]}</b>
          </li>
        ))}
      </ul>
    </aside>
  );
}

function CharacterChat({
  farm,
  definition,
  character,
  onClose,
}: {
  farm: string;
  definition: CharacterDefinition;
  character: CharacterState;
  onClose: () => void;
}) {
  const [input, setInput] = useState("");
  const [submitError, setSubmitError] = useState<string | null>(null);
  const transcriptRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const wasBusyRef = useRef(false);
  const characterAgent = useAgent({
    agent: definition.agent,
    name: definition.id,
    basePath: `agents/farm-game/${farm}/sub/${definition.route}/${definition.id}`,
  });
  const {
    messages,
    sendMessage,
    status,
    isStreaming,
    isRecovering,
    error,
    connectionError,
  } = useAgentChat({ agent: characterAgent });
  const connected = characterAgent.identified && !connectionError;
  const busy =
    status === "submitted" ||
    status === "streaming" ||
    isStreaming ||
    isRecovering;
  const failure = submitError ?? error?.message ?? connectionError?.message ?? null;

  useEffect(() => {
    const transcript = transcriptRef.current;
    if (transcript) transcript.scrollTop = transcript.scrollHeight;
  }, [messages, isStreaming]);

  useEffect(() => {
    if (busy) {
      wasBusyRef.current = true;
    } else if (wasBusyRef.current) {
      wasBusyRef.current = false;
      inputRef.current?.blur();
    }
  }, [busy]);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const text = input.trim();
    if (!text || busy || !connected) return;
    setInput("");
    setSubmitError(null);
    try {
      await sendMessage({ text });
    } catch (cause) {
      setInput((current) => current || text);
      setSubmitError(
        cause instanceof Error ? cause.message : `${definition.name} could not hear you.`,
      );
    }
  }

  return (
    <aside
      className="spirit-panel character-panel"
      aria-labelledby="character-title"
      aria-busy={busy}
    >
      <header>
        <div
          className="character-portrait pixel"
          style={{ backgroundImage: artUrl(definition.portrait) }}
          aria-hidden="true"
        />
        <div>
          <p>{definition.role}</p>
          <h2 id="character-title">{definition.name}</h2>
        </div>
        <span
          className={`link-light ${connected ? "is-live" : ""}`}
          title={
            connected
              ? "Agent connected"
              : connectionError
                ? "Agent connection failed"
                : "Agent connecting"
          }
        />
      </header>

      <div className="character-status">
        <span>Agent energy {character.energy}/{character.maxEnergy}</span>
        <p>{character.activity}</p>
      </div>

      <div className="transcript" ref={transcriptRef} aria-live="polite">
        {messages.length === 0 && (
          <div className="welcome-message">
            <span>Independent AI character</span>
            <h3>Talk to {definition.name}.</h3>
            <p>
              Ask about the valley, or ask {definition.name} to {definition.request}.
              Each villager handles their own trade through MCP tools.
            </p>
          </div>
        )}

        {messages.map((message) => {
          const text = message.parts
            .flatMap((part) => (part.type === "text" ? [part.text] : []))
            .join("");
          if (!text) return null;
          return (
            <article key={message.id} className={`message message-${message.role}`}>
              <span>{message.role === "user" ? "Wisp" : definition.name}</span>
              <p>{text}</p>
            </article>
          );
        })}

        {busy && (
          <div className="thinking" aria-label={`${definition.name} is choosing an MCP tool`}>
            <i /><i /><i />
            <span>reading the shared world</span>
          </div>
        )}
      </div>

      {failure && <p className="error-message" role="alert">{failure}</p>}

      <form onSubmit={(event) => void submit(event)}>
        <label htmlFor={`command-${definition.id}`}>Talk to {definition.name}</label>
        <div className="composer">
          <input
            ref={inputRef}
            id={`command-${definition.id}`}
            value={input}
            onChange={(event) => setInput(event.target.value)}
            placeholder={`Ask ${definition.name} to ${definition.request}...`}
            autoComplete="off"
            autoFocus
          />
          <button type="submit" disabled={!input.trim() || busy || !connected}>
            Say
          </button>
        </div>
      </form>

      <footer>
        <button type="button" onClick={onClose}>End chat</button>
        {connectionError && (
          <button type="button" onClick={() => window.location.reload()}>
            Reconnect
          </button>
        )}
        <span>Durable {definition.agent} conversation</span>
      </footer>
    </aside>
  );
}

export function App() {
  const [farm, setFarm] = useState<string | null>(null);
  if (farm === null) return <TitleScreen onPlay={setFarm} />;
  return <Valley farm={farm} />;
}

function Valley({ farm }: { farm: string }) {
  const [state, setState] = useState<GameState | null>(null);
  const [mcp, setMcp] = useState<McpCatalog | null>(null);
  const [input, setInput] = useState("");
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [movementStatus, setMovementStatus] = useState({ id: 0, text: "" });
  const [selectedCharacter, setSelectedCharacter] = useState<CharacterId | null>(null);
  const transcriptRef = useRef<HTMLDivElement>(null);
  const autoScrollRef = useRef(true);
  const movementBarRef = useRef<HTMLDivElement>(null);
  const interactionReturnRef = useRef<HTMLElement | null>(null);

  const agent = useAgent<GameState>({
    agent: "FarmGame",
    name: farm,
    onStateUpdate: setState,
    onMcpUpdate: (catalog) => setMcp(catalog as McpCatalog),
  });
  const {
    messages,
    sendMessage,
    clearHistory,
    status,
    isStreaming,
    isRecovering,
    error,
    connectionError,
  } = useAgentChat({ agent });

  const connected = agent.identified && !connectionError;
  const busy =
    status === "submitted" ||
    status === "streaming" ||
    isStreaming ||
    isRecovering;
  const mcpServer = mcp?.servers["mosslight-tools"];
  const toolsReady = mcpServer?.state === "ready";
  const toolsFailed = mcpServer?.state === "failed";
  const failure = submitError ?? error?.message ?? connectionError?.message ?? null;
  const nearbyCharacter = state
    ? (Object.keys(CHARACTERS) as CharacterId[]).find((id) =>
        isNearby(state.player, state.characters[id]),
      ) ?? null
    : null;

  useEffect(() => {
    const transcript = transcriptRef.current;
    if (transcript && autoScrollRef.current) {
      transcript.scrollTop = transcript.scrollHeight;
    }
  }, [messages, isStreaming]);

  function trackTranscriptScroll() {
    const transcript = transcriptRef.current;
    if (!transcript) return;
    autoScrollRef.current =
      transcript.scrollHeight - transcript.scrollTop - transcript.clientHeight < 80;
  }

  async function move(direction: "north" | "south" | "west" | "east") {
    if (!connected) return;
    setSubmitError(null);
    try {
      const result = await agent.stub.move_player(direction);
      setMovementStatus((current) => ({
        id: current.id + 1,
        text: `${result.message} Position ${result.player.x}, ${result.player.y}.`,
      }));
    } catch (cause) {
      const message = cause instanceof Error ? cause.message : "Wisp could not move.";
      setSubmitError(message);
      setMovementStatus((current) => ({ id: current.id + 1, text: message }));
    }
  }

  function openCharacter(character: CharacterId) {
    interactionReturnRef.current =
      document.activeElement instanceof HTMLElement &&
      document.activeElement !== document.body
        ? document.activeElement
        : movementBarRef.current;
    setSelectedCharacter(character);
  }

  function closeCharacter() {
    const returnTarget = interactionReturnRef.current;
    setSelectedCharacter(null);
    requestAnimationFrame(() => {
      const canRestore = returnTarget?.isConnected && !returnTarget.matches(":disabled");
      (canRestore ? returnTarget : movementBarRef.current)?.focus();
    });
  }

  useEffect(() => {
    function handleKeyDown(event: KeyboardEvent) {
      if (
        document.querySelector(".character-panel[aria-busy='true']") ||
        event.defaultPrevented ||
        event.isComposing ||
        event.metaKey ||
        event.ctrlKey ||
        event.altKey ||
        event.shiftKey
      ) {
        return;
      }
      const target = event.target;
      if (
        target instanceof HTMLElement &&
        target.closest(
          "input, textarea, summary, .transcript, .character-panel, [contenteditable='true']",
        )
      ) {
        return;
      }

      const directions = {
        w: "north",
        arrowup: "north",
        s: "south",
        arrowdown: "south",
        a: "west",
        arrowleft: "west",
        d: "east",
        arrowright: "east",
      } as const;
      const direction = directions[event.key.toLowerCase() as keyof typeof directions];
      if (direction) {
        event.preventDefault();
        void move(direction);
        return;
      }
      if (event.key.toLowerCase() === "e" && nearbyCharacter) {
        event.preventDefault();
        openCharacter(nearbyCharacter);
      }
    }

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [agent, connected, nearbyCharacter]);

  async function send(text: string) {
    const message = text.trim();
    if (!message || busy || !connected) return false;
    setSubmitError(null);
    try {
      await sendMessage({ text: message });
      return true;
    } catch (cause) {
      setSubmitError(
        cause instanceof Error ? cause.message : "The valley could not hear you.",
      );
      return false;
    }
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const message = input;
    setInput("");
    if (!(await send(message))) {
      setInput((current) => current || message);
    }
  }

  async function reset() {
    if (busy || !connected) return;
    setSubmitError(null);
    try {
      clearHistory();
      await agent.stub.reset_world();
    } catch (cause) {
      setSubmitError(cause instanceof Error ? cause.message : "Reset failed.");
    }
  }

  if (!state) {
    return (
      <main className="loading-screen">
        <div className="loading-lantern"><i /></div>
        {failure ? (
          <>
            <p role="alert">{failure}</p>
            <button type="button" onClick={() => window.location.reload()}>
              Retry
            </button>
          </>
        ) : (
          <p>Following the lantern trail...</p>
        )}
      </main>
    );
  }

  return (
    <main className="valley-shell">
      <header className="valley-header">
        <div className="brand-mark" aria-hidden="true"><i /></div>
        <div className="brand-copy">
          <p>An agentic pocket farm</p>
          <h1>Mosslight Valley</h1>
        </div>
        <div className="day-card">
          <span>{state.season}</span>
          <strong>Day {state.day}</strong>
          <small>{state.weather}</small>
        </div>
      </header>

      <section className="status-ribbon" aria-label="Farm status">
        <div className="energy-stat">
          <span>Energy</span>
          <div className="energy-pips" aria-label={`${state.energy} of ${state.maxEnergy}`}>
            {Array.from({ length: state.maxEnergy }, (_, index) => (
              <i className={index < state.energy ? "is-full" : ""} key={index} />
            ))}
          </div>
        </div>
        <div><span>Coins</span><strong>{state.coins}</strong></div>
      </section>

      <div className="game-layout">
        <section className="world-panel" aria-labelledby="world-title">
          <div className="panel-heading">
            <div>
              <span>Live shared world</span>
              <h2 id="world-title">{farm}</h2>
            </div>
            <div
              className={`mcp-badge ${toolsReady ? "is-ready" : ""} ${toolsFailed ? "is-failed" : ""}`}
              title={mcpServer?.error ?? undefined}
            >
              <i />
              <span>
                {toolsReady
                  ? `${mcp?.tools.length ?? 0} MCP tools ready`
                  : toolsFailed
                    ? "MCP tools unavailable"
                    : "MCP tools waking"}
              </span>
            </div>
          </div>

          <FarmMap state={state} onTalk={openCharacter} />
          <MiniMap state={state} />
          <InventoryHotbar inventory={state.inventory} />

          <div
            className="movement-bar"
            aria-label="Movement controls"
            ref={movementBarRef}
            tabIndex={-1}
          >
            <div className="d-pad">
              <button type="button" disabled={!connected} onClick={() => void move("north")} aria-label="Move north">W</button>
              <button type="button" disabled={!connected} onClick={() => void move("west")} aria-label="Move west">A</button>
              <button type="button" disabled={!connected} onClick={() => void move("south")} aria-label="Move south">S</button>
              <button type="button" disabled={!connected} onClick={() => void move("east")} aria-label="Move east">D</button>
            </div>
            <div className="movement-copy">
              <strong>Move Wisp with keys or the D-pad</strong>
              <span aria-live="polite">
                <span key={movementStatus.id}>
                  {movementStatus.text || "Walk beside a character, then use E or Talk."}
                </span>
              </span>
            </div>
            {nearbyCharacter && (
              <button
                type="button"
                className="talk-prompt"
                disabled={!connected}
                onClick={() => openCharacter(nearbyCharacter)}
              >
                E · Talk to {CHARACTERS[nearbyCharacter].name}
              </button>
            )}
          </div>

          <div className="event-log">
            <span>Field notes</span>
            <p>{state.lastAction}</p>
            <code>{state.lastTool}</code>
          </div>

          <details className="game-instructions">
            <summary>How to play</summary>
            <p>
              Move Wisp with WASD, the arrow keys, or the touch D-pad. Use the
              mini-map to find Mira, Bramble, Nori, and Tansy, then walk beside one
              and use E or the Talk prompt to start a durable AI conversation.
            </p>
            <p>
              Ask characters about the world or request help. They inspect and
              change the shared farm through MCP tools, can travel to jobs on their
              own, and remember their individual conversations. Farm actions need
              energy; fish beside any pond edge, forage beside the forest, and rest
              to start a new day and restore everyone.
            </p>
          </details>

          <div className="quick-actions" aria-label="Suggested actions">
            {QUICK_ACTIONS.map(([label, prompt]) => (
              <button
                type="button"
                key={label}
                disabled={busy || !connected || !toolsReady}
                onClick={() => void send(prompt)}
              >
                {label}
              </button>
            ))}
          </div>
        </section>

        {selectedCharacter ? (
          <CharacterChat
            farm={farm}
            key={selectedCharacter}
            definition={CHARACTERS[selectedCharacter]}
            character={state.characters[selectedCharacter]}
            onClose={closeCharacter}
          />
        ) : (
        <aside className="spirit-panel" aria-labelledby="spirit-title">
          <header>
            <div
              className="spirit-portrait pixel"
              style={{ backgroundImage: artUrl(LANTERN_SPIRIT_PORTRAIT) }}
              aria-hidden="true"
            />
            <div>
              <p>AIChatAgent game master</p>
              <h2 id="spirit-title">Lantern Spirit</h2>
            </div>
            <span
              className={`link-light ${connected ? "is-live" : ""}`}
              title={connected ? "Connected" : connectionError ? "Connection failed" : "Connecting"}
            />
          </header>

          <div
            className="transcript"
            ref={transcriptRef}
            onScroll={trackTranscriptScroll}
            aria-live="polite"
          >
            {messages.length === 0 && (
              <div className="welcome-message">
                <span>Quest 01</span>
                <h3>Bring the old farm back to life.</h3>
                <p>
                  Tell the spirit what you want to do. It will choose an MCP tool,
                  update the durable world, and remember the adventure.
                </p>
              </div>
            )}

            {messages.map((message) => {
              const text = message.parts
                .flatMap((part) => (part.type === "text" ? [part.text] : []))
                .join("");
              if (!text) return null;
              return (
                <article key={message.id} className={`message message-${message.role}`}>
                  <span>{message.role === "user" ? "Wisp" : "Spirit"}</span>
                  <p>{text}</p>
                </article>
              );
            })}

            {busy && (
              <div className="thinking" aria-label="The spirit is choosing a tool">
                <i /><i /><i />
                <span>consulting the tool satchel</span>
              </div>
            )}
          </div>

          {failure && <p className="error-message" role="alert">{failure}</p>}
          {connectionError && (
            <button type="button" onClick={() => window.location.reload()}>
              Reconnect
            </button>
          )}

          <form onSubmit={(event) => void submit(event)}>
            <label htmlFor="command">What should Wisp do?</label>
            <div className="composer">
              <input
                id="command"
                value={input}
                onChange={(event) => setInput(event.target.value)}
                placeholder="Try: water the crops..."
                autoComplete="off"
              />
              <button type="submit" disabled={!input.trim() || busy || !connected}>
                Ask
              </button>
            </div>
          </form>

          <footer>
            <button type="button" onClick={() => void reset()} disabled={busy || !connected}>
              New farm
            </button>
            <span>Chat and world state are durable</span>
          </footer>
        </aside>
        )}
      </div>
    </main>
  );
}
