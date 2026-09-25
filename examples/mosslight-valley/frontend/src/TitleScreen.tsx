import { useEffect, useRef, useState } from "react";
import type { FormEvent } from "react";
import { BRAMBLE, GRASS_TUFTS, MIRA, NORI, TANSY, WISP } from "./assets/pixel";
import { artUrl } from "./pixelArt";

const VILLAGERS = [
  { name: "Mira", art: MIRA, trade: "Plants, waters, and harvests the crops." },
  { name: "Bramble", art: BRAMBLE, trade: "Forages glowberries in Fernwood." },
  { name: "Nori", art: NORI, trade: "Fishes the silver pond." },
  { name: "Tansy", art: TANSY, trade: "Sells the harvest at the market." },
];

const LAST_FARM_KEY = "mosslight:last-farm";
const MAX_FARM_NAME_LENGTH = 32;

/** Farm names become Durable Object names in the URL path, so keep them URL-safe. */
export function farmSlug(name: string) {
  return name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, MAX_FARM_NAME_LENGTH)
    .replace(/-+$/, "");
}

export function TitleScreen({ onPlay }: { onPlay: (farm: string) => void }) {
  const [view, setView] = useState<"menu" | "instructions">("menu");
  const [farmName, setFarmName] = useState(
    () => localStorage.getItem(LAST_FARM_KEY) ?? "",
  );
  const farmRef = useRef<HTMLInputElement>(null);
  const backRef = useRef<HTMLButtonElement>(null);
  const farm = farmSlug(farmName);

  useEffect(() => {
    (view === "menu" ? farmRef : backRef).current?.focus();
  }, [view]);

  function play(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!farm) return;
    localStorage.setItem(LAST_FARM_KEY, farmName.trim());
    onPlay(farm);
  }

  useEffect(() => {
    if (view !== "instructions") return;
    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") setView("menu");
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [view]);

  return (
    <main
      className="title-screen"
      style={{
        backgroundImage: `linear-gradient(rgba(23, 41, 29, 0.35), rgba(23, 41, 29, 0.35)), ${artUrl(GRASS_TUFTS)}`,
      }}
    >
      <section className="title-card" aria-labelledby="title-heading">
        <div className="title-brand">
          <div className="brand-mark" aria-hidden="true" />
          <div>
            <p>An agentic pocket farm</p>
            <h1 id="title-heading">Mosslight Valley</h1>
          </div>
        </div>

        {view === "menu" ? (
          <>
            <ul className="title-parade" aria-hidden="true">
              {[WISP.south, MIRA, BRAMBLE, NORI, TANSY].map((art, index) => (
                <li
                  className="pixel"
                  key={index}
                  style={{ backgroundImage: artUrl(art) }}
                />
              ))}
            </ul>
            <form className="title-actions" onSubmit={play}>
              <label className="title-farm" htmlFor="farm-name">
                <span>Farm name</span>
                <input
                  ref={farmRef}
                  id="farm-name"
                  value={farmName}
                  onChange={(event) => setFarmName(event.target.value)}
                  placeholder="e.g. mossy-meadow"
                  maxLength={48}
                  autoComplete="off"
                  spellCheck={false}
                  aria-describedby="farm-name-hint"
                />
                <small id="farm-name-hint">
                  {farm
                    ? `Opens /${farm}. Anyone who enters the same name shares this farm.`
                    : "Name your farm to start, or enter a friend's farm name to join it."}
                </small>
              </label>
              <button
                type="submit"
                className="title-button is-primary"
                disabled={!farm}
              >
                Play
              </button>
              <button
                type="button"
                className="title-button"
                onClick={() => setView("instructions")}
              >
                Instructions
              </button>
            </form>
          </>
        ) : (
          <>
            <div className="title-instructions">
              <h2>How to play</h2>
              <ol>
                <li>
                  Move Wisp with <kbd>W</kbd> <kbd>A</kbd> <kbd>S</kbd> <kbd>D</kbd>,
                  the arrow keys, or the touch D-pad.
                </li>
                <li>
                  Ask the Lantern Spirit to plant, water, harvest, forage, fish,
                  sell, or inspect the farm.
                </li>
                <li>
                  Walk beside a villager and press <kbd>E</kbd> or tap Talk to chat.
                  Each one remembers your conversation.
                </li>
                <li>
                  Farm work costs energy. Rest to start a new day, restore everyone,
                  and help watered crops grow.
                </li>
              </ol>
              <h3>The villagers</h3>
              <ul className="title-villagers">
                {VILLAGERS.map((villager) => (
                  <li key={villager.name}>
                    <i
                      className="pixel"
                      style={{ backgroundImage: artUrl(villager.art) }}
                      aria-hidden="true"
                    />
                    <div>
                      <strong>{villager.name}</strong>
                      <span>{villager.trade}</span>
                    </div>
                  </li>
                ))}
              </ul>
            </div>
            <button
              ref={backRef}
              type="button"
              className="title-button"
              onClick={() => setView("menu")}
            >
              Back
            </button>
          </>
        )}

        <footer>Python Agents SDK · Workers AI · MCP</footer>
      </section>
    </main>
  );
}
