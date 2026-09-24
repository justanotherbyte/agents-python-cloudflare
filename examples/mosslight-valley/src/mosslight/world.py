from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

__all__ = [
    "ACTION_DESTINATIONS",
    "BLOCKED_TILES",
    "CHARACTER_ACTION_DESTINATIONS",
    "GRID_HEIGHT",
    "GRID_WIDTH",
    "act_on_farm",
    "initial_world",
    "inspect_farm",
    "move_world",
    "near",
    "reset_world",
    "upgrade_world",
]

WEATHER = ("soft sun", "leafy breeze", "silver rain", "soft sun")
GRID_WIDTH = 18
GRID_HEIGHT = 12
MOVE_DELTAS = {
    "north": (0, -1),
    "south": (0, 1),
    "west": (-1, 0),
    "east": (1, 0),
}
PLOT_TILES = (
    (3, 5),
    (4, 5),
    (5, 5),
    (6, 5),
    (3, 6),
    (4, 6),
    (5, 6),
    (6, 6),
)
POND_TILES = {(x, y) for x in range(9, 12) for y in range(2, 5)}
FOREST_TILES = {(x, y) for x in range(16, 19) for y in range(1, 5)} | {
    (x, y) for x in range(15, 19) for y in range(9, 13)
}
FOREST_TILES.discard((16, 2))
FOREST_TILES.discard((16, 3))
HOUSE_TILES = {(x, y) for x in range(1, 4) for y in range(1, 3)}
MARKET_TILE = (15, 2)
BLOCKED_TILES = POND_TILES | FOREST_TILES | HOUSE_TILES | {MARKET_TILE}
ACTION_DESTINATIONS = {
    "plant": (4, 4),
    "water": (4, 4),
    "harvest": (4, 4),
    "forage": (14, 9),
    "fish": (8, 3),
    "sell": (14, 2),
    "rest": (4, 2),
}
CHARACTER_ACTION_DESTINATIONS = {
    "mira": ACTION_DESTINATIONS,
    "bramble": {
        "plant": (7, 6),
        "water": (7, 6),
        "harvest": (7, 6),
        "forage": (14, 10),
        "fish": (12, 4),
        "sell": (14, 3),
        "rest": (5, 3),
    },
    "nori": {
        "plant": (7, 5),
        "water": (7, 5),
        "harvest": (7, 5),
        "forage": (14, 11),
        "fish": (12, 3),
        "sell": (15, 1),
        "rest": (12, 6),
    },
    "tansy": {
        "plant": (6, 7),
        "water": (6, 7),
        "harvest": (6, 7),
        "forage": (14, 12),
        "fish": (12, 2),
        "sell": (16, 2),
        "rest": (13, 2),
    },
}


def initial_world() -> dict[str, Any]:
    return {
        "day": 1,
        "season": "Leafrise",
        "weather": WEATHER[0],
        "generation": 0,
        "energy": 8,
        "maxEnergy": 8,
        "coins": 24,
        "revision": 0,
        "player": {"x": 5, "y": 4, "facing": "south"},
        "characters": {
            "mira": {
                "name": "Mira",
                "x": 6,
                "y": 3,
                "energy": 6,
                "maxEnergy": 6,
                "activity": "Sorting moon-turnip seeds",
            },
            "bramble": {
                "name": "Bramble",
                "x": 14,
                "y": 8,
                "energy": 6,
                "maxEnergy": 6,
                "activity": "Listening for crow songs",
            },
            "nori": {
                "name": "Nori",
                "x": 12,
                "y": 5,
                "energy": 6,
                "maxEnergy": 6,
                "activity": "Mending a silver fishing net",
            },
            "tansy": {
                "name": "Tansy",
                "x": 16,
                "y": 2,
                "energy": 6,
                "maxEnergy": 6,
                "activity": "Chalking prices on the market board",
            },
        },
        "inventory": {
            "seeds": 6,
            "turnips": 0,
            "berries": 0,
            "fish": 0,
            "wood": 0,
        },
        "plots": [{"crop": None, "stage": 0, "watered": False} for _ in range(8)],
        "friendship": {"mira": 0, "bramble": 0, "nori": 0, "tansy": 0},
        "lastAction": "Wisp found the old farm key beneath a mossy stone.",
        "lastTool": "world_boot",
        "turnReceipts": {},
        "characterReceipts": {},
    }


def upgrade_world(current: Mapping[str, Any]) -> dict[str, Any]:
    defaults = initial_world()
    state = deepcopy(dict(current))
    state.setdefault("generation", defaults["generation"])
    state.setdefault("revision", defaults["revision"])
    player = state.setdefault("player", defaults["player"])
    player.setdefault("facing", "south")
    characters = state.setdefault("characters", {})
    occupied = {(player["x"], player["y"])}
    occupied.update(
        (character["x"], character["y"]) for character in characters.values()
    )
    for character_id, character in defaults["characters"].items():
        if character_id in characters:
            continue
        added = deepcopy(character)
        preferred = (added["x"], added["y"])
        if preferred in occupied or preferred in BLOCKED_TILES:
            preferred = min(
                (
                    (x, y)
                    for y in range(1, GRID_HEIGHT + 1)
                    for x in range(1, GRID_WIDTH + 1)
                    if (x, y) not in occupied and (x, y) not in BLOCKED_TILES
                ),
                key=lambda point: (
                    abs(point[0] - added["x"]) + abs(point[1] - added["y"]),
                    point[1],
                    point[0],
                ),
            )
        added.update(x=preferred[0], y=preferred[1])
        characters[character_id] = added
        occupied.add(preferred)
    friendship = state.setdefault("friendship", {})
    for character_id in defaults["friendship"]:
        friendship.setdefault(character_id, 0)
    state.setdefault("turnReceipts", {})
    state.setdefault("characterReceipts", {})
    return state


def _distance(left: Mapping[str, Any], right: Mapping[str, Any]) -> int:
    return abs(int(left["x"]) - int(right["x"])) + abs(int(left["y"]) - int(right["y"]))


def near(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return _distance(left, right) <= 1


def move_world(
    current: Mapping[str, Any], direction: str
) -> tuple[dict[str, Any], bool, str]:
    state = upgrade_world(current)
    delta = MOVE_DELTAS.get(direction)
    if delta is None:
        return state, False, "Unknown direction"

    player = state["player"]
    destination = (player["x"] + delta[0], player["y"] + delta[1])
    occupied = {
        (character["x"], character["y"]) for character in state["characters"].values()
    }
    if (
        destination[0] < 1
        or destination[0] > GRID_WIDTH
        or destination[1] < 1
        or destination[1] > GRID_HEIGHT
        or destination in BLOCKED_TILES
        or destination in occupied
    ):
        return state, False, "That way is blocked"

    player["facing"] = direction
    player.update(x=destination[0], y=destination[1])
    state["revision"] += 1
    state["lastAction"] = f"Wisp walks {direction} through the valley."
    state["lastTool"] = "move_player"
    return state, True, state["lastAction"]


def reset_world(current: Mapping[str, Any]) -> dict[str, Any]:
    state = initial_world()
    state["generation"] = int(current.get("generation", 0)) + 1
    state["revision"] = int(current.get("revision", 0)) + 1
    return state


def _npc_route(
    state: Mapping[str, Any], actor: str, destination: tuple[int, int]
) -> list[tuple[int, int]] | None:
    start_state = state["characters"][actor]
    start = (int(start_state["x"]), int(start_state["y"]))
    occupied = {(state["player"]["x"], state["player"]["y"])}
    occupied.update(
        (character["x"], character["y"])
        for character_id, character in state["characters"].items()
        if character_id != actor
    )
    if destination in BLOCKED_TILES or destination in occupied:
        return None

    frontier = [start]
    routes = {start: [start]}
    while frontier:
        current = frontier.pop(0)
        if current == destination:
            return routes[current]
        for delta in MOVE_DELTAS.values():
            neighbor = (current[0] + delta[0], current[1] + delta[1])
            if (
                neighbor in routes
                or neighbor in BLOCKED_TILES
                or neighbor in occupied
                or neighbor[0] < 1
                or neighbor[0] > GRID_WIDTH
                or neighbor[1] < 1
                or neighbor[1] > GRID_HEIGHT
            ):
                continue
            routes[neighbor] = [*routes[current], neighbor]
            frontier.append(neighbor)
    return None


def _tool_result(
    state: dict[str, Any],
    narration: str,
    action: str,
) -> dict[str, Any]:
    state["lastAction"] = narration
    state["lastTool"] = action
    return {
        "content": [{"type": "text", "text": narration}],
        "structuredContent": {
            "state": state,
            "action": action,
            "narration": narration,
        },
    }


def _actor_name(state: Mapping[str, Any], actor: str) -> str:
    if actor == "wisp":
        return "Wisp"
    character = state["characters"].get(actor)
    return str(character["name"]) if character else actor.title()


def _actor_position(state: Mapping[str, Any], actor: str) -> Mapping[str, Any]:
    return state["player"] if actor == "wisp" else state["characters"][actor]


def _spend_energy(state: dict[str, Any], actor: str, amount: int) -> bool:
    source = state if actor == "wisp" else state["characters"][actor]
    if source["energy"] < amount:
        return False
    source["energy"] -= amount
    return True


def _target_character(target: str) -> str | None:
    lowered = target.lower()
    return next(
        (name for name in ("bramble", "nori", "tansy", "mira") if name in lowered),
        None,
    )


def _prepare_action(
    state: dict[str, Any], actor: str, action: str, target: str
) -> str | None:
    destinations = CHARACTER_ACTION_DESTINATIONS.get(actor, ACTION_DESTINATIONS)
    destination = destinations.get(action)
    if action == "talk":
        villager = _target_character(target)
        if villager is None:
            return "Choose Mira, Bramble, Nori, or Tansy to talk with."
        destination = (
            state["characters"][villager]["x"],
            state["characters"][villager]["y"],
        )
        if actor != "wisp":
            routes = []
            for delta_x, delta_y in MOVE_DELTAS.values():
                adjacent = (destination[0] + delta_x, destination[1] + delta_y)
                route = _npc_route(state, actor, adjacent)
                if route is not None:
                    routes.append((len(route), adjacent, route))
            if not routes:
                return f"{_actor_name(state, actor)} cannot find a clear path to talk."
            _, destination, route = min(routes)
            actor_position = _actor_position(state, actor)
            actor_position.update(x=destination[0], y=destination[1])
            actor_position["activity"] = f"Travelled {len(route) - 1} steps to talk"
            return None
    if destination is None:
        return None

    actor_position = _actor_position(state, actor)
    if actor == "wisp" and action in {"plant", "water", "harvest"}:
        if not any(
            near(actor_position, {"x": plot_x, "y": plot_y})
            for plot_x, plot_y in PLOT_TILES
        ):
            return f"Walk closer before trying to {action}."
        return None
    if actor == "wisp" and action in {"fish", "forage"}:
        terrain = POND_TILES if action == "fish" else FOREST_TILES
        if not any(
            near(actor_position, {"x": tile_x, "y": tile_y})
            for tile_x, tile_y in terrain
        ):
            place = "pond" if action == "fish" else "forest"
            return f"Walk beside the {place} before trying to {action}."
        return None
    target_position = {"x": destination[0], "y": destination[1]}
    if actor == "wisp":
        if not near(actor_position, target_position):
            return f"Walk closer before trying to {action}."
        return None

    route = _npc_route(state, actor, destination)
    if route is None:
        return f"{_actor_name(state, actor)} cannot find a clear path to {action}."
    actor_position.update(x=destination[0], y=destination[1])
    actor_position["activity"] = f"Travelled {len(route) - 1} steps to {action}"
    return None


def _action_plots(state: dict[str, Any], actor: str) -> list[dict[str, Any]]:
    plots = state["plots"]
    if actor != "wisp":
        return plots
    actor_position = _actor_position(state, actor)
    return [
        plot
        for plot, (plot_x, plot_y) in zip(plots, PLOT_TILES, strict=True)
        if near(actor_position, {"x": plot_x, "y": plot_y})
    ]


def inspect_farm(state: dict[str, Any]) -> dict[str, Any]:
    planted = sum(plot["crop"] is not None for plot in state["plots"])
    ready = sum(plot["stage"] >= 3 for plot in state["plots"])
    narration = (
        f"The farm ledger shows {planted} planted plots, {ready} ready to harvest, "
        f"{state['energy']} energy, and {state['coins']} coins."
    )
    return _tool_result(state, narration, "inspect_farm")


def act_on_farm(
    current: Mapping[str, Any],
    action: str,
    target: str,
    actor: str = "wisp",
) -> dict[str, Any]:
    state = upgrade_world(current)
    inventory = state["inventory"]
    plots = state["plots"]
    if actor != "wisp" and actor not in state["characters"]:
        return _tool_result(state, f"Unknown valley character: {actor}.", action)
    name = _actor_name(state, actor)
    blocked = _prepare_action(state, actor, action, target)
    if blocked is not None:
        return _tool_result(state, blocked, action)
    available_plots = _action_plots(state, actor)

    if action == "plant":
        empty = next((plot for plot in available_plots if plot["crop"] is None), None)
        if empty is None:
            return _tool_result(state, "Every garden plot is already occupied.", action)
        if inventory["seeds"] < 1:
            return _tool_result(state, "The seed pouch is empty.", action)
        if not _spend_energy(state, actor, 1):
            return _tool_result(state, f"{name} is too tired to plant today.", action)
        inventory["seeds"] -= 1
        empty.update(crop="turnip", stage=0, watered=False)
        narration = f"{name} presses a moon-turnip seed into the warm garden soil."
    elif action == "water":
        thirsty = [
            plot for plot in available_plots if plot["crop"] and not plot["watered"]
        ]
        if not thirsty:
            return _tool_result(state, "No planted crops need water right now.", action)
        if not _spend_energy(state, actor, 1):
            return _tool_result(
                state, f"The watering can feels too heavy for {name} today.", action
            )
        for plot in thirsty:
            plot["watered"] = True
        narration = (
            f"{name} settles rain-bright water over {len(thirsty)} garden plots."
        )
    elif action == "harvest":
        ready = [
            plot for plot in available_plots if plot["crop"] and plot["stage"] >= 3
        ]
        if not ready:
            return _tool_result(state, "Nothing is ripe enough to harvest yet.", action)
        if not _spend_energy(state, actor, 1):
            return _tool_result(
                state, f"{name} needs rest before gathering crops.", action
            )
        inventory["turnips"] += len(ready)
        for plot in ready:
            plot.update(crop=None, stage=0, watered=False)
        narration = f"{name} gathers {len(ready)} moon-turnips into a wicker basket."
    elif action == "forage":
        if not _spend_energy(state, actor, 2):
            return _tool_result(
                state,
                f"{name} decides the forest path can wait until tomorrow.",
                action,
            )
        berries = 2 + state["day"] % 2
        wood = 1 if state["day"] % 2 == 0 else 0
        inventory["berries"] += berries
        inventory["wood"] += wood
        extra = " and a fallen branch" if wood else ""
        narration = f"Beneath the ferns, {name} finds {berries} glowberries{extra}."
    elif action == "fish":
        if not _spend_energy(state, actor, 2):
            return _tool_result(
                state, f"{name} is too sleepy to watch a fishing float.", action
            )
        inventory["fish"] += 1
        narration = f"A tiny sunperch flashes gold as {name} lifts it from the pond."
    elif action == "talk":
        villager = _target_character(target)
        if villager is None:
            return _tool_result(
                state, "Choose Mira, Bramble, Nori, or Tansy to talk with.", action
            )
        state["friendship"][villager] += 1
        conversations = {
            "mira": (
                "Mira shares a seed-saving trick and a warm cup of clover tea "
                f"with {name}."
            ),
            "bramble": (
                f"Bramble tells {name} the crows are excellent poets, "
                "if a little repetitive."
            ),
            "nori": f"Nori teaches {name} how to read ripples beneath the reeds.",
            "tansy": f"Tansy gives {name} a cheerful lesson in market bargaining.",
        }
        narration = conversations[villager]
    elif action == "sell":
        turnips = inventory["turnips"]
        berries = inventory["berries"]
        fish = inventory["fish"]
        earnings = turnips * 12 + berries * 4 + fish * 9
        if earnings == 0:
            return _tool_result(state, "The market basket is empty.", action)
        inventory.update(turnips=0, berries=0, fish=0)
        state["coins"] += earnings
        narration = f"The village market pays {earnings} coins for {name}'s harvest."
    elif action == "rest":
        state["day"] += 1
        state["weather"] = WEATHER[(state["day"] - 1) % len(WEATHER)]
        state["energy"] = state["maxEnergy"]
        for character in state["characters"].values():
            character["energy"] = character["maxEnergy"]
        for plot in plots:
            if plot["crop"] and plot["watered"]:
                plot["stage"] = min(3, plot["stage"] + 1)
            plot["watered"] = False
        narration = (
            f"Day {state['day']} begins with {state['weather']} "
            "and a chorus of finches."
        )
    else:
        return _tool_result(
            state, f"The farm tools do not know how to {action}.", action
        )

    if actor != "wisp":
        state["characters"][actor]["activity"] = narration
    return _tool_result(state, narration, action)
