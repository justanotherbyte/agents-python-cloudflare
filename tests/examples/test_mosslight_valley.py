from __future__ import annotations

import asyncio
import importlib.util
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import workers

added_worker_entrypoint = not hasattr(workers, "WorkerEntrypoint")
if added_worker_entrypoint:
    workers.WorkerEntrypoint = type("WorkerEntrypoint", (), {})


src = Path(__file__).parents[2] / "examples" / "mosslight-valley" / "src"
sys.path.insert(0, str(src))
try:
    conversation = importlib.import_module("mosslight.conversation")
    mcp_server = importlib.import_module("mosslight.mcp_server")
    valley_agents = importlib.import_module("mosslight.valley_agents")
    world = importlib.import_module("mosslight.world")

    entry_spec = importlib.util.spec_from_file_location(
        "mosslight_valley_entry", src / "entry.py"
    )
    assert entry_spec is not None and entry_spec.loader is not None
    entry = importlib.util.module_from_spec(entry_spec)
    entry_spec.loader.exec_module(entry)
finally:
    sys.path.remove(str(src))
    if added_worker_entrypoint:
        del workers.WorkerEntrypoint


def _next_state(state: dict, action: str) -> dict:
    result = world.act_on_farm(state, action, "", "mira")
    return result["structuredContent"]["state"]


def test_entry_exposes_all_runtime_classes() -> None:
    assert entry.FarmTools is mcp_server.FarmTools
    assert entry.FarmGame is valley_agents.FarmGame
    assert entry.MiraAgent is valley_agents.MiraAgent
    assert entry.BrambleAgent is valley_agents.BrambleAgent
    assert entry.NoriAgent is valley_agents.NoriAgent
    assert entry.TansyAgent is valley_agents.TansyAgent
    assert entry.Default.__name__ == "Default"


def test_crop_cycle_keeps_input_immutable() -> None:
    initial = world.initial_world()

    state = _next_state(initial, "plant")
    assert initial["plots"][0]["crop"] is None
    assert state["plots"][0]["crop"] == "turnip"

    for _ in range(3):
        state = _next_state(state, "water")
        state = _next_state(state, "rest")

    state = _next_state(state, "harvest")
    assert state["inventory"]["turnips"] == 1
    assert state["plots"][0]["crop"] is None


def test_movement_is_cardinal_server_authoritative_and_collision_aware() -> None:
    initial = world.initial_world()

    moved, success, _ = world.move_world(initial, "north")
    assert success is True
    assert moved["player"] == {"x": 5, "y": 3, "facing": "north"}
    assert moved["revision"] == 1
    assert initial["player"] == {"x": 5, "y": 4, "facing": "south"}

    blocked, success, message = world.move_world(moved, "east")
    assert success is False
    assert message == "That way is blocked"
    assert blocked["player"]["x"] == 5
    assert blocked["player"]["y"] == 3
    assert blocked["player"]["facing"] == "north"
    assert blocked["revision"] == 1


def test_character_destinations_are_passable_and_rest_spots_distinct() -> None:
    jobs = world.CHARACTER_ACTION_DESTINATIONS.values()
    rest_spots = [destinations["rest"] for destinations in jobs]

    assert len(rest_spots) == len(set(rest_spots))
    for destinations in jobs:
        assert not set(destinations.values()) & world.BLOCKED_TILES


CHARACTER_JOBS = [
    (actor, action)
    for actor, destinations in world.CHARACTER_ACTION_DESTINATIONS.items()
    for action in destinations
]


@pytest.mark.parametrize(("actor", "action"), CHARACTER_JOBS)
def test_every_character_can_reach_each_of_their_jobs(actor: str, action: str) -> None:
    result = world.act_on_farm(world.initial_world(), action, "", actor)
    structured = result["structuredContent"]
    character = structured["state"]["characters"][actor]

    assert "cannot find a clear path" not in structured["narration"]
    assert (character["x"], character["y"]) == (
        world.CHARACTER_ACTION_DESTINATIONS[actor][action]
    )


@pytest.mark.parametrize(("actor", "action"), CHARACTER_JOBS)
def test_wisp_standing_beside_a_character_never_boxes_them_in(
    actor: str, action: str
) -> None:
    initial = world.initial_world()
    home = initial["characters"][actor]
    occupied = {(other["x"], other["y"]) for other in initial["characters"].values()}
    approaches = [
        (home["x"] + dx, home["y"] + dy)
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0))
        if (home["x"] + dx, home["y"] + dy) not in world.BLOCKED_TILES | occupied
    ]
    assert approaches

    for x, y in approaches:
        state = world.initial_world()
        state["player"].update(x=x, y=y)
        narration = world.act_on_farm(state, action, "", actor)["structuredContent"][
            "narration"
        ]
        assert "cannot find a clear path" not in narration, (x, y)


@pytest.mark.parametrize(
    ("actor", "action", "expected"),
    (
        ("tansy", "plant", "Tansy leaves the planting to Mira."),
        ("mira", "fish", "Mira leaves the fishing to Nori."),
        ("nori", "sell", "Nori leaves the selling to Tansy."),
        ("bramble", "water", "Bramble leaves the watering to Mira."),
    ),
)
def test_characters_decline_work_outside_their_specialty(
    actor: str, action: str, expected: str
) -> None:
    initial = world.initial_world()

    structured = world.act_on_farm(initial, action, "", actor)["structuredContent"]

    assert structured["narration"] == expected
    state = structured["state"]
    assert state["inventory"] == initial["inventory"]
    assert state["plots"] == initial["plots"]
    assert state["characters"][actor] == initial["characters"][actor]


def test_npc_talk_moves_next_to_the_target() -> None:
    initial = world.initial_world()

    result = world.act_on_farm(initial, "talk", "Talk to Nori", "bramble")
    state = result["structuredContent"]["state"]

    assert world.near(state["characters"]["bramble"], state["characters"]["nori"])
    assert state["friendship"]["nori"] == 1


def test_talk_requires_an_explicit_character() -> None:
    initial = world.initial_world()

    result = world.act_on_farm(initial, "talk", "", "wisp")

    assert result["structuredContent"]["narration"].startswith("Choose Mira")
    assert result["structuredContent"]["state"]["friendship"] == initial["friendship"]


def test_reset_advances_revision_to_fence_stale_character_work() -> None:
    state = world.initial_world()
    state["generation"] = 7
    state["revision"] = 41
    state["coins"] = 999

    reset = world.reset_world(state)

    assert reset["generation"] == 8
    assert reset["revision"] == 42
    assert reset["coins"] == 24


def test_npc_action_moves_npc_and_uses_its_energy_not_wisps() -> None:
    initial = world.initial_world()

    result = world.act_on_farm(initial, "forage", "", "bramble")
    state = result["structuredContent"]["state"]

    assert state["player"] == initial["player"]
    assert state["energy"] == initial["energy"]
    assert state["characters"]["bramble"]["energy"] == 4
    assert (
        state["characters"]["bramble"]["x"],
        state["characters"]["bramble"]["y"],
    ) == (14, 10)
    assert state["inventory"]["berries"] == 3


def test_npc_action_rejects_an_occupied_destination() -> None:
    initial = world.initial_world()
    initial["player"].update(x=14, y=10)

    result = world.act_on_farm(initial, "forage", "", "bramble")

    assert "cannot find a clear path" in result["structuredContent"]["narration"]
    assert initial["characters"]["bramble"]["energy"] == 6
    assert result["structuredContent"]["state"]["characters"]["bramble"]["energy"] == 6


@pytest.mark.parametrize(("action", "place"), (("fish", "pond"), ("forage", "forest")))
def test_wisp_must_walk_beside_action_terrain(action: str, place: str) -> None:
    initial = world.initial_world()

    result = world.act_on_farm(initial, action, "", "wisp")

    assert result["structuredContent"]["narration"] == (
        f"Walk beside the {place} before trying to {action}."
    )


@pytest.mark.parametrize(
    ("action", "position", "item"),
    (("fish", (8, 2), "fish"), ("forage", (14, 9), "berries")),
)
def test_wisp_can_gather_beside_any_matching_terrain(
    action: str, position: tuple[int, int], item: str
) -> None:
    state = world.initial_world()
    state["player"].update(x=position[0], y=position[1])

    result = world.act_on_farm(state, action, "", "wisp")

    assert result["structuredContent"]["state"]["inventory"][item] > 0


def test_plant_uses_an_empty_plot_next_to_wisp() -> None:
    state = world.initial_world()
    state["player"].update(x=6, y=4)

    result = world.act_on_farm(state, "plant", "", "wisp")
    structured = result["structuredContent"]

    assert "get closer" not in structured["narration"].lower()
    assert structured["state"]["plots"][3]["crop"] == "turnip"


@pytest.mark.asyncio
async def test_character_commit_cannot_cross_reset_generation() -> None:
    class Mcp:
        def __init__(self) -> None:
            self.calls = 0

        async def call_tool(self, _server_id: str, _name: str, arguments: dict):
            self.calls += 1
            return world.act_on_farm(
                arguments["state"],
                arguments["action"],
                arguments["target"],
                arguments["actor"],
            )

    class Parent:
        def __init__(self) -> None:
            self.state = world.initial_world()
            self.state["generation"] = 2
            self.state["revision"] = 9
            self.mcp = Mcp()

        def set_state(self, state: dict) -> None:
            self.state = state

    parent = Parent()
    result = await valley_agents.FarmGame.commit_character_action(
        parent,
        "mira",
        "old-turn",
        1,
        9,
        "plant",
        "",
    )

    assert result["applied"] is False
    assert result["cancelled"] is True
    assert parent.mcp.calls == 0
    assert parent.state["inventory"]["seeds"] == 6


@pytest.mark.asyncio
async def test_character_commit_rechecks_proximity() -> None:
    class Mcp:
        async def call_tool(self, *_args, **_kwargs):
            raise AssertionError("MCP must not run after the player walks away")

    class Parent:
        def __init__(self) -> None:
            self.state = world.initial_world()
            self.mcp = Mcp()

    parent = Parent()
    result = await valley_agents.FarmGame.commit_character_action(
        parent,
        "mira",
        "far-turn",
        0,
        0,
        "plant",
        "",
    )

    assert result["applied"] is False
    assert result["cancelled"] is True
    assert "too far away" in result["reason"]


def test_character_turn_validation_rechecks_generation_and_proximity() -> None:
    class Parent:
        state = world.initial_world()

    parent = Parent()
    reset_result = valley_agents.FarmGame.validate_character_turn(parent, "mira", 1, 0)
    assert reset_result["valid"] is False
    assert "previous farm" in reset_result["reason"]

    far_result = valley_agents.FarmGame.validate_character_turn(parent, "mira", 0, 0)
    assert far_result["valid"] is False
    assert "too far away" in far_result["reason"]

    parent.state["player"].update(x=5, y=3)
    assert (
        valley_agents.FarmGame.validate_character_turn(parent, "mira", 0, 0)["valid"]
        is True
    )

    parent.state["revision"] = 1
    changed = valley_agents.FarmGame.validate_character_turn(parent, "mira", 0, 0)
    assert changed["valid"] is False
    assert changed["retry"] is True


def test_turn_receipts_are_replayable_and_bounded() -> None:
    state = world.initial_world()
    for index in range(conversation.MAX_TURN_RECEIPTS + 1):
        conversation.remember_turn(
            state, f"message-{index}", f"reply-{index}", "act_on_farm"
        )

    assert conversation.turn_receipt(state, "message-0") is None
    assert conversation.turn_receipt(
        state, f"message-{conversation.MAX_TURN_RECEIPTS}"
    ) == (
        f"reply-{conversation.MAX_TURN_RECEIPTS}",
        "act_on_farm",
    )


def test_character_receipt_is_fenced_by_farm_generation() -> None:
    state = {"turnReceipts": {}}
    conversation.remember_turn(state, "message", "reply", "inspect_farm", generation=3)

    assert conversation.turn_receipt(state, "message", 3) == (
        "reply",
        "inspect_farm",
    )
    assert conversation.turn_receipt(state, "message", 4) is None


def test_selling_the_harvest_selects_sell() -> None:
    assert conversation.choose_action("Please sell the harvest")[0] == "sell"
    assert conversation.choose_action("Harvest every ripe crop")[0] == "harvest"


def test_friendship_does_not_select_the_shipping_action() -> None:
    assert conversation.choose_action("How is our friendship?")[0] is None


@pytest.mark.asyncio
async def test_character_dialogue_uses_persona_world_context_and_gateway() -> None:
    class Ai:
        def __init__(self) -> None:
            self.call = None

        async def run(self, model: str, inputs: dict, options: dict):
            self.call = (model, inputs, options)
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "The roots look ready for a patient hand.",
                        }
                    }
                ]
            }

    ai = Ai()
    snapshot = world.initial_world()
    messages = [
        {
            "id": "older-user",
            "role": "user",
            "parts": [{"type": "text", "text": "How are the turnips?"}],
        },
        {
            "id": "current-user",
            "role": "user",
            "parts": [{"type": "text", "text": "Please water them"}],
        },
    ]

    reply = await conversation.model_character_reply(
        ai,
        "mira",
        "current-user",
        "Please water them",
        "Mira watered two thirsty plots.",
        snapshot,
        messages,
        acted=True,
    )

    assert reply == "The roots look ready for a patient hand."
    model, inputs, options = ai.call
    assert model == "@cf/zai-org/glm-4.7-flash"
    assert options == {"gateway": {"id": "default", "skipCache": True}}
    assert inputs["max_completion_tokens"] == 160
    assert inputs["chat_template_kwargs"] == {"enable_thinking": False}
    assert "patient botanist" in inputs["messages"][0]["content"]
    assert inputs["messages"][1] == {
        "role": "user",
        "content": "How are the turnips?",
    }
    current = inputs["messages"][-1]["content"]
    assert "Mira watered two thirsty plots." in current
    assert "Friendship with Wisp is 0" in current
    assert "Please water them" in current
    assert current.count("Please water them") == 1


@pytest.mark.asyncio
async def test_character_dialogue_reports_plainly_when_inference_fails(caplog) -> None:
    class Ai:
        async def run(self, *_args, **_kwargs):
            raise RuntimeError("inference unavailable")

    reply = await conversation.model_character_reply(
        Ai(),
        "nori",
        "current-user",
        "How is the pond?",
        "The pond is calm today.",
        world.initial_world(),
        [],
        acted=False,
    )

    assert reply == "The pond is calm today."
    assert "character dialogue inference failed" in caplog.text


@pytest.mark.parametrize(
    ("character", "persona"),
    (
        ("mira", "patient botanist"),
        ("bramble", "playful forest forager"),
        ("nori", "quiet fisher"),
        ("tansy", "brisk market keeper"),
    ),
)
def test_each_character_has_a_distinct_model_persona(
    character: str, persona: str
) -> None:
    prompt = conversation._model_messages(
        character,
        "message",
        "Hello",
        "The farm is quiet.",
        world.initial_world(),
        [],
        acted=False,
    )[0]["content"]

    assert persona in prompt


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("world_change", "expected"),
    (
        (
            "reset",
            ["That conversation belonged to the previous farm. Please ask again."],
        ),
        ("revision", ["The world moved beneath your feet. Please ask again."]),
        ("before_model", ["The world moved beneath your feet. Please ask again."]),
        ("abort", []),
    ),
)
async def test_character_does_not_persist_reply_across_world_change(
    world_change: str, expected: list[str]
) -> None:
    class Ai:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def run(self, *_args, **_kwargs):
            self.started.set()
            await self.release.wait()
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Those old roots still look healthy.",
                        }
                    }
                ]
            }

    class Mcp:
        def get_connection(self, _server_id: str):
            return SimpleNamespace(state=valley_agents.MCPConnectionState.READY)

        async def call_tool(self, _server_id: str, _name: str, arguments: dict):
            return world.inspect_farm(arguments["state"])

    class Parent:
        def __init__(self) -> None:
            self.state = world.initial_world()
            self.state["player"].update(x=5, y=3)
            self.snapshot_calls = 0
            self.post_inference_snapshot = asyncio.Event()
            self.release_snapshot = asyncio.Event()

        async def world_snapshot(self):
            self.snapshot_calls += 1
            if world_change == "before_model" and self.snapshot_calls == 2:
                self.state["revision"] += 1
            if world_change == "abort" and self.snapshot_calls == 3:
                self.post_inference_snapshot.set()
                await self.release_snapshot.wait()
            return deepcopy(self.state)

        async def validate_character_turn(
            self, actor: str, generation: int, revision: int
        ):
            return valley_agents.FarmGame.validate_character_turn(
                self, actor, generation, revision
            )

    class Character:
        character_id = "mira"

        def __init__(self, parent: Parent, ai: Ai) -> None:
            self.state = {"turnReceipts": {}, "turnGenerations": {}}
            self.messages = [
                {
                    "id": "user-1",
                    "role": "user",
                    "parts": [{"type": "text", "text": "How is the farm?"}],
                }
            ]
            self.mcp = Mcp()
            self.env = SimpleNamespace(AI=ai)
            self.parent = parent

        async def parent_agent(self, _parent_type):
            return self.parent

        def set_state(self, state: dict) -> None:
            self.state = state

    parent = Parent()
    ai = Ai()
    character = Character(parent, ai)
    options = SimpleNamespace(aborted=False, continuation=False)

    async def collect_reply() -> list[str]:
        return [
            chunk
            async for chunk in valley_agents.MiraAgent.on_chat_message(
                character, options
            )
        ]

    turn = asyncio.create_task(collect_reply())
    if world_change != "before_model":
        await ai.started.wait()
        if world_change == "reset":
            parent.state = world.reset_world(parent.state)
        elif world_change == "revision":
            parent.state["revision"] += 1
        ai.release.set()
    if world_change == "abort":
        await parent.post_inference_snapshot.wait()
        options.aborted = True
        parent.release_snapshot.set()

    assert await turn == expected
    assert ai.started.is_set() is (world_change != "before_model")
    assert character.state["turnReceipts"] == {}


@pytest.mark.asyncio
async def test_recovered_turn_reuses_receipt_without_repeating_text() -> None:
    class Mcp:
        def __init__(self) -> None:
            self.calls = 0

        def get_connection(self, _server_id: str):
            return SimpleNamespace(state=valley_agents.MCPConnectionState.READY)

        async def call_tool(self, _server_id: str, _name: str, arguments: dict):
            self.calls += 1
            return world.act_on_farm(
                arguments["state"], arguments["action"], arguments["target"]
            )

    class Agent:
        def __init__(self) -> None:
            self.state = world.initial_world()
            self.messages = [
                {
                    "id": "user-1",
                    "role": "user",
                    "parts": [{"type": "text", "text": "Plant a turnip seed"}],
                }
            ]
            self.mcp = Mcp()

        def set_state(self, state: dict) -> None:
            self.state = state

    agent = Agent()
    options = SimpleNamespace(aborted=False, continuation=False)
    first = [
        chunk async for chunk in valley_agents.FarmGame.on_chat_message(agent, options)
    ]
    narration = "".join(first)
    assert agent.mcp.calls == 1
    assert agent.state["inventory"]["seeds"] == 5

    agent.messages.append(
        {
            "id": "assistant-1",
            "role": "assistant",
            "parts": [{"type": "text", "text": narration[:-10]}],
        }
    )
    options.continuation = True
    recovered = [
        chunk async for chunk in valley_agents.FarmGame.on_chat_message(agent, options)
    ]

    assert agent.mcp.calls == 1
    assert narration[:-10] + "".join(recovered) == narration


@pytest.mark.asyncio
async def test_mcp_protocol_version_notifications_and_unknown_methods() -> None:
    tools = object.__new__(mcp_server.FarmTools)

    initialized = await tools.handle_mcp_message(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "unsupported"},
        }
    )
    assert initialized["result"]["protocolVersion"] == mcp_server.MCP_PROTOCOL_VERSION

    notification = await tools.handle_mcp_message(
        {"jsonrpc": "2.0", "method": "unknown/event"}
    )
    assert notification is None

    unknown = await tools.handle_mcp_message(
        {"jsonrpc": "2.0", "id": 2, "method": "unknown/request"}
    )
    assert unknown["error"]["code"] == -32601

    malformed = await tools.handle_mcp_message(
        {"jsonrpc": "1.0", "id": 3, "method": "tools/list"}
    )
    assert malformed["error"]["code"] == -32600
