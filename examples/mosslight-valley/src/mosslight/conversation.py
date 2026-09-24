from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = [
    "MAX_TURN_RECEIPTS",
    "character_reply",
    "choose_action",
    "latest_assistant_text",
    "latest_player_message",
    "remember_turn",
    "runtime_value",
    "turn_receipt",
]

MAX_TURN_RECEIPTS = 32


def latest_player_message(messages: list[dict[str, Any]]) -> tuple[str, str]:
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        message_id = message.get("id")
        if not isinstance(message_id, str):
            continue
        parts = message.get("parts")
        if not isinstance(parts, list):
            continue
        text = "".join(
            part.get("text", "")
            for part in parts
            if isinstance(part, Mapping) and part.get("type") == "text"
        )
        if text.strip():
            return message_id, text.strip()
    return "", ""


def latest_assistant_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") != "assistant":
            continue
        parts = message.get("parts")
        if not isinstance(parts, list):
            continue
        return "".join(
            part.get("text", "")
            for part in parts
            if isinstance(part, Mapping) and part.get("type") == "text"
        )
    return ""


def turn_receipt(
    state: Mapping[str, Any],
    message_id: str,
    expected_generation: int | None = None,
) -> tuple[str, str] | None:
    receipts = state.get("turnReceipts")
    if not isinstance(receipts, Mapping):
        return None
    receipt = receipts.get(message_id)
    if not isinstance(receipt, Mapping):
        return None
    if (
        expected_generation is not None
        and receipt.get("generation") != expected_generation
    ):
        return None
    narration = receipt.get("narration")
    tool_name = receipt.get("toolName")
    if not isinstance(narration, str) or not isinstance(tool_name, str):
        return None
    return narration, tool_name


def remember_turn(
    state: dict[str, Any],
    message_id: str,
    narration: str,
    tool_name: str,
    generation: int | None = None,
) -> None:
    receipts = dict(state.get("turnReceipts") or {})
    receipt = {"narration": narration, "toolName": tool_name}
    if generation is not None:
        receipt["generation"] = generation
    receipts[message_id] = receipt
    while len(receipts) > MAX_TURN_RECEIPTS:
        del receipts[next(iter(receipts))]
    state["turnReceipts"] = receipts


def choose_action(message: str) -> tuple[str | None, str]:
    text = message.lower()
    choices = (
        ("harvest", ("harvest", "pick crops", "gather crops")),
        ("plant", ("plant", "sow", "seed")),
        ("water", ("water", "watering")),
        ("forage", ("forage", "forest", "berries", "gather wood")),
        ("fish", ("fish", "pond", "fishing")),
        ("talk", ("talk", "mira", "bramble", "nori", "tansy", "visit")),
        ("sell", ("sell", "market")),
        ("rest", ("rest", "sleep", "next day", "bed")),
        ("inspect", ("inspect", "status", "look around", "what should", "farm")),
    )
    for action, words in choices:
        if any(word in text for word in words):
            return action, message
    return None, message


def runtime_value(value: Any) -> Any:
    converter = getattr(value, "to_py", None)
    return converter() if callable(converter) else value


def character_reply(
    character: str,
    player_message: str,
    world_report: str,
    *,
    acted: bool,
) -> str:
    if acted:
        if character == "mira":
            return f'Mira brushes soil from her gloves. "{world_report}"'
        if character == "bramble":
            return f'Bramble tips his leaf-woven hat. "{world_report}"'
        if character == "nori":
            return f'Nori coils a silver line around one hand. "{world_report}"'
        return f'Tansy closes her coin ledger. "{world_report}"'

    text = player_message.lower()
    if character == "mira":
        if "hello" in text or "hi" in text:
            opening = "It is good to see you, Wisp."
        elif "seed" in text or "crop" in text:
            opening = "Healthy roots begin with patient hands."
        else:
            opening = "The valley notices every small kindness."
        return f'Mira smiles. "{opening} {world_report}"'

    if character == "bramble":
        if "crow" in text:
            opening = "The crows say your timing is improving."
        elif "forest" in text or "berry" in text:
            opening = "Fernwood leaves a trail for anyone quiet enough to see it."
        else:
            opening = "Every good farm needs one unnecessary mystery."
        return f'Bramble grins. "{opening} {world_report}"'

    if character == "nori":
        opening = (
            "The pond answers patient questions."
            if "fish" in text or "pond" in text
            else "Even still water is busy underneath."
        )
        return f'Nori watches the reeds. "{opening} {world_report}"'

    opening = (
        "A fair price leaves both baskets lighter."
        if "sell" in text or "coin" in text or "market" in text
        else "There is always room for one more good trade."
    )
    return f'Tansy taps her ledger. "{opening} {world_report}"'
