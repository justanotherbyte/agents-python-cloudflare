from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from agents import AIChatAgent, ChatOptions, rpc_callable
from agents.mcp.client import MCPConnectionState

from . import conversation, mcp_server, world

__all__ = ("BrambleAgent", "FarmGame", "MiraAgent", "NoriAgent", "TansyAgent")


async def ensure_mcp_connection(agent: AIChatAgent) -> None:
    servers = {server.id for server in agent.mcp.list_servers()}
    if mcp_server.MCP_SERVER_ID not in servers:
        await agent.mcp.register_rpc_server(
            mcp_server.MCP_SERVER_ID,
            name="Mosslight farm tools",
            rpc_name="shared-world",
            binding_name="FARM_TOOLS",
        )
        await agent.mcp.connect_to_server(mcp_server.MCP_SERVER_ID)
        return

    connection = agent.mcp.get_connection(mcp_server.MCP_SERVER_ID)
    if connection is None or connection.state != MCPConnectionState.READY:
        await agent.mcp.establish_connection(mcp_server.MCP_SERVER_ID)


class FarmGame(AIChatAgent):
    max_persisted_messages = 120

    def initial_state(self):
        return world.initial_world()

    async def on_start(self):
        await ensure_mcp_connection(self)
        await self.sub_agent(MiraAgent, "mira")
        await self.sub_agent(BrambleAgent, "bramble")
        await self.sub_agent(NoriAgent, "nori")
        await self.sub_agent(TansyAgent, "tansy")

    def _handle_state_update(self, connection, data):
        # The world is tool-owned; clients may observe it but cannot replace it.
        was_readonly = self.is_connection_readonly(connection)
        if not was_readonly:
            self.set_connection_readonly(connection)
        try:
            super()._handle_state_update(connection, data)
        finally:
            if not was_readonly:
                self.set_connection_readonly(connection, False)

    @rpc_callable()
    def reset_world(self):
        self._turn_queue.reset()
        for abort in self._aborts.values():
            abort.set()
        state = world.reset_world(self.state)
        self.set_state(state)
        return state

    @rpc_callable()
    def move_player(self, direction: str):
        state, moved, message = world.move_world(self.state, direction)
        if moved:
            self.set_state(state)
        return {"moved": moved, "message": message, "player": state["player"]}

    def world_snapshot(self):
        return deepcopy(self.state)

    def validate_character_turn(
        self,
        actor: str,
        expected_generation: int,
        expected_revision: int,
    ):
        current = self.state
        if current["generation"] != expected_generation:
            return {
                "valid": False,
                "reason": "That request belongs to the previous farm.",
            }
        if actor not in current["characters"]:
            return {"valid": False, "reason": "That character is unavailable."}
        if not world.near(current["player"], current["characters"][actor]):
            name = current["characters"][actor]["name"]
            return {"valid": False, "reason": f"{name} is too far away now."}
        if current["revision"] != expected_revision:
            return {
                "valid": False,
                "retry": True,
                "reason": "The world changed while we were looking. Trying again.",
                "state": current,
            }
        return {"valid": True}

    async def commit_character_action(
        self,
        actor: str,
        message_id: str,
        expected_generation: int,
        expected_revision: int,
        action: str,
        target: str,
    ):
        current = self.state
        if (
            actor not in current["characters"]
            or action not in world.ACTION_DESTINATIONS
        ):
            return {"applied": False, "state": current}
        if current["generation"] != expected_generation:
            return {
                "applied": False,
                "cancelled": True,
                "reason": "That request belongs to the previous farm.",
                "state": current,
            }
        receipt_key = f"{actor}:{message_id}"
        receipt = current["characterReceipts"].get(receipt_key)
        if receipt is not None:
            return {
                "applied": True,
                "duplicate": True,
                "state": current,
                **receipt,
            }
        if not world.near(current["player"], current["characters"][actor]):
            name = current["characters"][actor]["name"]
            return {
                "applied": False,
                "cancelled": True,
                "reason": f"{name} is too far away now.",
                "state": current,
            }
        if current["revision"] != expected_revision:
            return {"applied": False, "state": current}

        result = await self.mcp.call_tool(
            mcp_server.MCP_SERVER_ID,
            "act_on_farm",
            {
                "state": current,
                "action": action,
                "target": target,
                "actor": actor,
            },
        )
        structured = result.get("structuredContent")
        if not isinstance(structured, Mapping):
            return {"applied": False, "state": current}
        candidate_state = structured.get("state")
        narration = structured.get("narration")
        if not isinstance(candidate_state, Mapping) or not isinstance(narration, str):
            return {"applied": False, "state": current}
        if self.state["revision"] != expected_revision:
            return {"applied": False, "state": self.state}

        state = dict(candidate_state)
        state["player"] = deepcopy(current["player"])
        state["revision"] = expected_revision + 1
        state["characterReceipts"] = deepcopy(current["characterReceipts"])
        receipt = {"narration": narration, "toolName": "act_on_farm"}
        state["characterReceipts"][receipt_key] = receipt
        while len(state["characterReceipts"]) > conversation.MAX_TURN_RECEIPTS:
            del state["characterReceipts"][next(iter(state["characterReceipts"]))]
        state["lastAction"] = narration
        state["lastTool"] = f"{actor} / act_on_farm"
        self.set_state(state)
        return {
            "applied": True,
            "duplicate": False,
            "state": state,
            **receipt,
        }

    async def on_chat_message(self, options: ChatOptions):
        if options.aborted:
            return

        message_id, player_message = conversation.latest_player_message(self.messages)
        receipt = conversation.turn_receipt(self.state, message_id)
        if receipt is not None:
            narration, tool_name = receipt
            reply = f"{narration}  [MCP / {tool_name}]"
            if options.continuation:
                prior_text = conversation.latest_assistant_text(self.messages)
                if reply.startswith(prior_text):
                    remainder = reply[len(prior_text) :]
                    if remainder:
                        yield remainder
                return
            yield reply
            return

        action, target = conversation.choose_action(player_message)
        if action is None:
            yield "The valley spirit tilts its lantern. "
            yield (
                "Try asking me to plant, water, harvest, forage, fish, visit Mira "
                "or Bramble, sell goods, inspect the farm, or rest."
            )
            return

        connection = self.mcp.get_connection(mcp_server.MCP_SERVER_ID)
        if connection is None or connection.state != MCPConnectionState.READY:
            try:
                await self.mcp.establish_connection(mcp_server.MCP_SERVER_ID)
            except Exception:  # noqa: BLE001
                yield "The farm tool satchel could not open. Please try again."
                return
            connection = self.mcp.get_connection(mcp_server.MCP_SERVER_ID)
            if connection is None or connection.state != MCPConnectionState.READY:
                yield "The farm tool satchel is still waking up. Please try that again."
                return

        tool_name = "inspect_farm" if action == "inspect" else "act_on_farm"
        revision = self.state["revision"]
        arguments: dict[str, Any] = {"state": self.state}
        if action != "inspect":
            arguments.update(action=action, target=target, actor="wisp")

        result = await self.mcp.call_tool(
            mcp_server.MCP_SERVER_ID, tool_name, arguments
        )
        structured = result.get("structuredContent")
        if not isinstance(structured, Mapping):
            yield "The farm tools returned without a new page for the ledger."
            return
        state = structured.get("state")
        narration = structured.get("narration")
        if not isinstance(state, Mapping) or not isinstance(narration, str):
            yield "The valley's reply was too tangled to understand."
            return

        if options.aborted:
            return
        if self.state["revision"] != revision:
            yield "The world moved beneath your feet. Try that action once more."
            return

        next_state = dict(state)
        next_state["revision"] = revision + 1
        conversation.remember_turn(next_state, message_id, narration, tool_name)
        self.set_state(next_state)
        yield narration
        yield f"  [MCP / {tool_name}]"


class ValleyCharacter(AIChatAgent):
    character_id = ""
    max_persisted_messages = 80

    def initial_state(self):
        return {"turnReceipts": {}, "turnGenerations": {}}

    async def on_start(self):
        await ensure_mcp_connection(self)

    async def on_chat_message(self, options: ChatOptions):
        if options.aborted:
            return

        message_id, player_message = conversation.latest_player_message(self.messages)
        parent = await self.parent_agent(FarmGame)
        snapshot = conversation.runtime_value(await parent.world_snapshot())
        turn_generations = dict(self.state.get("turnGenerations") or {})
        accepted_generation = turn_generations.get(message_id)
        receipt_generation = (
            accepted_generation
            if accepted_generation is not None
            else snapshot["generation"]
        )
        receipt = conversation.turn_receipt(self.state, message_id, receipt_generation)
        if receipt is not None:
            reply, tool_name = receipt
            complete = f"{reply}  [MCP / {tool_name}]"
            if options.continuation:
                prior_text = conversation.latest_assistant_text(self.messages)
                if complete.startswith(prior_text):
                    remainder = complete[len(prior_text) :]
                    if remainder:
                        yield remainder
                return
            yield complete
            return

        if accepted_generation is None:
            accepted_generation = snapshot["generation"]
            turn_generations[message_id] = accepted_generation
            while len(turn_generations) > conversation.MAX_TURN_RECEIPTS:
                del turn_generations[next(iter(turn_generations))]
            next_state = deepcopy(self.state)
            next_state["turnGenerations"] = turn_generations
            self.set_state(next_state)
        elif accepted_generation != snapshot["generation"]:
            yield "That conversation belonged to the previous farm. Please ask again."
            return

        character = snapshot["characters"][self.character_id]
        if not world.near(snapshot["player"], character):
            reply = (
                f'{character["name"]} calls, "Come a little closer so I can hear you."'
            )
            next_state = deepcopy(self.state)
            conversation.remember_turn(
                next_state,
                message_id,
                reply,
                "proximity_check",
                accepted_generation,
            )
            self.set_state(next_state)
            yield reply
            return

        connection = self.mcp.get_connection(mcp_server.MCP_SERVER_ID)
        if connection is None or connection.state != MCPConnectionState.READY:
            try:
                await self.mcp.establish_connection(mcp_server.MCP_SERVER_ID)
            except Exception:  # noqa: BLE001
                yield (
                    f'{character["name"]} checks an empty tool satchel. "Try me again."'
                )
                return

        action, target = conversation.choose_action(player_message)
        if action == "talk":
            action = None
        tool_name = "inspect_farm" if action in (None, "inspect") else "act_on_farm"
        narration = ""

        if tool_name == "inspect_farm":
            for attempt in range(2):
                attempt_narration = ""
                result = await self.mcp.call_tool(
                    mcp_server.MCP_SERVER_ID,
                    tool_name,
                    {"state": snapshot},
                )
                structured = result.get("structuredContent")
                if isinstance(structured, Mapping):
                    report = structured.get("narration")
                    if isinstance(report, str):
                        attempt_narration = report
                guard = conversation.runtime_value(
                    await parent.validate_character_turn(
                        self.character_id,
                        accepted_generation,
                        snapshot["revision"],
                    )
                )
                if guard.get("valid"):
                    narration = attempt_narration
                    break
                if guard.get("retry") and attempt == 0:
                    snapshot = guard["state"]
                    continue
                narration = str(guard["reason"])
                tool_name = "world_guard"
                snapshot = conversation.runtime_value(await parent.world_snapshot())
                break
        else:
            for _ in range(2):
                generation = snapshot["generation"]
                revision = snapshot["revision"]
                result = await self.mcp.call_tool(
                    mcp_server.MCP_SERVER_ID,
                    tool_name,
                    {
                        "state": snapshot,
                        "action": action,
                        "target": target,
                        "actor": self.character_id,
                    },
                )
                structured = result.get("structuredContent")
                if not isinstance(structured, Mapping):
                    break
                candidate = structured.get("state")
                report = structured.get("narration")
                if not isinstance(candidate, Mapping) or not isinstance(report, str):
                    break
                commit = conversation.runtime_value(
                    await parent.commit_character_action(
                        self.character_id,
                        message_id,
                        generation,
                        revision,
                        action,
                        target,
                    )
                )
                if commit.get("applied"):
                    narration = str(commit["narration"])
                    snapshot = commit["state"]
                    break
                if commit.get("cancelled"):
                    narration = str(commit["reason"])
                    tool_name = "world_guard"
                    snapshot = commit["state"]
                    break
                snapshot = commit["state"]

        if not narration:
            narration = "The farm tools rustle, but return no clear answer."
        report_revision = snapshot["revision"]
        snapshot = conversation.runtime_value(await parent.world_snapshot())
        if snapshot["generation"] != accepted_generation:
            yield "That conversation belonged to the previous farm. Please ask again."
            return
        if snapshot["revision"] != report_revision:
            yield "The world moved beneath your feet. Please ask again."
            return
        reply = await conversation.model_character_reply(
            self.env.AI,
            self.character_id,
            message_id,
            player_message,
            narration,
            snapshot,
            self.messages,
            acted=tool_name == "act_on_farm",
        )
        current = conversation.runtime_value(await parent.world_snapshot())
        if options.aborted:
            return
        if current["generation"] != accepted_generation:
            yield "That conversation belonged to the previous farm. Please ask again."
            return
        if current["revision"] != snapshot["revision"]:
            yield "The world moved beneath your feet. Please ask again."
            return
        next_state = deepcopy(self.state)
        conversation.remember_turn(
            next_state,
            message_id,
            reply,
            tool_name,
            accepted_generation,
        )
        self.set_state(next_state)
        yield reply
        yield f"  [MCP / {tool_name}]"


class MiraAgent(ValleyCharacter):
    character_id = "mira"


class BrambleAgent(ValleyCharacter):
    character_id = "bramble"


class NoriAgent(ValleyCharacter):
    character_id = "nori"


class TansyAgent(ValleyCharacter):
    character_id = "tansy"
