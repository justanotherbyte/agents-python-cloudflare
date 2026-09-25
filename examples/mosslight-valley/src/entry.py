# ruff: noqa: F401 - the Worker runtime exports these Durable Object classes.
from agents import route_agent_request
from mosslight.mcp_server import FarmTools
from mosslight.valley_agents import (
    BrambleAgent,
    FarmGame,
    MiraAgent,
    NoriAgent,
    TansyAgent,
)
from workers import Response, WorkerEntrypoint


class Default(WorkerEntrypoint):
    async def fetch(self, request):
        return await route_agent_request(request, self.env) or Response(
            "Not Found", status=404
        )
