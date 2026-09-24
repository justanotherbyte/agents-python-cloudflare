from agents import route_agent_request
from workers import Response, WorkerEntrypoint

from mosslight.mcp_server import FarmTools
from mosslight.valley_agents import (
    BrambleAgent,
    FarmGame,
    MiraAgent,
    NoriAgent,
    TansyAgent,
)

__all__ = [
    "BrambleAgent",
    "Default",
    "FarmGame",
    "FarmTools",
    "MiraAgent",
    "NoriAgent",
    "TansyAgent",
]


class Default(WorkerEntrypoint):
    async def fetch(self, request):
        return await route_agent_request(request, self.env) or Response(
            "Not Found", status=404
        )
