from agents import Agent, route_agent_request, rpc_callable
from workers import Response, WorkerEntrypoint


class Counter(Agent):
    def initial_state(self):
        return {"count": 0}

    @rpc_callable()
    def increment(self, amount: int = 1) -> int:
        state = self.state
        state["count"] += amount
        self.set_state(state)
        return state["count"]


class Default(WorkerEntrypoint):
    async def fetch(self, request):
        return await route_agent_request(request, self.env) or Response(
            "Not Found", status=404
        )
