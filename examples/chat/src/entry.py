from agents import AIChatAgent, ChatOptions, route_agent_request
from workers import Response, WorkerEntrypoint


class Assistant(AIChatAgent):
    max_persisted_messages = 100

    async def on_chat_message(self, options: ChatOptions):
        if options.aborted:
            return

        yield "Hello"
        yield " from a Python Agent."


class Default(WorkerEntrypoint):
    async def fetch(self, request):
        return await route_agent_request(request, self.env) or Response(
            "Not Found", status=404
        )
