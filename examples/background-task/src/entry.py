from agents import Agent, route_agent_request, rpc_callable
from agents.tasks import TaskStep, task_definition
from workers import Response, WorkerEntrypoint


class Reports(Agent):
    def initial_state(self):
        return {"last_report": None}

    @task_definition()
    async def build_report(self, input: dict[str, int], step: TaskStep):
        @step.do("calculate")
        async def calculate():
            return input["value"] * 2

        report = {"value": input["value"], "doubled": await calculate()}

        @step.do("publish")
        async def publish():
            self.set_state({"last_report": report})
            return report

        return await publish()

    @rpc_callable()
    async def start_report(self, job_id: str, value: int) -> str:
        receipt = await self.tasks.run(
            "build_report",
            {"value": value},
            idempotency_key=f"report:{job_id}",
        )
        return receipt.run_id


class Default(WorkerEntrypoint):
    async def fetch(self, request):
        return await route_agent_request(request, self.env) or Response(
            "Not Found", status=404
        )
