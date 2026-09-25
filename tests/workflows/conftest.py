from __future__ import annotations

import workers


if not hasattr(workers, "WorkflowEntrypoint"):

    class WorkflowEntrypoint:
        def __init__(self, ctx: object, env: object) -> None:
            self.ctx = ctx
            self.env = env

    workers.WorkflowEntrypoint = WorkflowEntrypoint
