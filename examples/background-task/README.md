# Replayable background task

`Reports` accepts work through an RPC method and returns as soon as the Task run
is durably recorded. The Task journals each named step, so completed steps are
reused if execution restarts. Its final step publishes the result through Agent
state.

Run the Worker:

```bash
uv sync
uv run pywrangler dev
```

Start a report from a React application with `agents@0.22.0`:

```tsx
import { useAgent } from "agents/react";

type ReportState = {
  last_report: { value: number; doubled: number } | null;
};

export function ReportButton({ jobId }: { jobId: string }) {
  const agent = useAgent<ReportState>({ agent: "Reports", name: "main" });

  return (
    <button
      onClick={() => void agent.stub.start_report(jobId, 21)}
    >
      Result: {agent.state?.last_report?.doubled ?? "not started"}
    </button>
  );
}
```

Generate one stable job ID for each logical report and pass that same ID when
retrying its submission. The idempotency key then joins the existing Task run
instead of creating duplicate work.
