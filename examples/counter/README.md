# Stateful counter

`Counter` stores a JSON state object in its Durable Object and exposes
`increment()` as an RPC method. Calling `set_state()` persists the new count and
broadcasts it to connected clients.

Run the Worker:

```bash
uv sync
uv run pywrangler dev
```

Connect from a React application with `agents@0.22.0`:

```tsx
import { useAgent } from "agents/react";

export function CounterButton() {
  const agent = useAgent<{ count: number }>({
    agent: "Counter",
    name: "main"
  });

  return (
    <button onClick={() => void agent.stub.increment(1)}>
      Count: {agent.state?.count ?? 0}
    </button>
  );
}
```

The `main` counter instance is available at `/agents/counter/main`.
