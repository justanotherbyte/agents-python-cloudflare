# Chat agent

`Assistant` streams two text chunks for each user message. The SDK persists the
conversation and lets a reconnecting client replay chunks stored before a
connection drop.

Run the Worker:

```bash
uv sync
uv run pywrangler dev
```

Connect from a React application with `agents@0.22.0`,
`@cloudflare/ai-chat@0.10.1`, and their peer dependencies:

```tsx
import { useAgentChat } from "@cloudflare/ai-chat/react";
import { useAgent } from "agents/react";

export function ChatButton() {
  const agent = useAgent({ agent: "Assistant", name: "main" });
  const { messages, sendMessage, status } = useAgentChat({ agent });

  return (
    <>
      <button
        disabled={status !== "ready"}
        onClick={() => void sendMessage({ text: "Hello" })}
      >
        Send message
      </button>
      <pre>{JSON.stringify(messages, null, 2)}</pre>
    </>
  );
}
```

The `main` conversation is available at `/agents/assistant/main`.
