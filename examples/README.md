# Examples

These small Workers show the main ways to use the Python Agents SDK:

| Example | What it shows |
| --- | --- |
| [`counter`](counter/) | Durable state and a callable RPC method |
| [`chat`](chat/) | Persisted, resumable chat responses |
| [`background-task`](background-task/) | Replayable background work with durable steps |

Each directory is an independent Worker project. Its `pyproject.toml` resolves
`agents-py` from this repository as a non-editable package, so you can run an
example directly from a checkout:

```bash
cd examples/counter
uv sync
uv run pywrangler dev
```

The browser snippets use `agents` 0.22.0. The chat example additionally uses
`@cloudflare/ai-chat` 0.10.1.

These examples leave the Agent routes unauthenticated to keep the SDK usage
clear. Before deployment, add authentication, validate WebSocket `Origin`, and
authorize each requested Agent instance name so users cannot open another
user's state or conversation.
