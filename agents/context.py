from __future__ import annotations

import inspect
import math
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Protocol, TypedDict, cast

from ._context_shaping import _shape_messages
from ._session_json import dumps_session_json
from .sessions import SessionMessage

_JS_WHITESPACE = re.compile(
    "[\\u0009-\\u000d\\u0020\\u00a0\\u1680\\u2000-\\u200a"
    "\\u2028\\u2029\\u202f\\u205f\\u3000\\ufeff]+"
)


class ContextProvider(Protocol):
    async def get(self) -> str | None: ...


class WritableContextProvider(ContextProvider, Protocol):
    async def set(self, content: str) -> None: ...


class SearchProvider(ContextProvider, Protocol):
    async def search(self, query: str) -> str | None: ...


class SqlProvider(Protocol):
    def sql(self, query: str, *params: object) -> list[dict[str, object]]: ...


@dataclass(frozen=True, kw_only=True)
class ContextConfig:
    label: str
    description: str | None = None
    max_tokens: int | float | None = None
    provider: ContextProvider | None = None


@dataclass
class ContextBlock:
    label: str
    description: str | None
    content: str
    tokens: int
    max_tokens: int | float | None
    writable: bool
    is_searchable: bool


class _ModelContext(TypedDict):
    system: str
    messages: list[SessionMessage]


class _ContextTool:
    def __init__(
        self,
        description: str,
        input_schema: dict[str, object],
        execute: Callable[[Mapping[str, object]], Awaitable[str]],
    ) -> None:
        self.description = description
        self.input_schema = input_schema
        self._execute = execute

    async def execute(self, arguments: Mapping[str, object]) -> str:
        return await self._execute(arguments)


class ContextBlocks:
    """Compose labelled provider content into one durable frozen prompt."""

    def __init__(
        self,
        configs: Sequence[ContextConfig],
        prompt_store: WritableContextProvider | None = None,
        default_provider: Callable[[str], ContextProvider] | None = None,
    ) -> None:
        self._configs = list(configs)
        self._blocks: dict[str, ContextBlock] = {}
        self._snapshot: str | None = None
        self._loaded = False
        self._prompt_store = prompt_store
        self._default_provider = default_provider

    async def load(self) -> None:
        self._configs = [
            self._with_default_provider(config) for config in self._configs
        ]
        for config in self._configs:
            self._blocks[config.label] = await self._load_block(config)
        self._loaded = True

    async def add_block(self, config: ContextConfig) -> ContextBlock:
        if not self._loaded:
            await self.load()
        if any(existing.label == config.label for existing in self._configs):
            raise ValueError(f'Block "{config.label}" already exists')
        configured = self._with_default_provider(config)
        self._configs.append(configured)
        block = await self._load_block(configured)
        self._blocks[configured.label] = block
        return block

    def remove_block(self, label: str) -> bool:
        for index, config in enumerate(self._configs):
            if config.label != label:
                continue
            self._configs.pop(index)
            self._blocks.pop(label, None)
            return True
        return False

    def get_block(self, label: str) -> ContextBlock | None:
        return self._blocks.get(label)

    def get_blocks(self) -> list[ContextBlock]:
        return list(self._blocks.values())

    async def set_block(self, label: str, content: str) -> ContextBlock:
        if not self._loaded:
            await self.load()
        config = self._config(label)
        existing = self._blocks.get(label)
        if existing is None or not existing.writable:
            raise ValueError(f'Block "{label}" is readonly')
        if existing.is_searchable:
            raise ValueError(
                f'Block "{label}" is a keyed provider. Use set_search_entry() instead.'
            )

        tokens = _estimate_tokens(content)
        max_tokens = config.max_tokens if config is not None else existing.max_tokens
        if max_tokens is not None and tokens > max_tokens:
            raise ValueError(
                f'Block "{label}" exceeds max_tokens: {tokens} > {max_tokens}'
            )
        block = ContextBlock(
            label=label,
            description=(
                config.description if config is not None else existing.description
            ),
            content=content,
            tokens=tokens,
            max_tokens=max_tokens,
            writable=True,
            is_searchable=False,
        )
        self._blocks[label] = block
        if config is not None and _is_writable(config.provider):
            await cast(WritableContextProvider, config.provider).set(content)
        return block

    async def append_to_block(self, label: str, content: str) -> ContextBlock:
        if not self._loaded:
            await self.load()
        existing = self._blocks.get(label)
        if existing is None:
            raise ValueError(f'Block "{label}" not found')
        separator = "\n" if existing.content and not content.startswith("\n") else ""
        return await self.set_block(label, existing.content + separator + content)

    async def freeze_system_prompt(self) -> str:
        if self._prompt_store is not None:
            stored = await self._prompt_store.get()
            if stored is not None:
                return stored
        if self._snapshot is not None:
            return self._snapshot
        if not self._loaded:
            await self.load()
        self._snapshot = self._render_prompt()
        if self._prompt_store is not None:
            await self._prompt_store.set(self._snapshot)
        return self._snapshot

    async def refresh_system_prompt(self) -> str:
        self._loaded = False
        await self.load()
        self._snapshot = self._render_prompt()
        if self._prompt_store is not None:
            await self._prompt_store.set(self._snapshot)
        return self._snapshot

    async def assemble(
        self,
        messages: Sequence[SessionMessage],
        *,
        keep_recent: int = 4,
        max_tool_output_chars: int = 500,
        max_text_chars: int = 10_000,
    ) -> _ModelContext:
        """Combine the frozen prompt with non-destructively shaped history."""
        system = await self.freeze_system_prompt()
        shaped = _shape_messages(
            cast(Sequence[dict[str, object]], messages),
            keep_recent=keep_recent,
            max_tool_output_chars=max_tool_output_chars,
            max_text_chars=max_text_chars,
        )
        return _ModelContext(
            system=system,
            messages=cast(list[SessionMessage], shaped),
        )

    async def tools(self) -> dict[str, _ContextTool]:
        if not self._loaded:
            await self.load()
        blocks = list(self._blocks.values())
        writable = [block for block in blocks if block.writable]
        searchable = [block.label for block in blocks if block.is_searchable]
        tools = {}
        if writable:
            tools["set_context"] = self._set_context_tool(writable)
        if searchable:
            tools["search_context"] = self._search_context_tool(searchable)
        return tools

    def _with_default_provider(self, config: ContextConfig) -> ContextConfig:
        if config.provider is not None or self._default_provider is None:
            return config
        return replace(config, provider=self._default_provider(config.label))

    async def _load_block(self, config: ContextConfig) -> ContextBlock:
        provider = config.provider
        init = getattr(provider, "init", None)
        if callable(init):
            initialized = init(config.label)
            if inspect.isawaitable(initialized):
                close = getattr(initialized, "close", None)
                if callable(close):
                    close()
                raise TypeError("context provider init must be synchronous")
        content = "" if provider is None else (await provider.get()) or ""
        searchable = _is_searchable(provider)
        return ContextBlock(
            label=config.label,
            description=config.description,
            content=content,
            tokens=_estimate_tokens(content),
            max_tokens=config.max_tokens,
            writable=_is_writable(provider),
            is_searchable=searchable,
        )

    def _config(self, label: str) -> ContextConfig | None:
        return next((config for config in self._configs if config.label == label), None)

    def _render_prompt(self) -> str:
        separator = "═" * 46
        rendered = []
        for block in self._blocks.values():
            if not block.content and not block.writable and not block.is_searchable:
                continue
            header = block.label.upper()
            if block.description:
                header += f" ({block.description})"
            if block.max_tokens:
                percent = _js_round(block.tokens / block.max_tokens * 100)
                header += (
                    f" [{_js_number(percent)}% — {block.tokens}/"
                    f"{_js_number(block.max_tokens)} tokens]"
                )
            if block.is_searchable:
                header += " [searchable]"
            elif not block.writable:
                header += " [readonly]"
            else:
                header += " [writable]"
            rendered.append(f"{separator}\n{header}\n{separator}\n{block.content}")
        return "\n\n".join(rendered)

    async def _set_search_entry(self, label: str, key: str, content: str) -> None:
        if not self._loaded:
            await self.load()
        config = self._config(label)
        existing = self._blocks.get(label)
        if existing is None or not existing.is_searchable:
            raise ValueError(f'Block "{label}" is not a search provider')
        provider = None if config is None else config.provider
        setter = getattr(provider, "set", None)
        if not _is_searchable(provider) or not callable(setter):
            raise ValueError(f'Block "{label}" does not support writes')
        await setter(key, content)
        summary = await cast(SearchProvider, provider).get()
        existing.content = summary or ""
        existing.tokens = _estimate_tokens(existing.content)

    async def _search_context(self, label: str, query: str) -> str | None:
        if not self._loaded:
            await self.load()
        config = self._config(label)
        provider = None if config is None else config.provider
        if not _is_searchable(provider):
            raise ValueError(f'Block "{label}" is not a search provider')
        return await cast(SearchProvider, provider).search(query)

    def _set_context_tool(self, writable: list[ContextBlock]) -> _ContextTool:
        descriptions = []
        keyed = []
        for block in writable:
            kind = "searchable, keyed entries" if block.is_searchable else "writable"
            description = block.description or "no description"
            descriptions.append(f'- "{block.label}" ({kind}): {description}')
            if block.is_searchable:
                keyed.append(block)
        properties: dict[str, object] = {
            "label": {
                "type": "string",
                "enum": [block.label for block in writable],
                "description": "Block label to write to",
            },
            "content": {
                "type": "string",
                "description": "The main content to write to the block.",
            },
            "action": {
                "type": "string",
                "enum": ["replace", "append"],
                "description": "replace (default) or append",
            },
        }
        if keyed:
            labels = ", ".join(f'"{block.label}"' for block in keyed)
            properties["metadata"] = {
                "type": "object",
                "description": (
                    "Optional metadata for keyed entries (searchable blocks: "
                    f"{labels}). A title keeps updates stable; a description helps "
                    "the model pick the right entry."
                ),
                "properties": {
                    "title": {
                        "type": "string",
                        "description": (
                            "Short title. Used as a stable identifier — entries with "
                            "the same title are updated in place, different titles "
                            "create new entries."
                        ),
                    },
                    "description": {
                        "type": "string",
                        "description": (
                            "One-line summary shown alongside the title in the system "
                            "prompt so the model can decide when to load the entry."
                        ),
                    },
                },
            }
        metadata_hint = ""
        if keyed:
            metadata_hint = (
                "\n\nFor searchable blocks, pass `metadata: { title, description }` "
                "— title stabilises updates, description helps the model pick "
                "entries. Metadata is optional."
            )
        description = (
            "Write to a context block. Available blocks:\n"
            + "\n".join(descriptions)
            + "\n\nWrites are durable and persist across sessions."
            + metadata_hint
        )

        async def execute(arguments: Mapping[str, object]) -> str:
            try:
                label = cast(str, arguments["label"])
                content = cast(str, arguments["content"])
                block = self._blocks.get(label)
                if block is None:
                    return f'Error: block "{label}" not found'
                if block.is_searchable:
                    metadata = arguments.get("metadata")
                    title = (
                        metadata.get("title") if isinstance(metadata, Mapping) else None
                    )
                    key = _context_entry_key(
                        title if isinstance(title, str) else None,
                        content,
                    )
                    await self._set_search_entry(label, key, content)
                    return f'Indexed "{key}" in {label}.'
                updated = (
                    await self.append_to_block(label, content)
                    if arguments.get("action") == "append"
                    else await self.set_block(label, content)
                )
                if updated.max_tokens:
                    percent = _js_round(updated.tokens / updated.max_tokens * 100)
                    usage = (
                        f"{_js_number(percent)}% "
                        f"({updated.tokens}/{_js_number(updated.max_tokens)} tokens)"
                    )
                else:
                    usage = f"{updated.tokens} tokens"
                return f"Written to {label}. Usage: {usage}"
            except Exception as error:
                return f"Error: {error}"

        return _ContextTool(
            description,
            {
                "type": "object",
                "properties": properties,
                "required": ["label", "content"],
            },
            execute,
        )

    def _search_context_tool(self, labels: list[str]) -> _ContextTool:
        quoted = ", ".join(f'"{label}"' for label in labels)
        description = (
            "Search for information in a searchable context block. "
            f"ONLY these blocks are searchable: {quoted}. "
            "Other blocks cannot be searched."
        )

        async def execute(arguments: Mapping[str, object]) -> str:
            try:
                label = cast(str, arguments["label"])
                query = cast(str, arguments["query"])
                if label not in labels:
                    return (
                        f'Error: "{label}" is not searchable. '
                        f"Searchable blocks: {', '.join(labels)}"
                    )
                result = await self._search_context(label, query)
                return result or "No results found."
            except Exception as error:
                return f"Error: {error}"

        return _ContextTool(
            description,
            {
                "type": "object",
                "properties": {
                    "label": {
                        "type": "string",
                        "enum": labels,
                        "description": "Searchable block label",
                    },
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["label", "query"],
            },
            execute,
        )


class AgentContextProvider:
    """Store one context block lazily in Durable Object SQLite."""

    def __init__(self, sql: SqlProvider, label: str | None = None) -> None:
        self._sql = sql
        self._label = "" if label is None else label
        self._initialized = False

    def init(self, label: str) -> None:
        if not self._label:
            self._label = label

    async def get(self) -> str | None:
        self._ensure_table()
        rows = self._sql.sql(
            "SELECT content FROM cf_agents_context_blocks WHERE label = ?",
            self._label,
        )
        return None if not rows else cast(str, rows[0]["content"])

    async def set(self, content: str) -> None:
        self._ensure_table()
        self._sql.sql(
            "INSERT INTO cf_agents_context_blocks (label, content) VALUES (?, ?) "
            "ON CONFLICT(label) DO UPDATE SET content = ?, "
            "updated_at = CURRENT_TIMESTAMP",
            self._label,
            content,
            content,
        )

    def _ensure_table(self) -> None:
        if self._initialized:
            return
        self._sql.sql("""
        CREATE TABLE IF NOT EXISTS cf_agents_context_blocks (
          label TEXT PRIMARY KEY,
          content TEXT NOT NULL,
          updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)
        self._initialized = True


class AgentSearchProvider:
    """Index one labelled context collection lazily with SQLite FTS5."""

    def __init__(self, sql: SqlProvider) -> None:
        self._sql = sql
        self._label = ""
        self._initialized = False

    def init(self, label: str) -> None:
        self._label = label

    async def get(self) -> str | None:
        self._ensure_table()
        rows = self._sql.sql(
            "SELECT COUNT(*) AS count FROM cf_agents_search_fts WHERE label = ?",
            self._label,
        )
        count = cast(int, rows[0]["count"]) if rows else 0
        return None if count == 0 else f"{count} entries indexed."

    async def search(self, query: str) -> str | None:
        self._ensure_table()
        terms = []
        for term in _JS_WHITESPACE.split(query):
            if term:
                escaped = term.replace('"', '""')
                terms.append(f'"{escaped}"')
        if not terms:
            return None
        try:
            rows = self._sql.sql(
                "SELECT key, content FROM cf_agents_search_fts "
                "WHERE cf_agents_search_fts MATCH ? AND label = ? "
                "ORDER BY rank LIMIT 10",
                " ".join(terms),
                self._label,
            )
        except Exception:
            return None
        if not rows:
            return None
        return "\n\n".join(f"[{row['key']}]\n{row['content']}" for row in rows)

    async def set(self, key: str, content: str) -> None:
        self._ensure_table()
        self._sql.sql(
            "DELETE FROM cf_agents_search_fts WHERE label = ? AND key = ?",
            self._label,
            key,
        )
        self._sql.sql(
            "INSERT INTO cf_agents_search_fts (label, key, content) VALUES (?, ?, ?)",
            self._label,
            key,
            content,
        )

    def _ensure_table(self) -> None:
        if self._initialized:
            return
        self._sql.sql("""
        CREATE VIRTUAL TABLE IF NOT EXISTS cf_agents_search_fts
        USING fts5(
          label UNINDEXED,
          key UNINDEXED,
          content,
          tokenize='porter unicode61'
        )
        """)
        self._sql.sql("DROP TABLE IF EXISTS cf_agents_search_entries")
        self._initialized = True


def _is_writable(provider: object | None) -> bool:
    return callable(getattr(provider, "set", None))


def _is_searchable(provider: object | None) -> bool:
    return callable(getattr(provider, "search", None))


def _estimate_tokens(content: str) -> int:
    if not content:
        return 0
    utf16_units = len(content.encode("utf-16-le", errors="surrogatepass")) // 2
    words = len([part for part in _JS_WHITESPACE.split(content) if part])
    return math.ceil(max(utf16_units / 4, words * 1.3))


def _js_round(value: float) -> int | float:
    if not math.isfinite(value):
        return value
    return math.floor(value + 0.5)


def _js_number(value: int | float) -> str:
    if isinstance(value, int):
        return str(value)
    if math.isnan(value):
        return "NaN"
    if value == math.inf:
        return "Infinity"
    if value == -math.inf:
        return "-Infinity"
    return dumps_session_json(value)


def _context_entry_key(title: str | None, content: str) -> str:
    if title is not None and _js_trim(title):
        slug = _slugify(title)
        return slug or f"entry-{_stable_hash(title)}"
    slug = _slugify(content) or "entry"
    return f"{slug}-{_stable_hash(content)}"


def _slugify(value: str) -> str:
    encoded = value.encode("utf-16-le", errors="surrogatepass")[: 60 * 2]
    prefix = encoded.decode("utf-16-le", errors="surrogatepass").lower()
    return re.sub(r"[^a-z0-9]+", "-", prefix).strip("-")


def _js_trim(value: str) -> str:
    return _JS_WHITESPACE.sub(" ", value).strip(" ")


def _stable_hash(value: str) -> str:
    encoded = value.encode("utf-16-le", errors="surrogatepass")
    hash_value = 0x811C9DC5
    for index in range(0, len(encoded), 2):
        code_unit = encoded[index] | encoded[index + 1] << 8
        hash_value ^= code_unit
        hash_value = hash_value * 0x01000193 & 0xFFFFFFFF
    return _base36(hash_value)


def _base36(value: int) -> str:
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    if value == 0:
        return "0"
    encoded = ""
    while value:
        value, remainder = divmod(value, 36)
        encoded = digits[remainder] + encoded
    return encoded


__all__ = [
    "AgentContextProvider",
    "AgentSearchProvider",
    "ContextBlock",
    "ContextBlocks",
    "ContextConfig",
    "ContextProvider",
    "SearchProvider",
    "SqlProvider",
    "WritableContextProvider",
]
