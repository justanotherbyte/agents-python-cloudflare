from __future__ import annotations

from collections.abc import Callable

import pytest

import agents.context as context_module
from agents.context import ContextBlocks, ContextConfig


class ReadonlyProvider:
    def __init__(self, value: str | None) -> None:
        self.value = value
        self.init_labels = []

    def init(self, label: str) -> None:
        self.init_labels.append(label)

    async def get(self) -> str | None:
        return self.value


class MemoryProvider(ReadonlyProvider):
    async def set(self, content: str) -> None:
        self.value = content


class FailingProvider(MemoryProvider):
    async def set(self, content: str) -> None:
        raise RuntimeError("write failed")


class MemorySearchProvider(ReadonlyProvider):
    def __init__(self, entries: dict[str, str] | None = None) -> None:
        super().__init__(None)
        self.entries = dict(entries or {})

    async def get(self) -> str | None:
        if not self.entries:
            return None
        return f"{len(self.entries)} entries indexed."

    async def search(self, query: str) -> str | None:
        matches = [
            f"[{key}]\n{content}"
            for key, content in self.entries.items()
            if query.lower() in content.lower()
        ]
        return "\n\n".join(matches) or None

    async def set(self, key: str, content: str) -> None:
        self.entries[key] = content


class ReadonlySearchProvider(MemorySearchProvider):
    set: None = None


def _factory(providers: dict[str, MemoryProvider]) -> Callable[[str], MemoryProvider]:
    def create(label: str) -> MemoryProvider:
        provider = MemoryProvider(None)
        providers[label] = provider
        return provider

    return create


def test_context_module_exports_only_the_public_concern_surface():
    assert set(context_module.__all__) == {
        "AgentContextProvider",
        "AgentSearchProvider",
        "ContextBlock",
        "ContextBlocks",
        "ContextConfig",
        "ContextProvider",
        "SearchProvider",
        "SqlProvider",
        "WritableContextProvider",
    }


@pytest.mark.asyncio
async def test_prompt_rendering_freezing_and_refresh_are_exact():
    soul = ReadonlyProvider("You are helpful.")
    memory = MemoryProvider("likes TypeScript")
    blocks = ContextBlocks(
        [
            ContextConfig(label="soul", provider=soul),
            ContextConfig(
                label="memory",
                description="Facts",
                max_tokens=1_100,
                provider=memory,
            ),
        ]
    )

    frozen = await blocks.freeze_system_prompt()

    separator = "═" * 46
    assert frozen == (
        f"{separator}\nSOUL [readonly]\n{separator}\nYou are helpful.\n\n"
        f"{separator}\nMEMORY (Facts) [0% — 4/1100 tokens] [writable]\n"
        f"{separator}\nlikes TypeScript"
    )
    await blocks.set_block("memory", "likes Workers")
    assert await blocks.freeze_system_prompt() == frozen
    assert "likes Workers" in await blocks.refresh_system_prompt()
    assert soul.init_labels == ["soul", "soul"]
    assert memory.init_labels == ["memory", "memory"]


@pytest.mark.asyncio
async def test_empty_persisted_prompt_wins_without_loading_blocks():
    prompt_store = MemoryProvider(None)
    first = ContextBlocks([], prompt_store)

    assert await first.freeze_system_prompt() == ""
    assert await prompt_store.get() == ""

    provider = ReadonlyProvider("must not load")
    second = ContextBlocks(
        [ContextConfig(label="new", provider=provider)],
        prompt_store,
    )
    assert await second.freeze_system_prompt() == ""
    assert provider.init_labels == []


@pytest.mark.asyncio
async def test_default_providers_are_retained_and_constructor_inputs_are_copied():
    providers = {}
    configs = [ContextConfig(label="memory")]
    blocks = ContextBlocks(configs, default_provider=_factory(providers))
    configs.append(ContextConfig(label="late"))

    await blocks.load()
    await blocks.set_block("memory", "remembered")
    await blocks.load()

    assert [block.label for block in blocks.get_blocks()] == ["memory"]
    assert providers["memory"].value == "remembered"
    assert providers["memory"].init_labels == ["memory", "memory"]


@pytest.mark.asyncio
async def test_block_mutations_preserve_order_and_do_not_change_frozen_snapshot():
    first = MemoryProvider("one")
    second = MemoryProvider("two")
    blocks = ContextBlocks(
        [
            ContextConfig(label="first", provider=first),
            ContextConfig(label="second", provider=second),
        ]
    )
    frozen = await blocks.freeze_system_prompt()

    await blocks.append_to_block("first", "more")
    added = await blocks.add_block(
        ContextConfig(label="third", provider=MemoryProvider("three"))
    )
    assert added.content == "three"
    assert blocks.remove_block("second") is True
    assert blocks.remove_block("missing") is False
    assert [block.label for block in blocks.get_blocks()] == ["first", "third"]
    assert first.value == "one\nmore"
    assert await blocks.freeze_system_prompt() == frozen
    assert "one\nmore" in await blocks.refresh_system_prompt()
    assert "SECOND" not in await blocks.refresh_system_prompt()


@pytest.mark.asyncio
async def test_readonly_limits_and_failed_writes_have_target_state_ordering():
    failing = FailingProvider("")
    blocks = ContextBlocks(
        [
            ContextConfig(label="soul", provider=ReadonlyProvider("identity")),
            ContextConfig(label="limited", max_tokens=2, provider=MemoryProvider("")),
            ContextConfig(label="failing", provider=failing),
        ]
    )
    await blocks.load()

    with pytest.raises(ValueError, match="readonly"):
        await blocks.set_block("soul", "changed")
    with pytest.raises(ValueError, match="exceeds max_tokens"):
        await blocks.set_block("limited", "one two three")
    with pytest.raises(RuntimeError, match="write failed"):
        await blocks.set_block("failing", "visible before persistence")
    assert blocks.get_block("failing").content == "visible before persistence"
    with pytest.raises(ValueError, match="not found"):
        await blocks.append_to_block("missing", "value")


@pytest.mark.asyncio
async def test_token_counts_and_numeric_rendering_follow_javascript():
    blocks = ContextBlocks(
        [
            ContextConfig(
                label="bom",
                max_tokens=1_100.0,
                provider=ReadonlyProvider("a\ufeffb"),
            ),
            ContextConfig(label="next-line", provider=ReadonlyProvider("a\u0085b")),
        ]
    )

    prompt = await blocks.freeze_system_prompt()

    assert blocks.get_block("bom").tokens == 3
    assert blocks.get_block("next-line").tokens == 2
    assert "3/1100 tokens" in prompt


@pytest.mark.asyncio
async def test_tools_follow_provider_capabilities_and_execute_writes_and_searches():
    memory = MemoryProvider("")
    knowledge = MemorySearchProvider()
    readonly_search = ReadonlySearchProvider({"guide": "deployment notes"})
    blocks = ContextBlocks(
        [
            ContextConfig(label="memory", description="Facts", provider=memory),
            ContextConfig(label="knowledge", provider=knowledge),
            ContextConfig(label="docs", provider=readonly_search),
        ]
    )

    tools = await blocks.tools()

    assert set(tools) == {"set_context", "search_context"}
    assert tools["set_context"].input_schema["required"] == ["label", "content"]
    assert tools["set_context"].input_schema["properties"]["label"]["enum"] == [
        "memory",
        "knowledge",
    ]
    assert (
        await tools["set_context"].execute(
            {"label": "memory", "content": "first", "action": "replace"}
        )
        == "Written to memory. Usage: 2 tokens"
    )
    assert (
        await tools["set_context"].execute(
            {"label": "memory", "content": "second", "action": "append"}
        )
        == "Written to memory. Usage: 3 tokens"
    )
    assert memory.value == "first\nsecond"

    assert (
        await tools["set_context"].execute(
            {
                "label": "knowledge",
                "content": "Deploy on Friday",
                "metadata": {"title": "Deploy Plan", "description": "ignored"},
            }
        )
        == 'Indexed "deploy-plan" in knowledge.'
    )
    assert blocks.get_block("knowledge").content == "1 entries indexed."
    assert "Deploy on Friday" in await tools["search_context"].execute(
        {"label": "knowledge", "query": "deploy"}
    )
    assert (
        await tools["set_context"].execute(
            {"label": "knowledge", "content": "😀 knowledge"}
        )
        == 'Indexed "knowledge-knsqss" in knowledge.'
    )
    assert (
        await tools["set_context"].execute(
            {
                "label": "knowledge",
                "content": "boundary",
                "metadata": {"title": "A" * 59 + "😀tail"},
            }
        )
        == f'Indexed "{"a" * 59}" in knowledge.'
    )
    assert (
        await tools["search_context"].execute(
            {"label": "knowledge", "query": "missing"}
        )
        == "No results found."
    )
    assert (
        await tools["search_context"].execute({"label": "memory", "query": "first"})
        == 'Error: "memory" is not searchable. Searchable blocks: knowledge, docs'
    )


@pytest.mark.asyncio
async def test_empty_search_blocks_render_and_readonly_search_has_no_write_tool():
    blocks = ContextBlocks(
        [ContextConfig(label="docs", provider=ReadonlySearchProvider())]
    )

    prompt = await blocks.freeze_system_prompt()
    tools = await blocks.tools()

    assert "DOCS [searchable]" in prompt
    assert set(tools) == {"search_context"}
