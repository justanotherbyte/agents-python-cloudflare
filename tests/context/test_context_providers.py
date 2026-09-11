from __future__ import annotations

import types

import fakes
import pytest

from agents import Agent
from agents.context import AgentContextProvider, AgentSearchProvider


class Sql:
    def __init__(self, ctx: fakes.FakeCtx) -> None:
        self._sql = ctx.storage.sql

    def sql(self, query: str, *params: object) -> list[dict[str, object]]:
        return self._sql.exec(query, *params).toArray()


def _tables(ctx: fakes.FakeCtx) -> set[str]:
    return {
        row["name"]
        for row in ctx.storage.sql.exec(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).toArray()
    }


@pytest.mark.asyncio
async def test_context_provider_is_lazy_upserts_and_keeps_labels_isolated():
    ctx = fakes.FakeCtx()
    sql = Sql(ctx)
    soul = AgentContextProvider(sql, "soul")
    memory = AgentContextProvider(sql)
    soul.init("ignored")
    memory.init("memory")

    assert "cf_agents_context_blocks" not in _tables(ctx)
    assert await memory.get() is None
    assert "cf_agents_context_blocks" in _tables(ctx)
    await soul.set("identity")
    await memory.set("facts")
    await memory.set("updated facts")

    assert await soul.get() == "identity"
    assert await memory.get() == "updated facts"
    assert ctx.storage.sql.exec(
        "SELECT label, content FROM cf_agents_context_blocks ORDER BY label"
    ).toArray() == [
        {"label": "memory", "content": "updated facts"},
        {"label": "soul", "content": "identity"},
    ]
    ddl = ctx.storage.sql.exec(
        "SELECT sql FROM sqlite_master WHERE name = 'cf_agents_context_blocks'"
    ).toArray()[0]["sql"]
    assert " ".join(ddl.split()) == (
        "CREATE TABLE cf_agents_context_blocks ( label TEXT PRIMARY KEY, "
        "content TEXT NOT NULL, updated_at DATETIME DEFAULT CURRENT_TIMESTAMP )"
    )


@pytest.mark.asyncio
async def test_context_providers_accept_the_agent_sql_surface_directly():
    ctx = fakes.FakeCtx()
    agent = Agent(ctx, types.SimpleNamespace())
    context = AgentContextProvider(agent)
    search = AgentSearchProvider(agent)
    context.init("memory")
    search.init("knowledge")

    await context.set("durable facts")
    await search.set("guide", "searchable facts")

    assert await context.get() == "durable facts"
    assert await search.search("searchable") == "[guide]\nsearchable facts"


@pytest.mark.asyncio
async def test_search_provider_owns_fts_replaces_entries_and_isolates_labels():
    ctx = fakes.FakeCtx()
    ctx.storage.sql.exec("CREATE TABLE cf_agents_search_entries (id TEXT)")
    sql = Sql(ctx)
    docs = AgentSearchProvider(sql)
    notes = AgentSearchProvider(sql)
    docs.init("docs")
    notes.init("notes")

    assert "cf_agents_search_fts" not in _tables(ctx)
    assert await docs.get() is None
    assert "cf_agents_search_fts" in _tables(ctx)
    assert "cf_agents_search_entries" not in _tables(ctx)
    await docs.set("readme", "Durable Objects keep state on the edge")
    await docs.set("guide", "Workers run close to the user")
    await notes.set("private", "Durable notes")

    assert await docs.get() == "2 entries indexed."
    assert await notes.get() == "1 entries indexed."
    assert await docs.search("durable") == (
        "[readme]\nDurable Objects keep state on the edge"
    )
    assert await docs.search("durable\ufeffobjects") == (
        "[readme]\nDurable Objects keep state on the edge"
    )
    assert await notes.search("workers") is None
    await docs.set("readme", "rewritten body")
    assert await docs.get() == "2 entries indexed."
    assert await docs.search("durable") is None


@pytest.mark.asyncio
async def test_search_quotes_terms_limits_results_and_swallows_query_failures(
    monkeypatch,
):
    ctx = fakes.FakeCtx()
    provider = AgentSearchProvider(Sql(ctx))
    provider.init("knowledge")
    for index in range(12):
        await provider.set(f"entry-{index}", f'say "hello" shared {index}')

    results = await provider.search('"hello" shared')
    assert results is not None
    assert results.count("[entry-") == 10
    assert await provider.search("   ") is None
    execute = ctx.storage.sql.exec

    def fail_search(query, *params):
        if "MATCH" in query:
            raise RuntimeError("query failed")
        return execute(query, *params)

    monkeypatch.setattr(ctx.storage.sql, "exec", fail_search)
    assert await provider.search("shared") is None
