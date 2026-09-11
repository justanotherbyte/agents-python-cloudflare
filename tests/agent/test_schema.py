from __future__ import annotations

import types

import fakes
import pytest

from agents import Agent
from agents.core.agent import PARENT_PATH_ROW_ID


SCHEMA_VERSION = 11
SCHEMA_VERSION_ROW_ID = "cf_schema_version"
LEGACY_SCHEMA_VERSION_ROW_ID = "cf_py_schema_version"

CORE_TABLES = {
    "cf_agents_state",
    "cf_agents_queues",
    "cf_agents_workflows",
    "cf_agents_runs",
    "cf_agents_facet_runs",
    "cf_agents_fibers",
    "cf_agent_tool_runs",
    "cf_agents_mcp_servers",
}

CORE_INDEXES = {
    "idx_workflows_status": ["status"],
    "idx_workflows_name": ["workflow_name"],
    "idx_facet_runs_owner_path_key": ["owner_path_key"],
    "idx_fibers_status_created": ["status", "created_at", "fiber_id"],
    "idx_fibers_name_status_created": [
        "name",
        "status",
        "created_at",
        "fiber_id",
    ],
    "idx_fibers_status_completed": ["status", "completed_at", "created_at"],
    "idx_agent_tool_runs_parent_tool_call_id": [
        "parent_tool_call_id",
        "display_order",
    ],
}

CORE_COLUMNS = {
    "cf_agents_state": ["id", "state"],
    "cf_agents_queues": [
        "id",
        "payload",
        "callback",
        "created_at",
        "retry_options",
    ],
    "cf_agents_workflows": [
        "id",
        "workflow_id",
        "workflow_name",
        "status",
        "metadata",
        "error_name",
        "error_message",
        "created_at",
        "updated_at",
        "completed_at",
    ],
    "cf_agents_runs": ["id", "name", "snapshot", "created_at"],
    "cf_agents_facet_runs": [
        "owner_path",
        "owner_path_key",
        "run_id",
        "created_at",
    ],
    "cf_agents_fibers": [
        "fiber_id",
        "idempotency_key",
        "name",
        "status",
        "snapshot",
        "metadata_json",
        "error_message",
        "created_at",
        "started_at",
        "completed_at",
    ],
    "cf_agent_tool_runs": [
        "run_id",
        "parent_tool_call_id",
        "agent_type",
        "input_preview",
        "input_redacted",
        "status",
        "summary",
        "output_json",
        "error_message",
        "interrupted_reason",
        "child_still_running",
        "display_metadata",
        "display_order",
        "started_at",
        "completed_at",
        "detached",
        "detached_on_finish",
        "detached_notify_source",
        "detached_max_budget_at",
        "finish_claimed_at",
        "finish_delivered_at",
        "give_up_claimed_at",
        "give_up_delivered_at",
        "detached_no_progress_budget_ms",
        "last_progress_at",
        "detached_on_milestones",
    ],
    "cf_agents_mcp_servers": [
        "id",
        "name",
        "server_url",
        "callback_url",
        "client_id",
        "auth_url",
        "server_options",
    ],
}


def _tables(agent: Agent) -> set[str]:
    return {
        row["name"]
        for row in agent.sql("SELECT name FROM sqlite_master WHERE type = 'table'")
    }


def _columns(agent: Agent, table: str) -> list[str]:
    return [row["name"] for row in agent.sql(f"PRAGMA table_info({table})")]


def _column_info(agent: Agent, table: str) -> dict[str, dict]:
    return {row["name"]: row for row in agent.sql(f"PRAGMA table_info({table})")}


def _marker(agent: Agent, row_id: str = SCHEMA_VERSION_ROW_ID):
    rows = agent.sql(
        "SELECT state FROM cf_agents_state WHERE id = ?",
        row_id,
    )
    return rows[0]["state"] if rows else None


def _index_columns(agent: Agent, name: str) -> list[str]:
    return [row["name"] for row in agent.sql(f"PRAGMA index_info({name})")]


def _create_state_table(conn) -> None:
    conn.execute(
        "CREATE TABLE cf_agents_state (id TEXT PRIMARY KEY NOT NULL, state TEXT)"
    )


def _write_marker(conn, row_id: str, value: str) -> None:
    conn.execute(
        "INSERT INTO cf_agents_state (id, state) VALUES (?, ?)",
        (row_id, value),
    )


def _create_typescript_v11_fixture(conn) -> None:
    conn.executescript(
        """
        CREATE TABLE cf_agents_state (
            id TEXT PRIMARY KEY NOT NULL,
            state TEXT
        );
        CREATE TABLE cf_agents_queues (
            id TEXT PRIMARY KEY NOT NULL,
            payload TEXT,
            callback TEXT,
            created_at INTEGER DEFAULT (unixepoch()),
            retry_options TEXT
        );
        CREATE TABLE cf_agents_workflows (
            id TEXT PRIMARY KEY NOT NULL,
            workflow_id TEXT NOT NULL UNIQUE,
            workflow_name TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN (
                'queued', 'running', 'paused', 'errored', 'terminated',
                'complete', 'waiting', 'waitingForPause', 'unknown'
            )),
            metadata TEXT,
            error_name TEXT,
            error_message TEXT,
            created_at INTEGER NOT NULL DEFAULT (unixepoch()),
            updated_at INTEGER NOT NULL DEFAULT (unixepoch()),
            completed_at INTEGER
        );
        CREATE INDEX idx_workflows_status
            ON cf_agents_workflows(status);
        CREATE INDEX idx_workflows_name
            ON cf_agents_workflows(workflow_name);
        CREATE TABLE cf_agents_runs (
            id TEXT PRIMARY KEY NOT NULL,
            name TEXT NOT NULL,
            snapshot TEXT,
            created_at INTEGER NOT NULL
        );
        CREATE TABLE cf_agents_facet_runs (
            owner_path TEXT NOT NULL,
            owner_path_key TEXT NOT NULL,
            run_id TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            PRIMARY KEY (owner_path_key, run_id)
        );
        CREATE INDEX idx_facet_runs_owner_path_key
            ON cf_agents_facet_runs(owner_path_key);
        CREATE TABLE cf_agents_fibers (
            fiber_id TEXT PRIMARY KEY,
            idempotency_key TEXT UNIQUE,
            name TEXT NOT NULL,
            status TEXT NOT NULL,
            snapshot TEXT,
            metadata_json TEXT,
            error_message TEXT,
            created_at INTEGER NOT NULL,
            started_at INTEGER,
            completed_at INTEGER
        );
        CREATE INDEX idx_fibers_status_created
            ON cf_agents_fibers(status, created_at, fiber_id);
        CREATE INDEX idx_fibers_name_status_created
            ON cf_agents_fibers(name, status, created_at, fiber_id);
        CREATE INDEX idx_fibers_status_completed
            ON cf_agents_fibers(status, completed_at, created_at);
        CREATE TABLE cf_agent_tool_runs (
            run_id TEXT PRIMARY KEY,
            parent_tool_call_id TEXT,
            agent_type TEXT NOT NULL,
            input_preview TEXT,
            input_redacted INTEGER NOT NULL DEFAULT 1,
            status TEXT NOT NULL,
            summary TEXT,
            output_json TEXT,
            error_message TEXT,
            interrupted_reason TEXT,
            child_still_running INTEGER,
            display_metadata TEXT,
            display_order INTEGER NOT NULL DEFAULT 0,
            started_at INTEGER NOT NULL,
            completed_at INTEGER,
            detached INTEGER NOT NULL DEFAULT 0,
            detached_on_finish TEXT,
            detached_notify_source TEXT,
            detached_max_budget_at INTEGER,
            finish_claimed_at INTEGER,
            finish_delivered_at INTEGER,
            give_up_claimed_at INTEGER,
            give_up_delivered_at INTEGER,
            detached_no_progress_budget_ms INTEGER,
            last_progress_at INTEGER,
            detached_on_milestones TEXT
        );
        CREATE INDEX idx_agent_tool_runs_parent_tool_call_id
            ON cf_agent_tool_runs(parent_tool_call_id, display_order);
        CREATE TABLE cf_agents_mcp_servers (
            id TEXT PRIMARY KEY NOT NULL,
            name TEXT NOT NULL,
            server_url TEXT NOT NULL,
            callback_url TEXT NOT NULL,
            client_id TEXT,
            auth_url TEXT,
            server_options TEXT
        );
        INSERT INTO cf_agents_state (id, state)
            VALUES ('cf_schema_version', '11');
        INSERT INTO cf_agent_tool_runs (
            run_id, agent_type, status, started_at
        ) VALUES ('typescript-run', 'child', 'running', 1);
        """
    )


def _create_python_v3_fixture(conn) -> None:
    _create_state_table(conn)
    _write_marker(conn, LEGACY_SCHEMA_VERSION_ROW_ID, "3")
    _write_marker(conn, "cf_state_was_changed", "1")
    conn.executescript(
        """
        CREATE TABLE cf_agents_queues (
            id TEXT PRIMARY KEY NOT NULL,
            payload TEXT,
            callback TEXT,
            created_at INTEGER DEFAULT (unixepoch())
        );
        CREATE TABLE cf_agent_tool_runs (
            run_id TEXT PRIMARY KEY,
            parent_tool_call_id TEXT,
            agent_type TEXT NOT NULL,
            input_preview TEXT,
            input_redacted INTEGER NOT NULL DEFAULT 1,
            status TEXT NOT NULL,
            summary TEXT,
            output_json TEXT,
            error_message TEXT,
            interrupted_reason TEXT,
            child_still_running INTEGER,
            display_metadata TEXT,
            display_order INTEGER NOT NULL DEFAULT 0,
            chunks_mirrored INTEGER NOT NULL DEFAULT 0,
            started_at INTEGER NOT NULL,
            completed_at INTEGER
        );
        INSERT INTO cf_agent_tool_runs (
            run_id, agent_type, status, chunks_mirrored, started_at
        ) VALUES ('python-run', 'child', 'completed', 1, 1);
        """
    )


def test_fresh_agent_creates_complete_core_v11_and_stamps_shared_marker():
    agent = fakes.build_agent()

    assert CORE_TABLES <= _tables(agent)
    assert _marker(agent) == str(SCHEMA_VERSION)
    assert _marker(agent, LEGACY_SCHEMA_VERSION_ROW_ID) is None

    for table, columns in CORE_COLUMNS.items():
        actual = _columns(agent, table)
        if table == "cf_agent_tool_runs":
            assert actual[: len(columns)] == columns
            assert actual[len(columns) :] == ["chunks_mirrored"]
        else:
            assert actual == columns

    for name, columns in CORE_INDEXES.items():
        assert _index_columns(agent, name) == columns

    workflow_sql = agent.sql(
        "SELECT sql FROM sqlite_master WHERE type = 'table' "
        "AND name = 'cf_agents_workflows'"
    )[0]["sql"]
    assert "'waitingForPause'" in workflow_sql

    tool_columns = _column_info(agent, "cf_agent_tool_runs")
    assert tool_columns["run_id"] == {
        "cid": 0,
        "name": "run_id",
        "type": "TEXT",
        "notnull": 0,
        "dflt_value": None,
        "pk": 1,
    }
    assert tool_columns["input_redacted"]["notnull"] == 1
    assert tool_columns["input_redacted"]["dflt_value"] == "1"
    assert tool_columns["detached"]["notnull"] == 1
    assert tool_columns["detached"]["dflt_value"] == "0"


@pytest.mark.parametrize("stored", ["malformed", "-7"])
def test_malformed_and_negative_shared_markers_migrate_from_zero(stored: str):
    conn = fakes.new_sqlite()
    _create_state_table(conn)
    _write_marker(conn, SCHEMA_VERSION_ROW_ID, stored)

    agent = fakes.build_agent(conn=conn)

    assert agent._read_schema_version() == SCHEMA_VERSION
    assert _marker(agent) == str(SCHEMA_VERSION)
    assert CORE_TABLES <= _tables(agent)


def test_future_shared_marker_is_preserved_without_core_downgrade():
    conn = fakes.new_sqlite()
    _create_state_table(conn)
    future = str(SCHEMA_VERSION + 4)
    _write_marker(conn, SCHEMA_VERSION_ROW_ID, future)

    agent = fakes.build_agent(conn=conn)

    assert agent._read_schema_version() == SCHEMA_VERSION + 4
    assert _marker(agent) == future
    assert "cf_agents_workflows" not in _tables(agent)


def test_python_v3_object_runs_complete_shared_migration_and_preserves_rows():
    conn = fakes.new_sqlite()
    _create_python_v3_fixture(conn)

    agent = fakes.build_agent(conn=conn)

    assert _marker(agent) == str(SCHEMA_VERSION)
    assert _marker(agent, LEGACY_SCHEMA_VERSION_ROW_ID) == "3"
    assert _marker(agent, "cf_state_was_changed") is None
    assert "retry_options" in _columns(agent, "cf_agents_queues")
    assert set(CORE_COLUMNS["cf_agent_tool_runs"]) <= set(
        _columns(agent, "cf_agent_tool_runs")
    )
    row = agent.sql(
        "SELECT run_id, chunks_mirrored, detached "
        "FROM cf_agent_tool_runs WHERE run_id = 'python-run'"
    )[0]
    assert row == {
        "run_id": "python-run",
        "chunks_mirrored": 1,
        "detached": 0,
    }


def test_reduced_python_agent_tool_table_reconciles_before_index_creation():
    conn = fakes.new_sqlite()
    _create_state_table(conn)
    _write_marker(conn, LEGACY_SCHEMA_VERSION_ROW_ID, "3")
    conn.executescript(
        """
        CREATE TABLE cf_agent_tool_runs (
            run_id TEXT PRIMARY KEY,
            agent_type TEXT NOT NULL,
            status TEXT NOT NULL
        );
        INSERT INTO cf_agent_tool_runs (run_id, agent_type, status)
            VALUES ('reduced-run', 'child', 'running');
        """
    )

    agent = fakes.build_agent(conn=conn)

    assert _marker(agent) == "11"
    assert set(CORE_COLUMNS["cf_agent_tool_runs"]) <= set(
        _columns(agent, "cf_agent_tool_runs")
    )
    assert _index_columns(agent, "idx_agent_tool_runs_parent_tool_call_id") == [
        "parent_tool_call_id",
        "display_order",
    ]
    assert agent.sql(
        "SELECT run_id, input_redacted, display_order, started_at, detached "
        "FROM cf_agent_tool_runs WHERE run_id = 'reduced-run'"
    ) == [
        {
            "run_id": "reduced-run",
            "input_redacted": 1,
            "display_order": 0,
            "started_at": 0,
            "detached": 0,
        }
    ]


def test_typescript_v11_object_reopens_without_core_rewrite():
    conn = fakes.new_sqlite()
    _create_typescript_v11_fixture(conn)

    agent = fakes.build_agent(conn=conn)

    assert _marker(agent) == "11"
    assert agent.sql(
        "SELECT run_id, detached FROM cf_agent_tool_runs "
        "WHERE run_id = 'typescript-run'"
    ) == [{"run_id": "typescript-run", "detached": 0}]
    assert "chunks_mirrored" in _columns(agent, "cf_agent_tool_runs")
    assert {
        "cf_agent_tool_chunks",
        "cf_agent_tool_terminals",
    } <= _tables(agent)


def test_v11_precolumn_agent_tool_table_reconciles_without_marker_rewrite():
    conn = fakes.new_sqlite()
    _create_typescript_v11_fixture(conn)
    for column in (
        "detached",
        "detached_on_finish",
        "detached_notify_source",
        "detached_max_budget_at",
        "finish_claimed_at",
        "finish_delivered_at",
        "give_up_claimed_at",
        "give_up_delivered_at",
        "detached_no_progress_budget_ms",
        "last_progress_at",
        "detached_on_milestones",
    ):
        conn.execute(f"ALTER TABLE cf_agent_tool_runs DROP COLUMN {column}")

    agent = fakes.build_agent(conn=conn)

    assert _marker(agent) == "11"
    assert set(CORE_COLUMNS["cf_agent_tool_runs"]) <= set(
        _columns(agent, "cf_agent_tool_runs")
    )
    assert agent.sql(
        "SELECT run_id, detached FROM cf_agent_tool_runs "
        "WHERE run_id = 'typescript-run'"
    ) == [{"run_id": "typescript-run", "detached": 0}]


@pytest.mark.asyncio
async def test_interrupted_migration_leaves_marker_unstamped_and_retries_on_wake(
    monkeypatch,
):
    conn = fakes.new_sqlite()
    ctx = fakes.FakeCtx(conn=conn)
    original_exec = ctx.storage.sql.exec
    failures = 0

    def fail_mid_migration(query, *params):
        nonlocal failures
        if "ADD COLUMN detached_on_finish" in query:
            failures += 1
            raise RuntimeError("migration interrupted")
        return original_exec(query, *params)

    monkeypatch.setattr(ctx.storage.sql, "exec", fail_mid_migration)

    first = Agent(ctx, types.SimpleNamespace())

    assert failures == 0
    with pytest.raises(RuntimeError, match="migration interrupted"):
        await first._ensure_initialized()
    assert failures == 1
    assert _marker(first) is None

    reincarnated = Agent(fakes.FakeCtx(conn=conn), types.SimpleNamespace())
    await reincarnated._ensure_initialized()

    assert _marker(reincarnated) == "11"
    assert CORE_TABLES <= _tables(reincarnated)
    assert set(CORE_COLUMNS["cf_agent_tool_runs"]) <= set(
        _columns(reincarnated, "cf_agent_tool_runs")
    )
    assert _index_columns(reincarnated, "idx_agent_tool_runs_parent_tool_call_id") == [
        "parent_tool_call_id",
        "display_order",
    ]


@pytest.mark.asyncio
async def test_python_compatibility_preparation_retries_after_shared_stamp():
    class RetryCompatibilityAgent(Agent):
        attempts = 0

        def _prepare_agent_tool_compat_schema(self):
            type(self).attempts += 1
            if type(self).attempts == 1:
                raise RuntimeError("compatibility preparation interrupted")
            super()._prepare_agent_tool_compat_schema()

    conn = fakes.new_sqlite()

    first = RetryCompatibilityAgent(fakes.FakeCtx(conn=conn), types.SimpleNamespace())

    with pytest.raises(RuntimeError, match="compatibility preparation interrupted"):
        await first._ensure_initialized()

    assert _marker(first) == "11"
    assert "chunks_mirrored" not in _columns(first, "cf_agent_tool_runs")

    reincarnated = RetryCompatibilityAgent(
        fakes.FakeCtx(conn=conn), types.SimpleNamespace()
    )
    await reincarnated._ensure_initialized()

    assert RetryCompatibilityAgent.attempts == 2
    assert _marker(reincarnated) == "11"
    assert "chunks_mirrored" in _columns(reincarnated, "cf_agent_tool_runs")


@pytest.mark.asyncio
async def test_facet_ancestry_is_persisted_before_full_startup_preparation():
    class BrokenFacet(Agent):
        def _run_schema_migration(self):
            raise RuntimeError("full preparation failed")

    facet = BrokenFacet(
        fakes.FakeCtx(name="cf-agents:v2:leaf:digest"),
        types.SimpleNamespace(),
    )
    parent_path = '[{"className":"RootAgent","name":"root"}]'

    with pytest.raises(RuntimeError, match="full preparation failed"):
        await facet._cf_init_as_facet("leaf", parent_path)

    assert facet._read_state_cell(PARENT_PATH_ROW_ID) == parent_path
    assert facet.parent_path == [{"className": "RootAgent", "name": "root"}]
    assert facet._lifecycle._ready is False


def test_migration_is_idempotent():
    agent = fakes.build_agent()

    agent._run_schema_migration()
    agent._run_schema_migration()

    assert _marker(agent) == "11"
    assert CORE_TABLES <= _tables(agent)
