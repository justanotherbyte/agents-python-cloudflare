from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..lifecycle.fiber_schema import prepare_fiber_schema

CORE_SCHEMA_VERSION_ROW_ID = "cf_schema_version"
CORE_SCHEMA_VERSION = 11

type SqlExecutor = Callable[..., list[dict[str, Any]]]


def parse_schema_version(raw: Any) -> int:
    try:
        version = int(raw)
    except (TypeError, ValueError):
        return 0
    return version if version >= 0 else 0


def read_core_schema_version(sql: SqlExecutor) -> int:
    rows = sql(
        "SELECT state FROM cf_agents_state WHERE id = ?",
        CORE_SCHEMA_VERSION_ROW_ID,
    )
    return parse_schema_version(rows[0]["state"] if rows else None)


def prepare_core_state_schema(sql: SqlExecutor) -> None:
    sql("""
    CREATE TABLE IF NOT EXISTS cf_agents_state (
        id TEXT PRIMARY KEY NOT NULL,
        state TEXT
    )
    """)


def prepare_core_schema(sql: SqlExecutor) -> int:
    prepare_core_state_schema(sql)

    stored = read_core_schema_version(sql)
    if stored > CORE_SCHEMA_VERSION:
        return stored
    if stored == CORE_SCHEMA_VERSION:
        _prepare_agent_tool_columns(sql)
        return stored

    sql("""
    CREATE TABLE IF NOT EXISTS cf_agents_mcp_servers (
        id TEXT PRIMARY KEY NOT NULL,
        name TEXT NOT NULL,
        server_url TEXT NOT NULL,
        callback_url TEXT NOT NULL,
        client_id TEXT,
        auth_url TEXT,
        server_options TEXT
    )
    """)
    sql("""
    CREATE TABLE IF NOT EXISTS cf_agents_queues (
        id TEXT PRIMARY KEY NOT NULL,
        payload TEXT,
        callback TEXT,
        created_at INTEGER DEFAULT (unixepoch())
    )
    """)
    _add_column(
        sql,
        "ALTER TABLE cf_agents_queues ADD COLUMN retry_options TEXT",
    )

    sql("""
    CREATE TABLE IF NOT EXISTS cf_agents_workflows (
        id TEXT PRIMARY KEY NOT NULL,
        workflow_id TEXT NOT NULL UNIQUE,
        workflow_name TEXT NOT NULL,
        status TEXT NOT NULL CHECK(status IN (
            'queued', 'running', 'paused', 'errored', 'terminated', 'complete',
            'waiting', 'waitingForPause', 'unknown'
        )),
        metadata TEXT,
        error_name TEXT,
        error_message TEXT,
        created_at INTEGER NOT NULL DEFAULT (unixepoch()),
        updated_at INTEGER NOT NULL DEFAULT (unixepoch()),
        completed_at INTEGER
    )
    """)
    sql(
        "CREATE INDEX IF NOT EXISTS idx_workflows_status ON cf_agents_workflows(status)"
    )
    sql(
        "CREATE INDEX IF NOT EXISTS idx_workflows_name "
        "ON cf_agents_workflows(workflow_name)"
    )
    sql("DELETE FROM cf_agents_state WHERE id = 'cf_state_was_changed'")

    prepare_fiber_schema(sql)
    sql("""
    CREATE TABLE IF NOT EXISTS cf_agents_facet_runs (
        owner_path TEXT NOT NULL,
        owner_path_key TEXT NOT NULL,
        run_id TEXT NOT NULL,
        created_at INTEGER NOT NULL,
        PRIMARY KEY (owner_path_key, run_id)
    )
    """)
    sql(
        "CREATE INDEX IF NOT EXISTS idx_facet_runs_owner_path_key "
        "ON cf_agents_facet_runs(owner_path_key)"
    )

    sql("""
    CREATE TABLE IF NOT EXISTS cf_agent_tool_runs (
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
        completed_at INTEGER
    )
    """)
    _prepare_agent_tool_columns(sql)

    sql(
        "INSERT OR REPLACE INTO cf_agents_state (id, state) VALUES (?, ?)",
        CORE_SCHEMA_VERSION_ROW_ID,
        str(CORE_SCHEMA_VERSION),
    )
    return CORE_SCHEMA_VERSION


def _add_column(sql: SqlExecutor, statement: str) -> None:
    try:
        sql(statement)
    except Exception as error:
        if "duplicate column" not in str(error).lower():
            raise


def _prepare_agent_tool_columns(sql: SqlExecutor) -> None:
    columns = {row["name"] for row in sql("PRAGMA table_info(cf_agent_tool_runs)")}
    for name, statement in _AGENT_TOOL_COLUMN_ADDITIONS:
        if name not in columns:
            _add_column(sql, statement)

    columns = {row["name"] for row in sql("PRAGMA table_info(cf_agent_tool_runs)")}
    missing = _AGENT_TOOL_REQUIRED_COLUMNS - columns
    if missing:
        names = ", ".join(sorted(missing))
        raise RuntimeError(f"cf_agent_tool_runs is missing required columns: {names}")
    sql(
        "CREATE INDEX IF NOT EXISTS idx_agent_tool_runs_parent_tool_call_id "
        "ON cf_agent_tool_runs(parent_tool_call_id, display_order)"
    )


_AGENT_TOOL_REQUIRED_COLUMNS = {
    "run_id",
    "agent_type",
    "status",
}

_AGENT_TOOL_COLUMN_ADDITIONS = (
    (
        "parent_tool_call_id",
        "ALTER TABLE cf_agent_tool_runs ADD COLUMN parent_tool_call_id TEXT",
    ),
    (
        "input_preview",
        "ALTER TABLE cf_agent_tool_runs ADD COLUMN input_preview TEXT",
    ),
    (
        "input_redacted",
        "ALTER TABLE cf_agent_tool_runs ADD COLUMN input_redacted "
        "INTEGER NOT NULL DEFAULT 1",
    ),
    ("summary", "ALTER TABLE cf_agent_tool_runs ADD COLUMN summary TEXT"),
    (
        "output_json",
        "ALTER TABLE cf_agent_tool_runs ADD COLUMN output_json TEXT",
    ),
    (
        "error_message",
        "ALTER TABLE cf_agent_tool_runs ADD COLUMN error_message TEXT",
    ),
    (
        "interrupted_reason",
        "ALTER TABLE cf_agent_tool_runs ADD COLUMN interrupted_reason TEXT",
    ),
    (
        "child_still_running",
        "ALTER TABLE cf_agent_tool_runs ADD COLUMN child_still_running INTEGER",
    ),
    (
        "display_metadata",
        "ALTER TABLE cf_agent_tool_runs ADD COLUMN display_metadata TEXT",
    ),
    (
        "display_order",
        "ALTER TABLE cf_agent_tool_runs ADD COLUMN display_order "
        "INTEGER NOT NULL DEFAULT 0",
    ),
    (
        "started_at",
        "ALTER TABLE cf_agent_tool_runs ADD COLUMN started_at "
        "INTEGER NOT NULL DEFAULT 0",
    ),
    (
        "completed_at",
        "ALTER TABLE cf_agent_tool_runs ADD COLUMN completed_at INTEGER",
    ),
    (
        "detached",
        "ALTER TABLE cf_agent_tool_runs ADD COLUMN detached INTEGER NOT NULL DEFAULT 0",
    ),
    (
        "detached_on_finish",
        "ALTER TABLE cf_agent_tool_runs ADD COLUMN detached_on_finish TEXT",
    ),
    (
        "detached_notify_source",
        "ALTER TABLE cf_agent_tool_runs ADD COLUMN detached_notify_source TEXT",
    ),
    (
        "detached_max_budget_at",
        "ALTER TABLE cf_agent_tool_runs ADD COLUMN detached_max_budget_at INTEGER",
    ),
    (
        "finish_claimed_at",
        "ALTER TABLE cf_agent_tool_runs ADD COLUMN finish_claimed_at INTEGER",
    ),
    (
        "finish_delivered_at",
        "ALTER TABLE cf_agent_tool_runs ADD COLUMN finish_delivered_at INTEGER",
    ),
    (
        "give_up_claimed_at",
        "ALTER TABLE cf_agent_tool_runs ADD COLUMN give_up_claimed_at INTEGER",
    ),
    (
        "give_up_delivered_at",
        "ALTER TABLE cf_agent_tool_runs ADD COLUMN give_up_delivered_at INTEGER",
    ),
    (
        "detached_no_progress_budget_ms",
        "ALTER TABLE cf_agent_tool_runs "
        "ADD COLUMN detached_no_progress_budget_ms INTEGER",
    ),
    (
        "last_progress_at",
        "ALTER TABLE cf_agent_tool_runs ADD COLUMN last_progress_at INTEGER",
    ),
    (
        "detached_on_milestones",
        "ALTER TABLE cf_agent_tool_runs ADD COLUMN detached_on_milestones TEXT",
    ),
)
