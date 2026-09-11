from __future__ import annotations

from collections.abc import Callable
from typing import Any


type SqlExecutor = Callable[..., list[dict[str, Any]]]


def prepare_fiber_schema(sql: SqlExecutor) -> None:
    sql("""
    CREATE TABLE IF NOT EXISTS cf_agents_runs (
        id TEXT PRIMARY KEY NOT NULL,
        name TEXT NOT NULL,
        snapshot TEXT,
        created_at INTEGER NOT NULL
    )
    """)
    sql("""
    CREATE TABLE IF NOT EXISTS cf_agents_fibers (
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
    )
    """)
    sql(
        "CREATE INDEX IF NOT EXISTS idx_fibers_status_created "
        "ON cf_agents_fibers(status, created_at, fiber_id)"
    )
    sql(
        "CREATE INDEX IF NOT EXISTS idx_fibers_name_status_created "
        "ON cf_agents_fibers(name, status, created_at, fiber_id)"
    )
    sql(
        "CREATE INDEX IF NOT EXISTS idx_fibers_status_completed "
        "ON cf_agents_fibers(status, completed_at, created_at)"
    )
