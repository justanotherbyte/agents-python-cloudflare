from __future__ import annotations

from datetime import UTC, datetime

import fakes
import pytest

from agents.core.schema import prepare_core_schema
from agents.workflows import (
    WorkflowErrorInfo,
    WorkflowInstanceStatus,
    WorkflowLedger,
    WorkflowQueryCriteria,
)


class Sql:
    def __init__(self) -> None:
        self.ctx = fakes.FakeCtx()
        prepare_core_schema(self.execute)

    def execute(self, query: str, *params: object) -> list[dict]:
        return self.ctx.storage.sql.exec(query, *params).toArray()


def ledger() -> tuple[WorkflowLedger, Sql]:
    sql = Sql()
    ids = iter([f"row-{index}" for index in range(20)])
    return WorkflowLedger(
        sql, id_factory=lambda: next(ids), now_seconds=lambda: 100
    ), sql


def seed(
    sql: Sql,
    workflow_id: str,
    *,
    workflow_name: str = "REPORTS",
    status: str = "queued",
    metadata: str | None = None,
    created_at: int = 100,
    updated_at: int = 100,
    completed_at: int | None = None,
    error_name: str | None = None,
    error_message: str | None = None,
) -> None:
    sql.execute(
        """
        INSERT INTO cf_agents_workflows
          (id, workflow_id, workflow_name, status, metadata, error_name,
           error_message, created_at, updated_at, completed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        f"internal-{workflow_id}",
        workflow_id,
        workflow_name,
        status,
        metadata,
        error_name,
        error_message,
        created_at,
        updated_at,
        completed_at,
    )


def test_track_and_get_project_the_cross_runtime_row():
    workflows, _ = ledger()

    workflows.track("wf-1", "REPORTS", {"tenant": "a", "attempt": 2})
    workflows.update_status(
        "wf-1",
        WorkflowInstanceStatus(
            status="errored",
            error=WorkflowErrorInfo(name="RangeError", message="bad input"),
        ),
    )

    info = workflows.get("wf-1")
    assert info is not None
    assert info.id == "row-0"
    assert info.workflow_id == "wf-1"
    assert info.workflow_name == "REPORTS"
    assert info.status == "errored"
    assert info.metadata == {"tenant": "a", "attempt": 2}
    assert info.error == WorkflowErrorInfo(name="RangeError", message="bad input")
    assert info.created_at == datetime.fromtimestamp(100, UTC)
    assert info.updated_at == datetime.fromtimestamp(100, UTC)
    assert info.completed_at == datetime.fromtimestamp(100, UTC)
    assert workflows.get("missing") is None


def test_duplicate_tracking_has_a_stable_domain_error():
    workflows, _ = ledger()
    workflows.track("wf-1", "REPORTS")

    with pytest.raises(ValueError, match='Workflow with ID "wf-1" is already'):
        workflows.track("wf-1", "REPORTS")


def test_filtered_pages_use_total_ordering_and_an_opaque_cursor():
    workflows, sql = ledger()
    seed(
        sql,
        "wf-a",
        status="complete",
        metadata='{"tenant":"a","live":true}',
        created_at=1,
    )
    seed(
        sql,
        "wf-b",
        status="running",
        metadata='{"tenant":"a","live":true}',
        created_at=2,
    )
    seed(
        sql,
        "wf-c",
        status="complete",
        metadata='{"tenant":"b","live":true}',
        created_at=3,
    )
    seed(
        sql,
        "wf-d",
        status="complete",
        metadata='{"tenant":"a","live":false}',
        created_at=4,
    )
    seed(
        sql,
        "wf-e",
        status="complete",
        metadata='{"tenant":"a","live":true}',
        created_at=5,
    )

    criteria = WorkflowQueryCriteria(
        status=("complete", "errored"),
        workflow_name="REPORTS",
        metadata={"tenant": "a", "live": True},
        limit=2,
    )
    first = workflows.list(criteria)

    assert [item.workflow_id for item in first.workflows] == ["wf-e", "wf-a"]
    assert first.total == 2
    assert first.next_cursor is None


def test_metadata_filters_treat_json_punctuation_as_literal_key_text():
    workflows, sql = ledger()
    workflows.track(
        "wf-literal",
        "REPORTS",
        {
            "a.b": "dot",
            'a["b"]': "brackets",
            'say"hello': "quote",
            "a": {"b": "nested"},
        },
    )
    workflows.track("wf-nested", "REPORTS", {"a": {"b": "dot"}})

    for key, value in (
        ("a.b", "dot"),
        ('a["b"]', "brackets"),
        ('say"hello', "quote"),
    ):
        page = workflows.list(WorkflowQueryCriteria(metadata={key: value}))
        assert [item.workflow_id for item in page.workflows] == ["wf-literal"]


def test_cursor_pagination_breaks_created_time_ties_with_workflow_id():
    workflows, sql = ledger()
    for workflow_id in ("wf-a", "wf-b", "wf-c", "wf-d", "wf-e"):
        seed(sql, workflow_id, status="complete", created_at=10)

    first = workflows.list(WorkflowQueryCriteria(limit=2, order_by="asc"))
    second = workflows.list(
        WorkflowQueryCriteria(limit=2, order_by="asc", cursor=first.next_cursor)
    )
    third = workflows.list(
        WorkflowQueryCriteria(limit=2, order_by="asc", cursor=second.next_cursor)
    )

    assert [item.workflow_id for item in first.workflows] == ["wf-a", "wf-b"]
    assert [item.workflow_id for item in second.workflows] == ["wf-c", "wf-d"]
    assert [item.workflow_id for item in third.workflows] == ["wf-e"]
    assert first.total == second.total == third.total == 5
    assert first.next_cursor is not None
    assert second.next_cursor is not None
    assert third.next_cursor is None


def test_limits_default_to_fifty_and_cap_at_one_hundred():
    workflows, sql = ledger()
    for index in range(110):
        seed(sql, f"wf-{index:03d}", created_at=index)

    assert len(workflows.list().workflows) == 50
    assert len(workflows.list(WorkflowQueryCriteria(limit=10_000)).workflows) == 100


def test_invalid_cursor_is_rejected_before_querying():
    workflows, _ = ledger()

    with pytest.raises(ValueError, match="Invalid pagination cursor"):
        workflows.list(WorkflowQueryCriteria(cursor="not-base64"))


def test_delete_operations_report_exact_rows_and_respect_filters():
    workflows, sql = ledger()
    seed(sql, "wf-a", status="complete", created_at=1)
    seed(sql, "wf-b", status="errored", created_at=2)
    seed(sql, "wf-c", status="running", created_at=3)

    assert workflows.delete("wf-c") is True
    assert workflows.delete("wf-c") is False
    assert (
        workflows.delete_many(
            status=("complete",), created_before=datetime.fromtimestamp(2, UTC)
        )
        == 1
    )
    assert [item.workflow_id for item in workflows.list().workflows] == ["wf-b"]
