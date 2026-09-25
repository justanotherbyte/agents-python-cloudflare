from __future__ import annotations

import pytest

from agents.chat.recovery import RecoveryPolicy, evaluate_incident, incident_id


def _existing(**updates):
    incident = {
        "incidentId": "root:user",
        "requestId": "request-1",
        "recoveryRootRequestId": "root",
        "recoveryKind": "retry",
        "attempt": 2,
        "maxAttempts": 10,
        "status": "attempting",
        "firstSeenAt": 1,
        "lastAttemptAt": 1,
        "lastProgressAt": 1,
        "progress": 0,
        "workBaseline": 0,
    }
    incident.update(updates)
    return incident


def _evaluate(existing, *, progress=0, awaiting=False, now=100_000):
    return evaluate_incident(
        request_id="request-2",
        recovery_root_request_id="root",
        latest_user_message_id="user",
        recovery_kind="retry",
        existing=existing,
        progress=progress,
        awaiting_client_interaction=awaiting,
        now=now,
        policy=RecoveryPolicy(),
    )


def test_continuations_share_one_logical_recovery_incident():
    assert incident_id("first", "root", "user") == incident_id(
        "continuation",
        "root",
        "user",
    )


def test_client_interaction_parks_without_spending_an_attempt():
    decision = _evaluate(
        _existing(attempt=10, oomAttempts=4),
        progress=2_000,
        awaiting=True,
        now=1_000_000,
    )

    assert decision.exhausted is False
    assert decision.incident["attempt"] == 10
    assert decision.incident["lastProgressAt"] == 1_000_000


def test_durable_progress_resets_attempt_and_no_progress_clock():
    decision = _evaluate(_existing(attempt=9, progress=3), progress=4)

    assert decision.exhausted is False
    assert decision.incident["attempt"] == 1
    assert decision.incident["lastProgressAt"] == 100_000


@pytest.mark.parametrize(
    ("existing", "progress", "now", "reason"),
    [
        (
            _existing(attempt=10, lastProgressAt=100_000),
            0,
            100_000,
            "max_attempts_exceeded",
        ),
        (
            _existing(lastProgressAt=1),
            0,
            5 * 60 * 1_000 + 2,
            "no_progress_timeout",
        ),
        (
            _existing(progress=0, lastProgressAt=100_000),
            1_001,
            100_000,
            "work_budget_exceeded",
        ),
        (
            _existing(oomAttempts=4, lastProgressAt=100_000),
            0,
            100_000,
            "out_of_memory",
        ),
    ],
)
def test_recovery_budget_exhaustion(existing, progress, now, reason):
    decision = _evaluate(existing, progress=progress, now=now)

    assert decision.exhausted is True
    assert decision.incident["status"] == "exhausted"
    assert decision.incident["reason"] == reason
