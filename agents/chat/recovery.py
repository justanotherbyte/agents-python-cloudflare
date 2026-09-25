from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import quote


CHAT_RECOVERY_INCIDENT_KEY_PREFIX = "cf:chat-recovery:incident:"
CHAT_RECOVERY_PROGRESS_KEY = "cf:chat-recovery:progress"
CHAT_RECOVERING_KEY = "cf:chat:recovering"
CHAT_RECOVERY_INCIDENT_TTL_MS = 60 * 60 * 1000
CHAT_RECOVERING_FLAG_TTL_MS = 15 * 60 * 1000
CHAT_RECOVERY_ALARM_DEBOUNCE_MS = 30 * 1000

type RecoveryKind = Literal["retry", "continue"]


@dataclass(frozen=True, slots=True)
class RecoveryPolicy:
    max_attempts: int = 10
    no_progress_timeout_ms: int = 5 * 60 * 1000
    max_work: int = 1_000
    max_oom_retries: int = 3


@dataclass(frozen=True, slots=True)
class RecoveryDecision:
    incident: dict[str, Any]
    exhausted: bool


def incident_id(
    request_id: str,
    recovery_root_request_id: str | None,
    latest_user_message_id: str | None,
) -> str:
    return f"{recovery_root_request_id or request_id}:{latest_user_message_id or ''}"


def incident_key(value: str) -> str:
    return f"{CHAT_RECOVERY_INCIDENT_KEY_PREFIX}{quote(value, safe='')}"


def evaluate_incident(
    *,
    request_id: str,
    recovery_root_request_id: str | None,
    latest_user_message_id: str | None,
    recovery_kind: RecoveryKind,
    existing: dict[str, Any] | None,
    progress: int,
    awaiting_client_interaction: bool,
    now: int,
    policy: RecoveryPolicy,
) -> RecoveryDecision:
    value = incident_id(
        request_id,
        recovery_root_request_id,
        latest_user_message_id,
    )
    previous_progress = int((existing or {}).get("progress") or 0)
    made_progress = existing is not None and progress > previous_progress
    first_seen = int((existing or {}).get("firstSeenAt") or now)
    last_progress = int((existing or {}).get("lastProgressAt") or first_seen)
    if made_progress or awaiting_client_interaction:
        last_progress = now
    baseline_value = (existing or {}).get("workBaseline")
    baseline = int(baseline_value) if type(baseline_value) is int else progress
    work = max(previous_progress, progress) - baseline
    debounced = (
        existing is not None
        and not made_progress
        and now - int(existing.get("lastAttemptAt") or 0)
        < CHAT_RECOVERY_ALARM_DEBOUNCE_MS
    )
    previous_attempt = int((existing or {}).get("attempt") or 0)
    attempt = (
        previous_attempt
        if awaiting_client_interaction
        else 1
        if made_progress
        else previous_attempt
        if debounced
        else previous_attempt + 1
    )
    oom_attempts = int((existing or {}).get("oomAttempts") or 0)
    reason = None
    if not awaiting_client_interaction:
        if oom_attempts > policy.max_oom_retries:
            reason = "out_of_memory"
        elif (
            existing is not None and now - last_progress > policy.no_progress_timeout_ms
        ):
            reason = "no_progress_timeout"
        elif existing is not None and work > policy.max_work:
            reason = "work_budget_exceeded"
        elif attempt > policy.max_attempts:
            reason = "max_attempts_exceeded"
    exhausted = reason is not None
    incident: dict[str, Any] = {
        "incidentId": value,
        "requestId": request_id,
        "recoveryRootRequestId": recovery_root_request_id or request_id,
        "recoveryKind": recovery_kind,
        "attempt": attempt,
        "maxAttempts": policy.max_attempts,
        "status": "exhausted" if exhausted else "attempting",
        "firstSeenAt": first_seen,
        "lastAttemptAt": now,
        "lastProgressAt": last_progress,
        "progress": max(previous_progress, progress),
        "workBaseline": baseline,
    }
    if oom_attempts:
        incident["oomAttempts"] = oom_attempts
    if reason is not None:
        incident["reason"] = reason
    return RecoveryDecision(incident, exhausted)
