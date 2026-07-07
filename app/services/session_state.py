"""Manage small, temporary Streamlit session values predictably.

Purpose
-------
Keep connection checks and later analysis summaries in the current browser
session only. Raw uploaded files, complete article text, secrets, and provider
responses must never be stored here.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, MutableMapping


API_CHECK_KEY = "rm_api_check"
ANALYSIS_HISTORY_KEY = "rm_analysis_history"
MAXIMUM_SESSION_HISTORY = 20


def initialize_session_state(state: MutableMapping[str, Any]) -> None:
    """Create the approved session keys without replacing existing values."""

    state.setdefault(API_CHECK_KEY, None)
    state.setdefault(ANALYSIS_HISTORY_KEY, [])


def store_api_check(
    state: MutableMapping[str, Any],
    *,
    health: dict[str, Any] | None,
    readiness: dict[str, Any] | None,
    problem: dict[str, Any] | None,
) -> None:
    """Store one bounded connection-check result for the current session."""

    initialize_session_state(state)
    state[API_CHECK_KEY] = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "health": dict(health) if health is not None else None,
        "readiness": dict(readiness) if readiness is not None else None,
        "problem": dict(problem) if problem is not None else None,
    }


def get_api_check(state: MutableMapping[str, Any]) -> dict[str, Any] | None:
    """Return the current-session connection result when present."""

    initialize_session_state(state)
    value = state.get(API_CHECK_KEY)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("The Streamlit API check state has an invalid shape.")
    return dict(value)


def clear_api_check(state: MutableMapping[str, Any]) -> None:
    """Remove only the current API check without affecting other widgets."""

    initialize_session_state(state)
    state[API_CHECK_KEY] = None


def add_analysis_summary(
    state: MutableMapping[str, Any],
    summary: dict[str, Any],
) -> None:
    """Append one licence-safe summary and enforce the session history limit.

    The caller must provide a summary, not raw article text or file bytes. The
    function rejects known unsafe keys to prevent accidental session storage.
    """

    initialize_session_state(state)
    forbidden_keys = {"raw_text", "file_bytes", "api_key", "token", "secret"}
    normalized_keys = {str(key).lower() for key in summary}
    blocked = sorted(forbidden_keys & normalized_keys)
    if blocked:
        raise ValueError(
            "Analysis summary contains forbidden session fields: "
            + ", ".join(blocked)
        )

    history = state[ANALYSIS_HISTORY_KEY]
    if not isinstance(history, list):
        raise ValueError("The Streamlit analysis history has an invalid shape.")
    record = dict(summary)
    record.setdefault("recorded_at_utc", datetime.now(timezone.utc).isoformat())
    history.append(record)
    del history[:-MAXIMUM_SESSION_HISTORY]


def clear_analysis_history(state: MutableMapping[str, Any]) -> None:
    """Delete current-session summaries without touching persistent storage."""

    initialize_session_state(state)
    state[ANALYSIS_HISTORY_KEY] = []
