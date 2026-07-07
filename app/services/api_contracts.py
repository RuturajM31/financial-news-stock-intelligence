"""Validate the small public data contracts used by the Streamlit client.

Purpose
-------
The FastAPI server already validates its own schemas. The Streamlit layer still
checks the fields it depends on so an unexpected or incomplete response fails
closed instead of producing a misleading page.

Inputs and downstream use
-------------------------
Functions receive decoded JSON dictionaries from ``api_client.py``. They return
immutable values used by status cards, session state, and later analysis pages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


PASSED_STATUS = "PASSED"
FAILED_STATUS = "FAILED"


def _require_mapping(value: Any, location: str) -> Mapping[str, Any]:
    """Return a mapping or raise a clear contract error."""

    if not isinstance(value, Mapping):
        raise ValueError(f"{location} must be a JSON object.")
    return value


def _require_text(
    mapping: Mapping[str, Any],
    key: str,
    location: str,
) -> str:
    """Read one required non-empty text field from a response mapping."""

    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location}.{key} must be non-empty text.")
    return value.strip()


def _optional_text(
    mapping: Mapping[str, Any],
    key: str,
) -> str | None:
    """Read one optional text field while rejecting a wrong data type."""

    value = mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{key} must be text when provided.")
    stripped = value.strip()
    return stripped or None


@dataclass(frozen=True)
class ApiProblemDetails:
    """Store one safe error returned by FastAPI or created by the UI client."""

    error_code: str
    what_failed: str
    where_failed: str
    why_failed: str
    safe_next_step: str
    request_id: str | None = None

    def as_dict(self) -> dict[str, str | None]:
        """Return a JSON-safe copy for temporary Streamlit session state."""

        return {
            "error_code": self.error_code,
            "what_failed": self.what_failed,
            "where_failed": self.where_failed,
            "why_failed": self.why_failed,
            "safe_next_step": self.safe_next_step,
            "request_id": self.request_id,
        }


@dataclass(frozen=True)
class HealthStatus:
    """Store the public FastAPI health response used by the UI."""

    status: str
    service: str
    version: str
    request_id: str | None = None

    def as_dict(self) -> dict[str, str | None]:
        """Return a JSON-safe value for session state."""

        return {
            "status": self.status,
            "service": self.service,
            "version": self.version,
            "request_id": self.request_id,
        }


@dataclass(frozen=True)
class ReadinessStatus:
    """Store the public readiness result and its component states."""

    status: str
    components: dict[str, str]
    details: dict[str, Any]
    request_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a copy safe for current-session storage."""

        return {
            "status": self.status,
            "components": dict(self.components),
            "details": dict(self.details),
            "request_id": self.request_id,
        }


def parse_problem_payload(
    payload: Mapping[str, Any] | None,
    fallback_code: str,
    fallback_location: str,
    fallback_reason: str,
    fallback_next_step: str,
    request_id: str | None = None,
) -> ApiProblemDetails:
    """Parse FastAPI's safe error shape without exposing raw response text."""

    mapping: Mapping[str, Any] = payload if isinstance(payload, Mapping) else {}
    return ApiProblemDetails(
        error_code=_optional_text(mapping, "error_code") or fallback_code,
        what_failed=(
            _optional_text(mapping, "what_failed") or "The API request failed."
        ),
        where_failed=(
            _optional_text(mapping, "where_failed") or fallback_location
        ),
        why_failed=_optional_text(mapping, "why_failed") or fallback_reason,
        safe_next_step=(
            _optional_text(mapping, "safe_next_step") or fallback_next_step
        ),
        request_id=(
            _optional_text(mapping, "request_id")
            or request_id
        ),
    )


def parse_health_response(
    payload: Mapping[str, Any],
    request_id: str | None = None,
) -> HealthStatus:
    """Validate the exact fields needed by the live health card."""

    mapping = _require_mapping(payload, "health response")
    status = _require_text(mapping, "status", "health response")
    if status != PASSED_STATUS:
        raise ValueError("The FastAPI health response did not pass.")
    return HealthStatus(
        status=status,
        service=_require_text(mapping, "service", "health response"),
        version=_require_text(mapping, "version", "health response"),
        request_id=request_id,
    )


def parse_readiness_response(
    payload: Mapping[str, Any],
    request_id: str | None = None,
) -> ReadinessStatus:
    """Validate readiness status and every named component value."""

    mapping = _require_mapping(payload, "readiness response")
    status = _require_text(mapping, "status", "readiness response")
    components_raw = _require_mapping(
        mapping.get("components"),
        "readiness response.components",
    )
    components: dict[str, str] = {}
    for name, value in components_raw.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Readiness component names must be non-empty text.")
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Readiness component {name!r} must contain a text status."
            )
        components[name.strip()] = value.strip()

    details_raw = mapping.get("details", {})
    details = dict(_require_mapping(details_raw, "readiness response.details"))
    return ReadinessStatus(
        status=status,
        components=components,
        details=details,
        request_id=request_id,
    )


def validate_success_payload(
    payload: Mapping[str, Any],
    endpoint_name: str,
) -> dict[str, Any]:
    """Require a passed status before later pages use an analysis response."""

    mapping = _require_mapping(payload, f"{endpoint_name} response")
    status = _require_text(mapping, "status", f"{endpoint_name} response")
    if status != PASSED_STATUS:
        raise ValueError(f"{endpoint_name} response status was not PASSED.")
    return dict(mapping)
