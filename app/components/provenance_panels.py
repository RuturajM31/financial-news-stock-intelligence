"""Sanitize and render licence-safe source and model provenance.

FastAPI already applies the provider boundary. Streamlit repeats a smaller
presentation check so tokens, secret-like fields, local paths, and oversized
values cannot be displayed or included in a download by mistake.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from html import escape
from typing import Any, Mapping, Sequence

SENSITIVE_KEY_PARTS = (
    "api_key",
    "apikey",
    "token",
    "secret",
    "password",
    "credential",
    "private_key",
)
MAXIMUM_DEPTH = 5
MAXIMUM_MAPPING_ITEMS = 120
MAXIMUM_SEQUENCE_ITEMS = 100
MAXIMUM_TEXT_LENGTH = 1000


def _is_sensitive_key(key: str) -> bool:
    """Return whether a key name is unsafe for public display."""

    normalized = key.strip().lower().replace("-", "_")
    return any(part in normalized for part in SENSITIVE_KEY_PARTS)


def _safe_text(value: str, location: str) -> str:
    """Return bounded text and reject local-path or secret-like content."""

    stripped = value.strip()
    if len(stripped) > MAXIMUM_TEXT_LENGTH:
        raise ValueError(f"{location} is longer than the approved display limit.")
    lowered = stripped.lower()
    if "/users/" in lowered or "\\users\\" in lowered:
        raise ValueError(f"{location} contains a local user path.")
    if "bearer " in lowered or "x-api-key" in lowered:
        raise ValueError(f"{location} contains a private authentication value.")
    return stripped


def sanitize_public_value(value: Any, *, location: str = "provenance", depth: int = 0) -> Any:
    """Return a bounded public copy of nested JSON-like provenance values."""

    if depth > MAXIMUM_DEPTH:
        raise ValueError(f"{location} is nested too deeply.")
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{location} contains a non-finite number.")
        return value
    if isinstance(value, str):
        return _safe_text(value, location)
    if isinstance(value, Mapping):
        if len(value) > MAXIMUM_MAPPING_ITEMS:
            raise ValueError(f"{location} contains too many fields.")
        result: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key).strip()
            if not key:
                raise ValueError(f"{location} contains an empty field name.")
            if _is_sensitive_key(key):
                continue
            result[key] = sanitize_public_value(
                raw_value,
                location=f"{location}.{key}",
                depth=depth + 1,
            )
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) > MAXIMUM_SEQUENCE_ITEMS:
            raise ValueError(f"{location} contains too many list values.")
        return [
            sanitize_public_value(
                item,
                location=f"{location}[{index}]",
                depth=depth + 1,
            )
            for index, item in enumerate(value)
        ]
    raise ValueError(f"{location} contains an unsupported value type.")


@dataclass(frozen=True)
class ProvenanceView:
    """Store sanitized provenance ready for display and download."""

    data: dict[str, Any]

    def rows(self) -> list[dict[str, str]]:
        """Flatten nested values into a stable, readable evidence table."""

        rows: list[dict[str, str]] = []

        def walk(value: Any, path: str) -> None:
            """Append one nested value to the flat evidence table."""

            if isinstance(value, Mapping):
                for key in sorted(value):
                    next_path = f"{path} → {key}" if path else str(key)
                    walk(value[key], next_path)
                return
            if isinstance(value, list):
                for index, item in enumerate(value):
                    walk(item, f"{path} → item {index + 1}")
                return
            rendered = json.dumps(value, ensure_ascii=False) if value is not None else "Not recorded"
            rows.append({"Evidence field": path, "Recorded value": rendered})

        walk(self.data, "")
        return rows


def parse_provenance_response(payload: Mapping[str, Any]) -> ProvenanceView:
    """Validate the FastAPI wrapper and return sanitized public provenance."""

    if not isinstance(payload, Mapping):
        raise ValueError("Provenance response must be a JSON object.")
    if payload.get("status") != "PASSED":
        raise ValueError("Provenance response.status must be PASSED.")
    raw = payload.get("provenance")
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("Provenance response.provenance must be a non-empty object.")
    sanitized = sanitize_public_value(raw)
    if not isinstance(sanitized, dict) or not sanitized:
        raise ValueError("No public provenance remained after safety checks.")
    return ProvenanceView(data=sanitized)


def render_provenance_flow(st: Any) -> None:
    """Render the fixed source-to-screen path in simple words."""

    steps = (
        ("1", "Source", "Public news or a user-supplied document"),
        ("2", "Text", "Safe extraction and input checks"),
        ("3", "Sentiment", "Positive, neutral, and negative meaning"),
        ("4", "Session", "Correct market-session mapping"),
        ("5", "Forecast", "Down, Flat, and Up probabilities"),
        ("6", "Evidence", "Earlier events, drivers, and limitations"),
        ("7", "Display", "Plain-language result and safe downloads"),
    )
    cards = "".join(
        f"""
        <div class="rm-provenance-step">
          <span>{number}</span><strong>{escape(title)}</strong><small>{escape(body)}</small>
        </div>
        """
        for number, title, body in steps
    )
    st.markdown(
        f'<section class="rm-provenance-flow">{cards}</section>',
        unsafe_allow_html=True,
    )


def render_provenance_view(st: Any, view: ProvenanceView) -> None:
    """Render verification status, grouped evidence, and exact fallback rows."""

    st.markdown(
        """
        <section class="rm-verification-hero">
          <div class="rm-verification-seal" aria-hidden="true">✓</div>
          <div>
            <p class="rm-panel-kicker">LICENCE-SAFE EVIDENCE</p>
            <h2>Verified project provenance</h2>
            <p>FastAPI returned the approved public evidence boundary.</p>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    render_provenance_flow(st)

    st.markdown("### Recorded evidence")
    for key in sorted(view.data):
        with st.expander(key.replace("_", " ").title()):
            value = view.data[key]
            if isinstance(value, (dict, list)):
                st.json(value, expanded=True)
            else:
                st.write(value)

    with st.expander("Open the accessible evidence table"):
        st.dataframe(view.rows(), use_container_width=True, hide_index=True)

    st.markdown(
        """
        <section class="rm-evidence-note">
          <strong>Conclusion</strong><br>
          The page shows the recorded source, model, checks, and use limits that
          FastAPI marked safe for public display. Missing fields are not guessed.
        </section>
        """,
        unsafe_allow_html=True,
    )
