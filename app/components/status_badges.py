"""Render clear service states and safe four-part error messages."""

from __future__ import annotations

from html import escape
from typing import Any

from app.components.loading_states import loading_context
from app.components.result_cards import render_result_card
from app.services.api_client import FinancialNewsApiClient, StreamlitApiError
from app.services.api_contracts import ApiProblemDetails
from app.services.session_state import get_api_check, store_api_check


_ALLOWED_STATUS_TONES = frozenset({"passed", "failed", "warning", "unknown"})


def build_status_badge_html(label: str, tone: str) -> str:
    """Build one accessible text badge with an approved status tone."""

    if tone not in _ALLOWED_STATUS_TONES:
        raise ValueError(f"Unsupported status-badge tone: {tone!r}.")
    if not label.strip():
        raise ValueError("Status-badge text must not be empty.")
    return (
        f'<span class="rm-status-badge rm-status-{escape(tone)}">'
        f'{escape(label.strip())}</span>'
    )


def render_problem(st: Any, problem: ApiProblemDetails) -> None:
    """Show what, where, why, and the exact safe next step."""

    request_line = ""
    if problem.request_id:
        request_line = (
            "<p><strong>Request ID:</strong> "
            f"{escape(problem.request_id)}</p>"
        )
    st.markdown(
        f"""
        <section class="rm-panel rm-problem-panel">
          {build_status_badge_html("Connection failed", "failed")}
          <h3>{escape(problem.what_failed)}</h3>
          <p><strong>Where:</strong> {escape(problem.where_failed)}</p>
          <p><strong>Why:</strong> {escape(problem.why_failed)}</p>
          <p><strong>Safe next step:</strong> {escape(problem.safe_next_step)}</p>
          {request_line}
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_configuration_problem(
    st: Any,
    reason: str,
) -> None:
    """Show a safe API-setting failure without exposing secret values."""

    render_problem(
        st,
        ApiProblemDetails(
            error_code="streamlit_api_settings_invalid",
            what_failed="The live-service settings could not be loaded.",
            where_failed="Streamlit FastAPI settings",
            why_failed=reason,
            safe_next_step=(
                "Correct the named private setting and restart Streamlit."
            ),
        ),
    )


def _run_connection_check(
    client: FinancialNewsApiClient,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Check public health and lightweight readiness in a fixed order."""

    health = client.health()
    readiness = client.readiness()
    return health.as_dict(), readiness.as_dict()


def render_api_connection_panel(
    st: Any,
    client: FinancialNewsApiClient,
    *,
    auto_check: bool = True,
) -> None:
    """Render one live FastAPI status panel and keep its result in the session.

    The first Executive Overview visit runs one check automatically. Later
    Streamlit reruns reuse the current-session result until the user presses the
    refresh button. No model inference is started by these public endpoints.
    """

    st.markdown(
        """
        <section class="rm-section-heading rm-section-gap">
          <p class="rm-eyebrow">LIVE SERVICE CONNECTION</p>
          <h2>Is the verified backend ready?</h2>
          <p>
            This check confirms that Streamlit can reach FastAPI and that the
            approved files and worker entry points are ready. It does not run a
            prediction or expose a private key.
          </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    existing = get_api_check(st.session_state)
    refresh_requested = st.button(
        "Check live services",
        key="rm_check_live_services",
        type="secondary",
    )
    should_check = refresh_requested or (auto_check and existing is None)

    if should_check:
        try:
            with loading_context(st, "check_services"):
                health, readiness = _run_connection_check(client)
            store_api_check(
                st.session_state,
                health=health,
                readiness=readiness,
                problem=None,
            )
        except StreamlitApiError as error:
            store_api_check(
                st.session_state,
                health=None,
                readiness=None,
                problem=error.problem.as_dict(),
            )
        existing = get_api_check(st.session_state)

    if existing is None:
        st.info("Press ‘Check live services’ to test the FastAPI connection.")
        return

    problem = existing.get("problem")
    if isinstance(problem, dict):
        render_problem(
            st,
            ApiProblemDetails(
                error_code=str(problem.get("error_code") or "api_check_failed"),
                what_failed=str(
                    problem.get("what_failed") or "The API check failed."
                ),
                where_failed=str(
                    problem.get("where_failed") or "Streamlit API connection"
                ),
                why_failed=str(
                    problem.get("why_failed") or "No safe reason was returned."
                ),
                safe_next_step=str(
                    problem.get("safe_next_step")
                    or "Check FastAPI and try the connection again."
                ),
                request_id=(
                    str(problem["request_id"])
                    if problem.get("request_id")
                    else None
                ),
            ),
        )
        return

    health = existing.get("health")
    readiness = existing.get("readiness")
    if not isinstance(health, dict) or not isinstance(readiness, dict):
        render_configuration_problem(
            st,
            "The saved connection result has an invalid shape.",
        )
        return

    columns = st.columns(2, gap="medium")
    with columns[0]:
        render_result_card(
            st,
            "FastAPI process",
            "Healthy",
            (
                f"Service {health.get('service', 'unknown')} is answering. "
                f"Version: {health.get('version', 'unknown')}."
            ),
            tone="verified",
            eyebrow="LIVE CHECK",
        )
    components = readiness.get("components")
    component_values = (
        list(components.values()) if isinstance(components, dict) else []
    )
    all_ready = bool(component_values) and all(
        value == "PASSED" for value in component_values
    )
    with columns[1]:
        render_result_card(
            st,
            "Backend readiness",
            "Ready" if all_ready else "Needs attention",
            (
                "All reported readiness checks passed."
                if all_ready
                else "One or more readiness checks did not report PASSED."
            ),
            tone="verified" if all_ready else "neutral",
            eyebrow="LIGHTWEIGHT CHECK",
        )

    conclusion = (
        "Streamlit can reach the verified FastAPI backend. Analysis pages may "
        "use this connection once their packages are installed."
        if all_ready
        else "FastAPI answered, but the readiness details need review before analysis."
    )
    badge_label = "Connection verified" if all_ready else "Review needed"
    badge_tone = "passed" if all_ready else "warning"
    badge_html = build_status_badge_html(badge_label, badge_tone)
    st.markdown(
        f"""
        <section class="rm-panel rm-section-gap">
          {badge_html}
          <h3>Conclusion</h3>
          <p>{escape(conclusion)}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
