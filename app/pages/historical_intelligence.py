"""Render strictly earlier historical evidence for one financial-news event."""

from __future__ import annotations

from datetime import date
from html import escape
from typing import Any, Mapping, Sequence

from app.components.historical_charts import (
    render_outcome_balance,
    render_phrase_chips,
    render_return_range,
)
from app.components.historical_event_cards import (
    HistoricalMatchView,
    parse_historical_matches,
    render_historical_event_cards,
)
from app.components.intelligence_inputs import render_intelligence_inputs
from app.components.loading_states import loading_context
from app.components.status_badges import render_problem
from app.services.api_client import FinancialNewsApiClient, StreamlitApiError
from app.services.api_contracts import ApiProblemDetails


MAXIMUM_MATCHES = 20


def _require_text(mapping: Mapping[str, Any], key: str, location: str) -> str:
    """Read one non-empty historical response field."""

    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location}.{key} must be non-empty text.")
    return value.strip()


def _parse_text_list(value: Any, location: str) -> tuple[str, ...]:
    """Validate one list of limitation or phrase text."""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{location} must be a list of text values.")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{location} contains an invalid value.")
        result.append(item.strip())
    return tuple(result)


def parse_historical_response(
    payload: Mapping[str, Any],
) -> tuple[
    str,
    str,
    tuple[HistoricalMatchView, ...],
    tuple[str, ...],
    tuple[str, ...],
    str,
    str,
]:
    """Validate the response and return all fields used by the page."""

    if not isinstance(payload, Mapping):
        raise ValueError("historical response must be a JSON object.")
    if payload.get("status") != "PASSED":
        raise ValueError("historical response.status must be PASSED.")
    ticker = _require_text(payload, "ticker", "historical response")
    query_target_session_date = _require_text(
        payload,
        "query_target_session_date",
        "historical response",
    )
    try:
        query_date = date.fromisoformat(query_target_session_date)
    except ValueError as error:
        raise ValueError(
            "historical response.query_target_session_date must be an ISO date."
        ) from error

    matches = parse_historical_matches(payload.get("matches"))
    article_ids: set[str] = set()
    for match in matches:
        if match.ticker != ticker:
            raise ValueError("Every historical match must use the response ticker.")
        if date.fromisoformat(match.target_session_date) >= query_date:
            raise ValueError(
                "Every historical match must be earlier than the query session."
            )
        if match.article_id in article_ids:
            raise ValueError("Historical matches must not contain duplicate events.")
        article_ids.add(match.article_id)

    return (
        ticker,
        query_target_session_date,
        matches,
        _parse_text_list(
            payload.get("important_phrases", []),
            "historical response.important_phrases",
        ),
        _parse_text_list(
            payload.get("limitations"),
            "historical response.limitations",
        ),
        _require_text(payload, "reference_scope", "historical response"),
        _require_text(payload, "disclaimer", "historical response"),
    )


def _render_problem(st: Any, reason: str, location: str) -> None:
    """Render a safe four-part message for local page failures."""

    render_problem(
        st,
        ApiProblemDetails(
            error_code="streamlit_historical_page_failed",
            what_failed="The historical evidence could not be displayed.",
            where_failed=location,
            why_failed=reason,
            safe_next_step=(
                "Correct the request or restore the verified historical response, "
                "then run the search again."
            ),
        ),
    )


def _render_scope_summary(
    st: Any,
    *,
    ticker: str,
    target_date: str,
    evidence_count: int,
    reference_scope: str,
) -> None:
    """Show the earlier-only boundary before rendering any historical result."""

    st.markdown(
        f"""
        <section class="rm-panel rm-history-summary">
          <div>
            <p class="rm-panel-kicker">EARLIER-ONLY EVIDENCE</p>
            <h3>{escape(ticker)} · {evidence_count} matched events</h3>
            <p>Current target session · {escape(target_date)}</p>
          </div>
          <div class="rm-verification-badge">Future events excluded</div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"Reference scope: {reference_scope.replace('_', ' ')}")


def render_historical_intelligence_page(
    st: Any,
    client: FinancialNewsApiClient,
) -> None:
    """Collect one event and render strictly earlier matched evidence."""

    st.markdown(
        """
        <section class="rm-section-heading">
          <p class="rm-eyebrow">HISTORICAL INTELLIGENCE</p>
          <h2>Compare this event with verified earlier events</h2>
          <p>
            The search uses the current event only as a query. Every displayed
            match must come from an earlier target session, which protects the
            result from future information.
          </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    first, second = st.columns(2)
    with first:
        limit = st.slider(
            "Maximum earlier events",
            min_value=1,
            max_value=MAXIMUM_MATCHES,
            value=5,
        )
    with second:
        minimum_similarity_percent = st.slider(
            "Minimum similarity",
            min_value=0,
            max_value=100,
            value=0,
            help="A higher value returns fewer but more similar events.",
        )

    try:
        submission = render_intelligence_inputs(
            st,
            form_key="rm_historical_form",
            button_label="Find earlier events",
        )
    except ValueError as error:
        _render_problem(st, str(error), "Historical Intelligence input")
        return
    if submission is None:
        st.info(
            "Add the event details above. The page will show only earlier matches "
            "returned by the verified FastAPI service."
        )
        return

    try:
        with loading_context(st, "Finding similar earlier events..."):
            payload = client.historical_intelligence(
                submission.text,
                submission.ticker,
                submission.published_at,
                limit=limit,
                minimum_similarity=minimum_similarity_percent / 100,
            )
        (
            ticker,
            target_date,
            matches,
            phrases,
            limitations,
            reference_scope,
            disclaimer,
        ) = parse_historical_response(payload)
        if ticker != submission.ticker:
            raise ValueError(
                "The historical response ticker does not match the submitted ticker."
            )
    except StreamlitApiError as error:
        render_problem(st, error.problem)
        return
    except ValueError as error:
        _render_problem(
            st,
            str(error),
            "Streamlit historical response validation",
        )
        return

    _render_scope_summary(
        st,
        ticker=ticker,
        target_date=target_date,
        evidence_count=len(matches),
        reference_scope=reference_scope,
    )
    render_return_range(st, matches)
    render_outcome_balance(st, matches)
    render_phrase_chips(st, phrases)

    st.markdown(
        """
        <section class="rm-section-heading rm-section-gap">
          <p class="rm-eyebrow">MATCHED EVENTS</p>
          <h2>Earlier evidence used for comparison</h2>
        </section>
        """,
        unsafe_allow_html=True,
    )
    render_historical_event_cards(st, matches)

    for limitation in limitations:
        st.warning(limitation)
    st.caption(disclaimer)
    st.info(
        "A rotatable 3D event-cluster view will be added with the shared advanced "
        "visual system. This page already provides the required clear 2D view."
    )
