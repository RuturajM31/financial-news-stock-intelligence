"""Render the live Down, Flat, and Up forecast workflow."""

from __future__ import annotations

from typing import Any

from app.components.intelligence_inputs import render_intelligence_inputs
from app.components.loading_states import loading_context
from app.components.movement_results import (
    MovementView,
    parse_movement_response,
    render_market_session_timeline,
    render_movement_result,
)
from app.components.status_badges import render_problem
from app.services.api_client import FinancialNewsApiClient, StreamlitApiError
from app.services.api_contracts import ApiProblemDetails
from app.services.session_state import add_analysis_summary


CURRENT_FORECAST_KEY = "rm_current_movement_forecast"


def _render_local_problem(st: Any, reason: str, location: str) -> None:
    """Render one safe local validation or response error."""

    render_problem(
        st,
        ApiProblemDetails(
            error_code="streamlit_movement_page_failed",
            what_failed="The forecast could not be displayed.",
            where_failed=location,
            why_failed=reason,
            safe_next_step=(
                "Correct the input or restore the verified FastAPI movement "
                "response, then run the forecast again."
            ),
        ),
    )


def _store_safe_forecast(
    state: Any,
    view: MovementView,
    published_at: str,
) -> None:
    """Store only result fields; article text is deliberately excluded."""

    summary = view.as_summary()
    summary["published_at"] = published_at
    state[CURRENT_FORECAST_KEY] = summary
    add_analysis_summary(state, summary)


def _render_saved_forecast(st: Any, state: Any) -> None:
    """Show a compact reminder when a safe result exists in this session."""

    saved = state.get(CURRENT_FORECAST_KEY)
    if not isinstance(saved, dict):
        return
    st.caption(
        "Current session result: "
        f"{saved.get('ticker', 'Ticker')} · {saved.get('direction', 'Result')} · "
        f"{saved.get('confidence_percent', 0):.1f}% support."
    )


def render_forecasts_page(st: Any, client: FinancialNewsApiClient) -> None:
    """Collect one event, call FastAPI, and explain the movement result."""

    st.markdown(
        """
        <section class="rm-section-heading">
          <p class="rm-eyebrow">MARKET MOVEMENT FORECAST</p>
          <h2>See the model's Down, Flat, and Up view</h2>
          <p>
            Add the news text, ticker, and exact publication time. FastAPI maps
            the event to a verified market session and runs the saved movement
            model outside the Streamlit process.
          </p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "Research use only. The result describes model evidence and is not "
        "investment advice."
    )
    _render_saved_forecast(st, st.session_state)

    try:
        submission = render_intelligence_inputs(
            st,
            form_key="rm_forecast_form",
            button_label="Run movement forecast",
        )
    except ValueError as error:
        _render_local_problem(st, str(error), "Forecast input")
        return
    if submission is None:
        st.info(
            "Add a verified historical event above. Article text is not saved in "
            "the browser session after the request."
        )
        return

    try:
        with loading_context(st, "Generating the movement forecast..."):
            payload = client.movement_prediction(
                submission.text,
                submission.ticker,
                submission.published_at,
            )
        view = parse_movement_response(payload)
        if view.ticker != submission.ticker:
            raise ValueError(
                "The movement response ticker does not match the submitted ticker."
            )
    except StreamlitApiError as error:
        render_problem(st, error.problem)
        return
    except ValueError as error:
        _render_local_problem(
            st,
            str(error),
            "Streamlit movement response validation",
        )
        return

    _store_safe_forecast(st.session_state, view, submission.published_at)
    render_movement_result(st, view)
    render_market_session_timeline(
        st,
        published_at=submission.published_at,
        target_session_date=view.target_session_date,
    )
    st.info(
        "Use Historical Intelligence to compare the same event with strictly "
        "earlier verified events. The article text must be entered again because "
        "the app does not store raw news text."
    )
