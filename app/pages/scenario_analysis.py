"""Render transparent downside, central, and upside research scenarios."""

from __future__ import annotations

from typing import Any

from app.branding import PortfolioBrand
from app.components.loading_states import loading_context
from app.components.report_downloads import render_scenario_downloads
from app.components.scenario_inputs import render_scenario_inputs
from app.components.scenario_results import parse_scenario_response, render_scenario_result
from app.components.status_badges import render_problem
from app.services.api_client import FinancialNewsApiClient, StreamlitApiError
from app.services.session_state import add_analysis_summary


def render_scenario_analysis_page(
    st: Any,
    api_client: FinancialNewsApiClient,
    brand: PortfolioBrand,
) -> None:
    """Collect assumptions, call FastAPI, and explain the verified range."""

    st.markdown("## Scenario Analysis")
    st.write(
        "Explore three historical research examples using your investment amount, "
        "share price, costs, and optional tax assumption."
    )
    st.warning(
        "This page is a research calculator. It is not investment advice and does "
        "not predict an exact future return."
    )

    try:
        submission = render_scenario_inputs(st)
    except ValueError as error:
        st.warning(str(error))
        return
    if submission is None:
        st.info(
            "Add the news, market timing, and portfolio assumptions, then build "
            "the three research scenarios."
        )
        return

    try:
        with loading_context(st, "scenario_analysis"):
            response = api_client.scenario_analysis(submission.as_api_payload())
        view = parse_scenario_response(
            response,
            currency=submission.currency,
            investment_amount=submission.investment_amount,
        )
    except StreamlitApiError as error:
        render_problem(st, error.problem)
        return
    except ValueError as error:
        st.error(
            "The scenario result did not match the approved display contract. "
            f"Reason: {error}"
        )
        return

    render_scenario_result(st, view)
    render_scenario_downloads(st, view, brand)
    add_analysis_summary(
        st.session_state,
        {
            "analysis_type": "scenario",
            "ticker": view.prediction.ticker,
            "target_session_date": view.prediction.target_session_date,
            "direction": view.prediction.direction,
            "evidence_count": view.evidence_count,
        },
    )
