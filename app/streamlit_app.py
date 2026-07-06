"""Run the portfolio-quality Streamlit application through Package 7.

The entry point loads the approved design, builds one safe FastAPI client, and
routes all completed pages. Model inference and historical retrieval remain in
the isolated FastAPI workers; Streamlit only validates and presents responses.

Run from the project root with:
    .venv-streamlit/bin/python -m streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from app.branding import get_portfolio_brand
from app.components.status_badges import render_configuration_problem
from app.configuration import get_app_settings
from app.layout import (
    apply_premium_theme,
    render_page_header,
    render_planned_page,
    render_portfolio_footer,
)
from app.navigation import get_navigation_item, render_sidebar_navigation
from app.pages.about_ruturaj import render_about_ruturaj_page
from app.pages.analyze import render_analyze_page
from app.pages.executive_overview import render_executive_overview
from app.pages.explainability import render_explainability_page
from app.pages.forecasts import render_forecasts_page
from app.pages.historical_intelligence import render_historical_intelligence_page
from app.pages.model_comparison import render_model_comparison_page
from app.pages.model_training import render_model_training_page
from app.pages.provenance import render_provenance_page
from app.pages.scenario_analysis import render_scenario_analysis_page
from app.services.api_client import FinancialNewsApiClient
from app.services.api_settings import get_api_client_settings
from app.services.session_state import initialize_session_state


def _read_optional_streamlit_secrets() -> Mapping[str, Any]:
    """Return private Streamlit settings or an empty mapping when none exist."""

    try:
        return st.secrets.to_dict()
    except FileNotFoundError:
        return {}
    except Exception as error:
        if type(error).__name__ == "StreamlitSecretNotFoundError":
            return {}
        raise


def _build_api_client() -> tuple[FinancialNewsApiClient | None, str | None]:
    """Build one immutable API client or return a safe settings error."""

    try:
        settings = get_api_client_settings(
            secrets=_read_optional_streamlit_secrets()
        )
        return FinancialNewsApiClient(settings), None
    except ValueError as error:
        return None, str(error)


def _require_client(
    api_client: FinancialNewsApiClient | None,
    api_settings_error: str | None,
) -> FinancialNewsApiClient | None:
    """Render one configuration problem and return no client when unavailable."""

    if api_client is not None:
        return api_client
    render_configuration_problem(
        st,
        api_settings_error
        or "The FastAPI settings were not available for this protected page.",
    )
    return None


def main() -> None:
    """Render one complete Streamlit page rerun with fail-closed routing."""

    app_settings = get_app_settings(PROJECT_ROOT)
    brand = get_portfolio_brand()

    st.set_page_config(
        page_title=app_settings.page_title,
        page_icon=app_settings.page_icon,
        layout=app_settings.layout,
        initial_sidebar_state=app_settings.initial_sidebar_state,
    )
    apply_premium_theme(st, app_settings)
    initialize_session_state(st.session_state)
    selected_key = render_sidebar_navigation(st, brand)
    selected_page = get_navigation_item(selected_key)
    api_client, api_settings_error = _build_api_client()

    render_page_header(st, brand, selected_page)
    if selected_key == "executive_overview":
        render_executive_overview(
            st,
            brand,
            api_client=api_client,
            api_settings_error=api_settings_error,
        )
    elif selected_key == "analyze":
        client = _require_client(api_client, api_settings_error)
        if client is not None:
            render_analyze_page(st, client)
    elif selected_key == "forecasts":
        client = _require_client(api_client, api_settings_error)
        if client is not None:
            render_forecasts_page(st, client)
    elif selected_key == "historical_intelligence":
        client = _require_client(api_client, api_settings_error)
        if client is not None:
            render_historical_intelligence_page(st, client)
    elif selected_key == "explainability":
        client = _require_client(api_client, api_settings_error)
        if client is not None:
            render_explainability_page(st, client)
    elif selected_key == "model_training":
        render_model_training_page(st)
    elif selected_key == "model_comparison":
        client = _require_client(api_client, api_settings_error)
        if client is not None:
            render_model_comparison_page(st, client)
    elif selected_key == "scenario_analysis":
        client = _require_client(api_client, api_settings_error)
        if client is not None:
            render_scenario_analysis_page(st, client, brand)
    elif selected_key == "provenance":
        client = _require_client(api_client, api_settings_error)
        if client is not None:
            render_provenance_page(st, client)
    elif selected_key == "about_ruturaj":
        render_about_ruturaj_page(st)
    else:
        render_planned_page(st, selected_page)

    render_portfolio_footer(st, brand)


if __name__ == "__main__":
    main()
