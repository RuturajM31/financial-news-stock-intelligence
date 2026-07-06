"""Render licence-safe source, model, and verification evidence."""

from __future__ import annotations

from typing import Any

from app.components.loading_states import loading_context
from app.components.provenance_panels import (
    parse_provenance_response,
    render_provenance_view,
)
from app.components.report_downloads import render_provenance_download
from app.components.status_badges import render_problem
from app.services.api_client import FinancialNewsApiClient, StreamlitApiError


def render_provenance_page(st: Any, api_client: FinancialNewsApiClient) -> None:
    """Load, sanitize, display, and download FastAPI public provenance."""

    st.markdown("## Evidence & Verification")
    st.write(
        "See how the application connects sources, models, checks, and use "
        "limits without exposing private provider data."
    )
    try:
        with loading_context(st, "provenance"):
            payload = api_client.provenance()
        view = parse_provenance_response(payload)
    except StreamlitApiError as error:
        render_problem(st, error.problem)
        return
    except ValueError as error:
        st.error(
            "The provenance result did not match the approved public contract. "
            f"Reason: {error}"
        )
        return

    render_provenance_view(st, view)
    render_provenance_download(st, view)
    st.caption(
        "Private tokens, local file paths, raw Tiingo values, and restricted "
        "provider rows are not shown or included in the download."
    )
