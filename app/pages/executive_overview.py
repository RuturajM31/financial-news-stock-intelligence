"""Render the Executive Overview for the verified Streamlit product.

The page combines live FastAPI health, approved model roles, current delivery
status, and small current-session analysis summaries. It does not load models or
read model artifacts inside Streamlit.
"""

from __future__ import annotations

from html import escape
from typing import Any

from app.branding import PortfolioBrand
from app.components.result_cards import render_result_card
from app.components.status_badges import (
    render_api_connection_panel,
    render_configuration_problem,
)
from app.services.api_client import FinancialNewsApiClient
from app.services.session_state import ANALYSIS_HISTORY_KEY


def _render_model_role_cards(st: Any) -> None:
    """Show verified model roles without overstating unfinished comparisons."""

    st.markdown(
        """
        <section class="rm-section-heading rm-section-gap">
          <p class="rm-eyebrow">VERIFIED MODEL ROLES</p>
          <h2>Different winners for quality and live use</h2>
          <p>
            The project separates the model with the best tested quality from
            the model that gives the best balance for a live application.
          </p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    columns = st.columns(3, gap="medium")
    with columns[0]:
        render_result_card(
            st,
            "Live sentiment model",
            "DistilBERT",
            (
                "Selected for the live service because it remains strong while "
                "using less memory and responding faster than full BERT."
            ),
            tone="verified",
            eyebrow="DEPLOYMENT CHAMPION",
        )
    with columns[1]:
        render_result_card(
            st,
            "Sentiment quality benchmark",
            "BERT",
            (
                "Produced the highest tested sentiment quality and remains the "
                "reference model for the full comparison page."
            ),
            tone="info",
            eyebrow="QUALITY CHAMPION",
        )
    with columns[2]:
        render_result_card(
            st,
            "Movement model",
            "Stability Soft Vote",
            (
                "The verified movement champion is recorded technically as "
                "stability_soft_vote_rf_sgd."
            ),
            tone="positive",
            eyebrow="MOVEMENT CHAMPION",
        )


def _render_product_flow(st: Any) -> None:
    """Explain the end-to-end user journey in simple product language."""

    st.markdown(
        """
        <section class="rm-panel rm-panel-primary rm-section-gap">
          <div class="rm-panel-kicker">HOW THE PRODUCT WORKS</div>
          <h3>From news input to a clear research result</h3>
          <div class="rm-explanation-grid">
            <div>
              <span>1 · ADD NEWS</span>
              <strong>Paste text, upload a file, or add a public URL.</strong>
            </div>
            <div>
              <span>2 · UNDERSTAND THE TONE</span>
              <strong>See bearish, neutral, and bullish probabilities.</strong>
            </div>
            <div>
              <span>3 · REVIEW THE FORECAST</span>
              <strong>See Down, Flat, and Up chances in Package 4.</strong>
            </div>
          </div>
          <p>
            Every result explains what it shows, why it matters, and the clear
            conclusion. Uncertainty is stated directly rather than hidden.
          </p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _render_recent_session_history(st: Any) -> None:
    """Show bounded, licence-safe summaries from the current browser session."""

    value = st.session_state.get(ANALYSIS_HISTORY_KEY, [])
    if not isinstance(value, list):
        st.warning("The current-session analysis history could not be read safely.")
        return

    st.markdown(
        """
        <section class="rm-section-heading rm-section-gap">
          <p class="rm-eyebrow">CURRENT SESSION</p>
          <h2>Recent sentiment checks</h2>
          <p>
            Only small result summaries are kept in this browser session. Raw
            article text, file bytes, API keys, and provider data are not stored.
          </p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    if not value:
        st.info(
            "No analysis has been completed in this session. Open Analyze News "
            "from the sidebar to create the first result."
        )
        return

    for record in reversed(value[-5:]):
        if not isinstance(record, dict):
            continue
        label = escape(str(record.get("label", "Unknown")))
        confidence = record.get("confidence_percent")
        confidence_text = (
            f"{float(confidence):.1f}%"
            if isinstance(confidence, (int, float)) and not isinstance(confidence, bool)
            else "Unknown"
        )
        source = escape(str(record.get("source_kind", "Unknown source")))
        result_count = record.get("result_count", 1)
        count_text = (
            str(result_count)
            if isinstance(result_count, int) and not isinstance(result_count, bool)
            else "1"
        )
        st.markdown(
            f"""
            <section class="rm-panel">
              <div class="rm-panel-kicker">{source.upper()}</div>
              <h3>{label} · {confidence_text}</h3>
              <p>Results returned: {escape(count_text)}. Stored as a summary only.</p>
            </section>
            """,
            unsafe_allow_html=True,
        )


def _render_package_status(st: Any, brand: PortfolioBrand) -> None:
    """Show the current verified Streamlit delivery boundary."""

    st.markdown(
        f"""
        <section class="rm-panel rm-roadmap-panel rm-section-gap">
          <div class="rm-panel-kicker">STREAMLIT DELIVERY STATUS</div>
          <h3>Packages 1, 2, and 3</h3>
          <p>
            The visual foundation and FastAPI connection are verified. This
            package adds the Executive Overview and live sentiment analysis.
            Forecasts, history, explanations, model comparison, scenarios, and
            evidence pages remain clearly marked for later packages.
          </p>
          <div class="rm-verified-line">
            <span aria-hidden="true">✓</span>
            Designed and built by {escape(brand.owner_name)}
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_executive_overview(
    st: Any,
    brand: PortfolioBrand,
    *,
    api_client: FinancialNewsApiClient | None,
    api_settings_error: str | None,
) -> None:
    """Render one complete overview with live status and honest next steps."""

    st.markdown(
        """
        <section class="rm-section-heading">
          <p class="rm-eyebrow">EXECUTIVE OVERVIEW</p>
          <h2>Financial news intelligence that explains its conclusions</h2>
          <p>
            Analyze financial wording through the verified FastAPI service,
            review model confidence, and understand the limits before using the
            result as research evidence.
          </p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    _render_product_flow(st)

    if api_client is not None:
        render_api_connection_panel(st, api_client)
    elif api_settings_error:
        render_configuration_problem(st, api_settings_error)
    else:
        render_configuration_problem(
            st,
            "The FastAPI settings were not available for the live service check.",
        )

    _render_model_role_cards(st)
    _render_recent_session_history(st)
    _render_package_status(st, brand)
