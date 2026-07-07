"""Collect one checked explanation request without storing submitted text."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, time
from typing import Any

from app.components.intelligence_inputs import (
    MAXIMUM_TEXT_CHARACTERS,
    SUPPORTED_TIMEZONES,
    build_published_at,
    validate_article_text,
    validate_ticker,
)


@dataclass(frozen=True)
class ExplainabilitySubmission:
    """Store transient input for one FastAPI explanation request."""

    text: str
    ticker: str
    published_at: str
    top_n: int


def render_explainability_inputs(st: Any) -> ExplainabilitySubmission | None:
    """Render a bounded form and return data only after explicit submission."""

    with st.form("rm_explainability_form", clear_on_submit=False):
        text = st.text_area(
            "Financial-news text",
            height=220,
            max_chars=MAXIMUM_TEXT_CHARACTERS,
            placeholder="Paste the news text used for the movement explanation.",
            help=(
                "The text is sent to FastAPI for this request. It is not stored "
                "in Streamlit session history."
            ),
        )
        first, second = st.columns(2)
        with first:
            ticker = st.text_input(
                "Ticker",
                max_chars=15,
                placeholder="AAPL",
            )
            publication_date = st.date_input(
                "Publication date",
                value=date.today(),
                key="rm_explainability_date",
            )
        with second:
            publication_time = st.time_input(
                "Publication time",
                value=time(hour=9, minute=0),
                step=60,
                key="rm_explainability_time",
            )
            timezone_name = st.selectbox(
                "Publication timezone",
                SUPPORTED_TIMEZONES,
                index=1,
                key="rm_explainability_timezone",
            )
        top_n = st.slider(
            "Number of drivers",
            min_value=3,
            max_value=12,
            value=6,
            help="Choose how many general and current-result drivers to show.",
        )
        submitted = st.form_submit_button(
            "Explain this forecast",
            use_container_width=True,
        )

    if not submitted:
        return None
    return ExplainabilitySubmission(
        text=validate_article_text(text),
        ticker=validate_ticker(ticker),
        published_at=build_published_at(
            publication_date,
            publication_time,
            timezone_name,
        ),
        top_n=top_n,
    )
