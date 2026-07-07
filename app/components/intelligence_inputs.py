"""Collect and validate one movement-intelligence request.

Purpose
-------
Forecast and historical pages need the same article text, ticker, and publication
moment. This module keeps those rules in one place so both pages send consistent,
timezone-aware requests to FastAPI.

Privacy boundary
----------------
The returned text is used only for the current button action. Callers must not
store article text in Streamlit session history, logs, or downloadable evidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


MAXIMUM_TEXT_CHARACTERS = 20_000
TICKER_PATTERN = re.compile(r"^[A-Z][A-Z0-9.\-]{0,14}$")
SUPPORTED_TIMEZONES = (
    "UTC",
    "Europe/Berlin",
    "America/New_York",
    "Europe/London",
    "Asia/Kolkata",
)


@dataclass(frozen=True)
class IntelligenceSubmission:
    """Store one checked request that can be sent to FastAPI.

    Attributes:
        text: Current article text. It must not be stored after the request.
        ticker: Uppercase market ticker accepted by the FastAPI schema.
        published_at: ISO-8601 timestamp including an explicit timezone offset.
    """

    text: str
    ticker: str
    published_at: str


def validate_article_text(value: str) -> str:
    """Return stripped article text or raise a clear input error."""

    if not isinstance(value, str):
        raise ValueError("Article text must be text.")
    normalized = value.strip()
    if not normalized:
        raise ValueError("Add the financial-news text before running the analysis.")
    if len(normalized) > MAXIMUM_TEXT_CHARACTERS:
        raise ValueError(
            "Article text is too long. Use no more than "
            f"{MAXIMUM_TEXT_CHARACTERS:,} characters."
        )
    return normalized


def validate_ticker(value: str) -> str:
    """Normalize one ticker and reject unsupported characters."""

    if not isinstance(value, str):
        raise ValueError("Ticker must be text.")
    normalized = value.strip().upper()
    if not normalized:
        raise ValueError("Add a ticker before running the analysis.")
    if not TICKER_PATTERN.fullmatch(normalized):
        raise ValueError(
            "Ticker may contain uppercase letters, numbers, a dot, or a dash."
        )
    return normalized


def build_published_at(
    publication_date: date,
    publication_time: time,
    timezone_name: str,
) -> str:
    """Build one timezone-aware ISO timestamp for the FastAPI request.

    The model maps news to a market session. A timezone is therefore required;
    silently treating a local clock value as UTC could select the wrong session.
    """

    if timezone_name not in SUPPORTED_TIMEZONES:
        raise ValueError("Choose one of the supported publication timezones.")
    try:
        timezone = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as error:
        raise ValueError(
            "The selected timezone is not available in this Python environment."
        ) from error
    combined = datetime.combine(publication_date, publication_time, timezone)
    return combined.isoformat()


def render_intelligence_inputs(
    st: Any,
    *,
    form_key: str,
    button_label: str,
) -> IntelligenceSubmission | None:
    """Render one bounded form and return data only after its button is pressed."""

    with st.form(form_key, clear_on_submit=False):
        text = st.text_area(
            "Financial-news text",
            height=220,
            max_chars=MAXIMUM_TEXT_CHARACTERS,
            placeholder=(
                "Paste the news text used for the movement and historical analysis."
            ),
            help=(
                "The text is sent to the verified FastAPI service for this request. "
                "It is not kept in session history."
            ),
        )
        first, second = st.columns(2)
        with first:
            ticker = st.text_input(
                "Ticker",
                max_chars=15,
                placeholder="AAPL",
                help="Use the market ticker for the company in the news.",
            )
            publication_date = st.date_input(
                "Publication date",
                value=date.today(),
            )
        with second:
            publication_time = st.time_input(
                "Publication time",
                value=time(hour=9, minute=0),
                step=60,
            )
            timezone_name = st.selectbox(
                "Publication timezone",
                SUPPORTED_TIMEZONES,
                index=1,
                help=(
                    "The timezone is required so the news is mapped to the correct "
                    "market session."
                ),
            )
        submitted = st.form_submit_button(button_label, use_container_width=True)

    if not submitted:
        return None
    return IntelligenceSubmission(
        text=validate_article_text(text),
        ticker=validate_ticker(ticker),
        published_at=build_published_at(
            publication_date,
            publication_time,
            timezone_name,
        ),
    )
