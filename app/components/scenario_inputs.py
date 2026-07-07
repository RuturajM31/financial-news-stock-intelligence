"""Collect and validate one research-only investment scenario request.

Purpose
-------
The scenario page combines the verified movement request with user-controlled
portfolio assumptions. This module keeps all input rules in one place and sends
only checked values to FastAPI.

Privacy boundary
----------------
Article text is returned only for the current button action. It must not be
stored in session history, logs, reports, or downloadable evidence.
"""

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

SUPPORTED_CURRENCIES = ("EUR", "USD", "GBP", "INR")
MAXIMUM_INVESTMENT_AMOUNT = 100_000_000.0
MAXIMUM_SHARE_PRICE = 10_000_000.0
MAXIMUM_FEE = 1_000_000.0


@dataclass(frozen=True)
class ScenarioSubmission:
    """Store one validated scenario request for the protected FastAPI route."""

    text: str
    ticker: str
    published_at: str
    investment_amount: float
    share_price: float
    currency: str
    allow_fractional_shares: bool
    share_precision: int
    entry_fee: float
    exit_fee: float
    tax_rate_percent: float | None

    def as_api_payload(self) -> dict[str, Any]:
        """Return the exact request fields accepted by FastAPI."""

        return {
            "text": self.text,
            "ticker": self.ticker,
            "published_at": self.published_at,
            "investment_amount": self.investment_amount,
            "share_price": self.share_price,
            "currency": self.currency,
            "allow_fractional_shares": self.allow_fractional_shares,
            "share_precision": self.share_precision,
            "entry_fee": self.entry_fee,
            "exit_fee": self.exit_fee,
            "tax_rate_percent": self.tax_rate_percent,
        }


def _bounded_positive(value: Any, label: str, maximum: float) -> float:
    """Return one positive finite amount within an approved upper boundary."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number.")
    result = float(value)
    if result <= 0.0:
        raise ValueError(f"{label} must be above zero.")
    if result > maximum:
        raise ValueError(f"{label} is above the approved maximum.")
    return result


def _bounded_fee(value: Any, label: str) -> float:
    """Return one non-negative fee within the approved display boundary."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number.")
    result = float(value)
    if result < 0.0:
        raise ValueError(f"{label} cannot be negative.")
    if result > MAXIMUM_FEE:
        raise ValueError(f"{label} is above the approved maximum.")
    return result


def validate_scenario_values(
    *,
    investment_amount: Any,
    share_price: Any,
    currency: str,
    allow_fractional_shares: bool,
    share_precision: Any,
    entry_fee: Any,
    exit_fee: Any,
    tax_rate_percent: Any,
) -> dict[str, Any]:
    """Validate portfolio assumptions and return normalized API values."""

    investment = _bounded_positive(
        investment_amount,
        "Investment amount",
        MAXIMUM_INVESTMENT_AMOUNT,
    )
    price = _bounded_positive(share_price, "Share price", MAXIMUM_SHARE_PRICE)
    entry = _bounded_fee(entry_fee, "Entry fee")
    exit_value = _bounded_fee(exit_fee, "Exit fee")
    if entry >= investment:
        raise ValueError("Entry fee must be below the investment amount.")

    normalized_currency = str(currency).strip().upper()
    if normalized_currency not in SUPPORTED_CURRENCIES:
        raise ValueError("Choose one of the supported currencies.")

    if isinstance(share_precision, bool) or not isinstance(share_precision, int):
        raise ValueError("Share precision must be a whole number.")
    if not 0 <= share_precision <= 8:
        raise ValueError("Share precision must be between 0 and 8.")

    tax: float | None
    if tax_rate_percent is None:
        tax = None
    else:
        if isinstance(tax_rate_percent, bool) or not isinstance(
            tax_rate_percent,
            (int, float),
        ):
            raise ValueError("Tax rate must be a number.")
        tax = float(tax_rate_percent)
        if not 0.0 <= tax <= 100.0:
            raise ValueError("Tax rate must be between 0 and 100 percent.")

    return {
        "investment_amount": investment,
        "share_price": price,
        "currency": normalized_currency,
        "allow_fractional_shares": bool(allow_fractional_shares),
        "share_precision": share_precision,
        "entry_fee": entry,
        "exit_fee": exit_value,
        "tax_rate_percent": tax,
    }


def render_scenario_inputs(st: Any) -> ScenarioSubmission | None:
    """Render the complete scenario form and return data after submission."""

    with st.form("rm_scenario_form", clear_on_submit=False):
        st.markdown("### News and market timing")
        text = st.text_area(
            "Financial-news text",
            height=190,
            max_chars=MAXIMUM_TEXT_CHARACTERS,
            placeholder="Paste the financial news used for this research scenario.",
            help=(
                "The text is sent to FastAPI for this request only. It is not "
                "included in the report or session history."
            ),
        )
        left, right = st.columns(2)
        with left:
            ticker = st.text_input("Ticker", max_chars=15, placeholder="AAPL")
            publication_date = st.date_input("Publication date", value=date.today())
        with right:
            publication_time = st.time_input(
                "Publication time",
                value=time(hour=9, minute=0),
                step=60,
            )
            timezone_name = st.selectbox(
                "Publication timezone",
                SUPPORTED_TIMEZONES,
                index=1,
            )

        st.markdown("### Portfolio assumptions")
        first, second, third = st.columns(3)
        with first:
            investment_amount = st.number_input(
                "Investment amount",
                min_value=1.0,
                max_value=MAXIMUM_INVESTMENT_AMOUNT,
                value=10_000.0,
                step=100.0,
            )
            currency = st.selectbox("Currency", SUPPORTED_CURRENCIES, index=0)
        with second:
            share_price = st.number_input(
                "Share price",
                min_value=0.01,
                max_value=MAXIMUM_SHARE_PRICE,
                value=100.0,
                step=1.0,
            )
            allow_fractional_shares = st.checkbox(
                "Allow fractional shares",
                value=True,
            )
        with third:
            share_precision = st.number_input(
                "Share decimal places",
                min_value=0,
                max_value=8,
                value=6,
                step=1,
                disabled=not allow_fractional_shares,
            )
            apply_tax = st.checkbox("Include a simple gain tax", value=False)

        with st.expander("Optional costs and tax"):
            fee_left, fee_middle, fee_right = st.columns(3)
            with fee_left:
                entry_fee = st.number_input(
                    "Entry fee",
                    min_value=0.0,
                    max_value=MAXIMUM_FEE,
                    value=0.0,
                    step=1.0,
                )
            with fee_middle:
                exit_fee = st.number_input(
                    "Exit fee",
                    min_value=0.0,
                    max_value=MAXIMUM_FEE,
                    value=0.0,
                    step=1.0,
                )
            with fee_right:
                tax_rate = st.number_input(
                    "Tax rate (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=0.0,
                    step=0.5,
                    disabled=not apply_tax,
                )

        submitted = st.form_submit_button(
            "Build research scenarios",
            type="primary",
            use_container_width=True,
        )

    if not submitted:
        return None

    checked = validate_scenario_values(
        investment_amount=investment_amount,
        share_price=share_price,
        currency=currency,
        allow_fractional_shares=allow_fractional_shares,
        share_precision=int(share_precision),
        entry_fee=entry_fee,
        exit_fee=exit_fee,
        tax_rate_percent=tax_rate if apply_tax else None,
    )
    return ScenarioSubmission(
        text=validate_article_text(text),
        ticker=validate_ticker(ticker),
        published_at=build_published_at(
            publication_date,
            publication_time,
            timezone_name,
        ),
        **checked,
    )
