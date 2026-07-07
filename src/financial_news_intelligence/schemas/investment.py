"""Schemas for evidence-based investment scenario calculations."""

from __future__ import annotations

import math
import re
from datetime import datetime
from enum import Enum

from pydantic import Field, field_validator, model_validator

from financial_news_intelligence.schemas.common import (
    ProjectSchema,
    SentimentLabel,
)


# ============================================================
# 1. CONTROLLED SCENARIO LEVELS
# ============================================================

class ScenarioLevel(str, Enum):
    """Historical return points presented to the user."""

    LOW = "low"
    BASE = "base"
    HIGH = "high"


# ============================================================
# 2. USER INVESTMENT REQUEST
# ============================================================

class InvestmentRequest(ProjectSchema):
    """User-controlled assumptions for a portfolio scenario."""

    investment_amount: float = Field(gt=0)
    share_price: float = Field(gt=0)
    currency: str = "EUR"

    allow_fractional_shares: bool = True
    share_precision: int = Field(default=6, ge=0, le=8)

    entry_fee: float = Field(default=0, ge=0)
    exit_fee: float = Field(default=0, ge=0)
    tax_rate_pct: float | None = Field(default=None, ge=0, le=100)

    @field_validator("currency")
    @classmethod
    def normalize_currency(cls, value: str) -> str:
        """Require a three-letter currency code."""

        normalized_value = value.strip().upper()

        if not re.fullmatch(r"[A-Z]{3}", normalized_value):
            raise ValueError("currency must be a three-letter code.")

        return normalized_value

    @model_validator(mode="after")
    def validate_available_capital(self) -> "InvestmentRequest":
        """The entry fee cannot consume the entire investment amount."""

        if self.entry_fee >= self.investment_amount:
            raise ValueError(
                "entry_fee must be below investment_amount."
            )

        return self


# ============================================================
# 3. ONE SCENARIO OUTCOME
# ============================================================

class InvestmentOutcome(ProjectSchema):
    """Calculated portfolio value for one historical-return scenario."""

    scenario: ScenarioLevel
    historical_return_pct: float

    initial_investment: float = Field(gt=0)
    share_price: float = Field(gt=0)
    shares_purchased: float = Field(ge=0)
    share_cost: float = Field(ge=0)
    cash_balance: float = Field(ge=0)

    entry_fee: float = Field(ge=0)
    exit_fee: float = Field(ge=0)
    estimated_tax: float = Field(ge=0)

    gross_final_value: float = Field(ge=0)
    net_final_value: float = Field(ge=0)
    gain_loss: float
    gain_loss_pct: float

    @field_validator(
        "historical_return_pct",
        "gain_loss",
        "gain_loss_pct",
    )
    @classmethod
    def require_finite_value(cls, value: float) -> float:
        """Scenario calculations must remain finite."""

        if not math.isfinite(value):
            raise ValueError("Scenario values must be finite.")

        return float(value)

    @field_validator("historical_return_pct")
    @classmethod
    def validate_historical_return(cls, value: float) -> float:
        """A stock-value scenario cannot fall below a total loss."""

        if value < -100:
            raise ValueError(
                "historical_return_pct cannot be below -100%."
            )

        return value

    @model_validator(mode="after")
    def validate_outcome_arithmetic(self) -> "InvestmentOutcome":
        """Detect altered costs, final values, or gain/loss figures."""

        expected_available_capital = (
            self.initial_investment - self.entry_fee
        )
        expected_allocated_capital = (
            self.share_cost + self.cash_balance
        )

        if not math.isclose(
            expected_available_capital,
            expected_allocated_capital,
            abs_tol=0.0001,
        ):
            raise ValueError(
                "Share cost and cash balance do not match available capital."
            )

        expected_gross_final_value = (
            self.share_cost
            * (1 + self.historical_return_pct / 100)
            + self.cash_balance
        )

        if not math.isclose(
            self.gross_final_value,
            expected_gross_final_value,
            abs_tol=0.0001,
        ):
            raise ValueError(
                "gross_final_value does not match the return scenario."
            )

        value_after_exit_fee = max(
            self.gross_final_value - self.exit_fee,
            0.0,
        )
        maximum_taxable_gain = max(
            value_after_exit_fee - self.initial_investment,
            0.0,
        )

        if self.estimated_tax > maximum_taxable_gain + 0.0001:
            raise ValueError(
                "estimated_tax exceeds the positive pre-tax gain."
            )

        expected_net_final_value = max(
            value_after_exit_fee - self.estimated_tax,
            0.0,
        )

        if not math.isclose(
            self.net_final_value,
            expected_net_final_value,
            abs_tol=0.0001,
        ):
            raise ValueError(
                "net_final_value does not match fees and estimated tax."
            )

        expected_gain_loss = (
            self.net_final_value - self.initial_investment
        )

        if not math.isclose(
            self.gain_loss,
            expected_gain_loss,
            abs_tol=0.0001,
        ):
            raise ValueError(
                "gain_loss does not match net_final_value."
            )

        expected_gain_loss_pct = (
            expected_gain_loss
            / self.initial_investment
            * 100
        )

        if not math.isclose(
            self.gain_loss_pct,
            expected_gain_loss_pct,
            abs_tol=0.0001,
        ):
            raise ValueError(
                "gain_loss_pct does not match the calculated gain/loss."
            )

        return self


# ============================================================
# 4. COMPLETE SCENARIO RESULT
# ============================================================

class InvestmentScenarioResult(ProjectSchema):
    """Low, base, and high outcomes derived from one evidence cohort."""

    generated_at: datetime
    ticker: str
    sentiment_label: SentimentLabel
    holding_period: str = "next_trading_session_open_to_close"

    request: InvestmentRequest
    historical_sample_size: int = Field(ge=1)
    historical_evidence_checksum_sha256: str

    low: InvestmentOutcome
    base: InvestmentOutcome
    high: InvestmentOutcome

    assumptions: tuple[str, ...] = Field(min_length=1)
    disclaimer: str = Field(min_length=1)

    @field_validator("generated_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        """Scenario generation time must identify its timezone."""

        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("generated_at must be timezone-aware.")

        return value

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, value: str) -> str:
        """Normalize the result ticker."""

        normalized_value = value.strip().upper()

        if not re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,14}", normalized_value):
            raise ValueError("Ticker contains unsupported characters.")

        return normalized_value

    @field_validator("historical_evidence_checksum_sha256")
    @classmethod
    def validate_checksum(cls, value: str) -> str:
        """Preserve the exact historical evidence identity."""

        normalized_value = value.lower()

        if not re.fullmatch(r"[0-9a-f]{64}", normalized_value):
            raise ValueError(
                "historical_evidence_checksum_sha256 must be SHA-256."
            )

        return normalized_value

    @model_validator(mode="after")
    def validate_scenario_order(self) -> "InvestmentScenarioResult":
        """Check scenario labels and historical-return ordering."""

        if self.low.scenario != ScenarioLevel.LOW:
            raise ValueError("low outcome must use the low scenario label.")

        if self.base.scenario != ScenarioLevel.BASE:
            raise ValueError("base outcome must use the base scenario label.")

        if self.high.scenario != ScenarioLevel.HIGH:
            raise ValueError("high outcome must use the high scenario label.")

        if not (
            self.low.historical_return_pct
            <= self.base.historical_return_pct
            <= self.high.historical_return_pct
        ):
            raise ValueError(
                "Scenario returns must be ordered low, base, high."
            )

        return self
