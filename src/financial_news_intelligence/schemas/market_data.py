"""Schemas for verified prices and historical return labels."""

import math
import re
from datetime import date, datetime
from typing_extensions import Self

from pydantic import Field, field_validator, model_validator

from financial_news_intelligence.schemas.common import (
    MovementLabel,
    ProjectSchema,
)
from financial_news_intelligence.schemas.provenance import (
    SourceProvenance,
    VerificationStatus,
)


# ============================================================
# 1. DAILY PRICE BAR
# ============================================================

class PriceBar(ProjectSchema):
    """One validated daily stock-price record."""

    ticker: str
    session_date: date

    open_price: float
    high_price: float
    low_price: float
    close_price: float
    adjusted_close: float | None = None

    volume: int = Field(ge=0)

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, value: str) -> str:
        """Normalize and validate a US ticker symbol."""

        normalized_value = value.strip().upper()

        if not re.fullmatch(
            r"[A-Z][A-Z0-9.\-]{0,14}",
            normalized_value,
        ):
            raise ValueError("Ticker contains unsupported characters.")

        return normalized_value

    @field_validator(
        "open_price",
        "high_price",
        "low_price",
        "close_price",
        "adjusted_close",
    )
    @classmethod
    def validate_price(
        cls,
        value: float | None,
    ) -> float | None:
        """Reject missing, infinite, negative, or zero prices."""

        if value is None:
            return None

        if not math.isfinite(value) or value <= 0:
            raise ValueError("Prices must be finite and greater than zero.")

        return float(value)

    @model_validator(mode="after")
    def validate_ohlc_relationships(self) -> Self:
        """Check that open and close fit inside the daily range."""

        if self.low_price > self.high_price:
            raise ValueError("Low price cannot exceed high price.")

        for price_name, price_value in (
            ("open", self.open_price),
            ("close", self.close_price),
        ):
            if not self.low_price <= price_value <= self.high_price:
                raise ValueError(
                    f"{price_name} price must be between low and high."
                )

        return self


# ============================================================
# 2. PRICE CROSS-CHECK EVIDENCE
# ============================================================

class PriceCrossCheck(ProjectSchema):
    """Evidence that secondary prices matched a primary source."""

    source_id: str = Field(min_length=1)
    source_name: str = Field(min_length=1)
    source_url: str

    verification_status: VerificationStatus
    checked_at: datetime

    matched_sessions: int = Field(ge=1)
    maximum_difference_pct: float = Field(ge=0)
    tolerance_pct: float = Field(gt=0)
    passed: bool

    @field_validator("checked_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        """Cross-check timestamps must identify their timezone."""

        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("checked_at must be timezone-aware.")

        return value

    @model_validator(mode="after")
    def validate_cross_check(self) -> Self:
        """Only successful primary-source checks are accepted."""

        if (
            self.verification_status
            != VerificationStatus.VERIFIED_PRIMARY
        ):
            raise ValueError(
                "Price cross-check source must be verified primary."
            )

        if not self.passed:
            raise ValueError("Price cross-check must pass.")

        if self.maximum_difference_pct > self.tolerance_pct:
            raise ValueError(
                "Maximum price difference exceeds the tolerance."
            )

        return self


# ============================================================
# 3. VERIFIED PRICE HISTORY
# ============================================================

class MarketPriceHistory(ProjectSchema):
    """Validated price history with provenance and cross-check evidence."""

    ticker: str
    start_date: date
    end_date: date

    bars: tuple[PriceBar, ...] = Field(min_length=1)

    provenance: SourceProvenance
    cross_check: PriceCrossCheck

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, value: str) -> str:
        """Normalize the history ticker."""

        normalized_value = value.strip().upper()

        if not re.fullmatch(
            r"[A-Z][A-Z0-9.\-]{0,14}",
            normalized_value,
        ):
            raise ValueError("Ticker contains unsupported characters.")

        return normalized_value

    @model_validator(mode="after")
    def validate_history(self) -> Self:
        """Verify ordering, identity, range, and provenance links."""

        if self.start_date > self.end_date:
            raise ValueError("start_date cannot be after end_date.")

        session_dates = [bar.session_date for bar in self.bars]

        if session_dates != sorted(session_dates):
            raise ValueError("Price bars must be ordered by session date.")

        if len(session_dates) != len(set(session_dates)):
            raise ValueError("Price history contains duplicate sessions.")

        for bar in self.bars:
            if bar.ticker != self.ticker:
                raise ValueError(
                    "Every price bar must use the history ticker."
                )

            if not self.start_date <= bar.session_date <= self.end_date:
                raise ValueError(
                    "Price bar lies outside the requested date range."
                )

        if self.provenance.raw_record_count != len(self.bars):
            raise ValueError(
                "Provenance record count must equal the price-bar count."
            )

        if not self.provenance.cross_checked:
            raise ValueError(
                "Market prices must include a completed cross-check."
            )

        if (
            self.provenance.cross_check_source_url
            != self.cross_check.source_url
        ):
            raise ValueError(
                "Cross-check URL must match the provenance record."
            )

        return self


# ============================================================
# 4. HISTORICAL MOVEMENT LABEL
# ============================================================

class ReturnLabel(ProjectSchema):
    """Actual open-to-close return for an article's target session."""

    ticker: str
    target_session: date

    open_price: float = Field(gt=0)
    close_price: float = Field(gt=0)

    return_pct: float
    direction: MovementLabel
    flat_threshold_pct: float = Field(ge=0)

    calculation_method: str = "open_to_close"

    price_source_id: str = Field(min_length=1)
    price_checksum_sha256: str

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, value: str) -> str:
        """Normalize the label ticker."""

        return value.strip().upper()

    @field_validator("return_pct")
    @classmethod
    def validate_return(cls, value: float) -> float:
        """Return percentages must be finite."""

        if not math.isfinite(value):
            raise ValueError("return_pct must be finite.")

        return float(value)

    @field_validator("price_checksum_sha256")
    @classmethod
    def validate_checksum(cls, value: str) -> str:
        """Preserve the link to the exact price payload."""

        normalized_value = value.lower()

        if not re.fullmatch(r"[0-9a-f]{64}", normalized_value):
            raise ValueError(
                "price_checksum_sha256 must be a SHA-256 value."
            )

        return normalized_value

    @model_validator(mode="after")
    def validate_calculation(self) -> Self:
        """Detect altered returns or inconsistent movement labels."""

        expected_return = (
            (self.close_price - self.open_price)
            / self.open_price
            * 100
        )

        if not math.isclose(
            self.return_pct,
            expected_return,
            abs_tol=0.000001,
        ):
            raise ValueError(
                "return_pct does not match open and close prices."
            )

        if self.return_pct > self.flat_threshold_pct:
            expected_direction = MovementLabel.UP
        elif self.return_pct < -self.flat_threshold_pct:
            expected_direction = MovementLabel.DOWN
        else:
            expected_direction = MovementLabel.FLAT

        if self.direction != expected_direction:
            raise ValueError(
                "Movement direction does not match the return threshold."
            )

        return self
