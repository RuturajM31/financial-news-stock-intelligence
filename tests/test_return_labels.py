"""Tests for historical open-to-close movement labels."""

from datetime import date, datetime, timezone

import pytest
from pydantic import ValidationError

from financial_news_intelligence.data.return_labels import (
    calculate_open_to_close_return,
    classify_return_direction,
    create_return_label,
)
from financial_news_intelligence.schemas.common import MovementLabel
from financial_news_intelligence.schemas.market_data import (
    MarketPriceHistory,
    PriceBar,
    PriceCrossCheck,
    ReturnLabel,
)
from financial_news_intelligence.schemas.provenance import (
    DataPurpose,
    SourceProvenance,
    VerificationStatus,
)


@pytest.fixture
def verified_price_history() -> MarketPriceHistory:
    """Create deterministic verified prices for label tests."""

    provenance = SourceProvenance(
        source_id="stooq",
        source_name="Stooq Market Data",
        source_url=(
            "https://stooq.com/q/d/l/"
            "?s=aapl.us&d1=20240102&d2=20240104&i=d"
        ),
        retrieved_at=datetime(
            2024,
            1,
            5,
            12,
            0,
            tzinfo=timezone.utc,
        ),
        as_of_date=date(2024, 1, 4),
        verification_status=(
            VerificationStatus.VERIFIED_SECONDARY
        ),
        allowed_purposes=(
            DataPurpose.TRAINING,
            DataPurpose.INVESTMENT_SCENARIOS,
            DataPurpose.REPORTING,
        ),
        requires_cross_check=True,
        checksum_sha256="a" * 64,
        raw_record_count=3,
        cross_checked=True,
        cross_check_source_url=(
            "https://www.nasdaq.com/"
            "market-activity/stocks/aapl/historical"
        ),
    )

    cross_check = PriceCrossCheck(
        source_id="nasdaq",
        source_name="Nasdaq",
        source_url=(
            "https://www.nasdaq.com/"
            "market-activity/stocks/aapl/historical"
        ),
        verification_status=(
            VerificationStatus.VERIFIED_PRIMARY
        ),
        checked_at=datetime(
            2024,
            1,
            5,
            12,
            5,
            tzinfo=timezone.utc,
        ),
        matched_sessions=3,
        maximum_difference_pct=0.0,
        tolerance_pct=0.10,
        passed=True,
    )

    return MarketPriceHistory(
        ticker="AAPL",
        start_date=date(2024, 1, 2),
        end_date=date(2024, 1, 4),
        bars=(
            PriceBar(
                ticker="AAPL",
                session_date=date(2024, 1, 2),
                open_price=100.0,
                high_price=103.0,
                low_price=99.0,
                close_price=102.0,
                volume=1000,
            ),
            PriceBar(
                ticker="AAPL",
                session_date=date(2024, 1, 3),
                open_price=100.0,
                high_price=101.0,
                low_price=98.0,
                close_price=99.0,
                volume=1200,
            ),
            PriceBar(
                ticker="AAPL",
                session_date=date(2024, 1, 4),
                open_price=100.0,
                high_price=101.0,
                low_price=99.0,
                close_price=100.4,
                volume=900,
            ),
        ),
        provenance=provenance,
        cross_check=cross_check,
    )


# ============================================================
# 1. RETURN FORMULA
# ============================================================

def test_open_to_close_return_formula() -> None:
    """A move from 100 to 102 should equal positive two percent."""

    result = calculate_open_to_close_return(
        100.0,
        102.0,
    )

    assert result == 2.0


# ============================================================
# 2. MOVEMENT CLASSIFICATION
# ============================================================

@pytest.mark.parametrize(
    ("return_pct", "expected_direction"),
    [
        (0.60, MovementLabel.UP),
        (-0.60, MovementLabel.DOWN),
        (0.50, MovementLabel.FLAT),
    ],
)
def test_return_direction_classification(
    return_pct: float,
    expected_direction: MovementLabel,
) -> None:
    """The configured threshold should define Up, Flat, and Down."""

    result = classify_return_direction(
        return_pct,
        flat_threshold_pct=0.50,
    )

    assert result == expected_direction


# ============================================================
# 3. VERIFIED TARGET LABEL
# ============================================================

def test_target_session_creates_traceable_label(
    verified_price_history: MarketPriceHistory,
) -> None:
    """The selected session should produce a sourced movement label."""

    label = create_return_label(
        price_history=verified_price_history,
        target_session=date(2024, 1, 2),
        flat_threshold_pct=0.50,
    )

    assert label.return_pct == 2.0
    assert label.direction == MovementLabel.UP
    assert label.price_source_id == "stooq"
    assert label.price_checksum_sha256 == "a" * 64


# ============================================================
# 4. MISSING TARGET SESSION
# ============================================================

def test_missing_target_session_is_rejected(
    verified_price_history: MarketPriceHistory,
) -> None:
    """A label cannot be fabricated when its trading day is absent."""

    with pytest.raises(
        LookupError,
        match="absent from price history",
    ):
        create_return_label(
            price_history=verified_price_history,
            target_session=date(2024, 1, 5),
            flat_threshold_pct=0.50,
        )


# ============================================================
# 5. INVALID THRESHOLD
# ============================================================

def test_negative_flat_threshold_is_rejected() -> None:
    """Movement thresholds cannot be negative."""

    with pytest.raises(
        ValueError,
        match="non-negative",
    ):
        classify_return_direction(
            1.0,
            flat_threshold_pct=-0.50,
        )


# ============================================================
# 6. ALTERED RETURN DETECTION
# ============================================================

def test_tampered_return_label_is_rejected() -> None:
    """The schema should detect a return that disagrees with prices."""

    with pytest.raises(
        ValidationError,
        match="does not match",
    ):
        ReturnLabel(
            ticker="AAPL",
            target_session=date(2024, 1, 2),
            open_price=100.0,
            close_price=102.0,
            return_pct=99.0,
            direction=MovementLabel.UP,
            flat_threshold_pct=0.50,
            price_source_id="stooq",
            price_checksum_sha256="a" * 64,
        )
