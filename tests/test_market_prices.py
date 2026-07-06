"""Tests for verified historical market-price handling."""

from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from financial_news_intelligence.data.market_prices import (
    MarketPriceError,
    PriceCrossCheckError,
    build_stooq_download_url,
    fetch_verified_price_history,
    normalize_us_ticker,
    parse_stooq_csv,
    verify_price_cross_check,
)
from financial_news_intelligence.schemas.provenance import (
    VerificationStatus,
)


VALID_PRICE_CSV = """Date,Open,High,Low,Close,Volume
2024-01-02,100.00,103.00,99.00,102.00,1000000
2024-01-03,102.00,104.00,101.00,103.00,1200000
"""


class FakeResponse:
    """Small requests-compatible response used without internet access."""

    def __init__(self, text: str) -> None:
        self.text = text

    def raise_for_status(self) -> None:
        """The fake response represents HTTP success."""


def fake_get(
    url: str,
    **kwargs: object,
) -> FakeResponse:
    """Return deterministic CSV instead of making a live request."""

    assert "aapl.us" in url
    assert kwargs["timeout"] == 20

    return FakeResponse(VALID_PRICE_CSV)


# ============================================================
# 1. TICKER AND URL
# ============================================================

def test_ticker_and_download_url_are_normalized() -> None:
    """Ticker casing and Stooq request dates should be deterministic."""

    assert normalize_us_ticker(" aapl ") == "AAPL"

    url = build_stooq_download_url(
        "AAPL",
        date(2024, 1, 2),
        date(2024, 1, 3),
    )

    assert "s=aapl.us" in url
    assert "d1=20240102" in url
    assert "d2=20240103" in url
    assert "i=d" in url


# ============================================================
# 2. VALID CSV
# ============================================================

def test_valid_csv_creates_ordered_price_bars() -> None:
    """Provider rows should become validated chronological records."""

    price_bars = parse_stooq_csv(
        VALID_PRICE_CSV,
        "AAPL",
    )

    assert len(price_bars) == 2
    assert price_bars[0].session_date == date(2024, 1, 2)
    assert price_bars[0].open_price == 100.0
    assert price_bars[1].close_price == 103.0


# ============================================================
# 3. MISSING COLUMNS
# ============================================================

def test_missing_csv_columns_are_rejected() -> None:
    """Incomplete provider responses cannot enter the pipeline."""

    invalid_csv = """Date,Close
2024-01-02,102.00
"""

    with pytest.raises(
        MarketPriceError,
        match="missing columns",
    ):
        parse_stooq_csv(invalid_csv, "AAPL")


# ============================================================
# 4. IMPOSSIBLE OHLC VALUES
# ============================================================

def test_impossible_price_range_is_rejected() -> None:
    """A close outside the daily high-low range is invalid."""

    invalid_csv = """Date,Open,High,Low,Close,Volume
2024-01-02,100.00,101.00,99.00,105.00,1000
"""

    with pytest.raises(
        MarketPriceError,
        match="between low and high",
    ):
        parse_stooq_csv(invalid_csv, "AAPL")


# ============================================================
# 5. SUCCESSFUL PRIMARY CROSS-CHECK
# ============================================================

def test_primary_price_cross_check_passes() -> None:
    """Matching Nasdaq sample values should verify the provider data."""

    price_bars = parse_stooq_csv(
        VALID_PRICE_CSV,
        "AAPL",
    )

    evidence = verify_price_cross_check(
        price_bars=price_bars,
        reference_closes={
            date(2024, 1, 2): 102.01,
            date(2024, 1, 3): 103.00,
        },
        cross_check_source_url=(
            "https://www.nasdaq.com/"
            "market-activity/stocks/aapl/historical"
        ),
        checked_at=datetime(
            2024,
            1,
            4,
            12,
            0,
            tzinfo=timezone.utc,
        ),
        tolerance_pct=0.10,
    )

    assert evidence.passed is True
    assert evidence.matched_sessions == 2
    assert (
        evidence.verification_status
        == VerificationStatus.VERIFIED_PRIMARY
    )


# ============================================================
# 6. PRICE MISMATCH
# ============================================================

def test_large_cross_check_difference_is_rejected() -> None:
    """Materially different reference prices must block the dataset."""

    price_bars = parse_stooq_csv(
        VALID_PRICE_CSV,
        "AAPL",
    )

    with pytest.raises(
        PriceCrossCheckError,
        match="cross-check failed",
    ):
        verify_price_cross_check(
            price_bars=price_bars,
            reference_closes={
                date(2024, 1, 2): 90.00,
            },
            cross_check_source_url=(
                "https://www.nasdaq.com/"
                "market-activity/stocks/aapl/historical"
            ),
            checked_at=datetime(
                2024,
                1,
                4,
                tzinfo=timezone.utc,
            ),
            tolerance_pct=0.10,
        )


# ============================================================
# 7. SECONDARY CROSS-CHECK SOURCE
# ============================================================

def test_cross_check_source_must_be_primary() -> None:
    """The secondary provider cannot verify its own values."""

    price_bars = parse_stooq_csv(
        VALID_PRICE_CSV,
        "AAPL",
    )

    with pytest.raises(
        PriceCrossCheckError,
        match="verified primary",
    ):
        verify_price_cross_check(
            price_bars=price_bars,
            reference_closes={
                date(2024, 1, 2): 102.00,
            },
            cross_check_source_url=(
                "https://stooq.com/q/d/?s=aapl.us"
            ),
            checked_at=datetime(
                2024,
                1,
                4,
                tzinfo=timezone.utc,
            ),
        )


# ============================================================
# 8. COMPLETE FETCH, PROVENANCE, AND CACHE
# ============================================================

def test_verified_fetch_creates_provenance_and_cache(
    tmp_path: Path,
) -> None:
    """A fully checked response should be accepted and cached."""

    history = fetch_verified_price_history(
        ticker="AAPL",
        start_date=date(2024, 1, 2),
        end_date=date(2024, 1, 3),
        reference_closes={
            date(2024, 1, 2): 102.00,
            date(2024, 1, 3): 103.00,
        },
        cross_check_source_url=(
            "https://www.nasdaq.com/"
            "market-activity/stocks/aapl/historical"
        ),
        retrieved_at=datetime(
            2024,
            1,
            4,
            12,
            0,
            tzinfo=timezone.utc,
        ),
        checked_at=datetime(
            2024,
            1,
            4,
            12,
            5,
            tzinfo=timezone.utc,
        ),
        tolerance_pct=0.10,
        cache_dir=tmp_path,
        request_get=fake_get,
    )

    assert history.ticker == "AAPL"
    assert len(history.bars) == 2
    assert history.provenance.source_id == "stooq"
    assert history.provenance.cross_checked is True
    assert history.provenance.raw_record_count == 2
    assert len(history.provenance.checksum_sha256) == 64

    raw_cache_files = list(tmp_path.glob("*.csv"))
    metadata_cache_files = list(tmp_path.glob("*.json"))

    assert len(raw_cache_files) == 1
    assert len(metadata_cache_files) == 1
    assert raw_cache_files[0].read_text(
        encoding="utf-8"
    ) == VALID_PRICE_CSV
