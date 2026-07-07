"""Tests for verified sources, checksums, and provenance gates."""

from datetime import date, datetime, timezone

import pytest
from pydantic import ValidationError

from financial_news_intelligence.data.provenance import (
    SourceUsageError,
    UnverifiedSourceError,
    assert_usage_allowed,
    assess_source_url,
    build_provenance,
    compute_sha256,
    load_verified_source_registry,
)
from financial_news_intelligence.schemas.provenance import (
    DataPurpose,
    SourceProvenance,
    VerificationStatus,
)


# ============================================================
# 1. REGISTRY LOADING
# ============================================================

def test_verified_source_registry_loads() -> None:
    """The permanent registry should contain core authority sources."""

    registry = load_verified_source_registry()

    assert "sec_edgar" in registry
    assert "nyse" in registry
    assert "nasdaq" in registry
    assert "stooq" in registry


# ============================================================
# 2. PRIMARY SOURCE RECOGNITION
# ============================================================

def test_sec_url_is_verified_primary() -> None:
    """SEC subdomains should be recognized as primary sources."""

    assessment = assess_source_url(
        "https://data.sec.gov/submissions/CIK0000320193.json"
    )

    assert assessment.source_id == "sec_edgar"
    assert (
        assessment.verification_status
        == VerificationStatus.VERIFIED_PRIMARY
    )
    assert DataPurpose.TRAINING in assessment.allowed_purposes


# ============================================================
# 3. PRIMARY DATA USAGE
# ============================================================

def test_primary_source_is_allowed_for_training() -> None:
    """Verified primary data may enter protected pipelines."""

    provenance = build_provenance(
        source_url=(
            "https://www.sec.gov/files/company_tickers.json"
        ),
        raw_payload={"ticker": "AAPL"},
        retrieved_at=datetime(
            2024,
            1,
            20,
            12,
            0,
            tzinfo=timezone.utc,
        ),
        as_of_date=date(2024, 1, 19),
        raw_record_count=1,
    )

    assert_usage_allowed(
        provenance,
        DataPurpose.TRAINING,
    )

    assert_usage_allowed(
        provenance,
        DataPurpose.INVESTMENT_SCENARIOS,
    )


# ============================================================
# 4. SECONDARY SOURCE CROSS-CHECK
# ============================================================

def test_secondary_source_requires_cross_check() -> None:
    """Secondary price data stays blocked until independently checked."""

    unconfirmed_provenance = build_provenance(
        source_url="https://stooq.com/q/d/l/?s=aapl.us",
        raw_payload="Date,Close\n2024-01-19,191.56",
        retrieved_at=datetime(
            2024,
            1,
            20,
            12,
            0,
            tzinfo=timezone.utc,
        ),
        as_of_date=date(2024, 1, 19),
        raw_record_count=1,
    )

    with pytest.raises(
        SourceUsageError,
        match="requires a documented cross-check",
    ):
        assert_usage_allowed(
            unconfirmed_provenance,
            DataPurpose.INVESTMENT_SCENARIOS,
        )

    confirmed_provenance = build_provenance(
        source_url="https://stooq.com/q/d/l/?s=aapl.us",
        raw_payload="Date,Close\n2024-01-19,191.56",
        retrieved_at=datetime(
            2024,
            1,
            20,
            12,
            0,
            tzinfo=timezone.utc,
        ),
        as_of_date=date(2024, 1, 19),
        raw_record_count=1,
        cross_checked=True,
        cross_check_source_url=(
            "https://www.nasdaq.com/market-activity/stocks/aapl"
        ),
    )

    assert_usage_allowed(
        confirmed_provenance,
        DataPurpose.INVESTMENT_SCENARIOS,
    )


# ============================================================
# 5. UNKNOWN SOURCE REJECTION
# ============================================================

def test_unknown_source_is_rejected() -> None:
    """Unregistered domains must not enter the project pipeline."""

    with pytest.raises(
        UnverifiedSourceError,
        match="not approved",
    ):
        assess_source_url(
            "https://unknown-example.invalid/prices.csv"
        )


# ============================================================
# 6. DETERMINISTIC CHECKSUM
# ============================================================

def test_checksum_is_reproducible() -> None:
    """Dictionary key order must not change the checksum."""

    first_payload = {
        "ticker": "AAPL",
        "close": 191.56,
    }

    second_payload = {
        "close": 191.56,
        "ticker": "AAPL",
    }

    assert compute_sha256(first_payload) == compute_sha256(
        second_payload
    )


# ============================================================
# 7. REQUIRED METADATA
# ============================================================

def test_missing_source_name_is_rejected() -> None:
    """A provenance record must identify its source clearly."""

    with pytest.raises(ValidationError):
        SourceProvenance(
            source_id="sec_edgar",
            source_name="",
            source_url="https://www.sec.gov/example.json",
            retrieved_at=datetime(
                2024,
                1,
                20,
                tzinfo=timezone.utc,
            ),
            as_of_date=date(2024, 1, 19),
            verification_status=(
                VerificationStatus.VERIFIED_PRIMARY
            ),
            allowed_purposes=(DataPurpose.REPORTING,),
            checksum_sha256="a" * 64,
            raw_record_count=1,
        )


# ============================================================
# 8. TEMPORAL INTEGRITY
# ============================================================

def test_future_as_of_date_is_rejected() -> None:
    """Data cannot claim an as-of date after it was retrieved."""

    with pytest.raises(
        ValidationError,
        match="as_of_date cannot be later",
    ):
        build_provenance(
            source_url=(
                "https://www.sec.gov/files/company_tickers.json"
            ),
            raw_payload={"ticker": "AAPL"},
            retrieved_at=datetime(
                2024,
                1,
                20,
                tzinfo=timezone.utc,
            ),
            as_of_date=date(2024, 1, 21),
            raw_record_count=1,
        )
