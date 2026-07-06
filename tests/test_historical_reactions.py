"""Tests for the historical news reaction intelligence engine."""

from datetime import datetime, timedelta, timezone

import pytest

from financial_news_intelligence.data.training_dataset import (
    build_news_reaction_record,
)
from financial_news_intelligence.schemas.common import (
    MovementLabel,
    SentimentLabel,
)
from financial_news_intelligence.schemas.market_data import ReturnLabel
from financial_news_intelligence.schemas.provenance import (
    DataPurpose,
    SourceProvenance,
    VerificationStatus,
)
from financial_news_intelligence.services.historical_reactions import (
    InsufficientHistoricalEvidenceError,
    build_historical_reaction_cohort,
    calculate_linear_quantile,
)


# ============================================================
# 1. TEST DATA HELPERS
# ============================================================

def make_record(
    *,
    index: int,
    published_at: datetime,
    return_pct: float,
    ticker: str = "AAPL",
    sentiment_label: SentimentLabel = SentimentLabel.BULLISH,
):
    """Create one verified historical reaction record."""

    open_price = 100.0
    close_price = open_price * (1 + return_pct / 100)

    if return_pct > 0.5:
        movement = MovementLabel.UP
    elif return_pct < -0.5:
        movement = MovementLabel.DOWN
    else:
        movement = MovementLabel.FLAT

    news_provenance = SourceProvenance(
        source_id="sec_edgar",
        source_name="U.S. Securities and Exchange Commission",
        source_url=f"https://www.sec.gov/Archives/example-{index}.htm",
        retrieved_at=published_at + timedelta(hours=1),
        as_of_date=published_at.date(),
        verification_status=VerificationStatus.VERIFIED_PRIMARY,
        allowed_purposes=(DataPurpose.TRAINING,),
        requires_cross_check=False,
        checksum_sha256=(f"{index + 1:064x}"[-64:]),
        raw_record_count=1,
    )

    return_label = ReturnLabel(
        ticker=ticker,
        target_session=published_at.date(),
        open_price=open_price,
        close_price=close_price,
        return_pct=return_pct,
        direction=movement,
        flat_threshold_pct=0.5,
        price_source_id="stooq",
        price_checksum_sha256="a" * 64,
    )

    return build_news_reaction_record(
        article_text=f"Historical article {index} for {ticker}.",
        published_at=published_at,
        company="Example Company",
        ticker=ticker,
        sentiment_label=sentiment_label,
        sentiment_confidence=0.90,
        return_label=return_label,
        news_provenance=news_provenance,
    )


# ============================================================
# 2. QUANTILE FORMULA
# ============================================================

def test_linear_quantile_uses_interpolation() -> None:
    """The median and edge quantiles should be deterministic."""

    values = [-2.0, 0.0, 2.0, 4.0]

    assert calculate_linear_quantile(values, 0.50) == 1.0
    assert calculate_linear_quantile(values, 0.25) == -0.5


# ============================================================
# 3. LEAKAGE-SAFE COHORT
# ============================================================

def test_cohort_excludes_future_and_unrelated_records() -> None:
    """Only earlier records with matching ticker and sentiment may enter."""

    cutoff = datetime(
        2024,
        2,
        1,
        tzinfo=timezone.utc,
    )

    eligible_records = [
        make_record(
            index=index,
            published_at=cutoff - timedelta(days=10 - index),
            return_pct=float(index - 2),
        )
        for index in range(5)
    ]

    future_record = make_record(
        index=20,
        published_at=cutoff + timedelta(days=1),
        return_pct=10.0,
    )
    other_ticker = make_record(
        index=21,
        published_at=cutoff - timedelta(days=1),
        return_pct=8.0,
        ticker="MSFT",
    )
    bearish_record = make_record(
        index=22,
        published_at=cutoff - timedelta(days=1),
        return_pct=-8.0,
        sentiment_label=SentimentLabel.BEARISH,
    )

    candidates = [
        *eligible_records,
        future_record,
        other_ticker,
        bearish_record,
    ]

    scores = {
        record.article_id: 0.90
        for record in candidates
    }

    cohort = build_historical_reaction_cohort(
        query_article_id="f" * 64,
        ticker="AAPL",
        sentiment_label=SentimentLabel.BULLISH,
        cutoff_at=cutoff,
        candidate_records=candidates,
        similarity_scores=scores,
        minimum_sample_size=5,
    )

    assert cohort.sample_size == 5
    assert future_record.article_id not in cohort.matched_article_ids
    assert other_ticker.article_id not in cohort.matched_article_ids
    assert bearish_record.article_id not in cohort.matched_article_ids
    assert cohort.latest_evidence_published_at < cutoff


# ============================================================
# 4. HISTORICAL SCENARIO STATISTICS
# ============================================================

def test_cohort_calculates_low_median_and_high_returns() -> None:
    """Selected historical returns should form ordered scenarios."""

    cutoff = datetime(
        2024,
        2,
        1,
        tzinfo=timezone.utc,
    )
    returns = [-2.0, -1.0, 0.0, 1.0, 2.0]

    records = [
        make_record(
            index=index,
            published_at=cutoff - timedelta(days=20 - index),
            return_pct=return_pct,
        )
        for index, return_pct in enumerate(returns)
    ]

    cohort = build_historical_reaction_cohort(
        query_article_id="f" * 64,
        ticker="AAPL",
        sentiment_label=SentimentLabel.BULLISH,
        cutoff_at=cutoff,
        candidate_records=records,
        similarity_scores={
            record.article_id: 0.80 + index * 0.01
            for index, record in enumerate(records)
        },
        minimum_sample_size=5,
        lower_quantile=0.10,
        upper_quantile=0.90,
    )

    assert cohort.low_return_pct == -1.6
    assert cohort.median_return_pct == 0.0
    assert cohort.high_return_pct == 1.6
    assert len(cohort.evidence_checksum_sha256) == 64


# ============================================================
# 5. MINIMUM-SAMPLE PROTECTION
# ============================================================

def test_small_historical_cohort_is_rejected() -> None:
    """A tiny cohort must not produce confident-looking scenarios."""

    cutoff = datetime(
        2024,
        2,
        1,
        tzinfo=timezone.utc,
    )
    records = [
        make_record(
            index=index,
            published_at=cutoff - timedelta(days=5 - index),
            return_pct=float(index),
        )
        for index in range(2)
    ]

    with pytest.raises(
        InsufficientHistoricalEvidenceError,
        match="at least 5",
    ):
        build_historical_reaction_cohort(
            query_article_id="f" * 64,
            ticker="AAPL",
            sentiment_label=SentimentLabel.BULLISH,
            cutoff_at=cutoff,
            candidate_records=records,
            similarity_scores={
                record.article_id: 0.90
                for record in records
            },
            minimum_sample_size=5,
        )


# ============================================================
# 6. SIMILARITY THRESHOLD
# ============================================================

def test_low_similarity_records_are_excluded() -> None:
    """Evidence below the configured similarity floor should be ignored."""

    cutoff = datetime(
        2024,
        2,
        1,
        tzinfo=timezone.utc,
    )
    records = [
        make_record(
            index=index,
            published_at=cutoff - timedelta(days=10 - index),
            return_pct=float(index),
        )
        for index in range(5)
    ]

    scores = {
        record.article_id: 0.90
        for record in records
    }
    scores[records[0].article_id] = 0.20

    with pytest.raises(InsufficientHistoricalEvidenceError):
        build_historical_reaction_cohort(
            query_article_id="f" * 64,
            ticker="AAPL",
            sentiment_label=SentimentLabel.BULLISH,
            cutoff_at=cutoff,
            candidate_records=records,
            similarity_scores=scores,
            minimum_similarity=0.70,
            minimum_sample_size=5,
        )


# ============================================================
# 7. DETERMINISTIC EVIDENCE CHECKSUM
# ============================================================

def test_candidate_order_does_not_change_evidence_checksum() -> None:
    """Equivalent evidence should produce the same checksum."""

    cutoff = datetime(
        2024,
        2,
        1,
        tzinfo=timezone.utc,
    )
    records = [
        make_record(
            index=index,
            published_at=cutoff - timedelta(days=10 - index),
            return_pct=float(index - 2),
        )
        for index in range(5)
    ]
    scores = {
        record.article_id: 0.80 + index * 0.01
        for index, record in enumerate(records)
    }

    first = build_historical_reaction_cohort(
        query_article_id="f" * 64,
        ticker="AAPL",
        sentiment_label=SentimentLabel.BULLISH,
        cutoff_at=cutoff,
        candidate_records=records,
        similarity_scores=scores,
        minimum_sample_size=5,
    )
    second = build_historical_reaction_cohort(
        query_article_id="f" * 64,
        ticker="AAPL",
        sentiment_label=SentimentLabel.BULLISH,
        cutoff_at=cutoff,
        candidate_records=list(reversed(records)),
        similarity_scores=scores,
        minimum_sample_size=5,
    )

    assert first.evidence_checksum_sha256 == (
        second.evidence_checksum_sha256
    )
