"""Tests for the verified news-to-market reaction dataset."""

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from financial_news_intelligence.data.training_dataset import (
    assign_chronological_splits,
    build_news_reaction_record,
    create_article_id,
    export_training_dataset,
    normalize_article_text,
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
from financial_news_intelligence.schemas.training_data import DatasetSplit


# ============================================================
# 1. TEST DATA HELPERS
# ============================================================

def make_news_provenance(
    published_at: datetime,
) -> SourceProvenance:
    """Create one verified primary news-source record."""

    return SourceProvenance(
        source_id="sec_edgar",
        source_name="U.S. Securities and Exchange Commission",
        source_url="https://www.sec.gov/Archives/example-filing.htm",
        retrieved_at=published_at + timedelta(hours=1),
        as_of_date=published_at.date(),
        verification_status=VerificationStatus.VERIFIED_PRIMARY,
        allowed_purposes=(DataPurpose.TRAINING,),
        requires_cross_check=False,
        checksum_sha256="b" * 64,
        raw_record_count=1,
    )


def make_record(index: int):
    """Prepare one deterministic verified reaction record."""

    published_at = datetime(
        2024,
        1,
        2 + index,
        13,
        0,
        tzinfo=timezone.utc,
    )

    return_label = ReturnLabel(
        ticker="AAPL",
        target_session=published_at.date(),
        open_price=100.0,
        close_price=101.0,
        return_pct=1.0,
        direction=MovementLabel.UP,
        flat_threshold_pct=0.5,
        price_source_id="stooq",
        price_checksum_sha256="a" * 64,
    )

    return build_news_reaction_record(
        article_text=f"Apple filing number {index} reported stronger revenue.",
        published_at=published_at,
        company="Apple Inc.",
        ticker="AAPL",
        sentiment_label=SentimentLabel.BULLISH,
        sentiment_confidence=0.90,
        return_label=return_label,
        news_provenance=make_news_provenance(published_at),
    )


# ============================================================
# 2. NORMALIZATION AND DETERMINISTIC ID
# ============================================================

def test_article_normalization_and_id_are_deterministic() -> None:
    """Equivalent whitespace should produce the same article identity."""

    published_at = datetime(
        2024,
        1,
        2,
        13,
        0,
        tzinfo=timezone.utc,
    )

    first_text = "Apple   reported\nstrong revenue."
    second_text = "Apple reported strong revenue."

    first_id = create_article_id(
        article_text=first_text,
        published_at=published_at,
        ticker="AAPL",
        source_url="https://www.sec.gov/example",
    )
    second_id = create_article_id(
        article_text=second_text,
        published_at=published_at,
        ticker="AAPL",
        source_url="https://www.sec.gov/example",
    )

    assert normalize_article_text(first_text) == second_text
    assert first_id == second_id
    assert len(first_id) == 64


# ============================================================
# 3. VERIFIED ARTICLE-REACTION JOIN
# ============================================================

def test_verified_article_and_return_create_one_record() -> None:
    """The builder should preserve article and market evidence together."""

    record = make_record(0)

    assert record.ticker == "AAPL"
    assert record.movement_label == MovementLabel.UP
    assert record.return_pct == 1.0
    assert record.news_provenance.source_id == "sec_edgar"
    assert record.price_checksum_sha256 == "a" * 64


# ============================================================
# 4. CHRONOLOGICAL SPLITS
# ============================================================

def test_splits_are_chronological_and_non_overlapping() -> None:
    """Older articles must remain before validation and test articles."""

    records = [make_record(index) for index in reversed(range(10))]

    splits = assign_chronological_splits(records)

    assert len(splits[DatasetSplit.TRAIN]) == 7
    assert len(splits[DatasetSplit.VALIDATION]) == 1
    assert len(splits[DatasetSplit.TEST]) == 2

    train_ids = {
        record.article_id
        for record in splits[DatasetSplit.TRAIN]
    }
    validation_ids = {
        record.article_id
        for record in splits[DatasetSplit.VALIDATION]
    }
    test_ids = {
        record.article_id
        for record in splits[DatasetSplit.TEST]
    }

    assert train_ids.isdisjoint(validation_ids)
    assert train_ids.isdisjoint(test_ids)
    assert validation_ids.isdisjoint(test_ids)

    assert (
        splits[DatasetSplit.TRAIN][-1].published_at
        < splits[DatasetSplit.VALIDATION][0].published_at
        < splits[DatasetSplit.TEST][0].published_at
    )


# ============================================================
# 5. DUPLICATE PROTECTION
# ============================================================

def test_duplicate_article_ids_are_rejected() -> None:
    """The same article cannot appear twice across model splits."""

    record = make_record(0)

    with pytest.raises(ValueError, match="Duplicate article IDs"):
        assign_chronological_splits([record, record, make_record(1)])


# ============================================================
# 6. JSONL EXPORT AND MANIFEST
# ============================================================

def test_export_writes_all_splits_and_checksums(
    tmp_path: Path,
) -> None:
    """Exported files should match the manifest and retain split labels."""

    records = [make_record(index) for index in range(10)]

    manifest = export_training_dataset(
        records,
        output_dir=tmp_path,
        dataset_name="verified_reactions",
        created_at=datetime(
            2024,
            2,
            1,
            tzinfo=timezone.utc,
        ),
    )

    assert manifest.total_records == 10
    assert sum(manifest.split_counts.values()) == 10
    assert len(manifest.dataset_checksum_sha256) == 64

    for split in DatasetSplit:
        split_path = tmp_path / manifest.split_files[split.value]

        assert split_path.exists()
        assert split_path.read_text(encoding="utf-8").count("\n") == (
            manifest.split_counts[split.value]
        )

    assert (tmp_path / "verified_reactions_manifest.json").exists()


# ============================================================
# 7. TICKER JOIN PROTECTION
# ============================================================

def test_article_and_return_tickers_must_match() -> None:
    """A market label from another company must not be joined silently."""

    published_at = datetime(
        2024,
        1,
        2,
        13,
        0,
        tzinfo=timezone.utc,
    )

    return_label = ReturnLabel(
        ticker="MSFT",
        target_session=date(2024, 1, 2),
        open_price=100.0,
        close_price=101.0,
        return_pct=1.0,
        direction=MovementLabel.UP,
        flat_threshold_pct=0.5,
        price_source_id="stooq",
        price_checksum_sha256="a" * 64,
    )

    with pytest.raises(ValueError, match="must match"):
        build_news_reaction_record(
            article_text="Apple reported stronger revenue.",
            published_at=published_at,
            company="Apple Inc.",
            ticker="AAPL",
            sentiment_label=SentimentLabel.BULLISH,
            sentiment_confidence=0.90,
            return_label=return_label,
            news_provenance=make_news_provenance(published_at),
        )
