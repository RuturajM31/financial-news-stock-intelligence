"""Build reproducible news-to-market reaction training datasets."""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from financial_news_intelligence.data.provenance import assert_usage_allowed
from financial_news_intelligence.schemas.common import SentimentLabel
from financial_news_intelligence.schemas.market_data import ReturnLabel
from financial_news_intelligence.schemas.provenance import (
    DataPurpose,
    SourceProvenance,
)
from financial_news_intelligence.schemas.training_data import (
    DatasetManifest,
    DatasetSplit,
    NewsReactionRecord,
)


# ============================================================
# 1. ARTICLE NORMALIZATION AND IDENTITY
# ============================================================

def normalize_article_text(article_text: str) -> str:
    """
    Normalize whitespace without changing the article's words.

    Input:  Extracted article text.
    Output: Stable single-space text used for IDs and model records.
    """

    normalized_text = re.sub(r"\s+", " ", article_text).strip()

    if not normalized_text:
        raise ValueError("article_text cannot be empty.")

    return normalized_text


def create_article_id(
    *,
    article_text: str,
    published_at: datetime,
    ticker: str,
    source_url: str,
) -> str:
    """Create a deterministic SHA-256 article identifier."""

    if published_at.tzinfo is None or published_at.utcoffset() is None:
        raise ValueError("published_at must be timezone-aware.")

    canonical_payload = {
        "article_text": normalize_article_text(article_text),
        "published_at_utc": published_at.astimezone(timezone.utc).isoformat(),
        "source_url": source_url.strip(),
        "ticker": ticker.strip().upper(),
    }

    encoded_payload = json.dumps(
        canonical_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")

    return hashlib.sha256(encoded_payload).hexdigest()


# ============================================================
# 2. VERIFIED REACTION RECORD
# ============================================================

def build_news_reaction_record(
    *,
    article_text: str,
    published_at: datetime,
    company: str,
    ticker: str,
    sentiment_label: SentimentLabel,
    sentiment_confidence: float,
    return_label: ReturnLabel,
    news_provenance: SourceProvenance,
) -> NewsReactionRecord:
    """
    Join one verified article with one verified market-reaction label.

    Grain: one article, one company ticker, one target trading session.
    Downstream use: chronological model datasets and historical cohorts.
    """

    # Unverified news is blocked before it reaches model training.
    assert_usage_allowed(
        news_provenance,
        DataPurpose.TRAINING,
    )

    normalized_ticker = ticker.strip().upper()

    if normalized_ticker != return_label.ticker:
        raise ValueError(
            "Article ticker must match the verified return-label ticker."
        )

    normalized_text = normalize_article_text(article_text)

    article_id = create_article_id(
        article_text=normalized_text,
        published_at=published_at,
        ticker=normalized_ticker,
        source_url=news_provenance.source_url,
    )

    return NewsReactionRecord(
        article_id=article_id,
        article_text=normalized_text,
        published_at=published_at,
        company=company,
        ticker=normalized_ticker,
        sentiment_label=sentiment_label,
        sentiment_confidence=sentiment_confidence,
        target_session=return_label.target_session,
        open_price=return_label.open_price,
        close_price=return_label.close_price,
        return_pct=return_label.return_pct,
        movement_label=return_label.direction,
        flat_threshold_pct=return_label.flat_threshold_pct,
        news_provenance=news_provenance,
        price_source_id=return_label.price_source_id,
        price_checksum_sha256=(
            return_label.price_checksum_sha256
        ),
    )


# ============================================================
# 3. LEAKAGE-SAFE CHRONOLOGICAL SPLITS
# ============================================================

def assign_chronological_splits(
    records: Sequence[NewsReactionRecord],
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> dict[DatasetSplit, tuple[NewsReactionRecord, ...]]:
    """
    Split records by publication time instead of random shuffling.

    Earlier articles enter training, later articles enter validation,
    and the newest articles remain untouched for final testing.
    """

    if len(records) < 3:
        raise ValueError(
            "At least three records are required for three dataset splits."
        )

    if (
        not math.isfinite(train_fraction)
        or not math.isfinite(validation_fraction)
        or train_fraction <= 0
        or validation_fraction <= 0
        or train_fraction + validation_fraction >= 1
    ):
        raise ValueError(
            "Split fractions must be positive and sum to less than one."
        )

    article_ids = [record.article_id for record in records]

    if len(article_ids) != len(set(article_ids)):
        raise ValueError("Duplicate article IDs are not allowed.")

    ordered_records = sorted(
        records,
        key=lambda record: (
            record.published_at.astimezone(timezone.utc),
            record.article_id,
        ),
    )

    total_records = len(ordered_records)
    train_count = max(1, int(total_records * train_fraction))
    validation_count = max(
        1,
        int(total_records * validation_fraction),
    )

    # Preserve at least one untouched test record for small datasets.
    while train_count + validation_count >= total_records:
        if train_count > validation_count and train_count > 1:
            train_count -= 1
        elif validation_count > 1:
            validation_count -= 1
        else:
            raise ValueError(
                "Dataset is too small for the requested split fractions."
            )

    train_end = train_count
    validation_end = train_count + validation_count

    raw_splits = {
        DatasetSplit.TRAIN: ordered_records[:train_end],
        DatasetSplit.VALIDATION: ordered_records[
            train_end:validation_end
        ],
        DatasetSplit.TEST: ordered_records[validation_end:],
    }

    # Store the assigned split inside each exported record.
    return {
        split: tuple(
            record.model_copy(
                update={"dataset_split": split}
            )
            for record in split_records
        )
        for split, split_records in raw_splits.items()
    }


# ============================================================
# 4. JSONL EXPORT AND CHECKSUM MANIFEST
# ============================================================

def _sha256_bytes(payload: bytes) -> str:
    """Calculate a SHA-256 checksum for exported bytes."""

    return hashlib.sha256(payload).hexdigest()


def _serialize_jsonl(
    records: Sequence[NewsReactionRecord],
) -> bytes:
    """Serialize records deterministically as newline-delimited JSON."""

    lines = [
        json.dumps(
            record.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        for record in records
    ]

    return ("\n".join(lines) + "\n").encode("utf-8")


def export_training_dataset(
    records: Sequence[NewsReactionRecord],
    *,
    output_dir: Path,
    dataset_name: str,
    created_at: datetime | None = None,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> DatasetManifest:
    """
    Export chronological JSONL splits and a reproducible manifest.

    Files are used next by tokenizer, model-training, and evaluation jobs.
    """

    normalized_name = dataset_name.strip()

    if not normalized_name:
        raise ValueError("dataset_name cannot be empty.")

    if created_at is None:
        created_at = datetime.now(timezone.utc)

    if created_at.tzinfo is None or created_at.utcoffset() is None:
        raise ValueError("created_at must be timezone-aware.")

    split_records = assign_chronological_splits(
        records,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    split_counts: dict[str, int] = {}
    split_files: dict[str, str] = {}
    split_checksums: dict[str, str] = {}

    for split in DatasetSplit:
        payload = _serialize_jsonl(split_records[split])
        file_name = f"{normalized_name}_{split.value}.jsonl"
        file_path = output_dir / file_name

        file_path.write_bytes(payload)

        split_counts[split.value] = len(split_records[split])
        split_files[split.value] = file_name
        split_checksums[split.value] = _sha256_bytes(payload)

    # The combined checksum proves which exact split files belong together.
    combined_checksum_payload = json.dumps(
        split_checksums,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    ordered_records = sorted(
        records,
        key=lambda record: record.published_at,
    )

    manifest = DatasetManifest(
        dataset_name=normalized_name,
        created_at=created_at,
        total_records=len(records),
        split_counts=split_counts,
        split_files=split_files,
        split_checksums_sha256=split_checksums,
        dataset_checksum_sha256=_sha256_bytes(
            combined_checksum_payload
        ),
        earliest_published_at=ordered_records[0].published_at,
        latest_published_at=ordered_records[-1].published_at,
        assumptions=(
            "Records are sorted by article publication time.",
            "No random shuffle is used before splitting.",
            "Every record passed provenance and return-formula validation.",
        ),
    )

    manifest_path = output_dir / f"{normalized_name}_manifest.json"
    manifest_path.write_text(
        manifest.model_dump_json(indent=2),
        encoding="utf-8",
    )

    return manifest
