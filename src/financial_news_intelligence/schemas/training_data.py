"""Schemas for verified news-to-market reaction training data."""

from __future__ import annotations

import math
import re
from datetime import date, datetime
from enum import Enum
from pydantic import Field, field_validator, model_validator

from financial_news_intelligence.schemas.common import (
    MovementLabel,
    ProjectSchema,
    SentimentLabel,
)
from financial_news_intelligence.schemas.provenance import SourceProvenance


# ============================================================
# 1. CONTROLLED DATASET SPLITS
# ============================================================

class DatasetSplit(str, Enum):
    """Chronological dataset partitions used during model development."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


# ============================================================
# 2. VERIFIED ARTICLE-REACTION RECORD
# ============================================================

class NewsReactionRecord(ProjectSchema):
    """One article linked to its verified target-session market reaction."""

    article_id: str
    article_text: str = Field(min_length=1)
    published_at: datetime

    company: str = Field(min_length=1)
    ticker: str

    sentiment_label: SentimentLabel
    sentiment_confidence: float = Field(ge=0, le=1)

    target_session: date
    open_price: float = Field(gt=0)
    close_price: float = Field(gt=0)
    return_pct: float
    movement_label: MovementLabel
    flat_threshold_pct: float = Field(ge=0)

    news_provenance: SourceProvenance
    price_source_id: str = Field(min_length=1)
    price_checksum_sha256: str

    dataset_split: DatasetSplit | None = None

    @field_validator("article_id", "price_checksum_sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        """Require deterministic 64-character SHA-256 identifiers."""

        normalized_value = value.lower()

        if not re.fullmatch(r"[0-9a-f]{64}", normalized_value):
            raise ValueError("Value must be a valid SHA-256 checksum.")

        return normalized_value

    @field_validator("published_at")
    @classmethod
    def require_publication_timezone(cls, value: datetime) -> datetime:
        """Publication times must identify their timezone explicitly."""

        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("published_at must be timezone-aware.")

        return value

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, value: str) -> str:
        """Normalize and validate a US ticker symbol."""

        normalized_value = value.strip().upper()

        if not re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,14}", normalized_value):
            raise ValueError("Ticker contains unsupported characters.")

        return normalized_value

    @field_validator("return_pct")
    @classmethod
    def require_finite_return(cls, value: float) -> float:
        """Reject infinite and non-numeric return values."""

        if not math.isfinite(value):
            raise ValueError("return_pct must be finite.")

        return float(value)

    @model_validator(mode="after")
    def validate_reaction_relationships(self) -> "NewsReactionRecord":
        """Protect temporal order, formulas, and movement labels."""

        if self.published_at.date() > self.target_session:
            raise ValueError(
                "target_session cannot occur before article publication."
            )

        if self.news_provenance.as_of_date > self.published_at.date():
            raise ValueError(
                "News provenance cannot have a future as-of date."
            )

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
            expected_label = MovementLabel.UP
        elif self.return_pct < -self.flat_threshold_pct:
            expected_label = MovementLabel.DOWN
        else:
            expected_label = MovementLabel.FLAT

        if self.movement_label != expected_label:
            raise ValueError(
                "movement_label does not match the configured threshold."
            )

        return self


# ============================================================
# 3. REPRODUCIBLE DATASET MANIFEST
# ============================================================

class DatasetManifest(ProjectSchema):
    """Metadata and checksums for exported chronological dataset splits."""

    dataset_name: str = Field(min_length=1)
    created_at: datetime

    total_records: int = Field(ge=3)
    split_counts: dict[str, int]
    split_files: dict[str, str]
    split_checksums_sha256: dict[str, str]
    dataset_checksum_sha256: str

    earliest_published_at: datetime
    latest_published_at: datetime

    split_strategy: str = "chronological"
    assumptions: tuple[str, ...] = ()

    @field_validator(
        "created_at",
        "earliest_published_at",
        "latest_published_at",
    )
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        """All manifest timestamps must be timezone-aware."""

        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Manifest timestamps must be timezone-aware.")

        return value

    @field_validator("dataset_checksum_sha256")
    @classmethod
    def validate_dataset_checksum(cls, value: str) -> str:
        """Require a valid checksum for the combined dataset evidence."""

        normalized_value = value.lower()

        if not re.fullmatch(r"[0-9a-f]{64}", normalized_value):
            raise ValueError(
                "dataset_checksum_sha256 must be a SHA-256 value."
            )

        return normalized_value

    @model_validator(mode="after")
    def validate_manifest_relationships(self) -> "DatasetManifest":
        """Check split coverage, files, checksums, and date boundaries."""

        expected_keys = {split.value for split in DatasetSplit}

        if set(self.split_counts) != expected_keys:
            raise ValueError("split_counts must contain train, validation, and test.")

        if set(self.split_files) != expected_keys:
            raise ValueError("split_files must contain train, validation, and test.")

        if set(self.split_checksums_sha256) != expected_keys:
            raise ValueError(
                "split_checksums_sha256 must contain all dataset splits."
            )

        if sum(self.split_counts.values()) != self.total_records:
            raise ValueError("Split counts must sum to total_records.")

        if any(count < 1 for count in self.split_counts.values()):
            raise ValueError("Every chronological split must contain records.")

        for checksum in self.split_checksums_sha256.values():
            if not re.fullmatch(r"[0-9a-f]{64}", checksum.lower()):
                raise ValueError("Every split checksum must be SHA-256.")

        if self.earliest_published_at > self.latest_published_at:
            raise ValueError(
                "earliest_published_at cannot follow latest_published_at."
            )

        return self
