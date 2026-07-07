"""Schemas for leakage-safe historical news reaction intelligence."""

from __future__ import annotations

import math
import re
from datetime import datetime

from pydantic import Field, field_validator, model_validator

from financial_news_intelligence.schemas.common import (
    ProjectSchema,
    SentimentLabel,
)


# ============================================================
# 1. HISTORICAL REACTION COHORT
# ============================================================

class ReactionCohort(ProjectSchema):
    """Comparable historical reactions used to form return scenarios."""

    query_article_id: str
    ticker: str
    sentiment_label: SentimentLabel
    cutoff_at: datetime

    minimum_similarity: float = Field(ge=0, le=1)
    minimum_sample_size: int = Field(ge=1)
    sample_size: int = Field(ge=1)

    lower_quantile: float = Field(ge=0, le=0.5)
    upper_quantile: float = Field(ge=0.5, le=1)

    low_return_pct: float
    median_return_pct: float
    high_return_pct: float

    matched_article_ids: tuple[str, ...] = Field(min_length=1)
    matched_returns_pct: tuple[float, ...] = Field(min_length=1)
    matched_similarity_scores: tuple[float, ...] = Field(min_length=1)

    latest_evidence_published_at: datetime
    evidence_checksum_sha256: str

    calculation_method: str = "linear_quantiles"
    limitations: tuple[str, ...] = ()

    @field_validator("query_article_id", "evidence_checksum_sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        """Article and evidence identities must be SHA-256 values."""

        normalized_value = value.lower()

        if not re.fullmatch(r"[0-9a-f]{64}", normalized_value):
            raise ValueError("Value must be a valid SHA-256 checksum.")

        return normalized_value

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, value: str) -> str:
        """Normalize the cohort ticker."""

        normalized_value = value.strip().upper()

        if not re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,14}", normalized_value):
            raise ValueError("Ticker contains unsupported characters.")

        return normalized_value

    @field_validator("cutoff_at", "latest_evidence_published_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        """Leakage checks require explicit, comparable timezones."""

        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Cohort timestamps must be timezone-aware.")

        return value

    @field_validator(
        "low_return_pct",
        "median_return_pct",
        "high_return_pct",
    )
    @classmethod
    def validate_return(cls, value: float) -> float:
        """Historical stock returns must be finite and not below -100%."""

        if not math.isfinite(value) or value < -100:
            raise ValueError(
                "Historical returns must be finite and at least -100%."
            )

        return float(value)

    @model_validator(mode="after")
    def validate_cohort_relationships(self) -> "ReactionCohort":
        """Check sample coverage, leakage boundaries, and scenario order."""

        if self.sample_size < self.minimum_sample_size:
            raise ValueError(
                "sample_size cannot be below minimum_sample_size."
            )

        if not (
            len(self.matched_article_ids)
            == len(self.matched_returns_pct)
            == len(self.matched_similarity_scores)
            == self.sample_size
        ):
            raise ValueError(
                "Evidence arrays must all match sample_size."
            )

        if len(set(self.matched_article_ids)) != self.sample_size:
            raise ValueError("Historical evidence contains duplicate articles.")

        if any(
            not math.isfinite(value) or value < -100
            for value in self.matched_returns_pct
        ):
            raise ValueError("Matched returns contain invalid values.")

        if any(
            not math.isfinite(score)
            or score < self.minimum_similarity
            or score > 1
            for score in self.matched_similarity_scores
        ):
            raise ValueError(
                "Matched similarity scores violate the cohort threshold."
            )

        if self.latest_evidence_published_at >= self.cutoff_at:
            raise ValueError(
                "Historical evidence must be published before cutoff_at."
            )

        if not (
            self.low_return_pct
            <= self.median_return_pct
            <= self.high_return_pct
        ):
            raise ValueError(
                "Historical scenarios must be ordered low, median, high."
            )

        if self.lower_quantile >= self.upper_quantile:
            raise ValueError(
                "lower_quantile must be below upper_quantile."
            )

        return self
