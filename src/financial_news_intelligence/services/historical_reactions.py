"""Build leakage-safe historical news reaction cohorts."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from typing import Mapping, Sequence

from financial_news_intelligence.schemas.common import SentimentLabel
from financial_news_intelligence.schemas.historical_intelligence import (
    ReactionCohort,
)
from financial_news_intelligence.schemas.training_data import (
    NewsReactionRecord,
)


# ============================================================
# 1. DOMAIN-SPECIFIC ERROR
# ============================================================

class InsufficientHistoricalEvidenceError(ValueError):
    """Raised when too few comparable historical reactions are available."""


# ============================================================
# 2. QUANTILE CALCULATION
# ============================================================

def calculate_linear_quantile(
    values: Sequence[float],
    quantile: float,
) -> float:
    """
    Calculate a deterministic linearly interpolated quantile.

    This avoids a heavy numerical dependency in the service layer.
    """

    if not values:
        raise ValueError("At least one value is required.")

    if not math.isfinite(quantile) or not 0 <= quantile <= 1:
        raise ValueError("quantile must be between zero and one.")

    ordered_values = sorted(float(value) for value in values)

    if any(not math.isfinite(value) for value in ordered_values):
        raise ValueError("Quantile values must be finite.")

    if len(ordered_values) == 1:
        return ordered_values[0]

    position = (len(ordered_values) - 1) * quantile
    lower_index = math.floor(position)
    upper_index = math.ceil(position)

    if lower_index == upper_index:
        return ordered_values[lower_index]

    interpolation_weight = position - lower_index

    return (
        ordered_values[lower_index]
        + (
            ordered_values[upper_index]
            - ordered_values[lower_index]
        )
        * interpolation_weight
    )


# ============================================================
# 3. EVIDENCE CHECKSUM
# ============================================================

def _compute_evidence_checksum(
    evidence_rows: Sequence[dict[str, object]],
) -> str:
    """Hash the exact article IDs, returns, scores, and price evidence."""

    canonical_payload = json.dumps(
        list(evidence_rows),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")

    return hashlib.sha256(canonical_payload).hexdigest()


# ============================================================
# 4. HISTORICAL COHORT BUILDER
# ============================================================

def build_historical_reaction_cohort(
    *,
    query_article_id: str,
    ticker: str,
    sentiment_label: SentimentLabel,
    cutoff_at: datetime,
    candidate_records: Sequence[NewsReactionRecord],
    similarity_scores: Mapping[str, float],
    minimum_similarity: float = 0.70,
    minimum_sample_size: int = 5,
    maximum_sample_size: int = 100,
    lower_quantile: float = 0.10,
    upper_quantile: float = 0.90,
) -> ReactionCohort:
    """
    Select comparable earlier articles and calculate return scenarios.

    Leakage rule: records published at or after cutoff_at are excluded.
    Grain: one query article and one historical evidence cohort.
    """

    if cutoff_at.tzinfo is None or cutoff_at.utcoffset() is None:
        raise ValueError("cutoff_at must be timezone-aware.")

    if not 0 <= minimum_similarity <= 1:
        raise ValueError("minimum_similarity must be between zero and one.")

    if minimum_sample_size < 1:
        raise ValueError("minimum_sample_size must be positive.")

    if maximum_sample_size < minimum_sample_size:
        raise ValueError(
            "maximum_sample_size cannot be below minimum_sample_size."
        )

    if not 0 <= lower_quantile < 0.5:
        raise ValueError("lower_quantile must be below 0.5.")

    if not 0.5 < upper_quantile <= 1:
        raise ValueError("upper_quantile must be above 0.5.")

    normalized_ticker = ticker.strip().upper()
    eligible_records: list[tuple[NewsReactionRecord, float]] = []

    for record in candidate_records:
        # The query article itself can never become its own evidence.
        if record.article_id == query_article_id:
            continue

        if record.ticker != normalized_ticker:
            continue

        if record.sentiment_label != sentiment_label:
            continue

        if record.published_at >= cutoff_at:
            continue

        similarity_score = similarity_scores.get(record.article_id)

        if similarity_score is None:
            continue

        if (
            not math.isfinite(similarity_score)
            or similarity_score < 0
            or similarity_score > 1
        ):
            raise ValueError(
                f"Invalid similarity score for article {record.article_id}."
            )

        if similarity_score < minimum_similarity:
            continue

        eligible_records.append((record, similarity_score))

    # Most similar records are preferred. Recent records break equal scores.
    eligible_records.sort(
        key=lambda item: (
            item[1],
            item[0].published_at.astimezone(timezone.utc),
            item[0].article_id,
        ),
        reverse=True,
    )

    selected_records = eligible_records[:maximum_sample_size]

    if len(selected_records) < minimum_sample_size:
        raise InsufficientHistoricalEvidenceError(
            "Historical cohort contains "
            f"{len(selected_records)} records; "
            f"at least {minimum_sample_size} are required."
        )

    returns = [
        record.return_pct
        for record, _ in selected_records
    ]

    evidence_rows = [
        {
            "article_id": record.article_id,
            "published_at_utc": record.published_at.astimezone(
                timezone.utc
            ).isoformat(),
            "return_pct": record.return_pct,
            "similarity_score": similarity_score,
            "price_checksum_sha256": record.price_checksum_sha256,
        }
        for record, similarity_score in selected_records
    ]

    return ReactionCohort(
        query_article_id=query_article_id,
        ticker=normalized_ticker,
        sentiment_label=sentiment_label,
        cutoff_at=cutoff_at,
        minimum_similarity=minimum_similarity,
        minimum_sample_size=minimum_sample_size,
        sample_size=len(selected_records),
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
        low_return_pct=round(
            calculate_linear_quantile(returns, lower_quantile),
            6,
        ),
        median_return_pct=round(
            calculate_linear_quantile(returns, 0.50),
            6,
        ),
        high_return_pct=round(
            calculate_linear_quantile(returns, upper_quantile),
            6,
        ),
        matched_article_ids=tuple(
            record.article_id
            for record, _ in selected_records
        ),
        matched_returns_pct=tuple(returns),
        matched_similarity_scores=tuple(
            similarity_score
            for _, similarity_score in selected_records
        ),
        latest_evidence_published_at=max(
            record.published_at
            for record, _ in selected_records
        ),
        evidence_checksum_sha256=_compute_evidence_checksum(
            evidence_rows
        ),
        limitations=(
            "Historical similarity does not guarantee a future outcome.",
            "Only records published before the query cutoff are used.",
            "Returns describe the configured target-session window.",
        ),
    )
