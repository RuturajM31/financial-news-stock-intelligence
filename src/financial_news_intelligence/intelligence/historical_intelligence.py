"""Create leakage-safe phrases, historical matches, and company context.

Purpose
-------
Turn verified SEC event evidence into three auditable intelligence tables:
class-associated sentiment phrases, similar earlier events and reactions, and
company context derived only from verified reference-period evidence.

Leakage boundary
----------------
Global phrases and company context use train plus validation events only.
Historical matches use the same reference pool and must be strictly earlier
than each historical-audit query session. Test-period events never become
reference evidence.

Limitations
-----------
TF-IDF similarity measures word overlap, not semantic truth or causality.
Company context contains observed SEC-event evidence only; it does not invent
financial statements, valuations, recommendations, or live company facts.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SENTIMENT_LABELS = ("Bearish", "Neutral", "Bullish")


class HistoricalIntelligenceError(RuntimeError):
    """Raised when intelligence violates time, coverage, or source rules."""


def _validated_probability_weights(
    reference_events: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """Return normalized sentiment-probability weights for every class.

    The fallback is used only when the reference period lacks one or more hard
    sentiment labels. It does not invent labels: every weight comes from the
    verified pre-existing sentiment probabilities saved with each SEC event.
    """

    probability_columns = {
        "Bearish": "prob_bearish",
        "Neutral": "prob_neutral",
        "Bullish": "prob_bullish",
    }
    missing = sorted(set(probability_columns.values()) - set(reference_events.columns))
    if missing:
        raise HistoricalIntelligenceError(
            "Reference events are missing a hard sentiment class and the "
            f"probability evidence needed for fallback: {missing}"
        )

    ordered_columns = [probability_columns[label] for label in SENTIMENT_LABELS]
    probabilities = reference_events[ordered_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )
    values = probabilities.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise HistoricalIntelligenceError(
            "Sentiment probability fallback contains non-finite values."
        )
    if (values < 0.0).any() or (values > 1.0).any():
        raise HistoricalIntelligenceError(
            "Sentiment probability fallback contains values outside [0, 1]."
        )
    if not np.allclose(values.sum(axis=1), 1.0, atol=1e-6):
        raise HistoricalIntelligenceError(
            "Sentiment probability fallback rows do not sum to one."
        )

    weights: dict[str, np.ndarray] = {}
    for label, column in probability_columns.items():
        class_weights = probabilities[column].to_numpy(dtype=float)
        if class_weights.sum() <= 0.0 or (1.0 - class_weights).sum() <= 0.0:
            raise HistoricalIntelligenceError(
                f"Sentiment probability fallback has no contrast for {label}."
            )
        weights[label] = class_weights
    return weights


def _weighted_tfidf_mean(matrix: Any, weights: np.ndarray) -> np.ndarray:
    """Return one probability-weighted mean over sparse TF-IDF rows."""

    weight_total = float(weights.sum())
    if weight_total <= 0.0:
        raise HistoricalIntelligenceError(
            "Sentiment phrase weights must have a positive total."
        )
    weighted = matrix.multiply(weights[:, np.newaxis]).sum(axis=0)
    return np.asarray(weighted).ravel() / weight_total


def sentiment_phrases(
    reference_events: pd.DataFrame,
    top_n_per_class: int = 15,
    minimum_phrases_per_class: int = 3,
) -> pd.DataFrame:
    """Rank train/validation phrases for each sentiment class.

    Hard labels are used when all three classes occur in the reference period.
    When a hard class is absent, verified per-event sentiment probabilities are
    used as soft reference-only weights. This preserves three-class coverage
    without reading test rows or fabricating a missing label.
    """

    required = {"text", "sentiment_label", "target_session_date"}
    missing = sorted(required - set(reference_events.columns))
    if missing or reference_events.empty:
        raise HistoricalIntelligenceError(
            f"Reference event evidence is incomplete for phrases: {missing}"
        )
    if top_n_per_class < minimum_phrases_per_class or minimum_phrases_per_class < 1:
        raise HistoricalIntelligenceError(
            "Phrase coverage thresholds are invalid."
        )

    # Normalize reference values once. Only train/validation rows supplied by
    # the caller can contribute to either hard-label or probability weighting.
    texts = reference_events["text"].fillna("").astype(str).str.strip()
    labels = reference_events["sentiment_label"].astype(str).str.strip()
    observed_labels = set(labels)
    unknown_labels = sorted(observed_labels - set(SENTIMENT_LABELS))
    if not observed_labels or unknown_labels:
        raise HistoricalIntelligenceError(
            f"Reference events contain invalid sentiment labels: {unknown_labels}"
        )
    if texts.str.len().lt(2).all():
        raise HistoricalIntelligenceError("Reference event text is empty.")

    hard_label_coverage_complete = observed_labels == set(SENTIMENT_LABELS)
    probability_fallback_used = not hard_label_coverage_complete
    if hard_label_coverage_complete:
        weights_by_class = {
            label: labels.eq(label).to_numpy(dtype=float)
            for label in SENTIMENT_LABELS
        }
        method = "reference_only_class_mean_tfidf_difference"
    else:
        weights_by_class = _validated_probability_weights(reference_events)
        method = "reference_only_probability_weighted_tfidf_difference"

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_features=5000,
        stop_words="english",
    )
    try:
        matrix = vectorizer.fit_transform(texts)
    except ValueError as exc:
        raise HistoricalIntelligenceError(
            f"Sentiment phrase vocabulary is unavailable: {exc}"
        ) from exc
    feature_names = np.asarray(vectorizer.get_feature_names_out())
    if feature_names.size < minimum_phrases_per_class:
        raise HistoricalIntelligenceError(
            "Sentiment phrase vocabulary is too small for required coverage."
        )

    rows: list[dict[str, Any]] = []
    for sentiment_label in SENTIMENT_LABELS:
        class_weights = weights_by_class[sentiment_label]
        complement_weights = 1.0 - class_weights
        inside = _weighted_tfidf_mean(matrix, class_weights)
        outside = _weighted_tfidf_mean(matrix, complement_weights)
        association = inside - outside

        # Positive contrast remains the primary evidence. Probability fallback
        # may use high weighted relevance only when a missing hard class leaves
        # fewer than the required number of positive-contrast phrases.
        positive_indices = [
            int(index)
            for index in np.argsort(-association)
            if association[index] > 0.0
        ]
        selected_indices = positive_indices[:top_n_per_class]
        selection_basis = {
            index: "positive_tfidf_contrast" for index in selected_indices
        }
        if probability_fallback_used and len(selected_indices) < minimum_phrases_per_class:
            for index in np.argsort(-inside):
                feature_index = int(index)
                if feature_index in selection_basis or inside[feature_index] <= 0.0:
                    continue
                selected_indices.append(feature_index)
                selection_basis[feature_index] = "probability_weighted_relevance"
                if len(selected_indices) >= top_n_per_class:
                    break

        if len(selected_indices) < minimum_phrases_per_class:
            raise HistoricalIntelligenceError(
                f"Only {len(selected_indices)} phrases were produced for "
                f"{sentiment_label}."
            )

        hard_label_count = int(labels.eq(sentiment_label).sum())
        for rank, feature_index in enumerate(selected_indices, start=1):
            rows.append(
                {
                    "sentiment_label": sentiment_label,
                    "rank": rank,
                    "phrase": str(feature_names[feature_index]),
                    "association_score": float(association[feature_index]),
                    "relevance_score": float(inside[feature_index]),
                    "reference_record_count": int(len(reference_events)),
                    "hard_label_record_count": hard_label_count,
                    "effective_reference_weight": float(class_weights.sum()),
                    "probability_fallback_used": probability_fallback_used,
                    "selection_basis": selection_basis[feature_index],
                    "method": method,
                    "interpretation": (
                        "Reference-only phrase association; probability fallback "
                        "uses verified sentiment probabilities and is not causality."
                    ),
                }
            )
            if rank >= top_n_per_class:
                break
    return pd.DataFrame(rows)


def earlier_only_matches(
    reference_events: pd.DataFrame,
    query_events: pd.DataFrame,
    test_predictions: pd.DataFrame,
    top_n_per_record: int = 5,
    minimum_matches_per_record: int = 3,
) -> pd.DataFrame:
    """Return same-ticker TF-IDF matches from reference evidence only."""

    event_required = {
        "article_id",
        "ticker",
        "target_session_date",
        "text",
        "source_url",
        "movement_label",
        "reaction_return",
    }
    prediction_required = {
        "record_id",
        "ticker",
        "target_session_date",
        "predicted_movement",
    }
    if event_required - set(reference_events.columns):
        raise HistoricalIntelligenceError("Reference history schema is incomplete.")
    if event_required - set(query_events.columns):
        raise HistoricalIntelligenceError("Query event schema is incomplete.")
    if prediction_required - set(test_predictions.columns):
        raise HistoricalIntelligenceError("Test prediction schema is incomplete.")
    if top_n_per_record < minimum_matches_per_record or minimum_matches_per_record < 1:
        raise HistoricalIntelligenceError("Historical match thresholds are invalid.")

    reference = reference_events.copy()
    queries = query_events.copy()
    predictions = test_predictions.copy()
    for frame in (reference, queries, predictions):
        frame["target_session_date"] = pd.to_datetime(
            frame["target_session_date"],
            errors="coerce",
        ).dt.normalize()
    date_frames = (reference, queries, predictions)
    if any(
        frame["target_session_date"].isna().any()
        for frame in date_frames
    ):
        raise HistoricalIntelligenceError("Historical intelligence dates are invalid.")

    # Process each saved test identifier independently so missing coverage can
    # be reported against the exact record that failed.
    rows: list[dict[str, Any]] = []
    for prediction in predictions.itertuples(index=False):
        candidates = reference[
            (reference["ticker"] == prediction.ticker)
            & (reference["target_session_date"] < prediction.target_session_date)
        ].copy()
        if len(candidates) < minimum_matches_per_record:
            raise HistoricalIntelligenceError(
                f"Record {prediction.record_id} has only {len(candidates)} "
                "same-ticker earlier reference events."
            )

        # The query text may contain several SEC events mapped to one test day.
        # Combining them creates one transparent ticker-session query document.
        query_text = " ".join(
            queries[
                (queries["ticker"] == prediction.ticker)
                & (
                    queries["target_session_date"]
                    == prediction.target_session_date
                )
            ]["text"].fillna("").astype(str)
        ).strip()
        if not query_text:
            raise HistoricalIntelligenceError(
                f"Test query text is missing for record {prediction.record_id}."
            )

        corpus = candidates["text"].fillna("").astype(str).tolist()
        vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
            max_features=5000,
            stop_words="english",
        )
        try:
            matrix = vectorizer.fit_transform(corpus + [query_text])
        except ValueError as exc:
            raise HistoricalIntelligenceError(
                f"Historical similarity vocabulary failed: {exc}"
            ) from exc
        similarities = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
        ranked_indices = np.argsort(-similarities)[:top_n_per_record]

        for rank, candidate_index in enumerate(ranked_indices, start=1):
            candidate = candidates.iloc[int(candidate_index)]
            if candidate["target_session_date"] >= prediction.target_session_date:
                raise HistoricalIntelligenceError(
                    "Historical retrieval included a non-earlier event."
                )
            rows.append(
                {
                    "record_id": int(prediction.record_id),
                    "ticker": prediction.ticker,
                    "query_session_date": str(prediction.target_session_date.date()),
                    "predicted_movement": prediction.predicted_movement,
                    "historical_rank": rank,
                    "historical_article_id": candidate["article_id"],
                    "historical_session_date": str(
                        candidate["target_session_date"].date()
                    ),
                    "historical_source_url": candidate["source_url"],
                    "historical_movement_label": candidate["movement_label"],
                    "historical_reaction_return": float(
                        candidate["reaction_return"]
                    ),
                    "text_similarity": float(similarities[candidate_index]),
                    "candidate_scope": "train_validation_reference_only",
                    "retrieval_rule": (
                        "same_ticker_reference_and_strictly_earlier_only"
                    ),
                }
            )

    result = pd.DataFrame(rows)
    coverage = result.groupby("record_id", observed=True).size()
    expected_ids = set(predictions["record_id"].astype(int))
    if set(coverage.index.astype(int)) != expected_ids:
        raise HistoricalIntelligenceError("Historical match coverage is incomplete.")
    if coverage.lt(minimum_matches_per_record).any():
        raise HistoricalIntelligenceError("A test record has too few matches.")
    return result


def company_context(
    reference_events: pd.DataFrame,
    expected_tickers: set[str] | None = None,
) -> pd.DataFrame:
    """Summarize only train/validation SEC-event and company evidence."""

    required = {
        "ticker",
        "company",
        "target_session_date",
        "movement_label",
        "reaction_return",
        "source_name",
    }
    missing = sorted(required - set(reference_events.columns))
    if missing or reference_events.empty:
        raise HistoricalIntelligenceError(
            f"Company context evidence is incomplete: {missing}"
        )
    frame = reference_events.copy()
    frame["target_session_date"] = pd.to_datetime(
        frame["target_session_date"],
        errors="coerce",
    ).dt.normalize()
    frame["reaction_return"] = pd.to_numeric(
        frame["reaction_return"],
        errors="coerce",
    )
    if frame[["target_session_date", "reaction_return"]].isna().any().any():
        raise HistoricalIntelligenceError("Company context values are invalid.")

    # Build one deterministic row per ticker rather than merging separate
    # company-name and movement-count tables with fragile cardinality.
    rows: list[dict[str, Any]] = []
    for ticker, ticker_frame in frame.groupby("ticker", observed=True):
        # Choose the most frequently observed verified company name. Sorting
        # breaks ties deterministically without creating a one-to-one merge bug.
        company_counts = ticker_frame["company"].astype(str).value_counts()
        maximum_count = int(company_counts.max())
        canonical_company = sorted(
            company_counts[company_counts == maximum_count].index
        )[0]
        movement_counts = {
            label: int((ticker_frame["movement_label"] == label).sum())
            for label in ("Down", "Flat", "Up")
        }
        rows.append(
            {
                "ticker": str(ticker),
                "company": canonical_company,
                "verified_event_count": int(len(ticker_frame)),
                "company_name_variants": int(company_counts.size),
                "first_event_date": str(
                    ticker_frame["target_session_date"].min().date()
                ),
                "last_event_date": str(
                    ticker_frame["target_session_date"].max().date()
                ),
                "observed_reaction_mean": float(
                    ticker_frame["reaction_return"].mean()
                ),
                "observed_reaction_median": float(
                    ticker_frame["reaction_return"].median()
                ),
                "observed_reaction_std": float(
                    ticker_frame["reaction_return"].std(ddof=0)
                ),
                "down_event_count": movement_counts["Down"],
                "flat_event_count": movement_counts["Flat"],
                "up_event_count": movement_counts["Up"],
                "context_source": (
                    "train_validation_verified_sec_event_evidence_only"
                ),
                "limitation": (
                    "Observed SEC-event context only; no invented fundamentals."
                ),
            }
        )

    result = pd.DataFrame(rows).sort_values("ticker").reset_index(drop=True)
    if result["ticker"].duplicated().any():
        raise HistoricalIntelligenceError("Company context is not one row per ticker.")
    if expected_tickers is not None and set(result["ticker"]) != expected_tickers:
        raise HistoricalIntelligenceError("Company context ticker coverage changed.")
    return result
