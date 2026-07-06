"""Explain the verified movement champion without changing it.

Purpose
-------
Produce global model-native drivers and local model-agnostic perturbation
explanations for historical-audit predictions. Local explanations replace one
raw feature at a time with a train-plus-validation reference value and measure
the change in predicted-class probability.

Inputs and grain
----------------
Inputs are the fitted non-baseline champion, the split model table, and the
exact approved feature lists. Global output grain is one transformed feature.
Local output grain is one test record, raw feature, and predicted class.

Limitations
-----------
Perturbation effects describe model sensitivity, not causal effects. Correlated
features can share information, so one-feature changes must not be interpreted
as proof that a feature caused a market move.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from financial_news_intelligence.models.movement_dataset import LABEL_ORDER


class MovementExplainabilityError(RuntimeError):
    """Raised when champion explanations cannot be produced safely."""


def global_drivers(
    importance_frame: pd.DataFrame,
    top_n: int = 50,
    minimum_nonzero_features: int = 3,
) -> pd.DataFrame:
    """Return ranked non-negative and meaningfully nonzero global drivers."""

    required = {"feature", "importance", "method"}
    missing = sorted(required - set(importance_frame.columns))
    if missing:
        raise MovementExplainabilityError(
            f"Global importance is missing columns: {missing}"
        )
    if top_n < minimum_nonzero_features or minimum_nonzero_features < 1:
        raise MovementExplainabilityError("Global driver thresholds are invalid.")

    # Copy before conversion so the caller's model-native evidence remains
    # unchanged for independent verification.
    result = importance_frame.copy()
    result["importance"] = pd.to_numeric(result["importance"], errors="coerce")
    if result["importance"].isna().any() or (result["importance"] < 0).any():
        raise MovementExplainabilityError("Global importance values are invalid.")
    result = result.sort_values(
        ["importance", "feature"],
        ascending=[False, True],
    ).head(top_n)
    nonzero_count = int(result["importance"].gt(0).sum())
    if nonzero_count < minimum_nonzero_features:
        raise MovementExplainabilityError(
            f"Only {nonzero_count} nonzero global drivers were produced."
        )

    result.insert(0, "rank", np.arange(1, len(result) + 1))
    result["interpretation"] = (
        "Higher values indicate model sensitivity, not causal influence."
    )
    return result.reset_index(drop=True)


def _reference_values(
    reference_frame: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    text_features: list[str],
) -> dict[str, Any]:
    """Calculate train-plus-validation reference values for raw features."""

    values: dict[str, Any] = {}
    for feature in numeric_features:
        numeric_values = pd.to_numeric(reference_frame[feature], errors="coerce")
        median = float(numeric_values.median())
        if not np.isfinite(median):
            raise MovementExplainabilityError(
                f"Numeric reference value is unavailable: {feature}"
            )
        values[feature] = median
    for feature in categorical_features:
        modes = reference_frame[feature].dropna().astype(str).mode()
        if modes.empty:
            raise MovementExplainabilityError(
                f"Categorical reference value is unavailable: {feature}"
            )
        values[feature] = str(sorted(modes.tolist())[0])

    # An empty filing-text reference removes all SEC text n-grams while keeping
    # the other raw features unchanged. This measures text sensitivity without
    # borrowing any test-period phrase or creating synthetic market evidence.
    for feature in text_features:
        if feature not in reference_frame.columns:
            raise MovementExplainabilityError(
                f"Text reference column is unavailable: {feature}"
            )
        values[feature] = ""
    return values


def local_perturbation_drivers(
    champion: Pipeline,
    model_table: pd.DataFrame,
    test_predictions: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    text_features: list[str] | None = None,
    top_n_per_record: int = 5,
    minimum_nonzero_per_record: int = 1,
) -> pd.DataFrame:
    """Measure raw-feature probability sensitivity for every test record."""

    approved_text = list(text_features or [])
    features = numeric_features + categorical_features + approved_text
    required_prediction_columns = {
        "record_id",
        "ticker",
        "target_session_date",
        "predicted_movement",
    }
    missing = sorted(required_prediction_columns - set(test_predictions.columns))
    if missing:
        raise MovementExplainabilityError(
            f"Test predictions are missing columns: {missing}"
        )
    if top_n_per_record < minimum_nonzero_per_record:
        raise MovementExplainabilityError("Local driver thresholds are invalid.")

    reference_frame = model_table[
        model_table["split"].isin(["train", "validation"])
    ].copy()
    test_frame = model_table[model_table["split"] == "test"].copy()
    if reference_frame.empty or test_frame.empty:
        raise MovementExplainabilityError("Reference or test split is empty.")

    # Join stable record identifiers from saved predictions back to the test
    # feature rows. This prevents iteration order from silently changing IDs.
    test_keys = test_predictions[
        ["record_id", "ticker", "target_session_date", "predicted_movement"]
    ].copy()
    test_keys["target_session_date"] = pd.to_datetime(
        test_keys["target_session_date"],
        errors="coerce",
    ).dt.normalize()
    test_frame["target_session_date"] = pd.to_datetime(
        test_frame["target_session_date"],
        errors="coerce",
    ).dt.normalize()
    test_frame = test_frame.merge(
        test_keys,
        on=["ticker", "target_session_date"],
        how="inner",
        validate="one_to_one",
    )
    if len(test_frame) != len(test_predictions):
        raise MovementExplainabilityError("Test prediction and feature rows differ.")

    reference_values = _reference_values(
        reference_frame,
        numeric_features,
        categorical_features,
        approved_text,
    )
    class_names = [
        str(value) for value in champion.named_steps["classifier"].classes_
    ]
    if set(class_names) != set(LABEL_ORDER):
        raise MovementExplainabilityError("Champion class order changed.")

    # Explain every test record separately; no aggregate explanation may hide
    # a record that produced only zero or missing effects.
    rows: list[dict[str, Any]] = []
    for record in test_frame.itertuples(index=False):
        raw_values = {feature: getattr(record, feature) for feature in features}
        predicted_class = str(record.predicted_movement)
        class_index = class_names.index(predicted_class)

        # Build one batch containing the original row followed by one copy for
        # each single-feature perturbation. One model call per record is much
        # faster and more stable than one call per feature, while the semantic
        # meaning of every effect remains unchanged.
        batch_rows = [dict(raw_values)]
        for feature in features:
            perturbed_values = dict(raw_values)
            perturbed_values[feature] = reference_values[feature]
            batch_rows.append(perturbed_values)
        probability_batch = champion.predict_proba(pd.DataFrame(batch_rows))
        baseline_probability = float(probability_batch[0, class_index])
        perturbed_probabilities = probability_batch[1:, class_index]
        effects: list[dict[str, Any]] = []

        for feature_index, feature in enumerate(features):
            original_value = raw_values[feature]
            perturbed_probability = float(perturbed_probabilities[feature_index])
            effect = baseline_probability - perturbed_probability
            # Long SEC descriptions are shortened in the explanation table.
            # The model still receives the complete original string above.
            display_original = original_value
            if feature in approved_text:
                display_original = str(original_value)[:160]
            effects.append(
                {
                    "record_id": int(record.record_id),
                    "ticker": record.ticker,
                    "target_session_date": str(record.target_session_date.date()),
                    "predicted_class": predicted_class,
                    "predicted_probability": baseline_probability,
                    "feature": feature,
                    "original_value": display_original,
                    "reference_value": reference_values[feature],
                    "probability_effect": float(effect),
                    "absolute_effect": float(abs(effect)),
                    "direction": (
                        "supports_prediction"
                        if effect >= 0
                        else "opposes_prediction"
                    ),
                    "reference_scope": "train_validation_only",
                    "method": "single_feature_reference_perturbation",
                }
            )

        effects.sort(
            key=lambda value: (-value["absolute_effect"], value["feature"])
        )
        selected = effects[:top_n_per_record]
        nonzero_count = sum(row["absolute_effect"] > 1e-12 for row in selected)
        if nonzero_count < minimum_nonzero_per_record:
            raise MovementExplainabilityError(
                f"Record {record.record_id} has no meaningful local driver."
            )
        for rank, effect_row in enumerate(selected, start=1):
            effect_row["rank_within_record"] = rank
            rows.append(effect_row)

    result = pd.DataFrame(rows)
    coverage = result.groupby("record_id", observed=True).size()
    expected_ids = set(test_predictions["record_id"].astype(int))
    if set(coverage.index.astype(int)) != expected_ids:
        raise MovementExplainabilityError("Local explanation coverage is incomplete.")
    if coverage.lt(top_n_per_record).any():
        raise MovementExplainabilityError("A test record has too few local drivers.")
    return result
