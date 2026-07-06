"""Build evidence-based downside, base, and upside reaction scenarios.

Purpose
-------
Create transparent historical scenarios for each historical-audit prediction.
Scenarios combine model probabilities with earlier reference-period reaction
statistics. They are research evidence, not investment advice or guarantees.

Inputs and grain
----------------
Reference events must come from train plus validation only. Prediction grain is
one historical-audit ticker-session. Output grain is one scenario row per test
record.

Fallback and limitations
------------------------
The preferred history is same-ticker reference evidence. When one movement
class is absent for that ticker, the class median may fall back to all-ticker
reference evidence and the fallback is recorded explicitly. No test-period,
same-date, contemporary, or future reaction evidence is used.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class InvestmentScenarioError(RuntimeError):
    """Raised when scenario evidence is incomplete or temporally unsafe."""


def build_scenarios(
    reference_events: pd.DataFrame,
    test_predictions: pd.DataFrame,
    minimum_same_ticker_history: int = 20,
) -> pd.DataFrame:
    """Build one strictly reference-based scenario for every test record."""

    event_required = {
        "ticker",
        "target_session_date",
        "movement_label",
        "reaction_return",
    }
    prediction_required = {
        "record_id",
        "ticker",
        "target_session_date",
        "predicted_movement",
        "prob_down",
        "prob_flat",
        "prob_up",
    }
    if event_required - set(reference_events.columns):
        raise InvestmentScenarioError("Reference event schema is incomplete.")
    if prediction_required - set(test_predictions.columns):
        raise InvestmentScenarioError("Prediction evidence schema is incomplete.")
    if minimum_same_ticker_history < 1:
        raise InvestmentScenarioError("Minimum history must be positive.")

    events = reference_events.copy()
    predictions = test_predictions.copy()
    events["target_session_date"] = pd.to_datetime(
        events["target_session_date"],
        errors="coerce",
    ).dt.normalize()
    events["reaction_return"] = pd.to_numeric(
        events["reaction_return"],
        errors="coerce",
    )
    predictions["target_session_date"] = pd.to_datetime(
        predictions["target_session_date"],
        errors="coerce",
    ).dt.normalize()
    if events[["target_session_date", "reaction_return"]].isna().any().any():
        raise InvestmentScenarioError("Reference reaction evidence is invalid.")
    if predictions["target_session_date"].isna().any():
        raise InvestmentScenarioError("Prediction dates are invalid.")

    # Create exactly one scenario row for each persisted test prediction.
    rows: list[dict[str, Any]] = []
    for prediction in predictions.itertuples(index=False):
        # The reference frame is already train/validation-only, but the date
        # comparison provides a second independent temporal safety check.
        earlier = events[
            events["target_session_date"] < prediction.target_session_date
        ].copy()
        same_ticker = earlier[earlier["ticker"] == prediction.ticker].copy()
        if len(same_ticker) < minimum_same_ticker_history:
            raise InvestmentScenarioError(
                f"Record {prediction.record_id} has only {len(same_ticker)} "
                "same-ticker reference reactions."
            )

        # Class medians provide the probability-weighted base case. Any
        # all-ticker fallback is recorded instead of being hidden.
        class_medians: dict[str, float] = {}
        fallback_classes: list[str] = []
        for label in ("Down", "Flat", "Up"):
            values = same_ticker.loc[
                same_ticker["movement_label"] == label,
                "reaction_return",
            ]
            if values.empty:
                values = earlier.loc[
                    earlier["movement_label"] == label,
                    "reaction_return",
                ]
                fallback_classes.append(label)
            if values.empty:
                raise InvestmentScenarioError(
                    f"No earlier reference evidence exists for class {label}."
                )
            class_medians[label] = float(values.median())

        probabilities = {
            "Down": float(prediction.prob_down),
            "Flat": float(prediction.prob_flat),
            "Up": float(prediction.prob_up),
        }
        if any(not np.isfinite(value) or value < 0 for value in probabilities.values()):
            raise InvestmentScenarioError("Prediction probabilities are invalid.")
        if not np.isclose(sum(probabilities.values()), 1.0, atol=1e-6):
            raise InvestmentScenarioError("Prediction probabilities do not sum to one.")

        base_return = sum(
            probabilities[label] * class_medians[label]
            for label in ("Down", "Flat", "Up")
        )
        downside_return = float(same_ticker["reaction_return"].quantile(0.10))
        upside_return = float(same_ticker["reaction_return"].quantile(0.90))
        if downside_return > upside_return:
            raise InvestmentScenarioError("Scenario quantiles are reversed.")

        rows.append(
            {
                "record_id": int(prediction.record_id),
                "ticker": prediction.ticker,
                "target_session_date": str(prediction.target_session_date.date()),
                "predicted_movement": prediction.predicted_movement,
                "prob_down": probabilities["Down"],
                "prob_flat": probabilities["Flat"],
                "prob_up": probabilities["Up"],
                "downside_return": downside_return,
                "base_return": float(base_return),
                "upside_return": upside_return,
                "same_ticker_history_count": int(len(same_ticker)),
                "reference_history_end_date": str(
                    same_ticker["target_session_date"].max().date()
                ),
                "class_median_fallbacks": ",".join(sorted(fallback_classes)),
                "history_scope": "train_validation_same_ticker_earlier_only",
                "scenario_method": (
                    "reference_quantiles_plus_probability_weighted_class_medians"
                ),
                "disclaimer": (
                    "Historical research scenario; not investment advice or a "
                    "forecast guarantee."
                ),
            }
        )

    result = pd.DataFrame(rows)
    if result["record_id"].duplicated().any():
        raise InvestmentScenarioError("Scenario output is not one row per record.")
    if set(result["record_id"].astype(int)) != set(
        predictions["record_id"].astype(int)
    ):
        raise InvestmentScenarioError("Scenario coverage is incomplete.")
    return result.sort_values("record_id").reset_index(drop=True)
