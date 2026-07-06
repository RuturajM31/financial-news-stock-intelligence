"""Load and serve the verified stock-movement champion.

Purpose
-------
Use the saved scikit-learn champion, frozen decision policy, historical model
table, and verified price-session calendar without retraining or changing any
artifact.

Prediction boundary
-------------------
The current evidence ends in 2020. The API therefore supports research
predictions only for ticker-session rows already present in the verified model
table. It fails closed for live or unsupported dates instead of silently using
stale market features.
"""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

# The isolated movement worker receives a package-level ``pyarrow`` shim
# through ``PYTHONPATH`` before this module is imported. The read-only Mac
# diagnostic proved that this filesystem-level boundary allows the verified
# joblib artifact to load. No second meta-path blocker is installed here.

import numpy as np
import pandas as pd

from .artifacts import ArtifactPaths
from .errors import ApiProblem


LABEL_ORDER = ("Down", "Flat", "Up")
SENTIMENT_ORDER = ("Bearish", "Neutral", "Bullish")
MARKET_TIMEZONE = ZoneInfo("America/New_York")
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30
MINIMUM_SCENARIO_HISTORY = 20


class MovementRuntime:
    """Verified movement model, reference frames, and explanation methods."""

    def __init__(
        self,
        artifact_paths: ArtifactPaths,
        preloaded_bundle: dict[str, Any],
    ) -> None:
        """Initialize from a bundle loaded before pandas entered the worker.

        The worker loads and validates the owner-controlled joblib artifact first.
        This constructor then receives the validated dictionary and reads only the
        verified CSV evidence required for prediction and intelligence features.
        """

        self.paths = artifact_paths
        self.bundle = preloaded_bundle
        self.pipeline = self.bundle["pipeline"]
        self.numeric_features = list(self.bundle["numeric_features"])
        self.categorical_features = list(self.bundle["categorical_features"])
        self.text_features = list(self.bundle.get("text_features", []))
        self.features = (
            self.numeric_features + self.categorical_features + self.text_features
        )
        self.decision_policy = dict(self.bundle["decision_policy"])
        self.champion_name = str(self.bundle["champion_name"])
        self.model_table = pd.read_csv(artifact_paths.movement_model_table)
        self.test_predictions = pd.read_csv(
            artifact_paths.movement_test_predictions
        )
        self.news = pd.read_csv(artifact_paths.foundation_news)
        self.prices = pd.read_csv(artifact_paths.foundation_prices)
        self.global_driver_frame = pd.read_csv(artifact_paths.global_drivers)
        self.sentiment_phrase_frame = pd.read_csv(artifact_paths.sentiment_phrases)
        self._normalize_frames()
        self._validate_schemas()

    def _normalize_frames(self) -> None:
        """Normalize dates and identifiers without changing saved files."""

        for frame in (self.model_table, self.test_predictions, self.news, self.prices):
            if "ticker" in frame.columns:
                frame["ticker"] = frame["ticker"].astype(str).str.upper().str.strip()
        for frame in (self.model_table, self.test_predictions, self.news):
            if "target_session_date" in frame.columns:
                frame["target_session_date"] = pd.to_datetime(
                    frame["target_session_date"], errors="coerce"
                ).dt.normalize()
        self.prices["session_date"] = pd.to_datetime(
            self.prices["session_date"], errors="coerce"
        ).dt.normalize()
        self.news["published_at_utc"] = pd.to_datetime(
            self.news["published_at_utc"], utc=True, errors="coerce"
        )

    def _validate_schemas(self) -> None:
        """Require every field used for prediction, history, and scenarios."""

        model_required = set(self.features) | {
            "ticker",
            "target_session_date",
            "split",
        }
        news_required = {
            "article_id",
            "ticker",
            "target_session_date",
            "published_at_utc",
            "text",
            "source_url",
            "sentiment_label",
            "prob_bearish",
            "prob_neutral",
            "prob_bullish",
            "movement_label",
            "reaction_return",
        }
        price_required = {"ticker", "session_date"}
        missing_groups = {
            "model_table": sorted(model_required - set(self.model_table.columns)),
            "news": sorted(news_required - set(self.news.columns)),
            "prices": sorted(price_required - set(self.prices.columns)),
        }
        missing_groups = {
            key: value for key, value in missing_groups.items() if value
        }
        if missing_groups:
            raise ApiProblem(
                503,
                "movement_schema_changed",
                "Movement runtime validation failed.",
                "Verified movement and foundation tables",
                f"Required columns are missing: {missing_groups}",
                "Restore verified artifacts or rebuild the API against new contracts.",
            )
        if any(
            frame[column].isna().any()
            for frame, column in (
                (self.model_table, "target_session_date"),
                (self.news, "target_session_date"),
                (self.news, "published_at_utc"),
                (self.prices, "session_date"),
            )
        ):
            raise ApiProblem(
                503,
                "movement_dates_invalid",
                "Movement runtime validation failed.",
                "Verified movement and foundation tables",
                "At least one required date could not be parsed.",
                "Rerun independent movement verification and restore artifacts.",
            )

    def target_session(
        self,
        ticker: str,
        published_at: datetime,
    ) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Map publication time to the first verified session open after it."""

        publication_utc = pd.Timestamp(published_at).tz_convert("UTC")
        sessions = self.prices[self.prices["ticker"] == ticker][
            "session_date"
        ].drop_duplicates().sort_values()
        if sessions.empty:
            raise ApiProblem(
                422,
                "ticker_not_supported",
                "Movement prediction failed.",
                "Verified price-session evidence",
                f"No verified sessions exist for ticker {ticker}.",
                "Use one of the tickers recorded in the verified foundation.",
            )
        session_rows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        for session_date in sessions:
            local_open = pd.Timestamp(
                year=session_date.year,
                month=session_date.month,
                day=session_date.day,
                hour=MARKET_OPEN_HOUR,
                minute=MARKET_OPEN_MINUTE,
                tz=MARKET_TIMEZONE,
            )
            session_rows.append((session_date, local_open.tz_convert("UTC")))
        future = [row for row in session_rows if row[1] > publication_utc]
        if not future:
            latest = sessions.max().date()
            raise ApiProblem(
                422,
                "live_session_unavailable",
                "Movement prediction failed.",
                "Verified historical price-session evidence",
                f"No verified session occurs after the publication time. The latest "
                f"available session is {latest}.",
                "Use a publication time inside the verified historical window. Live "
                "forecasting requires a later provider-refresh phase.",
            )
        target_date, target_open = future[0]
        available = self.model_table[
            (self.model_table["ticker"] == ticker)
            & (self.model_table["target_session_date"] == target_date)
            & (self.model_table["split"] == "test")
        ]
        if len(available) != 1:
            raise ApiProblem(
                422,
                "historical_feature_row_unavailable",
                "Movement prediction failed.",
                "Verified movement model table",
                f"Ticker {ticker} and session {target_date.date()} do not have one "
                "verified historical-audit feature row.",
                (
                    "Choose a publication time that maps to the held-out "
                    "historical-audit block."
                ),
            )
        return target_date, target_open

    @staticmethod
    def _sentiment_event_features(
        text: str,
        sentiment: Mapping[str, Any],
        hours_to_open: float,
    ) -> dict[str, Any]:
        """Build one-event aggregate features using only pre-open information."""

        bearish = float(sentiment["prob_bearish"])
        neutral = float(sentiment["prob_neutral"])
        bullish = float(sentiment["prob_bullish"])
        values = np.asarray([bearish, neutral, bullish], dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ApiProblem(
                503,
                "sentiment_probabilities_invalid",
                "Movement feature preparation failed.",
                "DistilBERT sentiment result",
                "The sentiment probabilities are invalid.",
                "Run the FastAPI verifier and inspect the sentiment worker.",
            )
        if not np.isclose(values.sum(), 1.0, atol=1e-6):
            raise ApiProblem(
                503,
                "sentiment_probability_total_invalid",
                "Movement feature preparation failed.",
                "DistilBERT sentiment result",
                "The sentiment probabilities do not total one.",
                "Run the FastAPI verifier and inspect the sentiment worker.",
            )
        clipped = np.clip(values, 1e-12, 1.0)
        entropy = float(-(clipped * np.log(clipped)).sum() / np.log(3.0))
        label = str(sentiment["label"])
        shares = {name: float(label == name) for name in SENTIMENT_ORDER}
        text_length = len(text)
        return {
            "article_count": 1,
            "unique_source_count": 1,
            "text_length_mean": float(text_length),
            "text_length_max": int(text_length),
            "event_text": text,
            "prob_bearish_mean": bearish,
            "prob_neutral_mean": neutral,
            "prob_bullish_mean": bullish,
            "net_sentiment_mean": bullish - bearish,
            "net_sentiment_std": 0.0,
            "sentiment_entropy_mean": entropy,
            "sentiment_confidence_mean": float(values.max()),
            "sentiment_confidence_max": float(values.max()),
            "sentiment_probability_spread_mean": float(values.max() - values.min()),
            "bearish_event_share": shares["Bearish"],
            "neutral_event_share": shares["Neutral"],
            "bullish_event_share": shares["Bullish"],
            "hours_to_open_min": hours_to_open,
            "hours_to_open_mean": hours_to_open,
            "hours_to_open_max": hours_to_open,
            "article_count_log1p": math.log1p(1),
            "is_multi_event_day": 0,
        }

    def build_feature_frame(
        self,
        text: str,
        ticker: str,
        published_at: datetime,
        sentiment: Mapping[str, Any],
    ) -> tuple[pd.DataFrame, pd.Timestamp]:
        """Create one approved feature row without target-price information."""

        target_date, target_open = self.target_session(ticker, published_at)
        publication_utc = pd.Timestamp(published_at).tz_convert("UTC")
        hours_to_open = float((target_open - publication_utc).total_seconds() / 3600)
        if hours_to_open <= 0:
            raise ApiProblem(
                422,
                "publication_not_before_open",
                "Movement feature preparation failed.",
                "Market-session mapping",
                (
                    "The publication time is not strictly before the selected "
                    "session open."
                ),
                "Provide the original timezone-aware publication time and retry.",
            )
        source_rows = self.model_table[
            (self.model_table["ticker"] == ticker)
            & (self.model_table["target_session_date"] == target_date)
            & (self.model_table["split"] == "test")
        ]
        if len(source_rows) != 1:
            raise ApiProblem(
                422,
                "historical_feature_row_unavailable",
                "Movement feature preparation failed.",
                "Verified historical-audit feature row",
                "The selected ticker-session does not map to exactly one test row.",
                "Choose another verified historical-audit session.",
            )
        source_row = source_rows.iloc[0]
        row = {feature: source_row[feature] for feature in self.features}
        row.update(
            {
                key: value
                for key, value in self._sentiment_event_features(
                    text,
                    sentiment,
                    hours_to_open,
                ).items()
                if key in self.features
            }
        )
        row["ticker"] = ticker
        row["event_text"] = text
        frame = pd.DataFrame([row], columns=self.features)
        if frame[self.features].isna().any().any():
            missing = frame.columns[frame.isna().any()].tolist()
            raise ApiProblem(
                422,
                "movement_features_missing",
                "Movement feature preparation failed.",
                "Historical reference feature row",
                f"Required pre-session features are missing: {missing}",
                (
                    "Choose another verified historical-audit session or "
                    "refresh the foundation."
                ),
            )
        return frame, target_date

    def _aligned_raw_probabilities(self, frame: pd.DataFrame) -> np.ndarray:
        """Align model probabilities to Down, Flat, Up."""

        raw = np.asarray(self.pipeline.predict_proba(frame[self.features]), dtype=float)
        class_names = [
            str(value) for value in self.pipeline.named_steps["classifier"].classes_
        ]
        if set(class_names) != set(LABEL_ORDER):
            raise ApiProblem(
                503,
                "movement_classifier_classes_changed",
                "Movement prediction failed.",
                "Loaded movement classifier",
                f"Classifier classes changed: {class_names}",
                "Restore the verified movement model artifact.",
            )
        aligned = raw[:, [class_names.index(label) for label in LABEL_ORDER]]
        if aligned.shape != (len(frame), 3) or not np.isfinite(aligned).all():
            raise ApiProblem(
                503,
                "movement_probability_shape_invalid",
                "Movement prediction failed.",
                "Loaded movement classifier",
                "The probability matrix is invalid.",
                "Restore the verified movement model artifact.",
            )
        return aligned

    def _apply_policy(self, raw_probabilities: np.ndarray, ticker: str) -> np.ndarray:
        """Apply the frozen validation-only logit offsets exactly once."""

        global_offsets = self.decision_policy.get("global_logit_offsets")
        ticker_offsets = self.decision_policy.get("ticker_logit_offsets", {})
        if not isinstance(global_offsets, Mapping) or not isinstance(
            ticker_offsets, Mapping
        ):
            raise ApiProblem(
                503,
                "movement_decision_policy_invalid",
                "Movement prediction failed.",
                "Saved movement decision policy",
                "The decision policy is incomplete.",
                "Restore the verified movement model artifact.",
            )
        offsets = np.asarray(
            [float(global_offsets[label]) for label in LABEL_ORDER], dtype=float
        )
        local = ticker_offsets.get(ticker, {})
        if local:
            offsets += np.asarray(
                [float(local.get(label, 0.0)) for label in LABEL_ORDER],
                dtype=float,
            )
        scores = np.log(np.clip(raw_probabilities[0], 1e-12, 1.0)) + offsets
        scores -= scores.max()
        adjusted = np.exp(scores)
        adjusted /= adjusted.sum()
        if not np.isclose(adjusted.sum(), 1.0, atol=1e-6):
            raise ApiProblem(
                503,
                "movement_probability_total_invalid",
                "Movement prediction failed.",
                "Saved movement decision policy",
                "Adjusted probabilities do not total one.",
                "Restore the verified movement model artifact.",
            )
        return adjusted

    def predict_from_frame(self, frame: pd.DataFrame, ticker: str) -> dict[str, Any]:
        """Predict one movement result from an approved feature frame."""

        adjusted = self._apply_policy(self._aligned_raw_probabilities(frame), ticker)
        position = int(adjusted.argmax())
        return {
            "direction": LABEL_ORDER[position],
            "confidence": float(adjusted[position]),
            "prob_down": float(adjusted[0]),
            "prob_flat": float(adjusted[1]),
            "prob_up": float(adjusted[2]),
        }

    def predict(
        self,
        text: str,
        ticker: str,
        published_at: datetime,
        sentiment: Mapping[str, Any],
    ) -> tuple[dict[str, Any], pd.DataFrame, pd.Timestamp]:
        """Build one safe row and return its frozen-policy prediction."""

        frame, target_date = self.build_feature_frame(
            text,
            ticker,
            published_at,
            sentiment,
        )
        return self.predict_from_frame(frame, ticker), frame, target_date

    def local_drivers(
        self,
        frame: pd.DataFrame,
        predicted_class: str,
        top_n: int,
    ) -> list[dict[str, Any]]:
        """Measure one-feature probability sensitivity against reference values."""

        reference = self.model_table[
            self.model_table["split"].isin(["train", "validation"])
        ]
        if reference.empty:
            raise ApiProblem(
                503,
                "explanation_reference_missing",
                "Local explanation failed.",
                "Movement model table",
                "The train-validation reference frame is empty.",
                "Restore the verified movement model table.",
            )
        class_names = [
            str(value) for value in self.pipeline.named_steps["classifier"].classes_
        ]
        class_index = class_names.index(predicted_class)
        baseline = float(self.pipeline.predict_proba(frame)[0, class_index])
        rows: list[dict[str, Any]] = []
        for feature in self.features:
            perturbed = frame.copy()
            if feature in self.numeric_features:
                replacement = float(
                    pd.to_numeric(reference[feature], errors="coerce").median()
                )
            elif feature in self.categorical_features:
                modes = reference[feature].dropna().astype(str).mode()
                if modes.empty:
                    continue
                replacement = str(sorted(modes.tolist())[0])
            else:
                replacement = ""
            perturbed.loc[0, feature] = replacement
            changed = float(
                self.pipeline.predict_proba(perturbed)[0, class_index]
            )
            effect = baseline - changed
            rows.append(
                {
                    "feature": feature,
                    "probability_effect": float(effect),
                    "absolute_effect": float(abs(effect)),
                    "direction": (
                        "supports_prediction" if effect >= 0 else "opposes_prediction"
                    ),
                    "method": "single_feature_reference_perturbation",
                    "interpretation": (
                        "Probability sensitivity against a train-validation reference; "
                        "not causal influence."
                    ),
                }
            )
        rows.sort(key=lambda item: (-item["absolute_effect"], item["feature"]))
        for rank, row in enumerate(rows[:top_n], start=1):
            row["rank"] = rank
        return rows[:top_n]

    def global_drivers(self, top_n: int) -> list[dict[str, Any]]:
        """Return ranked persisted global sensitivity evidence."""

        required = {"feature", "importance", "method"}
        if required - set(self.global_driver_frame.columns):
            raise ApiProblem(
                503,
                "global_driver_schema_changed",
                "Global explanation failed.",
                "Persisted movement global drivers",
                "Required global-driver columns are missing.",
                "Restore the verified global-driver artifact.",
            )
        ordered = self.global_driver_frame.copy()
        ordered["importance"] = pd.to_numeric(
            ordered["importance"], errors="coerce"
        )
        ordered = ordered.dropna(subset=["importance"]).sort_values(
            ["importance", "feature"], ascending=[False, True]
        ).head(top_n)
        return [
            {
                "rank": rank,
                "feature": str(row.feature),
                "importance": float(row.importance),
                "method": str(row.method),
                "interpretation": (
                    "Higher values indicate model sensitivity, not causal influence."
                ),
            }
            for rank, row in enumerate(ordered.itertuples(index=False), start=1)
        ]

    def verify_saved_prediction(self) -> dict[str, Any]:
        """Recompute one saved historical-audit row from the loaded champion."""

        if self.test_predictions.empty:
            raise ApiProblem(
                503,
                "saved_predictions_empty",
                "Movement verification failed.",
                "Saved test predictions",
                "No saved prediction row is available.",
                "Restore the verified test-prediction artifact.",
            )
        saved = self.test_predictions.iloc[0]
        frame = self.model_table[
            (self.model_table["ticker"] == saved["ticker"])
            & (
                self.model_table["target_session_date"]
                == saved["target_session_date"]
            )
            & (self.model_table["split"] == "test")
        ]
        if len(frame) != 1:
            raise ApiProblem(
                503,
                "saved_prediction_feature_missing",
                "Movement verification failed.",
                "Movement model table",
                "The saved test prediction does not map to one feature row.",
                "Restore the verified movement artifacts.",
            )
        prediction = self.predict_from_frame(frame[self.features], str(saved["ticker"]))
        expected = np.asarray(
            [saved["prob_down"], saved["prob_flat"], saved["prob_up"]],
            dtype=float,
        )
        actual = np.asarray(
            [prediction["prob_down"], prediction["prob_flat"], prediction["prob_up"]],
            dtype=float,
        )
        if not np.allclose(expected, actual, atol=1e-10):
            raise ApiProblem(
                503,
                "saved_prediction_changed",
                "Movement verification failed.",
                "Reloaded champion prediction",
                "The recomputed probabilities differ from saved evidence.",
                "Stop API startup and rerun independent movement verification.",
            )
        return {
            "ticker": str(saved["ticker"]),
            "target_session_date": str(saved["target_session_date"].date()),
            "direction": prediction["direction"],
        }
