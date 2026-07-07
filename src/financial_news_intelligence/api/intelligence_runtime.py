"""Historical retrieval, scenario analysis, and provenance services."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import math

from .artifacts import ArtifactPaths
from .errors import ApiProblem
from .movement_runtime import MINIMUM_SCENARIO_HISTORY, MovementRuntime


class IntelligenceRuntime:
    """Serve earlier-only intelligence from verified reference evidence."""

    def __init__(
        self,
        movement: MovementRuntime,
        artifact_paths: ArtifactPaths,
    ) -> None:
        self.movement = movement
        self.paths = artifact_paths
        try:
            self.provenance = json.loads(
                artifact_paths.provenance.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise ApiProblem(
                503,
                "provenance_load_failed",
                "Provenance loading failed.",
                str(artifact_paths.provenance),
                (
                    "The verified provenance file could not be read: "
                    f"{type(exc).__name__}."
                ),
                "Restore the verified provenance artifact.",
            ) from exc
        if self.provenance.get("status") != "provenance_verified":
            raise ApiProblem(
                503,
                "provenance_not_verified",
                "Provenance loading failed.",
                str(artifact_paths.provenance),
                "The provenance status is not provenance_verified.",
                "Restore the verified provenance artifact.",
            )

    def _reference_events(self, cutoff: pd.Timestamp | None = None) -> pd.DataFrame:
        """Return events whose ticker-session keys are train or validation only."""

        keys = self.movement.model_table[
            self.movement.model_table["split"].isin(["train", "validation"])
        ][["ticker", "target_session_date"]].drop_duplicates()
        events = self.movement.news.merge(
            keys,
            on=["ticker", "target_session_date"],
            how="inner",
            validate="many_to_one",
        )
        if cutoff is not None:
            events = events[events["target_session_date"] < cutoff]
        return events.sort_values(
            ["target_session_date", "ticker", "article_id"]
        ).reset_index(drop=True)

    def historical_matches(
        self,
        text: str,
        ticker: str,
        cutoff: pd.Timestamp,
        limit: int,
        minimum_similarity: float,
        sentiment_label: str,
    ) -> dict[str, Any]:
        """Return same-ticker reference events strictly earlier than the query."""

        candidates = self._reference_events(cutoff)
        candidates = candidates[candidates["ticker"] == ticker].copy()
        if len(candidates) < 3:
            raise ApiProblem(
                422,
                "historical_evidence_insufficient",
                "Historical intelligence failed.",
                "Train-validation same-ticker reference events",
                f"Only {len(candidates)} earlier events exist for ticker {ticker}.",
                "Choose another verified historical cutoff or supported ticker.",
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
            matrix = vectorizer.fit_transform(corpus + [text])
        except ValueError as exc:
            raise ApiProblem(
                422,
                "historical_vocabulary_unavailable",
                "Historical intelligence failed.",
                "TF-IDF historical similarity",
                f"A comparison vocabulary could not be built: {exc}",
                "Use longer financial-news text and retry.",
            ) from exc
        similarities = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
        ranked = np.argsort(-similarities)
        matches: list[dict[str, Any]] = []
        for index in ranked:
            score = float(similarities[int(index)])
            if score < minimum_similarity:
                continue
            row = candidates.iloc[int(index)]
            matches.append(
                {
                    "article_id": str(row["article_id"]),
                    "ticker": ticker,
                    "target_session_date": row["target_session_date"].date(),
                    "source_url": str(row["source_url"]),
                    "sentiment_label": str(row["sentiment_label"]),
                    "movement_label": str(row["movement_label"]),
                    "reaction_return_percent": float(row["reaction_return"]) * 100.0,
                    "similarity_score": score,
                }
            )
            if len(matches) >= limit:
                break
        if not matches:
            raise ApiProblem(
                422,
                "historical_similarity_below_threshold",
                "Historical intelligence failed.",
                "Same-ticker earlier-event retrieval",
                "No earlier event met the requested similarity threshold.",
                "Lower minimum_similarity or provide more specific filing text.",
            )

        phrase_frame = self.movement.sentiment_phrase_frame
        phrase_required = {"sentiment_label", "rank", "phrase"}
        phrases: list[str] = []
        if phrase_required.issubset(phrase_frame.columns):
            selected = phrase_frame[
                phrase_frame["sentiment_label"].astype(str) == sentiment_label
            ].copy()
            selected["rank"] = pd.to_numeric(selected["rank"], errors="coerce")
            phrases = (
                selected.dropna(subset=["rank"])
                .sort_values("rank")["phrase"]
                .astype(str)
                .head(10)
                .tolist()
            )
        return {"matches": matches, "important_phrases": phrases}


    @staticmethod
    def _calculate_outcome(
        request: Mapping[str, Any],
        historical_return_percent: float,
        scenario: str,
    ) -> dict[str, Any]:
        """Apply one historical return without inventing market assumptions.

        Formula:
        ``final stock value = share cost × (1 + return / 100)``.
        Entry fees reduce capital before purchase. Exit fees reduce final value.
        Optional tax applies only to a positive scenario gain.
        """

        investment_amount = float(request["investment_amount"])
        share_price = float(request["share_price"])
        entry_fee = float(request["entry_fee"])
        exit_fee = float(request["exit_fee"])
        tax_rate = request.get("tax_rate_percent")
        available_capital = investment_amount - entry_fee
        raw_shares = available_capital / share_price
        if bool(request["allow_fractional_shares"]):
            precision_factor = 10 ** int(request["share_precision"])
            shares = math.floor(raw_shares * precision_factor) / precision_factor
        else:
            shares = float(math.floor(raw_shares))
        if shares <= 0:
            raise ApiProblem(
                422,
                "investment_amount_insufficient",
                "Scenario analysis failed.",
                "Share purchase calculation",
                "Available capital cannot purchase one permitted share unit.",
                "Increase the investment amount, lower the share price, "
                "or allow fractional shares.",
            )
        share_cost = shares * share_price
        cash_balance = available_capital - share_cost
        if abs(cash_balance) < 1e-8:
            cash_balance = 0.0
        final_stock_value = share_cost * (1 + historical_return_percent / 100)
        gross_final_value = final_stock_value + cash_balance
        value_after_exit_fee = max(gross_final_value - exit_fee, 0.0)
        pre_tax_gain = value_after_exit_fee - investment_amount
        estimated_tax = 0.0
        if tax_rate is not None:
            estimated_tax = max(pre_tax_gain, 0.0) * (float(tax_rate) / 100)
        net_final_value = max(value_after_exit_fee - estimated_tax, 0.0)
        gain_loss = net_final_value - investment_amount
        return {
            "scenario": scenario,
            "historical_return_percent": historical_return_percent,
            "shares_purchased": shares,
            "cash_balance": cash_balance,
            "estimated_tax": estimated_tax,
            "net_final_value": net_final_value,
            "gain_loss": gain_loss,
            "gain_loss_percent": gain_loss / investment_amount * 100,
        }

    def scenario(
        self,
        ticker: str,
        cutoff: pd.Timestamp,
        probabilities: Mapping[str, float],
        request: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Build downside, base, and upside outcomes from earlier reactions."""

        earlier = self._reference_events(cutoff)
        same_ticker = earlier[earlier["ticker"] == ticker].copy()
        if len(same_ticker) < MINIMUM_SCENARIO_HISTORY:
            raise ApiProblem(
                422,
                "scenario_history_insufficient",
                "Scenario analysis failed.",
                "Same-ticker earlier reference reactions",
                f"Only {len(same_ticker)} events exist; at least "
                f"{MINIMUM_SCENARIO_HISTORY} are required.",
                "Choose another verified historical cutoff or supported ticker.",
            )
        class_medians: dict[str, float] = {}
        fallback_classes: list[str] = []
        for label in ("Down", "Flat", "Up"):
            values = pd.to_numeric(
                same_ticker.loc[
                    same_ticker["movement_label"] == label,
                    "reaction_return",
                ],
                errors="coerce",
            ).dropna()
            if values.empty:
                values = pd.to_numeric(
                    earlier.loc[
                        earlier["movement_label"] == label,
                        "reaction_return",
                    ],
                    errors="coerce",
                ).dropna()
                fallback_classes.append(label)
            if values.empty:
                raise ApiProblem(
                    422,
                    "scenario_class_evidence_missing",
                    "Scenario analysis failed.",
                    "Earlier movement-class reactions",
                    f"No reference reaction exists for class {label}.",
                    "Choose another verified cutoff or rebuild the foundation.",
                )
            class_medians[label] = float(values.median())
        base_return = sum(
            float(probabilities[label]) * class_medians[label]
            for label in ("Down", "Flat", "Up")
        )
        return_points = {
            "low": float(
                pd.to_numeric(same_ticker["reaction_return"]).quantile(0.10)
            ),
            "base": float(base_return),
            "high": float(
                pd.to_numeric(same_ticker["reaction_return"]).quantile(0.90)
            ),
        }
        outcomes: list[dict[str, Any]] = []
        for level, decimal_return in return_points.items():
            outcomes.append(
                self._calculate_outcome(
                    request=request,
                    historical_return_percent=decimal_return * 100.0,
                    scenario=level,
                )
            )
        return {
            "evidence_count": int(len(same_ticker)),
            "evidence_end_date": same_ticker["target_session_date"].max().date(),
            "class_median_fallbacks": fallback_classes,
            "outcomes": outcomes,
            "method": (
                "Earlier same-ticker 10th and 90th reaction quantiles plus "
                "probability-weighted movement-class medians."
            ),
        }
