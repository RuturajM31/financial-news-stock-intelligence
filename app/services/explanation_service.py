"""Validate explanation evidence and calculate bounded word influence.

Purpose
-------
FastAPI owns model inference and movement-driver evidence. This module checks the
small response fields used by Streamlit and provides an optional word-influence
method that calls the verified sentiment API repeatedly. It never loads a model.

Word-influence method
---------------------
The method removes one word at a time and measures how much the selected
sentiment probability changes. This is an occlusion check, not raw attention.
It is capped at twelve words so one action stays below the current request
budget and remains understandable to the user.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from app.components.movement_results import MovementView, parse_movement_response
from app.components.sentiment_results import SentimentView, parse_sentiment_response


MAXIMUM_INFLUENCE_WORDS = 12
MINIMUM_INFLUENCE_WORDS = 3
_WORD_PATTERN = re.compile(r"\b[\w'-]+\b", re.UNICODE)
_FEATURE_LABELS = {
    "net_sentiment_mean": "Average news sentiment",
    "net_sentiment_std": "Sentiment variation",
    "event_text": "News wording",
    "historical_return_mean": "Earlier return average",
    "historical_return_std": "Earlier return variation",
    "similarity_score": "Similarity to earlier events",
    "article_count": "Earlier event count",
}


def _require_mapping(value: Any, location: str) -> Mapping[str, Any]:
    """Return one JSON object or raise a clear contract error."""

    if not isinstance(value, Mapping):
        raise ValueError(f"{location} must be a JSON object.")
    return value


def _require_text(mapping: Mapping[str, Any], key: str, location: str) -> str:
    """Read one required non-empty text field."""

    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location}.{key} must be non-empty text.")
    return value.strip()


def _optional_finite_number(
    mapping: Mapping[str, Any],
    key: str,
    location: str,
) -> float | None:
    """Read one optional finite number while rejecting booleans and NaN."""

    value = mapping.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location}.{key} must be a number when provided.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{location}.{key} must be finite.")
    return result


def humanize_feature_name(value: str) -> str:
    """Return a plain label while preserving the verified source key elsewhere."""

    normalized = value.strip()
    if not normalized:
        raise ValueError("A driver feature name must not be empty.")
    if normalized in _FEATURE_LABELS:
        return _FEATURE_LABELS[normalized]
    return normalized.replace("_", " ").replace("-", " ").strip().title()


@dataclass(frozen=True)
class DriverView:
    """Store one checked global or local movement driver."""

    rank: int
    feature_key: str
    label: str
    importance: float | None
    probability_effect: float | None
    absolute_effect: float | None
    direction: str | None
    method: str
    interpretation: str

    @property
    def signed_effect(self) -> float:
        """Return the local probability effect or zero when it is unavailable."""

        return self.probability_effect if self.probability_effect is not None else 0.0

    @property
    def strength(self) -> float:
        """Return the strongest available non-negative driver measure."""

        candidates = (
            self.absolute_effect,
            abs(self.probability_effect) if self.probability_effect is not None else None,
            self.importance,
        )
        for candidate in candidates:
            if candidate is not None:
                return max(0.0, candidate)
        return 0.0


@dataclass(frozen=True)
class ExplainabilityView:
    """Store one verified movement prediction and its driver evidence."""

    prediction: MovementView
    global_drivers: tuple[DriverView, ...]
    local_drivers: tuple[DriverView, ...]
    reference_scope: str
    limitation: str


@dataclass(frozen=True)
class WordInfluenceItem:
    """Store one word occurrence and its measured effect on the selected label."""

    position: int
    word: str
    raw_effect: float
    influence_percent: float
    direction: str


@dataclass(frozen=True)
class WordInfluenceResult:
    """Store a transient word-influence result that must not enter session history."""

    label: str
    baseline_probability: float
    deployment_model: str
    items: tuple[WordInfluenceItem, ...]
    method: str = "leave_one_word_out_occlusion"

    @property
    def strongest_supporter(self) -> WordInfluenceItem | None:
        """Return the largest positive effect when one exists."""

        supporters = [item for item in self.items if item.raw_effect > 0.0]
        return max(supporters, key=lambda item: item.raw_effect, default=None)


def _parse_driver(value: Any, location: str) -> DriverView:
    """Validate one driver returned by the FastAPI explanation route."""

    mapping = _require_mapping(value, location)
    rank = mapping.get("rank")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        raise ValueError(f"{location}.rank must be a positive whole number.")
    feature = _require_text(mapping, "feature", location)
    direction = mapping.get("direction")
    if direction is not None and (
        not isinstance(direction, str) or not direction.strip()
    ):
        raise ValueError(f"{location}.direction must be text when provided.")
    driver = DriverView(
        rank=rank,
        feature_key=feature,
        label=humanize_feature_name(feature),
        importance=_optional_finite_number(mapping, "importance", location),
        probability_effect=_optional_finite_number(
            mapping,
            "probability_effect",
            location,
        ),
        absolute_effect=_optional_finite_number(
            mapping,
            "absolute_effect",
            location,
        ),
        direction=direction.strip() if isinstance(direction, str) else None,
        method=_require_text(mapping, "method", location),
        interpretation=_require_text(mapping, "interpretation", location),
    )
    if driver.importance is not None and driver.importance < 0.0:
        raise ValueError(f"{location}.importance must not be negative.")
    if driver.absolute_effect is not None and driver.absolute_effect < 0.0:
        raise ValueError(f"{location}.absolute_effect must not be negative.")
    return driver


def _parse_driver_list(value: Any, location: str) -> tuple[DriverView, ...]:
    """Validate a non-empty ordered driver list with unique ranks."""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{location} must be a list.")
    if not value:
        raise ValueError(f"{location} must not be empty.")
    drivers = tuple(
        _parse_driver(item, f"{location}[{index}]")
        for index, item in enumerate(value)
    )
    ranks = [driver.rank for driver in drivers]
    if len(ranks) != len(set(ranks)):
        raise ValueError(f"{location} contains duplicate ranks.")
    if ranks != sorted(ranks):
        raise ValueError(f"{location} must be ordered by rank.")
    return drivers


def parse_explainability_response(payload: Mapping[str, Any]) -> ExplainabilityView:
    """Validate the exact explanation fields required by the page."""

    mapping = _require_mapping(payload, "explainability response")
    if mapping.get("status") != "PASSED":
        raise ValueError("explainability response.status must be PASSED.")
    prediction = parse_movement_response(
        _require_mapping(mapping.get("prediction"), "explainability prediction")
    )
    return ExplainabilityView(
        prediction=prediction,
        global_drivers=_parse_driver_list(
            mapping.get("global_drivers"),
            "global drivers",
        ),
        local_drivers=_parse_driver_list(
            mapping.get("local_drivers"),
            "local drivers",
        ),
        reference_scope=_require_text(
            mapping,
            "reference_scope",
            "explainability response",
        ),
        limitation=_require_text(
            mapping,
            "limitation",
            "explainability response",
        ),
    )


def _word_matches(text: str) -> tuple[re.Match[str], ...]:
    """Return visible word matches while rejecting an unsafe request size."""

    if not isinstance(text, str):
        raise ValueError("Word-influence text must be text.")
    normalized = text.strip()
    matches = tuple(_WORD_PATTERN.finditer(normalized))
    if len(matches) < MINIMUM_INFLUENCE_WORDS:
        raise ValueError(
            "Use at least three words for the word-influence check."
        )
    if len(matches) > MAXIMUM_INFLUENCE_WORDS:
        raise ValueError(
            "Use no more than twelve words for the word-influence check."
        )
    return matches


def _probability_for_label(view: SentimentView, label: str) -> float:
    """Return the selected sentiment probability from one validated response."""

    return view.probabilities[label]


def _remove_word(text: str, match: re.Match[str]) -> str:
    """Remove one word occurrence and normalize only the surrounding spaces."""

    result = f"{text[:match.start()]} {text[match.end():]}"
    return " ".join(result.split())


def compute_word_influence(api_client: Any, text: str) -> WordInfluenceResult:
    """Measure the effect of removing each word through the verified API.

    The function performs one baseline request and one request for each word.
    Raw text and intermediate responses are held only for this call. The caller
    must not place the result or submitted text in persistent session history.
    """

    normalized = text.strip()
    matches = _word_matches(normalized)
    baseline_views = parse_sentiment_response(api_client.sentiment_text(normalized))
    if len(baseline_views) != 1:
        raise ValueError("Word influence requires one sentiment result.")
    baseline = baseline_views[0]
    baseline_probability = _probability_for_label(baseline, baseline.label)

    measured: list[tuple[int, str, float]] = []
    for position, match in enumerate(matches, start=1):
        changed_text = _remove_word(normalized, match)
        changed_views = parse_sentiment_response(
            api_client.sentiment_text(changed_text)
        )
        if len(changed_views) != 1:
            raise ValueError("A word-influence request returned multiple results.")
        changed_probability = _probability_for_label(
            changed_views[0],
            baseline.label,
        )
        measured.append(
            (position, match.group(0), baseline_probability - changed_probability)
        )

    total_magnitude = sum(abs(effect) for _, _, effect in measured)
    items: list[WordInfluenceItem] = []
    for position, word, effect in measured:
        percentage = (
            abs(effect) / total_magnitude * 100.0
            if total_magnitude > 0.0
            else 0.0
        )
        direction = "supports" if effect > 0 else "opposes" if effect < 0 else "neutral"
        items.append(
            WordInfluenceItem(
                position=position,
                word=word,
                raw_effect=effect,
                influence_percent=percentage,
                direction=direction,
            )
        )
    return WordInfluenceResult(
        label=baseline.label,
        baseline_probability=baseline_probability,
        deployment_model=baseline.deployment_model,
        items=tuple(items),
    )
