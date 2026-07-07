"""Validate and present FastAPI sentiment results in simple language.

Purpose
-------
The FastAPI sentiment routes return either one result or a batch of results for
an uploaded file. This module checks every value before it reaches the page and
turns the probabilities into clear cards, bars, and a written conclusion.

The module does not calculate sentiment. It displays only values returned by the
verified DistilBERT service.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from app.components.chart_explanations import (
    ChartExplanation,
    render_chart_explanation,
)
from app.components.result_cards import render_result_card


_ALLOWED_LABELS = ("Bearish", "Neutral", "Bullish")
_LABEL_TO_KEY = {
    "Bearish": "bearish",
    "Neutral": "neutral",
    "Bullish": "bullish",
}
_LABEL_TO_TONE = {
    "Bearish": "negative",
    "Neutral": "neutral",
    "Bullish": "positive",
}


@dataclass(frozen=True)
class SentimentView:
    """Store one checked sentiment result for display and safe session use."""

    label: str
    confidence: float
    bearish: float
    neutral: float
    bullish: float
    deployment_model: str
    source_type: str
    warnings: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe value that contains no submitted article text."""

        return {
            "status": "PASSED",
            "label": self.label,
            "confidence": self.confidence,
            "probabilities": {
                "bearish": self.bearish,
                "neutral": self.neutral,
                "bullish": self.bullish,
            },
            "deployment_model": self.deployment_model,
            "source_type": self.source_type,
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SentimentView":
        """Rebuild a view from one previously checked session-safe mapping."""

        return _parse_single_sentiment(value, location="saved sentiment result")

    @property
    def probabilities(self) -> dict[str, float]:
        """Return the three display probabilities in a stable order."""

        return {
            "Bearish": self.bearish,
            "Neutral": self.neutral,
            "Bullish": self.bullish,
        }


def _require_mapping(value: Any, location: str) -> Mapping[str, Any]:
    """Return one mapping or reject an unexpected response shape."""

    if not isinstance(value, Mapping):
        raise ValueError(f"{location} must be a JSON object.")
    return value


def _require_text(mapping: Mapping[str, Any], key: str, location: str) -> str:
    """Read one required non-empty text field."""

    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location}.{key} must be non-empty text.")
    return value.strip()


def _require_probability(
    mapping: Mapping[str, Any],
    key: str,
    location: str,
) -> float:
    """Read one numeric probability and require a value from zero to one."""

    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location}.{key} must be a number.")
    result = float(value)
    if result < 0.0 or result > 1.0:
        raise ValueError(f"{location}.{key} must be between 0 and 1.")
    return result


def _parse_warnings(value: Any, location: str) -> tuple[str, ...]:
    """Return short warning text while rejecting an unexpected data type."""

    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{location}.warnings must be a list of text values.")
    warnings: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{location}.warnings contains an invalid value.")
        warnings.append(item.strip())
    return tuple(warnings)


def _parse_single_sentiment(
    value: Mapping[str, Any],
    *,
    location: str,
) -> SentimentView:
    """Validate one single-result contract returned by FastAPI."""

    mapping = _require_mapping(value, location)
    status = mapping.get("status", "PASSED")
    if status != "PASSED":
        raise ValueError(f"{location}.status must be PASSED.")

    label = _require_text(mapping, "label", location)
    if label not in _ALLOWED_LABELS:
        raise ValueError(f"{location}.label contains an unsupported value.")

    probabilities = _require_mapping(
        mapping.get("probabilities"),
        f"{location}.probabilities",
    )
    bearish = _require_probability(probabilities, "bearish", location)
    neutral = _require_probability(probabilities, "neutral", location)
    bullish = _require_probability(probabilities, "bullish", location)
    total = bearish + neutral + bullish
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"{location} probabilities must total 1.0.")

    confidence = _require_probability(mapping, "confidence", location)
    selected_probability = {
        "Bearish": bearish,
        "Neutral": neutral,
        "Bullish": bullish,
    }[label]
    if abs(confidence - selected_probability) > 1e-6:
        raise ValueError(
            f"{location}.confidence must match the selected label probability."
        )

    deployment_model = _require_text(mapping, "deployment_model", location)
    source_type = _require_text(mapping, "source_type", location)
    warnings = _parse_warnings(mapping.get("warnings", []), location)
    return SentimentView(
        label=label,
        confidence=confidence,
        bearish=bearish,
        neutral=neutral,
        bullish=bullish,
        deployment_model=deployment_model,
        source_type=source_type,
        warnings=warnings,
    )


def parse_sentiment_response(payload: Mapping[str, Any]) -> tuple[SentimentView, ...]:
    """Validate one text/URL result or all results in one file response."""

    mapping = _require_mapping(payload, "sentiment response")
    if mapping.get("status") != "PASSED":
        raise ValueError("sentiment response.status must be PASSED.")

    if "results" not in mapping:
        return (_parse_single_sentiment(mapping, location="sentiment response"),)

    results_raw = mapping.get("results")
    if not isinstance(results_raw, Sequence) or isinstance(
        results_raw,
        (str, bytes),
    ):
        raise ValueError("sentiment response.results must be a list.")
    if not results_raw:
        raise ValueError("sentiment response.results must not be empty.")

    declared_count = mapping.get("result_count")
    if isinstance(declared_count, bool) or not isinstance(declared_count, int):
        raise ValueError("sentiment response.result_count must be a whole number.")
    if declared_count != len(results_raw):
        raise ValueError(
            "sentiment response.result_count does not match the returned results."
        )

    return tuple(
        _parse_single_sentiment(item, location=f"sentiment result {index}")
        for index, item in enumerate(results_raw, start=1)
    )


def _ranked_probabilities(view: SentimentView) -> list[tuple[str, float]]:
    """Return probabilities from highest to lowest with stable tie handling."""

    label_order = {label: index for index, label in enumerate(_ALLOWED_LABELS)}
    return sorted(
        view.probabilities.items(),
        key=lambda item: (-item[1], label_order[item[0]]),
    )


def build_sentiment_conclusion(view: SentimentView) -> ChartExplanation:
    """Explain the strongest result and its distance from the next result."""

    ranked = _ranked_probabilities(view)
    first_label, first_value = ranked[0]
    second_label, second_value = ranked[1]
    margin = first_value - second_value

    if margin < 0.10:
        conclusion = (
            f"The result leans {first_label.lower()}, but it is close to "
            f"{second_label.lower()}. Treat the result as uncertain."
        )
        uncertainty = (
            f"The leading result is only {margin * 100:.1f} percentage points "
            "above the next result."
        )
    elif first_value < 0.70:
        conclusion = (
            f"The text is mainly {first_label.lower()}, with moderate confidence."
        )
        uncertainty = (
            "The result is clear enough to describe the text, but it should not "
            "be treated as a market forecast."
        )
    else:
        conclusion = (
            f"The text is strongly {first_label.lower()} according to the live "
            "sentiment model."
        )
        uncertainty = (
            "Sentiment describes the submitted wording. It does not guarantee a "
            "share-price reaction."
        )

    return ChartExplanation(
        what_it_shows=(
            "The three bars show how the model divided its view between bearish, "
            "neutral, and bullish meaning."
        ),
        why_it_matters=(
            "The gap between the first and second result shows whether the wording "
            "has a clear tone or a mixed tone."
        ),
        conclusion=conclusion,
        uncertainty=uncertainty,
    )


def _format_probability(value: float) -> str:
    """Return one percentage with enough detail for close comparisons."""

    return f"{value * 100:.1f}%"


def render_sentiment_result(
    st: Any,
    views: Sequence[SentimentView],
) -> SentimentView:
    """Render one selected result and return the view currently on screen."""

    if not views:
        raise ValueError("At least one sentiment result is required.")

    selected_index = 0
    if len(views) > 1:
        options = list(range(len(views)))
        selected_index = st.selectbox(
            "Choose a file result",
            options,
            format_func=lambda index: (
                f"Result {index + 1} · {views[index].label} · "
                f"{_format_probability(views[index].confidence)}"
            ),
            key="rm_selected_file_sentiment_result",
        )
        st.caption(
            f"The file returned {len(views)} separate text results. "
            "Choose one to inspect."
        )

    view = views[selected_index]
    columns = st.columns(3, gap="medium")
    with columns[0]:
        render_result_card(
            st,
            "Sentiment",
            view.label,
            "The main tone found in the submitted wording.",
            tone=_LABEL_TO_TONE[view.label],
            eyebrow="LIVE RESULT",
        )
    with columns[1]:
        render_result_card(
            st,
            "Confidence",
            _format_probability(view.confidence),
            "The probability assigned to the selected sentiment label.",
            tone="info",
            eyebrow="MODEL VIEW",
        )
    with columns[2]:
        render_result_card(
            st,
            "Live model",
            view.deployment_model,
            "The approved sentiment model used by FastAPI.",
            tone="verified",
            eyebrow="VERIFIED SERVICE",
        )

    st.markdown("### Sentiment balance")
    for label in _ALLOWED_LABELS:
        value = view.probabilities[label]
        st.progress(value, text=f"{label}: {_format_probability(value)}")

    render_chart_explanation(st, build_sentiment_conclusion(view))
    for warning in view.warnings:
        st.warning(warning)
    return view
