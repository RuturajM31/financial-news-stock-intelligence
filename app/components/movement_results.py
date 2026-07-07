"""Validate and render one Down, Flat, or Up movement result.

Purpose
-------
FastAPI owns prediction and schema validation. This module repeats the small
checks the page depends on, then turns the verified response into plain-language
cards and conclusions without exposing raw internal values.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from html import escape
from typing import Any, Mapping, Sequence

from app.components.chart_explanations import (
    ChartExplanation,
    render_chart_explanation,
)


ALLOWED_DIRECTIONS = ("Down", "Flat", "Up")


def _require_mapping(value: Any, location: str) -> Mapping[str, Any]:
    """Return a mapping or raise a clear response-contract error."""

    if not isinstance(value, Mapping):
        raise ValueError(f"{location} must be a JSON object.")
    return value


def _require_text(mapping: Mapping[str, Any], key: str, location: str) -> str:
    """Read one required non-empty text field."""

    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location}.{key} must be non-empty text.")
    return value.strip()




def _require_iso_date(
    mapping: Mapping[str, Any],
    key: str,
    location: str,
) -> str:
    """Read one ISO calendar date and reject ambiguous display text."""

    value = _require_text(mapping, key, location)
    try:
        date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{location}.{key} must be an ISO calendar date.") from error
    return value


def _require_probability(
    mapping: Mapping[str, Any],
    key: str,
    location: str,
) -> float:
    """Read one probability from zero to one while rejecting booleans."""

    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location}.{key} must be a number.")
    probability = float(value)
    if probability < 0.0 or probability > 1.0:
        raise ValueError(f"{location}.{key} must be between 0 and 1.")
    return probability


def _parse_text_list(value: Any, location: str) -> tuple[str, ...]:
    """Validate a list of safe warning strings."""

    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{location} must be a list of text values.")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{location} contains an invalid value.")
        result.append(item.strip())
    return tuple(result)


@dataclass(frozen=True)
class MovementView:
    """Store the checked movement fields used by the Streamlit page."""

    ticker: str
    target_session_date: str
    direction: str
    confidence: float
    down: float
    flat: float
    up: float
    champion_model: str
    research_mode: str
    warnings: tuple[str, ...]
    disclaimer: str

    @property
    def probabilities(self) -> dict[str, float]:
        """Return a new direction-to-probability mapping."""

        return {"Down": self.down, "Flat": self.flat, "Up": self.up}

    def as_summary(self) -> dict[str, Any]:
        """Return safe values that may be stored in current session history."""

        return {
            "ticker": self.ticker,
            "target_session_date": self.target_session_date,
            "direction": self.direction,
            "confidence_percent": round(self.confidence * 100, 2),
            "champion_model": self.champion_model,
        }


def parse_movement_response(payload: Mapping[str, Any]) -> MovementView:
    """Validate the exact movement fields required by the UI."""

    mapping = _require_mapping(payload, "movement response")
    if mapping.get("status") != "PASSED":
        raise ValueError("movement response.status must be PASSED.")
    direction = _require_text(mapping, "direction", "movement response")
    if direction not in ALLOWED_DIRECTIONS:
        raise ValueError("movement response.direction is unsupported.")

    probabilities = _require_mapping(
        mapping.get("probabilities"),
        "movement response.probabilities",
    )
    down = _require_probability(probabilities, "down", "movement response")
    flat = _require_probability(probabilities, "flat", "movement response")
    up = _require_probability(probabilities, "up", "movement response")
    if abs((down + flat + up) - 1.0) > 1e-6:
        raise ValueError("movement probabilities must total 1.0.")

    confidence = _require_probability(mapping, "confidence", "movement response")
    selected = {"Down": down, "Flat": flat, "Up": up}[direction]
    if abs(confidence - selected) > 1e-6:
        raise ValueError("movement confidence must match the selected direction.")

    return MovementView(
        ticker=_require_text(mapping, "ticker", "movement response"),
        target_session_date=_require_iso_date(
            mapping,
            "target_session_date",
            "movement response",
        ),
        direction=direction,
        confidence=confidence,
        down=down,
        flat=flat,
        up=up,
        champion_model=_require_text(
            mapping,
            "champion_model",
            "movement response",
        ),
        research_mode=_require_text(mapping, "research_mode", "movement response"),
        warnings=_parse_text_list(
            mapping.get("warnings", []),
            "movement response.warnings",
        ),
        disclaimer=_require_text(mapping, "disclaimer", "movement response"),
    )


def build_movement_conclusion(view: MovementView) -> ChartExplanation:
    """Explain the leading direction, margin, and uncertainty in simple words."""

    order = {name: index for index, name in enumerate(ALLOWED_DIRECTIONS)}
    ranked = sorted(
        view.probabilities.items(),
        key=lambda item: (-item[1], order[item[0]]),
    )
    first_name, first_value = ranked[0]
    second_name, second_value = ranked[1]
    margin = first_value - second_value

    if margin < 0.08:
        conclusion = (
            f"The result leans {first_name}, but it is close to {second_name}. "
            "Treat this forecast as uncertain."
        )
        uncertainty = (
            f"The lead is only {margin * 100:.1f} percentage points. Small new "
            "information could change the result."
        )
    elif first_value < 0.60:
        conclusion = (
            f"The model currently prefers {first_name}, with moderate confidence."
        )
        uncertainty = (
            "The result has a clear leader, but the other outcomes still hold "
            "meaningful probability."
        )
    else:
        conclusion = f"The model gives its strongest support to {first_name}."
        uncertainty = (
            "A strong model result is not a guaranteed market move. Unexpected "
            "news and market conditions can still change the outcome."
        )

    return ChartExplanation(
        what_it_shows=(
            "The chart divides the model result between Down, Flat, and Up for "
            "the mapped market session."
        ),
        why_it_matters=(
            "The gap between the highest and second-highest values shows whether "
            "the model has a clear preference or a close decision."
        ),
        conclusion=conclusion,
        uncertainty=uncertainty,
    )


def render_movement_hero(st: Any, view: MovementView) -> None:
    """Render the selected result, confidence, target date, and model name."""

    tone = {"Down": "down", "Flat": "flat", "Up": "up"}[view.direction]
    st.markdown(
        f"""
        <section class="rm-forecast-hero rm-forecast-{tone}">
          <div>
            <p class="rm-panel-kicker">NEXT VERIFIED SESSION</p>
            <h2>{escape(view.ticker)} · {escape(view.direction)}</h2>
            <p class="rm-forecast-confidence">{view.confidence * 100:.1f}% model support</p>
          </div>
          <div class="rm-forecast-meta">
            <span>Target session</span>
            <strong>{escape(view.target_session_date)}</strong>
            <span>Movement model</span>
            <strong>{escape(view.champion_model)}</strong>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_movement_warnings(st: Any, view: MovementView) -> None:
    """Render backend warnings and the mandatory research disclaimer."""

    for warning in view.warnings:
        st.warning(warning)
    st.caption(view.disclaimer)




def _bar(label: str, value: float, css_class: str) -> str:
    """Return one accessible probability bar using a bounded percentage."""

    percent = max(0.0, min(100.0, value * 100))
    return f"""
    <div class="rm-probability-row">
      <div class="rm-probability-label">
        <span>{escape(label)}</span><strong>{percent:.1f}%</strong>
      </div>
      <div class="rm-probability-track" role="img"
           aria-label="{escape(label)} probability {percent:.1f} percent">
        <span class="rm-probability-fill {css_class}" style="width:{percent:.2f}%"></span>
      </div>
    </div>
    """


def render_probability_visual(st: Any, view: MovementView) -> None:
    """Render probability ribbon, exact bars, and the lead over second place."""

    probabilities = view.probabilities
    ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    lead = (ranked[0][1] - ranked[1][1]) * 100
    st.markdown(
        f"""
        <section class="rm-panel rm-forecast-probabilities">
          <div class="rm-forecast-chart-header">
            <div>
              <p class="rm-panel-kicker">FORECAST BALANCE</p>
              <h3>Down, Flat, and Up probability</h3>
            </div>
            <div class="rm-margin-pill">Lead over second place · {lead:.1f} pts</div>
          </div>
          <div class="rm-probability-ribbon" role="img"
               aria-label="Down {probabilities['Down'] * 100:.1f} percent, Flat
               {probabilities['Flat'] * 100:.1f} percent, Up
               {probabilities['Up'] * 100:.1f} percent">
            <span class="rm-ribbon-down" style="width:{probabilities['Down'] * 100:.2f}%"></span>
            <span class="rm-ribbon-flat" style="width:{probabilities['Flat'] * 100:.2f}%"></span>
            <span class="rm-ribbon-up" style="width:{probabilities['Up'] * 100:.2f}%"></span>
          </div>
          <div class="rm-probability-bars">
            {_bar('Down', probabilities['Down'], 'rm-fill-down')}
            {_bar('Flat', probabilities['Flat'], 'rm-fill-flat')}
            {_bar('Up', probabilities['Up'], 'rm-fill-up')}
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_market_session_timeline(
    st: Any,
    *,
    published_at: str,
    target_session_date: str,
) -> None:
    """Explain how publication time maps to the forecast session."""

    try:
        publication = datetime.fromisoformat(published_at)
        publication_label = publication.strftime("%d %b %Y · %H:%M %Z")
    except ValueError:
        publication_label = published_at
    st.markdown(
        f"""
        <section class="rm-panel rm-session-timeline">
          <p class="rm-panel-kicker">MARKET SESSION MAP</p>
          <h3>From publication time to forecast session</h3>
          <div class="rm-timeline-track" aria-label="Market session timeline">
            <div class="rm-timeline-step rm-timeline-complete">
              <span>1</span><strong>News published</strong>
              <small>{escape(publication_label)}</small>
            </div>
            <div class="rm-timeline-line"></div>
            <div class="rm-timeline-step rm-timeline-complete">
              <span>2</span><strong>Session rules checked</strong>
              <small>Timezone and market calendar applied</small>
            </div>
            <div class="rm-timeline-line"></div>
            <div class="rm-timeline-step rm-timeline-target">
              <span>3</span><strong>Forecast target</strong>
              <small>{escape(target_session_date)}</small>
            </div>
          </div>
          <p class="rm-chart-note">
            The publication time matters because news before and after the market
            closes can belong to different target sessions.
          </p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_movement_result(st: Any, view: MovementView) -> None:
    """Render all verified movement result sections in the correct order."""

    render_movement_hero(st, view)
    render_probability_visual(st, view)
    render_chart_explanation(st, build_movement_conclusion(view))
    render_movement_warnings(st, view)
