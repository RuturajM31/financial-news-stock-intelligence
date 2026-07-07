"""Create historical evidence summaries from verified earlier-event matches."""

from __future__ import annotations

from collections import Counter
from html import escape
from statistics import median
from typing import Any, Sequence

from app.components.chart_explanations import (
    ChartExplanation,
    render_chart_explanation,
)
from app.components.historical_event_cards import HistoricalMatchView


def _scale_return(value: float, minimum: float, maximum: float) -> float:
    """Map a return into a zero-to-one range without dividing by zero."""

    if maximum == minimum:
        return 0.5
    return (value - minimum) / (maximum - minimum)


def render_return_range(
    st: Any,
    matches: Sequence[HistoricalMatchView],
) -> None:
    """Render the observed return range and explain its practical meaning."""

    if not matches:
        return
    returns = [match.reaction_return_percent for match in matches]
    minimum = min(returns)
    maximum = max(returns)
    middle = median(returns)
    marker = _scale_return(middle, minimum, maximum) * 100
    st.markdown(
        f"""
        <section class="rm-panel rm-return-range">
          <p class="rm-panel-kicker">HISTORICAL REACTION RANGE</p>
          <h3>What happened after the matched events</h3>
          <div class="rm-range-labels">
            <span>{minimum:+.2f}%</span>
            <strong>Middle result · {middle:+.2f}%</strong>
            <span>{maximum:+.2f}%</span>
          </div>
          <div class="rm-range-track" role="img"
               aria-label="Historical return range from {minimum:+.2f} to
               {maximum:+.2f} percent with middle result {middle:+.2f} percent">
            <span class="rm-range-gradient"></span>
            <span class="rm-range-marker" style="left:{marker:.2f}%"></span>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    render_chart_explanation(
        st,
        ChartExplanation(
            what_it_shows=(
                "The line shows the lowest, middle, and highest next-session "
                "returns from the matched earlier events."
            ),
            why_it_matters=(
                "A range is more honest than one average because similar events "
                "did not all produce the same market reaction."
            ),
            conclusion=(
                f"The matched events ranged from {minimum:+.2f}% to "
                f"{maximum:+.2f}%, with a middle result of {middle:+.2f}%."
            ),
            uncertainty=(
                "These are earlier observations, not a promise that the current "
                "event will stay inside this range."
            ),
        ),
    )


def render_outcome_balance(
    st: Any,
    matches: Sequence[HistoricalMatchView],
) -> None:
    """Show how often earlier matches ended Down, Flat, or Up."""

    if not matches:
        return
    counts = Counter(match.movement_label for match in matches)
    total = len(matches)
    rows = []
    for label, css_class in (
        ("Down", "rm-fill-down"),
        ("Flat", "rm-fill-flat"),
        ("Up", "rm-fill-up"),
    ):
        count = counts.get(label, 0)
        percent = count / total * 100
        rows.append(
            f"""
            <div class="rm-probability-row">
              <div class="rm-probability-label">
                <span>{label}</span><strong>{count} of {total}</strong>
              </div>
              <div class="rm-probability-track">
                <span class="rm-probability-fill {css_class}"
                      style="width:{percent:.2f}%"></span>
              </div>
            </div>
            """
        )
    most_common_label, most_common_count = max(
        counts.items(),
        key=lambda item: (item[1], item[0]),
    )
    st.markdown(
        f"""
        <section class="rm-panel">
          <p class="rm-panel-kicker">EARLIER OUTCOME BALANCE</p>
          <h3>How the matched sessions ended</h3>
          {''.join(rows)}
        </section>
        """,
        unsafe_allow_html=True,
    )
    render_chart_explanation(
        st,
        ChartExplanation(
            what_it_shows=(
                "The bars count how many matched earlier events ended Down, "
                "Flat, or Up in the next session."
            ),
            why_it_matters=(
                "The count shows whether the earlier evidence points mainly in "
                "one direction or remains mixed."
            ),
            conclusion=(
                f"{escape(most_common_label)} was the most common earlier result "
                f"with {most_common_count} of {total} matches."
            ),
            uncertainty=(
                "A small evidence count or a close split should be treated as weak "
                "support rather than a strong conclusion."
            ),
        ),
    )


def render_phrase_chips(st: Any, phrases: Sequence[str]) -> None:
    """Render only phrases returned by the verified historical endpoint."""

    if not phrases:
        st.info("No verified reference-period phrases were returned for this event.")
        return
    chips = "".join(
        f'<span class="rm-phrase-chip">{escape(phrase)}</span>' for phrase in phrases
    )
    st.markdown(
        f"""
        <section class="rm-panel">
          <p class="rm-panel-kicker">VERIFIED REFERENCE PHRASES</p>
          <h3>Wording linked to the earlier evidence</h3>
          <div class="rm-phrase-chip-grid">{chips}</div>
          <p class="rm-chart-note">
            These phrases come from the approved earlier-only reference scope.
            They are not invented by the interface.
          </p>
        </section>
        """,
        unsafe_allow_html=True,
    )
