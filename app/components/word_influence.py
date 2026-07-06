"""Render verified word influence without presenting raw attention as importance."""

from __future__ import annotations

from html import escape
from typing import Any

from app.components.chart_explanations import (
    ChartExplanation,
    render_chart_explanation,
)
from app.services.explanation_service import WordInfluenceResult


def build_word_influence_html(result: WordInfluenceResult) -> str:
    """Return escaped word chips with text labels and measured percentages."""

    chips: list[str] = []
    for item in result.items:
        strength = max(0.08, min(1.0, item.influence_percent / 100.0))
        css_class = {
            "supports": "rm-word-supports",
            "opposes": "rm-word-opposes",
            "neutral": "rm-word-neutral",
        }[item.direction]
        chips.append(
            f'<span class="rm-word-chip {css_class}" '
            f'style="--rm-word-strength:{strength:.4f}" '
            f'aria-label="{escape(item.word)} '
            f'{item.influence_percent:.1f} percent '
            f'{escape(item.direction)}">{escape(item.word)}'
            f'<small>{item.influence_percent:.1f}%</small></span>'
        )
    return (
        '<section class="rm-panel rm-word-influence">'
        '<p class="rm-panel-kicker">VERIFIED WORD INFLUENCE</p>'
        '<h3>Which words changed the result?</h3>'
        '<div class="rm-word-cloud">'
        + "".join(chips)
        + '</div></section>'
    )


def build_word_influence_table(result: WordInfluenceResult) -> list[dict[str, Any]]:
    """Return a non-color table with exact effect and percentage values."""

    return [
        {
            "Position": item.position,
            "Word": item.word,
            "Share of measured influence": round(item.influence_percent, 2),
            "Effect on selected result (points)": round(item.raw_effect * 100.0, 4),
            "Direction": item.direction.title(),
        }
        for item in result.items
    ]


def render_word_influence_result(st: Any, result: WordInfluenceResult) -> None:
    """Render the word map, exact table, method, conclusion, and limitation."""

    st.markdown(build_word_influence_html(result), unsafe_allow_html=True)
    strongest = result.strongest_supporter
    conclusion = (
        f'“{strongest.word}” gave the largest positive support to the '
        f"{result.label.lower()} result in this short sentence."
        if strongest is not None
        else "No word produced a positive change in the selected result."
    )
    render_chart_explanation(
        st,
        ChartExplanation(
            what_it_shows=(
                "The percentages divide the measured change across the words. "
                "They total 100% when at least one word changes the result."
            ),
            why_it_matters=(
                "The check shows which words changed the live model result when "
                "each word was removed one at a time."
            ),
            conclusion=conclusion,
            uncertainty=(
                "These are relative influence percentages, not confidence and "
                "not proof of meaning. Removing a word also changes the sentence."
            ),
        ),
    )
    st.caption(
        f"Live model: {result.deployment_model} · Method: remove one word at a "
        "time and measure the probability change."
    )
    with st.expander("Open the exact word table"):
        st.dataframe(build_word_influence_table(result), width="stretch")
