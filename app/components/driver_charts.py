"""Build accessible 2D chart specifications for movement-driver evidence.

The functions return plain Plotly-compatible dictionaries. Plotly is imported by
Streamlit only when the chart is rendered, which keeps module tests lightweight.
Every visual is followed by the shared plain-language explanation panel.
"""

from __future__ import annotations

from typing import Any, Sequence

from app.components.chart_explanations import (
    ChartExplanation,
    render_chart_explanation,
)
from app.services.explanation_service import DriverView


_PLOTLY_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "scrollZoom": True,
    "toImageButtonOptions": {"format": "png", "filename": "ruturaj-model-drivers"},
}


def _normalized_strengths(drivers: Sequence[DriverView]) -> list[float]:
    """Return strengths as percentages of the strongest visible driver."""

    strongest = max((driver.strength for driver in drivers), default=0.0)
    if strongest <= 0.0:
        return [0.0 for _ in drivers]
    return [driver.strength / strongest * 100.0 for driver in drivers]


def build_global_driver_figure(drivers: Sequence[DriverView]) -> dict[str, Any]:
    """Return a horizontal ranking of factors that usually matter."""

    if not drivers:
        raise ValueError("Global drivers must not be empty.")
    strengths = _normalized_strengths(drivers)
    ordered = list(zip(drivers, strengths, strict=True))[::-1]
    return {
        "data": [
            {
                "type": "bar",
                "orientation": "h",
                "x": [round(value, 4) for _, value in ordered],
                "y": [driver.label for driver, _ in ordered],
                "customdata": [
                    [driver.feature_key, driver.method, driver.interpretation]
                    for driver, _ in ordered
                ],
                "hovertemplate": (
                    "<b>%{y}</b><br>Relative strength: %{x:.1f}%"
                    "<br>Source key: %{customdata[0]}"
                    "<br>Method: %{customdata[1]}"
                    "<br>%{customdata[2]}<extra></extra>"
                ),
            }
        ],
        "layout": {
            "title": {"text": "Factors that usually matter", "x": 0.01},
            "xaxis": {"title": "Relative strength", "range": [0, 105]},
            "yaxis": {"title": ""},
            "margin": {"l": 20, "r": 20, "t": 60, "b": 45},
            "height": max(360, len(drivers) * 56),
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(0,0,0,0)",
            "showlegend": False,
        },
    }


def build_local_driver_figure(drivers: Sequence[DriverView]) -> dict[str, Any]:
    """Return a signed bar chart for factors affecting the current result."""

    if not drivers:
        raise ValueError("Local drivers must not be empty.")
    values = [driver.signed_effect * 100.0 for driver in drivers]
    return {
        "data": [
            {
                "type": "bar",
                "orientation": "h",
                "x": values[::-1],
                "y": [driver.label for driver in drivers][::-1],
                "customdata": [
                    [driver.direction or "not stated", driver.method, driver.interpretation]
                    for driver in drivers
                ][::-1],
                "hovertemplate": (
                    "<b>%{y}</b><br>Change in selected result: %{x:.2f} points"
                    "<br>Direction: %{customdata[0]}"
                    "<br>Method: %{customdata[1]}"
                    "<br>%{customdata[2]}<extra></extra>"
                ),
            }
        ],
        "layout": {
            "title": {"text": "Factors that affected this result", "x": 0.01},
            "xaxis": {
                "title": "Change in selected probability (percentage points)",
                "zeroline": True,
            },
            "yaxis": {"title": ""},
            "margin": {"l": 20, "r": 20, "t": 60, "b": 55},
            "height": max(360, len(drivers) * 56),
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(0,0,0,0)",
            "showlegend": False,
        },
    }


def build_driver_table(drivers: Sequence[DriverView]) -> list[dict[str, Any]]:
    """Return a non-color table fallback with the original source keys."""

    rows: list[dict[str, Any]] = []
    for driver in drivers:
        rows.append(
            {
                "Rank": driver.rank,
                "Plain label": driver.label,
                "Source key": driver.feature_key,
                "Strength": round(driver.strength, 6),
                "Effect": (
                    round(driver.probability_effect * 100.0, 4)
                    if driver.probability_effect is not None
                    else None
                ),
                "Direction": driver.direction or "Not stated",
                "Method": driver.method,
            }
        )
    return rows


def render_driver_charts(
    st: Any,
    global_drivers: Sequence[DriverView],
    local_drivers: Sequence[DriverView],
) -> None:
    """Render both driver scopes, their conclusions, and table fallbacks."""

    first, second = st.tabs(
        ["Factors that usually matter", "Factors affecting this result"]
    )
    with first:
        st.plotly_chart(
            build_global_driver_figure(global_drivers),
            width="stretch",
            config=_PLOTLY_CONFIG,
            key="rm_global_driver_chart",
        )
        render_chart_explanation(
            st,
            ChartExplanation(
                what_it_shows=(
                    "The ranking shows which inputs have mattered most across "
                    "the verified reference data."
                ),
                why_it_matters=(
                    "It explains the model's general behaviour rather than only "
                    "one forecast."
                ),
                conclusion=(
                    f"{global_drivers[0].label} is the strongest general driver "
                    "in the displayed evidence."
                ),
                uncertainty=(
                    "General importance does not prove that a factor caused a "
                    "real market move."
                ),
            ),
        )
        with st.expander("Open the exact table"):
            st.dataframe(build_driver_table(global_drivers), width="stretch")

    with second:
        st.plotly_chart(
            build_local_driver_figure(local_drivers),
            width="stretch",
            config=_PLOTLY_CONFIG,
            key="rm_local_driver_chart",
        )
        strongest = max(local_drivers, key=lambda driver: driver.strength)
        render_chart_explanation(
            st,
            ChartExplanation(
                what_it_shows=(
                    "Bars to the right support the selected forecast. Bars to "
                    "the left reduce support for it."
                ),
                why_it_matters=(
                    "This separates the factors that affected the current result "
                    "from the factors that matter in general."
                ),
                conclusion=(
                    f"{strongest.label} had the largest measured effect among "
                    "the displayed current-result drivers."
                ),
                uncertainty=(
                    "The values describe model sensitivity. They do not prove "
                    "cause and effect."
                ),
            ),
        )
        with st.expander("Open the exact table"):
            st.dataframe(build_driver_table(local_drivers), width="stretch")
