"""Build one purposeful 3D driver landscape with an always-visible 2D fallback."""

from __future__ import annotations

from typing import Any, Sequence

from app.components.chart_explanations import (
    ChartExplanation,
    render_chart_explanation,
)
from app.services.explanation_service import DriverView


_PLOTLY_3D_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "scrollZoom": True,
    "toImageButtonOptions": {
        "format": "png",
        "filename": "ruturaj-driver-landscape-3d",
    },
}


def build_driver_landscape_rows(
    global_drivers: Sequence[DriverView],
    local_drivers: Sequence[DriverView],
) -> list[dict[str, Any]]:
    """Join general and current-result evidence by the verified feature key."""

    global_by_key = {driver.feature_key: driver for driver in global_drivers}
    local_by_key = {driver.feature_key: driver for driver in local_drivers}
    feature_keys = sorted(set(global_by_key) | set(local_by_key))
    rows: list[dict[str, Any]] = []
    for feature_key in feature_keys:
        global_driver = global_by_key.get(feature_key)
        local_driver = local_by_key.get(feature_key)
        label_source = global_driver or local_driver
        if label_source is None:
            continue
        rows.append(
            {
                "feature_key": feature_key,
                "label": label_source.label,
                "general_strength": (
                    global_driver.strength * 100.0 if global_driver else 0.0
                ),
                "current_strength": (
                    local_driver.strength * 100.0 if local_driver else 0.0
                ),
                "current_effect": (
                    local_driver.signed_effect * 100.0 if local_driver else 0.0
                ),
            }
        )
    return rows


def build_driver_landscape_figure(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Return a rotatable 3D scatter specification for joined driver evidence."""

    if len(rows) < 3:
        raise ValueError("At least three joined driver rows are needed for 3D.")
    marker_sizes = [
        max(8.0, min(34.0, 8.0 + row["current_strength"] * 0.6))
        for row in rows
    ]
    return {
        "data": [
            {
                "type": "scatter3d",
                "mode": "markers+text",
                "x": [row["general_strength"] for row in rows],
                "y": [row["current_strength"] for row in rows],
                "z": [row["current_effect"] for row in rows],
                "text": [row["label"] for row in rows],
                "customdata": [row["feature_key"] for row in rows],
                "textposition": "top center",
                "marker": {
                    "size": marker_sizes,
                    "opacity": 0.86,
                },
                "hovertemplate": (
                    "<b>%{text}</b><br>Usually matters: %{x:.2f}"
                    "<br>Current strength: %{y:.2f}"
                    "<br>Current effect: %{z:.2f} points"
                    "<br>Source key: %{customdata}<extra></extra>"
                ),
            }
        ],
        "layout": {
            "title": {"text": "3D driver landscape", "x": 0.01},
            "scene": {
                "xaxis": {"title": "Usually matters"},
                "yaxis": {"title": "Current strength"},
                "zaxis": {"title": "Current effect"},
                "camera": {"eye": {"x": 1.45, "y": 1.45, "z": 1.15}},
            },
            "height": 620,
            "margin": {"l": 0, "r": 0, "t": 60, "b": 0},
            "paper_bgcolor": "rgba(0,0,0,0)",
            "showlegend": False,
        },
    }


def build_driver_landscape_table(
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return a readable 2D fallback for the same 3D driver values."""

    return [
        {
            "Driver": row["label"],
            "Usually matters": round(row["general_strength"], 4),
            "Current strength": round(row["current_strength"], 4),
            "Current effect (points)": round(row["current_effect"], 4),
            "Source key": row["feature_key"],
        }
        for row in rows
    ]


def render_driver_landscape(
    st: Any,
    global_drivers: Sequence[DriverView],
    local_drivers: Sequence[DriverView],
) -> None:
    """Render the interactive 3D view, a larger view, and a 2D fallback."""

    rows = build_driver_landscape_rows(global_drivers, local_drivers)
    st.markdown("### Explore the driver landscape")
    st.caption(
        "Drag to rotate. Scroll to zoom. Use the camera button to reset the view."
    )
    if len(rows) >= 3:
        large_view = st.toggle(
            "Use a larger 3D view",
            value=False,
            key="rm_large_driver_landscape",
        )
        figure = build_driver_landscape_figure(rows)
        if large_view:
            figure["layout"]["height"] = 820
        st.plotly_chart(
            figure,
            width="stretch",
            config=_PLOTLY_3D_CONFIG,
            key="rm_driver_landscape_3d",
        )
    else:
        st.info(
            "The API returned fewer than three joined drivers, so the 3D view "
            "is hidden. The exact 2D table remains available below."
        )

    render_chart_explanation(
        st,
        ChartExplanation(
            what_it_shows=(
                "Each point compares how much a driver usually matters with how "
                "strongly it affected this result."
            ),
            why_it_matters=(
                "A driver can be important in general but have little effect on "
                "one forecast, or the reverse."
            ),
            conclusion=(
                "Points that are high on both strength measures deserve the most "
                "attention when reading this explanation."
            ),
            uncertainty=(
                "The 3D position describes model evidence only. It does not prove "
                "that the driver caused a market move."
            ),
        ),
    )
    with st.expander("Open the accessible 2D fallback", expanded=False):
        st.dataframe(build_driver_landscape_table(rows), width="stretch")
