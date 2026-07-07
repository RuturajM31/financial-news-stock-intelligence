"""Verify every 3D view has clear controls and a readable 2D alternative."""

from __future__ import annotations

from tests.streamlit_test_support import source_text

THREE_D_FILES = (
    "app/components/charts_3d.py",
    "app/components/model_comparison_charts.py",
    "app/components/scenario_results.py",
)


def test_all_3d_modules_use_plotly_scatter3d() -> None:
    """Require the approved interactive 3D chart type in each 3D module."""

    for relative in THREE_D_FILES:
        source = source_text(relative)
        assert "scatter3d" in source.lower(), relative


def test_driver_3d_view_has_an_accessible_table_fallback() -> None:
    """Require exact driver values when a user cannot use the 3D view."""

    source = source_text("app/components/charts_3d.py")
    assert "build_driver_landscape_table" in source
    assert "2D fallback" in source


def test_model_comparison_3d_view_has_exact_2d_rows() -> None:
    """Keep BERT comparison values available outside the rotatable chart."""

    source = source_text("app/components/model_comparison_charts.py")
    assert "comparison_rows" in source
    assert "build_3d_tradeoff_figure" in source


def test_scenario_3d_view_has_an_accessible_2d_fallback() -> None:
    """Require a readable scenario table beside the 3D relationship view."""

    source = source_text("app/components/scenario_results.py")
    assert "Open the accessible 2D fallback" in source
    assert "build_scenario_3d_figure" in source


def test_3d_views_include_plain_language_explanations() -> None:
    """Require written meaning and conclusions near advanced visuals."""

    combined = "\n".join(source_text(relative) for relative in THREE_D_FILES)
    assert "What this" in combined or "what_it_shows" in combined
    assert "Conclusion" in combined or "conclusion" in combined


def test_3d_views_do_not_use_invented_random_values() -> None:
    """Keep all chart points tied to verified API or benchmark evidence."""

    combined = "\n".join(source_text(relative).lower() for relative in THREE_D_FILES)
    assert "random.random" not in combined
    assert "numpy.random" not in combined
    assert "np.random" not in combined
