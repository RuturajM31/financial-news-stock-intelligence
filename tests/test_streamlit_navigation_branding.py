"""Verify the retained public navigation and portfolio identity."""

from __future__ import annotations

import inspect

from app.public_cloud_app import _render_sidebar, render_public_streamlit_cloud_app


EXPECTED_PAGES = (
    "Overview",
    "Analyze Article",
    "Model Results",
    "About / Architecture",
)
RETIRED_PAGE_LABELS = (
    "Forecasts",
    "Historical Intelligence",
    "Explainability",
    "Scenario Analysis",
    "Model Comparison",
    "Model Training / Evidence",
    "Provenance",
    "Architecture / System Design",
    "3D Intelligence",
    "About / Project Purpose",
    "Visual QA / Page Audit",
)


def test_sidebar_lists_only_the_four_retained_pages_in_order() -> None:
    """Keep the visible route order stable across Streamlit reruns."""

    source = inspect.getsource(_render_sidebar)
    positions = [source.index(page) for page in EXPECTED_PAGES]
    assert positions == sorted(positions)
    assert 'key="public_dashboard_page"' in source
    assert all(label not in source for label in RETIRED_PAGE_LABELS)


def test_route_map_has_one_renderer_for_each_retained_page() -> None:
    """Prevent a sidebar option from pointing at a missing renderer."""

    source = inspect.getsource(render_public_streamlit_cloud_app)
    assert all(f'"{page}":' in source for page in EXPECTED_PAGES)
    assert source.count("_render_sentiment_architecture_page") == 1


def test_public_branding_keeps_the_named_product_and_owner() -> None:
    """Keep the portfolio identity inside the rendered sidebar source."""

    source = inspect.getsource(_render_sidebar)
    assert "Financial News Sentiment Analyzer" in source
    assert "Ruturaj Mokashi" in source
