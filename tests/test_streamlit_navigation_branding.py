"""Verify portfolio branding, page order, and completed-page status."""

from __future__ import annotations

from tests.streamlit_test_support import import_project_module, source_text

EXPECTED_KEYS = (
    "executive_overview",
    "analyze",
    "forecasts",
    "historical_intelligence",
    "explainability",
    "model_training",
    "model_comparison",
    "scenario_analysis",
    "provenance",
    "about_ruturaj",
)


def test_navigation_contains_every_completed_page_in_a_clear_order() -> None:
    """Require the agreed recruiter-friendly page order without duplicates."""

    navigation = import_project_module("app.navigation")
    items = navigation.get_navigation_items()
    assert tuple(item.key for item in items) == EXPECTED_KEYS
    assert len({item.key for item in items}) == len(items)


def test_all_navigation_pages_are_available_after_package_seven() -> None:
    """Ensure no completed page is still labelled as a future placeholder."""

    navigation = import_project_module("app.navigation")
    assert all(item.is_available for item in navigation.get_navigation_items())


def test_navigation_labels_are_readable_and_descriptive() -> None:
    """Require visible names and simple summaries for every page."""

    navigation = import_project_module("app.navigation")
    for item in navigation.get_navigation_items():
        assert item.label.strip()
        assert len(item.summary.split()) >= 6
        assert item.key not in item.display_label


def test_branding_uses_ruturaj_mokashi_as_the_owner() -> None:
    """Require the approved portfolio identity across the application."""

    branding = import_project_module("app.branding")
    brand = branding.get_portfolio_brand()
    assert brand.owner_name == "Ruturaj Mokashi"
    assert "Financial News" in brand.product_name
    assert "end-to-end" in brand.portfolio_statement


def test_branding_html_escapes_untrusted_text() -> None:
    """Prevent future branding text changes from injecting active HTML."""

    branding = import_project_module("app.branding")
    unsafe = branding.PortfolioBrand(
        owner_name="<script>alert(1)</script>",
        product_name="<b>Product</b>",
        short_name="Safe",
        tagline="<img src=x>",
        portfolio_statement="<iframe>",
    )
    header = branding.build_brand_header_html(unsafe)
    footer = branding.build_footer_html(unsafe)
    assert "<script>" not in header
    assert "&lt;script&gt;" in header
    assert "<iframe>" not in footer


def test_about_page_separates_completed_and_future_work() -> None:
    """Prevent portfolio claims from presenting later phases as finished."""

    about = source_text("app/pages/about_ruturaj.py")
    assert "Ruturaj Mokashi" in about
    assert "Completed and verified" in about
    assert "Next phases" in about
    assert "public deployment" in about.lower()
