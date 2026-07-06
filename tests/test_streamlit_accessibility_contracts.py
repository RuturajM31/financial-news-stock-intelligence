"""Verify readable labels, keyboard focus, responsive rules, and fallbacks."""

from __future__ import annotations

from tests.streamlit_test_support import source_text


def test_theme_has_visible_keyboard_focus_rules() -> None:
    """Require a visible focus indicator for keyboard users."""

    css = source_text("app/styles/premium_theme.css")
    assert ":focus-visible" in css
    assert "outline" in css


def test_theme_respects_reduced_motion_preference() -> None:
    """Disable decorative movement when the operating system requests it."""

    css = source_text("app/styles/premium_theme.css")
    assert "prefers-reduced-motion" in css


def test_theme_contains_a_small_screen_layout() -> None:
    """Require a responsive rule for phone and narrow-window use."""

    css = source_text("app/styles/premium_theme.css")
    assert "@media" in css
    assert "max-width" in css


def test_probability_views_include_text_labels_not_color_only() -> None:
    """Keep Down, Flat, Up, and sentiment classes readable without color."""

    movement = source_text("app/components/movement_results.py")
    sentiment = source_text("app/components/sentiment_results.py")
    for label in ("Down", "Flat", "Up"):
        assert label in movement
    for label in ("Bearish", "Neutral", "Bullish"):
        assert label in sentiment


def test_word_chips_include_accessible_labels() -> None:
    """Require screen-reader text for influence percentage and direction."""

    source = source_text("app/components/word_influence.py")
    assert "aria-label" in source
    assert "percent" in source
    assert "direction" in source


def test_brand_header_has_a_labelled_main_title() -> None:
    """Connect the hero section to a stable visible product heading."""

    source = source_text("app/branding.py")
    assert 'aria-labelledby="rm-product-title"' in source
    assert 'id="rm-product-title"' in source


def test_tables_are_available_for_advanced_visuals() -> None:
    """Require readable tables for drivers, evidence, words, and scenarios."""

    files = (
        "app/components/charts_3d.py",
        "app/components/word_influence.py",
        "app/components/provenance_panels.py",
        "app/components/scenario_results.py",
    )
    combined = "\n".join(source_text(relative) for relative in files)
    assert combined.count("dataframe") >= 4


def test_errors_use_four_plain_language_parts() -> None:
    """Require what, where, why, and next-step guidance for visible failures."""

    source = source_text("app/components/status_badges.py")
    for field in ("problem.what_failed", "problem.where_failed", "problem.why_failed", "problem.safe_next_step"):
        assert field in source
    for label in ("Where:", "Why:", "Safe next step:"):
        assert label in source
