"""Verify plain-language chart explanations and word-influence boundaries."""

from __future__ import annotations

import pytest

from tests.streamlit_test_support import import_project_module, source_text


def test_chart_explanation_requires_all_three_core_sections() -> None:
    """Reject charts that do not explain meaning, value, and conclusion."""

    module = import_project_module("app.components.chart_explanations")
    explanation = module.ChartExplanation("", "Why", "Conclusion")
    with pytest.raises(ValueError, match="must not be empty"):
        explanation.validate()


def test_chart_explanation_html_contains_plain_language_headings() -> None:
    """Require the three agreed explanation headings below each chart."""

    module = import_project_module("app.components.chart_explanations")
    html = module.build_chart_explanation_html(
        module.ChartExplanation(
            what_it_shows="One result",
            why_it_matters="It changes the decision",
            conclusion="Use the faster model",
            uncertainty="The result is close",
        )
    )
    assert "What this chart shows" in html
    assert "Why it matters" in html
    assert "Conclusion" in html
    assert "Uncertainty" in html


def test_chart_explanation_escapes_active_html() -> None:
    """Prevent explanation text from injecting browser content."""

    module = import_project_module("app.components.chart_explanations")
    html = module.build_chart_explanation_html(
        module.ChartExplanation("<script>x</script>", "safe", "safe")
    )
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_attention_requires_a_complete_verified_contract() -> None:
    """Do not show partial or invented attention values."""

    module = import_project_module("app.components.attention_explorer")
    assert not module.has_verified_attention({})
    assert not module.has_verified_attention(
        {"attention": {"tokens": ["profit"], "values": [], "method": "layer mean"}}
    )
    assert module.has_verified_attention(
        {
            "attention": {
                "tokens": ["profit", "rose"],
                "values": [0.4, 0.6],
                "method": "verified layer mean",
            }
        }
    )


def test_attention_copy_states_that_attention_is_not_true_importance() -> None:
    """Keep the scientific limitation visible in simple words."""

    source = source_text("app/components/attention_explorer.py").lower()
    assert "does not always show the true reason" in source
    assert "not a percentage of true" in source


def test_word_influence_has_a_non_color_table_fallback() -> None:
    """Require exact text values in addition to visual word chips."""

    source = source_text("app/components/word_influence.py")
    assert "build_word_influence_table" in source
    assert "Direction" in source
    assert "Share of measured influence" in source


def test_word_influence_copy_does_not_call_percentages_attention() -> None:
    """Keep measured removal effects separate from raw model attention."""

    source = source_text("app/components/word_influence.py").lower()
    assert "not confidence" in source
    assert "remove one word" in source
