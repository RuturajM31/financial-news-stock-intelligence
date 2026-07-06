"""Verify every completed Streamlit page and the application router."""

from __future__ import annotations

import ast

from tests.streamlit_test_support import (
    assert_module_documented,
    source_text,
    source_tree,
)

REQUIRED_PAGES = {
    "executive_overview": ("app/pages/executive_overview.py", "render_executive_overview"),
    "analyze": ("app/pages/analyze.py", "render_analyze_page"),
    "forecasts": ("app/pages/forecasts.py", "render_forecasts_page"),
    "historical_intelligence": (
        "app/pages/historical_intelligence.py",
        "render_historical_intelligence_page",
    ),
    "explainability": ("app/pages/explainability.py", "render_explainability_page"),
    "model_training": ("app/pages/model_training.py", "render_model_training_page"),
    "model_comparison": (
        "app/pages/model_comparison.py",
        "render_model_comparison_page",
    ),
    "scenario_analysis": (
        "app/pages/scenario_analysis.py",
        "render_scenario_analysis_page",
    ),
    "provenance": ("app/pages/provenance.py", "render_provenance_page"),
    "about_ruturaj": ("app/pages/about_ruturaj.py", "render_about_ruturaj_page"),
}


def test_all_completed_page_files_exist_and_are_documented() -> None:
    """Require one documented module for every completed navigation page."""

    for relative, _ in REQUIRED_PAGES.values():
        assert_module_documented(relative)


def test_each_page_exposes_its_expected_render_function() -> None:
    """Require the public render function named in the page contract."""

    for relative, function_name in REQUIRED_PAGES.values():
        tree = source_tree(relative)
        functions = {
            node.name for node in tree.body if isinstance(node, ast.FunctionDef)
        }
        assert function_name in functions, f"Missing {function_name} in {relative}"


def test_entry_point_imports_every_completed_page() -> None:
    """Require the entry point to import every page renderer explicitly."""

    entry = source_text("app/streamlit_app.py")
    for _, function_name in REQUIRED_PAGES.values():
        assert function_name in entry


def test_entry_point_routes_every_completed_page_key() -> None:
    """Require one explicit route branch for every completed page key."""

    entry = source_text("app/streamlit_app.py")
    for key in REQUIRED_PAGES:
        assert f'selected_key == "{key}"' in entry


def test_completed_pages_do_not_load_model_libraries_directly() -> None:
    """Keep model runtimes behind FastAPI rather than inside Streamlit pages."""

    forbidden = (
        "import torch",
        "from torch",
        "import transformers",
        "from transformers",
        "import sklearn",
        "from sklearn",
        "import joblib",
    )
    for relative, _ in REQUIRED_PAGES.values():
        lowered = source_text(relative).lower()
        assert not any(fragment in lowered for fragment in forbidden)


def test_completed_pages_use_plain_language_error_renderer() -> None:
    """Require protected pages to use the shared safe problem renderer."""

    protected = (
        "app/pages/analyze.py",
        "app/pages/forecasts.py",
        "app/pages/historical_intelligence.py",
        "app/pages/explainability.py",
        "app/pages/model_comparison.py",
        "app/pages/scenario_analysis.py",
        "app/pages/provenance.py",
    )
    combined = "\n".join(source_text(relative) for relative in protected)
    assert "render_problem" in combined
    assert "traceback" not in combined.lower()
