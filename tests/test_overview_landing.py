from __future__ import annotations

import inspect
from pathlib import Path

from streamlit.testing.v1 import AppTest

from app.public_cloud_app import _render_sentiment_overview_page


APP = Path(__file__).resolve().parents[1] / "app" / "public_cloud_app.py"


def _overview() -> AppTest:
    app = AppTest.from_file(str(APP), default_timeout=45)
    app.session_state["public_dashboard_page"] = "Overview"
    app.run(timeout=45)
    assert not app.exception
    return app


def test_overview_is_lightweight_and_uses_no_model_inference() -> None:
    source = inspect.getsource(_render_sentiment_overview_page)
    assert "_load_public_bert_runtime" not in source
    assert "analyze_article_with_bert" not in source
    assert "contextual_tokens" not in source


def test_overview_displays_verified_current_metrics_and_scope() -> None:
    app = _overview()
    rendered = " ".join(element.value for element in app.markdown)
    assert "90.93%" in rendered
    assert "0.8864" in rendered
    assert "3,448" in rendered
    assert "1.61 ms" in rendered
    assert "CURRENT RUN" in rendered
    assert "HISTORICAL" in rendered
    assert "Interface preview" in rendered
    assert "does not predict stock prices" in rendered
    assert "Example company" not in rendered
    assert "NVIDIA · NVDA" not in rendered
    assert not app.tabs


def test_primary_overview_ctas_navigate_to_existing_routes() -> None:
    app = _overview()
    app.button(key="overview_analyze").click().run(timeout=45)
    assert app.session_state["public_dashboard_page"] == "Analyze Article"

    app = _overview()
    app.button(key="overview_models").click().run(timeout=45)
    assert app.session_state["public_dashboard_page"] == "Model Results"

    app = _overview()
    app.button(key="overview_close_arch").click().run(timeout=45)
    assert app.session_state["public_dashboard_page"] == "About / Architecture"


def test_sample_story_cta_loads_presentation_sample_without_analysis() -> None:
    app = _overview()
    app.button(key="overview_open_sample").click().run(timeout=45)
    assert app.session_state["public_dashboard_page"] == "Analyze Article"
    assert app.session_state["an_source_type"] == "Built-in sample"
    assert len(app.session_state["an_loaded_body"].split()) >= 100
    assert app.session_state["an_results_generated"] is False
    assert app.session_state["an_result_bert"] is None
