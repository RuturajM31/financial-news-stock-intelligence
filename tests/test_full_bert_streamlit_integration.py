from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


APP = Path(__file__).resolve().parents[1] / "app" / "public_cloud_app.py"


def _open_page(name: str) -> AppTest:
    app = AppTest.from_file(str(APP), default_timeout=45)
    app.session_state["public_dashboard_page"] = name
    app.run(timeout=45)
    assert not app.exception
    return app


def test_all_public_routes_render() -> None:
    for page in ("Overview", "Model Results", "About / Architecture"):
        app = _open_page(page)
        assert page in app.session_state["public_dashboard_page"]


def test_model_results_experiment_lab_renders_verified_metrics() -> None:
    app = _open_page("Model Results")
    rendered = " ".join(element.value for element in app.markdown)
    assert "90.93%" in rendered
    assert "0.8864" in rendered
    assert "91.31%" in rendered
    assert "0.8900" in rendered
    assert "checkpoint-453" in rendered
    assert "80.93 seconds" in rendered
    assert [tab.label for tab in app.tabs] == [
        "Executive Results", "Training Dynamics", "Benchmark & Efficiency", "Error Analysis"
    ]
    assert len(app.get("plotly_chart")) >= 8
    assert not app.exception

def test_sample_runs_real_full_bert_and_persists() -> None:
    app = _open_page("Analyze Article")
    app.segmented_control(key="an_input_choice").set_value("Presentation sample").run(timeout=45)
    app.button(key="an_load_sample").click().run(timeout=45)
    assert len(app.session_state["an_loaded_body"].split()) >= 100
    app.button(key="an_analyze_article").click().run(timeout=45)
    assert not app.exception
    assert app.session_state["an_results_generated"] is True
    assert app.session_state["an_result_bert"].label in {"Bearish", "Neutral", "Bullish"}
    assert len(app.session_state["an_result_bert"].sentences) >= 4
    assert sum(app.session_state["an_result_bert"].probabilities.values()) == pytest.approx(1.0)
    assert all(len(item.embedding) == 768 for item in app.session_state["an_result_bert"].sentences)
    assert [tab.label for tab in app.tabs] == [
        "Summary", "Sentence Evidence", "Semantic & Token View", "Lexical Cues", "Article Reader"
    ]
    assert len(app.get("plotly_chart")) >= 4
    signature = app.session_state["an_result_signature"]
    app.run(timeout=45)
    assert app.session_state["an_result_signature"] == signature
    assert app.session_state["an_results_generated"] is True


def test_manual_change_clears_stale_result() -> None:
    app = _open_page("Analyze Article")
    app.segmented_control(key="an_input_choice").set_value("Presentation sample").run(timeout=45)
    app.button(key="an_load_sample").click().run(timeout=45)
    app.button(key="an_analyze_article").click().run(timeout=45)
    app.segmented_control(key="an_input_choice").set_value("Paste article").run(timeout=45)
    app.text_area(key="an_manual_body").set_value(
        "The company published a routine filing with factual details and no change to guidance. "
        "Management will discuss the quarter next week after the scheduled board meeting. "
        "The notice contains administrative information for shareholders and describes the reporting timetable."
    ).run(timeout=45)
    assert app.session_state["an_results_generated"] is False
    assert app.session_state["an_result_bert"] is None
    assert app.session_state["an_selected_sentence"] == 0
    assert app.session_state["an_token_network"] is None
