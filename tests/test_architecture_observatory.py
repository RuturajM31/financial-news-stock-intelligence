from __future__ import annotations

import inspect
from pathlib import Path

from streamlit.testing.v1 import AppTest

from app.public_cloud_app import _render_sentiment_architecture_page

APP = Path(__file__).resolve().parents[1] / "app" / "public_cloud_app.py"


def _architecture() -> AppTest:
    app = AppTest.from_file(str(APP), default_timeout=45)
    app.session_state["public_dashboard_page"] = "About / Architecture"
    app.run(timeout=45)
    assert not app.exception
    return app


def _rendered(app: AppTest) -> str:
    return " ".join(element.value for element in app.markdown)


def test_architecture_page_is_inference_free() -> None:
    source = inspect.getsource(_render_sentiment_architecture_page)
    assert "_load_public_bert_runtime" not in source
    assert "analyze_article_with_bert" not in source


def test_runtime_blueprint_and_stage_inspector() -> None:
    app = _architecture()
    rendered = _rendered(app)
    for text in ("One model artifact. Two connected systems.", "Article source", "Clean extraction",
                 "Sentence segmentation", "Batched Full BERT inference", "Sentence class scores",
                 "Article aggregation", "Evidence workspace", "FULL BERT ARTIFACT",
                 "0 Bearish · 1 Neutral · 2 Bullish", "INPUT", "PROCESS", "OUTPUT", "STATUS"):
        assert text in rendered
    app.segmented_control(key="ar_runtime_node").set_value("Evidence workspace").run(timeout=45)
    assert app.session_state["ar_runtime_node"] == "Evidence workspace"
    assert not app.exception


def test_training_blueprint_branches_and_separates_metrics() -> None:
    app = _architecture()
    app.segmented_control(key="ar_view_mode").set_value("Training system").run(timeout=45)
    rendered = _rendered(app)
    for text in ("3,453", "3,448", "5 duplicates removed", "Full BERT", "DistilBERT", "BERT-LoRA",
                 "Bearish 420", "Neutral 2,141", "Bullish 887", "90.93%", "0.8864",
                 "checkpoint-453", "80.93 seconds", "91.31%", "0.8900",
                 "HISTORICAL BEST · SEPARATE RUN"):
        assert text in rendered
    app.segmented_control(key="ar_training_node").set_value("Champion model artifact").run(timeout=45)
    assert not app.exception


def test_deployment_blueprint_separates_current_and_target() -> None:
    app = _architecture()
    app.segmented_control(key="ar_view_mode").set_value("Deployment topology").run(timeout=45)
    rendered = _rendered(app)
    for text in ("CURRENT WORKING TOPOLOGY", "Operational locally", "TARGET PRODUCTION TOPOLOGY",
                 "not yet publicly validated", "FastAPI model service", "model artifact distribution",
                 "hosted inference service", "security and request controls", "monitoring and scaling"):
        assert text in rendered
    assert "Streamlit Community Cloud" not in rendered


def test_evidence_lineage_guarantees_and_collapsed_details() -> None:
    app = _architecture()
    rendered = _rendered(app)
    for text in ("1.93 s", "1.61 ms / sentence", "418.58 MiB", "90.93%", "0.8864",
                 "449.8 MB", "Not recorded", "One continuous artifact trail",
                 "Fixed held-out evaluation", "Transparent aggregation", "Product safety",
                 "does not predict stock prices"):
        assert text.lower() in rendered.lower()
    labels = {item.label for item in app.expander}
    assert {"Artifact registry", "Technology by system layer", "Full integrity contract"} <= labels
    source = inspect.getsource(_render_sentiment_architecture_page)
    assert "expanded=True" not in source


def test_architecture_ctas_navigate_to_existing_routes() -> None:
    for key, route in (("ar_run_analysis", "Analyze Article"), ("ar_model_results", "Model Results"), ("ar_overview", "Overview")):
        app = _architecture()
        app.button(key=key).click().run(timeout=45)
        assert app.session_state["public_dashboard_page"] == route
        assert not app.exception