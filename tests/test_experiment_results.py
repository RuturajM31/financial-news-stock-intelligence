from __future__ import annotations

import json
from pathlib import Path

import pytest

from financial_news_intelligence.models.experiment_results import (
    ExperimentDataError,
    diagnostic_findings,
    error_flows,
    leaderboard,
    load_experiment_lab_data,
    normalize_confusion,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def lab():
    return load_experiment_lab_data(ROOT)


def test_current_and_historical_metrics_are_separate(lab) -> None:
    assert lab.current.accuracy == pytest.approx(0.9092664092664092)
    assert lab.current.macro_f1 == pytest.approx(0.8864301276065981)
    assert lab.historical["Full BERT historical"].accuracy == pytest.approx(0.9131274131274131)
    assert lab.historical["Full BERT historical"].macro_f1 == pytest.approx(0.8899574644672684)
    assert lab.historical["DistilBERT historical"].accuracy == pytest.approx(0.8996138996138996)
    assert lab.historical["DistilBERT historical"].macro_f1 == pytest.approx(0.8778560157683483)
    assert lab.historical["BERT-LoRA historical"].accuracy == pytest.approx(0.8416988416988417)
    assert lab.historical["BERT-LoRA historical"].macro_f1 == pytest.approx(0.8153521710896103)


def test_leaderboard_uses_verified_historical_macro_f1(lab) -> None:
    assert [item.name for item in leaderboard(lab)] == [
        "Full BERT historical", "DistilBERT historical", "BERT-LoRA historical"
    ]


def test_current_history_checkpoint_runtime_and_size_are_verified(lab) -> None:
    assert lab.best_checkpoint == "checkpoint-453"
    assert lab.manifest["timing"]["training_seconds"] == pytest.approx(80.93103929999779)
    assert lab.benchmark["model_load_seconds"] == pytest.approx(1.9254996000017854)
    assert lab.benchmark["warm_sentence_milliseconds"] == pytest.approx(1.6101000001071952)
    assert lab.benchmark["model_artifact_size_bytes"] == 438_912_889
    assert all("eval_accuracy" not in item or item["epoch"] in {1.0, 2.0, 3.0} for item in lab.history)


def test_confusion_matrix_and_normalization_reconcile(lab) -> None:
    matrix = lab.current.confusion_matrix
    assert matrix == ((58, 1, 4), (10, 296, 16), (5, 11, 117))
    assert normalize_confusion(matrix, "Counts") == [[58, 1, 4], [10, 296, 16], [5, 11, 117]]
    rows = normalize_confusion(matrix, "Normalized by actual class")
    columns = normalize_confusion(matrix, "Normalized by predicted class")
    assert all(sum(row) == pytest.approx(100.0) for row in rows)
    assert all(sum(columns[row][column] for row in range(3)) == pytest.approx(100.0) for column in range(3))


def test_per_class_metrics_and_error_flows_match_saved_evaluation(lab) -> None:
    assert lab.current.per_class["Bearish"]["f1"] == pytest.approx(0.8529411764705882)
    assert lab.current.per_class["Neutral"]["precision"] == pytest.approx(0.961038961038961)
    assert lab.current.per_class["Bullish"]["support"] == 133
    assert error_flows(lab.current.confusion_matrix) == [
        ("Bearish", "Neutral", 1), ("Bearish", "Bullish", 4),
        ("Neutral", "Bearish", 10), ("Neutral", "Bullish", 16),
        ("Bullish", "Bearish", 5), ("Bullish", "Neutral", 11),
    ]


def test_diagnostic_findings_are_calculated(lab) -> None:
    findings = diagnostic_findings(lab.current)
    assert findings["strongest"] == {"precision": "Neutral", "recall": "Bearish", "f1": "Neutral"}
    assert findings["largest_flow"] == ("Neutral", "Bullish", 16)
    assert findings["total_errors"] == 47
    assert findings["largest_error_share"] == pytest.approx(16 / 47)
    assert findings["error_pattern"] == "Neutral versus directional sentiment"


def test_manifest_dataset_counts_reconcile(lab) -> None:
    summary = lab.manifest["dataset_summary"]
    assert sum(split["records"] for split in summary.values()) == 3448
    assert sum(split["Bearish"] for split in summary.values()) == 420
    assert sum(split["Neutral"] for split in summary.values()) == 2141
    assert sum(split["Bullish"] for split in summary.values()) == 887


def test_invalid_current_artifact_is_rejected_without_substitution(tmp_path: Path) -> None:
    (tmp_path / "reports" / "metrics").mkdir(parents=True)
    (tmp_path / "artifacts" / "manifests").mkdir(parents=True)
    (tmp_path / "reports" / "metrics" / "bert_sentiment_current_run_metrics.json").write_text(
        json.dumps({"test_metrics": {"test_accuracy": float("nan")}, "test_evaluation": {}}),
        encoding="utf-8",
    )
    with pytest.raises(ExperimentDataError):
        load_experiment_lab_data(tmp_path)
