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
    """Load the immutable experiment artifacts shared by this module's checks."""

    return load_experiment_lab_data(ROOT)


def test_current_and_historical_metrics_are_separate(lab) -> None:
    current_result = lab.current
    historical_results = lab.historical
    assert current_result.accuracy == pytest.approx(0.9092664092664092)
    assert current_result.macro_f1 == pytest.approx(0.8864301276065981)
    assert historical_results["Full BERT historical"].accuracy == pytest.approx(0.9131274131274131)
    assert historical_results["Full BERT historical"].macro_f1 == pytest.approx(0.8899574644672684)
    assert historical_results["DistilBERT historical"].accuracy == pytest.approx(0.8996138996138996)
    assert historical_results["DistilBERT historical"].macro_f1 == pytest.approx(0.8778560157683483)
    assert historical_results["BERT-LoRA historical"].accuracy == pytest.approx(0.8416988416988417)
    assert historical_results["BERT-LoRA historical"].macro_f1 == pytest.approx(0.8153521710896103)


def test_leaderboard_uses_verified_historical_macro_f1(lab) -> None:
    leaderboard_names = [
        leaderboard_entry.name
        for leaderboard_entry in leaderboard(lab)
    ]
    assert leaderboard_names == [
        "Full BERT historical",
        "DistilBERT historical",
        "BERT-LoRA historical",
    ]


def test_current_history_checkpoint_runtime_and_size_are_verified(lab) -> None:
    experiment_manifest = lab.manifest
    benchmark_report = lab.benchmark
    assert lab.best_checkpoint == "checkpoint-453"
    assert experiment_manifest["timing"]["training_seconds"] == pytest.approx(80.93103929999779)
    assert benchmark_report["model_load_seconds"] == pytest.approx(1.9254996000017854)
    assert benchmark_report["warm_sentence_milliseconds"] == pytest.approx(1.6101000001071952)
    assert benchmark_report["model_artifact_size_bytes"] == 438_912_889
    assert all(
        "eval_accuracy" not in history_entry
        or history_entry["epoch"] in {1.0, 2.0, 3.0}
        for history_entry in lab.history
    )


def test_confusion_matrix_and_normalization_reconcile(lab) -> None:
    confusion_matrix = lab.current.confusion_matrix
    assert confusion_matrix == ((58, 1, 4), (10, 296, 16), (5, 11, 117))
    assert normalize_confusion(confusion_matrix, "Counts") == [
        [58, 1, 4],
        [10, 296, 16],
        [5, 11, 117],
    ]
    actual_class_rows = normalize_confusion(
        confusion_matrix,
        "Normalized by actual class",
    )
    predicted_class_columns = normalize_confusion(
        confusion_matrix,
        "Normalized by predicted class",
    )
    assert all(
        sum(normalized_row) == pytest.approx(100.0)
        for normalized_row in actual_class_rows
    )
    assert all(
        sum(
            predicted_class_columns[row_index][column_index]
            for row_index in range(3)
        ) == pytest.approx(100.0)
        for column_index in range(3)
    )


def test_per_class_metrics_and_error_flows_match_saved_evaluation(lab) -> None:
    per_class_metrics = lab.current.per_class
    expected_error_flows = [
        ("Bearish", "Neutral", 1),
        ("Bearish", "Bullish", 4),
        ("Neutral", "Bearish", 10),
        ("Neutral", "Bullish", 16),
        ("Bullish", "Bearish", 5),
        ("Bullish", "Neutral", 11),
    ]
    assert per_class_metrics["Bearish"]["f1"] == pytest.approx(0.8529411764705882)
    assert per_class_metrics["Neutral"]["precision"] == pytest.approx(0.961038961038961)
    assert per_class_metrics["Bullish"]["support"] == 133
    assert error_flows(lab.current.confusion_matrix) == expected_error_flows


def test_diagnostic_findings_are_calculated(lab) -> None:
    diagnostic_result = diagnostic_findings(lab.current)
    assert diagnostic_result["strongest"] == {
        "precision": "Neutral",
        "recall": "Bearish",
        "f1": "Neutral",
    }
    assert diagnostic_result["largest_flow"] == ("Neutral", "Bullish", 16)
    assert diagnostic_result["total_errors"] == 47
    assert diagnostic_result["largest_error_share"] == pytest.approx(16 / 47)
    assert diagnostic_result["error_pattern"] == "Neutral versus directional sentiment"


def test_manifest_dataset_counts_reconcile(lab) -> None:
    dataset_summary = lab.manifest["dataset_summary"]
    assert sum(split_summary["records"] for split_summary in dataset_summary.values()) == 3448
    assert sum(split_summary["Bearish"] for split_summary in dataset_summary.values()) == 420
    assert sum(split_summary["Neutral"] for split_summary in dataset_summary.values()) == 2141
    assert sum(split_summary["Bullish"] for split_summary in dataset_summary.values()) == 887


def test_invalid_current_artifact_is_rejected_without_substitution(tmp_path: Path) -> None:
    metrics_directory = tmp_path / "reports" / "metrics"
    manifests_directory = tmp_path / "artifacts" / "manifests"
    metrics_directory.mkdir(parents=True)
    manifests_directory.mkdir(parents=True)
    current_metrics_path = metrics_directory / "bert_sentiment_current_run_metrics.json"
    current_metrics_path.write_text(
        json.dumps({"test_metrics": {"test_accuracy": float("nan")}, "test_evaluation": {}}),
        encoding="utf-8",
    )
    with pytest.raises(ExperimentDataError):
        load_experiment_lab_data(tmp_path)
