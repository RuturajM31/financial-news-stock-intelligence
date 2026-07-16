"""Strict loading and calculations for the public AI Experiment Lab."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any


LABEL_ORDER = ("Bearish", "Neutral", "Bullish")


class ExperimentDataError(ValueError):
    """Raised when a required verified experiment artifact is invalid."""


@dataclass(frozen=True)
class ModelMetrics:
    name: str
    accuracy: float
    macro_f1: float
    macro_precision: float | None
    macro_recall: float | None
    confusion_matrix: tuple[tuple[int, ...], ...]
    per_class: dict[str, dict[str, float]]
    source: str


@dataclass(frozen=True)
class ExperimentLabData:
    current: ModelMetrics
    historical: dict[str, ModelMetrics]
    history: tuple[dict[str, Any], ...]
    best_checkpoint: str
    manifest: dict[str, Any]
    benchmark: dict[str, Any]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentDataError(f"Could not read verified artifact {path.name}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ExperimentDataError(f"Verified artifact {path.name} must contain a JSON object.")
    return payload


def _finite(value: Any, field: str, *, optional: bool = False) -> float | None:
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ExperimentDataError(f"Invalid numeric value for {field}.")
    return float(value)


def _load_metrics(path: Path, name: str) -> ModelMetrics:
    payload = _read_json(path)
    metrics = payload.get("test_metrics")
    evaluation = payload.get("test_evaluation")
    if not isinstance(metrics, dict) or not isinstance(evaluation, dict):
        raise ExperimentDataError(f"{path.name} is missing test metrics or evaluation data.")
    labels = tuple(evaluation.get("label_order", ()))
    if labels != LABEL_ORDER:
        raise ExperimentDataError(f"{path.name} has an unexpected label order: {labels!r}")
    raw_matrix = evaluation.get("confusion_matrix")
    if not isinstance(raw_matrix, list) or len(raw_matrix) != len(LABEL_ORDER):
        raise ExperimentDataError(f"{path.name} has an invalid confusion matrix.")
    matrix: list[tuple[int, ...]] = []
    for row in raw_matrix:
        if not isinstance(row, list) or len(row) != len(LABEL_ORDER):
            raise ExperimentDataError(f"{path.name} has an invalid confusion matrix row.")
        clean_row = tuple(int(value) for value in row)
        if any(value < 0 for value in clean_row):
            raise ExperimentDataError(f"{path.name} has a negative confusion-matrix count.")
        matrix.append(clean_row)
    report = evaluation.get("classification_report", {})
    per_class: dict[str, dict[str, float]] = {}
    for label in LABEL_ORDER:
        values = report.get(label) if isinstance(report, dict) else None
        if not isinstance(values, dict):
            values = evaluation.get("per_class", {}).get(label, {})
        if not isinstance(values, dict):
            values = {}
        per_class[label] = {
            "precision": _finite(values.get("precision", metrics.get(f"test_{label.lower()}_precision")), f"{label} precision"),
            "recall": _finite(values.get("recall", metrics.get(f"test_{label.lower()}_recall")), f"{label} recall"),
            "f1": _finite(values.get("f1-score", values.get("f1", metrics.get(f"test_{label.lower()}_f1"))), f"{label} F1"),
            "support": _finite(values.get("support", sum(matrix[LABEL_ORDER.index(label)])), f"{label} support"),
        }
    return ModelMetrics(
        name=name,
        accuracy=_finite(metrics.get("test_accuracy"), f"{name} accuracy"),
        macro_f1=_finite(metrics.get("test_macro_f1"), f"{name} macro-F1"),
        macro_precision=_finite(metrics.get("test_macro_precision"), f"{name} macro precision", optional=True),
        macro_recall=_finite(metrics.get("test_macro_recall"), f"{name} macro recall", optional=True),
        confusion_matrix=tuple(matrix), per_class=per_class, source=path.name,
    )


def load_experiment_lab_data(project_root: Path) -> ExperimentLabData:
    """Load current, historical and runtime artifacts without cross-substitution."""

    metrics_root = project_root / "reports" / "metrics"
    manifest_root = project_root / "artifacts" / "manifests"
    current = _load_metrics(metrics_root / "bert_sentiment_current_run_metrics.json", "Full BERT current")
    historical = {
        "Full BERT historical": _load_metrics(metrics_root / "bert_sentiment_metrics.json", "Full BERT historical"),
        "DistilBERT historical": _load_metrics(metrics_root / "distilbert_sentiment_metrics.json", "DistilBERT historical"),
        "BERT-LoRA historical": _load_metrics(metrics_root / "bert_lora_sentiment_metrics.json", "BERT-LoRA historical"),
    }
    history_payload = _read_json(metrics_root / "bert_sentiment_current_run_history.json")
    log_history = history_payload.get("log_history")
    if not isinstance(log_history, list):
        raise ExperimentDataError("Current trainer history is missing log_history.")
    history = tuple(item for item in log_history if isinstance(item, dict))
    best_checkpoint = Path(str(history_payload.get("best_model_checkpoint", "Not recorded"))).name
    manifest = _read_json(manifest_root / "bert_sentiment_current_run_manifest.json")
    benchmark = _read_json(metrics_root / "bert_sentiment_current_run_benchmark.json")
    for field in ("model_load_seconds", "warm_sentence_milliseconds", "cuda_peak_allocated_bytes"):
        if field in benchmark:
            _finite(benchmark[field], field)
    return ExperimentLabData(current, historical, history, best_checkpoint, manifest, benchmark)


def leaderboard(data: ExperimentLabData) -> list[ModelMetrics]:
    """Rank historical model experiments by verified macro-F1."""

    return sorted(data.historical.values(), key=lambda item: (-item.macro_f1, -item.accuracy, item.name))


def normalize_confusion(matrix: tuple[tuple[int, ...], ...], mode: str) -> list[list[float]]:
    """Return counts, actual-row percentages, or predicted-column percentages."""

    values = [[float(value) for value in row] for row in matrix]
    if mode == "Counts":
        return values
    if mode == "Normalized by actual class":
        return [[100 * value / sum(row) if sum(row) else 0.0 for value in row] for row in values]
    if mode == "Normalized by predicted class":
        columns = [sum(values[row][column] for row in range(len(values))) for column in range(len(values))]
        return [[100 * value / columns[column] if columns[column] else 0.0 for column, value in enumerate(row)] for row in values]
    raise ValueError(f"Unknown confusion-matrix mode: {mode}")


def error_flows(matrix: tuple[tuple[int, ...], ...]) -> list[tuple[str, str, int]]:
    """Return each non-zero off-diagonal error direction exactly once."""

    return [
        (LABEL_ORDER[row], LABEL_ORDER[column], int(matrix[row][column]))
        for row in range(len(LABEL_ORDER)) for column in range(len(LABEL_ORDER))
        if row != column and matrix[row][column] > 0
    ]


def diagnostic_findings(metrics: ModelMetrics) -> dict[str, Any]:
    """Calculate evidence-based class and error findings from stored evaluation data."""

    class_rows = metrics.per_class
    strongest = {
        measure: max(LABEL_ORDER, key=lambda label: class_rows[label][measure])
        for measure in ("precision", "recall", "f1")
    }
    weakest_label, weakest_measure = min(
        ((label, measure) for label in LABEL_ORDER for measure in ("precision", "recall", "f1")),
        key=lambda pair: class_rows[pair[0]][pair[1]],
    )
    flows = error_flows(metrics.confusion_matrix)
    largest = max(flows, key=lambda item: item[2]) if flows else ("None", "None", 0)
    total_errors = sum(item[2] for item in flows)
    neutral_errors = sum(count for source, target, count in flows if "Neutral" in (source, target))
    directional_errors = total_errors - neutral_errors
    return {
        "strongest": strongest,
        "weakest": (weakest_label, weakest_measure, class_rows[weakest_label][weakest_measure]),
        "largest_flow": largest,
        "largest_error_share": largest[2] / total_errors if total_errors else 0.0,
        "error_pattern": "Neutral versus directional sentiment" if neutral_errors >= directional_errors else "direct Bearish-versus-Bullish confusion",
        "total_errors": total_errors,
    }
