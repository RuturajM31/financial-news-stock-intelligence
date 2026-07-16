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


def _validate_label_order(test_evaluation: dict[str, Any], source_name: str) -> None:
    """Require the saved class order used throughout public inference."""

    saved_label_order = tuple(test_evaluation.get("label_order", ()))
    if saved_label_order != LABEL_ORDER:
        raise ExperimentDataError(
            f"{source_name} has an unexpected label order: {saved_label_order!r}"
        )


def _validate_confusion_matrix(
    test_evaluation: dict[str, Any],
    source_name: str,
) -> tuple[tuple[int, ...], ...]:
    """Validate dimensions and non-negative integer counts in the saved matrix."""

    raw_confusion_matrix = test_evaluation.get("confusion_matrix")
    expected_dimension = len(LABEL_ORDER)

    if (
        not isinstance(raw_confusion_matrix, list)
        or len(raw_confusion_matrix) != expected_dimension
    ):
        raise ExperimentDataError(
            f"{source_name} has an invalid confusion matrix."
        )

    validated_confusion_matrix: list[tuple[int, ...]] = []

    for raw_matrix_row in raw_confusion_matrix:
        if (
            not isinstance(raw_matrix_row, list)
            or len(raw_matrix_row) != expected_dimension
        ):
            raise ExperimentDataError(
                f"{source_name} has an invalid confusion matrix row."
            )

        validated_matrix_row = tuple(
            int(raw_value) for raw_value in raw_matrix_row
        )
        if any(count < 0 for count in validated_matrix_row):
            raise ExperimentDataError(
                f"{source_name} has a negative confusion-matrix count."
            )

        validated_confusion_matrix.append(validated_matrix_row)

    return tuple(validated_confusion_matrix)


def _resolve_class_metrics(
    class_label: str,
    classification_report: Any,
    test_evaluation: dict[str, Any],
) -> dict[str, Any]:
    """Use the classification report first, preserving the saved fallback."""

    class_metrics = None
    if isinstance(classification_report, dict):
        class_metrics = classification_report.get(class_label)

    if not isinstance(class_metrics, dict):
        saved_per_class_metrics = test_evaluation.get("per_class", {})
        if isinstance(saved_per_class_metrics, dict):
            class_metrics = saved_per_class_metrics.get(class_label, {})

    if not isinstance(class_metrics, dict):
        return {}

    return class_metrics


def _extract_per_class_metrics(
    test_metrics: dict[str, Any],
    test_evaluation: dict[str, Any],
    validated_confusion_matrix: tuple[tuple[int, ...], ...],
) -> dict[str, dict[str, float]]:
    """Resolve and validate precision, recall, F1, and support per class."""

    classification_report = test_evaluation.get("classification_report", {})
    per_class_metrics: dict[str, dict[str, float]] = {}

    for class_index, class_label in enumerate(LABEL_ORDER):
        class_metrics = _resolve_class_metrics(
            class_label,
            classification_report,
            test_evaluation,
        )
        metric_prefix = f"test_{class_label.lower()}"
        default_support = sum(validated_confusion_matrix[class_index])

        per_class_metrics[class_label] = {
            "precision": _finite(
                class_metrics.get(
                    "precision",
                    test_metrics.get(f"{metric_prefix}_precision"),
                ),
                f"{class_label} precision",
            ),
            "recall": _finite(
                class_metrics.get(
                    "recall",
                    test_metrics.get(f"{metric_prefix}_recall"),
                ),
                f"{class_label} recall",
            ),
            "f1": _finite(
                class_metrics.get(
                    "f1-score",
                    class_metrics.get(
                        "f1",
                        test_metrics.get(f"{metric_prefix}_f1"),
                    ),
                ),
                f"{class_label} F1",
            ),
            "support": _finite(
                class_metrics.get("support", default_support),
                f"{class_label} support",
            ),
        }

    return per_class_metrics


def _load_metrics(path: Path, name: str) -> ModelMetrics:
    """Read, validate, and construct one model-metrics record."""

    metrics_payload = _read_json(path)
    test_metrics = metrics_payload.get("test_metrics")
    test_evaluation = metrics_payload.get("test_evaluation")

    # Reject partial artifacts instead of displaying mixed or incomplete data.
    if not isinstance(test_metrics, dict) or not isinstance(
        test_evaluation,
        dict,
    ):
        raise ExperimentDataError(
            f"{path.name} is missing test metrics or evaluation data."
        )

    _validate_label_order(test_evaluation, path.name)
    validated_confusion_matrix = _validate_confusion_matrix(
        test_evaluation,
        path.name,
    )
    per_class_metrics = _extract_per_class_metrics(
        test_metrics,
        test_evaluation,
        validated_confusion_matrix,
    )

    return ModelMetrics(
        name=name,
        accuracy=_finite(
            test_metrics.get("test_accuracy"),
            f"{name} accuracy",
        ),
        macro_f1=_finite(
            test_metrics.get("test_macro_f1"),
            f"{name} macro-F1",
        ),
        macro_precision=_finite(
            test_metrics.get("test_macro_precision"),
            f"{name} macro precision",
            optional=True,
        ),
        macro_recall=_finite(
            test_metrics.get("test_macro_recall"),
            f"{name} macro recall",
            optional=True,
        ),
        confusion_matrix=validated_confusion_matrix,
        per_class=per_class_metrics,
        source=path.name,
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
