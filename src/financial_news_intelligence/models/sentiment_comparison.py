"""
Compare DistilBERT, full BERT, and BERT-LoRA on one test split.

Purpose
-------
Create the final evidence table for sentiment-model selection after all three
experiments have completed on the same Financial PhraseBank test records.

Inputs and source variables
---------------------------
The module reads each model's immutable training manifest, metrics JSON, and
saved artifact directory. It does not rerun inference or change model files.

Processing and formulas
-----------------------
- Test grain is the confusion-matrix cell total.
- Inference latency is ``test_runtime * 1000 / test_records``.
- Measured peak process RSS is used when recorded by the benchmark runner.
- Estimated FP32 parameter memory remains a transparent legacy fallback.
- Trainable percentage is ``trainable_parameters / total_parameters * 100``.
- Quality ranking uses macro F1, weighted F1, accuracy, lower latency, then
  lower comparison memory.
- The deployment candidate must be within 0.02 macro-F1 points of the best
  model, then minimizes recorded comparison memory and latency.

Outputs and downstream use
--------------------------
- ``reports/metrics/sentiment_model_comparison.json``;
- ``artifacts/manifests/sentiment_model_champion.json``.

These files support the later FastAPI, Streamlit, model-card, and deployment
decisions. They do not alter the application automatically.

Assumptions and limitations
---------------------------
All models must use identical train, validation, and untouched test source
checksums. Test metrics must share one row count and 3x3 label order: Bearish,
Neutral, Bullish. Measured peak RSS includes the benchmark process, framework, model, and
evaluation workload. The verified legacy DistilBERT baseline may fall back
to an explicitly labelled FP32 parameter estimate when its older manifest
does not contain measured RSS. LoRA still requires base BERT at inference.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence


LABEL_ORDER = ("Bearish", "Neutral", "Bullish")
EXPECTED_CLASS_COUNT = len(LABEL_ORDER)
EXPECTED_SPLIT_RECORDS = {
    "train": 2_413,
    "validation": 517,
    "test": 518,
}
DEPLOYMENT_QUALITY_TOLERANCE = 0.02
BYTES_PER_FP32_PARAMETER = 4
MEBIBYTE = 1024 * 1024

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DISTILBERT_MANIFEST = (
    PROJECT_ROOT
    / "artifacts"
    / "manifests"
    / "distilbert_sentiment_training_manifest.json"
)
DISTILBERT_METRICS = (
    PROJECT_ROOT / "reports" / "metrics" / "distilbert_sentiment_metrics.json"
)
BERT_MANIFEST = (
    PROJECT_ROOT
    / "artifacts"
    / "manifests"
    / "bert_sentiment_training_manifest.json"
)
BERT_METRICS = (
    PROJECT_ROOT / "reports" / "metrics" / "bert_sentiment_metrics.json"
)
LORA_MANIFEST = (
    PROJECT_ROOT
    / "artifacts"
    / "manifests"
    / "bert_lora_sentiment_training_manifest.json"
)
LORA_METRICS = (
    PROJECT_ROOT / "reports" / "metrics" / "bert_lora_sentiment_metrics.json"
)

COMPARISON_FILE = (
    PROJECT_ROOT
    / "reports"
    / "metrics"
    / "sentiment_model_comparison.json"
)
CHAMPION_MANIFEST_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "manifests"
    / "sentiment_model_champion.json"
)


class SentimentComparisonError(RuntimeError):
    """Raised when model evidence cannot support a fair comparison."""


@dataclass(frozen=True)
class ModelEvidence:
    """Store one normalized row in the three-model comparison."""

    model_key: str
    experiment_name: str
    model_family: str
    benchmark_role: str
    model_id: str
    model_revision: str
    status: str
    test_records: int
    test_accuracy: float
    test_macro_f1: float
    test_weighted_f1: float
    confusion_matrix: list[list[int]]
    per_class_metrics: dict[str, dict[str, float | int]]
    total_parameters: int
    trainable_parameters: int
    trainable_percentage: float
    training_seconds: float
    test_runtime_seconds: float
    inference_milliseconds_per_record: float
    estimated_fp32_parameter_memory_mib: float
    measured_peak_process_rss_mib: float | None
    comparison_memory_mib: float
    comparison_memory_source: str
    artifact_size_bytes: int
    artifact_size_mib: float
    final_model_directory: str
    manifest_path: str
    metrics_path: str
    manifest_sha256: str
    metrics_sha256: str
    artifact_files: list[dict[str, Any]]
    artifact_checksum_source: str
    source_files: dict[str, dict[str, Any]]


def require_regular_file(file_path: Path, description: str) -> None:
    """Require one existing non-symlink file before reading it."""

    if not file_path.exists():
        raise SentimentComparisonError(f"Missing {description}: {file_path}")
    if file_path.is_symlink() or not file_path.is_file():
        raise SentimentComparisonError(
            f"Unsafe {description}; expected a regular file: {file_path}"
        )


def sha256(file_path: Path) -> str:
    """Return one evidence file's hexadecimal SHA-256 checksum."""

    require_regular_file(file_path, "checksum source")
    digest = hashlib.sha256()
    with file_path.open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_object(file_path: Path, description: str) -> dict[str, Any]:
    """Load one UTF-8 JSON object with a precise failure message."""

    require_regular_file(file_path, description)
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SentimentComparisonError(
            f"Invalid {description}: {file_path}: {exc}"
        ) from exc

    if not isinstance(payload, dict):
        raise SentimentComparisonError(f"{description} must contain an object.")
    return payload


def require_mapping(
    payload: Mapping[str, Any],
    key: str,
    description: str,
) -> Mapping[str, Any]:
    """Require one nested JSON object used by the evidence contract."""

    value = payload.get(key)
    if not isinstance(value, dict):
        raise SentimentComparisonError(
            f"Missing or invalid {description}: {key}"
        )
    return value


def finite_positive(value: Any, field_name: str) -> float:
    """Require one positive numeric timing or size value."""

    if not isinstance(value, (int, float)):
        raise SentimentComparisonError(f"Missing numeric field: {field_name}")
    numeric = float(value)
    if numeric <= 0:
        raise SentimentComparisonError(f"{field_name} must be positive.")
    return numeric


def probability(value: Any, field_name: str) -> float:
    """Require one evaluation metric inside the inclusive 0..1 range."""

    if not isinstance(value, (int, float)):
        raise SentimentComparisonError(f"Missing numeric metric: {field_name}")
    numeric = float(value)
    if not 0.0 <= numeric <= 1.0:
        raise SentimentComparisonError(
            f"Metric {field_name} is outside 0..1: {numeric}"
        )
    return numeric


def artifact_size_bytes(directory: Path) -> int:
    """Sum regular files below one saved model or adapter directory."""

    if not directory.exists() or not directory.is_dir() or directory.is_symlink():
        raise SentimentComparisonError(
            f"Saved artifact directory is missing or unsafe: {directory}"
        )

    total = 0
    for file_path in directory.rglob("*"):
        if file_path.is_symlink():
            raise SentimentComparisonError(
                f"Symbolic links are not accepted in model evidence: {file_path}"
            )
        if file_path.is_file():
            total += file_path.stat().st_size

    if total <= 0:
        raise SentimentComparisonError(f"Artifact directory is empty: {directory}")
    return total


def artifact_inventory(directory: Path) -> list[dict[str, Any]]:
    """Checksum every regular file in one saved model directory."""

    if not directory.exists() or not directory.is_dir() or directory.is_symlink():
        raise SentimentComparisonError(
            f"Saved artifact directory is missing or unsafe: {directory}"
        )

    inventory: list[dict[str, Any]] = []
    for file_path in sorted(directory.rglob("*")):
        if file_path.is_symlink():
            raise SentimentComparisonError(
                f"Symbolic links are not accepted in model evidence: {file_path}"
            )
        if file_path.is_file():
            inventory.append(
                {
                    "path": file_path.relative_to(directory).as_posix(),
                    "sha256": sha256(file_path),
                    "size_bytes": file_path.stat().st_size,
                }
            )

    if not inventory:
        raise SentimentComparisonError(f"Artifact directory is empty: {directory}")
    return inventory


def normalize_confusion_matrix(
    metrics: Mapping[str, Any],
) -> tuple[list[list[int]], int]:
    """Require a non-negative 3x3 matrix and return its exact row count."""

    evaluation = require_mapping(metrics, "test_evaluation", "test evaluation")
    matrix = evaluation.get("confusion_matrix")
    if not (
        isinstance(matrix, list)
        and len(matrix) == EXPECTED_CLASS_COUNT
        and all(
            isinstance(row, list) and len(row) == EXPECTED_CLASS_COUNT
            for row in matrix
        )
    ):
        raise SentimentComparisonError("Expected one 3x3 confusion matrix.")

    normalized: list[list[int]] = []
    for row in matrix:
        if not all(isinstance(value, int) and value >= 0 for value in row):
            raise SentimentComparisonError(
                "Confusion-matrix values must be non-negative integers."
            )
        normalized.append(list(row))

    test_records = sum(sum(row) for row in normalized)
    if test_records <= 0:
        raise SentimentComparisonError("Confusion matrix contains no records.")
    return normalized, test_records


def canonical_label_name(raw_label: Any) -> str | None:
    """Map supported class identifiers to the approved display labels."""

    aliases = {
        "0": "Bearish",
        "bearish": "Bearish",
        "negative": "Bearish",
        "1": "Neutral",
        "neutral": "Neutral",
        "2": "Bullish",
        "bullish": "Bullish",
        "positive": "Bullish",
    }
    return aliases.get(str(raw_label).strip().casefold())


def calculated_per_class_metrics(
    confusion_matrix: list[list[int]],
) -> dict[str, dict[str, float | int]]:
    """Calculate class metrics independently from the saved matrix."""

    calculated: dict[str, dict[str, float | int]] = {}
    for class_id, label_name in enumerate(LABEL_ORDER):
        true_positive = confusion_matrix[class_id][class_id]
        false_positive = sum(
            confusion_matrix[row_id][class_id]
            for row_id in range(EXPECTED_CLASS_COUNT)
            if row_id != class_id
        )
        false_negative = sum(
            confusion_matrix[class_id][column_id]
            for column_id in range(EXPECTED_CLASS_COUNT)
            if column_id != class_id
        )
        support = sum(confusion_matrix[class_id])
        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative
        precision = (
            true_positive / precision_denominator
            if precision_denominator
            else 0.0
        )
        recall = (
            true_positive / recall_denominator
            if recall_denominator
            else 0.0
        )
        f1_denominator = precision + recall
        f1_score = (
            2.0 * precision * recall / f1_denominator
            if f1_denominator
            else 0.0
        )
        calculated[label_name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1_score),
            "support": int(support),
        }
    return calculated


def normalize_per_class_metrics(
    metrics: Mapping[str, Any],
    confusion_matrix: list[list[int]],
) -> dict[str, dict[str, float | int]]:
    """Normalize nested or scalar class evidence and verify matrix agreement.

    DistilBERT and full BERT use the established shared Trainer schema, while
    LoRA writes a title-case nested report. The comparison accepts both, then
    recalculates every value from the same confusion matrix so schema
    flexibility cannot hide inconsistent quality evidence.
    """

    evaluation = require_mapping(metrics, "test_evaluation", "test evaluation")
    raw_order = evaluation.get("label_order")
    if not isinstance(raw_order, list):
        raise SentimentComparisonError("Per-class label order is missing.")
    normalized_order = [canonical_label_name(value) for value in raw_order]
    if normalized_order != list(LABEL_ORDER):
        raise SentimentComparisonError(
            "Per-class label order must be Bearish, Neutral, Bullish."
        )

    candidates: dict[str, Mapping[str, Any]] = {}
    raw_per_class = evaluation.get("per_class")
    if isinstance(raw_per_class, dict):
        for raw_label, raw_values in raw_per_class.items():
            label_name = canonical_label_name(raw_label)
            if label_name is None or not isinstance(raw_values, dict):
                continue
            if label_name in candidates:
                raise SentimentComparisonError(
                    f"Duplicate per-class metrics for {label_name}."
                )
            candidates[label_name] = raw_values

    test_metrics = require_mapping(metrics, "test_metrics", "test metrics")
    if set(candidates) != set(LABEL_ORDER):
        candidates = {}
        for label_name in LABEL_ORDER:
            prefix = label_name.casefold()
            values = {
                "precision": test_metrics.get(f"test_{prefix}_precision"),
                "recall": test_metrics.get(f"test_{prefix}_recall"),
                "f1": test_metrics.get(f"test_{prefix}_f1"),
            }
            if all(value is not None for value in values.values()):
                candidates[label_name] = values

    if set(candidates) != set(LABEL_ORDER):
        raise SentimentComparisonError(
            "Per-class metrics must contain Bearish, Neutral, and Bullish "
            "as nested values or test_* scalar metrics."
        )

    calculated = calculated_per_class_metrics(confusion_matrix)
    normalized: dict[str, dict[str, float | int]] = {}
    tolerance = 1e-6
    for label_name in LABEL_ORDER:
        values = candidates[label_name]
        expected = calculated[label_name]
        reported_support = values.get("support", expected["support"])
        if reported_support != expected["support"]:
            raise SentimentComparisonError(
                f"{label_name} support must be {expected['support']}; "
                f"found {reported_support!r}."
            )
        normalized_values = {
            "precision": probability(
                values.get("precision"),
                f"{label_name}.precision",
            ),
            "recall": probability(
                values.get("recall"),
                f"{label_name}.recall",
            ),
            "f1": probability(values.get("f1"), f"{label_name}.f1"),
            "support": int(expected["support"]),
        }
        for metric_name in ("precision", "recall", "f1"):
            if abs(
                float(normalized_values[metric_name])
                - float(expected[metric_name])
            ) > tolerance:
                raise SentimentComparisonError(
                    f"{label_name} {metric_name} does not match the "
                    "confusion matrix."
                )
        normalized[label_name] = normalized_values
    return normalized


def resolve_memory_evidence(
    manifest: Mapping[str, Any],
    model_key: str,
    total_parameters: int,
) -> tuple[float, float | None, float, str]:
    """Resolve measured RSS or an explicitly labelled legacy fallback."""

    estimated = total_parameters * BYTES_PER_FP32_PARAMETER / MEBIBYTE
    memory = manifest.get("memory")
    if isinstance(memory, dict):
        peak = memory.get("peak_process_rss_mib")
        method = memory.get("measurement_method")
        if (
            isinstance(peak, (int, float))
            and float(peak) > 0
            and isinstance(method, str)
            and "ru_maxrss" in method
        ):
            measured = float(peak)
            return (
                estimated,
                measured,
                estimated,
                "fp32_parameter_estimate_consistent_fallback",
            )

    if model_key in {"bert", "bert_lora"}:
        raise SentimentComparisonError(
            f"{model_key} manifest lacks measured peak process RSS evidence."
        )
    return (
        estimated,
        None,
        estimated,
        "fp32_parameter_estimate_consistent_fallback",
    )


def resolve_test_runtime(
    manifest: Mapping[str, Any],
    metrics: Mapping[str, Any],
    test_metrics: Mapping[str, Any],
) -> float:
    """Resolve measured test inference time from supported evidence fields."""

    candidates: list[Any] = [test_metrics.get("test_runtime")]

    metrics_timing = metrics.get("timing")
    if isinstance(metrics_timing, dict):
        candidates.append(metrics_timing.get("test_inference_seconds"))

    manifest_timing = manifest.get("timing")
    if isinstance(manifest_timing, dict):
        candidates.append(manifest_timing.get("test_inference_seconds"))

    for candidate in candidates:
        if isinstance(candidate, (int, float)) and candidate > 0:
            return float(candidate)

    raise SentimentComparisonError("No positive test inference runtime was found.")


def line_count(file_path: Path) -> int:
    """Count non-empty source rows without loading a complete split."""

    require_regular_file(file_path, "dataset source")
    with file_path.open("r", encoding="utf-8") as source_file:
        count = sum(1 for line in source_file if line.strip())

    if count <= 0:
        raise SentimentComparisonError(
            f"Dataset source contains no records: {file_path}"
        )
    return count


def resolve_source_files(
    manifest: Mapping[str, Any],
    model_key: str,
) -> dict[str, dict[str, Any]]:
    """Verify all split paths, checksums, and optional row counts."""

    source_files = require_mapping(manifest, "source_files", "source files")
    verified: dict[str, dict[str, Any]] = {}

    for split_name in ("train", "validation", "test"):
        details = source_files.get(split_name)
        if not isinstance(details, dict):
            raise SentimentComparisonError(
                f"{model_key} manifest is missing {split_name} source evidence."
            )

        path_text = details.get("path")
        expected_checksum = details.get("checksum_sha256")
        expected_records = details.get("records")
        if not isinstance(path_text, str) or not path_text.strip():
            raise SentimentComparisonError(
                f"{model_key} {split_name} source path is missing."
            )
        if not (
            isinstance(expected_checksum, str)
            and len(expected_checksum) == 64
            and all(character in "0123456789abcdef" for character in expected_checksum)
        ):
            raise SentimentComparisonError(
                f"{model_key} {split_name} source checksum is invalid."
            )
        required_records = EXPECTED_SPLIT_RECORDS[split_name]
        if expected_records is not None and expected_records != required_records:
            raise SentimentComparisonError(
                f"{model_key} {split_name} must record exactly "
                f"{required_records} rows; found {expected_records!r}."
            )
        if expected_records is None and model_key != "distilbert":
            raise SentimentComparisonError(
                f"{model_key} {split_name} source row count is missing."
            )

        source_path = Path(path_text).expanduser().resolve()
        actual_checksum = sha256(source_path)
        actual_records = line_count(source_path)
        if actual_checksum != expected_checksum:
            raise SentimentComparisonError(
                f"{model_key} {split_name} source checksum changed."
            )
        if actual_records != required_records:
            raise SentimentComparisonError(
                f"{model_key} {split_name} split must contain exactly "
                f"{required_records} rows; found {actual_records}."
            )

        verified[split_name] = {
            "path": str(source_path),
            "checksum_sha256": actual_checksum,
            "records": actual_records,
        }

    return verified

def load_model_evidence(
    model_key: str,
    manifest_path: Path,
    metrics_path: Path,
) -> ModelEvidence:
    """Normalize one model's saved evidence into a comparison row."""

    manifest_path = manifest_path.resolve()
    metrics_path = metrics_path.resolve()
    manifest = load_json_object(manifest_path, f"{model_key} manifest")
    metrics = load_json_object(metrics_path, f"{model_key} metrics")

    if manifest.get("status") != "trained_and_evaluated":
        raise SentimentComparisonError(
            f"{model_key} status is not trained_and_evaluated."
        )

    required_text_fields = (
        "experiment_name",
        "model_id",
        "model_revision",
        "final_model_directory",
    )
    text_values: dict[str, str] = {}
    for field_name in required_text_fields:
        value = manifest.get(field_name)
        if not isinstance(value, str) or not value.strip():
            raise SentimentComparisonError(
                f"{model_key} manifest is missing {field_name}."
            )
        text_values[field_name] = value

    # The completed DistilBERT baseline predates the shared-engine identity
    # fields. Apply explicit backward-compatible defaults only for those two
    # descriptive values; all performance and provenance evidence remains
    # mandatory and comes from the original saved files.
    family_defaults = {
        "distilbert": "DistilBERT",
        "bert": "BERT",
        "bert_lora": "BERT-LoRA",
    }
    role_defaults = {
        "distilbert": "baseline_full_fine_tuning",
        "bert": "full_fine_tuning_comparison",
        "bert_lora": "parameter_efficient_comparison",
    }
    model_family = manifest.get("model_family", family_defaults.get(model_key))
    benchmark_role = manifest.get(
        "benchmark_role",
        role_defaults.get(model_key),
    )
    if not isinstance(model_family, str) or not model_family.strip():
        raise SentimentComparisonError(
            f"{model_key} model family could not be resolved."
        )
    if not isinstance(benchmark_role, str) or not benchmark_role.strip():
        raise SentimentComparisonError(
            f"{model_key} benchmark role could not be resolved."
        )

    parameters = require_mapping(manifest, "parameter_counts", "parameters")
    total_parameters = parameters.get("total_parameters")
    trainable_parameters = parameters.get("trainable_parameters")
    if not isinstance(total_parameters, int) or total_parameters <= 0:
        raise SentimentComparisonError(
            f"{model_key} total_parameters must be positive."
        )
    if not isinstance(trainable_parameters, int) or trainable_parameters <= 0:
        raise SentimentComparisonError(
            f"{model_key} trainable_parameters must be positive."
        )
    if trainable_parameters > total_parameters:
        raise SentimentComparisonError(
            f"{model_key} trainable parameters exceed total parameters."
        )

    timing = require_mapping(manifest, "timing", "training timing")
    training_seconds = finite_positive(
        timing.get("training_seconds"),
        f"{model_key}.training_seconds",
    )

    test_metrics = require_mapping(metrics, "test_metrics", "test metrics")
    accuracy = probability(
        test_metrics.get("test_accuracy"),
        f"{model_key}.test_accuracy",
    )
    macro_f1 = probability(
        test_metrics.get("test_macro_f1"),
        f"{model_key}.test_macro_f1",
    )
    weighted_f1 = probability(
        test_metrics.get("test_weighted_f1"),
        f"{model_key}.test_weighted_f1",
    )

    matrix, test_records = normalize_confusion_matrix(metrics)
    per_class_metrics = normalize_per_class_metrics(metrics, matrix)
    source_files = resolve_source_files(manifest, model_key)
    if source_files["test"]["records"] != test_records:
        raise SentimentComparisonError(
            f"{model_key} test source rows do not match its confusion matrix."
        )
    test_runtime = resolve_test_runtime(manifest, metrics, test_metrics)
    latency_ms = test_runtime * 1000.0 / test_records

    model_directory = Path(text_values["final_model_directory"]).expanduser()
    model_directory = model_directory.resolve()
    size_bytes = artifact_size_bytes(model_directory)
    actual_artifact_files = artifact_inventory(model_directory)
    recorded_artifact_files = manifest.get("artifact_files")
    if recorded_artifact_files is not None:
        if recorded_artifact_files != actual_artifact_files:
            raise SentimentComparisonError(
                f"{model_key} saved artifacts do not match manifest checksums."
            )
        artifact_checksum_source = "training_manifest_verified"
    elif model_key == "distilbert":
        artifact_checksum_source = "computed_during_comparison_legacy_baseline"
    else:
        raise SentimentComparisonError(
            f"{model_key} manifest artifact checksum inventory is missing."
        )

    (
        estimated_memory_mib,
        measured_peak_rss_mib,
        comparison_memory_mib,
        comparison_memory_source,
    ) = resolve_memory_evidence(manifest, model_key, total_parameters)

    return ModelEvidence(
        model_key=model_key,
        experiment_name=text_values["experiment_name"],
        model_family=model_family,
        benchmark_role=benchmark_role,
        model_id=text_values["model_id"],
        model_revision=text_values["model_revision"],
        status="trained_and_evaluated",
        test_records=test_records,
        test_accuracy=accuracy,
        test_macro_f1=macro_f1,
        test_weighted_f1=weighted_f1,
        confusion_matrix=matrix,
        per_class_metrics=per_class_metrics,
        total_parameters=total_parameters,
        trainable_parameters=trainable_parameters,
        trainable_percentage=(
            trainable_parameters / total_parameters * 100.0
        ),
        training_seconds=training_seconds,
        test_runtime_seconds=test_runtime,
        inference_milliseconds_per_record=latency_ms,
        estimated_fp32_parameter_memory_mib=estimated_memory_mib,
        measured_peak_process_rss_mib=measured_peak_rss_mib,
        comparison_memory_mib=comparison_memory_mib,
        comparison_memory_source=comparison_memory_source,
        artifact_size_bytes=size_bytes,
        artifact_size_mib=size_bytes / MEBIBYTE,
        final_model_directory=str(model_directory),
        manifest_path=str(manifest_path),
        metrics_path=str(metrics_path),
        manifest_sha256=sha256(manifest_path),
        metrics_sha256=sha256(metrics_path),
        artifact_files=actual_artifact_files,
        artifact_checksum_source=artifact_checksum_source,
        source_files=source_files,
    )



def apply_common_memory_basis(
    models: Sequence[ModelEvidence],
) -> list[ModelEvidence]:
    """Use one comparable memory basis across all three model rows."""

    if models and all(
        model.measured_peak_process_rss_mib is not None for model in models
    ):
        return [
            replace(
                model,
                comparison_memory_mib=float(
                    model.measured_peak_process_rss_mib
                ),
                comparison_memory_source=(
                    "measured_peak_process_rss_all_models"
                ),
            )
            for model in models
        ]

    return [
        replace(
            model,
            comparison_memory_mib=model.estimated_fp32_parameter_memory_mib,
            comparison_memory_source=(
                "fp32_parameter_estimate_consistent_fallback"
            ),
        )
        for model in models
    ]

def validate_common_dataset(models: Sequence[ModelEvidence]) -> int:
    """Require all models to use identical train, validation, and test data."""

    for split_name in ("train", "validation", "test"):
        checksums = {
            model.source_files[split_name]["checksum_sha256"]
            for model in models
        }
        record_counts = {
            model.source_files[split_name]["records"]
            for model in models
        }
        if len(checksums) != 1 or len(record_counts) != 1:
            details = {
                model.model_key: {
                    "checksum_sha256": model.source_files[split_name][
                        "checksum_sha256"
                    ],
                    "records": model.source_files[split_name]["records"],
                }
                for model in models
            }
            raise SentimentComparisonError(
                f"Models do not share one {split_name} split: {details}"
            )

    test_records = {model.test_records for model in models}
    if len(test_records) != 1:
        details = {model.model_key: model.test_records for model in models}
        raise SentimentComparisonError(
            f"Models do not share one test-row count: {details}"
        )
    return next(iter(test_records))


def validate_bert_base_alignment(models: Sequence[ModelEvidence]) -> None:
    """Require full BERT and BERT-LoRA to use one base ID and revision."""

    by_key = {model.model_key: model for model in models}
    try:
        bert = by_key["bert"]
        lora = by_key["bert_lora"]
    except KeyError as exc:
        raise SentimentComparisonError(
            "BERT and BERT-LoRA evidence are both required."
        ) from exc

    if bert.model_id != lora.model_id:
        raise SentimentComparisonError(
            "Full BERT and BERT-LoRA use different base model IDs."
        )
    if bert.model_revision != lora.model_revision:
        raise SentimentComparisonError(
            "Full BERT and BERT-LoRA use different base model revisions."
        )

def quality_sort_key(model: ModelEvidence) -> tuple[float, ...]:
    """Return the documented deterministic quality-ranking key."""

    return (
        -model.test_macro_f1,
        -model.test_weighted_f1,
        -model.test_accuracy,
        model.inference_milliseconds_per_record,
        model.comparison_memory_mib,
    )


def rank_models(
    models: Sequence[ModelEvidence],
) -> tuple[list[ModelEvidence], ModelEvidence, ModelEvidence]:
    """Return quality ranking plus quality and deployment selections."""

    if len(models) != 3:
        raise SentimentComparisonError(
            f"Exactly three model experiments are required, found {len(models)}."
        )

    ranked = sorted(models, key=quality_sort_key)
    quality_champion = ranked[0]
    minimum_macro_f1 = (
        quality_champion.test_macro_f1 - DEPLOYMENT_QUALITY_TOLERANCE
    )
    deployment_pool = [
        model for model in ranked if model.test_macro_f1 >= minimum_macro_f1
    ]
    deployment_champion = min(
        deployment_pool,
        key=lambda model: (
            model.comparison_memory_mib,
            model.inference_milliseconds_per_record,
            -model.test_macro_f1,
        ),
    )
    return ranked, quality_champion, deployment_champion


def write_json_atomic(file_path: Path, payload: Mapping[str, Any]) -> None:
    """Write one JSON object atomically to prevent partial decision evidence."""

    file_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = file_path.with_suffix(file_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary_path.replace(file_path)


def build_comparison(
    comparison_file: Path = COMPARISON_FILE,
    champion_file: Path = CHAMPION_MANIFEST_FILE,
) -> dict[str, Any]:
    """Load three experiments, rank them, and save the final evidence."""

    evidence = [
        load_model_evidence(
            "distilbert",
            DISTILBERT_MANIFEST,
            DISTILBERT_METRICS,
        ),
        load_model_evidence("bert", BERT_MANIFEST, BERT_METRICS),
        load_model_evidence("bert_lora", LORA_MANIFEST, LORA_METRICS),
    ]
    test_records = validate_common_dataset(evidence)
    validate_bert_base_alignment(evidence)
    evidence = apply_common_memory_basis(evidence)
    ranked, quality_champion, deployment_champion = rank_models(evidence)

    comparison_payload = {
        "status": "comparison_completed",
        "label_order": list(LABEL_ORDER),
        "test_records": test_records,
        "deployment_quality_tolerance_macro_f1": (
            DEPLOYMENT_QUALITY_TOLERANCE
        ),
        "quality_ranking": [model.model_key for model in ranked],
        "quality_champion": quality_champion.model_key,
        "deployment_champion": deployment_champion.model_key,
        "selection_rule": {
            "quality": (
                "macro_f1 desc, weighted_f1 desc, accuracy desc, "
                "latency asc, comparison memory asc"
            ),
            "deployment": (
                "within 0.02 macro F1 of best, then comparison memory asc, "
                "latency asc, macro F1 desc"
            ),
        },
        "dataset_sources": quality_champion.source_files,
        "models": [asdict(model) for model in ranked],
        "memory_method": (
            "Use measured peak process RSS only when all three manifests "
            "contain it. Otherwise use the same FP32 parameter-memory "
            "estimate for every model, while retaining measured RSS as "
            "separate evidence for BERT and LoRA."
        ),
        "lora_limitation": (
            "The LoRA adapter artifact requires the recorded base BERT model."
        ),
    }

    champion_payload = {
        "status": "champion_selected",
        "official_quality_champion": quality_champion.model_key,
        "recommended_deployment_model": deployment_champion.model_key,
        "test_records": test_records,
        "label_order": list(LABEL_ORDER),
        "quality_champion_metrics": {
            "accuracy": quality_champion.test_accuracy,
            "macro_f1": quality_champion.test_macro_f1,
            "weighted_f1": quality_champion.test_weighted_f1,
            "inference_milliseconds_per_record": (
                quality_champion.inference_milliseconds_per_record
            ),
            "comparison_memory_mib": quality_champion.comparison_memory_mib,
            "comparison_memory_source": (
                quality_champion.comparison_memory_source
            ),
        },
        "deployment_model_metrics": {
            "accuracy": deployment_champion.test_accuracy,
            "macro_f1": deployment_champion.test_macro_f1,
            "weighted_f1": deployment_champion.test_weighted_f1,
            "inference_milliseconds_per_record": (
                deployment_champion.inference_milliseconds_per_record
            ),
            "comparison_memory_mib": (
                deployment_champion.comparison_memory_mib
            ),
            "comparison_memory_source": (
                deployment_champion.comparison_memory_source
            ),
        },
        "comparison_file": str(comparison_file.resolve()),
        "source_evidence": {
            model.model_key: {
                "manifest_path": model.manifest_path,
                "manifest_sha256": model.manifest_sha256,
                "metrics_path": model.metrics_path,
                "metrics_sha256": model.metrics_sha256,
                "artifact_checksum_source": model.artifact_checksum_source,
                "artifact_files": model.artifact_files,
                "comparison_memory_mib": model.comparison_memory_mib,
                "comparison_memory_source": model.comparison_memory_source,
            }
            for model in evidence
        },
        "automatic_deployment_change": False,
    }

    write_json_atomic(comparison_file, comparison_payload)
    write_json_atomic(champion_file, champion_payload)
    return comparison_payload
