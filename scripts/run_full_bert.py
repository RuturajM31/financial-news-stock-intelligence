#!/usr/bin/env python3
"""
Run and verify the complete full-BERT sentiment benchmark.

Purpose
-------
Train ``google-bert/bert-base-uncased`` on the verified Financial PhraseBank
splits through the existing compact BERT wrapper, then review every required
model, metrics, timing, parameter, and provenance artifact.

Inputs and source variables
---------------------------
``BertTrainingConfig`` supplies the approved model identifier, train,
validation, and untouched test files, the shared random seed, class-weighting
rule, batch settings, sequence length, learning rate, and three-epoch budget.
The runner discovers the configured test file from the dataclass instead of
copying a private path into this script.

Processing and data journey
---------------------------
1. Confirm execution inside the isolated Transformer environment.
2. Protect existing full-BERT evidence from accidental replacement.
3. Validate the BERT configuration before model download or training.
4. Call the verified BERT wrapper and shared Transformer engine.
5. Load the saved manifest and metrics from disk.
6. Verify model identity, output paths, parameter counts, metrics, confusion
   matrix, evaluated row count, and final tokenizer/model files.

Outputs and downstream use
--------------------------
The run creates the full-BERT checkpoints, final model, metrics JSON, and
reproducibility manifest already defined by ``BertTrainingConfig``. These
artifacts are consumed by the LoRA comparison and final champion-selection
stage.

Safety, assumptions, and limitations
------------------------------------
The runner does not alter DistilBERT or BERT-smoke artifacts. Existing
full-BERT outputs are rejected unless ``--replace-existing`` is supplied.
``--verify-only`` reads saved evidence without training. A successful run
proves the benchmark completed; champion status is decided only after LoRA
and the common comparison stage also pass.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import resource
import shutil
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, Mapping


EXPECTED_CLASS_COUNT = 3
EXPECTED_MODEL_ID = "google-bert/bert-base-uncased"
EXPECTED_EXPERIMENT_NAME = "BERT Financial Sentiment"
EXPECTED_MODEL_FAMILY = "BERT"
EXPECTED_BENCHMARK_ROLE = "full_fine_tuning_comparison"
EXPECTED_SPLIT_RECORDS = {
    "train": 2_413,
    "validation": 517,
    "test": 518,
}
LABEL_ORDER = ("Bearish", "Neutral", "Bullish")
MEBIBYTE = 1024 * 1024

REQUIRED_FINAL_MODEL_FILES = {
    "config.json",
    "model.safetensors",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.txt",
}

TEST_FIELD_CANDIDATES = (
    "test_file",
    "test_path",
    "test_data_file",
)


class FullBertBenchmarkError(RuntimeError):
    """Raised when training preparation or saved evidence is unsafe."""


def ensure_isolated_environment() -> None:
    """
    Reject environments that expose scikit-learn.

    The project uses an isolated Transformer environment on Intel macOS to
    prevent PyTorch and scikit-learn from loading conflicting OpenMP runtimes
    into the same process.
    """

    if importlib.util.find_spec("sklearn") is not None:
        raise FullBertBenchmarkError(
            "scikit-learn is visible. Run with .venv-distilbert/bin/python."
        )


def require_regular_file(file_path: Path, description: str) -> None:
    """Require one existing non-symlink file before it is read."""

    if not file_path.exists():
        raise FullBertBenchmarkError(f"Missing {description}: {file_path}")
    if file_path.is_symlink() or not file_path.is_file():
        raise FullBertBenchmarkError(
            f"Unsafe {description}; expected a regular file: {file_path}"
        )


def calculate_sha256(file_path: Path) -> str:
    """Return the checksum binding saved evidence to one source file."""

    require_regular_file(file_path, "checksum source")
    digest = hashlib.sha256()
    with file_path.open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(file_path: Path, payload: Mapping[str, Any]) -> None:
    """Write one JSON object atomically so partial evidence cannot survive."""

    temporary_path = file_path.with_suffix(file_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary_path.replace(file_path)


def current_peak_rss_mib() -> float:
    """Return peak process RSS normalized to MiB on macOS and Linux."""

    raw_value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    bytes_value = raw_value if sys.platform == "darwin" else raw_value * 1024.0
    return bytes_value / MEBIBYTE


def build_artifact_inventory(directory: Path) -> list[dict[str, Any]]:
    """Checksum every regular saved-model file using relative POSIX paths."""

    if not directory.exists() or not directory.is_dir() or directory.is_symlink():
        raise FullBertBenchmarkError(
            f"Saved model directory is missing or unsafe: {directory}"
        )

    inventory: list[dict[str, Any]] = []
    for file_path in sorted(directory.rglob("*")):
        if file_path.is_symlink():
            raise FullBertBenchmarkError(
                f"Symbolic links are not accepted in model evidence: {file_path}"
            )
        if file_path.is_file():
            inventory.append(
                {
                    "path": file_path.relative_to(directory).as_posix(),
                    "sha256": calculate_sha256(file_path),
                    "size_bytes": file_path.stat().st_size,
                }
            )

    if not inventory:
        raise FullBertBenchmarkError(f"Saved model directory is empty: {directory}")
    return inventory


def enrich_manifest_evidence(
    config: Any,
    baseline_peak_rss_mib: float,
    completed_peak_rss_mib: float,
) -> None:
    """Add measured process memory and model-file checksums after training."""

    manifest_path = Path(config.manifest_file).expanduser().resolve()
    final_model_dir = Path(config.final_model_dir).expanduser().resolve()
    manifest = load_json_object(manifest_path, "full-BERT manifest")
    source_files = require_mapping(manifest, "source_files", "source files")
    for split_name, required_records in EXPECTED_SPLIT_RECORDS.items():
        details = source_files.get(split_name)
        if not isinstance(details, dict):
            raise FullBertBenchmarkError(
                f"Manifest is missing {split_name} source evidence."
            )
        source_path_text = details.get("path")
        if not isinstance(source_path_text, str) or not source_path_text.strip():
            raise FullBertBenchmarkError(
                f"Manifest is missing the {split_name} source path."
            )
        source_path = Path(source_path_text).expanduser().resolve()
        actual_records = line_count(source_path)
        if actual_records != required_records:
            raise FullBertBenchmarkError(
                f"{split_name} split must contain exactly {required_records} "
                f"rows; found {actual_records}."
            )
        details["records"] = actual_records

    manifest["artifact_files"] = build_artifact_inventory(final_model_dir)
    manifest["memory"] = {
        "measurement_method": (
            "resource.getrusage(resource.RUSAGE_SELF).ru_maxrss"
        ),
        "measurement_scope": "full_training_validation_and_test_process",
        "baseline_peak_rss_mib": float(baseline_peak_rss_mib),
        "peak_process_rss_mib": float(completed_peak_rss_mib),
        "incremental_peak_rss_mib": float(
            max(0.0, completed_peak_rss_mib - baseline_peak_rss_mib)
        ),
        "platform_unit_normalization": (
            "bytes_on_macos; kibibytes_on_linux; normalized_to_mib"
        ),
    }
    write_json_atomic(manifest_path, manifest)


def load_json_object(file_path: Path, description: str) -> dict[str, Any]:
    """Load one required UTF-8 JSON object with a precise failure message."""

    require_regular_file(file_path, description)
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FullBertBenchmarkError(
            f"Invalid {description}: {file_path}: {exc}"
        ) from exc

    if not isinstance(payload, dict):
        raise FullBertBenchmarkError(f"{description} must contain an object.")
    return payload


def require_mapping(
    payload: Mapping[str, Any],
    key: str,
    description: str,
) -> Mapping[str, Any]:
    """Require one nested JSON object used by the evidence contract."""

    value = payload.get(key)
    if not isinstance(value, dict):
        raise FullBertBenchmarkError(f"Missing or invalid {description}: {key}")
    return value


def probability_metric(value: Any, metric_name: str) -> float:
    """Require one finite evaluation metric inside the inclusive 0..1 range."""

    if not isinstance(value, (int, float)):
        raise FullBertBenchmarkError(f"Missing numeric metric: {metric_name}")

    metric_value = float(value)
    if not 0.0 <= metric_value <= 1.0:
        raise FullBertBenchmarkError(
            f"Metric {metric_name} is outside 0..1: {metric_value}"
        )
    return metric_value


def line_count(file_path: Path) -> int:
    """Count non-empty JSONL records without loading the complete split."""

    require_regular_file(file_path, "test split")
    with file_path.open("r", encoding="utf-8") as source_file:
        count = sum(1 for line in source_file if line.strip())

    if count <= 0:
        raise FullBertBenchmarkError(f"Test split is empty: {file_path}")
    return count


def resolve_test_file(config: Any) -> Path:
    """
    Resolve the configured untouched test file from a dataclass instance.

    Conventional names are preferred. A semantic fallback accepts one
    unambiguous JSON or JSONL path field containing the word ``test``.
    """

    available_fields = {field.name for field in fields(config)}
    exact_matches = [
        name for name in TEST_FIELD_CANDIDATES if name in available_fields
    ]

    if len(exact_matches) == 1:
        test_path = Path(getattr(config, exact_matches[0]))
        return test_path.expanduser().resolve()

    semantic_matches: list[str] = []
    for field in fields(config):
        value = getattr(config, field.name)
        if not isinstance(value, (str, Path)):
            continue
        path_value = Path(value)
        if "test" in field.name.lower() and path_value.suffix in {".json", ".jsonl"}:
            semantic_matches.append(field.name)

    if len(semantic_matches) != 1:
        raise FullBertBenchmarkError(
            "Could not resolve one test split field. "
            f"Candidates={sorted(semantic_matches)}"
        )

    test_path = Path(getattr(config, semantic_matches[0]))
    return test_path.expanduser().resolve()


def output_paths(config: Any) -> tuple[Path, ...]:
    """Return every dedicated full-BERT path controlled by this runner."""

    return (
        Path(config.checkpoint_dir),
        Path(config.final_model_dir),
        Path(config.metrics_file),
        Path(config.manifest_file),
    )


def protect_or_replace_outputs(config: Any, replace_existing: bool) -> None:
    """Protect previous evidence unless full-BERT replacement is explicit."""

    existing = [path for path in output_paths(config) if path.exists()]

    if existing and not replace_existing:
        formatted = "\n".join(f"- {path}" for path in existing)
        raise FullBertBenchmarkError(
            "Full-BERT outputs already exist. Review them or use "
            f"--replace-existing.\n{formatted}"
        )

    if replace_existing:
        for path in existing:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()


def validate_source_evidence(
    manifest: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Verify train, validation, and test paths, checksums, and row counts."""

    source_files = require_mapping(manifest, "source_files", "source files")
    verified: dict[str, dict[str, Any]] = {}

    for split_name in ("train", "validation", "test"):
        details = source_files.get(split_name)
        if not isinstance(details, dict):
            raise FullBertBenchmarkError(
                f"Manifest is missing {split_name} source evidence."
            )

        path_text = details.get("path")
        expected_checksum = details.get("checksum_sha256")
        expected_records = details.get("records")
        if not isinstance(path_text, str) or not path_text.strip():
            raise FullBertBenchmarkError(
                f"Missing {split_name} source path."
            )
        if not isinstance(expected_checksum, str) or len(expected_checksum) != 64:
            raise FullBertBenchmarkError(
                f"Invalid {split_name} source checksum."
            )
        required_records = EXPECTED_SPLIT_RECORDS[split_name]
        if expected_records != required_records:
            raise FullBertBenchmarkError(
                f"{split_name} must record exactly {required_records} rows; "
                f"found {expected_records!r}."
            )

        source_path = Path(path_text).expanduser().resolve()
        actual_checksum = calculate_sha256(source_path)
        actual_records = line_count(source_path)
        if actual_checksum != expected_checksum:
            raise FullBertBenchmarkError(
                f"{split_name} source checksum changed."
            )
        if actual_records != required_records:
            raise FullBertBenchmarkError(
                f"{split_name} split must contain exactly {required_records} "
                f"rows; found {actual_records}."
            )

        verified[split_name] = {
            "path": str(source_path),
            "checksum_sha256": actual_checksum,
            "records": actual_records,
        }

    return verified


def validate_confusion_matrix(
    metrics: Mapping[str, Any],
    expected_records: int,
) -> list[list[int]]:
    """Require a non-negative 3x3 matrix containing every test record."""

    test_evaluation = require_mapping(
        metrics,
        "test_evaluation",
        "test evaluation",
    )
    matrix = test_evaluation.get("confusion_matrix")

    if not (
        isinstance(matrix, list)
        and len(matrix) == EXPECTED_CLASS_COUNT
        and all(
            isinstance(row, list) and len(row) == EXPECTED_CLASS_COUNT
            for row in matrix
        )
    ):
        raise FullBertBenchmarkError("Metrics do not contain a valid 3x3 matrix.")

    normalized: list[list[int]] = []
    for row in matrix:
        if not all(isinstance(value, int) and value >= 0 for value in row):
            raise FullBertBenchmarkError(
                "Confusion-matrix values must be non-negative integers."
            )
        normalized.append(list(row))

    evaluated_records = sum(sum(row) for row in normalized)
    if evaluated_records != expected_records:
        raise FullBertBenchmarkError(
            f"Expected {expected_records} test records, found {evaluated_records}."
        )
    return normalized


def canonical_label_name(raw_label: Any) -> str | None:
    """Map supported label keys to the approved display labels.

    The existing shared Transformer engine may serialize class keys as title
    case labels, lowercase labels, or numeric class identifiers. This helper
    normalizes those equivalent representations without weakening the fixed
    Bearish/Neutral/Bullish label order.
    """

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


def normalize_label_order(raw_order: Any) -> list[str]:
    """Require one label order that resolves to the approved three classes."""

    if not isinstance(raw_order, list):
        raise FullBertBenchmarkError("Per-class label order is missing.")
    normalized = [canonical_label_name(value) for value in raw_order]
    if normalized != list(LABEL_ORDER):
        raise FullBertBenchmarkError(
            "Per-class label order must be Bearish, Neutral, Bullish."
        )
    return list(LABEL_ORDER)


def calculated_per_class_metrics(
    confusion_matrix: list[list[int]],
) -> dict[str, dict[str, float | int]]:
    """Calculate precision, recall, F1, and support from the saved matrix."""

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


def nested_per_class_candidates(
    evaluation: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    """Normalize supported nested per-class key formats when present."""

    raw_per_class = evaluation.get("per_class")
    if not isinstance(raw_per_class, dict):
        return {}

    normalized: dict[str, Mapping[str, Any]] = {}
    for raw_label, raw_values in raw_per_class.items():
        label_name = canonical_label_name(raw_label)
        if label_name is None or not isinstance(raw_values, dict):
            continue
        if label_name in normalized:
            raise FullBertBenchmarkError(
                f"Duplicate per-class metrics for {label_name}."
            )
        normalized[label_name] = raw_values
    return normalized


def scalar_per_class_candidates(
    test_metrics: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    """Read the shared Trainer's test_bearish/test_neutral/test_bullish keys."""

    normalized: dict[str, Mapping[str, Any]] = {}
    for label_name in LABEL_ORDER:
        prefix = label_name.casefold()
        metric_values = {
            "precision": test_metrics.get(f"test_{prefix}_precision"),
            "recall": test_metrics.get(f"test_{prefix}_recall"),
            "f1": test_metrics.get(f"test_{prefix}_f1"),
        }
        if all(value is not None for value in metric_values.values()):
            normalized[label_name] = metric_values
    return normalized


def validate_per_class_metrics(
    metrics: Mapping[str, Any],
    confusion_matrix: list[list[int]],
) -> dict[str, dict[str, float | int]]:
    """Validate all three class metrics across supported evidence schemas.

    The existing BERT/DistilBERT engine stores scalar per-class metrics under
    ``test_metrics`` and may use lowercase or numeric keys in the nested
    report. LoRA stores title-case nested keys. Both are valid representations
    of the same experiment contract. Values are independently recalculated
    from the confusion matrix and must agree within floating-point tolerance.
    """

    evaluation = require_mapping(metrics, "test_evaluation", "test evaluation")
    normalize_label_order(evaluation.get("label_order"))
    test_metrics = require_mapping(metrics, "test_metrics", "test metrics")

    candidates = nested_per_class_candidates(evaluation)
    if set(candidates) != set(LABEL_ORDER):
        candidates = scalar_per_class_candidates(test_metrics)
    if set(candidates) != set(LABEL_ORDER):
        raise FullBertBenchmarkError(
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
            raise FullBertBenchmarkError(
                f"{label_name} support must be {expected['support']}; "
                f"found {reported_support!r}."
            )

        normalized_values = {
            "precision": probability_metric(
                values.get("precision"),
                f"{label_name}.precision",
            ),
            "recall": probability_metric(
                values.get("recall"),
                f"{label_name}.recall",
            ),
            "f1": probability_metric(
                values.get("f1"),
                f"{label_name}.f1",
            ),
            "support": int(expected["support"]),
        }
        for metric_name in ("precision", "recall", "f1"):
            if abs(
                float(normalized_values[metric_name])
                - float(expected[metric_name])
            ) > tolerance:
                raise FullBertBenchmarkError(
                    f"{label_name} {metric_name} does not match the "
                    "confusion matrix."
                )
        normalized[label_name] = normalized_values
    return normalized


def validate_manifest_artifact_inventory(
    manifest: Mapping[str, Any],
    final_model_dir: Path,
) -> list[dict[str, Any]]:
    """Require the manifest checksum inventory to match every saved file."""

    recorded = manifest.get("artifact_files")
    if not isinstance(recorded, list) or not recorded:
        raise FullBertBenchmarkError(
            "Manifest artifact_files checksum inventory is missing."
        )
    actual = build_artifact_inventory(final_model_dir)
    if recorded != actual:
        raise FullBertBenchmarkError(
            "Saved BERT model files do not match the manifest checksums."
        )
    return actual


def validate_memory_evidence(manifest: Mapping[str, Any]) -> dict[str, float | str]:
    """Require measured peak process memory from the benchmark run."""

    memory = require_mapping(manifest, "memory", "memory evidence")
    peak = memory.get("peak_process_rss_mib")
    baseline = memory.get("baseline_peak_rss_mib")
    incremental = memory.get("incremental_peak_rss_mib")
    for field_name, value in (
        ("peak_process_rss_mib", peak),
        ("baseline_peak_rss_mib", baseline),
        ("incremental_peak_rss_mib", incremental),
    ):
        if not isinstance(value, (int, float)) or float(value) < 0:
            raise FullBertBenchmarkError(
                f"Memory field {field_name} must be non-negative."
            )
    if float(peak) <= 0 or float(peak) < float(baseline):
        raise FullBertBenchmarkError("Measured peak RSS evidence is inconsistent.")
    method = memory.get("measurement_method")
    if not isinstance(method, str) or "ru_maxrss" not in method:
        raise FullBertBenchmarkError("Measured memory method is missing.")
    return {
        "measurement_method": method,
        "peak_process_rss_mib": float(peak),
        "baseline_peak_rss_mib": float(baseline),
        "incremental_peak_rss_mib": float(incremental),
    }


def validate_full_bert_artifacts(config: Any) -> dict[str, Any]:
    """
    Review full-BERT evidence and return the fields needed downstream.

    The validation uses the configured paths rather than assuming a working
    directory. The confusion matrix must cover the exact current test-file
    row count, preserving the experiment's data grain.
    """

    manifest_path = Path(config.manifest_file).expanduser().resolve()
    metrics_path = Path(config.metrics_file).expanduser().resolve()
    final_model_dir = Path(config.final_model_dir).expanduser().resolve()

    manifest = load_json_object(manifest_path, "full-BERT manifest")
    metrics = load_json_object(metrics_path, "full-BERT metrics")

    expected_manifest_values = {
        "status": "trained_and_evaluated",
        "model_id": EXPECTED_MODEL_ID,
        "experiment_name": EXPECTED_EXPERIMENT_NAME,
        "model_family": EXPECTED_MODEL_FAMILY,
        "benchmark_role": EXPECTED_BENCHMARK_ROLE,
    }
    for field_name, expected_value in expected_manifest_values.items():
        actual_value = manifest.get(field_name)
        if actual_value != expected_value:
            raise FullBertBenchmarkError(
                f"Unexpected manifest {field_name}: {actual_value!r}; "
                f"expected {expected_value!r}."
            )

    model_revision = manifest.get("model_revision")
    if not isinstance(model_revision, str) or not model_revision.strip():
        raise FullBertBenchmarkError("Manifest model_revision is missing.")

    recorded_model_dir = manifest.get("final_model_directory")
    if not isinstance(recorded_model_dir, str):
        raise FullBertBenchmarkError("Manifest final_model_directory is missing.")
    if Path(recorded_model_dir).expanduser().resolve() != final_model_dir:
        raise FullBertBenchmarkError(
            "Manifest final_model_directory does not match the configuration."
        )

    parameter_counts = require_mapping(
        manifest,
        "parameter_counts",
        "parameter counts",
    )
    total_parameters = parameter_counts.get("total_parameters")
    trainable_parameters = parameter_counts.get("trainable_parameters")
    if not isinstance(total_parameters, int) or total_parameters <= 0:
        raise FullBertBenchmarkError("total_parameters must be positive.")
    if trainable_parameters != total_parameters:
        raise FullBertBenchmarkError(
            "Full BERT must report every parameter as trainable."
        )

    timing = require_mapping(manifest, "timing", "training timing")
    training_seconds = timing.get("training_seconds")
    if not isinstance(training_seconds, (int, float)) or training_seconds <= 0:
        raise FullBertBenchmarkError("training_seconds must be positive.")

    test_metrics = require_mapping(metrics, "test_metrics", "test metrics")
    test_accuracy = probability_metric(
        test_metrics.get("test_accuracy"),
        "test_accuracy",
    )
    test_macro_f1 = probability_metric(
        test_metrics.get("test_macro_f1"),
        "test_macro_f1",
    )
    test_weighted_f1 = probability_metric(
        test_metrics.get("test_weighted_f1"),
        "test_weighted_f1",
    )

    test_runtime = test_metrics.get("test_runtime")
    if not isinstance(test_runtime, (int, float)) or test_runtime <= 0:
        raise FullBertBenchmarkError("test_runtime must be positive.")

    source_evidence = validate_source_evidence(manifest)
    test_file = resolve_test_file(config)
    expected_records = line_count(test_file)
    if Path(source_evidence["test"]["path"]).resolve() != test_file:
        raise FullBertBenchmarkError(
            "Configured test file does not match the manifest source."
        )
    confusion_matrix = validate_confusion_matrix(metrics, expected_records)
    per_class_metrics = validate_per_class_metrics(metrics, confusion_matrix)

    require_regular_file(final_model_dir / "config.json", "BERT configuration")
    actual_model_files = {
        path.name for path in final_model_dir.iterdir() if path.is_file()
    }
    missing_files = REQUIRED_FINAL_MODEL_FILES - actual_model_files
    if missing_files:
        raise FullBertBenchmarkError(
            "Missing final BERT files: " + ", ".join(sorted(missing_files))
        )

    artifact_files = validate_manifest_artifact_inventory(
        manifest,
        final_model_dir,
    )
    memory = validate_memory_evidence(manifest)

    return {
        "status": manifest["status"],
        "model_id": manifest["model_id"],
        "model_revision": model_revision,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "training_seconds": float(training_seconds),
        "test_records": expected_records,
        "test_accuracy": test_accuracy,
        "test_macro_f1": test_macro_f1,
        "test_weighted_f1": test_weighted_f1,
        "test_runtime": float(test_runtime),
        "confusion_matrix": confusion_matrix,
        "per_class_metrics": per_class_metrics,
        "source_files": source_evidence,
        "artifact_files": artifact_files,
        "memory": memory,
        "final_model_files": sorted(actual_model_files),
    }


def load_bert_contract() -> tuple[type[Any], Any, Any]:
    """Import the existing BERT configuration, validator, and runner lazily."""

    from financial_news_intelligence.models.bert_training import (
        BertTrainingConfig,
        run_bert_training,
        validate_bert_config,
    )

    return BertTrainingConfig, run_bert_training, validate_bert_config


def run_full_bert(
    replace_existing: bool = False,
    verify_only: bool = False,
) -> dict[str, Any]:
    """Train full BERT when requested, then validate every saved artifact."""

    ensure_isolated_environment()
    config_type, training_function, validation_function = load_bert_contract()
    config = config_type()
    validation_function(config)

    if not verify_only:
        protect_or_replace_outputs(config, replace_existing=replace_existing)
        baseline_peak_rss_mib = current_peak_rss_mib()
        training_function(config)
        enrich_manifest_evidence(
            config,
            baseline_peak_rss_mib=baseline_peak_rss_mib,
            completed_peak_rss_mib=current_peak_rss_mib(),
        )

    summary = validate_full_bert_artifacts(config)

    print("Status:", summary["status"])
    print("Model revision:", summary["model_revision"])
    print("Total parameters:", summary["total_parameters"])
    print("Training seconds:", summary["training_seconds"])
    print("Test records:", summary["test_records"])
    print("Test accuracy:", summary["test_accuracy"])
    print("Test macro F1:", summary["test_macro_f1"])
    print("Test weighted F1:", summary["test_weighted_f1"])
    print("Test runtime:", summary["test_runtime"])
    print("Confusion matrix:", summary["confusion_matrix"])
    print("Per-class metrics:", summary["per_class_metrics"])
    print("Peak process RSS MiB:", summary["memory"]["peak_process_rss_mib"])
    print("Final model files:", summary["final_model_files"])
    print("FULL BERT BENCHMARK: PASSED")
    return summary


def parse_arguments() -> argparse.Namespace:
    """Read explicit permissions for replacement or verification-only mode."""

    parser = argparse.ArgumentParser(
        description="Train or verify the full-BERT sentiment benchmark."
    )
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="Replace only the dedicated full-BERT outputs.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Validate existing full-BERT artifacts without training.",
    )
    return parser.parse_args()


def main() -> int:
    """Command-line entry point with one concise actionable failure marker."""

    arguments = parse_arguments()
    if arguments.replace_existing and arguments.verify_only:
        print(
            "FULL BERT BENCHMARK: FAILED: --replace-existing and "
            "--verify-only cannot be combined."
        )
        return 2

    try:
        run_full_bert(
            replace_existing=arguments.replace_existing,
            verify_only=arguments.verify_only,
        )
    except Exception as exc:
        print(
            "FULL BERT BENCHMARK: FAILED: "
            f"{type(exc).__name__}: {exc}"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
