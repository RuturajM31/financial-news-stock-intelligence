#!/usr/bin/env python3
"""
Run and verify the complete BERT-LoRA sentiment experiment.

Purpose
-------
Execute the reusable LoRA training module on the same verified Financial
PhraseBank splits used by DistilBERT and full BERT, then validate adapter,
metrics, timing, parameter-efficiency, and provenance evidence.

Inputs and data journey
-----------------------
``BertTrainingConfig`` supplies the shared source files and comparison rules.
``LoraTrainingConfig`` supplies rank, alpha, dropout, target modules, and
isolated output paths. The module performs schema detection, leakage control,
tokenization, weighted training, validation checkpoint selection, and
untouched test prediction.

Outputs and downstream use
--------------------------
The final adapter, tokenizer, metrics JSON, and manifest are reviewed here and
then consumed by ``run_sentiment_comparison.py``.

Safety and limitations
----------------------
The runner requires the isolated Transformer environment. Existing LoRA
evidence is protected unless ``--replace-existing`` is explicit.
``--verify-only`` never trains. LoRA still requires base BERT for deployment.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Mapping

from financial_news_intelligence.models.lora_training import (
    BASE_MODEL_ID,
    EXPECTED_CLASS_COUNT,
    EXPECTED_SPLIT_RECORDS,
    LABEL_ORDER,
    REQUIRED_ADAPTER_FILES,
    LoraTrainingConfig,
    LoraTrainingError,
    build_artifact_inventory,
    calculate_sha256,
    lora_output_paths,
    run_lora_training,
    validate_lora_config,
)


class LoraArtifactError(RuntimeError):
    """Raised when saved LoRA evidence is incomplete or inconsistent."""


def ensure_isolated_environment() -> None:
    """Reject environments that expose the known scikit-learn OpenMP risk."""

    if importlib.util.find_spec("sklearn") is not None:
        raise LoraArtifactError(
            "scikit-learn is visible. Run with .venv-distilbert/bin/python."
        )


def require_regular_file(file_path: Path, description: str) -> None:
    """Require one existing non-symlink file before reading it."""

    if not file_path.exists():
        raise LoraArtifactError(f"Missing {description}: {file_path}")
    if file_path.is_symlink() or not file_path.is_file():
        raise LoraArtifactError(
            f"Unsafe {description}; expected a regular file: {file_path}"
        )


def load_json_object(file_path: Path, description: str) -> dict[str, Any]:
    """Load one required UTF-8 JSON object."""

    require_regular_file(file_path, description)
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LoraArtifactError(
            f"Invalid {description}: {file_path}: {exc}"
        ) from exc

    if not isinstance(payload, dict):
        raise LoraArtifactError(f"{description} must contain an object.")
    return payload


def require_mapping(
    payload: Mapping[str, Any],
    key: str,
    description: str,
) -> Mapping[str, Any]:
    """Require one nested evidence object."""

    value = payload.get(key)
    if not isinstance(value, dict):
        raise LoraArtifactError(f"Missing or invalid {description}: {key}")
    return value


def probability_metric(value: Any, metric_name: str) -> float:
    """Require one evaluation metric inside the inclusive 0..1 range."""

    if not isinstance(value, (int, float)):
        raise LoraArtifactError(f"Missing numeric metric: {metric_name}")
    metric_value = float(value)
    if not 0.0 <= metric_value <= 1.0:
        raise LoraArtifactError(
            f"Metric {metric_name} is outside 0..1: {metric_value}"
        )
    return metric_value


def line_count(file_path: Path) -> int:
    """Count non-empty JSONL rows for source-evidence validation."""

    require_regular_file(file_path, "source split")
    with file_path.open("r", encoding="utf-8") as source_file:
        count = sum(1 for line in source_file if line.strip())
    if count <= 0:
        raise LoraArtifactError(f"Source split is empty: {file_path}")
    return count


def validate_source_evidence(
    manifest: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Verify train, validation, and test source paths and checksums."""

    source_files = require_mapping(manifest, "source_files", "source files")
    verified: dict[str, dict[str, Any]] = {}

    for split_name in ("train", "validation", "test"):
        details = source_files.get(split_name)
        if not isinstance(details, dict):
            raise LoraArtifactError(
                f"Manifest is missing {split_name} source evidence."
            )

        path_text = details.get("path")
        expected_checksum = details.get("checksum_sha256")
        expected_records = details.get("records")
        if not isinstance(path_text, str) or not path_text.strip():
            raise LoraArtifactError(f"Missing {split_name} source path.")
        if not isinstance(expected_checksum, str) or len(expected_checksum) != 64:
            raise LoraArtifactError(
                f"Invalid {split_name} source checksum."
            )
        required_records = EXPECTED_SPLIT_RECORDS[split_name]
        if expected_records != required_records:
            raise LoraArtifactError(
                f"{split_name} must record exactly {required_records} rows; "
                f"found {expected_records!r}."
            )

        source_path = Path(path_text).expanduser().resolve()
        actual_checksum = calculate_sha256(source_path)
        actual_records = line_count(source_path)
        if actual_checksum != expected_checksum:
            raise LoraArtifactError(f"{split_name} source checksum changed.")
        if actual_records != required_records:
            raise LoraArtifactError(
                f"{split_name} split must contain exactly {required_records} "
                f"rows; found {actual_records}."
            )

        verified[split_name] = {
            "path": str(source_path),
            "checksum_sha256": actual_checksum,
            "records": actual_records,
        }

    return verified


def validate_matrix(
    metrics: Mapping[str, Any],
    expected_records: int,
) -> list[list[int]]:
    """Require a non-negative 3x3 matrix containing every test prediction."""

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
        raise LoraArtifactError("Metrics do not contain a valid 3x3 matrix.")

    normalized: list[list[int]] = []
    for row in matrix:
        if not all(isinstance(value, int) and value >= 0 for value in row):
            raise LoraArtifactError(
                "Confusion-matrix values must be non-negative integers."
            )
        normalized.append(list(row))

    actual_records = sum(sum(row) for row in normalized)
    if actual_records != expected_records:
        raise LoraArtifactError(
            f"Expected {expected_records} test records, found {actual_records}."
        )
    return normalized


def validate_per_class_metrics(
    metrics: Mapping[str, Any],
    confusion_matrix: list[list[int]],
) -> dict[str, dict[str, float | int]]:
    """Validate precision, recall, F1, and support for each sentiment class."""

    evaluation = require_mapping(metrics, "test_evaluation", "test evaluation")
    if evaluation.get("label_order") != list(LABEL_ORDER):
        raise LoraArtifactError(
            "Per-class label order must be Bearish, Neutral, Bullish."
        )
    per_class = evaluation.get("per_class")
    if not isinstance(per_class, dict) or set(per_class) != set(LABEL_ORDER):
        raise LoraArtifactError(
            "Per-class metrics must contain Bearish, Neutral, and Bullish."
        )

    normalized: dict[str, dict[str, float | int]] = {}
    for class_id, label_name in enumerate(LABEL_ORDER):
        values = per_class.get(label_name)
        if not isinstance(values, dict):
            raise LoraArtifactError(f"Missing per-class metrics for {label_name}.")
        expected_support = sum(confusion_matrix[class_id])
        if values.get("support") != expected_support:
            raise LoraArtifactError(
                f"{label_name} support must be {expected_support}; "
                f"found {values.get('support')!r}."
            )
        normalized[label_name] = {
            "precision": probability_metric(
                values.get("precision"),
                f"{label_name}.precision",
            ),
            "recall": probability_metric(
                values.get("recall"),
                f"{label_name}.recall",
            ),
            "f1": probability_metric(values.get("f1"), f"{label_name}.f1"),
            "support": expected_support,
        }
    return normalized


def validate_memory_evidence(manifest: Mapping[str, Any]) -> dict[str, float | str]:
    """Require measured peak process memory from the LoRA run."""

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
            raise LoraArtifactError(
                f"Memory field {field_name} must be non-negative."
            )
    if float(peak) <= 0 or float(peak) < float(baseline):
        raise LoraArtifactError("Measured peak RSS evidence is inconsistent.")
    method = memory.get("measurement_method")
    if not isinstance(method, str) or "ru_maxrss" not in method:
        raise LoraArtifactError("Measured memory method is missing.")
    return {
        "measurement_method": method,
        "peak_process_rss_mib": float(peak),
        "baseline_peak_rss_mib": float(baseline),
        "incremental_peak_rss_mib": float(incremental),
    }


def validate_artifact_inventory(
    manifest: Mapping[str, Any],
    final_adapter_dir: Path,
) -> list[dict[str, Any]]:
    """Require every saved adapter file to match its recorded checksum."""

    recorded = manifest.get("artifact_files")
    if not isinstance(recorded, list) or not recorded:
        raise LoraArtifactError(
            "Manifest artifact_files checksum inventory is missing."
        )
    actual = build_artifact_inventory(final_adapter_dir)
    if recorded != actual:
        raise LoraArtifactError(
            "Saved LoRA adapter files do not match the manifest checksums."
        )
    return actual


def validate_lora_artifacts(
    config: LoraTrainingConfig | None = None,
) -> dict[str, Any]:
    """Review LoRA evidence and return the comparison-ready summary."""

    if config is None:
        config = LoraTrainingConfig()
    validate_lora_config(config)

    manifest = load_json_object(config.manifest_file, "LoRA manifest")
    metrics = load_json_object(config.metrics_file, "LoRA metrics")

    expected_values = {
        "status": "trained_and_evaluated",
        "experiment_name": config.experiment_name,
        "model_family": config.model_family,
        "benchmark_role": config.benchmark_role,
        "model_id": BASE_MODEL_ID,
        "adapter_method": "LoRA",
    }
    for field_name, expected_value in expected_values.items():
        actual_value = manifest.get(field_name)
        if actual_value != expected_value:
            raise LoraArtifactError(
                f"Unexpected manifest {field_name}: {actual_value!r}; "
                f"expected {expected_value!r}."
            )

    model_revision = manifest.get("model_revision")
    if (
        not isinstance(model_revision, str)
        or not model_revision.strip()
        or model_revision == "unresolved"
    ):
        raise LoraArtifactError("A resolved base-model revision is required.")

    recorded_directory = manifest.get("final_model_directory")
    if not isinstance(recorded_directory, str):
        raise LoraArtifactError("Manifest final_model_directory is missing.")
    if Path(recorded_directory).resolve() != config.final_adapter_dir.resolve():
        raise LoraArtifactError(
            "Manifest final_model_directory does not match the LoRA config."
        )

    parameter_counts = require_mapping(
        manifest,
        "parameter_counts",
        "parameter counts",
    )
    total_parameters = parameter_counts.get("total_parameters")
    trainable_parameters = parameter_counts.get("trainable_parameters")
    frozen_parameters = parameter_counts.get("frozen_parameters")

    if not isinstance(total_parameters, int) or total_parameters <= 0:
        raise LoraArtifactError("total_parameters must be positive.")
    if not isinstance(trainable_parameters, int) or trainable_parameters <= 0:
        raise LoraArtifactError("trainable_parameters must be positive.")
    if trainable_parameters >= total_parameters:
        raise LoraArtifactError("LoRA must train fewer than all parameters.")
    if frozen_parameters != total_parameters - trainable_parameters:
        raise LoraArtifactError("frozen_parameters is inconsistent.")

    timing = require_mapping(manifest, "timing", "timing")
    training_seconds = timing.get("training_seconds")
    inference_ms = timing.get("inference_milliseconds_per_record")
    test_records = timing.get("test_records")
    if not isinstance(training_seconds, (int, float)) or training_seconds <= 0:
        raise LoraArtifactError("training_seconds must be positive.")
    if not isinstance(inference_ms, (int, float)) or inference_ms <= 0:
        raise LoraArtifactError("Inference milliseconds must be positive.")
    if not isinstance(test_records, int) or test_records <= 0:
        raise LoraArtifactError("test_records must be a positive integer.")

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
        raise LoraArtifactError("test_runtime must be positive.")

    source_evidence = validate_source_evidence(manifest)
    if source_evidence["test"]["records"] != test_records:
        raise LoraArtifactError(
            "LoRA test source row count does not match timing evidence."
        )
    matrix = validate_matrix(metrics, expected_records=test_records)
    per_class_metrics = validate_per_class_metrics(metrics, matrix)

    require_regular_file(
        config.final_adapter_dir / "adapter_config.json",
        "LoRA adapter configuration",
    )
    actual_files = {
        path.name for path in config.final_adapter_dir.iterdir() if path.is_file()
    }
    missing_files = REQUIRED_ADAPTER_FILES - actual_files
    if missing_files:
        raise LoraArtifactError(
            "Missing final LoRA files: " + ", ".join(sorted(missing_files))
        )

    artifact_files = validate_artifact_inventory(
        manifest,
        config.final_adapter_dir,
    )
    memory = validate_memory_evidence(manifest)

    label_mapping = require_mapping(manifest, "label_mapping", "label mapping")
    recorded_order = label_mapping.get("id_to_label")
    expected_order = {str(index): label for index, label in enumerate(LABEL_ORDER)}
    if recorded_order != expected_order:
        raise LoraArtifactError("LoRA label order does not match the baseline.")

    return {
        "status": manifest["status"],
        "model_revision": model_revision,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "training_seconds": float(training_seconds),
        "test_records": test_records,
        "test_accuracy": test_accuracy,
        "test_macro_f1": test_macro_f1,
        "test_weighted_f1": test_weighted_f1,
        "test_runtime": float(test_runtime),
        "inference_milliseconds_per_record": float(inference_ms),
        "confusion_matrix": matrix,
        "per_class_metrics": per_class_metrics,
        "source_files": source_evidence,
        "artifact_files": artifact_files,
        "memory": memory,
        "final_adapter_files": sorted(actual_files),
    }


def execute_lora(
    replace_existing: bool = False,
    verify_only: bool = False,
) -> dict[str, Any]:
    """Train LoRA when requested, then validate every saved artifact."""

    ensure_isolated_environment()
    config = LoraTrainingConfig()

    if not verify_only:
        run_lora_training(config, replace_existing=replace_existing)

    summary = validate_lora_artifacts(config)
    print("Status:", summary["status"])
    print("Model revision:", summary["model_revision"])
    print("Total parameters:", summary["total_parameters"])
    print("Trainable parameters:", summary["trainable_parameters"])
    print("Training seconds:", summary["training_seconds"])
    print("Test records:", summary["test_records"])
    print("Test accuracy:", summary["test_accuracy"])
    print("Test macro F1:", summary["test_macro_f1"])
    print("Test weighted F1:", summary["test_weighted_f1"])
    print("Test runtime:", summary["test_runtime"])
    print("Confusion matrix:", summary["confusion_matrix"])
    print("Per-class metrics:", summary["per_class_metrics"])
    print("Peak process RSS MiB:", summary["memory"]["peak_process_rss_mib"])
    print("Final adapter files:", summary["final_adapter_files"])
    print("BERT LORA BENCHMARK: PASSED")
    return summary


def parse_arguments() -> argparse.Namespace:
    """Read explicit permissions for replacement or verification-only mode."""

    parser = argparse.ArgumentParser(
        description="Train or verify the BERT-LoRA sentiment benchmark."
    )
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="Replace only dedicated LoRA outputs.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Validate existing LoRA artifacts without training.",
    )
    return parser.parse_args()


def main() -> int:
    """Command-line entry point with one concise failure marker."""

    arguments = parse_arguments()
    if arguments.replace_existing and arguments.verify_only:
        print(
            "BERT LORA BENCHMARK: FAILED: --replace-existing and "
            "--verify-only cannot be combined."
        )
        return 2

    try:
        execute_lora(
            replace_existing=arguments.replace_existing,
            verify_only=arguments.verify_only,
        )
    except Exception as exc:
        print(
            "BERT LORA BENCHMARK: FAILED: "
            f"{type(exc).__name__}: {exc}"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
