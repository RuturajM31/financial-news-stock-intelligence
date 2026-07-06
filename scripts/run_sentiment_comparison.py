#!/usr/bin/env python3
"""
Create and verify the final DistilBERT, BERT, and LoRA comparison.

Purpose
-------
Read the three completed sentiment experiments, confirm identical data splits,
apply the documented quality and deployment rules, and save reproducible
comparison and champion-selection evidence.

Inputs and data journey
-----------------------
``three manifests + three metrics files + three artifact directories`` flow
through strict normalization, split-checksum validation, BERT-revision checks,
deterministic ranking, and atomic JSON output.

Outputs and downstream use
--------------------------
The comparison report supports model cards and technical review. The champion
manifest records the quality winner and deployment recommendation, but it does
not change the API or Streamlit application automatically.

Safety and limitations
----------------------
Existing comparison evidence is protected unless ``--replace-existing`` is
explicit. ``--verify-only`` validates saved outputs without rewriting them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from financial_news_intelligence.models.sentiment_comparison import (
    CHAMPION_MANIFEST_FILE,
    COMPARISON_FILE,
    DEPLOYMENT_QUALITY_TOLERANCE,
    EXPECTED_SPLIT_RECORDS,
    LABEL_ORDER,
    SentimentComparisonError,
    build_comparison,
)


class ComparisonArtifactError(RuntimeError):
    """Raised when saved comparison or champion evidence is invalid."""


def require_regular_file(file_path: Path, description: str) -> None:
    """Require one existing non-symlink file before reading it."""

    if not file_path.exists():
        raise ComparisonArtifactError(f"Missing {description}: {file_path}")
    if file_path.is_symlink() or not file_path.is_file():
        raise ComparisonArtifactError(
            f"Unsafe {description}; expected a regular file: {file_path}"
        )


def load_json_object(file_path: Path, description: str) -> dict[str, Any]:
    """Load one required UTF-8 JSON object."""

    require_regular_file(file_path, description)
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ComparisonArtifactError(
            f"Invalid {description}: {file_path}: {exc}"
        ) from exc

    if not isinstance(payload, dict):
        raise ComparisonArtifactError(f"{description} must contain an object.")
    return payload


def require_mapping(
    payload: Mapping[str, Any],
    key: str,
    description: str,
) -> Mapping[str, Any]:
    """Require one nested JSON object."""

    value = payload.get(key)
    if not isinstance(value, dict):
        raise ComparisonArtifactError(
            f"Missing or invalid {description}: {key}"
        )
    return value


def protect_outputs(replace_existing: bool) -> None:
    """Protect previous decision evidence unless replacement is explicit."""

    controlled_paths = (COMPARISON_FILE, CHAMPION_MANIFEST_FILE)
    existing = [path for path in controlled_paths if path.exists()]

    if existing and not replace_existing:
        formatted = "\n".join(f"- {path}" for path in existing)
        raise ComparisonArtifactError(
            "Comparison outputs already exist. Review them or use "
            f"--replace-existing.\n{formatted}"
        )

    if replace_existing:
        for path in existing:
            path.unlink()


def validate_comparison_outputs() -> dict[str, Any]:
    """Review saved ranking and champion evidence for internal consistency."""

    comparison = load_json_object(COMPARISON_FILE, "comparison report")
    champion = load_json_object(CHAMPION_MANIFEST_FILE, "champion manifest")

    if comparison.get("status") != "comparison_completed":
        raise ComparisonArtifactError("Comparison status is not complete.")
    if champion.get("status") != "champion_selected":
        raise ComparisonArtifactError("Champion status is not selected.")

    if comparison.get("label_order") != list(LABEL_ORDER):
        raise ComparisonArtifactError("Comparison label order is incorrect.")
    if champion.get("label_order") != list(LABEL_ORDER):
        raise ComparisonArtifactError("Champion label order is incorrect.")

    models = comparison.get("models")
    if not isinstance(models, list) or len(models) != 3:
        raise ComparisonArtifactError("Comparison must contain three models.")

    model_keys = [model.get("model_key") for model in models if isinstance(model, dict)]
    if set(model_keys) != {"distilbert", "bert", "bert_lora"}:
        raise ComparisonArtifactError(f"Unexpected comparison models: {model_keys}")

    ranking = comparison.get("quality_ranking")
    if not isinstance(ranking, list) or set(ranking) != set(model_keys):
        raise ComparisonArtifactError("Quality ranking is incomplete.")

    quality_champion = comparison.get("quality_champion")
    deployment_champion = comparison.get("deployment_champion")
    if quality_champion != ranking[0]:
        raise ComparisonArtifactError("Quality champion is not rank one.")
    if champion.get("official_quality_champion") != quality_champion:
        raise ComparisonArtifactError("Champion files disagree on quality winner.")
    if champion.get("recommended_deployment_model") != deployment_champion:
        raise ComparisonArtifactError(
            "Champion files disagree on deployment recommendation."
        )

    test_records = comparison.get("test_records")
    if test_records != EXPECTED_SPLIT_RECORDS["test"]:
        raise ComparisonArtifactError(
            "Comparison test_records must equal the approved 518-row split."
        )
    if champion.get("test_records") != test_records:
        raise ComparisonArtifactError("Champion test grain does not match.")

    dataset_sources = require_mapping(
        comparison,
        "dataset_sources",
        "dataset sources",
    )
    if set(dataset_sources) != {"train", "validation", "test"}:
        raise ComparisonArtifactError("Comparison dataset sources are incomplete.")

    for split_name, source_details in dataset_sources.items():
        if not isinstance(source_details, dict):
            raise ComparisonArtifactError(
                f"Invalid {split_name} dataset source evidence."
            )
        checksum = source_details.get("checksum_sha256")
        records = source_details.get("records")
        if not isinstance(checksum, str) or len(checksum) != 64:
            raise ComparisonArtifactError(
                f"Invalid {split_name} dataset checksum."
            )
        required_records = EXPECTED_SPLIT_RECORDS[split_name]
        if records != required_records:
            raise ComparisonArtifactError(
                f"{split_name} dataset must contain {required_records} rows."
            )

    for model in models:
        model_sources = model.get("source_files")
        if not isinstance(model_sources, dict):
            raise ComparisonArtifactError("A model is missing source files.")
        for split_name, expected_source in dataset_sources.items():
            model_source = model_sources.get(split_name)
            if not isinstance(model_source, dict):
                raise ComparisonArtifactError(
                    f"A model is missing {split_name} source evidence."
                )
            for field_name in ("checksum_sha256", "records"):
                if model_source.get(field_name) != expected_source.get(field_name):
                    raise ComparisonArtifactError(
                        f"Model {split_name} source evidence does not match."
                    )

    for model in models:
        confusion_matrix = model.get("confusion_matrix")
        if not (
            isinstance(confusion_matrix, list)
            and len(confusion_matrix) == 3
            and all(isinstance(row, list) and len(row) == 3 for row in confusion_matrix)
        ):
            raise ComparisonArtifactError("A model has an invalid confusion matrix.")

        per_class = model.get("per_class_metrics")
        if not isinstance(per_class, dict) or set(per_class) != set(LABEL_ORDER):
            raise ComparisonArtifactError("A model has incomplete per-class metrics.")
        for class_id, label_name in enumerate(LABEL_ORDER):
            class_metrics = per_class.get(label_name)
            if not isinstance(class_metrics, dict):
                raise ComparisonArtifactError(
                    f"A model is missing {label_name} class metrics."
                )
            if class_metrics.get("support") != sum(confusion_matrix[class_id]):
                raise ComparisonArtifactError(
                    f"A model has inconsistent {label_name} support."
                )
            for metric_name in ("precision", "recall", "f1"):
                value = class_metrics.get(metric_name)
                if not isinstance(value, (int, float)) or not 0 <= value <= 1:
                    raise ComparisonArtifactError(
                        f"A model has invalid {label_name} {metric_name}."
                    )

        artifact_files = model.get("artifact_files")
        if not isinstance(artifact_files, list) or not artifact_files:
            raise ComparisonArtifactError("A model has no artifact checksums.")
        for artifact in artifact_files:
            if not isinstance(artifact, dict):
                raise ComparisonArtifactError("Invalid artifact checksum entry.")
            checksum = artifact.get("sha256")
            size_bytes = artifact.get("size_bytes")
            relative_path = artifact.get("path")
            if not isinstance(relative_path, str) or not relative_path:
                raise ComparisonArtifactError("Artifact path is missing.")
            if not isinstance(checksum, str) or len(checksum) != 64:
                raise ComparisonArtifactError("Artifact checksum is invalid.")
            if not isinstance(size_bytes, int) or size_bytes < 0:
                raise ComparisonArtifactError("Artifact byte size is invalid.")

        memory_value = model.get("comparison_memory_mib")
        memory_source = model.get("comparison_memory_source")
        if not isinstance(memory_value, (int, float)) or memory_value <= 0:
            raise ComparisonArtifactError("A model has invalid memory evidence.")
        if memory_source not in {
            "measured_peak_process_rss_all_models",
            "fp32_parameter_estimate_consistent_fallback",
        }:
            raise ComparisonArtifactError("A model has unknown memory provenance.")
        if model.get("model_key") in {"bert", "bert_lora"}:
            measured_peak = model.get("measured_peak_process_rss_mib")
            if not isinstance(measured_peak, (int, float)) or measured_peak <= 0:
                raise ComparisonArtifactError(
                    "BERT and LoRA require measured peak RSS evidence."
                )

    memory_sources = {model["comparison_memory_source"] for model in models}
    if len(memory_sources) != 1:
        raise ComparisonArtifactError(
            "All models must use one common comparison-memory basis."
        )

    models_by_key = {model["model_key"]: model for model in models}
    bert = models_by_key["bert"]
    lora = models_by_key["bert_lora"]
    if bert.get("model_id") != lora.get("model_id"):
        raise ComparisonArtifactError("BERT base model IDs do not match.")
    if bert.get("model_revision") != lora.get("model_revision"):
        raise ComparisonArtifactError("BERT base model revisions do not match.")

    tolerance = comparison.get("deployment_quality_tolerance_macro_f1")
    if tolerance != DEPLOYMENT_QUALITY_TOLERANCE:
        raise ComparisonArtifactError("Deployment tolerance is incorrect.")

    source_evidence = require_mapping(
        champion,
        "source_evidence",
        "source evidence",
    )
    if set(source_evidence) != {"distilbert", "bert", "bert_lora"}:
        raise ComparisonArtifactError("Champion source evidence is incomplete.")
    for model_key, evidence in source_evidence.items():
        if not isinstance(evidence, dict):
            raise ComparisonArtifactError(
                f"Champion evidence is invalid for {model_key}."
            )
        for checksum_name in ("manifest_sha256", "metrics_sha256"):
            checksum = evidence.get(checksum_name)
            if not isinstance(checksum, str) or len(checksum) != 64:
                raise ComparisonArtifactError(
                    f"Champion {model_key} {checksum_name} is invalid."
                )
        artifact_files = evidence.get("artifact_files")
        if not isinstance(artifact_files, list) or not artifact_files:
            raise ComparisonArtifactError(
                f"Champion {model_key} artifact checksums are missing."
            )
        memory_source = evidence.get("comparison_memory_source")
        if memory_source not in {
            "measured_peak_process_rss_all_models",
            "fp32_parameter_estimate_consistent_fallback",
        }:
            raise ComparisonArtifactError(
                f"Champion {model_key} memory provenance is invalid."
            )

    if champion.get("automatic_deployment_change") is not False:
        raise ComparisonArtifactError(
            "Comparison must not change deployment automatically."
        )

    return {
        "test_records": test_records,
        "quality_ranking": ranking,
        "quality_champion": quality_champion,
        "deployment_champion": deployment_champion,
        "models": models,
    }


def execute_comparison(
    replace_existing: bool = False,
    verify_only: bool = False,
) -> dict[str, Any]:
    """Create decision evidence when requested, then validate both outputs."""

    if not verify_only:
        protect_outputs(replace_existing=replace_existing)
        build_comparison()

    summary = validate_comparison_outputs()

    print("Test records:", summary["test_records"])
    print("Quality ranking:", summary["quality_ranking"])
    print("Quality champion:", summary["quality_champion"])
    print("Deployment champion:", summary["deployment_champion"])
    for model in summary["models"]:
        print(
            "Model:",
            model["model_key"],
            "macro_f1=",
            round(model["test_macro_f1"], 6),
            "weighted_f1=",
            round(model["test_weighted_f1"], 6),
            "accuracy=",
            round(model["test_accuracy"], 6),
            "trainable=",
            model["trainable_parameters"],
            "latency_ms=",
            round(model["inference_milliseconds_per_record"], 6),
            "memory_mib=",
            round(model["comparison_memory_mib"], 3),
            "memory_source=",
            model["comparison_memory_source"],
        )
    print("SENTIMENT MODEL COMPARISON: PASSED")
    return summary


def parse_arguments() -> argparse.Namespace:
    """Read explicit permissions for replacement or verification-only mode."""

    parser = argparse.ArgumentParser(
        description="Create or verify the sentiment-model comparison."
    )
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="Replace only comparison and champion JSON files.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Validate existing comparison outputs without rewriting them.",
    )
    return parser.parse_args()


def main() -> int:
    """Command-line entry point with one concise failure marker."""

    arguments = parse_arguments()
    if arguments.replace_existing and arguments.verify_only:
        print(
            "SENTIMENT MODEL COMPARISON: FAILED: --replace-existing and "
            "--verify-only cannot be combined."
        )
        return 2

    try:
        execute_comparison(
            replace_existing=arguments.replace_existing,
            verify_only=arguments.verify_only,
        )
    except (SentimentComparisonError, ComparisonArtifactError) as exc:
        print(
            "SENTIMENT MODEL COMPARISON: FAILED: "
            f"{type(exc).__name__}: {exc}"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
