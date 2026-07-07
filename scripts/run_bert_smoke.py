#!/usr/bin/env python3
"""
Run a controlled full-BERT smoke experiment on balanced real data.

Purpose
-------
Prove that the approved BERT wrapper, verified Financial PhraseBank files,
shared Transformer engine, model download, tokenization, training,
evaluation, and artifact saving work together before the expensive full run.

Inputs
------
The runner reads the train, validation, and untouched test JSONL files already
configured by ``BertTrainingConfig``. It keeps every original record field and
selects a deterministic balanced sample from each split:

- training: 9 records per class, 27 records total;
- validation: 3 records per class, 9 records total;
- test: 3 records per class, 9 records total.

Processing
----------
1. Confirm the isolated Transformer environment does not expose scikit-learn.
2. Discover the three configured split-path fields without hard-coding a
   private implementation detail from the shared engine.
3. Read and validate the JSONL records.
4. Detect the three-class label field and build deterministic balanced files.
5. Create a BERT smoke configuration with one epoch and isolated outputs.
6. Call the verified BERT wrapper and shared training engine.
7. Validate the manifest, metrics, model, tokenizer, and evaluated row count.

Outputs
-------
- ``data/interim/bert_smoke`` balanced JSONL files;
- ``artifacts/models/bert_smoke`` checkpoints and final model;
- ``reports/metrics/bert_smoke_metrics.json``;
- ``artifacts/manifests/bert_smoke_manifest.json``.

Safety and limitations
----------------------
The runner never changes the source dataset or the completed DistilBERT and
full-BERT artifact locations. Existing BERT smoke outputs are protected by
default. ``--replace-existing`` must be supplied explicitly to replace only
the dedicated smoke outputs. Smoke metrics prove pipeline health, not model
quality, and must never be used for champion selection.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# Disable tokenizer worker forking before Transformers is imported. This keeps
# the tiny CPU smoke run deterministic and avoids noisy fork warnings.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from financial_news_intelligence.models.bert_training import (  # noqa: E402
    BERT_MODEL_ID,
    BertTrainingConfig,
    run_bert_training,
    validate_bert_config,
)


# ============================================================
# 1. CONTROLLED SMOKE SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SMOKE_DATA_DIR = PROJECT_ROOT / "data" / "interim" / "bert_smoke"
SMOKE_MODEL_ROOT = PROJECT_ROOT / "artifacts" / "models" / "bert_smoke"
SMOKE_CHECKPOINT_DIR = SMOKE_MODEL_ROOT / "checkpoints"
SMOKE_FINAL_MODEL_DIR = SMOKE_MODEL_ROOT / "final_model"
SMOKE_METRICS_FILE = (
    PROJECT_ROOT / "reports" / "metrics" / "bert_smoke_metrics.json"
)
SMOKE_MANIFEST_FILE = (
    PROJECT_ROOT / "artifacts" / "manifests" / "bert_smoke_manifest.json"
)

TRAIN_RECORDS_PER_CLASS = 9
VALIDATION_RECORDS_PER_CLASS = 3
TEST_RECORDS_PER_CLASS = 3
SMOKE_EPOCHS = 1.0
SMOKE_RANDOM_SEED = 42
EXPECTED_CLASS_COUNT = 3

SPLIT_SAMPLE_SIZES = {
    "train": TRAIN_RECORDS_PER_CLASS,
    "validation": VALIDATION_RECORDS_PER_CLASS,
    "test": TEST_RECORDS_PER_CLASS,
}

SPLIT_FILE_NAMES = {
    "train": "financial_phrasebank_train_smoke.jsonl",
    "validation": "financial_phrasebank_validation_smoke.jsonl",
    "test": "financial_phrasebank_test_smoke.jsonl",
}

REQUIRED_FINAL_MODEL_FILES = {
    "config.json",
    "model.safetensors",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.txt",
}

# Explicit names are tried first. The fallback resolver supports equivalent
# Path fields while still rejecting ambiguous configurations.
SPLIT_FIELD_CANDIDATES = {
    "train": (
        "train_file",
        "train_path",
        "train_data_file",
        "training_file",
        "training_path",
    ),
    "validation": (
        "validation_file",
        "validation_path",
        "validation_data_file",
        "valid_file",
        "valid_path",
    ),
    "test": (
        "test_file",
        "test_path",
        "test_data_file",
    ),
}

LABEL_FIELD_CANDIDATES = (
    "label_id",
    "label",
    "label_name",
    "sentiment_label",
    "sentiment",
)


# ============================================================
# 2. SMALL VALIDATION HELPERS
# ============================================================

class BertSmokeError(RuntimeError):
    """Raised when smoke preparation or artifact validation is unsafe."""


def ensure_isolated_environment() -> None:
    """
    Reject environments that expose scikit-learn.

    The verified Transformer environment intentionally excludes scikit-learn
    so PyTorch and scikit-learn cannot load conflicting OpenMP runtimes in the
    same process on this Intel macOS machine.
    """

    if importlib.util.find_spec("sklearn") is not None:
        raise BertSmokeError(
            "scikit-learn is visible in this environment. Run the smoke "
            "experiment with .venv-distilbert/bin/python."
        )


def require_regular_file(file_path: Path, description: str) -> None:
    """Require one existing non-symlink file before reading it."""

    if not file_path.exists():
        raise BertSmokeError(f"Missing {description}: {file_path}")

    if file_path.is_symlink() or not file_path.is_file():
        raise BertSmokeError(
            f"Unsafe {description}; expected a regular file: {file_path}"
        )


def scalar_label(value: Any) -> bool:
    """Return whether a JSON value is suitable as a class label."""

    if isinstance(value, (str, int, bool)):
        return True

    return isinstance(value, float) and value.is_integer()


def label_sort_key(value: Any) -> tuple[str, str]:
    """Create a deterministic ordering for string and numeric labels."""

    return type(value).__name__, str(value)


# ============================================================
# 3. CONFIGURATION FIELD DISCOVERY
# ============================================================


def path_like_config_fields(config: BertTrainingConfig) -> dict[str, Path]:
    """Return dataclass fields whose current values behave like file paths."""

    discovered: dict[str, Path] = {}

    for field in fields(config):
        value = getattr(config, field.name)
        if isinstance(value, (str, Path)):
            path_value = Path(value)
            if path_value.suffix.lower() in {".jsonl", ".json"}:
                discovered[field.name] = path_value

    return discovered


def resolve_split_field_names(
    config: BertTrainingConfig,
) -> dict[str, str]:
    """
    Map train, validation, and test to their dataclass field names.

    Exact conventional names are preferred. A semantic fallback accepts one
    unambiguous JSON/JSONL Path field containing the split word. Ambiguous or
    missing mappings fail before any data or artifact is changed.
    """

    available_fields = {field.name for field in fields(config)}
    path_fields = path_like_config_fields(config)
    resolved: dict[str, str] = {}

    for split_name, candidates in SPLIT_FIELD_CANDIDATES.items():
        exact_matches = [name for name in candidates if name in available_fields]

        if len(exact_matches) == 1:
            resolved[split_name] = exact_matches[0]
            continue

        semantic_matches = [
            name
            for name in path_fields
            if split_name in name.lower()
            or (split_name == "validation" and "valid" in name.lower())
        ]

        if len(semantic_matches) != 1:
            raise BertSmokeError(
                f"Could not resolve one {split_name} split field. "
                f"Candidates={sorted(semantic_matches)}"
            )

        resolved[split_name] = semantic_matches[0]

    if len(set(resolved.values())) != len(resolved):
        raise BertSmokeError(f"Split fields are not distinct: {resolved}")

    return resolved


def source_split_paths(
    config: BertTrainingConfig,
    split_fields: Mapping[str, str],
) -> dict[str, Path]:
    """Read and validate the configured source path for each split."""

    paths: dict[str, Path] = {}

    for split_name, field_name in split_fields.items():
        source_path = Path(getattr(config, field_name)).expanduser().resolve()
        require_regular_file(source_path, f"{split_name} source split")
        paths[split_name] = source_path

    return paths


# ============================================================
# 4. JSONL AND BALANCED-SAMPLE LOGIC
# ============================================================


def read_jsonl(file_path: Path) -> list[dict[str, Any]]:
    """Load non-empty JSONL objects and report the exact broken line."""

    require_regular_file(file_path, "JSONL input")
    records: list[dict[str, Any]] = []

    with file_path.open("r", encoding="utf-8") as source_file:
        for line_number, raw_line in enumerate(source_file, start=1):
            text = raw_line.strip()
            if not text:
                continue

            try:
                record = json.loads(text)
            except json.JSONDecodeError as exc:
                raise BertSmokeError(
                    f"Invalid JSON in {file_path} at line {line_number}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise BertSmokeError(
                    f"Expected an object in {file_path} at line {line_number}."
                )

            records.append(record)

    if not records:
        raise BertSmokeError(f"JSONL file contains no records: {file_path}")

    return records


def detect_label_field(records: Sequence[Mapping[str, Any]]) -> str:
    """
    Find one scalar field containing exactly the three sentiment classes.

    Preferred label names are checked in order. This supports integer labels,
    text labels, or records that contain both while remaining deterministic.
    """

    if not records:
        raise BertSmokeError("Cannot detect a label field in an empty dataset.")

    common_fields = set(records[0])
    for record in records[1:]:
        common_fields.intersection_update(record)

    ordered_candidates = [
        name for name in LABEL_FIELD_CANDIDATES if name in common_fields
    ]
    ordered_candidates.extend(
        sorted(
            name
            for name in common_fields
            if "label" in name.lower() and name not in ordered_candidates
        )
    )

    for field_name in ordered_candidates:
        values = [record[field_name] for record in records]
        if not all(scalar_label(value) for value in values):
            continue

        unique_values = {value for value in values}
        if len(unique_values) == EXPECTED_CLASS_COUNT:
            return field_name

    raise BertSmokeError(
        "No common scalar field with exactly three classes was found."
    )


def balanced_sample(
    records: Sequence[dict[str, Any]],
    label_field: str,
    records_per_class: int,
    random_seed: int,
) -> list[dict[str, Any]]:
    """
    Select the same requested count from each class deterministically.

    The source objects are copied intact. No text, label, provenance, checksum,
    or identifier field is altered.
    """

    if records_per_class <= 0:
        raise BertSmokeError("records_per_class must be positive.")

    grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)

    for record in records:
        if label_field not in record:
            raise BertSmokeError(
                f"Record is missing detected label field: {label_field}"
            )
        grouped[record[label_field]].append(dict(record))

    if len(grouped) != EXPECTED_CLASS_COUNT:
        raise BertSmokeError(
            f"Expected {EXPECTED_CLASS_COUNT} classes, found {len(grouped)}."
        )

    selected: list[dict[str, Any]] = []

    for class_index, label_value in enumerate(
        sorted(grouped, key=label_sort_key)
    ):
        class_records = list(grouped[label_value])
        if len(class_records) < records_per_class:
            raise BertSmokeError(
                f"Class {label_value!r} has {len(class_records)} records; "
                f"{records_per_class} are required."
            )

        class_random = random.Random(random_seed + class_index)
        class_random.shuffle(class_records)
        selected.extend(class_records[:records_per_class])

    final_random = random.Random(random_seed)
    final_random.shuffle(selected)
    return selected


def write_jsonl(records: Iterable[Mapping[str, Any]], file_path: Path) -> None:
    """Write UTF-8 JSONL atomically so partial files never look complete."""

    file_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = file_path.with_suffix(file_path.suffix + ".tmp")

    with temporary_path.open("w", encoding="utf-8", newline="\n") as target:
        for record in records:
            target.write(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            )

    temporary_path.replace(file_path)


def prepare_balanced_smoke_files(
    source_paths: Mapping[str, Path],
) -> tuple[dict[str, Path], dict[str, dict[str, int]], dict[str, str]]:
    """
    Build all balanced files and return paths, class counts, and label fields.

    All source splits must expose the same detected label field. This protects
    the shared training engine from receiving different schemas across train,
    validation, and test. No output file is written until all three inputs pass
    that validation.
    """

    split_order = ("train", "validation", "test")
    loaded_records = {
        split_name: read_jsonl(source_paths[split_name])
        for split_name in split_order
    }
    label_fields = {
        split_name: detect_label_field(loaded_records[split_name])
        for split_name in split_order
    }

    if len(set(label_fields.values())) != 1:
        raise BertSmokeError(
            "Dataset splits use inconsistent label fields: "
            f"{label_fields}"
        )

    smoke_paths: dict[str, Path] = {}
    split_counts: dict[str, dict[str, int]] = {}

    for split_index, split_name in enumerate(split_order):
        label_field = label_fields[split_name]
        selected = balanced_sample(
            records=loaded_records[split_name],
            label_field=label_field,
            records_per_class=SPLIT_SAMPLE_SIZES[split_name],
            random_seed=SMOKE_RANDOM_SEED + split_index * 100,
        )

        smoke_path = SMOKE_DATA_DIR / SPLIT_FILE_NAMES[split_name]
        write_jsonl(selected, smoke_path)

        smoke_paths[split_name] = smoke_path
        split_counts[split_name] = {
            str(label): int(count)
            for label, count in sorted(
                Counter(record[label_field] for record in selected).items(),
                key=lambda item: label_sort_key(item[0]),
            )
        }

    return smoke_paths, split_counts, label_fields


# ============================================================
# 5. SMOKE CONFIGURATION
# ============================================================


def build_smoke_config(
    base_config: BertTrainingConfig,
    split_fields: Mapping[str, str],
    smoke_paths: Mapping[str, Path],
) -> BertTrainingConfig:
    """
    Create a one-epoch BERT config with isolated data and artifact paths.

    All settings not listed below remain identical to the approved full-BERT
    configuration, preserving the real production code path during the smoke
    experiment.
    """

    overrides: dict[str, Any] = {
        "experiment_name": "BERT Financial Sentiment Smoke",
        "benchmark_role": "smoke_pipeline_validation",
        "number_of_epochs": SMOKE_EPOCHS,
        "checkpoint_dir": SMOKE_CHECKPOINT_DIR,
        "final_model_dir": SMOKE_FINAL_MODEL_DIR,
        "metrics_file": SMOKE_METRICS_FILE,
        "manifest_file": SMOKE_MANIFEST_FILE,
        "run_name": "bert_financial_phrasebank_smoke",
    }

    for split_name, field_name in split_fields.items():
        overrides[field_name] = smoke_paths[split_name]

    smoke_config = replace(base_config, **overrides)
    validate_bert_config(smoke_config)
    return smoke_config


# ============================================================
# 6. OUTPUT PROTECTION AND ARTIFACT REVIEW
# ============================================================


def smoke_output_paths() -> tuple[Path, ...]:
    """Return every dedicated path this smoke run is allowed to replace."""

    return (
        SMOKE_DATA_DIR,
        SMOKE_MODEL_ROOT,
        SMOKE_METRICS_FILE,
        SMOKE_MANIFEST_FILE,
    )


def protect_or_replace_outputs(replace_existing: bool) -> None:
    """Refuse existing evidence unless explicit smoke-only replacement is set."""

    existing = [path for path in smoke_output_paths() if path.exists()]

    if existing and not replace_existing:
        formatted = "\n".join(f"- {path}" for path in existing)
        raise BertSmokeError(
            "BERT smoke outputs already exist. Review them or rerun with "
            f"--replace-existing.\n{formatted}"
        )

    if replace_existing:
        for path in existing:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()


def load_json_object(file_path: Path, description: str) -> dict[str, Any]:
    """Load one required UTF-8 JSON object."""

    require_regular_file(file_path, description)

    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BertSmokeError(f"Invalid {description}: {file_path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise BertSmokeError(f"{description} must contain a JSON object.")

    return payload


def require_nested_mapping(
    payload: Mapping[str, Any],
    key: str,
    description: str,
) -> Mapping[str, Any]:
    """Require one nested JSON object used by the evidence contract."""

    value = payload.get(key)
    if not isinstance(value, dict):
        raise BertSmokeError(f"Missing or invalid {description}: {key}")
    return value


def validate_probability_metric(value: Any, metric_name: str) -> float:
    """Require one finite evaluation metric inside the inclusive 0..1 range."""

    if not isinstance(value, (int, float)):
        raise BertSmokeError(f"Missing numeric metric: {metric_name}")

    metric_value = float(value)
    if not 0.0 <= metric_value <= 1.0:
        raise BertSmokeError(
            f"Metric {metric_name} is outside 0..1: {metric_value}"
        )
    return metric_value


def validate_confusion_matrix(metrics: Mapping[str, Any]) -> list[list[int]]:
    """Require the test 3x3 matrix containing exactly nine evaluated records."""

    test_evaluation = require_nested_mapping(
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
        raise BertSmokeError("Metrics do not contain a valid 3x3 test matrix.")

    normalized: list[list[int]] = []
    for row in matrix:
        if not all(isinstance(value, int) and value >= 0 for value in row):
            raise BertSmokeError(
                "Confusion-matrix values must be non-negative integers."
            )
        normalized.append(list(row))

    expected_test_records = TEST_RECORDS_PER_CLASS * EXPECTED_CLASS_COUNT
    actual_test_records = sum(sum(row) for row in normalized)

    if actual_test_records != expected_test_records:
        raise BertSmokeError(
            f"Expected {expected_test_records} evaluated test records, "
            f"found {actual_test_records}."
        )

    return normalized


def validate_smoke_artifacts() -> dict[str, Any]:
    """Validate saved evidence and return a concise reviewed summary."""

    manifest = load_json_object(SMOKE_MANIFEST_FILE, "smoke manifest")
    metrics = load_json_object(SMOKE_METRICS_FILE, "smoke metrics")

    expected_manifest_values = {
        "status": "trained_and_evaluated",
        "model_id": BERT_MODEL_ID,
        "experiment_name": "BERT Financial Sentiment Smoke",
        "model_family": "BERT",
        "benchmark_role": "smoke_pipeline_validation",
    }
    for field_name, expected_value in expected_manifest_values.items():
        actual_value = manifest.get(field_name)
        if actual_value != expected_value:
            raise BertSmokeError(
                f"Unexpected manifest {field_name}: {actual_value!r}; "
                f"expected {expected_value!r}."
            )

    model_revision = manifest.get("model_revision")
    if not isinstance(model_revision, str) or not model_revision.strip():
        raise BertSmokeError("Manifest model_revision is missing.")

    recorded_model_directory = manifest.get("final_model_directory")
    if not isinstance(recorded_model_directory, str):
        raise BertSmokeError("Manifest final_model_directory is missing.")
    if Path(recorded_model_directory).resolve() != SMOKE_FINAL_MODEL_DIR.resolve():
        raise BertSmokeError(
            "Manifest final_model_directory does not match the smoke path."
        )

    parameter_counts = require_nested_mapping(
        manifest,
        "parameter_counts",
        "parameter counts",
    )
    total_parameters = parameter_counts.get("total_parameters")
    if not isinstance(total_parameters, int) or total_parameters <= 0:
        raise BertSmokeError("Manifest total_parameters must be positive.")

    timing = require_nested_mapping(manifest, "timing", "training timing")
    training_seconds = timing.get("training_seconds")
    if not isinstance(training_seconds, (int, float)) or training_seconds <= 0:
        raise BertSmokeError("Manifest training_seconds must be positive.")

    test_metrics = require_nested_mapping(metrics, "test_metrics", "test metrics")
    test_accuracy = validate_probability_metric(
        test_metrics.get("test_accuracy"),
        "test_accuracy",
    )
    test_macro_f1 = validate_probability_metric(
        test_metrics.get("test_macro_f1"),
        "test_macro_f1",
    )

    require_regular_file(
        SMOKE_FINAL_MODEL_DIR / "config.json",
        "final BERT smoke model configuration",
    )
    actual_model_files = {
        path.name
        for path in SMOKE_FINAL_MODEL_DIR.iterdir()
        if path.is_file()
    }
    missing_model_files = REQUIRED_FINAL_MODEL_FILES - actual_model_files
    if missing_model_files:
        raise BertSmokeError(
            "Missing final BERT smoke files: "
            + ", ".join(sorted(missing_model_files))
        )

    confusion_matrix = validate_confusion_matrix(metrics)

    return {
        "status": manifest["status"],
        "model_id": manifest["model_id"],
        "model_revision": model_revision,
        "total_parameters": total_parameters,
        "training_seconds": float(training_seconds),
        "test_accuracy": test_accuracy,
        "test_macro_f1": test_macro_f1,
        "confusion_matrix": confusion_matrix,
        "final_model_files": sorted(actual_model_files),
    }


# ============================================================
# 7. COMPLETE SMOKE WORKFLOW
# ============================================================


def run_smoke(replace_existing: bool = False) -> dict[str, Any]:
    """Prepare balanced files, run one epoch, and verify every output."""

    ensure_isolated_environment()
    protect_or_replace_outputs(replace_existing=replace_existing)

    base_config = BertTrainingConfig()
    validate_bert_config(base_config)

    split_fields = resolve_split_field_names(base_config)
    source_paths = source_split_paths(base_config, split_fields)
    smoke_paths, split_counts, label_fields = prepare_balanced_smoke_files(
        source_paths
    )
    smoke_config = build_smoke_config(base_config, split_fields, smoke_paths)

    print("BERT smoke model:", smoke_config.model_id)
    print("Training epochs:", smoke_config.number_of_epochs)
    print("Split fields:", split_fields)
    print("Detected label fields:", label_fields)
    print("Balanced class counts:", split_counts)

    # This is the only statement that downloads and trains the model. All
    # preparation and safety checks above complete before it can run.
    run_bert_training(smoke_config)

    summary = validate_smoke_artifacts()

    print("Status:", summary["status"])
    print("Model revision:", summary["model_revision"])
    print("Total parameters:", summary["total_parameters"])
    print("Training seconds:", summary["training_seconds"])
    print("Test accuracy:", summary["test_accuracy"])
    print("Test macro F1:", summary["test_macro_f1"])
    print("Confusion matrix:", summary["confusion_matrix"])
    print("Final model files:", summary["final_model_files"])
    print("BERT SMOKE TRAINING: PASSED")

    return summary


def parse_arguments() -> argparse.Namespace:
    """Read the one explicit permission controlling smoke-output replacement."""

    parser = argparse.ArgumentParser(
        description="Run the balanced full-BERT smoke experiment."
    )
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="Replace only the dedicated BERT smoke data and artifacts.",
    )
    return parser.parse_args()


def main() -> int:
    """Command-line entry point with a concise actionable failure message."""

    arguments = parse_arguments()

    try:
        run_smoke(replace_existing=arguments.replace_existing)
    except Exception as exc:
        print(
            "BERT SMOKE TRAINING: FAILED: "
            f"{type(exc).__name__}: {exc}"
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
