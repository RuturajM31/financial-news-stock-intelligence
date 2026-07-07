"""
Train a LoRA efficiency benchmark for financial-news sentiment.

Purpose
-------
Fine-tune lightweight Low-Rank Adaptation parameters on top of the approved
BERT-base model while keeping the same verified Financial PhraseBank splits,
three sentiment labels, random seed, sequence length, epoch count, class
weighting, checkpoint rule, and untouched test evaluation used by the full
models.

Inputs and source variables
---------------------------
The existing ``BertTrainingConfig`` provides the source split paths and shared
comparison settings. ``LoraTrainingConfig`` contains only the LoRA-specific
adapter design and dedicated artifact paths. JSONL records are preserved at
their original grain of one financial sentence per row.

Processing and data journey
---------------------------
``JSONL splits -> schema detection -> leakage check -> canonical text/labels
-> tokenization -> BERT base model -> LoRA query/value adapters -> weighted
cross-entropy training -> validation checkpoint selection -> untouched test
prediction -> metrics, adapter, tokenizer, and manifest``

Outputs and downstream use
--------------------------
- ``artifacts/models/bert_lora_sentiment`` checkpoints and final adapter;
- ``reports/metrics/bert_lora_sentiment_metrics.json``;
- ``artifacts/manifests/bert_lora_sentiment_training_manifest.json``.

The final comparison stage reads these files beside the verified DistilBERT
and full-BERT evidence.

Assumptions, thresholds, and limitations
----------------------------------------
The accepted label order is Bearish=0, Neutral=1, Bullish=2. Class weights use
``N / (K * class_count)`` where ``N`` is the training-row count and ``K=3``.
LoRA targets BERT attention ``query`` and ``value`` projections with rank 8.
The classifier is saved with the adapter. LoRA reduces trainable parameters,
but deployment still requires the base BERT weights. This experiment cannot
be selected as champion until the common comparison stage passes.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import resource
import shutil
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


# ============================================================
# 1. APPROVED EXPERIMENT CONTRACT
# ============================================================

LABEL_ORDER = ("Bearish", "Neutral", "Bullish")
LABEL_TO_ID = {label: index for index, label in enumerate(LABEL_ORDER)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}
EXPECTED_CLASS_COUNT = len(LABEL_ORDER)
EXPECTED_SPLIT_RECORDS = {
    "train": 2_413,
    "validation": 517,
    "test": 518,
}
MEBIBYTE = 1024 * 1024
BASE_MODEL_ID = "google-bert/bert-base-uncased"

PROJECT_ROOT = Path(__file__).resolve().parents[3]
LORA_MODEL_ROOT = PROJECT_ROOT / "artifacts" / "models" / "bert_lora_sentiment"
LORA_CHECKPOINT_DIR = LORA_MODEL_ROOT / "checkpoints"
LORA_FINAL_ADAPTER_DIR = LORA_MODEL_ROOT / "final_adapter"
LORA_METRICS_FILE = (
    PROJECT_ROOT / "reports" / "metrics" / "bert_lora_sentiment_metrics.json"
)
LORA_MANIFEST_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "manifests"
    / "bert_lora_sentiment_training_manifest.json"
)

TEXT_FIELD_CANDIDATES = (
    "text",
    "sentence",
    "news_text",
    "article_text",
    "content",
    "headline",
)

LABEL_FIELD_CANDIDATES = (
    "label_id",
    "label",
    "label_name",
    "sentiment_label",
    "sentiment",
)

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

REQUIRED_ADAPTER_FILES = {
    "adapter_config.json",
    "adapter_model.safetensors",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.txt",
}


@dataclass(frozen=True)
class LoraTrainingConfig:
    """Store the approved LoRA settings and dedicated output paths."""

    experiment_name: str = "BERT LoRA Financial Sentiment"
    model_family: str = "BERT-LoRA"
    benchmark_role: str = "parameter_efficient_comparison"
    base_model_id: str = BASE_MODEL_ID
    number_of_epochs: float = 3.0
    max_length: int = 128
    train_batch_size: int = 8
    evaluation_batch_size: int = 16
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    random_seed: int = 42
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    target_modules: tuple[str, ...] = ("query", "value")
    modules_to_save: tuple[str, ...] = ("classifier",)
    checkpoint_dir: Path = LORA_CHECKPOINT_DIR
    final_adapter_dir: Path = LORA_FINAL_ADAPTER_DIR
    metrics_file: Path = LORA_METRICS_FILE
    manifest_file: Path = LORA_MANIFEST_FILE
    run_name: str = "bert_lora_financial_phrasebank_full"


class LoraTrainingError(RuntimeError):
    """Raised when LoRA data, configuration, training, or evidence is unsafe."""


# ============================================================
# 2. FILE, JSON, AND CHECKSUM HELPERS
# ============================================================


def require_regular_file(file_path: Path, description: str) -> None:
    """Require one existing non-symlink file before reading it."""

    if not file_path.exists():
        raise LoraTrainingError(f"Missing {description}: {file_path}")
    if file_path.is_symlink() or not file_path.is_file():
        raise LoraTrainingError(
            f"Unsafe {description}; expected a regular file: {file_path}"
        )


def calculate_sha256(file_path: Path) -> str:
    """Return the SHA-256 checksum used to bind evidence to one source file."""

    require_regular_file(file_path, "checksum source")
    digest = hashlib.sha256()
    with file_path.open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_peak_rss_mib() -> float:
    """Return peak process RSS normalized to MiB on macOS and Linux."""

    raw_value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    bytes_value = raw_value if sys.platform == "darwin" else raw_value * 1024.0
    return bytes_value / MEBIBYTE


def build_artifact_inventory(directory: Path) -> list[dict[str, Any]]:
    """Checksum every regular adapter and tokenizer file recursively."""

    if not directory.exists() or not directory.is_dir() or directory.is_symlink():
        raise LoraTrainingError(
            f"Saved adapter directory is missing or unsafe: {directory}"
        )

    inventory: list[dict[str, Any]] = []
    for file_path in sorted(directory.rglob("*")):
        if file_path.is_symlink():
            raise LoraTrainingError(
                f"Symbolic links are not accepted in adapter evidence: {file_path}"
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
        raise LoraTrainingError(f"Saved adapter directory is empty: {directory}")
    return inventory


def write_json_atomic(file_path: Path, payload: Mapping[str, Any]) -> None:
    """Write one JSON object atomically so partial files never look complete."""

    file_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = file_path.with_suffix(file_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary_path.replace(file_path)


def read_jsonl(file_path: Path) -> list[dict[str, Any]]:
    """Read non-empty JSONL objects and report the exact malformed line."""

    require_regular_file(file_path, "JSONL split")
    records: list[dict[str, Any]] = []

    with file_path.open("r", encoding="utf-8") as source_file:
        for line_number, raw_line in enumerate(source_file, start=1):
            text = raw_line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError as exc:
                raise LoraTrainingError(
                    f"Invalid JSON in {file_path} at line {line_number}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise LoraTrainingError(
                    f"Expected an object in {file_path} at line {line_number}."
                )
            records.append(record)

    if not records:
        raise LoraTrainingError(f"JSONL split contains no records: {file_path}")
    return records


# ============================================================
# 3. SHARED BERT CONTRACT AND SPLIT DISCOVERY
# ============================================================


def load_bert_config() -> Any:
    """Create the existing full-BERT configuration through a lazy import."""

    from financial_news_intelligence.models.bert_training import (
        BertTrainingConfig,
        validate_bert_config,
    )

    config = BertTrainingConfig()
    validate_bert_config(config)
    return config


def path_like_config_fields(config: Any) -> dict[str, Path]:
    """Return JSON or JSONL dataclass fields that behave like source paths."""

    discovered: dict[str, Path] = {}
    for field in fields(config):
        value = getattr(config, field.name)
        if isinstance(value, (str, Path)):
            path_value = Path(value)
            if path_value.suffix.lower() in {".json", ".jsonl"}:
                discovered[field.name] = path_value
    return discovered


def resolve_split_field_names(config: Any) -> dict[str, str]:
    """Map train, validation, and test to unambiguous dataclass field names."""

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
            raise LoraTrainingError(
                f"Could not resolve one {split_name} split field. "
                f"Candidates={sorted(semantic_matches)}"
            )
        resolved[split_name] = semantic_matches[0]

    if len(set(resolved.values())) != len(resolved):
        raise LoraTrainingError(f"Split fields are not distinct: {resolved}")
    return resolved


def resolve_split_paths(config: Any) -> dict[str, Path]:
    """Read and validate the configured source path for every split."""

    field_names = resolve_split_field_names(config)
    paths: dict[str, Path] = {}
    for split_name, field_name in field_names.items():
        source_path = Path(getattr(config, field_name)).expanduser().resolve()
        require_regular_file(source_path, f"{split_name} split")
        paths[split_name] = source_path
    return paths


def validate_shared_protocol(bert_config: Any, lora_config: LoraTrainingConfig) -> None:
    """
    Preserve the shared experiment protocol and document the one LoRA override.

    LoRA keeps the full-BERT model, epochs, sequence length, batches, gradient
    accumulation, and seed. Its learning rate is intentionally adapter-specific
    and is therefore not required to match full fine-tuning.
    """

    required_matches = {
        "model_id": lora_config.base_model_id,
        "number_of_epochs": lora_config.number_of_epochs,
        "max_length": lora_config.max_length,
        "train_batch_size": lora_config.train_batch_size,
        "evaluation_batch_size": lora_config.evaluation_batch_size,
        "gradient_accumulation_steps": (
            lora_config.gradient_accumulation_steps
        ),
        "random_seed": lora_config.random_seed,
    }

    for field_name, expected_value in required_matches.items():
        actual_value = getattr(bert_config, field_name, None)
        if actual_value != expected_value:
            raise LoraTrainingError(
                f"Shared protocol mismatch for {field_name}: "
                f"{actual_value!r} != {expected_value!r}"
            )

    if getattr(bert_config, "use_class_weights", None) is not True:
        raise LoraTrainingError(
            "Full BERT must enable class weights for a fair LoRA comparison."
        )


# ============================================================
# 4. SCHEMA DETECTION, LABELS, AND LEAKAGE CONTROL
# ============================================================


def common_fields(records: Sequence[Mapping[str, Any]]) -> set[str]:
    """Return fields present in every row of one split."""

    if not records:
        raise LoraTrainingError("Cannot inspect an empty record collection.")

    shared = set(records[0])
    for record in records[1:]:
        shared.intersection_update(record)
    return shared


def detect_text_field(records: Sequence[Mapping[str, Any]]) -> str:
    """Find one common non-empty string field containing the sentence text."""

    shared = common_fields(records)
    preferred = [name for name in TEXT_FIELD_CANDIDATES if name in shared]
    other_fields = sorted(shared - set(preferred))

    for field_name in preferred + other_fields:
        values = [record[field_name] for record in records]
        if not all(isinstance(value, str) and value.strip() for value in values):
            continue
        unique_ratio = len(set(values)) / len(values)
        if unique_ratio >= 0.5:
            return field_name

    raise LoraTrainingError("No common non-empty text field was found.")


def scalar_label(value: Any) -> bool:
    """Return whether a JSON value can represent one class label."""

    if isinstance(value, bool):
        return False
    if isinstance(value, (str, int)):
        return True
    return isinstance(value, float) and value.is_integer()


def detect_label_field(records: Sequence[Mapping[str, Any]]) -> str:
    """Find one common scalar field containing exactly three classes."""

    shared = common_fields(records)
    preferred = [name for name in LABEL_FIELD_CANDIDATES if name in shared]
    preferred.extend(
        sorted(
            name
            for name in shared
            if "label" in name.lower() and name not in preferred
        )
    )

    for field_name in preferred:
        values = [record[field_name] for record in records]
        if not all(scalar_label(value) for value in values):
            continue
        if len(set(values)) == EXPECTED_CLASS_COUNT:
            return field_name

    raise LoraTrainingError("No common scalar field with three classes was found.")


def canonical_label_id(value: Any) -> int:
    """Convert numeric or named sentiment labels to the approved integer ID."""

    if isinstance(value, bool):
        raise LoraTrainingError(f"Boolean label is not supported: {value!r}")

    if isinstance(value, float) and value.is_integer():
        value = int(value)

    if isinstance(value, int):
        if value not in ID_TO_LABEL:
            raise LoraTrainingError(f"Label ID must be 0, 1, or 2: {value}")
        return value

    if isinstance(value, str):
        normalized = value.strip().casefold()
        for label_name, label_id in LABEL_TO_ID.items():
            if normalized == label_name.casefold():
                return label_id

    raise LoraTrainingError(f"Unsupported sentiment label: {value!r}")


def normalize_text(value: str) -> str:
    """Normalize whitespace and case for exact cross-split leakage checks."""

    return " ".join(value.split()).casefold()


def validate_no_cross_split_leakage(
    split_records: Mapping[str, Sequence[Mapping[str, Any]]],
    text_field: str,
) -> None:
    """Reject any normalized sentence appearing in more than one split."""

    normalized = {
        split_name: {
            normalize_text(str(record[text_field])) for record in records
        }
        for split_name, records in split_records.items()
    }

    split_pairs = (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    )
    for left_name, right_name in split_pairs:
        overlap = normalized[left_name] & normalized[right_name]
        if overlap:
            example = sorted(overlap)[0]
            raise LoraTrainingError(
                f"Sentence leakage between {left_name} and {right_name}: "
                f"{example[:80]!r}"
            )


def canonical_records(
    records: Sequence[Mapping[str, Any]],
    text_field: str,
    label_field: str,
) -> list[dict[str, Any]]:
    """Create the minimal text and integer-label rows consumed by Trainer."""

    canonical: list[dict[str, Any]] = []
    for record in records:
        sentence = str(record[text_field]).strip()
        if not sentence:
            raise LoraTrainingError("A source sentence is empty after stripping.")
        canonical.append(
            {
                "text": sentence,
                "labels": canonical_label_id(record[label_field]),
            }
        )
    return canonical


def prepare_dataset_records(
    split_paths: Mapping[str, Path],
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, dict[str, Any]],
]:
    """Load, validate, and canonicalize all three source splits."""

    split_order = ("train", "validation", "test")
    raw_records = {
        split_name: read_jsonl(split_paths[split_name])
        for split_name in split_order
    }
    text_fields = {
        split_name: detect_text_field(raw_records[split_name])
        for split_name in split_order
    }
    label_fields = {
        split_name: detect_label_field(raw_records[split_name])
        for split_name in split_order
    }

    if len(set(text_fields.values())) != 1:
        raise LoraTrainingError(f"Inconsistent text fields: {text_fields}")
    if len(set(label_fields.values())) != 1:
        raise LoraTrainingError(f"Inconsistent label fields: {label_fields}")

    text_field = text_fields["train"]
    label_field = label_fields["train"]
    validate_no_cross_split_leakage(raw_records, text_field)

    canonical = {
        split_name: canonical_records(
            raw_records[split_name],
            text_field,
            label_field,
        )
        for split_name in split_order
    }

    for split_name, required_records in EXPECTED_SPLIT_RECORDS.items():
        actual_records = len(canonical[split_name])
        if actual_records != required_records:
            raise LoraTrainingError(
                f"{split_name} split must contain exactly {required_records} "
                f"rows; found {actual_records}."
            )

    source_evidence = {
        split_name: {
            "path": str(split_paths[split_name]),
            "checksum_sha256": calculate_sha256(split_paths[split_name]),
            "records": len(canonical[split_name]),
            "text_field": text_field,
            "label_field": label_field,
            "label_counts": {
                ID_TO_LABEL[label_id]: int(count)
                for label_id, count in sorted(
                    Counter(row["labels"] for row in canonical[split_name]).items()
                )
            },
        }
        for split_name in split_order
    }
    return canonical, source_evidence


# ============================================================
# 5. CLASS WEIGHTS AND METRICS
# ============================================================


def calculate_class_weights(labels: Sequence[int]) -> np.ndarray:
    """Calculate balanced weights using N divided by K times class count."""

    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=3)
    if len(counts) != EXPECTED_CLASS_COUNT or np.any(counts <= 0):
        raise LoraTrainingError(
            f"All three classes must be present in training data: {counts.tolist()}"
        )

    total_records = int(counts.sum())
    weights = total_records / (EXPECTED_CLASS_COUNT * counts.astype(float))
    return weights.astype(np.float32)


def safe_divide(numerator: float, denominator: float) -> float:
    """Return zero when a precision or recall denominator is empty."""

    return numerator / denominator if denominator else 0.0


def classification_summary(
    labels: Sequence[int] | np.ndarray,
    predictions: Sequence[int] | np.ndarray,
) -> dict[str, Any]:
    """Calculate deterministic three-class metrics without scikit-learn."""

    actual = np.asarray(labels, dtype=np.int64)
    predicted = np.asarray(predictions, dtype=np.int64)
    if actual.shape != predicted.shape or actual.ndim != 1:
        raise LoraTrainingError("Labels and predictions must be equal 1D arrays.")
    if actual.size == 0:
        raise LoraTrainingError("Cannot evaluate an empty prediction set.")
    if np.any(actual < 0) or np.any(actual >= EXPECTED_CLASS_COUNT):
        raise LoraTrainingError("Actual labels fall outside the 0..2 range.")
    if np.any(predicted < 0) or np.any(predicted >= EXPECTED_CLASS_COUNT):
        raise LoraTrainingError("Predictions fall outside the 0..2 range.")

    matrix = np.zeros((EXPECTED_CLASS_COUNT, EXPECTED_CLASS_COUNT), dtype=int)
    for actual_id, predicted_id in zip(actual, predicted):
        matrix[actual_id, predicted_id] += 1

    per_class: dict[str, dict[str, float | int]] = {}
    precision_values: list[float] = []
    recall_values: list[float] = []
    f1_values: list[float] = []
    supports: list[int] = []

    for class_id, label_name in enumerate(LABEL_ORDER):
        true_positive = int(matrix[class_id, class_id])
        false_positive = int(matrix[:, class_id].sum() - true_positive)
        false_negative = int(matrix[class_id, :].sum() - true_positive)
        support = int(matrix[class_id, :].sum())

        precision = safe_divide(true_positive, true_positive + false_positive)
        recall = safe_divide(true_positive, true_positive + false_negative)
        f1_score = safe_divide(2 * precision * recall, precision + recall)

        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1_score)
        supports.append(support)
        per_class[label_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1_score,
            "support": support,
        }

    total = int(matrix.sum())
    support_array = np.asarray(supports, dtype=float)
    weights = support_array / total

    return {
        "accuracy": float(np.trace(matrix) / total),
        "macro_precision": float(np.mean(precision_values)),
        "macro_recall": float(np.mean(recall_values)),
        "macro_f1": float(np.mean(f1_values)),
        "weighted_precision": float(np.dot(weights, precision_values)),
        "weighted_recall": float(np.dot(weights, recall_values)),
        "weighted_f1": float(np.dot(weights, f1_values)),
        "confusion_matrix": matrix.tolist(),
        "per_class": per_class,
    }


def trainer_metric_values(summary: Mapping[str, Any]) -> dict[str, float]:
    """Return only scalar values accepted by Hugging Face Trainer."""

    scalar_names = (
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
    )
    result = {name: float(summary[name]) for name in scalar_names}

    per_class = summary["per_class"]
    for label_name in LABEL_ORDER:
        prefix = label_name.casefold()
        result[f"{prefix}_precision"] = float(
            per_class[label_name]["precision"]
        )
        result[f"{prefix}_recall"] = float(per_class[label_name]["recall"])
        result[f"{prefix}_f1"] = float(per_class[label_name]["f1"])
    return result


# ============================================================
# 6. CONFIGURATION AND OUTPUT PROTECTION
# ============================================================


def validate_lora_config(config: LoraTrainingConfig) -> None:
    """Validate model identity, numeric ranges, and isolated artifact paths."""

    if config.base_model_id != BASE_MODEL_ID:
        raise LoraTrainingError(f"LoRA benchmark must use {BASE_MODEL_ID}.")
    if config.number_of_epochs <= 0:
        raise LoraTrainingError("number_of_epochs must be positive.")
    if config.max_length <= 0:
        raise LoraTrainingError("max_length must be positive.")
    if config.train_batch_size <= 0 or config.evaluation_batch_size <= 0:
        raise LoraTrainingError("Batch sizes must be positive.")
    if config.gradient_accumulation_steps <= 0:
        raise LoraTrainingError("gradient_accumulation_steps must be positive.")
    if config.learning_rate <= 0:
        raise LoraTrainingError("learning_rate must be positive.")
    if not 0.0 <= config.lora_dropout < 1.0:
        raise LoraTrainingError("lora_dropout must be inside 0..1.")
    if config.lora_rank <= 0 or config.lora_alpha <= 0:
        raise LoraTrainingError("LoRA rank and alpha must be positive.")
    if not config.target_modules:
        raise LoraTrainingError("At least one LoRA target module is required.")

    controlled_paths = (
        config.checkpoint_dir,
        config.final_adapter_dir,
        config.metrics_file,
        config.manifest_file,
    )
    resolved_paths = [Path(path).expanduser().resolve() for path in controlled_paths]
    if len(set(resolved_paths)) != len(resolved_paths):
        raise LoraTrainingError("LoRA artifact paths must be distinct.")


def lora_output_paths(config: LoraTrainingConfig) -> tuple[Path, ...]:
    """Return every dedicated LoRA path that may be replaced explicitly."""

    return (
        config.checkpoint_dir,
        config.final_adapter_dir,
        config.metrics_file,
        config.manifest_file,
    )


def protect_or_replace_outputs(
    config: LoraTrainingConfig,
    replace_existing: bool,
) -> None:
    """Protect previous LoRA evidence unless replacement is explicit."""

    existing = [path for path in lora_output_paths(config) if path.exists()]
    if existing and not replace_existing:
        formatted = "\n".join(f"- {path}" for path in existing)
        raise LoraTrainingError(
            "LoRA outputs already exist. Review them or use "
            f"--replace-existing.\n{formatted}"
        )

    if replace_existing:
        for path in existing:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()


# ============================================================
# 7. TRAINING IMPLEMENTATION
# ============================================================


def dependency_versions() -> dict[str, str]:
    """Record the exact libraries that define the training environment."""

    import datasets
    import peft
    import torch
    import transformers

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "datasets": datasets.__version__,
        "peft": peft.__version__,
    }


def count_parameters(model: Any) -> dict[str, int]:
    """Count total and gradient-enabled parameters after LoRA injection."""

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "frozen_parameters": int(total - trainable),
    }


def build_training_arguments(config: LoraTrainingConfig) -> Any:
    """Create CPU Trainer arguments with macro-F1 checkpoint selection."""

    from transformers import TrainingArguments

    return TrainingArguments(
        output_dir=str(config.checkpoint_dir),
        run_name=config.run_name,
        num_train_epochs=config.number_of_epochs,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.evaluation_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=25,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        seed=config.random_seed,
        data_seed=config.random_seed,
        use_cpu=True,
        dataloader_num_workers=0,
        report_to=[],
        save_safetensors=True,
    )


def run_lora_training(
    config: LoraTrainingConfig | None = None,
    replace_existing: bool = False,
) -> dict[str, Any]:
    """Train, evaluate, save, and document the complete LoRA experiment."""

    if config is None:
        config = LoraTrainingConfig()

    validate_lora_config(config)
    protect_or_replace_outputs(config, replace_existing=replace_existing)
    baseline_peak_rss_mib = current_peak_rss_mib()

    # Heavy libraries are imported only after configuration and output safety
    # checks pass. This keeps unit tests fast and prevents accidental downloads.
    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
    )

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)

    bert_config = load_bert_config()
    validate_shared_protocol(bert_config, config)
    split_paths = resolve_split_paths(bert_config)
    canonical, source_evidence = prepare_dataset_records(split_paths)

    train_labels = [row["labels"] for row in canonical["train"]]
    class_weights = calculate_class_weights(train_labels)

    tokenizer = AutoTokenizer.from_pretrained(config.base_model_id)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        config.base_model_id,
        num_labels=EXPECTED_CLASS_COUNT,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )
    model_revision = getattr(base_model.config, "_commit_hash", None)
    if not isinstance(model_revision, str) or not model_revision:
        model_revision = "unresolved"

    adapter_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.target_modules),
        modules_to_save=list(config.modules_to_save),
        bias="none",
    )
    model = get_peft_model(base_model, adapter_config)
    parameter_counts = count_parameters(model)

    if parameter_counts["trainable_parameters"] >= parameter_counts[
        "total_parameters"
    ]:
        raise LoraTrainingError(
            "LoRA must train fewer parameters than the complete base model."
        )

    datasets = {
        split_name: Dataset.from_list(records)
        for split_name, records in canonical.items()
    }

    def tokenize(batch: Mapping[str, Sequence[str]]) -> Mapping[str, Any]:
        """Tokenize one batch using the shared fixed truncation threshold."""

        return tokenizer(
            list(batch["text"]),
            truncation=True,
            max_length=config.max_length,
        )

    tokenized = {
        split_name: dataset.map(
            tokenize,
            batched=True,
            remove_columns=["text"],
            desc="Tokenizing Financial PhraseBank for LoRA",
        )
        for split_name, dataset in datasets.items()
    }

    class WeightedTrainer(Trainer):
        """Apply the verified balanced class weights to cross-entropy loss."""

        def __init__(self, *args: Any, class_weight_values: np.ndarray, **kwargs: Any):
            """Store balanced class weights beside the standard Trainer state."""

            super().__init__(*args, **kwargs)
            self.class_weight_values = torch.tensor(
                class_weight_values,
                dtype=torch.float32,
            )

        def compute_loss(
            self,
            model: Any,
            inputs: dict[str, Any],
            return_outputs: bool = False,
            **_: Any,
        ) -> Any:
            """Calculate weighted cross-entropy for one training batch."""

            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")
            loss_function = torch.nn.CrossEntropyLoss(
                weight=self.class_weight_values.to(logits.device)
            )
            loss = loss_function(
                logits.view(-1, EXPECTED_CLASS_COUNT),
                labels.view(-1),
            )
            return (loss, outputs) if return_outputs else loss

    def compute_metrics(evaluation: Any) -> dict[str, float]:
        """Return scalar metrics used for validation and checkpoint selection."""

        predicted_ids = np.argmax(evaluation.predictions, axis=-1)
        summary = classification_summary(evaluation.label_ids, predicted_ids)
        return trainer_metric_values(summary)

    trainer = WeightedTrainer(
        model=model,
        args=build_training_arguments(config),
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        class_weight_values=class_weights,
    )

    training_started = time.perf_counter()
    training_result = trainer.train()
    training_seconds = time.perf_counter() - training_started

    validation_metrics = trainer.evaluate(tokenized["validation"])
    test_output = trainer.predict(tokenized["test"])
    predicted_ids = np.argmax(test_output.predictions, axis=-1)
    test_summary = classification_summary(test_output.label_ids, predicted_ids)

    config.final_adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(config.final_adapter_dir)
    tokenizer.save_pretrained(config.final_adapter_dir)
    artifact_files = build_artifact_inventory(config.final_adapter_dir)
    completed_peak_rss_mib = current_peak_rss_mib()

    test_runtime = float(test_output.metrics.get("test_runtime", 0.0))
    if not math.isfinite(test_runtime) or test_runtime <= 0:
        raise LoraTrainingError("Trainer did not report a positive test_runtime.")

    test_records = len(canonical["test"])
    timing = {
        "training_seconds": float(training_seconds),
        "test_inference_seconds": test_runtime,
        "test_records": test_records,
        "inference_milliseconds_per_record": (
            test_runtime * 1000.0 / test_records
        ),
    }

    metrics_payload = {
        "experiment_name": config.experiment_name,
        "validation_metrics": {
            key: float(value)
            for key, value in validation_metrics.items()
            if isinstance(value, (int, float))
        },
        "test_metrics": {
            **{
                key: float(value)
                for key, value in test_output.metrics.items()
                if isinstance(value, (int, float))
            },
            "test_accuracy": float(test_summary["accuracy"]),
            "test_macro_f1": float(test_summary["macro_f1"]),
            "test_weighted_f1": float(test_summary["weighted_f1"]),
        },
        "test_evaluation": {
            "confusion_matrix": test_summary["confusion_matrix"],
            "per_class": test_summary["per_class"],
            "label_order": list(LABEL_ORDER),
        },
        "timing": timing,
    }

    manifest_payload = {
        "status": "trained_and_evaluated",
        "experiment_name": config.experiment_name,
        "model_family": config.model_family,
        "benchmark_role": config.benchmark_role,
        "model_id": config.base_model_id,
        "model_revision": model_revision,
        "adapter_method": "LoRA",
        "label_mapping": {
            "label_to_id": LABEL_TO_ID,
            "id_to_label": {str(key): value for key, value in ID_TO_LABEL.items()},
        },
        "source_files": source_evidence,
        "class_weights": {
            LABEL_ORDER[index]: float(value)
            for index, value in enumerate(class_weights)
        },
        "lora_configuration": {
            "rank": config.lora_rank,
            "alpha": config.lora_alpha,
            "dropout": config.lora_dropout,
            "target_modules": list(config.target_modules),
            "modules_to_save": list(config.modules_to_save),
        },
        "training_configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "parameter_counts": parameter_counts,
        "timing": timing,
        "memory": {
            "measurement_method": (
                "resource.getrusage(resource.RUSAGE_SELF).ru_maxrss"
            ),
            "measurement_scope": "lora_training_validation_and_test_process",
            "baseline_peak_rss_mib": float(baseline_peak_rss_mib),
            "peak_process_rss_mib": float(completed_peak_rss_mib),
            "incremental_peak_rss_mib": float(
                max(0.0, completed_peak_rss_mib - baseline_peak_rss_mib)
            ),
            "platform_unit_normalization": (
                "bytes_on_macos; kibibytes_on_linux; normalized_to_mib"
            ),
        },
        "artifact_files": artifact_files,
        "environment": dependency_versions(),
        "final_model_directory": str(config.final_adapter_dir.resolve()),
        "metrics_file": str(config.metrics_file.resolve()),
        "deployment_note": (
            "The saved adapter requires the recorded base BERT model at "
            "inference time."
        ),
        "training_result": {
            key: float(value)
            for key, value in training_result.metrics.items()
            if isinstance(value, (int, float))
        },
    }

    write_json_atomic(config.metrics_file, metrics_payload)
    write_json_atomic(config.manifest_file, manifest_payload)
    return manifest_payload
