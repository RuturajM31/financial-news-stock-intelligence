"""Load and validate verified model-training evidence for Streamlit.

Purpose
-------
The model pages present a fixed benchmark completed before Streamlit was built.
This module validates the small JSON evidence file and exposes immutable view
objects. It does not load model artifacts, datasets, private caches, or secrets.

Failure behaviour
-----------------
Missing, malformed, non-finite, or internally inconsistent evidence fails
closed. The pages show a plain error instead of guessing or repairing values.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

EVIDENCE_PATH = Path(__file__).resolve().parents[1] / "data" / "sentiment_benchmark_evidence.json"
CLASS_ORDER = ("Bearish", "Neutral", "Bullish")
MODEL_ORDER = ("bert", "distilbert", "bert_lora")


def _finite_number(value: Any, location: str, *, minimum: float | None = None) -> float:
    """Return one finite number and enforce an optional lower boundary."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location} must be a number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{location} must be finite.")
    if minimum is not None and result < minimum:
        raise ValueError(f"{location} must be at least {minimum}.")
    return result


def _probability(value: Any, location: str) -> float:
    """Return one score constrained to the inclusive zero-to-one range."""

    result = _finite_number(value, location)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{location} must be between zero and one.")
    return result


@dataclass(frozen=True)
class EpochView:
    """Store one checked training-history row."""

    epoch: int
    average_training_loss: float
    validation_loss: float
    validation_accuracy: float
    validation_macro_f1: float


@dataclass(frozen=True)
class ClassResultView:
    """Store checked precision, recall, F1, and row count for one class."""

    label: str
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True)
class ModelEvidenceView:
    """Store the verified comparison fields for one sentiment model."""

    key: str
    display_name: str
    role: str
    accuracy: float
    macro_f1: float
    weighted_f1: float
    latency_ms_per_record: float
    memory_mib: float
    total_parameters: int
    trainable_parameters: int
    training_seconds: float | None
    test_runtime_seconds: float | None
    artifact_size_bytes: int | None
    confusion_matrix: tuple[tuple[int, ...], ...] | None
    per_class: tuple[ClassResultView, ...] | None
    training_history: tuple[EpochView, ...] | None
    selection_reason: str
    limitation: str

    @property
    def parameter_reduction_percent(self) -> float:
        """Return the share of parameters that did not require training."""

        if self.total_parameters <= 0:
            return 0.0
        return 100.0 * (1.0 - self.trainable_parameters / self.total_parameters)


@dataclass(frozen=True)
class BenchmarkEvidenceView:
    """Store the complete checked benchmark used by both portfolio pages."""

    benchmark_name: str
    completed_date: str
    dataset_name: str
    split_rows: Mapping[str, int]
    test_class_counts: Mapping[str, int]
    model_revision: str
    quality_champion: str
    deployment_champion: str
    models: tuple[ModelEvidenceView, ...]
    evidence_gaps: Mapping[str, str]
    verification: Mapping[str, Any]

    def model(self, key: str) -> ModelEvidenceView:
        """Return one model by stable key or fail clearly."""

        for model in self.models:
            if model.key == key:
                return model
        raise ValueError(f"Unknown benchmark model: {key!r}.")


def _parse_history(value: Any, location: str) -> tuple[EpochView, ...] | None:
    """Validate optional ordered training history."""

    if value is None:
        return None
    if not isinstance(value, list) or not value:
        raise ValueError(f"{location} must be a non-empty list when provided.")
    rows: list[EpochView] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError(f"{location}[{index}] must be an object.")
        epoch = item.get("epoch")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch != index + 1:
            raise ValueError(f"{location}[{index}].epoch must be {index + 1}.")
        rows.append(
            EpochView(
                epoch=epoch,
                average_training_loss=_finite_number(
                    item.get("average_training_loss"),
                    f"{location}[{index}].average_training_loss",
                    minimum=0.0,
                ),
                validation_loss=_finite_number(
                    item.get("validation_loss"),
                    f"{location}[{index}].validation_loss",
                    minimum=0.0,
                ),
                validation_accuracy=_probability(
                    item.get("validation_accuracy"),
                    f"{location}[{index}].validation_accuracy",
                ),
                validation_macro_f1=_probability(
                    item.get("validation_macro_f1"),
                    f"{location}[{index}].validation_macro_f1",
                ),
            )
        )
    return tuple(rows)


def _parse_confusion(value: Any, location: str) -> tuple[tuple[int, ...], ...] | None:
    """Validate an optional three-by-three confusion matrix."""

    if value is None:
        return None
    if not isinstance(value, list) or len(value) != len(CLASS_ORDER):
        raise ValueError(f"{location} must contain three rows.")
    result: list[tuple[int, ...]] = []
    for row_index, row in enumerate(value):
        if not isinstance(row, list) or len(row) != len(CLASS_ORDER):
            raise ValueError(f"{location}[{row_index}] must contain three values.")
        checked: list[int] = []
        for column_index, number in enumerate(row):
            if isinstance(number, bool) or not isinstance(number, int) or number < 0:
                raise ValueError(f"{location}[{row_index}][{column_index}] must be a non-negative whole number.")
            checked.append(number)
        result.append(tuple(checked))
    return tuple(result)


def _parse_per_class(value: Any, location: str) -> tuple[ClassResultView, ...] | None:
    """Validate optional class-level results in the fixed label order."""

    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != set(CLASS_ORDER):
        raise ValueError(f"{location} must contain Bearish, Neutral, and Bullish.")
    rows: list[ClassResultView] = []
    for label in CLASS_ORDER:
        item = value[label]
        if not isinstance(item, Mapping):
            raise ValueError(f"{location}.{label} must be an object.")
        support = item.get("support")
        if isinstance(support, bool) or not isinstance(support, int) or support < 1:
            raise ValueError(f"{location}.{label}.support must be positive.")
        rows.append(
            ClassResultView(
                label=label,
                precision=_probability(item.get("precision"), f"{location}.{label}.precision"),
                recall=_probability(item.get("recall"), f"{location}.{label}.recall"),
                f1=_probability(item.get("f1"), f"{location}.{label}.f1"),
                support=support,
            )
        )
    return tuple(rows)


def _optional_number(value: Any, location: str) -> float | None:
    """Validate an optional non-negative finite number."""

    if value is None:
        return None
    return _finite_number(value, location, minimum=0.0)


def _optional_integer(value: Any, location: str) -> int | None:
    """Validate an optional non-negative whole number."""

    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{location} must be a non-negative whole number.")
    return value


def load_benchmark_evidence(path: Path = EVIDENCE_PATH) -> BenchmarkEvidenceView:
    """Load, validate, and return the immutable benchmark evidence."""

    if not path.is_file():
        raise FileNotFoundError(f"Verified benchmark evidence is missing: {path.name}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("Benchmark evidence must be a JSON object.")
    if raw.get("evidence_status") != "verified_completed_benchmark":
        raise ValueError("Benchmark evidence is not marked as verified and completed.")

    dataset = raw.get("dataset")
    if not isinstance(dataset, Mapping):
        raise ValueError("dataset must be an object.")
    split_rows = dataset.get("split_rows")
    class_counts = dataset.get("test_class_counts")
    if not isinstance(split_rows, Mapping) or set(split_rows) != {"training", "validation", "test"}:
        raise ValueError("dataset.split_rows is incomplete.")
    checked_splits: dict[str, int] = {}
    for key, value in split_rows.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"dataset.split_rows.{key} must be positive.")
        checked_splits[str(key)] = value
    if not isinstance(class_counts, Mapping) or set(class_counts) != set(CLASS_ORDER):
        raise ValueError("dataset.test_class_counts is incomplete.")
    checked_counts: dict[str, int] = {}
    for label in CLASS_ORDER:
        count = class_counts[label]
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError(
                f"dataset.test_class_counts.{label} must be positive."
            )
        checked_counts[label] = count
    if sum(checked_counts.values()) != checked_splits["test"]:
        raise ValueError("Test class counts do not equal the test split size.")

    models_raw = raw.get("models")
    if not isinstance(models_raw, Mapping) or set(models_raw) != set(MODEL_ORDER):
        raise ValueError("models must contain BERT, DistilBERT, and BERT LoRA.")
    models: list[ModelEvidenceView] = []
    for key in MODEL_ORDER:
        item = models_raw[key]
        if not isinstance(item, Mapping):
            raise ValueError(f"models.{key} must be an object.")
        total_parameters = item.get("total_parameters")
        trainable_parameters = item.get("trainable_parameters")
        if isinstance(total_parameters, bool) or not isinstance(total_parameters, int) or total_parameters < 1:
            raise ValueError(f"models.{key}.total_parameters must be positive.")
        invalid_trainable_count = (
            isinstance(trainable_parameters, bool)
            or not isinstance(trainable_parameters, int)
            or not 0 < trainable_parameters <= total_parameters
        )
        if invalid_trainable_count:
            raise ValueError(f"models.{key}.trainable_parameters is invalid.")
        confusion = _parse_confusion(item.get("confusion_matrix"), f"models.{key}.confusion_matrix")
        per_class = _parse_per_class(item.get("per_class"), f"models.{key}.per_class")
        if confusion is not None and sum(sum(row) for row in confusion) != checked_splits["test"]:
            raise ValueError(f"models.{key}.confusion_matrix total is incorrect.")
        if per_class is not None and sum(row.support for row in per_class) != checked_splits["test"]:
            raise ValueError(f"models.{key}.per_class support is incorrect.")
        display_name = str(item.get("display_name", "")).strip()
        role = str(item.get("role", "")).strip()
        selection_reason = str(item.get("selection_reason", "")).strip()
        limitation = str(item.get("limitation", "")).strip()
        if not all((display_name, role, selection_reason, limitation)):
            raise ValueError(f"models.{key} contains an empty explanation field.")
        models.append(
            ModelEvidenceView(
                key=key,
                display_name=display_name,
                role=role,
                accuracy=_probability(item.get("accuracy"), f"models.{key}.accuracy"),
                macro_f1=_probability(item.get("macro_f1"), f"models.{key}.macro_f1"),
                weighted_f1=_probability(item.get("weighted_f1"), f"models.{key}.weighted_f1"),
                latency_ms_per_record=_finite_number(
                    item.get("latency_ms_per_record"),
                    f"models.{key}.latency_ms_per_record",
                    minimum=0.0,
                ),
                memory_mib=_finite_number(item.get("memory_mib"), f"models.{key}.memory_mib", minimum=0.0),
                total_parameters=total_parameters,
                trainable_parameters=trainable_parameters,
                training_seconds=_optional_number(item.get("training_seconds"), f"models.{key}.training_seconds"),
                test_runtime_seconds=_optional_number(
                    item.get("test_runtime_seconds"),
                    f"models.{key}.test_runtime_seconds",
                ),
                artifact_size_bytes=_optional_integer(
                    item.get("artifact_size_bytes"),
                    f"models.{key}.artifact_size_bytes",
                ),
                confusion_matrix=confusion,
                per_class=per_class,
                training_history=_parse_history(item.get("training_history"), f"models.{key}.training_history"),
                selection_reason=selection_reason,
                limitation=limitation,
            )
        )

    quality_champion = str(raw.get("quality_champion", "")).strip()
    deployment_champion = str(raw.get("deployment_champion", "")).strip()
    quality_ranking = raw.get("quality_ranking")
    if quality_champion not in MODEL_ORDER:
        raise ValueError("quality_champion is not a known model.")
    if deployment_champion not in MODEL_ORDER:
        raise ValueError("deployment_champion is not a known model.")
    if quality_ranking != list(MODEL_ORDER):
        raise ValueError("quality_ranking does not match the verified order.")

    gaps = raw.get("evidence_gaps")
    verification = raw.get("verification")
    if not isinstance(gaps, Mapping) or not isinstance(verification, Mapping):
        raise ValueError("Evidence gaps and verification must be objects.")
    return BenchmarkEvidenceView(
        benchmark_name=str(raw.get("benchmark_name", "")).strip(),
        completed_date=str(raw.get("benchmark_completed_utc_date", "")).strip(),
        dataset_name=str(dataset.get("name", "")).strip(),
        split_rows=checked_splits,
        test_class_counts=checked_counts,
        model_revision=str(raw.get("model_revision", "")).strip(),
        quality_champion=quality_champion,
        deployment_champion=deployment_champion,
        models=tuple(models),
        evidence_gaps={str(key): str(value) for key, value in gaps.items()},
        verification=dict(verification),
    )
