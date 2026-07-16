"""
Train and evaluate DistilBERT for financial-news sentiment.

Purpose
-------
This module fine-tunes DistilBERT on the verified Financial PhraseBank
train, validation, and test files.

The model predicts:

- Bearish;
- Neutral;
- Bullish.

Input
-----
Reproducible JSONL split files created by
financial_phrasebank_split.py:

- financial_phrasebank_train.jsonl;
- financial_phrasebank_validation.jsonl;
- financial_phrasebank_test.jsonl.

Processing
----------
1. Validate split files and detect leakage.
2. Resolve an exact Hugging Face model revision.
3. Tokenize financial sentences.
4. calculate class weights from the training split.
5. Fine-tune DistilBERT.
6. select the best checkpoint using validation macro F1.
7. evaluate the untouched test split.
8. record metrics, timing, parameters, checksums, and versions.

Output
------
- Fine-tuned model and tokenizer.
- Training checkpoints.
- Validation and test metrics.
- Confusion matrix and per-class metrics.
- Reproducibility manifest.

Important
---------
Importing this module does not download a model or begin training.
Training starts only when run_distilbert_training() is called.
"""

import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import numpy as np
import torch
from datasets import DatasetDict, load_dataset
from huggingface_hub import HfApi
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    set_seed,
)

from financial_news_intelligence.paths import (
    MANIFESTS_DIR,
    PROCESSED_DATA_DIR,
)


# ============================================================
# 1. LABEL DEFINITIONS
# ============================================================

LABEL_TO_ID = {
    "Bearish": 0,
    "Neutral": 1,
    "Bullish": 2,
}

ID_TO_LABEL = {
    label_id: label
    for label, label_id in LABEL_TO_ID.items()
}

LABEL_NAMES = [
    "Bearish",
    "Neutral",
    "Bullish",
]


# ============================================================
# 2. DEFAULT PROJECT PATHS
# ============================================================

DEFAULT_SPLIT_DIR = (
    PROCESSED_DATA_DIR / "transformer"
)

DEFAULT_TRAIN_FILE = (
    DEFAULT_SPLIT_DIR
    / "financial_phrasebank_train.jsonl"
)

DEFAULT_VALIDATION_FILE = (
    DEFAULT_SPLIT_DIR
    / "financial_phrasebank_validation.jsonl"
)

DEFAULT_TEST_FILE = (
    DEFAULT_SPLIT_DIR
    / "financial_phrasebank_test.jsonl"
)

DEFAULT_MODEL_ROOT = (
    MANIFESTS_DIR.parent
    / "models"
    / "distilbert_sentiment"
)

DEFAULT_REPORT_DIR = (
    MANIFESTS_DIR.parents[1]
    / "reports"
    / "metrics"
)


# ============================================================
# 3. TRAINING CONFIGURATION
# ============================================================

@dataclass
class DistilBertTrainingConfig:
    """
    Store every setting needed for one DistilBERT experiment.

    The settings are written into the training manifest so that the
    experiment can later be inspected and reproduced.
    """

    # Experiment identity written into the manifest.
    # BERT and LoRA override these values while reusing
    # the same verified training engine.
    experiment_name: str = (
        "DistilBERT Financial Sentiment"
    )

    model_family: str = "DistilBERT"

    benchmark_role: str = (
        "baseline_full_fine_tuning"
    )

    model_id: str = (
        "distilbert/distilbert-base-uncased"
    )

    # None means resolve and record the repository's exact commit
    # immediately before downloading the model.
    model_revision: str | None = None

    train_file: Path = DEFAULT_TRAIN_FILE
    validation_file: Path = DEFAULT_VALIDATION_FILE
    test_file: Path = DEFAULT_TEST_FILE

    checkpoint_dir: Path = field(
        default_factory=lambda: (
            DEFAULT_MODEL_ROOT / "checkpoints"
        )
    )

    final_model_dir: Path = field(
        default_factory=lambda: (
            DEFAULT_MODEL_ROOT / "final_model"
        )
    )

    metrics_file: Path = field(
        default_factory=lambda: (
            DEFAULT_REPORT_DIR
            / "distilbert_sentiment_metrics.json"
        )
    )

    manifest_file: Path = field(
        default_factory=lambda: (
            MANIFESTS_DIR
            / "distilbert_sentiment_training_manifest.json"
        )
    )

    run_name: str = (
        "distilbert_financial_phrasebank"
    )

    random_seed: int = 42
    max_length: int = 128

    train_batch_size: int = 8
    evaluation_batch_size: int = 16
    gradient_accumulation_steps: int = 2

    number_of_epochs: float = 3.0
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.10
    maximum_gradient_norm: float = 1.0

    logging_steps: int = 25
    save_total_limit: int = 2

    use_class_weights: bool = True
    force_cpu: bool = False
    full_determinism: bool = True
    overwrite_output_dir: bool = False


# ============================================================
# 4. CONFIGURATION VALIDATION
# ============================================================

def validate_training_config(
    config: DistilBertTrainingConfig,
) -> None:
    """
    Reject unsafe or impossible training settings.

    This validation runs before model download or training.
    """

    if not config.model_id.strip():
        raise ValueError(
            "model_id cannot be empty."
        )

    if not 8 <= config.max_length <= 512:
        raise ValueError(
            "max_length must be between 8 and 512."
        )

    positive_integer_fields = {
        "train_batch_size": config.train_batch_size,
        "evaluation_batch_size": (
            config.evaluation_batch_size
        ),
        "gradient_accumulation_steps": (
            config.gradient_accumulation_steps
        ),
        "logging_steps": config.logging_steps,
        "save_total_limit": config.save_total_limit,
    }

    for field_name, field_value in (
        positive_integer_fields.items()
    ):
        if field_value <= 0:
            raise ValueError(
                f"{field_name} must be greater than zero."
            )

    if config.number_of_epochs <= 0:
        raise ValueError(
            "number_of_epochs must be greater than zero."
        )

    if config.learning_rate <= 0:
        raise ValueError(
            "learning_rate must be greater than zero."
        )

    if config.weight_decay < 0:
        raise ValueError(
            "weight_decay cannot be negative."
        )

    if not 0 <= config.warmup_ratio < 1:
        raise ValueError(
            "warmup_ratio must be between 0 and 1."
        )

    if config.maximum_gradient_norm <= 0:
        raise ValueError(
            "maximum_gradient_norm must be greater than zero."
        )

    split_files = {
        config.train_file.resolve(),
        config.validation_file.resolve(),
        config.test_file.resolve(),
    }

    if len(split_files) != 3:
        raise ValueError(
            "Train, validation, and test files must be different."
        )


# ============================================================
# 5. DIGITAL FINGERPRINTS
# ============================================================

def create_sha256(content: bytes) -> str:
    """Create a SHA-256 fingerprint for file verification."""

    return hashlib.sha256(content).hexdigest()


def calculate_file_checksum(
    file_path: Path,
) -> str:
    """Calculate the SHA-256 checksum of one existing file."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    return create_sha256(
        file_path.read_bytes()
    )


# ============================================================
# 6. LOAD THE THREE DATASET SPLITS
# ============================================================

def load_phrasebank_splits(
    config: DistilBertTrainingConfig,
) -> DatasetDict:
    """
    Load train, validation, and test JSONL files.

    Output:
        Hugging Face DatasetDict containing three named splits.
    """

    data_files = {
        "train": str(config.train_file),
        "validation": str(
            config.validation_file
        ),
        "test": str(config.test_file),
    }

    for split_name, file_name in data_files.items():
        file_path = Path(file_name)

        if not file_path.exists():
            raise FileNotFoundError(
                f"{split_name} split file not found: "
                f"{file_path}"
            )

    dataset = load_dataset(
        "json",
        data_files=data_files,
    )

    validate_dataset_integrity(dataset)

    return dataset


# ============================================================
# 7. DATASET AND LEAKAGE VALIDATION
# ============================================================

def normalize_text(text: str) -> str:
    """Normalize text so duplicate sentences can be detected."""

    return " ".join(
        text.lower().split()
    )


def validate_dataset_integrity(
    dataset: DatasetDict,
) -> dict[str, dict[str, int]]:
    """
    Validate labels, split names, IDs, and sentence separation.

    Leakage rule:
        No record ID or normalized sentence may appear in more than
        one split.

    Output:
        Record and label counts for each split.
    """

    required_splits = {
        "train",
        "validation",
        "test",
    }

    if set(dataset.keys()) != required_splits:
        raise ValueError(
            "Dataset must contain train, validation, and test splits."
        )

    required_columns = {
        "record_id",
        "text",
        "label",
        "label_id",
        "split",
    }

    seen_record_ids: set[str] = set()
    seen_normalized_texts: set[str] = set()

    split_summary: dict[
        str,
        dict[str, int],
    ] = {}

    for split_name in (
        "train",
        "validation",
        "test",
    ):
        split_dataset = dataset[split_name]

        missing_columns = required_columns.difference(
            split_dataset.column_names
        )

        if missing_columns:
            raise ValueError(
                f"{split_name} split is missing columns: "
                + ", ".join(sorted(missing_columns))
            )

        label_counts = {
            "Bearish": 0,
            "Neutral": 0,
            "Bullish": 0,
        }

        for row_number, record in enumerate(
            split_dataset,
            start=1,
        ):
            record_id = str(
                record["record_id"]
            ).strip()

            text = " ".join(
                str(record["text"]).split()
            )

            label = str(
                record["label"]
            ).strip()

            label_id = int(
                record["label_id"]
            )

            declared_split = str(
                record["split"]
            ).strip()

            if not record_id:
                raise ValueError(
                    f"Empty record_id in {split_name} "
                    f"row {row_number}."
                )

            if not text:
                raise ValueError(
                    f"Empty text in {split_name} "
                    f"row {row_number}."
                )

            if declared_split != split_name:
                raise ValueError(
                    f"Record {record_id} declares split "
                    f"{declared_split}, but was loaded from "
                    f"{split_name}."
                )

            if label not in LABEL_TO_ID:
                raise ValueError(
                    f"Unsupported label in record "
                    f"{record_id}: {label}"
                )

            if label_id != LABEL_TO_ID[label]:
                raise ValueError(
                    f"Label and label_id disagree in "
                    f"record {record_id}."
                )

            if record_id in seen_record_ids:
                raise ValueError(
                    "Record ID leakage detected across "
                    f"dataset splits: {record_id}"
                )

            normalized_text = normalize_text(
                text
            )

            if normalized_text in seen_normalized_texts:
                raise ValueError(
                    "Sentence leakage detected across "
                    f"dataset splits: {record_id}"
                )

            seen_record_ids.add(record_id)
            seen_normalized_texts.add(
                normalized_text
            )

            label_counts[label] += 1

        if len(split_dataset) == 0:
            raise ValueError(
                f"{split_name} split is empty."
            )

        missing_labels = [
            label
            for label, count in label_counts.items()
            if count == 0
        ]

        if missing_labels:
            raise ValueError(
                f"{split_name} split has no examples for: "
                + ", ".join(missing_labels)
            )

        split_summary[split_name] = {
            "records": len(split_dataset),
            **label_counts,
        }

    return split_summary


# ============================================================
# 8. RESOLVE AN EXACT MODEL REVISION
# ============================================================

def resolve_model_revision(
    config: DistilBertTrainingConfig,
) -> str:
    """
    Resolve the exact Hugging Face commit used by the experiment.

    If config.model_revision is already a commit or tag, that
    revision is resolved and recorded.
    """

    requested_revision = (
        config.model_revision or "main"
    )

    if len(requested_revision) == 40 and all(
        character in "0123456789abcdef" for character in requested_revision.casefold()
    ):
        return requested_revision.casefold()

    model_info = HfApi().model_info(
        repo_id=config.model_id,
        revision=requested_revision,
    )

    if not model_info.sha:
        raise RuntimeError(
            "Could not resolve the model repository revision."
        )

    return model_info.sha


# ============================================================
# 9. TOKENIZER AND MODEL
# ============================================================

def load_tokenizer(
    config: DistilBertTrainingConfig,
    model_revision: str,
) -> Any:
    """Download the tokenizer from one exact model revision."""

    return AutoTokenizer.from_pretrained(
        config.model_id,
        revision=model_revision,
        use_fast=True,
    )


def load_model(
    config: DistilBertTrainingConfig,
    model_revision: str,
) -> torch.nn.Module:
    """
    Create a three-label DistilBERT classification model.

    The original language model weights are loaded from the pinned
    revision. The classification head is trained for this project.
    """

    return AutoModelForSequenceClassification.from_pretrained(
        config.model_id,
        revision=model_revision,
        num_labels=len(LABEL_NAMES),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )


# ============================================================
# 10. TOKENIZE THE DATASET
# ============================================================

def tokenize_phrasebank_splits(
    dataset: DatasetDict,
    tokenizer: Any,
    *,
    max_length: int,
) -> DatasetDict:
    """
    Convert sentence text into DistilBERT token IDs.

    Dynamic padding is applied later by the data collator, so short
    sentences are not padded to the maximum length unnecessarily.
    """

    def tokenize_batch(
        batch: dict[str, list[Any]],
    ) -> dict[str, Any]:
        tokenized_batch = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
        )

        # Trainer expects the target column to be named "labels".
        tokenized_batch["labels"] = batch[
            "label_id"
        ]

        return tokenized_batch

    source_columns = dataset[
        "train"
    ].column_names

    tokenized_dataset = dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=source_columns,
        load_from_cache_file=False,
        desc="Tokenizing Financial PhraseBank",
    )

    return tokenized_dataset


# ============================================================
# 11. CALCULATE CLASS WEIGHTS
# ============================================================

def calculate_class_weights(
    training_labels: Sequence[int],
) -> torch.Tensor:
    """
    Calculate balanced loss weights using NumPy only.

    Formula
    -------
    class weight =
        total training records
        /
        number of classes
        /
        records in that class

    A rare class receives a larger weight.
    A common class receives a smaller weight.

    Important
    ---------
    This implementation avoids importing scikit-learn into the same
    process as PyTorch. On this macOS environment, scikit-learn and
    PyTorch load incompatible OpenMP runtimes.
    """

    label_array = np.asarray(
        training_labels,
        dtype=np.int64,
    )

    if label_array.ndim != 1:
        raise ValueError(
            "Training labels must be one-dimensional."
        )

    if label_array.size == 0:
        raise ValueError(
            "Training labels cannot be empty."
        )

    if not np.isin(
        label_array,
        [0, 1, 2],
    ).all():
        raise ValueError(
            "Training labels may contain only 0, 1, and 2."
        )

    class_counts = np.bincount(
        label_array,
        minlength=len(LABEL_NAMES),
    ).astype(np.float64)

    if np.any(class_counts == 0):
        raise ValueError(
            "Training labels must contain classes 0, 1, and 2."
        )

    number_of_records = float(
        label_array.size
    )

    number_of_classes = float(
        len(LABEL_NAMES)
    )

    weights = (
        number_of_records
        / number_of_classes
        / class_counts
    )

    return torch.tensor(
        weights,
        dtype=torch.float32,
    )


# ============================================================
# 12. CLASS-WEIGHTED TRAINER
# ============================================================

class ClassWeightedTrainer(Trainer):
    """
    Hugging Face Trainer using weighted cross-entropy loss.

    The weights reduce the tendency to predict the largest Neutral
    class merely because it appears most frequently.
    """

    def __init__(
        self,
        *args: Any,
        class_weights: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            *args,
            **kwargs,
        )

        self.class_weights = class_weights

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Any | None = None,
    ) -> Any:
        """Calculate weighted cross-entropy for one batch."""

        del num_items_in_batch

        model_inputs = dict(inputs)

        labels = model_inputs.pop(
            "labels",
            None,
        )

        if labels is None:
            raise ValueError(
                "Training batch does not contain labels."
            )

        outputs = model(
            **model_inputs
        )

        logits = outputs.logits

        loss_function = torch.nn.CrossEntropyLoss(
            weight=(
                self.class_weights.to(logits.device)
                if self.class_weights is not None
                else None
            )
        )

        loss = loss_function(
            logits.view(
                -1,
                model.config.num_labels,
            ),
            labels.view(-1),
        )

        if return_outputs:
            return loss, outputs

        return loss


# ============================================================
# 13. NUMPY CLASSIFICATION STATISTICS
# ============================================================

def calculate_classification_statistics(
    true_labels: Sequence[int],
    predicted_labels: Sequence[int],
) -> dict[str, Any]:
    """
    Calculate classification statistics using NumPy only.

    Output
    ------
    Accuracy, per-class precision, recall, F1, support,
    macro averages, weighted averages, and confusion matrix.
    """

    true_array = np.asarray(
        true_labels,
        dtype=np.int64,
    )

    predicted_array = np.asarray(
        predicted_labels,
        dtype=np.int64,
    )

    if true_array.ndim != 1:
        raise ValueError(
            "True labels must be one-dimensional."
        )

    if predicted_array.ndim != 1:
        raise ValueError(
            "Predicted labels must be one-dimensional."
        )

    if true_array.size == 0:
        raise ValueError(
            "Evaluation labels cannot be empty."
        )

    if true_array.shape != predicted_array.shape:
        raise ValueError(
            "True and predicted labels must have equal length."
        )

    if not np.isin(
        true_array,
        [0, 1, 2],
    ).all():
        raise ValueError(
            "True labels may contain only 0, 1, and 2."
        )

    if not np.isin(
        predicted_array,
        [0, 1, 2],
    ).all():
        raise ValueError(
            "Predicted labels may contain only 0, 1, and 2."
        )

    number_of_classes = len(
        LABEL_NAMES
    )

    matrix = np.zeros(
        (
            number_of_classes,
            number_of_classes,
        ),
        dtype=np.int64,
    )

    # Rows represent true labels.
    # Columns represent predicted labels.
    np.add.at(
        matrix,
        (
            true_array,
            predicted_array,
        ),
        1,
    )

    true_positives = np.diag(
        matrix
    ).astype(np.float64)

    false_positives = (
        matrix.sum(axis=0)
        - true_positives
    )

    false_negatives = (
        matrix.sum(axis=1)
        - true_positives
    )

    support = matrix.sum(
        axis=1
    ).astype(np.float64)

    precision = np.divide(
        true_positives,
        true_positives + false_positives,
        out=np.zeros_like(
            true_positives,
            dtype=np.float64,
        ),
        where=(
            true_positives + false_positives
        ) != 0,
    )

    recall = np.divide(
        true_positives,
        true_positives + false_negatives,
        out=np.zeros_like(
            true_positives,
            dtype=np.float64,
        ),
        where=(
            true_positives + false_negatives
        ) != 0,
    )

    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(
            precision,
            dtype=np.float64,
        ),
        where=(
            precision + recall
        ) != 0,
    )

    total_records = float(
        matrix.sum()
    )

    accuracy = float(
        true_positives.sum()
        / total_records
    )

    macro_precision = float(
        precision.mean()
    )

    macro_recall = float(
        recall.mean()
    )

    macro_f1 = float(
        f1.mean()
    )

    support_weights = (
        support / total_records
    )

    weighted_precision = float(
        np.sum(
            precision * support_weights
        )
    )

    weighted_recall = float(
        np.sum(
            recall * support_weights
        )
    )

    weighted_f1 = float(
        np.sum(
            f1 * support_weights
        )
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support.astype(
            np.int64
        ),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "confusion_matrix": matrix,
    }


# ============================================================
# 14. TRAINER METRICS
# ============================================================

def compute_trainer_metrics(
    evaluation_prediction: EvalPrediction,
) -> dict[str, float]:
    """
    Calculate scalar metrics during validation and testing.

    Macro F1 treats all three classes equally.
    Weighted F1 also considers class frequency.
    """

    raw_predictions = (
        evaluation_prediction.predictions
    )

    if isinstance(
        raw_predictions,
        tuple,
    ):
        raw_predictions = raw_predictions[0]

    predicted_labels = np.argmax(
        raw_predictions,
        axis=-1,
    )

    statistics = (
        calculate_classification_statistics(
            evaluation_prediction.label_ids,
            predicted_labels,
        )
    )

    metrics: dict[str, float] = {
        "accuracy": float(
            statistics["accuracy"]
        ),
        "macro_precision": float(
            statistics["macro_precision"]
        ),
        "macro_recall": float(
            statistics["macro_recall"]
        ),
        "macro_f1": float(
            statistics["macro_f1"]
        ),
        "weighted_precision": float(
            statistics["weighted_precision"]
        ),
        "weighted_recall": float(
            statistics["weighted_recall"]
        ),
        "weighted_f1": float(
            statistics["weighted_f1"]
        ),
    }

    class_precision = statistics[
        "precision"
    ]

    class_recall = statistics[
        "recall"
    ]

    class_f1 = statistics[
        "f1"
    ]

    for label_id, label_name in enumerate(
        LABEL_NAMES
    ):
        metric_prefix = label_name.lower()

        metrics[
            f"{metric_prefix}_precision"
        ] = float(
            class_precision[label_id]
        )

        metrics[
            f"{metric_prefix}_recall"
        ] = float(
            class_recall[label_id]
        )

        metrics[
            f"{metric_prefix}_f1"
        ] = float(
            class_f1[label_id]
        )

    return metrics


# ============================================================
# 15. FULL TEST EVALUATION SUMMARY
# ============================================================

def build_evaluation_summary(
    true_labels: Sequence[int],
    predicted_labels: Sequence[int],
) -> dict[str, Any]:
    """
    Build a confusion matrix and complete per-class report.

    This richer structure is saved to JSON after final testing.
    """

    statistics = (
        calculate_classification_statistics(
            true_labels,
            predicted_labels,
        )
    )

    precision = statistics[
        "precision"
    ]

    recall = statistics[
        "recall"
    ]

    f1 = statistics[
        "f1"
    ]

    support = statistics[
        "support"
    ]

    report: dict[str, Any] = {}

    for label_id, label_name in enumerate(
        LABEL_NAMES
    ):
        report[label_name] = {
            "precision": float(
                precision[label_id]
            ),
            "recall": float(
                recall[label_id]
            ),
            "f1-score": float(
                f1[label_id]
            ),
            "support": int(
                support[label_id]
            ),
        }

    total_support = int(
        np.sum(support)
    )

    report["accuracy"] = float(
        statistics["accuracy"]
    )

    report["macro avg"] = {
        "precision": float(
            statistics["macro_precision"]
        ),
        "recall": float(
            statistics["macro_recall"]
        ),
        "f1-score": float(
            statistics["macro_f1"]
        ),
        "support": total_support,
    }

    report["weighted avg"] = {
        "precision": float(
            statistics["weighted_precision"]
        ),
        "recall": float(
            statistics["weighted_recall"]
        ),
        "f1-score": float(
            statistics["weighted_f1"]
        ),
        "support": total_support,
    }

    return {
        "label_order": LABEL_NAMES,
        "confusion_matrix": (
            statistics[
                "confusion_matrix"
            ].tolist()
        ),
        "classification_report": report,
    }


# ============================================================
# 15. MODEL PARAMETER COUNTS
# ============================================================

def count_model_parameters(
    model: torch.nn.Module,
) -> dict[str, int | float]:
    """
    Count total and trainable model parameters.

    The memory estimate counts parameter weights only in float32.
    It does not include gradients, optimizer states, or activations.
    """

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    estimated_parameter_memory_mb = (
        total_parameters
        * 4
        / (1024 ** 2)
    )

    return {
        "total_parameters": total_parameters,
        "trainable_parameters": (
            trainable_parameters
        ),
        "estimated_fp32_parameter_memory_mb": (
            round(
                estimated_parameter_memory_mb,
                2,
            )
        ),
    }


# ============================================================
# 16. CREATE TRAINING ARGUMENTS
# ============================================================

def create_training_arguments(
    config: DistilBertTrainingConfig,
) -> TrainingArguments:
    """
    Build reproducible Hugging Face Trainer settings.

    The validation metric used for selecting the best checkpoint is
    macro F1 because the dataset classes are imbalanced.
    """

    config.checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return TrainingArguments(
        output_dir=str(
            config.checkpoint_dir
        ),
        overwrite_output_dir=(
            config.overwrite_output_dir
        ),
        run_name=config.run_name,
        seed=config.random_seed,
        data_seed=config.random_seed,
        full_determinism=(
            config.full_determinism
        ),
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        max_grad_norm=(
            config.maximum_gradient_norm
        ),
        num_train_epochs=(
            config.number_of_epochs
        ),
        per_device_train_batch_size=(
            config.train_batch_size
        ),
        per_device_eval_batch_size=(
            config.evaluation_batch_size
        ),
        gradient_accumulation_steps=(
            config.gradient_accumulation_steps
        ),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=config.logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=(
            config.save_total_limit
        ),
        lr_scheduler_type="linear",
        optim="adamw_torch",
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        use_cpu=config.force_cpu,
        fp16=False,
        bf16=False,
        save_safetensors=True,
        report_to=[],
        push_to_hub=False,
    )


# ============================================================
# 17. CREATE THE TRAINER
# ============================================================

def create_trainer(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    tokenized_dataset: DatasetDict,
    training_arguments: TrainingArguments,
    class_weights: torch.Tensor | None,
) -> Trainer:
    """Create the class-weighted or standard Trainer."""

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
    )

    common_arguments = {
        "model": model,
        "args": training_arguments,
        "train_dataset": (
            tokenized_dataset["train"]
        ),
        "eval_dataset": (
            tokenized_dataset["validation"]
        ),
        "tokenizer": tokenizer,
        "data_collator": data_collator,
        "compute_metrics": (
            compute_trainer_metrics
        ),
    }

    if class_weights is not None:
        return ClassWeightedTrainer(
            **common_arguments,
            class_weights=class_weights,
        )

    return Trainer(
        **common_arguments
    )


# ============================================================
# 18. JSON SERIALIZATION HELPERS
# ============================================================

def make_json_safe(
    value: Any,
) -> Any:
    """Convert paths and NumPy values into JSON-compatible values."""

    if isinstance(value, Path):
        return str(value)

    if isinstance(
        value,
        np.generic,
    ):
        return value.item()

    if isinstance(value, dict):
        return {
            str(key): make_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(
        value,
        (list, tuple),
    ):
        return [
            make_json_safe(item)
            for item in value
        ]

    return value


def save_json(
    payload: dict[str, Any],
    output_path: Path,
) -> None:
    """Save a readable and deterministic JSON document."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            make_json_safe(payload),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def installed_version(
    package_name: str,
) -> str:
    """Return an installed package version or 'not-installed'."""

    try:
        return version(package_name)

    except PackageNotFoundError:
        return "not-installed"


# ============================================================
# 19. COMPLETE DISTILBERT TRAINING RUN
# ============================================================

def run_distilbert_training(
    config: DistilBertTrainingConfig | None = None,
) -> dict[str, Any]:
    """
    Run the complete DistilBERT sentiment experiment.

    This is the only function in this module that:

    - connects to Hugging Face;
    - downloads model assets;
    - begins model training;
    - writes final model artifacts.
    """

    if config is None:
        config = DistilBertTrainingConfig()

    validate_training_config(config)

    # Fix Python, NumPy, and PyTorch random seeds.
    set_seed(config.random_seed)

    # Load and verify untouched train, validation, and test files.
    dataset = load_phrasebank_splits(
        config
    )

    dataset_summary = validate_dataset_integrity(
        dataset
    )

    # Resolve one exact model commit before downloading weights.
    model_revision = resolve_model_revision(
        config
    )

    tokenizer = load_tokenizer(
        config,
        model_revision,
    )

    model = load_model(
        config,
        model_revision,
    )

    parameter_counts = count_model_parameters(
        model
    )

    training_labels = dataset[
        "train"
    ]["label_id"]

    class_weights = None

    if config.use_class_weights:
        class_weights = calculate_class_weights(
            training_labels
        )

    tokenized_dataset = tokenize_phrasebank_splits(
        dataset,
        tokenizer,
        max_length=config.max_length,
    )

    training_arguments = create_training_arguments(
        config
    )

    trainer = create_trainer(
        model=model,
        tokenizer=tokenizer,
        tokenized_dataset=tokenized_dataset,
        training_arguments=training_arguments,
        class_weights=class_weights,
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    training_started = perf_counter()

    training_result = trainer.train()

    training_seconds = (
        perf_counter() - training_started
    )

    # --------------------------------------------------------
    # Validate the best loaded checkpoint
    # --------------------------------------------------------

    validation_started = perf_counter()

    validation_metrics = trainer.evaluate(
        eval_dataset=(
            tokenized_dataset["validation"]
        ),
        metric_key_prefix="validation",
    )

    validation_seconds = (
        perf_counter() - validation_started
    )

    # --------------------------------------------------------
    # Evaluate the untouched test split once
    # --------------------------------------------------------

    test_started = perf_counter()

    test_prediction = trainer.predict(
        tokenized_dataset["test"],
        metric_key_prefix="test",
    )

    test_seconds = (
        perf_counter() - test_started
    )

    raw_test_predictions = (
        test_prediction.predictions
    )

    if isinstance(
        raw_test_predictions,
        tuple,
    ):
        raw_test_predictions = (
            raw_test_predictions[0]
        )

    predicted_test_labels = np.argmax(
        raw_test_predictions,
        axis=-1,
    )

    evaluation_summary = build_evaluation_summary(
        test_prediction.label_ids,
        predicted_test_labels,
    )

    test_record_count = len(
        tokenized_dataset["test"]
    )

    milliseconds_per_test_example = (
        test_seconds
        / test_record_count
        * 1000
    )

    # --------------------------------------------------------
    # Save model and tokenizer
    # --------------------------------------------------------

    config.final_model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    trainer.save_model(
        str(config.final_model_dir)
    )

    tokenizer.save_pretrained(
        str(config.final_model_dir)
    )

    trainer.save_state()

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    class_weight_values = None

    if class_weights is not None:
        class_weight_values = {
            label_name: float(
                class_weights[label_id]
            )
            for label_id, label_name in enumerate(
                LABEL_NAMES
            )
        }

    timing = {
        "training_seconds": training_seconds,
        "validation_seconds": (
            validation_seconds
        ),
        "test_inference_seconds": (
            test_seconds
        ),
        "test_milliseconds_per_example": (
            milliseconds_per_test_example
        ),
    }

    metrics_payload = {
        "training_metrics": (
            training_result.metrics
        ),
        "validation_metrics": (
            validation_metrics
        ),
        "test_metrics": (
            test_prediction.metrics
        ),
        "test_evaluation": (
            evaluation_summary
        ),
        "timing": timing,
    }

    save_json(
        metrics_payload,
        config.metrics_file,
    )

    # --------------------------------------------------------
    # Save reproducibility manifest
    # --------------------------------------------------------

    source_files = {
        "train": {
            "path": config.train_file,
            "checksum_sha256": (
                calculate_file_checksum(
                    config.train_file
                )
            ),
        },
        "validation": {
            "path": config.validation_file,
            "checksum_sha256": (
                calculate_file_checksum(
                    config.validation_file
                )
            ),
        },
        "test": {
            "path": config.test_file,
            "checksum_sha256": (
                calculate_file_checksum(
                    config.test_file
                )
            ),
        },
    }

    manifest = {
        "experiment_name": (
            config.experiment_name
        ),
        "model_family": (
            config.model_family
        ),
        "benchmark_role": (
            config.benchmark_role
        ),
        "status": "trained_and_evaluated",
        "model_id": config.model_id,
        "model_revision": model_revision,
        "labels": LABEL_TO_ID,
        "configuration": asdict(config),
        "dataset_summary": dataset_summary,
        "source_files": source_files,
        "class_weights": (
            class_weight_values
        ),
        "parameter_counts": (
            parameter_counts
        ),
        "device": str(
            trainer.args.device
        ),
        "timing": timing,
        "metrics_file": (
            config.metrics_file
        ),
        "final_model_directory": (
            config.final_model_dir
        ),
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": installed_version(
                "torch"
            ),
            "transformers": installed_version(
                "transformers"
            ),
            "datasets": installed_version(
                "datasets"
            ),
            "numpy": installed_version(
                "numpy"
            ),
            "huggingface_hub": installed_version(
                "huggingface-hub"
            ),
        },
    }

    save_json(
        manifest,
        config.manifest_file,
    )

    return make_json_safe(
        manifest
    )
