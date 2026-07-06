"""
Test the DistilBERT sentiment-training foundation.

Purpose
-------
These tests verify the reusable logic required before model training:

- configuration validation;
- balanced class-weight calculation;
- accuracy, precision, recall, and F1 calculations;
- confusion-matrix construction;
- Financial PhraseBank split validation;
- record-ID and sentence-leakage protection.

The tests do not:

- connect to Hugging Face;
- download model weights;
- tokenize the full dataset;
- start training;
- write model checkpoints.

Environment
-----------
Run this file with .venv-distilbert.

That environment intentionally excludes scikit-learn to prevent the
Intel OpenMP and LLVM OpenMP runtime collision found in the main
project environment.
"""

from pathlib import Path

import numpy as np
import pytest
from datasets import Dataset, DatasetDict

from financial_news_intelligence.models.distilbert_training import (
    DistilBertTrainingConfig,
    build_evaluation_summary,
    calculate_class_weights,
    calculate_classification_statistics,
    validate_dataset_integrity,
    validate_training_config,
)


# ============================================================
# 1. TEST DATA HELPERS
# ============================================================

LABELS = (
    ("Bearish", 0),
    ("Neutral", 1),
    ("Bullish", 2),
)


def create_split(
    split_name: str,
    record_prefix: str,
    *,
    first_text: str | None = None,
) -> Dataset:
    """
    Create one tiny valid sentiment split.

    Each split contains:
        one Bearish record;
        one Neutral record;
        one Bullish record.

    Inputs
    ------
    split_name:
        The declared dataset split.

    record_prefix:
        A unique prefix preventing accidental ID collisions.

    first_text:
        Optional text used to deliberately test sentence leakage.
    """

    records: list[dict[str, object]] = []

    for index, (label, label_id) in enumerate(
        LABELS,
        start=1,
    ):
        text = (
            first_text
            if index == 1 and first_text is not None
            else (
                f"{record_prefix} financial sentence "
                f"for {label}."
            )
        )

        records.append(
            {
                "record_id": (
                    f"{record_prefix}_{index:03d}"
                ),
                "text": text,
                "label": label,
                "label_id": label_id,
                "split": split_name,
            }
        )

    return Dataset.from_list(records)


def create_valid_dataset() -> DatasetDict:
    """
    Create three valid, separate dataset splits.

    Output
    ------
    A DatasetDict containing train, validation, and test data.
    """

    return DatasetDict(
        {
            "train": create_split(
                "train",
                "train",
            ),
            "validation": create_split(
                "validation",
                "validation",
            ),
            "test": create_split(
                "test",
                "test",
            ),
        }
    )


# ============================================================
# 2. CONFIGURATION VALIDATION
# ============================================================

def test_default_training_configuration_is_valid() -> None:
    """
    Prepare:
        Create the default DistilBERT configuration.

    Run:
        Validate its values.

    Check:
        No exception is raised and the expected model is selected.
    """

    config = DistilBertTrainingConfig()

    validate_training_config(config)

    assert config.model_id == (
        "distilbert/distilbert-base-uncased"
    )

    assert config.max_length == 128
    assert config.random_seed == 42


def test_split_files_must_be_different(
    tmp_path: Path,
) -> None:
    """
    Prepare:
        Point train, validation, and test to the same file.

    Run:
        Validate the configuration.

    Check:
        The unsafe configuration is rejected.
    """

    shared_file = tmp_path / "shared.jsonl"

    config = DistilBertTrainingConfig(
        train_file=shared_file,
        validation_file=shared_file,
        test_file=shared_file,
    )

    with pytest.raises(
        ValueError,
        match="must be different",
    ):
        validate_training_config(config)


# ============================================================
# 3. CLASS-WEIGHT CALCULATION
# ============================================================

def test_class_weights_follow_balanced_formula() -> None:
    """
    Prepare:
        Create labels with different class frequencies.

    Run:
        Calculate balanced class weights.

    Check:
        Rare classes receive larger weights.
    """

    labels = [
        0,
        0,
        1,
        1,
        1,
        1,
        2,
        2,
        2,
    ]

    weights = calculate_class_weights(
        labels
    )

    np.testing.assert_allclose(
        weights.numpy(),
        np.array(
            [
                1.5,
                0.75,
                1.0,
            ],
            dtype=np.float32,
        ),
        rtol=1e-6,
    )


def test_class_weights_require_all_labels() -> None:
    """
    Prepare:
        Create labels without the Bullish class.

    Run:
        Calculate class weights.

    Check:
        Training is blocked because one class is missing.
    """

    with pytest.raises(
        ValueError,
        match="classes 0, 1, and 2",
    ):
        calculate_class_weights(
            [
                0,
                0,
                1,
                1,
            ]
        )


# ============================================================
# 4. CLASSIFICATION METRICS
# ============================================================

def test_perfect_predictions_produce_perfect_metrics() -> None:
    """
    Prepare:
        Give every record the correct prediction.

    Run:
        Calculate classification statistics.

    Check:
        Accuracy and macro F1 equal one.
    """

    statistics = (
        calculate_classification_statistics(
            true_labels=[
                0,
                1,
                2,
            ],
            predicted_labels=[
                0,
                1,
                2,
            ],
        )
    )

    assert statistics["accuracy"] == 1.0
    assert statistics["macro_f1"] == 1.0
    assert statistics["weighted_f1"] == 1.0

    assert (
        statistics[
            "confusion_matrix"
        ].tolist()
        == [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
    )


def test_known_predictions_create_expected_matrix() -> None:
    """
    Prepare:
        Create one correct and one incorrect result per class.

    Run:
        Calculate metrics and build the evaluation report.

    Check:
        The matrix and macro F1 match the known result.
    """

    true_labels = [
        0,
        0,
        1,
        1,
        2,
        2,
    ]

    predicted_labels = [
        0,
        1,
        1,
        2,
        2,
        0,
    ]

    statistics = (
        calculate_classification_statistics(
            true_labels,
            predicted_labels,
        )
    )

    summary = build_evaluation_summary(
        true_labels,
        predicted_labels,
    )

    assert statistics["accuracy"] == 0.5
    assert statistics["macro_f1"] == 0.5

    assert summary["confusion_matrix"] == [
        [1, 1, 0],
        [0, 1, 1],
        [1, 0, 1],
    ]

    assert summary["label_order"] == [
        "Bearish",
        "Neutral",
        "Bullish",
    ]


# ============================================================
# 5. DATASET INTEGRITY
# ============================================================

def test_valid_dataset_returns_split_summary() -> None:
    """
    Prepare:
        Create three separate valid splits.

    Run:
        Validate dataset integrity.

    Check:
        Record and class counts are returned correctly.
    """

    dataset = create_valid_dataset()

    summary = validate_dataset_integrity(
        dataset
    )

    assert summary["train"] == {
        "records": 3,
        "Bearish": 1,
        "Neutral": 1,
        "Bullish": 1,
    }

    assert summary["validation"][
        "records"
    ] == 3

    assert summary["test"][
        "records"
    ] == 3


def test_sentence_leakage_between_splits_is_rejected() -> None:
    """
    Prepare:
        Put the same normalized sentence in train and validation.

    Run:
        Validate dataset integrity.

    Check:
        Cross-split sentence leakage is rejected.
    """

    shared_sentence = (
        "The company reported weaker quarterly revenue."
    )

    dataset = DatasetDict(
        {
            "train": create_split(
                "train",
                "train",
                first_text=shared_sentence,
            ),
            "validation": create_split(
                "validation",
                "validation",
                first_text=(
                    "  THE COMPANY reported weaker "
                    "quarterly revenue.  "
                ),
            ),
            "test": create_split(
                "test",
                "test",
            ),
        }
    )

    with pytest.raises(
        ValueError,
        match="Sentence leakage detected",
    ):
        validate_dataset_integrity(
            dataset
        )
