"""
Run full DistilBERT financial-sentiment training.

Purpose
-------
This runner launches the complete DistilBERT experiment using the
verified Financial PhraseBank train, validation, and test splits.

Inputs
------
- 2,413 training records.
- 517 validation records.
- 518 untouched testing records.

Each JSONL line represents one financial sentence with:

- record identifier;
- sentence text;
- Bearish, Neutral, or Bullish label;
- numerical label ID;
- assigned dataset split.

Processing
----------
1. Verify that the isolated Transformer environment is being used.
2. Verify that all three dataset files exist and are non-empty.
3. Prevent accidental overwriting of existing model artifacts.
4. Fine-tune DistilBERT for three epochs.
5. Select the best checkpoint using validation macro F1.
6. Evaluate the untouched test split.
7. Save model artifacts, metrics, timing, and provenance.

Outputs
-------
- Best training checkpoints.
- Final DistilBERT model and tokenizer.
- Validation and test metrics.
- Confusion matrix and per-class metrics.
- Reproducibility manifest.

Limitations
-----------
This Intel Mac uses CPU training, so the full run may take substantial
time.

No model-quality Intel Mac uses CPU training, so the full claim should be made until the full run finishes and
its saved metrics and artifacts are reviewed.
"""

import importlib.util
import json
import os
from pathlib import Path
from typing import Any

from threadpoolctl import threadpool_info


# ============================================================
# 1. TOKENIZER PROCESS SAFETY
# ============================================================

# Disable tokenizer worker parallelism before Transformers is loaded.
# This prevents fork-related warnings during evaluation.
os.environ.setdefault(
    "TOKENIZERS_PARALLELISM",
    "false",
)


# ============================================================
# 2. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SPLIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "transformer"
)

TRAIN_FILE = (
    SPLIT_DIR
    / "financial_phrasebank_train.jsonl"
)

VALIDATION_FILE = (
    SPLIT_DIR
    / "financial_phrasebank_validation.jsonl"
)

TEST_FILE = (
    SPLIT_DIR
    / "financial_phrasebank_test.jsonl"
)

MODEL_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
    / "distilbert_sentiment"
)

CHECKPOINT_DIR = (
    MODEL_ROOT / "checkpoints"
)

FINAL_MODEL_DIR = (
    MODEL_ROOT / "final_model"
)

METRICS_FILE = (
    PROJECT_ROOT
    / "reports"
    / "metrics"
    / "distilbert_sentiment_metrics.json"
)

MANIFEST_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "manifests"
    / "distilbert_sentiment_training_manifest.json"
)


# ============================================================
# 3. ENVIRONMENT VALIDATION
# ============================================================

def validate_isolated_environment() -> None:
    """
    Verify that the safe Transformer environment is active.

    Rules
    -----
    - scikit-learn must not be installed;
    - LLVM libomp must not be loaded.

    Reason
    ------
    PyTorch uses Intel OpenMP in this environment. Loading LLVM OpenMP
    in the same process could cause crashes or deadlocks.
    """

    sklearn_installed = (
        importlib.util.find_spec("sklearn")
        is not None
    )

    if sklearn_installed:
        raise RuntimeError(
            "scikit-learn is installed. Run this script with "
            ".venv-distilbert/bin/python."
        )

    openmp_entries = [
        entry
        for entry in threadpool_info()
        if entry.get("user_api") == "openmp"
    ]

    llvm_openmp_loaded = any(
        entry.get("prefix") == "libomp"
        for entry in openmp_entries
    )

    if llvm_openmp_loaded:
        raise RuntimeError(
            "Unsafe LLVM libomp runtime is loaded."
        )


# ============================================================
# 4. INPUT VALIDATION
# ============================================================

def validate_input_files() -> None:
    """
    Verify that all three dataset files exist and contain data.

    These files were created from the pinned Financial PhraseBank
    source and its reproducible stratified split.
    """

    input_files = {
        "train": TRAIN_FILE,
        "validation": VALIDATION_FILE,
        "test": TEST_FILE,
    }

    for split_name, file_path in input_files.items():
        if not file_path.exists():
            raise FileNotFoundError(
                f"{split_name} file not found: {file_path}"
            )

        if file_path.stat().st_size == 0:
            raise ValueError(
                f"{split_name} file is empty: {file_path}"
            )


# ============================================================
# 5. OUTPUT PROTECTION
# ============================================================

def validate_output_targets() -> None:
    """
    Prevent accidental replacement of completed experiment artifacts.

    Existing non-empty model directories or result files must be
    reviewed or removed deliberately before another full run.
    """

    artifact_directories = {
        "checkpoint": CHECKPOINT_DIR,
        "final model": FINAL_MODEL_DIR,
    }

    for artifact_name, directory in (
        artifact_directories.items()
    ):
        if (
            directory.exists()
            and any(directory.iterdir())
        ):
            raise FileExistsError(
                f"Existing {artifact_name} artifacts found: "
                f"{directory}"
            )

    result_files = {
        "metrics": METRICS_FILE,
        "manifest": MANIFEST_FILE,
    }

    for result_name, file_path in result_files.items():
        if file_path.exists():
            raise FileExistsError(
                f"Existing {result_name} file found: "
                f"{file_path}"
            )


# ============================================================
# 6. BUILD FULL TRAINING CONFIGURATION
# ============================================================

def build_training_config() -> Any:
    """
    Create the approved full DistilBERT configuration.

    Important settings
    ------------------
    - fixed random seed: 42;
    - maximum sequence length: 128;
    - three training epochs;
    - class-weighted loss;
    - best checkpoint selected by validation macro F1;
    - CPU execution on the verified Intel Mac environment.
    """

    from financial_news_intelligence.models.distilbert_training import (
        DistilBertTrainingConfig,
    )

    return DistilBertTrainingConfig(
        model_id=(
            "distilbert/distilbert-base-uncased"
        ),
        train_file=TRAIN_FILE,
        validation_file=VALIDATION_FILE,
        test_file=TEST_FILE,
        checkpoint_dir=CHECKPOINT_DIR,
        final_model_dir=FINAL_MODEL_DIR,
        metrics_file=METRICS_FILE,
        manifest_file=MANIFEST_FILE,
        run_name=(
            "distilbert_financial_phrasebank_full"
        ),
        random_seed=42,
        max_length=128,
        train_batch_size=8,
        evaluation_batch_size=16,
        gradient_accumulation_steps=2,
        number_of_epochs=3.0,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.10,
        maximum_gradient_norm=1.0,
        logging_steps=25,
        save_total_limit=2,
        use_class_weights=True,
        force_cpu=True,
        full_determinism=True,
        overwrite_output_dir=False,
    )


# ============================================================
# 7. READ SAVED JSON RESULTS
# ============================================================

def load_json_file(
    file_path: Path,
) -> dict[str, Any]:
    """Read one generated metrics or manifest JSON file."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"Expected result file not found: {file_path}"
        )

    return json.loads(
        file_path.read_text(
            encoding="utf-8",
        )
    )


# ============================================================
# 8. PRINT FINAL TRAINING SUMMARY
# ============================================================

def print_training_summary(
    manifest: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    """
    Print the main evidence required for model review.

    The complete per-class report and confusion matrix remain stored
    in the metrics JSON file.
    """

    validation_metrics = metrics[
        "validation_metrics"
    ]

    test_metrics = metrics[
        "test_metrics"
    ]

    test_evaluation = metrics[
        "test_evaluation"
    ]

    print()
    print("========================================")
    print("DISTILBERT FULL TRAINING SUMMARY")
    print("========================================")

    print(
        "Status:",
        manifest["status"],
    )

    print(
        "Model:",
        manifest["model_id"],
    )

    print(
        "Revision:",
        manifest["model_revision"],
    )

    print(
        "Device:",
        manifest["device"],
    )

    print(
        "Total parameters:",
        manifest[
            "parameter_counts"
        ]["total_parameters"],
    )

    print(
        "Training seconds:",
        round(
            float(
                manifest[
                    "timing"
                ]["training_seconds"]
            ),
            2,
        ),
    )

    print(
        "Validation macro F1:",
        round(
            float(
                validation_metrics[
                    "validation_macro_f1"
                ]
            ),
            6,
        ),
    )

    print(
        "Test accuracy:",
        round(
            float(
                test_metrics[
                    "test_accuracy"
                ]
            ),
            6,
        ),
    )

    print(
        "Test macro F1:",
        round(
            float(
                test_metrics[
                    "test_macro_f1"
                ]
            ),
            6,
        ),
    )

    print(
        "Test weighted F1:",
        round(
            float(
                test_metrics[
                    "test_weighted_f1"
                ]
            ),
            6,
        ),
    )

    print(
        "Confusion matrix:",
        test_evaluation[
            "confusion_matrix"
        ],
    )

    print(
        "Final model directory:",
        manifest[
            "final_model_directory"
        ],
    )

    print(
        "FULL DISTILBERT TRAINING: PASSED"
    )


# ============================================================
# 9. RUN THE COMPLETE EXPERIMENT
# ============================================================

def run_full_training() -> dict[str, Any]:
    """
    Validate, train, evaluate, save, and summarize DistilBERT.

    Output
    ------
    The generated reproducibility manifest.
    """

    validate_isolated_environment()
    validate_input_files()
    validate_output_targets()

    config = build_training_config()

    from financial_news_intelligence.models.distilbert_training import (
        run_distilbert_training,
    )

    manifest = run_distilbert_training(
        config
    )

    saved_manifest = load_json_file(
        MANIFEST_FILE
    )

    metrics = load_json_file(
        METRICS_FILE
    )

    print_training_summary(
        saved_manifest,
        metrics,
    )

    return manifest


# ============================================================
# 10. COMMAND-LINE ENTRY POINT
# ============================================================

def main() -> None:
    """Run full DistilBERT training from the command line."""

    run_full_training()


if __name__ == "__main__":
    main()
