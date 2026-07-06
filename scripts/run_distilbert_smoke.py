"""
Run a small end-to-end DistilBERT sentiment-training smoke test.

Purpose
-------
This runner verifies that the complete training pipeline can:

- read real Financial PhraseBank split records;
- create small balanced smoke datasets;
- download the pinned DistilBERT model and tokenizer;
- tokenize financial sentences;
- complete forward and backward training passes;
- evaluate validation and test data;
- save the best checkpoint and final model;
- write metrics and a reproducibility manifest.

Inputs
------
The verified Financial PhraseBank JSONL split files:

- financial_phrasebank_train.jsonl;
- financial_phrasebank_validation.jsonl;
- financial_phrasebank_test.jsonl.

Smoke dataset grain
-------------------
Each JSONL line represents one financial sentence with one sentiment
label.

The smoke run selects:

- 9 records per label for training;
- 3 records per label for validation;
- 3 records per label for testing.

Outputs
-------
- Temporary balanced smoke JSONL files.
- Best validation checkpoint.
- Final smoke model and tokenizer.
- Smoke metrics JSON.
- Smoke training manifest.

Limitations
-----------
The smoke dataset is intentionally tiny.

Its accuracy and F1 values are not evidence of production model
quality. The run proves pipeline execution only.
"""

import json
from pathlib import Path
from typing import Any

from financial_news_intelligence.models.distilbert_training import (
    DistilBertTrainingConfig,
    run_distilbert_training,
)


# ============================================================
# 1. PROJECT LOCATIONS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "transformer"
)

SMOKE_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "distilbert_smoke"
)

SMOKE_MODEL_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
    / "distilbert_smoke"
)

SMOKE_METRICS_FILE = (
    PROJECT_ROOT
    / "reports"
    / "metrics"
    / "distilbert_smoke_metrics.json"
)

SMOKE_MANIFEST_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "manifests"
    / "distilbert_smoke_manifest.json"
)


# ============================================================
# 2. SENTIMENT LABEL SETTINGS
# ============================================================

LABEL_ORDER = (
    "Bearish",
    "Neutral",
    "Bullish",
)


# ============================================================
# 3. LOAD JSONL RECORDS
# ============================================================

def load_jsonl_records(
    source_file: Path,
) -> list[dict[str, Any]]:
    """
    Load one JSON object from each non-empty line.

    Input
    -----
    source_file:
        Verified Financial PhraseBank split file.

    Output
    ------
    A list of sentiment records.

    Validation
    ----------
    The source file must exist and contain at least one record.
    """

    if not source_file.exists():
        raise FileNotFoundError(
            f"Smoke source file not found: {source_file}"
        )

    records: list[dict[str, Any]] = []

    with source_file.open(
        "r",
        encoding="utf-8",
    ) as input_file:
        for line_number, raw_line in enumerate(
            input_file,
            start=1,
        ):
            clean_line = raw_line.strip()

            if not clean_line:
                continue

            try:
                record = json.loads(clean_line)

            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: "
                    f"{source_file}"
                ) from error

            records.append(record)

    if not records:
        raise ValueError(
            f"Smoke source file contains no records: {source_file}"
        )

    return records


# ============================================================
# 4. SELECT A BALANCED SMOKE SAMPLE
# ============================================================

def select_balanced_records(
    records: list[dict[str, Any]],
    *,
    split_name: str,
    examples_per_label: int,
) -> list[dict[str, Any]]:
    """
    Select the first fixed number of records from each label.

    Inputs
    ------
    records:
        Records from one verified source split.

    split_name:
        train, validation, or test.

    examples_per_label:
        Number of records required from each sentiment class.

    Output
    ------
    A deterministic balanced sample.

    Assumption
    ----------
    Every source split contains enough examples for all three labels.
    """

    if examples_per_label <= 0:
        raise ValueError(
            "examples_per_label must be greater than zero."
        )

    selected: dict[
        str,
        list[dict[str, Any]],
    ] = {
        label: []
        for label in LABEL_ORDER
    }

    for source_record in records:
        label = str(
            source_record.get(
                "label",
                "",
            )
        )

        if label not in selected:
            continue

        if len(selected[label]) >= examples_per_label:
            continue

        prepared_record = dict(
            source_record
        )

        # The destination split must match the smoke file that will
        # contain this record.
        prepared_record["split"] = split_name

        selected[label].append(
            prepared_record
        )

        if all(
            len(label_records)
            == examples_per_label
            for label_records in selected.values()
        ):
            break

    missing_counts = {
        label: (
            examples_per_label
            - len(label_records)
        )
        for label, label_records in selected.items()
        if len(label_records) < examples_per_label
    }

    if missing_counts:
        raise ValueError(
            "Not enough records for balanced smoke data: "
            f"{missing_counts}"
        )

    balanced_records: list[
        dict[str, Any]
    ] = []

    # Fixed label order makes generated files reproducible.
    for label in LABEL_ORDER:
        balanced_records.extend(
            selected[label]
        )

    return balanced_records


# ============================================================
# 5. SAVE ONE SMOKE SPLIT
# ============================================================

def save_jsonl_records(
    records: list[dict[str, Any]],
    output_file: Path,
) -> None:
    """
    Save one record on each JSONL line.

    The file is overwritten intentionally because smoke datasets are
    deterministic temporary artifacts.
    """

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_lines = [
        json.dumps(
            record,
            sort_keys=True,
            ensure_ascii=False,
        )
        for record in records
    ]

    output_file.write_text(
        "\n".join(json_lines) + "\n",
        encoding="utf-8",
    )


# ============================================================
# 6. CREATE ALL SMOKE DATASETS
# ============================================================

def create_smoke_datasets() -> dict[str, Path]:
    """
    Create balanced train, validation, and test smoke files.

    Output
    ------
    Paths to the three generated JSONL files.
    """

    split_settings = {
        "train": {
            "source_name": (
                "financial_phrasebank_train.jsonl"
            ),
            "examples_per_label": 9,
        },
        "validation": {
            "source_name": (
                "financial_phrasebank_validation.jsonl"
            ),
            "examples_per_label": 3,
        },
        "test": {
            "source_name": (
                "financial_phrasebank_test.jsonl"
            ),
            "examples_per_label": 3,
        },
    }

    output_files: dict[str, Path] = {}

    for split_name, settings in split_settings.items():
        source_file = (
            SOURCE_DATA_DIR
            / str(settings["source_name"])
        )

        source_records = load_jsonl_records(
            source_file
        )

        balanced_records = select_balanced_records(
            source_records,
            split_name=split_name,
            examples_per_label=int(
                settings[
                    "examples_per_label"
                ]
            ),
        )

        output_file = (
            SMOKE_DATA_DIR
            / f"{split_name}.jsonl"
        )

        save_jsonl_records(
            balanced_records,
            output_file,
        )

        output_files[split_name] = output_file

        print(
            f"{split_name.upper()}: "
            f"{len(balanced_records)} records"
        )

    return output_files


# ============================================================
# 7. BUILD THE SMOKE CONFIGURATION
# ============================================================

def build_smoke_config(
    smoke_files: dict[str, Path],
) -> DistilBertTrainingConfig:
    """
    Create the small CPU-only DistilBERT configuration.

    Thresholds
    ----------
    - one training epoch;
    - sequence length limited to 64 tokens;
    - batch size of 3;
    - no gradient accumulation;
    - deterministic seed 42.

    These values minimize smoke runtime while still exercising the
    full training pipeline.
    """

    return DistilBertTrainingConfig(
        train_file=smoke_files["train"],
        validation_file=(
            smoke_files["validation"]
        ),
        test_file=smoke_files["test"],
        checkpoint_dir=(
            SMOKE_MODEL_DIR
            / "best_checkpoint"
        ),
        final_model_dir=(
            SMOKE_MODEL_DIR
            / "final_model"
        ),
        metrics_file=SMOKE_METRICS_FILE,
        manifest_file=SMOKE_MANIFEST_FILE,
        random_seed=42,
        max_length=64,
        train_batch_size=3,
        evaluation_batch_size=3,
        gradient_accumulation_steps=1,
        number_of_epochs=1,
        learning_rate=2e-5,
        logging_steps=1,
        use_class_weights=True,
        force_cpu=True,
        full_determinism=True,
        overwrite_output_dir=True,
    )


# ============================================================
# 8. RUN AND SUMMARIZE THE SMOKE TEST
# ============================================================

def run_smoke_test() -> dict[str, Any]:
    """
    Run the complete smoke experiment.

    Output
    ------
    The reproducibility manifest returned by the training pipeline.
    """

    smoke_files = create_smoke_datasets()

    config = build_smoke_config(
        smoke_files
    )

    manifest = run_distilbert_training(
        config
    )

    metrics = json.loads(
        SMOKE_METRICS_FILE.read_text(
            encoding="utf-8",
        )
    )

    print()
    print(
        "Training engine:",
        manifest.get(
            "training_engine",
            "huggingface_trainer",
        ),
    )

    print(
        "Device:",
        manifest["device"],
    )

    print(
        "Best epoch:",
        metrics.get(
            "best_epoch",
            "recorded by Trainer",
        ),
    )

    test_metrics = metrics.get(
        "test_metrics",
        {},
    )

    macro_f1 = (
        test_metrics.get("test_macro_f1")
        if "test_macro_f1" in test_metrics
        else test_metrics.get("macro_f1")
    )

    print(
        "Test macro F1:",
        macro_f1,
    )

    print(
        "Final model saved:",
        Path(
            manifest[
                "final_model_directory"
            ]
        ).exists(),
    )

    print(
        "DISTILBERT SMOKE TRAINING: PASSED"
    )

    return manifest


# ============================================================
# 9. COMMAND-LINE ENTRY POINT
# ============================================================

def main() -> None:
    """Run the smoke test when this file is executed directly."""

    run_smoke_test()


if __name__ == "__main__":
    main()
