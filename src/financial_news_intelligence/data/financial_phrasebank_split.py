"""
Create reproducible Financial PhraseBank dataset splits.

Purpose
-------
This module divides the normalized Financial PhraseBank records into:

- training data;
- validation data;
- testing data.

The split is stratified, which means every dataset part keeps a
similar proportion of Bearish, Neutral, and Bullish examples.

Input
-----
financial_phrasebank_75agree.jsonl created by the acquisition module.

Processing
----------
1. Load and validate every JSONL record.
2. Remove exact duplicate sentence text.
3. Reject duplicate text with conflicting sentiment labels.
4. Create a reproducible stratified 70 / 15 / 15 split.
5. Save train, validation, and test JSONL files.
6. Create checksums and a split manifest.

Output
------
- financial_phrasebank_train.jsonl
- financial_phrasebank_validation.jsonl
- financial_phrasebank_test.jsonl
- financial_phrasebank_split_manifest.json

Next use
--------
DistilBERT, full BERT, and LoRA experiments will use the same files.

Important
---------
Financial PhraseBank has no publication timestamps or event IDs.
Therefore, this module uses a fixed stratified random split rather
than chronological event-level splitting.
"""

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from sklearn.model_selection import train_test_split

from financial_news_intelligence.paths import (
    MANIFESTS_DIR,
    PROCESSED_DATA_DIR,
)


# ============================================================
# 1. DEFAULT FILE LOCATIONS
# ============================================================

# Normalized records created by financial_phrasebank.py.
DEFAULT_SOURCE_FILE = (
    PROCESSED_DATA_DIR
    / "financial_phrasebank"
    / "financial_phrasebank_75agree.jsonl"
)

# DistilBERT, BERT, and LoRA will read their split files here.
DEFAULT_OUTPUT_DIR = (
    PROCESSED_DATA_DIR / "transformer"
)

# The reproducibility manifest will be stored here.
DEFAULT_MANIFEST_DIR = MANIFESTS_DIR


# ============================================================
# 2. LABEL SETTINGS
# ============================================================

# The numerical label must agree with the project label.
LABEL_TO_ID = {
    "Bearish": 0,
    "Neutral": 1,
    "Bullish": 2,
}

# A fixed seed ensures the same records enter the same split.
DEFAULT_RANDOM_SEED = 42


# ============================================================
# 3. CREATE A DIGITAL FINGERPRINT
# ============================================================

def create_sha256(content: bytes) -> str:
    """
    Create a SHA-256 checksum.

    Same content:
        same checksum.

    Changed content:
        different checksum.
    """

    return hashlib.sha256(content).hexdigest()


# ============================================================
# 4. NORMALIZE SENTENCE TEXT
# ============================================================

def normalize_text(text: str) -> str:
    """
    Normalize text for duplicate detection.

    Example:
        "Revenue   increased strongly."
        becomes
        "revenue increased strongly."
    """

    return " ".join(text.lower().split())


# ============================================================
# 5. LOAD AND VALIDATE THE NORMALIZED DATASET
# ============================================================

def load_phrasebank_records(
    source_file: Path = DEFAULT_SOURCE_FILE,
) -> list[dict[str, object]]:
    """
    Load normalized Financial PhraseBank JSONL records.

    Input:
        Path to the normalized acquisition file.

    Output:
        Validated sentiment records.

    Validation:
        Every record must contain its ID, text, label, label ID,
        original source label, and sentence checksum.
    """

    if not source_file.exists():
        raise FileNotFoundError(
            f"Financial PhraseBank file not found: {source_file}"
        )

    required_fields = {
        "record_id",
        "text",
        "source_label",
        "label",
        "label_id",
        "text_checksum_sha256",
    }

    records: list[dict[str, object]] = []
    seen_record_ids: set[str] = set()

    with source_file.open(
        "r",
        encoding="utf-8",
    ) as dataset_file:
        for line_number, raw_line in enumerate(
            dataset_file,
            start=1,
        ):
            clean_line = raw_line.strip()

            if not clean_line:
                continue

            try:
                record = json.loads(clean_line)

            except json.JSONDecodeError as error:
                raise ValueError(
                    "Invalid JSON on Financial PhraseBank "
                    f"line {line_number}."
                ) from error

            missing_fields = required_fields.difference(record)

            if missing_fields:
                raise ValueError(
                    "Financial PhraseBank record is missing fields "
                    f"on line {line_number}: "
                    + ", ".join(sorted(missing_fields))
                )

            record_id = str(record["record_id"]).strip()
            text = " ".join(str(record["text"]).split())
            label = str(record["label"]).strip()

            if not record_id:
                raise ValueError(
                    f"Empty record_id on line {line_number}."
                )

            if record_id in seen_record_ids:
                raise ValueError(
                    f"Duplicate record_id found: {record_id}"
                )

            seen_record_ids.add(record_id)

            if not text:
                raise ValueError(
                    f"Empty sentence text on line {line_number}."
                )

            if label not in LABEL_TO_ID:
                raise ValueError(
                    f"Unsupported label on line {line_number}: {label}"
                )

            expected_label_id = LABEL_TO_ID[label]
            actual_label_id = int(record["label_id"])

            if actual_label_id != expected_label_id:
                raise ValueError(
                    "Label and label_id disagree "
                    f"on line {line_number}."
                )

            # Store a clean copy without changing the source file.
            clean_record = dict(record)
            clean_record["record_id"] = record_id
            clean_record["text"] = text
            clean_record["label"] = label
            clean_record["label_id"] = actual_label_id

            records.append(clean_record)

    if not records:
        raise ValueError(
            "Financial PhraseBank file contains no records."
        )

    return records


# ============================================================
# 6. REMOVE EXACT DUPLICATE SENTENCES
# ============================================================

def remove_duplicate_records(
    records: Sequence[dict[str, object]],
) -> tuple[list[dict[str, object]], int]:
    """
    Remove sentences containing exactly the same normalized text.

    Important:
        Identical text with different labels is rejected because the
        correct training answer would be ambiguous.

    Output:
        Unique records and the number of duplicates removed.
    """

    unique_records: list[dict[str, object]] = []

    # Store the label already connected to each normalized sentence.
    seen_text_labels: dict[str, str] = {}

    # Sorting makes duplicate handling reproducible.
    ordered_records = sorted(
        records,
        key=lambda record: str(record["record_id"]),
    )

    for record in ordered_records:
        normalized_text = normalize_text(
            str(record["text"])
        )

        current_label = str(record["label"])

        if normalized_text in seen_text_labels:
            previous_label = seen_text_labels[
                normalized_text
            ]

            # The same sentence cannot safely teach two answers.
            if previous_label != current_label:
                raise ValueError(
                    "Duplicate sentence has conflicting labels: "
                    f"{previous_label} and {current_label}."
                )

            # Same sentence and same label means this is a duplicate.
            continue

        seen_text_labels[normalized_text] = current_label
        unique_records.append(dict(record))

    duplicates_removed = (
        len(records) - len(unique_records)
    )

    return unique_records, duplicates_removed


# ============================================================
# 7. COUNT SENTIMENT LABELS
# ============================================================

def count_labels(
    records: Sequence[dict[str, object]],
) -> dict[str, int]:
    """
    Count Bearish, Neutral, and Bullish records.

    The fixed order makes manifests easier to compare.
    """

    counts = Counter(
        str(record["label"])
        for record in records
    )

    return {
        "Bearish": counts.get("Bearish", 0),
        "Neutral": counts.get("Neutral", 0),
        "Bullish": counts.get("Bullish", 0),
    }


# ============================================================
# 8. CREATE STRATIFIED DATASET SPLITS
# ============================================================

def split_phrasebank_records(
    records: Sequence[dict[str, object]],
    *,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> dict[str, list[dict[str, object]]]:
    """
    Divide records into stratified train, validation, and test splits.

    Stratified means:
        every split keeps approximately the same class proportions.

    Reproducible means:
        the same random seed creates the same split every time.
    """

    total_ratio = (
        train_ratio
        + validation_ratio
        + test_ratio
    )

    if round(total_ratio, 6) != 1.0:
        raise ValueError(
            "Train, validation, and test ratios must total 1.0."
        )

    if min(
        train_ratio,
        validation_ratio,
        test_ratio,
    ) <= 0:
        raise ValueError(
            "Every dataset split ratio must be greater than zero."
        )

    if len(records) < 3:
        raise ValueError(
            "At least three records are required."
        )

    # Sort before shuffling so the same input always starts identically.
    ordered_records = sorted(
        (dict(record) for record in records),
        key=lambda record: str(record["record_id"]),
    )

    labels = [
        str(record["label"])
        for record in ordered_records
    ]

    temporary_ratio = (
        validation_ratio + test_ratio
    )

    try:
        # First split:
        # 70% training and 30% temporary evaluation data.
        train_records, temporary_records = train_test_split(
            ordered_records,
            test_size=temporary_ratio,
            random_state=random_seed,
            shuffle=True,
            stratify=labels,
        )

        temporary_labels = [
            str(record["label"])
            for record in temporary_records
        ]

        # Divide the temporary 30% equally into validation and test.
        relative_test_ratio = (
            test_ratio / temporary_ratio
        )

        validation_records, test_records = train_test_split(
            temporary_records,
            test_size=relative_test_ratio,
            random_state=random_seed,
            shuffle=True,
            stratify=temporary_labels,
        )

    except ValueError as error:
        raise ValueError(
            "The dataset does not contain enough examples "
            "per label for stratified splitting."
        ) from error

    split_records = {
        "train": train_records,
        "validation": validation_records,
        "test": test_records,
    }

    # Attach the split name and sort records for stable output files.
    for split_name, records_in_split in split_records.items():
        prepared_records: list[dict[str, object]] = []

        for record in records_in_split:
            prepared_record = dict(record)
            prepared_record["split"] = split_name
            prepared_records.append(prepared_record)

        split_records[split_name] = sorted(
            prepared_records,
            key=lambda record: str(record["record_id"]),
        )

    return split_records


# ============================================================
# 9. SAVE ONE JSONL SPLIT
# ============================================================

def save_jsonl_split(
    records: Sequence[dict[str, object]],
    output_path: Path,
) -> str:
    """
    Save one record on each JSONL line.

    Output:
        SHA-256 checksum of the saved file.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_lines = [
        json.dumps(
            record,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        for record in records
    ]

    file_content = (
        "\n".join(json_lines) + "\n"
    ).encode("utf-8")

    output_path.write_bytes(file_content)

    return create_sha256(file_content)


# ============================================================
# 10. EXPORT THE COMPLETE SPLIT DATASET
# ============================================================

def export_phrasebank_splits(
    *,
    source_file: Path = DEFAULT_SOURCE_FILE,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    manifest_dir: Path = DEFAULT_MANIFEST_DIR,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = DEFAULT_RANDOM_SEED,
    created_at: datetime | None = None,
) -> dict[str, object]:
    """
    Create reproducible Transformer split files and their manifest.

    Input:
        Normalized Financial PhraseBank JSONL data.

    Output:
        Train, validation, test, and manifest files.

    Next:
        The same split files will be used by DistilBERT, BERT,
        and LoRA experiments.
    """

    if created_at is None:
        created_at = datetime.now(
            timezone.utc
        )

    if (
        created_at.tzinfo is None
        or created_at.utcoffset() is None
    ):
        raise ValueError(
            "created_at must include a timezone."
        )

    source_records = load_phrasebank_records(
        source_file
    )

    unique_records, duplicates_removed = (
        remove_duplicate_records(source_records)
    )

    splits = split_phrasebank_records(
        unique_records,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_details: dict[str, dict[str, object]] = {}

    # Save all three split files.
    for split_name in (
        "train",
        "validation",
        "test",
    ):
        records = splits[split_name]

        file_name = (
            f"financial_phrasebank_{split_name}.jsonl"
        )

        file_path = output_dir / file_name

        checksum = save_jsonl_split(
            records,
            file_path,
        )

        file_details[split_name] = {
            "file_name": file_name,
            "record_count": len(records),
            "label_counts": count_labels(records),
            "checksum_sha256": checksum,
        }

    # Create one checksum representing the full split package.
    dataset_summary = json.dumps(
        file_details,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    dataset_checksum = create_sha256(
        dataset_summary
    )

    manifest: dict[str, object] = {
        "dataset_name": "Financial PhraseBank",
        "configuration": "sentences_75agree",
        "created_at": created_at.isoformat(),
        "source_file": str(source_file),
        "source_checksum_sha256": create_sha256(
            source_file.read_bytes()
        ),
        "source_record_count": len(source_records),
        "duplicates_removed": duplicates_removed,
        "final_record_count": len(unique_records),
        "split_strategy": "stratified_random",
        "random_seed": random_seed,
        "ratios": {
            "train": train_ratio,
            "validation": validation_ratio,
            "test": test_ratio,
        },
        "overall_label_counts": count_labels(
            unique_records
        ),
        "files": file_details,
        "dataset_checksum_sha256": dataset_checksum,
        "limitations": [
            "No publication timestamps are available.",
            "No event-level grouping is available.",
            "The split is stratified and reproducible.",
            "The dataset is restricted to sentiment modelling.",
        ],
    }

    manifest_path = (
        manifest_dir
        / "financial_phrasebank_split_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return manifest
