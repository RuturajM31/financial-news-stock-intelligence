"""
Test reproducible Financial PhraseBank dataset splitting.

Purpose
-------
These tests verify that the split module:

- loads normalized sentiment records;
- removes exact duplicate sentences;
- rejects duplicate text with conflicting labels;
- preserves class proportions through stratified splitting;
- creates identical splits when the random seed is unchanged;
- writes train, validation, test, and manifest files;
- records accurate counts and SHA-256 checksums;
- rejects unsafe split settings and timestamps.

Important
---------
These tests use artificial records stored in temporary folders.

They do not modify the real Financial PhraseBank files and do not
train DistilBERT, BERT, or LoRA models.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from financial_news_intelligence.data.financial_phrasebank_split import (
    count_labels,
    create_sha256,
    export_phrasebank_splits,
    load_phrasebank_records,
    remove_duplicate_records,
    split_phrasebank_records,
)


# ============================================================
# 1. TEST RECORD HELPERS
# ============================================================

def make_record(
    record_number: int,
    label: str,
    *,
    text: str | None = None,
) -> dict[str, object]:
    """
    Create one normalized Financial PhraseBank-style record.

    Input:
        Record number, project label, and optional sentence text.

    Output:
        One complete record ready for testing.
    """

    label_to_id = {
        "Bearish": 0,
        "Neutral": 1,
        "Bullish": 2,
    }

    source_label = {
        "Bearish": "negative",
        "Neutral": "neutral",
        "Bullish": "positive",
    }[label]

    if text is None:
        text = (
            f"Financial sentence number {record_number} "
            f"has the label {label}."
        )

    return {
        "record_id": f"fpb_{record_number:06d}",
        "text": text,
        "source_label": source_label,
        "label": label,
        "label_id": label_to_id[label],
        "text_checksum_sha256": create_sha256(
            text.encode("utf-8")
        ),
    }


def make_balanced_records() -> list[dict[str, object]]:
    """
    Create sixty balanced records.

    Distribution:
        20 Bearish
        20 Neutral
        20 Bullish
    """

    records: list[dict[str, object]] = []

    record_number = 1

    for label in (
        "Bearish",
        "Neutral",
        "Bullish",
    ):
        for _ in range(20):
            records.append(
                make_record(
                    record_number,
                    label,
                )
            )

            record_number += 1

    return records


def write_jsonl(
    records: list[dict[str, object]],
    file_path: Path,
) -> None:
    """Save testing records as one JSON object per line."""

    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    lines = [
        json.dumps(
            record,
            sort_keys=True,
        )
        for record in records
    ]

    file_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


# ============================================================
# 2. LOAD NORMALIZED RECORDS
# ============================================================

def test_normalized_records_are_loaded(
    tmp_path: Path,
) -> None:
    """A valid JSONL source file should load successfully."""

    # Prepare three valid records.
    source_file = tmp_path / "phrasebank.jsonl"

    records = [
        make_record(1, "Bearish"),
        make_record(2, "Neutral"),
        make_record(3, "Bullish"),
    ]

    write_jsonl(
        records,
        source_file,
    )

    # Run the loader.
    loaded_records = load_phrasebank_records(
        source_file
    )

    # Check the result.
    assert len(loaded_records) == 3

    assert count_labels(loaded_records) == {
        "Bearish": 1,
        "Neutral": 1,
        "Bullish": 1,
    }


# ============================================================
# 3. EXACT DUPLICATE REMOVAL
# ============================================================

def test_exact_duplicate_sentence_is_removed() -> None:
    """The same normalized sentence and label should appear once."""

    shared_text = (
        "The company reported stronger quarterly revenue."
    )

    records = [
        make_record(
            1,
            "Bullish",
            text=shared_text,
        ),
        make_record(
            2,
            "Bullish",
            text=(
                "  THE COMPANY reported stronger "
                "quarterly revenue.  "
            ),
        ),
    ]

    unique_records, duplicates_removed = (
        remove_duplicate_records(records)
    )

    assert len(unique_records) == 1
    assert duplicates_removed == 1
    assert unique_records[0]["record_id"] == "fpb_000001"


# ============================================================
# 4. CONFLICTING DUPLICATE LABELS
# ============================================================

def test_conflicting_duplicate_labels_are_rejected() -> None:
    """Identical text cannot safely teach two different answers."""

    shared_text = (
        "The company announced its quarterly results."
    )

    records = [
        make_record(
            1,
            "Bullish",
            text=shared_text,
        ),
        make_record(
            2,
            "Bearish",
            text=shared_text,
        ),
    ]

    with pytest.raises(
        ValueError,
        match="conflicting labels",
    ):
        remove_duplicate_records(records)


# ============================================================
# 5. STRATIFIED AND REPRODUCIBLE SPLITTING
# ============================================================

def test_split_is_stratified_and_reproducible() -> None:
    """
    The same seed should create the same balanced split every time.

    Sixty balanced records become:

    - 42 training records;
    - 9 validation records;
    - 9 testing records.
    """

    records = make_balanced_records()

    first_result = split_phrasebank_records(
        records,
        random_seed=42,
    )

    second_result = split_phrasebank_records(
        records,
        random_seed=42,
    )

    assert len(first_result["train"]) == 42
    assert len(first_result["validation"]) == 9
    assert len(first_result["test"]) == 9

    assert count_labels(first_result["train"]) == {
        "Bearish": 14,
        "Neutral": 14,
        "Bullish": 14,
    }

    assert count_labels(first_result["validation"]) == {
        "Bearish": 3,
        "Neutral": 3,
        "Bullish": 3,
    }

    assert count_labels(first_result["test"]) == {
        "Bearish": 3,
        "Neutral": 3,
        "Bullish": 3,
    }

    # Compare record IDs to prove exact reproducibility.
    for split_name in (
        "train",
        "validation",
        "test",
    ):
        first_ids = [
            record["record_id"]
            for record in first_result[split_name]
        ]

        second_ids = [
            record["record_id"]
            for record in second_result[split_name]
        ]

        assert first_ids == second_ids


# ============================================================
# 6. COMPLETE FILE AND MANIFEST EXPORT
# ============================================================

def test_export_creates_files_counts_and_checksums(
    tmp_path: Path,
) -> None:
    """The complete exporter should produce verified split files."""

    records = make_balanced_records()

    # Add one same-label duplicate.
    duplicate = make_record(
        61,
        "Bearish",
        text=str(records[0]["text"]),
    )

    records.append(duplicate)

    source_file = tmp_path / "source.jsonl"
    output_dir = tmp_path / "transformer"
    manifest_dir = tmp_path / "manifests"

    write_jsonl(
        records,
        source_file,
    )

    manifest = export_phrasebank_splits(
        source_file=source_file,
        output_dir=output_dir,
        manifest_dir=manifest_dir,
        random_seed=42,
        created_at=datetime(
            2024,
            7,
            1,
            12,
            0,
            tzinfo=timezone.utc,
        ),
    )

    assert manifest["source_record_count"] == 61
    assert manifest["duplicates_removed"] == 1
    assert manifest["final_record_count"] == 60

    assert (
        manifest["files"]["train"]["record_count"]
        == 42
    )

    assert (
        manifest["files"]["validation"]["record_count"]
        == 9
    )

    assert (
        manifest["files"]["test"]["record_count"]
        == 9
    )

    # Check every saved file against its recorded checksum.
    for split_name in (
        "train",
        "validation",
        "test",
    ):
        file_details = manifest["files"][
            split_name
        ]

        file_path = (
            output_dir
            / file_details["file_name"]
        )

        assert file_path.exists()

        actual_checksum = create_sha256(
            file_path.read_bytes()
        )

        assert (
            file_details["checksum_sha256"]
            == actual_checksum
        )

    manifest_file = (
        manifest_dir
        / "financial_phrasebank_split_manifest.json"
    )

    assert manifest_file.exists()

    assert len(
        manifest["dataset_checksum_sha256"]
    ) == 64


# ============================================================
# 7. INVALID SPLIT RATIOS
# ============================================================

def test_invalid_split_ratios_are_rejected() -> None:
    """The train, validation, and test ratios must total 100%."""

    records = make_balanced_records()

    with pytest.raises(
        ValueError,
        match="must total 1.0",
    ):
        split_phrasebank_records(
            records,
            train_ratio=0.70,
            validation_ratio=0.20,
            test_ratio=0.20,
        )


# ============================================================
# 8. MISSING SOURCE FILE
# ============================================================

def test_missing_source_file_is_rejected(
    tmp_path: Path,
) -> None:
    """The split pipeline cannot run without normalized records."""

    missing_file = (
        tmp_path / "missing_phrasebank.jsonl"
    )

    with pytest.raises(
        FileNotFoundError,
        match="Financial PhraseBank file not found",
    ):
        load_phrasebank_records(
            missing_file
        )


# ============================================================
# 9. TIMEZONE-FREE CREATION TIME
# ============================================================

def test_created_at_requires_timezone(
    tmp_path: Path,
) -> None:
    """The split manifest timestamp must identify its timezone."""

    source_file = tmp_path / "source.jsonl"

    write_jsonl(
        make_balanced_records(),
        source_file,
    )

    with pytest.raises(
        ValueError,
        match="created_at must include a timezone",
    ):
        export_phrasebank_splits(
            source_file=source_file,
            output_dir=tmp_path / "output",
            manifest_dir=tmp_path / "manifests",
            created_at=datetime(
                2024,
                7,
                1,
                12,
                0,
            ),
        )
