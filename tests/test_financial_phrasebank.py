"""
Test Financial PhraseBank acquisition and preparation.

Purpose
-------
These tests prove that the acquisition module can:

- read the expected file from a ZIP archive;
- convert negative, neutral, and positive labels;
- create reproducible SHA-256 checksums;
- save normalized records as JSONL;
- create an acquisition manifest;
- reject broken ZIP files;
- reject archives missing the expected dataset file;
- reject unexpected record counts;
- require timezone-aware acquisition timestamps.

Test strategy
-------------
The tests create tiny temporary archives.

They do not download the real Financial PhraseBank dataset and do not
train DistilBERT, BERT, or LoRA models.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

import pytest

import financial_news_intelligence.data.financial_phrasebank as phrasebank
from financial_news_intelligence.data.financial_phrasebank import (
    ARCHIVE_MEMBER_NAME,
    acquire_financial_phrasebank,
    create_sha256,
    read_phrasebank_records,
    save_phrasebank_jsonl,
)


# ============================================================
# 1. TEST ARCHIVE HELPER
# ============================================================

def create_test_archive(
    archive_path: Path,
    lines: list[str],
    *,
    member_name: str = ARCHIVE_MEMBER_NAME,
) -> Path:
    """
    Create one small Financial PhraseBank-style ZIP archive.

    Input:
        Destination path, dataset lines, and archive member name.

    Output:
        Path to the generated ZIP file.
    """

    file_content = "\n".join(lines).encode(
        "iso-8859-1"
    )

    with ZipFile(
        archive_path,
        mode="w",
    ) as archive:
        archive.writestr(
            member_name,
            file_content,
        )

    return archive_path


# ============================================================
# 2. LABEL CONVERSION
# ============================================================

def test_records_are_read_and_labels_are_converted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    negative, neutral, and positive should become project labels.

    Prepare:
        A tiny ZIP containing three labelled sentences.

    Run:
        Read and normalize the archive.

    Check:
        Labels, numerical IDs, and checksums are correct.
    """

    archive_path = create_test_archive(
        tmp_path / "phrasebank.zip",
        [
            "The company reported a loss.@negative",
            "The results matched expectations.@neutral",
            "Revenue increased strongly.@positive",
        ],
    )

    # The real dataset expects 3,453 records.
    # This test temporarily expects only our three fake records.
    monkeypatch.setattr(
        phrasebank,
        "EXPECTED_RECORD_COUNT",
        3,
    )

    records = read_phrasebank_records(
        archive_path
    )

    assert len(records) == 3

    assert records[0]["label"] == "Bearish"
    assert records[0]["label_id"] == 0

    assert records[1]["label"] == "Neutral"
    assert records[1]["label_id"] == 1

    assert records[2]["label"] == "Bullish"
    assert records[2]["label_id"] == 2

    assert len(
        records[0]["text_checksum_sha256"]
    ) == 64


# ============================================================
# 3. JSONL WRITING
# ============================================================

def test_jsonl_file_is_created_with_matching_checksum(
    tmp_path: Path,
) -> None:
    """
    One normalized record should be stored on each JSONL line.

    Prepare:
        Two small normalized records.

    Run:
        Save them as JSONL.

    Check:
        Two lines exist and the checksum matches the file.
    """

    records = [
        {
            "record_id": "fpb_000001",
            "text": "The company reported a loss.",
            "source_label": "negative",
            "label": "Bearish",
            "label_id": 0,
            "text_checksum_sha256": "a" * 64,
        },
        {
            "record_id": "fpb_000002",
            "text": "Revenue increased strongly.",
            "source_label": "positive",
            "label": "Bullish",
            "label_id": 2,
            "text_checksum_sha256": "b" * 64,
        },
    ]

    output_path = tmp_path / "records.jsonl"

    checksum = save_phrasebank_jsonl(
        records,
        output_path,
    )

    lines = output_path.read_text(
        encoding="utf-8"
    ).splitlines()

    assert len(lines) == 2

    first_record = json.loads(lines[0])
    second_record = json.loads(lines[1])

    assert first_record["label"] == "Bearish"
    assert second_record["label"] == "Bullish"

    assert checksum == create_sha256(
        output_path.read_bytes()
    )


# ============================================================
# 4. MISSING ARCHIVE
# ============================================================

def test_missing_archive_is_rejected(
    tmp_path: Path,
) -> None:
    """A nonexistent source archive should raise a clear error."""

    missing_path = (
        tmp_path / "missing_phrasebank.zip"
    )

    with pytest.raises(
        FileNotFoundError,
        match="Dataset archive not found",
    ):
        read_phrasebank_records(
            missing_path
        )


# ============================================================
# 5. BROKEN ZIP FILE
# ============================================================

def test_invalid_zip_file_is_rejected(
    tmp_path: Path,
) -> None:
    """A text file pretending to be a ZIP must be rejected."""

    invalid_archive = (
        tmp_path / "invalid_phrasebank.zip"
    )

    invalid_archive.write_text(
        "This is not a ZIP archive.",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="not a valid ZIP",
    ):
        read_phrasebank_records(
            invalid_archive
        )


# ============================================================
# 6. MISSING DATASET MEMBER
# ============================================================

def test_archive_without_expected_member_is_rejected(
    tmp_path: Path,
) -> None:
    """The ZIP must contain Sentences_75Agree.txt."""

    archive_path = create_test_archive(
        tmp_path / "wrong_member.zip",
        [
            "Revenue increased strongly.@positive",
        ],
        member_name="unexpected_file.txt",
    )

    with pytest.raises(
        ValueError,
        match="does not contain",
    ):
        read_phrasebank_records(
            archive_path
        )


# ============================================================
# 7. UNEXPECTED RECORD COUNT
# ============================================================

def test_unexpected_record_count_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A changed record count should stop dataset acquisition."""

    archive_path = create_test_archive(
        tmp_path / "count_test.zip",
        [
            "Revenue increased strongly.@positive",
            "Results matched expectations.@neutral",
        ],
    )

    # Pretend that three records were expected.
    monkeypatch.setattr(
        phrasebank,
        "EXPECTED_RECORD_COUNT",
        3,
    )

    with pytest.raises(
        ValueError,
        match="Unexpected Financial PhraseBank record count",
    ):
        read_phrasebank_records(
            archive_path
        )


# ============================================================
# 8. COMPLETE ACQUISITION MANIFEST
# ============================================================

def test_complete_acquisition_creates_files_and_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The complete pipeline should create processed data and a manifest.

    Prepare:
        A fake downloaded archive and repository revision.

    Run:
        Execute the acquisition function.

    Check:
        Files, labels, licence, counts, and checksums are recorded.
    """

    archive_path = create_test_archive(
        tmp_path / "downloaded_phrasebank.zip",
        [
            "The company reported a loss.@negative",
            "The results matched expectations.@neutral",
            "Revenue increased strongly.@positive",
        ],
    )

    repository_revision = (
        "1234567890abcdef1234567890abcdef"
    )

    # Replace the live downloader with our local fake archive.
    monkeypatch.setattr(
        phrasebank,
        "download_phrasebank_archive",
        lambda raw_dir: (
            archive_path,
            repository_revision,
        ),
    )

    monkeypatch.setattr(
        phrasebank,
        "EXPECTED_RECORD_COUNT",
        3,
    )

    processed_dir = (
        tmp_path / "processed"
    )

    manifest_dir = (
        tmp_path / "manifests"
    )

    manifest = acquire_financial_phrasebank(
        raw_dir=tmp_path / "raw",
        processed_dir=processed_dir,
        manifest_dir=manifest_dir,
        acquired_at=datetime(
            2024,
            7,
            1,
            12,
            0,
            tzinfo=timezone.utc,
        ),
    )

    processed_file = (
        processed_dir
        / "financial_phrasebank_75agree.jsonl"
    )

    manifest_file = (
        manifest_dir
        / (
            "financial_phrasebank_"
            "acquisition_manifest.json"
        )
    )

    assert processed_file.exists()
    assert manifest_file.exists()

    assert manifest["record_count"] == 3

    assert manifest["label_counts"] == {
        "Bearish": 1,
        "Bullish": 1,
        "Neutral": 1,
    }

    assert manifest["license"] == (
        "CC BY-NC-SA 3.0"
    )

    assert (
        manifest["commercial_use_restricted"]
        is True
    )

    assert (
        manifest["repository_revision"]
        == repository_revision
    )

    assert len(
        manifest["raw_checksum_sha256"]
    ) == 64

    assert len(
        manifest["processed_checksum_sha256"]
    ) == 64


# ============================================================
# 9. TIMEZONE VALIDATION
# ============================================================

def test_acquisition_time_requires_timezone(
    tmp_path: Path,
) -> None:
    """A timezone-free acquisition timestamp is unsafe."""

    with pytest.raises(
        ValueError,
        match="acquired_at must include a timezone",
    ):
        acquire_financial_phrasebank(
            raw_dir=tmp_path / "raw",
            processed_dir=tmp_path / "processed",
            manifest_dir=tmp_path / "manifests",
            acquired_at=datetime(
                2024,
                7,
                1,
                12,
                0,
            ),
        )
