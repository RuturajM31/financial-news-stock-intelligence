"""
Test the Transformer dataset export process.

Purpose
-------
These tests confirm that the exporter creates valid training,
validation, testing, and manifest files.

The tests check:

- one article is written on each JSONL line;
- train, validation, and test files are created;
- manifest record counts are correct;
- SHA-256 checksums match the real files;
- exact duplicate articles are removed;
- unsafe dataset names are rejected;
- manifest timestamps must include a timezone.

Important
---------
These tests use small temporary files. They do not create the real
financial-news training dataset and do not train a Transformer model.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from financial_news_intelligence.data.transformer_export import (
    create_checksum,
    export_transformer_dataset,
    save_jsonl_file,
)
from financial_news_intelligence.schemas.common import (
    SentimentLabel,
)
from financial_news_intelligence.schemas.transformer_data import (
    TransformerExample,
)


# ============================================================
# 1. TEST ARTICLE HELPER
# ============================================================

def make_example(
    article_number: int,
    event_number: int,
    month: int,
    text: str | None = None,
) -> TransformerExample:
    """
    Create one small article card for testing.

    Input:
        Article number, event number, publication month,
        and optional article text.

    Output:
        One valid TransformerExample.
    """

    if text is None:
        text = (
            f"Company {event_number} reported financial results "
            f"for article {article_number}."
        )

    return TransformerExample(
        article_id=f"article_{article_number:03d}",
        event_id=f"event_{event_number:03d}",
        text=text,
        label=SentimentLabel.BULLISH,
        published_at=datetime(
            2024,
            month,
            1,
            9,
            0,
            tzinfo=timezone.utc,
        ),
        ticker="TEST",
        source_id="verified_test_source",
    )


def make_six_events() -> list[TransformerExample]:
    """
    Create six chronological events.

    Six events produce:

    - four training events;
    - one validation event;
    - one testing event.
    """

    return [
        make_example(1, 1, 1),
        make_example(2, 2, 2),
        make_example(3, 3, 3),
        make_example(4, 4, 4),
        make_example(5, 5, 5),
        make_example(6, 6, 6),
    ]


# ============================================================
# 2. SAVE ONE JSONL FILE
# ============================================================

def test_save_jsonl_file_writes_one_article_per_line(
    tmp_path: Path,
) -> None:
    """Each article should appear on exactly one JSONL line."""

    # Prepare two article cards.
    examples = [
        make_example(1, 1, 1),
        make_example(2, 2, 2),
    ]

    output_file = tmp_path / "articles.jsonl"

    # Run the JSONL writer.
    checksum = save_jsonl_file(
        examples,
        output_file,
    )

    # Read the generated lines.
    lines = output_file.read_text(
        encoding="utf-8"
    ).splitlines()

    # Check that two articles created two lines.
    assert len(lines) == 2

    first_article = json.loads(lines[0])
    second_article = json.loads(lines[1])

    assert first_article["article_id"] == "article_001"
    assert second_article["article_id"] == "article_002"

    # Check that the returned checksum matches the real file.
    assert checksum == create_checksum(
        output_file.read_bytes()
    )


# ============================================================
# 3. CREATE ALL DATASET FILES
# ============================================================

def test_export_creates_train_validation_test_and_manifest(
    tmp_path: Path,
) -> None:
    """The exporter should create all four required files."""

    # Prepare six chronological events.
    examples = make_six_events()

    dataset_dir = tmp_path / "dataset"
    manifest_dir = tmp_path / "manifests"

    # Run the complete exporter.
    manifest = export_transformer_dataset(
        examples,
        dataset_name="financial_sentiment",
        output_dir=dataset_dir,
        manifest_dir=manifest_dir,
        created_at=datetime(
            2024,
            7,
            1,
            12,
            0,
            tzinfo=timezone.utc,
        ),
    )

    # Check the three JSONL files.
    assert (
        dataset_dir
        / "financial_sentiment_train.jsonl"
    ).exists()

    assert (
        dataset_dir
        / "financial_sentiment_validation.jsonl"
    ).exists()

    assert (
        dataset_dir
        / "financial_sentiment_test.jsonl"
    ).exists()

    # Check the manifest file.
    manifest_file = (
        manifest_dir
        / "financial_sentiment_manifest.json"
    )

    assert manifest_file.exists()

    # Check the returned manifest name.
    assert manifest["dataset_name"] == "financial_sentiment"


# ============================================================
# 4. MANIFEST COUNTS
# ============================================================

def test_manifest_contains_correct_record_counts(
    tmp_path: Path,
) -> None:
    """The manifest should report the real split sizes."""

    # Prepare six events.
    examples = make_six_events()

    # Run the exporter.
    manifest = export_transformer_dataset(
        examples,
        dataset_name="count_test",
        output_dir=tmp_path / "dataset",
        manifest_dir=tmp_path / "manifests",
        created_at=datetime(
            2024,
            7,
            1,
            tzinfo=timezone.utc,
        ),
    )

    files = manifest["files"]

    # Six events with the default split become 4, 1, and 1.
    assert files["train"]["record_count"] == 4
    assert files["validation"]["record_count"] == 1
    assert files["test"]["record_count"] == 1

    assert manifest["source_record_count"] == 6
    assert manifest["final_record_count"] == 6
    assert manifest["duplicates_removed"] == 0


# ============================================================
# 5. CHECKSUMS MATCH THE REAL FILES
# ============================================================

def test_manifest_checksums_match_generated_files(
    tmp_path: Path,
) -> None:
    """Every manifest checksum should match its saved JSONL file."""

    examples = make_six_events()

    dataset_dir = tmp_path / "dataset"

    manifest = export_transformer_dataset(
        examples,
        dataset_name="checksum_test",
        output_dir=dataset_dir,
        manifest_dir=tmp_path / "manifests",
        created_at=datetime(
            2024,
            7,
            1,
            tzinfo=timezone.utc,
        ),
    )

    for split_name in (
        "train",
        "validation",
        "test",
    ):
        file_details = manifest["files"][split_name]

        file_path = (
            dataset_dir
            / file_details["file_name"]
        )

        actual_checksum = create_checksum(
            file_path.read_bytes()
        )

        assert (
            file_details["checksum_sha256"]
            == actual_checksum
        )

    # The complete dataset checksum must also be 64 characters.
    assert len(
        manifest["dataset_checksum_sha256"]
    ) == 64


# ============================================================
# 6. DUPLICATE ARTICLE REMOVAL
# ============================================================

def test_export_removes_exact_duplicate_articles(
    tmp_path: Path,
) -> None:
    """Identical article text should be exported only once."""

    examples = make_six_events()

    # Add a second article to event 1 with identical text.
    duplicate_article = make_example(
        article_number=7,
        event_number=1,
        month=1,
        text=examples[0].text,
    )

    examples.append(duplicate_article)

    manifest = export_transformer_dataset(
        examples,
        dataset_name="duplicate_test",
        output_dir=tmp_path / "dataset",
        manifest_dir=tmp_path / "manifests",
        created_at=datetime(
            2024,
            7,
            1,
            tzinfo=timezone.utc,
        ),
    )

    assert manifest["source_record_count"] == 7
    assert manifest["final_record_count"] == 6
    assert manifest["duplicates_removed"] == 1


# ============================================================
# 7. UNSAFE DATASET NAME
# ============================================================

def test_unsafe_dataset_name_is_rejected(
    tmp_path: Path,
) -> None:
    """Unsafe characters must not become part of a filename."""

    examples = make_six_events()

    with pytest.raises(
        ValueError,
        match="dataset_name",
    ):
        export_transformer_dataset(
            examples,
            dataset_name="../unsafe dataset",
            output_dir=tmp_path / "dataset",
            manifest_dir=tmp_path / "manifests",
        )


# ============================================================
# 8. TIMEZONE-FREE MANIFEST TIME
# ============================================================

def test_created_at_without_timezone_is_rejected(
    tmp_path: Path,
) -> None:
    """The manifest creation time must identify its timezone."""

    examples = make_six_events()

    with pytest.raises(
        ValueError,
        match="created_at must include a timezone",
    ):
        export_transformer_dataset(
            examples,
            dataset_name="timezone_test",
            output_dir=tmp_path / "dataset",
            manifest_dir=tmp_path / "manifests",
            created_at=datetime(
                2024,
                7,
                1,
                12,
                0,
            ),
        )
