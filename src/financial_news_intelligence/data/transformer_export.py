"""
Export prepared financial-news articles for Transformer training.

Purpose
-------
This module converts validated TransformerExample records into three
newline-delimited JSON files:

- training data;
- validation data;
- testing data.

It also creates a manifest containing record counts, filenames, and
SHA-256 checksums.

Input
-----
A collection of verified and labelled financial-news article cards.

Processing
----------
1. Use the leakage-safe chronological event-level split.
2. Save one JSON article on each line.
3. Calculate a checksum for every generated file.
4. Create a manifest describing the complete dataset.

Output
------
- <dataset_name>_train.jsonl
- <dataset_name>_validation.jsonl
- <dataset_name>_test.jsonl
- <dataset_name>_manifest.json

Next use
--------
DistilBERT and BERT will load these files during tokenization and
sentiment-model training.

Important
---------
This module prepares model data. It does not train a Transformer model.
"""

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from financial_news_intelligence.data.transformer_dataset import (
    split_transformer_examples,
)
from financial_news_intelligence.paths import (
    MANIFESTS_DIR,
    PROCESSED_DATA_DIR,
)
from financial_news_intelligence.schemas.training_data import (
    DatasetSplit,
)
from financial_news_intelligence.schemas.transformer_data import (
    TransformerExample,
    TransformerSplitConfig,
)


# ============================================================
# 1. DEFAULT OUTPUT FOLDERS
# ============================================================

# DistilBERT and BERT will read the generated JSONL files here.
DEFAULT_DATASET_DIR = PROCESSED_DATA_DIR / "transformer"

# Dataset manifests will be stored with other project manifests.
DEFAULT_MANIFEST_DIR = MANIFESTS_DIR


# ============================================================
# 2. CREATE A DIGITAL FILE FINGERPRINT
# ============================================================

def create_checksum(content: bytes) -> str:
    """
    Create a SHA-256 fingerprint for file content.

    Input:
        File content stored as bytes.

    Output:
        A 64-character checksum.

    Why:
        If the file changes, its checksum changes too.
    """

    return hashlib.sha256(content).hexdigest()


# ============================================================
# 3. SAVE ONE JSONL FILE
# ============================================================

def save_jsonl_file(
    examples: Sequence[TransformerExample],
    file_path: Path,
) -> str:
    """
    Save article cards as one JSON record per line.

    Input:
        Prepared article cards and the destination file path.

    Output:
        The SHA-256 checksum of the saved file.

    Next use:
        Hugging Face Datasets will load this JSONL file.
    """

    json_lines: list[str] = []

    for example in examples:
        # Convert dates, labels, and split values into JSON-safe values.
        article_data = example.model_dump(mode="json")

        # Create one compact and reproducible JSON line.
        json_line = json.dumps(
            article_data,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )

        json_lines.append(json_line)

    # JSONL stores one article on each line.
    file_text = "\n".join(json_lines) + "\n"
    file_content = file_text.encode("utf-8")

    file_path.write_bytes(file_content)

    return create_checksum(file_content)


# ============================================================
# 4. EXPORT THE COMPLETE TRANSFORMER DATASET
# ============================================================

def export_transformer_dataset(
    examples: Sequence[TransformerExample],
    *,
    dataset_name: str,
    config: TransformerSplitConfig | None = None,
    output_dir: Path = DEFAULT_DATASET_DIR,
    manifest_dir: Path = DEFAULT_MANIFEST_DIR,
    created_at: datetime | None = None,
) -> dict[str, object]:
    """
    Create training, validation, testing, and manifest files.

    Input:
        Verified and labelled article cards.

    Processing:
        1. Remove exact duplicate articles.
        2. Keep articles from the same event together.
        3. Split events chronologically.
        4. Save each split as a JSONL file.
        5. Calculate checksums.
        6. Save a dataset manifest.

    Output:
        A dictionary containing the dataset manifest.

    Next use:
        DistilBERT and BERT training.
    """

    clean_name = dataset_name.strip()

    # The dataset name becomes part of each generated filename.
    if not re.fullmatch(r"[A-Za-z0-9_-]+", clean_name):
        raise ValueError(
            "dataset_name may contain only letters, "
            "numbers, underscores, and hyphens."
        )

    # Use the current UTC time unless tests provide a fixed time.
    if created_at is None:
        created_at = datetime.now(timezone.utc)

    if created_at.tzinfo is None or created_at.utcoffset() is None:
        raise ValueError(
            "created_at must include a timezone."
        )

    # Use the leakage-safe splitting logic created earlier.
    splits = split_transformer_examples(
        examples,
        config,
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
    final_record_count = 0

    # Save the train, validation, and test files.
    for split in DatasetSplit:
        split_examples = splits[split]

        file_name = (
            f"{clean_name}_{split.value}.jsonl"
        )

        file_path = output_dir / file_name

        checksum = save_jsonl_file(
            split_examples,
            file_path,
        )

        record_count = len(split_examples)
        final_record_count += record_count

        file_details[split.value] = {
            "file_name": file_name,
            "record_count": record_count,
            "checksum_sha256": checksum,
        }

    # Create one fingerprint representing the complete dataset.
    dataset_summary = json.dumps(
        file_details,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    dataset_checksum = create_checksum(
        dataset_summary
    )

    manifest: dict[str, object] = {
        "dataset_name": clean_name,
        "created_at": created_at.isoformat(),
        "source_record_count": len(examples),
        "final_record_count": final_record_count,
        "duplicates_removed": (
            len(examples) - final_record_count
        ),
        "split_strategy": (
            "chronological_event_level"
        ),
        "files": file_details,
        "dataset_checksum_sha256": dataset_checksum,
    }

    manifest_path = (
        manifest_dir
        / f"{clean_name}_manifest.json"
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
