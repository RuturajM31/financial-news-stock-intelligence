"""
Acquire and prepare the Financial PhraseBank sentiment dataset.

Purpose
-------
This module downloads the original Financial PhraseBank archive from
a pinned Hugging Face repository revision.

It reads the sentences with at least 75% annotator agreement and
converts their original sentiment labels into the project labels:

- negative becomes Bearish;
- neutral becomes Neutral;
- positive becomes Bullish.

Input
-----
The original FinancialPhraseBank-v1.0.zip archive stored in the
takala/financial_phrasebank dataset repository.

Processing
----------
1. Find the exact repository revision.
2. Download the original archive.
3. Copy the archive into the project's raw-data folder.
4. Read Sentences_75Agree.txt.
5. Convert labels into project labels and numerical IDs.
6. Save the normalized records as JSONL.
7. Create a manifest with counts, checksums, licence, and provenance.

Output
------
- A preserved raw ZIP archive.
- financial_phrasebank_75agree.jsonl.
- financial_phrasebank_acquisition_manifest.json.

Next use
--------
A later module will divide the normalized records into reproducible,
stratified train, validation, and test datasets.

Important
---------
This dataset contains sentences and sentiment labels only.

It does not provide:
- article publication timestamps;
- company event identifiers;
- official train, validation, or test splits;
- verified stock-return outcomes.

It is used only for sentiment-model training and evaluation.
"""

import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from huggingface_hub import HfApi, hf_hub_download

from financial_news_intelligence.paths import (
    MANIFESTS_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)
from financial_news_intelligence.schemas.common import (
    SentimentLabel,
)


# ============================================================
# 1. DATASET SETTINGS
# ============================================================

# Public Hugging Face repository containing the original archive.
DATASET_REPOSITORY = "takala/financial_phrasebank"

# Location of the original ZIP file inside that repository.
ARCHIVE_FILE_NAME = "data/FinancialPhraseBank-v1.0.zip"

# File used for the selected 75% annotator-agreement collection.
ARCHIVE_MEMBER_NAME = (
    "FinancialPhraseBank-v1.0/"
    "Sentences_75Agree.txt"
)

# Expected number of records in the selected collection.
EXPECTED_RECORD_COUNT = 3453

# Licence recorded in the generated manifest.
DATASET_LICENSE = "CC BY-NC-SA 3.0"

# Project folder that preserves the original downloaded archive.
DEFAULT_RAW_DIR = (
    RAW_DATA_DIR / "financial_phrasebank"
)

# Project folder that stores the cleaned JSONL records.
DEFAULT_PROCESSED_DIR = (
    PROCESSED_DATA_DIR / "financial_phrasebank"
)


# ============================================================
# 2. LABEL CONVERSION
# ============================================================

# Convert original labels into the project's sentiment vocabulary.
SOURCE_TO_PROJECT_LABEL = {
    "negative": SentimentLabel.BEARISH,
    "neutral": SentimentLabel.NEUTRAL,
    "positive": SentimentLabel.BULLISH,
}

# Numerical IDs required later during Transformer training.
PROJECT_LABEL_TO_ID = {
    SentimentLabel.BEARISH: 0,
    SentimentLabel.NEUTRAL: 1,
    SentimentLabel.BULLISH: 2,
}


# ============================================================
# 3. CREATE A DIGITAL FINGERPRINT
# ============================================================

def create_sha256(content: bytes) -> str:
    """
    Create a SHA-256 checksum.

    Input:
        File or text content stored as bytes.

    Output:
        A 64-character digital fingerprint.

    Why:
        Any change to the content produces a different checksum.
    """

    return hashlib.sha256(content).hexdigest()


# ============================================================
# 4. DOWNLOAD THE ORIGINAL DATASET ARCHIVE
# ============================================================

def download_phrasebank_archive(
    raw_dir: Path = DEFAULT_RAW_DIR,
) -> tuple[Path, str]:
    """
    Download and preserve the original dataset ZIP.

    Input:
        Destination folder for raw project data.

    Output:
        Local archive path and exact repository revision.

    Reproducibility:
        The repository revision is discovered first and then used to
        pin the file download.
    """

    raw_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Ask Hugging Face for the exact current repository commit.
    repository_info = HfApi().dataset_info(
        DATASET_REPOSITORY
    )

    repository_revision = repository_info.sha

    if not repository_revision:
        raise RuntimeError(
            "Could not determine the dataset repository revision."
        )

    # Download the archive from the exact repository revision.
    cached_archive_path = Path(
        hf_hub_download(
            repo_id=DATASET_REPOSITORY,
            filename=ARCHIVE_FILE_NAME,
            repo_type="dataset",
            revision=repository_revision,
        )
    )

    # Copy the cached file into the project's own raw-data folder.
    project_archive_path = (
        raw_dir
        / (
            "FinancialPhraseBank-v1.0_"
            f"{repository_revision[:12]}.zip"
        )
    )

    shutil.copy2(
        cached_archive_path,
        project_archive_path,
    )

    return project_archive_path, repository_revision


# ============================================================
# 5. READ AND CONVERT THE RECORDS
# ============================================================

def read_phrasebank_records(
    archive_path: Path,
) -> list[dict[str, object]]:
    """
    Read and normalize the selected Financial PhraseBank records.

    Input:
        Downloaded Financial PhraseBank ZIP archive.

    Output:
        Clean records containing text, labels, IDs, and checksums.
    """

    if not archive_path.exists():
        raise FileNotFoundError(
            f"Dataset archive not found: {archive_path}"
        )

    try:
        with ZipFile(archive_path) as archive:
            # Stop clearly if the expected collection is missing.
            if ARCHIVE_MEMBER_NAME not in archive.namelist():
                raise ValueError(
                    "The archive does not contain "
                    "Sentences_75Agree.txt."
                )

            raw_content = archive.read(
                ARCHIVE_MEMBER_NAME
            )

    except BadZipFile as error:
        raise ValueError(
            "The downloaded dataset archive is not a valid ZIP file."
        ) from error

    # The original dataset file uses ISO-8859-1 encoding.
    file_text = raw_content.decode("iso-8859-1")

    normalized_records: list[dict[str, object]] = []

    for line_number, raw_line in enumerate(
        file_text.splitlines(),
        start=1,
    ):
        clean_line = raw_line.strip()

        # Ignore blank lines.
        if not clean_line:
            continue

        # Each original line ends with:
        # @negative, @neutral, or @positive.
        try:
            sentence, source_label = clean_line.rsplit(
                "@",
                1,
            )

        except ValueError as error:
            raise ValueError(
                "Invalid Financial PhraseBank record "
                f"on line {line_number}."
            ) from error

        # Remove repeated spaces and line breaks.
        clean_sentence = " ".join(
            sentence.split()
        )

        clean_source_label = (
            source_label.strip().lower()
        )

        if clean_source_label not in SOURCE_TO_PROJECT_LABEL:
            raise ValueError(
                "Unsupported sentiment label "
                f"on line {line_number}: "
                f"{clean_source_label}"
            )

        project_label = SOURCE_TO_PROJECT_LABEL[
            clean_source_label
        ]

        # Create a fingerprint for this exact sentence.
        text_checksum = create_sha256(
            clean_sentence.encode("utf-8")
        )

        normalized_records.append(
            {
                "record_id": (
                    f"fpb_{line_number:06d}"
                ),
                "text": clean_sentence,
                "source_label": clean_source_label,
                "label": project_label.value,
                "label_id": PROJECT_LABEL_TO_ID[
                    project_label
                ],
                "text_checksum_sha256": text_checksum,
            }
        )

    # Detect accidental use of the wrong file or an upstream change.
    if len(normalized_records) != EXPECTED_RECORD_COUNT:
        raise ValueError(
            "Unexpected Financial PhraseBank record count. "
            f"Expected {EXPECTED_RECORD_COUNT}, "
            f"received {len(normalized_records)}."
        )

    return normalized_records


# ============================================================
# 6. SAVE NORMALIZED JSONL RECORDS
# ============================================================

def save_phrasebank_jsonl(
    records: list[dict[str, object]],
    output_path: Path,
) -> str:
    """
    Save one normalized sentiment record on each JSONL line.

    Input:
        Converted Financial PhraseBank records.

    Output:
        SHA-256 checksum of the generated JSONL file.
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
# 7. COMPLETE DATASET ACQUISITION
# ============================================================

def acquire_financial_phrasebank(
    raw_dir: Path = DEFAULT_RAW_DIR,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    manifest_dir: Path = MANIFESTS_DIR,
    acquired_at: datetime | None = None,
) -> dict[str, object]:
    """
    Download, normalize, verify, and document the dataset.

    Output:
        Acquisition manifest containing source, revision, licence,
        record counts, label counts, and checksums.

    Next:
        The normalized JSONL file will be split reproducibly before
        DistilBERT, BERT, and LoRA training.
    """

    if acquired_at is None:
        acquired_at = datetime.now(
            timezone.utc
        )

    if (
        acquired_at.tzinfo is None
        or acquired_at.utcoffset() is None
    ):
        raise ValueError(
            "acquired_at must include a timezone."
        )

    # Download the original source archive.
    archive_path, repository_revision = (
        download_phrasebank_archive(raw_dir)
    )

    # Read and convert all selected sentiment records.
    normalized_records = read_phrasebank_records(
        archive_path
    )

    processed_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        processed_dir
        / "financial_phrasebank_75agree.jsonl"
    )

    # Save the cleaned records and calculate their checksum.
    processed_checksum = save_phrasebank_jsonl(
        normalized_records,
        output_path,
    )

    # Calculate a checksum for the original archive.
    raw_checksum = create_sha256(
        archive_path.read_bytes()
    )

    # Count Bullish, Neutral, and Bearish records.
    label_counts = Counter(
        str(record["label"])
        for record in normalized_records
    )

    manifest: dict[str, object] = {
        "dataset_name": "Financial PhraseBank",
        "configuration": "sentences_75agree",
        "repository_id": DATASET_REPOSITORY,
        "repository_revision": repository_revision,
        "source_file": ARCHIVE_FILE_NAME,
        "raw_archive_path": str(archive_path),
        "processed_file_path": str(output_path),
        "acquired_at": acquired_at.isoformat(),
        "license": DATASET_LICENSE,
        "commercial_use_restricted": True,
        "record_count": len(normalized_records),
        "label_counts": dict(
            sorted(label_counts.items())
        ),
        "raw_checksum_sha256": raw_checksum,
        "processed_checksum_sha256": (
            processed_checksum
        ),
        "label_mapping": {
            "negative": "Bearish",
            "neutral": "Neutral",
            "positive": "Bullish",
        },
        "limitations": [
            "Sentence-level sentiment benchmark.",
            "No publication timestamps.",
            "No event identifiers.",
            "No official dataset splits.",
            "Not a stock-return prediction dataset.",
        ],
    }

    manifest_path = (
        manifest_dir
        / (
            "financial_phrasebank_"
            "acquisition_manifest.json"
        )
    )

    # Save a readable manifest for humans and later model runs.
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
