"""
Focused tests for the controlled BERT smoke runner.

Purpose
-------
Protect the small but important layer that prepares balanced real-data files,
creates an isolated one-epoch BERT configuration, protects existing smoke
evidence, and validates the saved model and evaluation artifacts.

Inputs and data journey
-----------------------
The tests use tiny in-memory records and temporary files. They exercise the
same helper functions used by ``scripts/run_bert_smoke.py``:

``configuration -> split paths -> balanced records -> smoke configuration
-> artifact validation``

Outputs and downstream use
--------------------------
Passing tests provide evidence that the runner will preserve source records,
keep all three classes balanced, isolate smoke outputs, and reject incomplete
artifacts before the full BERT benchmark is allowed to start.

Safety and limitations
----------------------
No test downloads BERT, reads the real Financial PhraseBank files, starts a
Trainer, or modifies production artifacts. The real smoke run remains the
required end-to-end proof of model download, tokenization, training, and save.
"""

from __future__ import annotations

import importlib.util
import json
from collections import Counter
from dataclasses import fields
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from financial_news_intelligence.models.bert_training import (
    BERT_MODEL_ID,
    BertTrainingConfig,
)


# ============================================================
# 1. LOAD THE SCRIPT AS A TESTABLE MODULE
# ============================================================

RUNNER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_bert_smoke.py"


def load_runner() -> ModuleType:
    """Load the command-line script without executing its ``main`` function."""

    specification = importlib.util.spec_from_file_location(
        "bert_smoke_runner_under_test",
        RUNNER_PATH,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Could not load BERT smoke runner: {RUNNER_PATH}")

    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


runner = load_runner()


# ============================================================
# 2. REUSABLE TEST DATA
# ============================================================


def make_records(records_per_class: int = 12) -> list[dict[str, Any]]:
    """Create three balanced classes while preserving extra record fields."""

    records: list[dict[str, Any]] = []

    for label_id, label_name in enumerate(("Bearish", "Neutral", "Bullish")):
        for record_number in range(records_per_class):
            records.append(
                {
                    "sentence_id": f"{label_id}-{record_number}",
                    "text": f"Example {label_name} sentence {record_number}",
                    "label_id": label_id,
                    "label_name": label_name,
                    "source_checksum": f"checksum-{label_id}-{record_number}",
                }
            )

    return records


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write one readable JSON object for artifact-validation tests."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


# ============================================================
# 3. SPLIT AND LABEL DISCOVERY
# ============================================================


def test_split_fields_are_resolved_from_real_configuration() -> None:
    """Find the configured train, validation, and test paths unambiguously."""

    # Prepare: create the same configuration used by production smoke runs.
    config = BertTrainingConfig()

    # Run: discover the private split-path field names.
    resolved = runner.resolve_split_field_names(config)

    # Check: every split maps to a distinct dataclass field containing a path.
    config_field_names = {field.name for field in fields(config)}
    assert set(resolved) == {"train", "validation", "test"}
    assert len(set(resolved.values())) == 3
    assert set(resolved.values()).issubset(config_field_names)


def test_label_detection_prefers_numeric_label_id() -> None:
    """Choose the stable model label when text and numeric labels both exist."""

    # Prepare: records contain two valid three-class label fields.
    records = make_records()

    # Run: detect the field used for balancing.
    label_field = runner.detect_label_field(records)

    # Check: the preferred label_id field wins deterministically.
    assert label_field == "label_id"


# ============================================================
# 4. BALANCED SAMPLE BEHAVIOUR
# ============================================================


def test_balanced_sample_has_exact_counts_and_preserves_records() -> None:
    """Select equal class counts without deleting provenance fields."""

    # Prepare: build twelve records for each of three sentiment classes.
    records = make_records(records_per_class=12)

    # Run: select the nine-per-class training smoke sample.
    selected = runner.balanced_sample(
        records=records,
        label_field="label_id",
        records_per_class=9,
        random_seed=42,
    )

    # Check: the sample is 27 rows, exactly balanced, and structurally intact.
    assert len(selected) == 27
    assert Counter(record["label_id"] for record in selected) == {
        0: 9,
        1: 9,
        2: 9,
    }
    assert all("source_checksum" in record for record in selected)


def test_balanced_sample_is_deterministic() -> None:
    """Produce identical record order whenever the seed and input are equal."""

    records = make_records(records_per_class=12)

    first = runner.balanced_sample(records, "label_id", 3, 99)
    second = runner.balanced_sample(records, "label_id", 3, 99)

    assert [record["sentence_id"] for record in first] == [
        record["sentence_id"] for record in second
    ]


def test_balanced_sample_rejects_an_undersized_class() -> None:
    """Fail before writing files when any class cannot meet the requested size."""

    # Prepare: one class has only two records, but three are required.
    records = make_records(records_per_class=3)
    records = [
        record
        for record in records
        if not (record["label_id"] == 2 and record["sentence_id"] == "2-2")
    ]

    # Run and check: the exact class shortage must be reported.
    with pytest.raises(runner.BertSmokeError, match="3 are required"):
        runner.balanced_sample(records, "label_id", 3, 42)


# ============================================================
# 5. SMOKE CONFIGURATION ISOLATION
# ============================================================


def test_smoke_config_changes_only_approved_fields(tmp_path: Path) -> None:
    """Keep the real BERT protocol while changing data, epoch, and outputs."""

    # Prepare: discover the split fields and supply temporary balanced files.
    base_config = BertTrainingConfig()
    split_fields = runner.resolve_split_field_names(base_config)
    smoke_paths = {
        split_name: tmp_path / f"{split_name}.jsonl"
        for split_name in split_fields
    }
    for path in smoke_paths.values():
        path.write_text("{}\n", encoding="utf-8")

    # Run: create the controlled one-epoch smoke configuration.
    smoke_config = runner.build_smoke_config(
        base_config,
        split_fields,
        smoke_paths,
    )

    # Check: BERT identity is preserved and smoke-only values are isolated.
    assert smoke_config.model_id == BERT_MODEL_ID
    assert smoke_config.model_family == "BERT"
    assert smoke_config.number_of_epochs == 1.0
    assert smoke_config.experiment_name == "BERT Financial Sentiment Smoke"
    assert smoke_config.benchmark_role == "smoke_pipeline_validation"
    assert smoke_config.final_model_dir == runner.SMOKE_FINAL_MODEL_DIR

    for split_name, field_name in split_fields.items():
        assert getattr(smoke_config, field_name) == smoke_paths[split_name]


# ============================================================
# 6. OUTPUT PROTECTION
# ============================================================


def test_existing_smoke_output_is_protected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Refuse to overwrite previous smoke evidence without explicit approval."""

    protected_file = tmp_path / "existing_metrics.json"
    protected_file.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        runner,
        "smoke_output_paths",
        lambda: (protected_file,),
    )

    with pytest.raises(runner.BertSmokeError, match="already exist"):
        runner.protect_or_replace_outputs(replace_existing=False)

    assert protected_file.exists()


# ============================================================
# 7. SAVED-ARTIFACT CONTRACT
# ============================================================


def test_artifact_validation_accepts_complete_smoke_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Accept only a completed BERT run with model files and nine test rows."""

    # Prepare: redirect every artifact constant to temporary test locations.
    model_dir = tmp_path / "final_model"
    metrics_file = tmp_path / "metrics.json"
    manifest_file = tmp_path / "manifest.json"
    model_dir.mkdir(parents=True)

    monkeypatch.setattr(runner, "SMOKE_FINAL_MODEL_DIR", model_dir)
    monkeypatch.setattr(runner, "SMOKE_METRICS_FILE", metrics_file)
    monkeypatch.setattr(runner, "SMOKE_MANIFEST_FILE", manifest_file)

    for file_name in runner.REQUIRED_FINAL_MODEL_FILES:
        (model_dir / file_name).write_text("test", encoding="utf-8")

    write_json(
        manifest_file,
        {
            "status": "trained_and_evaluated",
            "model_id": BERT_MODEL_ID,
            "model_revision": "a" * 40,
            "experiment_name": "BERT Financial Sentiment Smoke",
            "model_family": "BERT",
            "benchmark_role": "smoke_pipeline_validation",
            "final_model_directory": str(model_dir),
            "parameter_counts": {"total_parameters": 109_000_000},
            "timing": {"training_seconds": 12.5},
        },
    )
    write_json(
        metrics_file,
        {
            "test_metrics": {
                "test_accuracy": 0.55,
                "test_macro_f1": 0.50,
            },
            "test_evaluation": {
                "confusion_matrix": [
                    [2, 1, 0],
                    [0, 2, 1],
                    [1, 0, 2],
                ]
            },
        },
    )

    # Run: review the same evidence required after real training.
    summary = runner.validate_smoke_artifacts()

    # Check: key metrics and the evaluated-row contract are returned.
    assert summary["status"] == "trained_and_evaluated"
    assert summary["model_id"] == BERT_MODEL_ID
    assert summary["total_parameters"] == 109_000_000
    assert sum(sum(row) for row in summary["confusion_matrix"]) == 9
