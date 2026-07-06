"""
Test LoRA data preparation, metrics, configuration, and artifact contracts.

Purpose
-------
Protect the logic that makes the LoRA experiment comparable with DistilBERT
and full BERT before any large model is downloaded.

Inputs and data journey
-----------------------
Tiny JSONL fixtures move through schema detection, canonical labels, leakage
checks, class weighting, deterministic metrics, output protection, and saved
adapter review.

Outputs and downstream use
--------------------------
Passing tests provide evidence that LoRA uses all three sentiment classes,
keeps test-row grain intact, trains only a subset of parameters, and produces
the exact metrics needed by final champion selection.

Safety and limitations
----------------------
No test imports Transformers, PEFT, or Datasets, and no test starts training.
The real three-epoch LoRA run remains the end-to-end proof.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

import financial_news_intelligence.models.lora_training as lora_module
from financial_news_intelligence.models.lora_training import (
    LABEL_ORDER,
    LoraTrainingConfig,
    LoraTrainingError,
    calculate_class_weights,
    canonical_label_id,
    classification_summary,
    detect_label_field,
    detect_text_field,
    protect_or_replace_outputs,
    validate_lora_config,
    validate_no_cross_split_leakage,
    validate_shared_protocol,
)


RUNNER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_lora_training.py"


def load_runner() -> ModuleType:
    """Load the LoRA command-line runner without executing its main function."""

    specification = importlib.util.spec_from_file_location(
        "lora_runner_under_test",
        RUNNER_PATH,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Could not load LoRA runner: {RUNNER_PATH}")

    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


runner = load_runner()


@pytest.fixture(autouse=True)
def use_small_split_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use three-row fixtures while production keeps 2,413/517/518."""

    small_counts = {"train": 3, "validation": 3, "test": 3}
    monkeypatch.setattr(lora_module, "EXPECTED_SPLIT_RECORDS", small_counts)
    monkeypatch.setattr(runner, "EXPECTED_SPLIT_RECORDS", small_counts)


def sample_records() -> list[dict[str, Any]]:
    """Return one small split with the approved three labels."""

    return [
        {"sentence": "Profit rose strongly.", "label_id": 2},
        {"sentence": "The outlook was unchanged.", "label_id": 1},
        {"sentence": "Costs increased sharply.", "label_id": 0},
    ]


def file_sha256(file_path: Path) -> str:
    """Return one fixture checksum for source-evidence validation."""

    return hashlib.sha256(file_path.read_bytes()).hexdigest()


def write_json(file_path: Path, payload: dict[str, Any]) -> None:
    """Write one deterministic JSON fixture."""

    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def build_lora_artifacts(tmp_path: Path) -> LoraTrainingConfig:
    """Create a complete three-record LoRA evidence fixture."""

    split_files: dict[str, Path] = {}
    for split_name, prefix in (
        ("train", "train"),
        ("validation", "valid"),
        ("test", "test"),
    ):
        split_file = tmp_path / f"{split_name}.jsonl"
        split_file.write_text(
            "\n".join(
                [
                    f'{{"sentence": "{prefix} a", "label_id": 0}}',
                    f'{{"sentence": "{prefix} b", "label_id": 1}}',
                    f'{{"sentence": "{prefix} c", "label_id": 2}}',
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        split_files[split_name] = split_file

    final_adapter = tmp_path / "models" / "final_adapter"
    final_adapter.mkdir(parents=True)
    for file_name in runner.REQUIRED_ADAPTER_FILES:
        (final_adapter / file_name).write_text("fixture", encoding="utf-8")

    config = LoraTrainingConfig(
        checkpoint_dir=tmp_path / "models" / "checkpoints",
        final_adapter_dir=final_adapter,
        metrics_file=tmp_path / "reports" / "lora_metrics.json",
        manifest_file=tmp_path / "manifests" / "lora_manifest.json",
    )

    write_json(
        config.metrics_file,
        {
            "test_metrics": {
                "test_accuracy": 2 / 3,
                "test_macro_f1": 0.61,
                "test_weighted_f1": 0.62,
                "test_runtime": 0.3,
            },
            "test_evaluation": {
                "confusion_matrix": [
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 1, 0],
                ],
                "label_order": list(LABEL_ORDER),
                "per_class": {
                    "Bearish": {
                        "precision": 1.0,
                        "recall": 1.0,
                        "f1": 1.0,
                        "support": 1,
                    },
                    "Neutral": {
                        "precision": 0.5,
                        "recall": 1.0,
                        "f1": 2 / 3,
                        "support": 1,
                    },
                    "Bullish": {
                        "precision": 0.0,
                        "recall": 0.0,
                        "f1": 0.0,
                        "support": 1,
                    },
                },
            },
        },
    )

    write_json(
        config.manifest_file,
        {
            "status": "trained_and_evaluated",
            "experiment_name": config.experiment_name,
            "model_family": config.model_family,
            "benchmark_role": config.benchmark_role,
            "model_id": config.base_model_id,
            "model_revision": "fixture-revision",
            "adapter_method": "LoRA",
            "final_model_directory": str(final_adapter.resolve()),
            "parameter_counts": {
                "total_parameters": 100,
                "trainable_parameters": 10,
                "frozen_parameters": 90,
            },
            "timing": {
                "training_seconds": 2.0,
                "test_records": 3,
                "inference_milliseconds_per_record": 100.0,
            },
            "memory": {
                "measurement_method": (
                    "resource.getrusage(resource.RUSAGE_SELF).ru_maxrss"
                ),
                "baseline_peak_rss_mib": 50.0,
                "peak_process_rss_mib": 100.0,
                "incremental_peak_rss_mib": 50.0,
            },
            "artifact_files": lora_module.build_artifact_inventory(final_adapter),
            "label_mapping": {
                "id_to_label": {
                    "0": "Bearish",
                    "1": "Neutral",
                    "2": "Bullish",
                }
            },
            "source_files": {
                split_name: {
                    "path": str(split_file.resolve()),
                    "checksum_sha256": file_sha256(split_file),
                    "records": 3,
                }
                for split_name, split_file in split_files.items()
            },
        },
    )
    return config


def test_default_lora_configuration_is_valid() -> None:
    """Prepare the default config, validate it, and check adapter settings."""

    # Prepare
    config = LoraTrainingConfig()

    # Run
    validate_lora_config(config)

    # Check
    assert config.lora_rank == 8
    assert config.lora_alpha == 16
    assert config.target_modules == ("query", "value")
    assert config.modules_to_save == ("classifier",)


def test_shared_protocol_requires_class_weighting() -> None:
    """Reject a LoRA comparison when full BERT disables class weights."""

    # Prepare
    lora_config = LoraTrainingConfig()
    bert_config = SimpleNamespace(
        model_id=lora_config.base_model_id,
        number_of_epochs=lora_config.number_of_epochs,
        max_length=lora_config.max_length,
        train_batch_size=lora_config.train_batch_size,
        evaluation_batch_size=lora_config.evaluation_batch_size,
        gradient_accumulation_steps=lora_config.gradient_accumulation_steps,
        random_seed=lora_config.random_seed,
        use_class_weights=False,
    )

    # Run
    with pytest.raises(LoraTrainingError) as captured:
        validate_shared_protocol(bert_config, lora_config)

    # Check
    assert "enable class weights" in str(captured.value)


def test_schema_detection_prefers_explicit_fields() -> None:
    """Detect the sentence and integer-label fields from complete records."""

    # Prepare
    records = sample_records()

    # Run
    text_field = detect_text_field(records)
    label_field = detect_label_field(records)

    # Check
    assert text_field == "sentence"
    assert label_field == "label_id"


def test_named_and_numeric_labels_share_one_mapping() -> None:
    """Convert both source formats to Bearish=0, Neutral=1, Bullish=2."""

    # Prepare
    source_labels = ("Bearish", "neutral", " BULLISH ", 0, 2.0)

    # Run
    converted = [canonical_label_id(value) for value in source_labels]

    # Check
    assert converted == [0, 1, 2, 0, 2]
    assert tuple(LABEL_ORDER) == ("Bearish", "Neutral", "Bullish")


def test_class_weights_follow_balanced_formula() -> None:
    """Calculate N divided by K times class count for an imbalanced sample."""

    # Prepare
    labels = [0, 1, 1, 2, 2, 2]

    # Run
    weights = calculate_class_weights(labels)

    # Check
    expected = np.asarray([2.0, 1.0, 2.0 / 3.0], dtype=np.float32)
    np.testing.assert_allclose(weights, expected, rtol=1e-6)


def test_cross_split_sentence_leakage_is_rejected() -> None:
    """Reject a sentence that appears in both training and validation."""

    # Prepare
    split_records = {
        "train": [{"sentence": "Shared sentence"}],
        "validation": [{"sentence": " shared   sentence "}],
        "test": [{"sentence": "Different sentence"}],
    }

    # Run
    with pytest.raises(LoraTrainingError) as captured:
        validate_no_cross_split_leakage(split_records, "sentence")

    # Check
    assert "Sentence leakage" in str(captured.value)


def test_known_predictions_produce_expected_metrics() -> None:
    """Check accuracy, matrix, and macro F1 on one transparent example."""

    # Prepare
    labels = [0, 1, 2, 2]
    predictions = [0, 1, 1, 2]

    # Run
    summary = classification_summary(labels, predictions)

    # Check
    assert summary["accuracy"] == pytest.approx(0.75)
    assert summary["confusion_matrix"] == [[1, 0, 0], [0, 1, 0], [0, 1, 1]]
    assert summary["macro_f1"] == pytest.approx(7 / 9)


def test_existing_lora_outputs_are_protected(tmp_path: Path) -> None:
    """Reject previous adapter evidence unless replacement is explicit."""

    # Prepare
    config = build_lora_artifacts(tmp_path)

    # Run
    with pytest.raises(LoraTrainingError) as captured:
        protect_or_replace_outputs(config, replace_existing=False)

    # Check
    assert "already exist" in str(captured.value)


def test_complete_lora_artifacts_pass_review(tmp_path: Path) -> None:
    """Return a comparison-ready summary for one complete fixture."""

    # Prepare
    config = build_lora_artifacts(tmp_path)

    # Run
    summary = runner.validate_lora_artifacts(config)

    # Check
    assert summary["status"] == "trained_and_evaluated"
    assert summary["trainable_parameters"] == 10
    assert summary["total_parameters"] == 100
    assert summary["test_records"] == 3
    assert summary["confusion_matrix"] == [[1, 0, 0], [0, 1, 0], [0, 1, 0]]


def test_full_parameter_training_is_rejected(tmp_path: Path) -> None:
    """Reject a manifest that does not demonstrate parameter efficiency."""

    # Prepare
    config = build_lora_artifacts(tmp_path)
    manifest = json.loads(config.manifest_file.read_text(encoding="utf-8"))
    manifest["parameter_counts"]["trainable_parameters"] = 100
    manifest["parameter_counts"]["frozen_parameters"] = 0
    write_json(config.manifest_file, manifest)

    # Run
    with pytest.raises(runner.LoraArtifactError) as captured:
        runner.validate_lora_artifacts(config)

    # Check
    assert "fewer than all" in str(captured.value)


def test_lora_approved_split_count_is_enforced(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Reject LoRA evidence when one source split has the wrong row count."""

    # Prepare
    config = build_lora_artifacts(tmp_path)
    monkeypatch.setattr(
        runner,
        "EXPECTED_SPLIT_RECORDS",
        {"train": 4, "validation": 3, "test": 3},
    )

    # Run
    with pytest.raises(runner.LoraArtifactError) as captured:
        runner.validate_lora_artifacts(config)

    # Check
    assert "exactly 4" in str(captured.value)


def test_tampered_lora_adapter_checksum_is_rejected(tmp_path: Path) -> None:
    """Reject an adapter file changed after its checksum was recorded."""

    # Prepare
    config = build_lora_artifacts(tmp_path)
    (config.final_adapter_dir / "adapter_model.safetensors").write_text(
        "tampered",
        encoding="utf-8",
    )

    # Run
    with pytest.raises(runner.LoraArtifactError) as captured:
        runner.validate_lora_artifacts(config)

    # Check
    assert "checksums" in str(captured.value)


def test_missing_lora_per_class_metrics_are_rejected(tmp_path: Path) -> None:
    """Reject LoRA metrics that omit class-level precision, recall, and F1."""

    # Prepare
    config = build_lora_artifacts(tmp_path)
    metrics = json.loads(config.metrics_file.read_text(encoding="utf-8"))
    metrics["test_evaluation"].pop("per_class")
    write_json(config.metrics_file, metrics)

    # Run
    with pytest.raises(runner.LoraArtifactError) as captured:
        runner.validate_lora_artifacts(config)

    # Check
    assert "Per-class metrics" in str(captured.value)


def test_missing_lora_memory_evidence_is_rejected(tmp_path: Path) -> None:
    """Reject LoRA evidence without measured peak process memory."""

    # Prepare
    config = build_lora_artifacts(tmp_path)
    manifest = json.loads(config.manifest_file.read_text(encoding="utf-8"))
    manifest.pop("memory")
    write_json(config.manifest_file, manifest)

    # Run
    with pytest.raises(runner.LoraArtifactError) as captured:
        runner.validate_lora_artifacts(config)

    # Check
    assert "memory evidence" in str(captured.value)
