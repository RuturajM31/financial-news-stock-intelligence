"""
Test the full-BERT benchmark runner without downloading or training a model.

Purpose
-------
Protect the safety layer around the existing BERT wrapper. The tests verify
output protection, test-file discovery, confusion-matrix grain, artifact
review, and delegation to the shared training engine.

Inputs and data journey
-----------------------
Temporary JSONL, manifest, metrics, and model files imitate one completed
experiment. Small dataclass configurations flow through the same helpers used
by ``scripts/run_full_bert.py``.

Outputs and downstream use
--------------------------
Passing tests show that the runner will reject unsafe replacement, incomplete
model evidence, incorrect test-row counts, and an invalid full-fine-tuning
parameter contract before LoRA or champion selection can begin.

Safety and limitations
----------------------
No test imports Transformers, accesses the network, or starts training. The
real full-BERT run remains the required end-to-end proof.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


RUNNER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_full_bert.py"


def load_runner() -> ModuleType:
    """Load the command-line script without executing its main function."""

    specification = importlib.util.spec_from_file_location(
        "full_bert_runner_under_test",
        RUNNER_PATH,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Could not load full-BERT runner: {RUNNER_PATH}")

    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


runner = load_runner()


@pytest.fixture(autouse=True)
def use_small_split_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use three-row fixtures while production keeps 2,413/517/518."""

    monkeypatch.setattr(
        runner,
        "EXPECTED_SPLIT_RECORDS",
        {"train": 3, "validation": 3, "test": 3},
    )


@dataclass
class FakeConfig:
    """Provide only the path fields required by the artifact validator."""

    test_file: Path
    checkpoint_dir: Path
    final_model_dir: Path
    metrics_file: Path
    manifest_file: Path


def file_sha256(file_path: Path) -> str:
    """Return one fixture file checksum for manifest evidence."""

    return hashlib.sha256(file_path.read_bytes()).hexdigest()


def write_json(file_path: Path, payload: dict[str, Any]) -> None:
    """Write one deterministic UTF-8 JSON object for a test fixture."""

    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def build_complete_artifacts(tmp_path: Path) -> FakeConfig:
    """Create one internally consistent three-record full-BERT experiment."""

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
                    f'{{"text": "{prefix} a", "label_id": 0}}',
                    f'{{"text": "{prefix} b", "label_id": 1}}',
                    f'{{"text": "{prefix} c", "label_id": 2}}',
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        split_files[split_name] = split_file

    test_file = split_files["test"]
    final_model_dir = tmp_path / "models" / "final_model"
    final_model_dir.mkdir(parents=True)
    for file_name in runner.REQUIRED_FINAL_MODEL_FILES:
        (final_model_dir / file_name).write_text("fixture", encoding="utf-8")

    metrics_file = tmp_path / "reports" / "bert_metrics.json"
    manifest_file = tmp_path / "manifests" / "bert_manifest.json"

    write_json(
        metrics_file,
        {
            "test_metrics": {
                "test_accuracy": 2 / 3,
                "test_macro_f1": 0.65,
                "test_weighted_f1": 0.66,
                "test_runtime": 0.5,
            },
            "test_evaluation": {
                "confusion_matrix": [
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 1, 0],
                ],
                "label_order": list(runner.LABEL_ORDER),
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
        manifest_file,
        {
            "status": "trained_and_evaluated",
            "model_id": runner.EXPECTED_MODEL_ID,
            "experiment_name": runner.EXPECTED_EXPERIMENT_NAME,
            "model_family": runner.EXPECTED_MODEL_FAMILY,
            "benchmark_role": runner.EXPECTED_BENCHMARK_ROLE,
            "model_revision": "fixture-revision",
            "final_model_directory": str(final_model_dir.resolve()),
            "parameter_counts": {
                "total_parameters": 100,
                "trainable_parameters": 100,
            },
            "timing": {"training_seconds": 2.0},
            "memory": {
                "measurement_method": (
                    "resource.getrusage(resource.RUSAGE_SELF).ru_maxrss"
                ),
                "baseline_peak_rss_mib": 50.0,
                "peak_process_rss_mib": 120.0,
                "incremental_peak_rss_mib": 70.0,
            },
            "artifact_files": runner.build_artifact_inventory(final_model_dir),
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

    return FakeConfig(
        test_file=test_file,
        checkpoint_dir=tmp_path / "models" / "checkpoints",
        final_model_dir=final_model_dir,
        metrics_file=metrics_file,
        manifest_file=manifest_file,
    )


def test_resolve_test_file_uses_dataclass_field(tmp_path: Path) -> None:
    """Prepare a config, resolve its test path, and check the exact result."""

    # Prepare
    config = build_complete_artifacts(tmp_path)

    # Run
    resolved = runner.resolve_test_file(config)

    # Check
    assert resolved == config.test_file.resolve()


def test_existing_outputs_are_protected(tmp_path: Path) -> None:
    """Reject an existing evidence file when replacement is not approved."""

    # Prepare
    config = build_complete_artifacts(tmp_path)

    # Run
    with pytest.raises(runner.FullBertBenchmarkError) as captured:
        runner.protect_or_replace_outputs(config, replace_existing=False)

    # Check
    assert "already exist" in str(captured.value)


def test_explicit_replacement_removes_only_controlled_paths(
    tmp_path: Path,
) -> None:
    """Remove dedicated outputs while preserving an unrelated sentinel."""

    # Prepare
    config = build_complete_artifacts(tmp_path)
    sentinel = tmp_path / "keep-me.txt"
    sentinel.write_text("safe", encoding="utf-8")

    # Run
    runner.protect_or_replace_outputs(config, replace_existing=True)

    # Check
    assert not config.final_model_dir.exists()
    assert not config.metrics_file.exists()
    assert not config.manifest_file.exists()
    assert sentinel.read_text(encoding="utf-8") == "safe"


def test_confusion_matrix_must_match_test_grain() -> None:
    """Reject a matrix whose cell total differs from the test row count."""

    # Prepare
    metrics = {
        "test_evaluation": {
            "confusion_matrix": [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ]
        }
    }

    # Run
    with pytest.raises(runner.FullBertBenchmarkError) as captured:
        runner.validate_confusion_matrix(metrics, expected_records=4)

    # Check
    assert "Expected 4" in str(captured.value)


def test_complete_artifacts_pass_review(tmp_path: Path) -> None:
    """Return a concise summary for one complete full-BERT fixture."""

    # Prepare
    config = build_complete_artifacts(tmp_path)

    # Run
    summary = runner.validate_full_bert_artifacts(config)

    # Check
    assert summary["status"] == "trained_and_evaluated"
    assert summary["test_records"] == 3
    assert summary["total_parameters"] == 100
    assert summary["trainable_parameters"] == 100
    assert summary["confusion_matrix"] == [[1, 0, 0], [0, 1, 0], [0, 1, 0]]


def test_source_evidence_without_record_counts_is_rejected(
    tmp_path: Path,
) -> None:
    """Reject manifests that do not prove the approved split counts."""

    # Prepare
    config = build_complete_artifacts(tmp_path)
    manifest = json.loads(config.manifest_file.read_text(encoding="utf-8"))
    for details in manifest["source_files"].values():
        details.pop("records")
    write_json(config.manifest_file, manifest)

    # Run
    with pytest.raises(runner.FullBertBenchmarkError) as captured:
        runner.validate_full_bert_artifacts(config)

    # Check
    assert "exactly" in str(captured.value)


def test_partial_fine_tuning_is_rejected(tmp_path: Path) -> None:
    """Reject a full-BERT manifest that reports frozen parameters."""

    # Prepare
    config = build_complete_artifacts(tmp_path)
    manifest = json.loads(config.manifest_file.read_text(encoding="utf-8"))
    manifest["parameter_counts"]["trainable_parameters"] = 10
    write_json(config.manifest_file, manifest)

    # Run
    with pytest.raises(runner.FullBertBenchmarkError) as captured:
        runner.validate_full_bert_artifacts(config)

    # Check
    assert "every parameter" in str(captured.value)


def test_run_delegates_once_before_artifact_review(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Pass the validated config to training exactly once, then review it."""

    # Prepare
    config = build_complete_artifacts(tmp_path)
    captured: dict[str, Any] = {"calls": 0}

    def fake_config_type() -> FakeConfig:
        """Return the prepared configuration without reading project files."""

        return config

    def fake_validate(received: FakeConfig) -> None:
        """Confirm the wrapper validates the exact prepared object."""

        assert received is config

    def fake_train(received: FakeConfig) -> dict[str, str]:
        """Record one delegated training call without starting a model."""

        captured["calls"] += 1
        assert received is config
        return {"status": "fixture"}

    monkeypatch.setattr(runner, "ensure_isolated_environment", lambda: None)
    monkeypatch.setattr(runner, "configure_current_run_evidence", lambda config: None)
    monkeypatch.setattr(runner, "preserve_current_run_history", lambda config: None)
    monkeypatch.setattr(
        runner,
        "load_bert_contract",
        lambda: (fake_config_type, fake_train, fake_validate),
    )
    monkeypatch.setattr(
        runner,
        "protect_or_replace_outputs",
        lambda received, replace_existing: None,
    )

    # Run
    summary = runner.run_full_bert()

    # Check
    assert captured["calls"] == 1
    assert summary["test_records"] == 3


def test_approved_split_count_is_enforced(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Reject a valid-looking manifest when one split has the wrong grain."""

    # Prepare
    config = build_complete_artifacts(tmp_path)
    monkeypatch.setattr(
        runner,
        "EXPECTED_SPLIT_RECORDS",
        {"train": 4, "validation": 3, "test": 3},
    )

    # Run
    with pytest.raises(runner.FullBertBenchmarkError) as captured:
        runner.validate_full_bert_artifacts(config)

    # Check
    assert "exactly 4" in str(captured.value)


def test_tampered_model_file_checksum_is_rejected(tmp_path: Path) -> None:
    """Reject a saved BERT file changed after manifest creation."""

    # Prepare
    config = build_complete_artifacts(tmp_path)
    (config.final_model_dir / "model.safetensors").write_text(
        "tampered",
        encoding="utf-8",
    )

    # Run
    with pytest.raises(runner.FullBertBenchmarkError) as captured:
        runner.validate_full_bert_artifacts(config)

    # Check
    assert "checksums" in str(captured.value)


def test_missing_full_bert_per_class_metrics_are_rejected(tmp_path: Path) -> None:
    """Reject aggregate-only metrics that omit class-level quality evidence."""

    # Prepare
    config = build_complete_artifacts(tmp_path)
    metrics = json.loads(config.metrics_file.read_text(encoding="utf-8"))
    metrics["test_evaluation"].pop("per_class")
    write_json(config.metrics_file, metrics)

    # Run
    with pytest.raises(runner.FullBertBenchmarkError) as captured:
        runner.validate_full_bert_artifacts(config)

    # Check
    assert "Per-class metrics" in str(captured.value)


def test_missing_full_bert_memory_evidence_is_rejected(tmp_path: Path) -> None:
    """Reject a full-BERT manifest without measured peak process memory."""

    # Prepare
    config = build_complete_artifacts(tmp_path)
    manifest = json.loads(config.manifest_file.read_text(encoding="utf-8"))
    manifest.pop("memory")
    write_json(config.manifest_file, manifest)

    # Run
    with pytest.raises(runner.FullBertBenchmarkError) as captured:
        runner.validate_full_bert_artifacts(config)

    # Check
    assert "memory evidence" in str(captured.value)

def test_lowercase_per_class_keys_are_normalized(tmp_path: Path) -> None:
    """Accept the established shared-engine lowercase class-key schema."""

    # Prepare
    config = build_complete_artifacts(tmp_path)
    metrics = json.loads(config.metrics_file.read_text(encoding="utf-8"))
    metrics["test_evaluation"]["per_class"] = {
        key.casefold(): value
        for key, value in metrics["test_evaluation"]["per_class"].items()
    }
    write_json(config.metrics_file, metrics)

    # Run
    summary = runner.validate_full_bert_artifacts(config)

    # Check
    assert set(summary["per_class_metrics"]) == set(runner.LABEL_ORDER)


def test_scalar_per_class_metrics_support_shared_engine(
    tmp_path: Path,
) -> None:
    """Use test_bearish/test_neutral/test_bullish metrics as a fallback."""

    # Prepare
    config = build_complete_artifacts(tmp_path)
    metrics = json.loads(config.metrics_file.read_text(encoding="utf-8"))
    nested = metrics["test_evaluation"].pop("per_class")
    for label_name, values in nested.items():
        prefix = label_name.casefold()
        metrics["test_metrics"][f"test_{prefix}_precision"] = values[
            "precision"
        ]
        metrics["test_metrics"][f"test_{prefix}_recall"] = values["recall"]
        metrics["test_metrics"][f"test_{prefix}_f1"] = values["f1"]
    write_json(config.metrics_file, metrics)

    # Run
    summary = runner.validate_full_bert_artifacts(config)

    # Check
    assert summary["per_class_metrics"]["Bearish"]["support"] == 1


def test_per_class_values_must_match_confusion_matrix(tmp_path: Path) -> None:
    """Reject present but internally inconsistent per-class evidence."""

    # Prepare
    config = build_complete_artifacts(tmp_path)
    metrics = json.loads(config.metrics_file.read_text(encoding="utf-8"))
    metrics["test_evaluation"]["per_class"]["Bearish"]["precision"] = 0.5
    write_json(config.metrics_file, metrics)

    # Run
    with pytest.raises(runner.FullBertBenchmarkError) as captured:
        runner.validate_full_bert_artifacts(config)

    # Check
    assert "confusion matrix" in str(captured.value)


def test_windows_memory_fallback_uses_psutil(monkeypatch: pytest.MonkeyPatch) -> None:
    """Measure current RSS when the Unix resource module is unavailable."""

    monkeypatch.setattr(runner, "resource", None)

    assert runner.current_peak_rss_mib() > 0
    assert "psutil.Process().memory_info().rss" in runner.memory_measurement_method()


def test_memory_validator_accepts_windows_psutil_evidence(tmp_path: Path) -> None:
    """Accept explicit current-RSS evidence while retaining Unix support."""

    config = build_complete_artifacts(tmp_path)
    manifest = json.loads(config.manifest_file.read_text(encoding="utf-8"))
    manifest["memory"]["measurement_method"] = (
        "psutil.Process().memory_info().rss (current RSS; peak unavailable)"
    )
    write_json(config.manifest_file, manifest)

    summary = runner.validate_full_bert_artifacts(config)

    assert "psutil" in summary["memory"]["measurement_method"]