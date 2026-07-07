"""
Test the final DistilBERT, BERT, and LoRA comparison contract.

Purpose
-------
Protect evidence normalization, complete split provenance, quality ranking,
deployment selection, base-model alignment, and atomic champion output.

Inputs and data journey
-----------------------
Tiny manifests, metrics files, source splits, and artifact directories imitate
three completed experiments. They flow through the same comparison module
used after the real training runs.

Outputs and downstream use
--------------------------
Passing tests show that model selection is deterministic, traceable, and
unable to compare different datasets or different BERT base revisions.

Safety and limitations
----------------------
The tests do not load weights, access the network, or alter deployment. Real
scores, timing, and saved artifacts remain the final source of truth.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

import financial_news_intelligence.models.sentiment_comparison as comparison


RUNNER_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_sentiment_comparison.py"
)


def load_runner() -> ModuleType:
    """Load the comparison runner without executing its command-line entry."""

    specification = importlib.util.spec_from_file_location(
        "comparison_runner_under_test",
        RUNNER_PATH,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Could not load comparison runner: {RUNNER_PATH}")

    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


runner = load_runner()


@pytest.fixture(autouse=True)
def use_small_split_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use three-row fixtures while production keeps 2,413/517/518."""

    small_counts = {"train": 3, "validation": 3, "test": 3}
    monkeypatch.setattr(comparison, "EXPECTED_SPLIT_RECORDS", small_counts)
    monkeypatch.setattr(runner, "EXPECTED_SPLIT_RECORDS", small_counts)


def write_json(file_path: Path, payload: dict[str, Any]) -> None:
    """Write one deterministic UTF-8 JSON fixture."""

    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def source_rows(split_name: str, record_count: int, variant: str) -> list[str]:
    """Create transparent source rows with all three sentiment classes."""

    rows = [
        f'{{"text": "{split_name} {variant} a", "label_id": 0}}',
        f'{{"text": "{split_name} {variant} b", "label_id": 1}}',
        f'{{"text": "{split_name} {variant} c", "label_id": 2}}',
    ]
    if record_count == 4:
        rows.append(
            f'{{"text": "{split_name} {variant} d", "label_id": 0}}'
        )
    elif record_count != 3:
        raise ValueError("Fixture supports only three or four records.")
    return rows


def build_model_evidence(
    tmp_path: Path,
    model_key: str,
    macro_f1: float,
    weighted_f1: float,
    accuracy: float,
    total_parameters: int,
    trainable_parameters: int,
    test_runtime: float,
    test_records: int = 3,
    model_id: str | None = None,
    model_revision: str | None = None,
    train_variant: str = "shared",
) -> tuple[Path, Path]:
    """Create one complete model directory, manifest, and metrics pair."""

    model_dir = tmp_path / model_key / "model"
    model_dir.mkdir(parents=True)
    (model_dir / "weights.bin").write_bytes(b"fixture-weights")

    source_evidence: dict[str, dict[str, Any]] = {}
    for split_name in ("train", "validation", "test"):
        record_count = test_records if split_name == "test" else 3
        variant = train_variant if split_name == "train" else "shared"
        split_file = tmp_path / model_key / f"{split_name}.jsonl"
        split_file.write_text(
            "\n".join(source_rows(split_name, record_count, variant)) + "\n",
            encoding="utf-8",
        )
        source_evidence[split_name] = {
            "path": str(split_file.resolve()),
            "checksum_sha256": hashlib.sha256(
                split_file.read_bytes()
            ).hexdigest(),
            "records": record_count,
        }

    manifest_path = tmp_path / model_key / "manifest.json"
    metrics_path = tmp_path / model_key / "metrics.json"
    write_json(
        manifest_path,
        {
            "status": "trained_and_evaluated",
            "experiment_name": f"{model_key} experiment",
            "model_family": model_key,
            "benchmark_role": "comparison_fixture",
            "model_id": model_id or f"fixture/{model_key}",
            "model_revision": model_revision or f"revision-{model_key}",
            "final_model_directory": str(model_dir.resolve()),
            "parameter_counts": {
                "total_parameters": total_parameters,
                "trainable_parameters": trainable_parameters,
            },
            "timing": {"training_seconds": 10.0},
            "memory": {
                "measurement_method": (
                    "resource.getrusage(resource.RUSAGE_SELF).ru_maxrss"
                ),
                "peak_process_rss_mib": float(total_parameters),
            },
            "artifact_files": comparison.artifact_inventory(model_dir),
            "source_files": source_evidence,
        },
    )

    matrix = (
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        if test_records == 3
        else [[2, 0, 0], [0, 1, 0], [0, 0, 1]]
    )
    write_json(
        metrics_path,
        {
            "test_metrics": {
                "test_accuracy": accuracy,
                "test_macro_f1": macro_f1,
                "test_weighted_f1": weighted_f1,
                "test_runtime": test_runtime,
            },
            "test_evaluation": {
                "confusion_matrix": matrix,
                "label_order": list(comparison.LABEL_ORDER),
                "per_class": {
                    label_name: {
                        "precision": 1.0,
                        "recall": 1.0,
                        "f1": 1.0,
                        "support": sum(matrix[class_id]),
                    }
                    for class_id, label_name in enumerate(comparison.LABEL_ORDER)
                },
            },
        },
    )
    return manifest_path, metrics_path


def load_fixture_model(
    tmp_path: Path,
    model_key: str,
    macro_f1: float = 0.88,
    weighted_f1: float = 0.90,
    accuracy: float = 0.89,
    total_parameters: int = 100,
    trainable_parameters: int = 100,
    test_runtime: float = 0.3,
    **kwargs: Any,
) -> comparison.ModelEvidence:
    """Build and normalize one model fixture in a single helper."""

    paths = build_model_evidence(
        tmp_path,
        model_key,
        macro_f1,
        weighted_f1,
        accuracy,
        total_parameters,
        trainable_parameters,
        test_runtime,
        **kwargs,
    )
    return comparison.load_model_evidence(model_key, *paths)


def test_model_evidence_calculates_latency_and_memory(tmp_path: Path) -> None:
    """Normalize one model and check the documented formulas."""

    # Prepare
    paths = build_model_evidence(
        tmp_path,
        "distilbert",
        0.88,
        0.90,
        0.89,
        1_048_576,
        1_048_576,
        0.3,
    )

    # Run
    evidence = comparison.load_model_evidence("distilbert", *paths)

    # Check
    assert evidence.test_records == 3
    assert evidence.inference_milliseconds_per_record == pytest.approx(100.0)
    assert evidence.estimated_fp32_parameter_memory_mib == pytest.approx(4.0)
    assert evidence.trainable_percentage == pytest.approx(100.0)
    assert set(evidence.source_files) == {"train", "validation", "test"}
    assert len(evidence.manifest_sha256) == 64
    assert len(evidence.metrics_sha256) == 64


def test_legacy_distilbert_identity_fields_are_resolved(tmp_path: Path) -> None:
    """Support the verified baseline created before new identity fields."""

    # Prepare
    manifest_path, metrics_path = build_model_evidence(
        tmp_path,
        "distilbert",
        0.88,
        0.90,
        0.89,
        60,
        60,
        0.2,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("model_family")
    manifest.pop("benchmark_role")
    write_json(manifest_path, manifest)

    # Run
    evidence = comparison.load_model_evidence(
        "distilbert",
        manifest_path,
        metrics_path,
    )

    # Check
    assert evidence.model_family == "DistilBERT"
    assert evidence.benchmark_role == "baseline_full_fine_tuning"


def test_common_dataset_rejects_mismatched_test_populations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Reject models evaluated on different untouched test populations."""

    # Prepare
    first = load_fixture_model(tmp_path, "first", test_records=3)
    monkeypatch.setattr(
        comparison,
        "EXPECTED_SPLIT_RECORDS",
        {"train": 3, "validation": 3, "test": 4},
    )
    second = load_fixture_model(tmp_path, "second", test_records=4)

    # Run
    with pytest.raises(comparison.SentimentComparisonError) as captured:
        comparison.validate_common_dataset([first, second])

    # Check
    assert "test split" in str(captured.value)


def test_common_dataset_rejects_mismatched_training_split(
    tmp_path: Path,
) -> None:
    """Reject models trained from different source sentences."""

    # Prepare
    first = load_fixture_model(tmp_path, "first", train_variant="shared")
    second = load_fixture_model(tmp_path, "second", train_variant="changed")

    # Run
    with pytest.raises(comparison.SentimentComparisonError) as captured:
        comparison.validate_common_dataset([first, second])

    # Check
    assert "train split" in str(captured.value)


def test_bert_base_revision_mismatch_is_rejected(tmp_path: Path) -> None:
    """Reject full BERT and LoRA evidence from different base revisions."""

    # Prepare
    distilbert = load_fixture_model(tmp_path, "distilbert")
    bert = load_fixture_model(
        tmp_path,
        "bert",
        model_id="google-bert/bert-base-uncased",
        model_revision="revision-one",
    )
    lora = load_fixture_model(
        tmp_path,
        "bert_lora",
        model_id="google-bert/bert-base-uncased",
        model_revision="revision-two",
    )

    # Run
    with pytest.raises(comparison.SentimentComparisonError) as captured:
        comparison.validate_bert_base_alignment([distilbert, bert, lora])

    # Check
    assert "different base model revisions" in str(captured.value)


def test_quality_ranking_prioritizes_macro_f1(tmp_path: Path) -> None:
    """Select the highest macro F1 even when another model is smaller."""

    # Prepare
    models = [
        load_fixture_model(tmp_path, "distilbert", 0.88, 0.90, 0.90, 60, 60),
        load_fixture_model(tmp_path, "bert", 0.90, 0.91, 0.91, 110, 110),
        load_fixture_model(tmp_path, "bert_lora", 0.89, 0.90, 0.90, 110, 5),
    ]

    # Run
    ranked, quality_champion, deployment_champion = comparison.rank_models(
        models
    )

    # Check
    assert [model.model_key for model in ranked] == [
        "bert",
        "bert_lora",
        "distilbert",
    ]
    assert quality_champion.model_key == "bert"
    assert deployment_champion.model_key == "distilbert"


def test_quality_tie_uses_weighted_f1_then_accuracy(tmp_path: Path) -> None:
    """Apply the documented quality tie-breakers in order."""

    # Prepare
    models = [
        load_fixture_model(tmp_path, "a", 0.9, 0.90, 0.95),
        load_fixture_model(tmp_path, "b", 0.9, 0.91, 0.90),
        load_fixture_model(tmp_path, "c", 0.8, 0.80, 0.80),
    ]

    # Run
    ranked, quality_champion, _ = comparison.rank_models(models)

    # Check
    assert ranked[0].model_key == "b"
    assert quality_champion.model_key == "b"


def test_build_comparison_writes_traceable_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Write ranking and champion files linked to source checksums."""

    # Prepare
    distil_paths = build_model_evidence(
        tmp_path,
        "distilbert",
        0.895,
        0.90,
        0.90,
        60,
        60,
        0.2,
    )
    shared_base_id = "google-bert/bert-base-uncased"
    shared_revision = "shared-bert-revision"
    bert_paths = build_model_evidence(
        tmp_path,
        "bert",
        0.91,
        0.92,
        0.92,
        110,
        110,
        0.4,
        model_id=shared_base_id,
        model_revision=shared_revision,
    )
    lora_paths = build_model_evidence(
        tmp_path,
        "bert_lora",
        0.90,
        0.91,
        0.91,
        110,
        5,
        0.4,
        model_id=shared_base_id,
        model_revision=shared_revision,
    )
    monkeypatch.setattr(comparison, "DISTILBERT_MANIFEST", distil_paths[0])
    monkeypatch.setattr(comparison, "DISTILBERT_METRICS", distil_paths[1])
    monkeypatch.setattr(comparison, "BERT_MANIFEST", bert_paths[0])
    monkeypatch.setattr(comparison, "BERT_METRICS", bert_paths[1])
    monkeypatch.setattr(comparison, "LORA_MANIFEST", lora_paths[0])
    monkeypatch.setattr(comparison, "LORA_METRICS", lora_paths[1])
    comparison_file = tmp_path / "comparison.json"
    champion_file = tmp_path / "champion.json"

    # Run
    result = comparison.build_comparison(comparison_file, champion_file)

    # Check
    assert result["quality_champion"] == "bert"
    assert result["deployment_champion"] == "distilbert"
    assert set(result["dataset_sources"]) == {"train", "validation", "test"}
    champion = json.loads(champion_file.read_text(encoding="utf-8"))
    assert champion["official_quality_champion"] == "bert"
    assert champion["recommended_deployment_model"] == "distilbert"
    assert champion["automatic_deployment_change"] is False
    assert len(champion["source_evidence"]["bert"]["metrics_sha256"]) == 64


def test_runner_rejects_incomplete_ranking(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Reject saved comparison evidence that omits one required model."""

    # Prepare
    comparison_file = tmp_path / "comparison.json"
    champion_file = tmp_path / "champion.json"
    write_json(
        comparison_file,
        {
            "status": "comparison_completed",
            "label_order": list(comparison.LABEL_ORDER),
            "test_records": 3,
            "deployment_quality_tolerance_macro_f1": 0.02,
            "quality_ranking": ["bert", "distilbert"],
            "quality_champion": "bert",
            "deployment_champion": "distilbert",
            "models": [
                {"model_key": "bert"},
                {"model_key": "distilbert"},
            ],
        },
    )
    write_json(
        champion_file,
        {
            "status": "champion_selected",
            "label_order": list(comparison.LABEL_ORDER),
            "test_records": 3,
            "official_quality_champion": "bert",
            "recommended_deployment_model": "distilbert",
            "source_evidence": {},
            "automatic_deployment_change": False,
        },
    )
    monkeypatch.setattr(runner, "COMPARISON_FILE", comparison_file)
    monkeypatch.setattr(runner, "CHAMPION_MANIFEST_FILE", champion_file)

    # Run
    with pytest.raises(runner.ComparisonArtifactError) as captured:
        runner.validate_comparison_outputs()

    # Check
    assert "three models" in str(captured.value)


def test_runner_rejects_missing_dataset_provenance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Reject comparison output that omits the shared split evidence."""

    # Prepare
    comparison_file = tmp_path / "comparison_missing_sources.json"
    champion_file = tmp_path / "champion_with_sources.json"
    models = [
        {
            "model_key": "distilbert",
            "model_id": "distilbert/distilbert-base-uncased",
            "model_revision": "distil-revision",
            "source_files": {},
        },
        {
            "model_key": "bert",
            "model_id": "google-bert/bert-base-uncased",
            "model_revision": "bert-revision",
            "source_files": {},
        },
        {
            "model_key": "bert_lora",
            "model_id": "google-bert/bert-base-uncased",
            "model_revision": "bert-revision",
            "source_files": {},
        },
    ]
    write_json(
        comparison_file,
        {
            "status": "comparison_completed",
            "label_order": list(comparison.LABEL_ORDER),
            "test_records": 3,
            "deployment_quality_tolerance_macro_f1": 0.02,
            "quality_ranking": ["bert", "bert_lora", "distilbert"],
            "quality_champion": "bert",
            "deployment_champion": "distilbert",
            "models": models,
        },
    )
    write_json(
        champion_file,
        {
            "status": "champion_selected",
            "label_order": list(comparison.LABEL_ORDER),
            "test_records": 3,
            "official_quality_champion": "bert",
            "recommended_deployment_model": "distilbert",
            "source_evidence": {
                "distilbert": {},
                "bert": {},
                "bert_lora": {},
            },
            "automatic_deployment_change": False,
        },
    )
    monkeypatch.setattr(runner, "COMPARISON_FILE", comparison_file)
    monkeypatch.setattr(runner, "CHAMPION_MANIFEST_FILE", champion_file)

    # Run
    with pytest.raises(runner.ComparisonArtifactError) as captured:
        runner.validate_comparison_outputs()

    # Check
    assert "dataset sources" in str(captured.value)


def test_new_bert_evidence_requires_measured_memory(tmp_path: Path) -> None:
    """Reject new full-BERT evidence that falls back to estimated memory."""

    # Prepare
    manifest_path, metrics_path = build_model_evidence(
        tmp_path,
        "bert",
        0.9,
        0.9,
        0.9,
        100,
        100,
        0.2,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("memory")
    write_json(manifest_path, manifest)

    # Run
    with pytest.raises(comparison.SentimentComparisonError) as captured:
        comparison.load_model_evidence("bert", manifest_path, metrics_path)

    # Check
    assert "measured peak process RSS" in str(captured.value)


def test_new_bert_evidence_requires_artifact_checksums(tmp_path: Path) -> None:
    """Reject new full-BERT evidence without a saved-file checksum inventory."""

    # Prepare
    manifest_path, metrics_path = build_model_evidence(
        tmp_path,
        "bert",
        0.9,
        0.9,
        0.9,
        100,
        100,
        0.2,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("artifact_files")
    write_json(manifest_path, manifest)

    # Run
    with pytest.raises(comparison.SentimentComparisonError) as captured:
        comparison.load_model_evidence("bert", manifest_path, metrics_path)

    # Check
    assert "artifact checksum inventory" in str(captured.value)


def test_legacy_distilbert_missing_record_fields_uses_verified_file_grain(
    tmp_path: Path,
) -> None:
    """Allow the older baseline only after its actual split sizes are checked."""

    # Prepare
    manifest_path, metrics_path = build_model_evidence(
        tmp_path,
        "distilbert",
        0.88,
        0.89,
        0.89,
        60,
        60,
        0.2,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for details in manifest["source_files"].values():
        details.pop("records")
    write_json(manifest_path, manifest)

    # Run
    evidence = comparison.load_model_evidence(
        "distilbert",
        manifest_path,
        metrics_path,
    )

    # Check
    assert evidence.source_files["train"]["records"] == 3
    assert evidence.source_files["validation"]["records"] == 3
    assert evidence.source_files["test"]["records"] == 3


def test_memory_basis_never_mixes_measured_and_estimated_values(
    tmp_path: Path,
) -> None:
    """Use one FP32 fallback basis when the legacy model lacks measured RSS."""

    # Prepare
    distil_manifest, distil_metrics = build_model_evidence(
        tmp_path,
        "distilbert",
        0.88,
        0.89,
        0.89,
        60,
        60,
        0.2,
    )
    distil_payload = json.loads(distil_manifest.read_text(encoding="utf-8"))
    distil_payload.pop("memory")
    write_json(distil_manifest, distil_payload)
    models = [
        comparison.load_model_evidence(
            "distilbert",
            distil_manifest,
            distil_metrics,
        ),
        load_fixture_model(tmp_path, "bert", total_parameters=110),
        load_fixture_model(tmp_path, "bert_lora", total_parameters=110),
    ]

    # Run
    normalized = comparison.apply_common_memory_basis(models)

    # Check
    assert {
        model.comparison_memory_source for model in normalized
    } == {"fp32_parameter_estimate_consistent_fallback"}
    assert all(
        model.comparison_memory_mib
        == model.estimated_fp32_parameter_memory_mib
        for model in normalized
    )

def test_comparison_accepts_lowercase_per_class_keys(tmp_path: Path) -> None:
    """Normalize established lowercase keys before three-model comparison."""

    # Prepare
    manifest_path, metrics_path = build_model_evidence(
        tmp_path,
        "distilbert",
        1.0,
        1.0,
        1.0,
        100,
        100,
        0.3,
    )
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["test_evaluation"]["per_class"] = {
        key.casefold(): value
        for key, value in metrics["test_evaluation"]["per_class"].items()
    }
    write_json(metrics_path, metrics)

    # Run
    evidence = comparison.load_model_evidence(
        "distilbert",
        manifest_path,
        metrics_path,
    )

    # Check
    assert set(evidence.per_class_metrics) == set(comparison.LABEL_ORDER)


def test_comparison_accepts_scalar_per_class_fallback(tmp_path: Path) -> None:
    """Read shared Trainer scalar metrics when no nested report is present."""

    # Prepare
    manifest_path, metrics_path = build_model_evidence(
        tmp_path,
        "bert",
        1.0,
        1.0,
        1.0,
        100,
        100,
        0.3,
    )
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    nested = metrics["test_evaluation"].pop("per_class")
    for label_name, values in nested.items():
        prefix = label_name.casefold()
        metrics["test_metrics"][f"test_{prefix}_precision"] = values[
            "precision"
        ]
        metrics["test_metrics"][f"test_{prefix}_recall"] = values["recall"]
        metrics["test_metrics"][f"test_{prefix}_f1"] = values["f1"]
    write_json(metrics_path, metrics)

    # Run
    evidence = comparison.load_model_evidence(
        "bert",
        manifest_path,
        metrics_path,
    )

    # Check
    assert evidence.per_class_metrics["Bullish"]["f1"] == pytest.approx(1.0)


def test_comparison_rejects_matrix_metric_disagreement(tmp_path: Path) -> None:
    """Reject class metrics that disagree with the saved confusion matrix."""

    # Prepare
    manifest_path, metrics_path = build_model_evidence(
        tmp_path,
        "bert",
        1.0,
        1.0,
        1.0,
        100,
        100,
        0.3,
    )
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["test_evaluation"]["per_class"]["Neutral"]["recall"] = 0.25
    write_json(metrics_path, metrics)

    # Run
    with pytest.raises(comparison.SentimentComparisonError) as captured:
        comparison.load_model_evidence("bert", manifest_path, metrics_path)

    # Check
    assert "confusion matrix" in str(captured.value)
