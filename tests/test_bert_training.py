"""
Focused tests for the compact BERT training wrapper.

Purpose
-------
Protect the contract between the BERT configuration wrapper and the verified
shared Transformer training engine.

Data journey
------------
``BertTrainingConfig`` defines BERT identity and output paths. The wrapper
validates that configuration, then passes it unchanged to the shared engine.
The shared engine later loads the verified dataset splits, tokenizes text,
trains the model, evaluates it, and saves reproducibility evidence.

Safety
------
These unit tests do not download model weights, read the training dataset,
train a neural network, create checkpoints, or modify production artifacts.
"""

from dataclasses import fields
from pathlib import Path
from typing import Any

import pytest

import financial_news_intelligence.models.bert_training as bert_module
from financial_news_intelligence.models.bert_training import (
    BERT_MODEL_ID,
    BertTrainingConfig,
    run_bert_training,
    validate_bert_config,
)
from financial_news_intelligence.models.distilbert_training import (
    DistilBertTrainingConfig,
)


# These fields must differ because they identify BERT or store BERT outputs.
BERT_OVERRIDE_FIELDS = {
    "experiment_name",
    "model_family",
    "benchmark_role",
    "model_id",
    "checkpoint_dir",
    "final_model_dir",
    "metrics_file",
    "manifest_file",
    "run_name",
}


def test_default_bert_identity_is_approved() -> None:
    """Validate the model identity written into BERT evidence files."""

    # Prepare: build the default BERT configuration.
    config = BertTrainingConfig()

    # Run: apply shared rules and the BERT-specific model-ID rule.
    validate_bert_config(config)

    # Check: the experiment must identify full BERT fine-tuning.
    assert config.model_id == BERT_MODEL_ID
    assert config.experiment_name == "BERT Financial Sentiment"
    assert config.model_family == "BERT"
    assert config.benchmark_role == "full_fine_tuning_comparison"


def test_distilbert_identity_remains_unchanged() -> None:
    """Protect the completed DistilBERT baseline after the shared edit."""

    # Prepare and run: create the original shared configuration.
    config = DistilBertTrainingConfig()

    # Check: BERT overrides must not leak into DistilBERT defaults.
    assert config.experiment_name == "DistilBERT Financial Sentiment"
    assert config.model_family == "DistilBERT"
    assert config.benchmark_role == "baseline_full_fine_tuning"


def test_bert_changes_only_approved_configuration_fields() -> None:
    """Keep all dataset, label, training, and evaluation rules comparable."""

    # Prepare: create both sides of the controlled benchmark.
    distilbert_config = DistilBertTrainingConfig()
    bert_config = BertTrainingConfig()

    # Run: inspect every field inherited from the verified shared config.
    shared_field_names = {
        field.name
        for field in fields(DistilBertTrainingConfig)
        if field.name not in BERT_OVERRIDE_FIELDS
    }

    # Check: every non-BERT-specific setting must remain identical.
    for field_name in sorted(shared_field_names):
        assert getattr(bert_config, field_name) == getattr(
            distilbert_config,
            field_name,
        ), f"Unexpected BERT override: {field_name}"


def test_bert_artifact_paths_are_exact_and_isolated() -> None:
    """Prevent BERT outputs from overwriting DistilBERT evidence."""

    # Prepare: create both experiment configurations.
    distilbert_config = DistilBertTrainingConfig()
    bert_config = BertTrainingConfig()

    # Check: every BERT destination must differ from DistilBERT.
    assert bert_config.checkpoint_dir != distilbert_config.checkpoint_dir
    assert bert_config.final_model_dir != distilbert_config.final_model_dir
    assert bert_config.metrics_file != distilbert_config.metrics_file
    assert bert_config.manifest_file != distilbert_config.manifest_file

    # Check exact directory and file names, not loose substring matches.
    assert bert_config.checkpoint_dir.parent.name == "bert_sentiment"
    assert bert_config.checkpoint_dir.name == "checkpoints"
    assert bert_config.final_model_dir.parent.name == "bert_sentiment"
    assert bert_config.final_model_dir.name == "final_model"
    assert bert_config.metrics_file.name == "bert_sentiment_metrics.json"
    assert (
        bert_config.manifest_file.name
        == "bert_sentiment_training_manifest.json"
    )


def test_unapproved_bert_model_is_rejected() -> None:
    """Stop an unreviewed checkpoint before model loading or training."""

    # Prepare: simulate an accidental model substitution.
    config = BertTrainingConfig(model_id="another/model")

    # Run and check: validation must fail before the shared engine runs.
    with pytest.raises(ValueError, match="BERT benchmark must use"):
        validate_bert_config(config)


def test_supplied_config_reaches_shared_engine_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify safe delegation while keeping production paths untouched."""

    captured: dict[str, Any] = {}

    def fake_shared_training(
        config: BertTrainingConfig,
    ) -> dict[str, Any]:
        """Record the input without loading data or training a model."""

        captured["config"] = config
        return {
            "status": "training_not_started",
            "model_id": config.model_id,
        }

    # Prepare: replace only the dependency imported by the BERT wrapper.
    monkeypatch.setattr(
        bert_module,
        "run_distilbert_training",
        fake_shared_training,
    )

    # Temporary paths guarantee that this unit test cannot touch real outputs.
    config = BertTrainingConfig(
        checkpoint_dir=tmp_path / "checkpoints",
        final_model_dir=tmp_path / "final_model",
        metrics_file=tmp_path / "metrics.json",
        manifest_file=tmp_path / "manifest.json",
    )

    # Run: call the public wrapper exactly as production code will.
    result = run_bert_training(config)

    # Check: the same validated object reached the shared engine.
    assert captured["config"] is config
    assert result == {
        "status": "training_not_started",
        "model_id": BERT_MODEL_ID,
    }


def test_default_config_is_created_when_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify the convenience path used by the full-training runner."""

    captured: dict[str, Any] = {}

    def fake_shared_training(
        config: BertTrainingConfig,
    ) -> dict[str, Any]:
        """Capture the generated config without performing real work."""

        captured["config"] = config
        return {"status": "training_not_started"}

    # Prepare: isolate the wrapper from the real training engine.
    monkeypatch.setattr(
        bert_module,
        "run_distilbert_training",
        fake_shared_training,
    )

    # Run: omit the optional configuration.
    result = run_bert_training()

    # Check: the wrapper created the approved default configuration.
    assert isinstance(captured["config"], BertTrainingConfig)
    assert captured["config"].model_id == BERT_MODEL_ID
    assert result == {"status": "training_not_started"}
