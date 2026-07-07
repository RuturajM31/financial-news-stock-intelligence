"""
Configure full BERT financial-sentiment training.

Purpose
-------
Select the approved full-BERT checkpoint and BERT-specific artifact paths,
then reuse the verified Transformer training engine.

Inputs
------
The shared engine loads the same verified Financial PhraseBank train,
validation, and untouched test splits used by DistilBERT.

Processing
----------
``BertTrainingConfig`` overrides only experiment identity, model identity,
and output destinations. Dataset, label, training, and evaluation settings
remain inherited from ``DistilBertTrainingConfig`` for a fair comparison.

Outputs
-------
A real training run writes BERT checkpoints, the final model and tokenizer,
metrics JSON, and a reproducibility manifest to BERT-specific locations.

Limitations
-----------
Importing this module does not download or train BERT. Full BERT is larger
than DistilBERT and is expected to require more CPU time and memory.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from financial_news_intelligence.models.distilbert_training import (
    DistilBertTrainingConfig,
    run_distilbert_training,
    validate_training_config,
)
from financial_news_intelligence.paths import MANIFESTS_DIR


# ============================================================
# 1. APPROVED MODEL
# ============================================================

BERT_MODEL_ID = "google-bert/bert-base-uncased"


# ============================================================
# 2. BERT-SPECIFIC OUTPUT LOCATIONS
# ============================================================

BERT_MODEL_ROOT = MANIFESTS_DIR.parent / "models" / "bert_sentiment"
BERT_REPORT_DIR = MANIFESTS_DIR.parents[1] / "reports" / "metrics"


# ============================================================
# 3. BERT TRAINING CONFIGURATION
# ============================================================

@dataclass
class BertTrainingConfig(DistilBertTrainingConfig):
    """
    Store the controlled full-BERT experiment settings.

    Only BERT identity and artifact paths differ from the verified
    DistilBERT configuration. All comparison-critical settings remain
    inherited from the shared configuration.
    """

    experiment_name: str = "BERT Financial Sentiment"
    model_family: str = "BERT"
    benchmark_role: str = "full_fine_tuning_comparison"
    model_id: str = BERT_MODEL_ID

    checkpoint_dir: Path = field(
        default_factory=lambda: BERT_MODEL_ROOT / "checkpoints"
    )
    final_model_dir: Path = field(
        default_factory=lambda: BERT_MODEL_ROOT / "final_model"
    )
    metrics_file: Path = field(
        default_factory=lambda: BERT_REPORT_DIR / "bert_sentiment_metrics.json"
    )
    manifest_file: Path = field(
        default_factory=lambda: (
            MANIFESTS_DIR / "bert_sentiment_training_manifest.json"
        )
    )
    run_name: str = "bert_financial_phrasebank_full"


# ============================================================
# 4. BERT-SPECIFIC VALIDATION
# ============================================================

def validate_bert_config(config: BertTrainingConfig) -> None:
    """
    Validate shared training rules and the approved BERT checkpoint.

    Parameters
    ----------
    config:
        BERT experiment settings supplied to the shared training engine.

    Raises
    ------
    ValueError:
        Raised before model loading when the model ID is not the approved
        BERT-base uncased checkpoint.
    """

    # Shared validation protects dataset files, hyperparameters, labels,
    # output locations, and the other rules already tested for DistilBERT.
    validate_training_config(config)

    if config.model_id != BERT_MODEL_ID:
        raise ValueError(
            "BERT benchmark must use "
            f"{BERT_MODEL_ID}."
        )


# ============================================================
# 5. COMPLETE BERT TRAINING ENTRY POINT
# ============================================================

def run_bert_training(
    config: BertTrainingConfig | None = None,
) -> dict[str, Any]:
    """
    Train, evaluate, save, and document full BERT.

    Data journey
    ------------
    1. Create the approved default BERT configuration when needed.
    2. Validate the shared protocol and BERT model identity.
    3. Pass the unchanged configuration to the verified shared engine.
    4. Return the generated reproducibility manifest.

    Importing this module does not call this function, so import checks are
    safe and do not start a download or training run.
    """

    if config is None:
        config = BertTrainingConfig()

    validate_bert_config(config)

    # The shared engine owns data loading, tokenization, class weighting,
    # training, evaluation, artifact saving, and manifest creation.
    return run_distilbert_training(config)
