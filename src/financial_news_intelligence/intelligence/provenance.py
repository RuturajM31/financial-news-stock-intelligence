"""Create source provenance, acceptance gates, and mandatory disclaimers.

Purpose
-------
Produce one machine-readable report describing the exact foundation manifest,
movement model, quality gates, explainability methods, historical-retrieval
rule, private-data boundary, assumptions, limitations, and deployment status.

Downstream use
--------------
FastAPI, Streamlit, documentation, and monitoring may surface this report later.
They must not claim that raw Tiingo values are publicly redistributable or that
historical associations prove causality.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping


class ProvenanceError(RuntimeError):
    """Raised when required provenance markers are missing or inconsistent."""


def build_provenance_report(
    foundation_manifest: Mapping[str, Any],
    movement_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a complete licence-safe provenance and disclaimer report."""

    if foundation_manifest.get("status") != "foundation_verified":
        raise ProvenanceError("Foundation manifest is not verified.")
    if movement_summary.get("status") != "trained_and_evaluated":
        raise ProvenanceError("Movement summary is not trained and evaluated.")
    quality_gates = movement_summary.get("quality_gates")
    if not isinstance(quality_gates, Mapping) or quality_gates.get("status") != "passed":
        raise ProvenanceError("Movement quality gates did not pass.")

    # Keep all provider, leakage, quality, and licensing statements in one
    # machine-readable object so later applications cannot silently omit them.
    return {
        "status": "provenance_verified",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "events": "SEC EDGAR official company disclosures",
            "prices": (
                "Tiingo EOD authenticated internal-use adjusted historical data"
            ),
            "sentiment": "existing local DistilBERT deployment champion",
            "movement_model": movement_summary.get("quality_champion"),
        },
        "data_grain": {
            "foundation_event": "one verified SEC event mapped to one session",
            "movement_model": "one canonical ticker and target-session date",
            "local_explanation": "one test record and raw feature",
            "historical_match": (
                "one test record and one earlier train/validation SEC event"
            ),
            "scenario": "one historical-audit record",
        },
        "selection_and_quality": {
            "candidate_selection_scope": (
                "purged expanding folds inside early development"
            ),
            "terminal_confirmation_scope": (
                "purged terminal development tournament over a fixed shortlist"
            ),
            "historical_audit_used_for_selection": False,
            "baseline_improvement_required": True,
            "all_three_prediction_classes_required": True,
            "quality_gates": dict(quality_gates),
        },
        "leakage_controls": [
            "foundation event time precedes target-session open",
            "market predictors are shifted one complete session",
            "chronological development and audit blocks use purged dates",
            "expanding development folds select the champion",
            "terminal development labels rank only the pre-registered shortlist",
            "historical audit is evaluated once and never selects a model",
            "global phrases and company context use train-validation only",
            "historical matches and scenarios use reference evidence only",
            "historical evidence is strictly earlier than each test session",
        ],
        "licence_boundary": {
            "tiingo_classification": "internal_use_only",
            "raw_tiingo_values_publicly_redistributable": False,
            "raw_tiingo_values_in_intelligence_outputs": False,
            "public_deployment_allowed_by_this_package": False,
        },
        "explainability_methods": {
            "global": "nonzero model-native importance",
            "local": "train-validation reference perturbation",
            "sentiment_phrases": (
                "train-validation class mean TF-IDF difference"
            ),
            "historical_similarity": (
                "same-ticker train-validation earlier-only TF-IDF cosine"
            ),
            "scenarios": "train-validation earlier empirical reactions",
        },
        "reproducibility": {
            "random_seed": movement_summary.get("random_seed"),
            "runtime_versions": movement_summary.get("runtime_versions"),
            "training_config": movement_summary.get("training_config"),
            "foundation_manifest_sha256": movement_summary.get(
                "foundation_manifest_sha256"
            ),
        },
        # These limitations are mandatory evidence, not optional UI copy.
        "limitations": [
            "SEC disclosures are not a complete representation of all market news.",
            "Historical associations and feature sensitivity are not causal proof.",
            "Model quality may degrade outside the 2015-2020 evidence window.",
            "Per-ticker sample sizes may be smaller than aggregate sample sizes.",
            "No result guarantees future price movement or investment performance.",
            "Public deployment requires a separate data-licensing review.",
        ],
        "mandatory_disclaimer": (
            "For educational and research use only. This output is not financial, "
            "investment, legal, or tax advice. Markets involve risk, and users must "
            "perform independent due diligence."
        ),
        "deployment_changed": False,
    }
