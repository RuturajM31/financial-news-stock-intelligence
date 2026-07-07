#!/usr/bin/env python3
"""Run movement modeling first and intelligence only after verification.

Purpose
-------
Provide one controlled command-line entry point for the combined strike. The
movement phase verifies the exact passed v8 foundation, creates purged splits,
ranks learned candidates with purged rolling validation, runs a frozen
shortlist tournament on terminal development data, and evaluates the known
historical audit once.
Hard quality gates and checksummed artifacts remain fail-closed. The
intelligence phase refuses to start until those artifacts are independently
recomputed and verified.

Inputs and outputs
------------------
The positional input is the project root. ``--replace-existing`` allows the
rollback-safe installer to replace controlled outputs after backing them up.
No API token is required because the verified foundation already exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_info

from financial_news_intelligence.models.movement_artifacts import (
    artifact_entry,
    load_json_object,
    protect_outputs,
    resolve_outputs,
    verify_inventory_entries,
    verify_manifest,
    write_csv,
    write_joblib,
    write_json,
)
from financial_news_intelligence.models.movement_dataset import (
    EXPECTED_FOUNDATION_MANIFEST_SHA256,
    LABEL_ORDER,
    assign_chronological_splits,
    build_model_table,
    feature_columns,
    filter_events_by_split,
    load_foundation_frames,
)
from financial_news_intelligence.models.movement_training import (
    TrainingConfig,
    classification_metrics,
    evaluate_quality_gates,
    global_importance,
    per_ticker_metrics,
    runtime_versions,
    train_and_evaluate,
    _predict_output,
)

PACKAGE_CONTRACT_VERSION = "8.3.0"
MINIMUM_MATCHES_PER_RECORD = 3
MINIMUM_GLOBAL_DRIVERS = 3
MINIMUM_LOCAL_DRIVERS_PER_RECORD = 5
FORBIDDEN_PUBLIC_COLUMNS = {
    "open",
    "high",
    "low",
    "close",
    "volume",
    "previous_close",
    "target_open",
    "target_close",
    "target_volume",
}


class CombinedRunError(RuntimeError):
    """Raised when phase ordering or saved semantic evidence is incomplete."""


def _allowed_decision_global_offsets(
    config: TrainingConfig,
) -> set[float]:
    """Return the frozen global offset grid from a typed training config.

    The runner reconstructs ``TrainingConfig`` from saved JSON before this
    check. Accessing the dataclass through attributes preserves the declared
    type contract and prevents dictionary-style subscripting errors during
    independent artifact verification.
    """

    # Normalize every configured value to float so persisted JSON numbers and
    # in-memory dataclass values are compared with one explicit representation.
    return {float(value) for value in config.decision_global_offsets}


def _validate_global_importance_records(
    records: list[dict[str, Any]],
) -> pd.DataFrame:
    """Validate movement-native importance without starting intelligence.

    Inputs are validation-only feature-importance records saved inside the
    champion bundle. The check mirrors the minimum schema and nonzero-driver
    requirements needed by the later explainability phase, but it deliberately
    avoids importing intelligence modules before movement verification passes.

    The returned frame is a validated copy used only for verification. The
    original records remain unchanged for the downstream explainability phase.
    """

    required_columns = {"feature", "importance", "method"}
    if not records or not all(isinstance(row, Mapping) for row in records):
        raise CombinedRunError("Validation-only global importance is invalid.")

    frame = pd.DataFrame(records).copy()
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        raise CombinedRunError(
            f"Validation-only global importance is missing columns: {missing}"
        )

    # Numeric conversion catches strings, missing values, and infinities before
    # explainability can consume the saved model-native evidence.
    frame["importance"] = pd.to_numeric(frame["importance"], errors="coerce")
    if frame["importance"].isna().any():
        raise CombinedRunError("Validation-only global importance is nonnumeric.")
    if not np.isfinite(frame["importance"].to_numpy()).all():
        raise CombinedRunError("Validation-only global importance is nonfinite.")
    if frame["importance"].lt(0).any():
        raise CombinedRunError("Validation-only global importance is negative.")

    nonzero_count = int(frame["importance"].gt(0).sum())
    if nonzero_count < MINIMUM_GLOBAL_DRIVERS:
        raise CombinedRunError(
            "Validation-only global importance has fewer than "
            f"{MINIMUM_GLOBAL_DRIVERS} nonzero drivers."
        )
    return frame.reset_index(drop=True)


def _openmp_runtime_report() -> dict[str, Any]:
    """Return loaded native thread pools and normalized OpenMP families.

    The report contains library paths and runtime families only. It contains no
    project data, credentials, labels, or market values.
    """

    pools = threadpool_info()
    openmp_rows: list[dict[str, Any]] = []
    families: set[str] = set()
    for row in pools:
        if str(row.get("user_api", "")).lower() != "openmp":
            continue
        filepath = str(row.get("filepath") or "")
        lowered = filepath.lower()
        if "libiomp" in lowered:
            family = "intel"
        elif "libomp" in lowered:
            family = "llvm"
        elif "libgomp" in lowered:
            family = "gnu"
        else:
            family = str(row.get("prefix") or "unknown").lower()
        families.add(family)
        openmp_rows.append(
            {
                "family": family,
                "prefix": row.get("prefix"),
                "filepath": filepath,
                "version": row.get("version"),
                "num_threads": row.get("num_threads"),
            }
        )
    return {
        "status": "compatible" if len(families) <= 1 else "conflict",
        "openmp_families": sorted(families),
        "openmp_libraries": openmp_rows,
        "threadpool_count": len(pools),
    }


def _assert_compatible_openmp_runtime() -> dict[str, Any]:
    """Fail before training when incompatible OpenMP runtimes are loaded."""

    report = _openmp_runtime_report()
    if report["status"] != "compatible":
        raise CombinedRunError(
            "Incompatible OpenMP runtimes are loaded together: "
            + ", ".join(report["openmp_families"])
        )
    return report

def _atomic_external_bytes(file_path: Path, content: bytes) -> None:
    """Write owner-only diagnostic evidence outside the project atomically."""

    file_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = file_path.with_name(
        f"{file_path.name}.strike_tmp.{os.getpid()}"
    )
    try:
        temporary.write_bytes(content)
        os.chmod(temporary, 0o600)
        temporary.replace(file_path)
        os.chmod(file_path, 0o600)
    finally:
        if temporary.exists() and temporary.is_file() and not temporary.is_symlink():
            temporary.unlink()


def _diagnostic_checksum(file_path: Path) -> str:
    """Return one diagnostic file checksum without loading it into memory."""

    digest = hashlib.sha256()
    with file_path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()



def write_external_diagnostics(
    diagnostic_dir: Path | None,
    result: Mapping[str, Any],
    split_report: Mapping[str, Any],
) -> Path | None:
    """Persist complete licence-safe diagnostics before project rollback.

    Inputs are the in-memory training result and split metadata. Outputs are
    owner-only CSV and JSON files outside the project transaction. Candidate,
    fold, convergence, confirmation, and historical-audit evidence are written
    without raw Tiingo price columns.
    """

    if diagnostic_dir is None:
        return None
    root = diagnostic_dir.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise CombinedRunError(f"Unsafe diagnostic directory: {root}")
    os.chmod(root, 0o700)

    candidate_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    convergence_rows: list[dict[str, Any]] = []
    for row in result.get("candidate_results", []):
        metrics = row.get("metrics") or {}
        candidate_rows.append(
            {
                "model_name": row.get("model_name"),
                "model_family": row.get("model_family"),
                "status": row.get("status"),
                "convergence_status": row.get("convergence_status"),
                "parameters_json": json.dumps(
                    row.get("parameters", {}),
                    sort_keys=True,
                ),
                "validation_accuracy": metrics.get("accuracy"),
                "validation_macro_f1": metrics.get("macro_f1"),
                "validation_weighted_f1": metrics.get("weighted_f1"),
                "validation_predicted_class_count": metrics.get(
                    "predicted_class_count"
                ),
                "minimum_fold_macro_f1": row.get(
                    "minimum_fold_macro_f1"
                ),
                "minimum_fold_weighted_f1": row.get(
                    "minimum_fold_weighted_f1"
                ),
                "mean_fold_macro_f1": row.get("mean_fold_macro_f1"),
                "median_fold_macro_f1": row.get(
                    "median_fold_macro_f1"
                ),
                "mean_fold_weighted_f1": row.get(
                    "mean_fold_weighted_f1"
                ),
                "median_fold_weighted_f1": row.get(
                    "median_fold_weighted_f1"
                ),
                "latest_fold_macro_f1": row.get(
                    "latest_fold_macro_f1"
                ),
                "latest_fold_weighted_f1": row.get(
                    "latest_fold_weighted_f1"
                ),
                "recency_weighted_fold_macro_f1": row.get(
                    "recency_weighted_fold_macro_f1"
                ),
                "recency_weighted_fold_weighted_f1": row.get(
                    "recency_weighted_fold_weighted_f1"
                ),
                "macro_f1_std": row.get("macro_f1_std"),
                "weighted_f1_std": row.get("weighted_f1_std"),
                "minimum_fold_predicted_class_count": row.get(
                    "minimum_fold_predicted_class_count"
                ),
                "training_seconds": row.get("training_seconds"),
                "latency_ms_per_record": row.get(
                    "latency_ms_per_record"
                ),
                "decision_policy_json": json.dumps(
                    row.get("decision_policy", {}),
                    sort_keys=True,
                ),
                "mean_max_probability": (
                    row.get("prediction_diagnostics") or {}
                ).get("mean_max_probability"),
                "mean_top_two_margin": (
                    row.get("prediction_diagnostics") or {}
                ).get("mean_top_two_margin"),
                "error": row.get("error"),
            }
        )
        for fold in row.get("fold_metrics") or []:
            fold_metrics = fold.get("metrics") or {}
            fold_rows.append(
                {
                    "model_name": row.get("model_name"),
                    "model_family": row.get("model_family"),
                    "fold_name": fold.get("fold_name"),
                    "train_start_date": fold.get("train_start_date"),
                    "train_end_date": fold.get("train_end_date"),
                    "validation_start_date": fold.get(
                        "validation_start_date"
                    ),
                    "validation_end_date": fold.get(
                        "validation_end_date"
                    ),
                    "validation_accuracy": fold_metrics.get("accuracy"),
                    "validation_macro_f1": fold_metrics.get("macro_f1"),
                    "validation_weighted_f1": fold_metrics.get(
                        "weighted_f1"
                    ),
                    "validation_predicted_class_count": fold_metrics.get(
                        "predicted_class_count"
                    ),
                    "convergence_status": fold.get("convergence_status"),
                    "training_seconds": fold.get("training_seconds"),
                    "latency_ms_per_record": fold.get(
                        "latency_ms_per_record"
                    ),
                }
            )
        convergence_rows.extend(row.get("convergence_diagnostics") or [])

    convergence_rows.extend(
        result.get("development_confirmation_convergence") or []
    )
    convergence_rows.extend(result.get("champion_refit_convergence") or [])

    paths = {
        "candidates": root / "movement_candidate_validation.csv",
        "folds": root / "movement_candidate_fold_metrics.csv",
        "convergence": root / "movement_convergence_diagnostics.csv",
        "policy_calibration": root / "movement_policy_calibration.csv",
        "policy_oof_predictions": (
            root / "movement_policy_oof_predictions.csv"
        ),
        "confirmation_metrics": (
            root / "movement_development_confirmation_metrics.csv"
        ),
        "confirmation_predictions": (
            root / "movement_development_confirmation_predictions.csv"
        ),
        "confirmation_tournament": (
            root / "movement_development_confirmation_tournament.csv"
        ),
        "summary": root / "movement_diagnostic_summary.json",
        "predictions": root / "movement_failed_test_predictions.csv",
        "metrics": root / "movement_test_metrics.csv",
        "class_metrics": root / "movement_test_class_metrics.csv",
        "confusion": root / "movement_test_confusion_matrix.csv",
        "ticker_metrics": root / "movement_test_ticker_metrics.csv",
        "openmp_trace": root / "movement_openmp_import_trace.json",
        "manifest": root / "movement_diagnostic_manifest.json",
    }
    _atomic_external_bytes(
        paths["candidates"],
        pd.DataFrame(candidate_rows).to_csv(index=False).encode("utf-8"),
    )
    _atomic_external_bytes(
        paths["folds"],
        pd.DataFrame(fold_rows).to_csv(index=False).encode("utf-8"),
    )
    _atomic_external_bytes(
        paths["convergence"],
        pd.DataFrame(convergence_rows).to_csv(index=False).encode("utf-8"),
    )

    policy_rows = result.get("decision_policy_candidates")
    if isinstance(policy_rows, list):
        _atomic_external_bytes(
            paths["policy_calibration"],
            pd.DataFrame(policy_rows).to_csv(index=False).encode("utf-8"),
        )

    policy_oof = result.get("decision_policy_oof_predictions")
    if isinstance(policy_oof, pd.DataFrame):
        forbidden = FORBIDDEN_PUBLIC_COLUMNS & set(policy_oof.columns)
        if forbidden:
            raise CombinedRunError(
                "Policy OOF diagnostics contain restricted columns: "
                f"{sorted(forbidden)}"
            )
        _atomic_external_bytes(
            paths["policy_oof_predictions"],
            policy_oof.to_csv(index=False).encode("utf-8"),
        )

    summary = {
        "status": result.get("status"),
        "champion_name": result.get("champion_name"),
        "evaluation_protocol": result.get("evaluation_protocol"),
        "historical_audit_pristine": result.get(
            "historical_audit_pristine"
        ),
        "historical_audit_used_for_selection": result.get(
            "historical_audit_used_for_selection"
        ),
        "rolling_fold_reports": result.get("rolling_fold_reports"),
        "validation_gates": result.get("validation_gates"),
        "development_confirmation_split": result.get(
            "development_confirmation_split"
        ),
        "development_confirmation_metrics": result.get(
            "development_confirmation_metrics"
        ),
        "baseline_development_confirmation_metrics": result.get(
            "baseline_development_confirmation_metrics"
        ),
        "development_confirmation_gates": result.get(
            "development_confirmation_gates"
        ),
        "development_confirmation_evaluation_count": result.get(
            "development_confirmation_evaluation_count"
        ),
        "development_confirmation_pristine": result.get(
            "development_confirmation_pristine"
        ),
        "development_confirmation_known_from_prior_run": result.get(
            "development_confirmation_known_from_prior_run"
        ),
        "development_confirmation_used_for_candidate_selection": (
            result.get(
                "development_confirmation_used_for_candidate_selection"
            )
        ),
        "development_confirmation_tournament": result.get(
            "development_confirmation_tournament"
        ),
        "quality_gates": result.get("quality_gates"),
        "test_metrics": result.get("test_metrics"),
        "historical_audit_metrics": result.get(
            "historical_audit_metrics"
        ),
        "baseline_test_metrics": result.get("baseline_test_metrics"),
        "baseline_historical_audit_metrics": result.get(
            "baseline_historical_audit_metrics"
        ),
        "per_ticker_test_metrics": result.get("per_ticker_test_metrics"),
        "per_ticker_historical_audit_metrics": result.get(
            "per_ticker_historical_audit_metrics"
        ),
        "split_report": split_report,
        "test_used_for_selection": result.get("test_used_for_selection"),
        "test_evaluation_count": result.get("test_evaluation_count"),
        "historical_audit_evaluation_count": result.get(
            "historical_audit_evaluation_count"
        ),
        "training_config": result.get("training_config"),
        "runtime_versions": result.get("runtime_versions"),
        "openmp_runtime_report": result.get("openmp_runtime_report"),
        "decision_policy": result.get("decision_policy"),
        "decision_policy_calibration": result.get(
            "decision_policy_calibration"
        ),
        "raw_tiingo_values_exported": False,
    }
    _atomic_external_bytes(
        paths["summary"],
        json.dumps(summary, indent=2, sort_keys=True).encode("utf-8"),
    )

    written_paths = [
        paths["candidates"],
        paths["folds"],
        paths["convergence"],
        paths["summary"],
    ]
    if isinstance(policy_rows, list):
        written_paths.append(paths["policy_calibration"])
    if isinstance(policy_oof, pd.DataFrame):
        written_paths.append(paths["policy_oof_predictions"])

    confirmation_metrics = result.get("development_confirmation_metrics")
    if isinstance(confirmation_metrics, Mapping):
        confirmation_row = {
            "accuracy": confirmation_metrics.get("accuracy"),
            "macro_precision": confirmation_metrics.get("macro_precision"),
            "macro_recall": confirmation_metrics.get("macro_recall"),
            "macro_f1": confirmation_metrics.get("macro_f1"),
            "weighted_f1": confirmation_metrics.get("weighted_f1"),
            "record_count": confirmation_metrics.get("record_count"),
            "predicted_class_count": confirmation_metrics.get(
                "predicted_class_count"
            ),
        }
        _atomic_external_bytes(
            paths["confirmation_metrics"],
            pd.DataFrame([confirmation_row]).to_csv(index=False).encode(
                "utf-8"
            ),
        )
        written_paths.append(paths["confirmation_metrics"])

    confirmation_predictions = result.get(
        "development_confirmation_predictions"
    )
    if isinstance(confirmation_predictions, pd.DataFrame):
        forbidden = FORBIDDEN_PUBLIC_COLUMNS & set(
            confirmation_predictions.columns
        )
        if forbidden:
            raise CombinedRunError(
                "Confirmation diagnostics contain restricted columns: "
                f"{sorted(forbidden)}"
            )
        _atomic_external_bytes(
            paths["confirmation_predictions"],
            confirmation_predictions.to_csv(index=False).encode("utf-8"),
        )
        written_paths.append(paths["confirmation_predictions"])

    tournament = result.get("development_confirmation_tournament")
    if isinstance(tournament, list):
        tournament_rows: list[dict[str, Any]] = []
        for row in tournament:
            metrics = row.get("metrics") or {}
            gates = row.get("confirmation_gates") or {}
            tournament_rows.append(
                {
                    "model_name": row.get("model_name"),
                    "model_family": row.get("model_family"),
                    "status": row.get("status"),
                    "confirmation_accuracy": metrics.get("accuracy"),
                    "confirmation_macro_f1": metrics.get("macro_f1"),
                    "confirmation_weighted_f1": metrics.get(
                        "weighted_f1"
                    ),
                    "confirmation_predicted_class_count": metrics.get(
                        "predicted_class_count"
                    ),
                    "gate_status": gates.get("status"),
                    "gate_failures_json": json.dumps(
                        gates.get("failures") or [],
                        sort_keys=True,
                    ),
                    "minimum_fold_macro_f1": row.get(
                        "minimum_fold_macro_f1"
                    ),
                    "minimum_fold_weighted_f1": row.get(
                        "minimum_fold_weighted_f1"
                    ),
                    "latest_fold_macro_f1": row.get(
                        "latest_fold_macro_f1"
                    ),
                    "latest_fold_weighted_f1": row.get(
                        "latest_fold_weighted_f1"
                    ),
                    "recency_weighted_fold_macro_f1": row.get(
                        "recency_weighted_fold_macro_f1"
                    ),
                    "recency_weighted_fold_weighted_f1": row.get(
                        "recency_weighted_fold_weighted_f1"
                    ),
                    "fit_seconds": row.get("fit_seconds"),
                    "latency_ms_per_record": row.get(
                        "latency_ms_per_record"
                    ),
                    "decision_policy_json": json.dumps(
                        row.get("decision_policy") or {},
                        sort_keys=True,
                    ),
                    "error": row.get("error"),
                }
            )
        _atomic_external_bytes(
            paths["confirmation_tournament"],
            pd.DataFrame(tournament_rows).to_csv(index=False).encode(
                "utf-8"
            ),
        )
        written_paths.append(paths["confirmation_tournament"])

    predictions = result.get("test_predictions")
    if isinstance(predictions, pd.DataFrame):
        forbidden = FORBIDDEN_PUBLIC_COLUMNS & set(predictions.columns)
        if forbidden:
            raise CombinedRunError(
                "Diagnostic predictions contain restricted columns: "
                f"{sorted(forbidden)}"
            )
        _atomic_external_bytes(
            paths["predictions"],
            predictions.to_csv(index=False).encode("utf-8"),
        )
        written_paths.append(paths["predictions"])

    test_metrics = result.get("test_metrics")
    if isinstance(test_metrics, Mapping):
        aggregate = pd.DataFrame(
            [
                {
                    "accuracy": test_metrics.get("accuracy"),
                    "macro_precision": test_metrics.get("macro_precision"),
                    "macro_recall": test_metrics.get("macro_recall"),
                    "macro_f1": test_metrics.get("macro_f1"),
                    "weighted_f1": test_metrics.get("weighted_f1"),
                    "record_count": test_metrics.get("record_count"),
                    "predicted_class_count": test_metrics.get(
                        "predicted_class_count"
                    ),
                }
            ]
        )
        _atomic_external_bytes(
            paths["metrics"],
            aggregate.to_csv(index=False).encode("utf-8"),
        )
        class_rows = []
        for label in LABEL_ORDER:
            evidence = (test_metrics.get("per_class") or {}).get(label, {})
            class_rows.append(
                {
                    "movement_label": label,
                    "precision": evidence.get("precision"),
                    "recall": evidence.get("recall"),
                    "f1": evidence.get("f1"),
                    "support": evidence.get("support"),
                    "predicted_support": evidence.get(
                        "predicted_support"
                    ),
                }
            )
        _atomic_external_bytes(
            paths["class_metrics"],
            pd.DataFrame(class_rows).to_csv(index=False).encode("utf-8"),
        )
        written_paths.extend([paths["metrics"], paths["class_metrics"]])
        recorded_matrix = test_metrics.get("confusion_matrix")
        if recorded_matrix is not None:
            matrix = np.asarray(recorded_matrix, dtype=int)
            if matrix.shape != (len(LABEL_ORDER), len(LABEL_ORDER)):
                raise CombinedRunError(
                    "Diagnostic confusion matrix has an invalid shape."
                )
            confusion_rows = []
            for actual_index, actual_label in enumerate(LABEL_ORDER):
                for predicted_index, predicted_label in enumerate(LABEL_ORDER):
                    confusion_rows.append(
                        {
                            "actual_movement": actual_label,
                            "predicted_movement": predicted_label,
                            "record_count": int(
                                matrix[actual_index, predicted_index]
                            ),
                        }
                    )
            _atomic_external_bytes(
                paths["confusion"],
                pd.DataFrame(confusion_rows).to_csv(index=False).encode(
                    "utf-8"
                ),
            )
            written_paths.append(paths["confusion"])

    ticker_rows = result.get("per_ticker_test_metrics")
    if isinstance(ticker_rows, list):
        normalized_tickers = []
        for row in ticker_rows:
            normalized_tickers.append(
                {
                    "ticker": row.get("ticker"),
                    "record_count": row.get("record_count"),
                    "accuracy": row.get("accuracy"),
                    "macro_f1": row.get("macro_f1"),
                    "weighted_f1": row.get("weighted_f1"),
                    "predicted_class_count": row.get(
                        "predicted_class_count"
                    ),
                    "actual_class_support_json": json.dumps(
                        row.get("actual_class_support", {}),
                        sort_keys=True,
                    ),
                    "predicted_class_support_json": json.dumps(
                        row.get("predicted_class_support", {}),
                        sort_keys=True,
                    ),
                }
            )
        _atomic_external_bytes(
            paths["ticker_metrics"],
            pd.DataFrame(normalized_tickers).to_csv(index=False).encode(
                "utf-8"
            ),
        )
        written_paths.append(paths["ticker_metrics"])

    # The native-runtime preflight runs before model training. When its trace
    # exists in the same owner-only directory, include it in the diagnostic
    # inventory so one manifest covers the complete controlled run evidence.
    if paths["openmp_trace"].is_file() and not paths["openmp_trace"].is_symlink():
        os.chmod(paths["openmp_trace"], 0o600)
        written_paths.append(paths["openmp_trace"])

    manifest = {
        "status": "diagnostic_evidence_written",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": [
            {
                "path": file_path.name,
                "sha256": _diagnostic_checksum(file_path),
                "size_bytes": file_path.stat().st_size,
                "mode": oct(file_path.stat().st_mode & 0o777),
            }
            for file_path in written_paths
        ],
        "restricted_raw_price_values_included": False,
    }
    _atomic_external_bytes(
        paths["manifest"],
        json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8"),
    )
    return paths["summary"]

def _close(left: float, right: float, tolerance: float = 1e-9) -> bool:
    """Return whether two finite numeric values agree within tolerance."""

    return bool(np.isclose(float(left), float(right), atol=tolerance, rtol=0.0))


def _assert_metric_mapping(
    recorded: Mapping[str, Any],
    recomputed: Mapping[str, Any],
    prefix: str,
) -> None:
    """Recursively compare saved and independently recomputed metric evidence."""

    if set(recorded) != set(recomputed):
        raise CombinedRunError(f"{prefix} metric keys changed.")
    for key, recorded_value in recorded.items():
        recomputed_value = recomputed[key]
        location = f"{prefix}.{key}"
        if isinstance(recorded_value, Mapping):
            if not isinstance(recomputed_value, Mapping):
                raise CombinedRunError(f"{location} type changed.")
            _assert_metric_mapping(recorded_value, recomputed_value, location)
        elif isinstance(recorded_value, list):
            if recorded_value != recomputed_value:
                raise CombinedRunError(f"{location} changed.")
        elif isinstance(recorded_value, (int, float)):
            if not _close(recorded_value, recomputed_value):
                raise CombinedRunError(f"{location} changed.")
        elif recorded_value != recomputed_value:
            raise CombinedRunError(f"{location} changed.")


def _recompute_split_report(model_table: pd.DataFrame) -> dict[str, Any]:
    """Rebuild split counts and date boundaries from the saved model table."""

    required = {"ticker", "target_session_date", "movement_label", "split"}
    missing = sorted(required - set(model_table.columns))
    if missing:
        raise CombinedRunError(f"Model table is missing columns: {missing}")
    frame = model_table.copy()
    frame["target_session_date"] = pd.to_datetime(
        frame["target_session_date"],
        errors="coerce",
    ).dt.normalize()
    if frame["target_session_date"].isna().any():
        raise CombinedRunError("Model table contains invalid target dates.")
    if set(frame["split"]) != {"train", "validation", "test"}:
        raise CombinedRunError("Model table does not contain exactly three splits.")

    report: dict[str, Any] = {}
    for split_name in ("train", "validation", "test"):
        split_rows = frame[frame["split"] == split_name]
        report[split_name] = {
            "start_date": str(split_rows["target_session_date"].min().date()),
            "end_date": str(split_rows["target_session_date"].max().date()),
            "unique_dates": int(split_rows["target_session_date"].nunique()),
            "rows": int(len(split_rows)),
            "class_counts": {
                label: int((split_rows["movement_label"] == label).sum())
                for label in LABEL_ORDER
            },
            "tickers": sorted(split_rows["ticker"].astype(str).unique()),
        }

    # Purged rows are intentionally not persisted in the model table. The saved
    # report remains the evidence for their count and exact dates.
    train_end = pd.Timestamp(report["train"]["end_date"])
    validation_start = pd.Timestamp(report["validation"]["start_date"])
    validation_end = pd.Timestamp(report["validation"]["end_date"])
    test_start = pd.Timestamp(report["test"]["start_date"])
    if not (train_end < validation_start < validation_end < test_start):
        raise CombinedRunError("Saved chronological split boundaries overlap.")
    return report


def _validate_minimal_predictions(predictions: pd.DataFrame) -> None:
    """Require one licence-safe prediction row per test ticker-session."""

    expected_columns = {
        "record_id",
        "ticker",
        "target_session_date",
        "actual_movement",
        "predicted_movement",
        "prob_down",
        "prob_flat",
        "prob_up",
    }
    if set(predictions.columns) != expected_columns:
        raise CombinedRunError("Test prediction schema changed.")
    if predictions.empty or predictions["record_id"].duplicated().any():
        raise CombinedRunError("Test prediction identifiers are empty or duplicated.")
    if predictions.duplicated(["ticker", "target_session_date"]).any():
        raise CombinedRunError("Test prediction grain is not ticker-session unique.")
    if not set(predictions["actual_movement"]).issubset(LABEL_ORDER):
        raise CombinedRunError("Actual test labels contain an unknown class.")
    if not set(predictions["predicted_movement"]).issubset(LABEL_ORDER):
        raise CombinedRunError("Predicted test labels contain an unknown class.")
    probabilities = predictions[["prob_down", "prob_flat", "prob_up"]].apply(
        pd.to_numeric,
        errors="coerce",
    )
    if probabilities.isna().any().any() or (probabilities < 0).any().any():
        raise CombinedRunError("Saved test probabilities are invalid.")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise CombinedRunError("Saved test probabilities do not sum to one.")
    probability_labels = np.asarray(LABEL_ORDER, dtype=object)[
        probabilities.to_numpy().argmax(axis=1)
    ]
    if not np.array_equal(
        probability_labels,
        predictions["predicted_movement"].to_numpy(),
    ):
        raise CombinedRunError(
            "Saved labels do not match the highest saved probability."
        )


def verify_movement_phase(project_root: Path) -> dict[str, Any]:
    """Independently verify movement artifacts before intelligence starts."""

    root = project_root.expanduser().resolve()
    outputs = resolve_outputs(root)
    summary = load_json_object(outputs["movement_metrics"], "movement metrics")
    if summary.get("status") != "trained_and_evaluated":
        raise CombinedRunError("Movement metrics are not complete.")
    if summary.get("test_used_for_selection") is not False:
        raise CombinedRunError(
            "Historical-audit data was used for model selection."
        )
    if summary.get("historical_audit_used_for_selection") is not False:
        raise CombinedRunError(
            "Historical-audit selection marker changed."
        )
    confirmation_count = int(
        summary.get("development_confirmation_evaluation_count", 0)
    )
    training_config = summary.get("training_config")
    if not isinstance(training_config, Mapping):
        raise CombinedRunError("Training configuration evidence is missing.")
    shortlist_size = int(training_config.get("terminal_shortlist_size", 0))
    if confirmation_count < 1 or confirmation_count > shortlist_size:
        raise CombinedRunError(
            "Development-confirmation tournament count is invalid."
        )
    confirmation_gates = summary.get("development_confirmation_gates")
    if not isinstance(confirmation_gates, Mapping):
        raise CombinedRunError(
            "Development-confirmation gate evidence is missing."
        )
    if confirmation_gates.get("status") != "passed":
        raise CombinedRunError(
            "Development-confirmation gates are not passed."
        )
    if confirmation_gates.get("used_for_candidate_selection") is not True:
        raise CombinedRunError(
            "Terminal development selection marker changed."
        )
    if (
        summary.get(
            "development_confirmation_used_for_candidate_selection"
        )
        is not True
    ):
        raise CombinedRunError(
            "Terminal development tournament provenance is missing."
        )
    if int(summary.get("test_evaluation_count", 0)) != 1:
        raise CombinedRunError(
            "Historical-audit evaluation count is not exactly one."
        )
    if int(summary.get("historical_audit_evaluation_count", 0)) != 1:
        raise CombinedRunError(
            "Historical-audit marker is not exactly one."
        )
    if summary.get("historical_audit_pristine") is not False:
        raise CombinedRunError(
            "Historical audit must be disclosed as previously inspected."
        )
    if summary.get("development_confirmation_pristine") is not False:
        raise CombinedRunError(
            "Development confirmation must be disclosed as known from v6."
        )
    if summary.get("development_confirmation_known_from_prior_run") is not True:
        raise CombinedRunError(
            "Prior-run development-confirmation provenance is missing."
        )
    if summary.get("evaluation_protocol") != (
        "purged_four_fold_recency_ranking_plus_oof_policy_"
        "calibration_plus_terminal_development_tournament_"
        "plus_known_historical_audit"
    ):
        raise CombinedRunError("Movement evaluation protocol changed.")
    if summary.get("quality_champion") == "prior_baseline":
        raise CombinedRunError("The saved champion is the prior baseline.")
    if training_config.get("logistic_solver") != "lbfgs":
        raise CombinedRunError("The non-Liblinear logistic solver changed.")
    if training_config.get("convergence_warnings_are_errors") is not True:
        raise CombinedRunError("Convergence warnings are not fail-closed.")
    if training_config.get("enable_ticker_offsets") is not False:
        raise CombinedRunError("Ticker-specific decision offsets are enabled.")
    expected_global_offsets = [
        -0.3,
        -0.2,
        -0.1,
        0.0,
        0.1,
        0.2,
        0.3,
    ]
    if list(training_config.get("decision_global_offsets", [])) != (
        expected_global_offsets
    ):
        raise CombinedRunError("Global decision-offset grid changed.")
    if list(training_config.get("decision_ticker_offsets", [])) != [0.0]:
        raise CombinedRunError("Ticker decision-offset search changed.")
    if int(training_config.get("rolling_validation_folds", 0)) != 4:
        raise CombinedRunError("Four-fold development selection changed.")
    if int(training_config.get("minimum_confirmation_dates", 0)) != 60:
        raise CombinedRunError("Terminal confirmation minimum changed.")
    if shortlist_size != 5:
        raise CombinedRunError("Terminal shortlist size changed.")
    if int(
        training_config.get("terminal_shortlist_max_per_family", 0)
    ) != 2:
        raise CombinedRunError("Terminal family-diversity limit changed.")
    if float(
        training_config.get("rolling_recency_weight_power", 0.0)
    ) != 1.0:
        raise CombinedRunError("Rolling recency weighting changed.")

    tournament = summary.get("development_confirmation_tournament")
    if not isinstance(tournament, list) or len(tournament) != (
        confirmation_count
    ):
        raise CombinedRunError(
            "Terminal development tournament evidence is incomplete."
        )

    policy_report = summary.get("decision_policy_calibration")
    if not isinstance(policy_report, Mapping):
        raise CombinedRunError("OOF policy-calibration evidence is missing.")
    if policy_report.get("fit_split") != "selection_oof":
        raise CombinedRunError("Policy calibration did not use selection OOF.")
    if policy_report.get("historical_audit_used_for_selection") is not False:
        raise CombinedRunError("Historical audit influenced policy calibration.")
    if policy_report.get("status") not in {
        "adjusted",
        "identity_fallback",
    }:
        raise CombinedRunError("Policy calibration status is invalid.")
    runtime_report = summary.get("openmp_runtime_report")
    if not isinstance(runtime_report, Mapping):
        raise CombinedRunError("OpenMP runtime evidence is missing.")
    if runtime_report.get("status") != "compatible":
        raise CombinedRunError("OpenMP runtime evidence reports a conflict.")
    for row in summary.get("candidate_results", []):
        if row.get("model_family") == "calibrated_linear_svc":
            raise CombinedRunError("Removed LinearSVC family was persisted.")
        if (
            row.get("status") == "passed"
            and row.get("convergence_status") != "converged"
        ):
            raise CombinedRunError(
                "A passed candidate lacks convergence evidence."
            )
    if (
        summary.get("foundation_manifest_sha256")
        != EXPECTED_FOUNDATION_MANIFEST_SHA256
    ):
        raise CombinedRunError("Movement metrics reference a different foundation.")

    # Verify the three movement artifacts before loading model or CSV content.
    movement_entries = summary.get("movement_artifacts")
    if not isinstance(movement_entries, list):
        raise CombinedRunError("Movement artifact inventory is missing.")
    expected_paths = {
        outputs[name].relative_to(root).as_posix()
        for name in ("model", "model_table", "test_predictions")
    }
    verify_inventory_entries(root, movement_entries, expected_paths)

    model_table = pd.read_csv(outputs["model_table"])
    predictions = pd.read_csv(outputs["test_predictions"])
    _validate_minimal_predictions(predictions)
    model_table["target_session_date"] = pd.to_datetime(
        model_table["target_session_date"],
        errors="coerce",
    ).dt.normalize()
    predictions["target_session_date"] = pd.to_datetime(
        predictions["target_session_date"],
        errors="coerce",
    ).dt.normalize()

    recomputed_report = _recompute_split_report(model_table)
    recorded_report = summary.get("split_report")
    if not isinstance(recorded_report, Mapping):
        raise CombinedRunError("Saved split report is missing.")
    for split_name in ("train", "validation", "test"):
        if recorded_report.get(split_name) != recomputed_report[split_name]:
            raise CombinedRunError(f"Saved {split_name} split report changed.")
    if int(recorded_report.get("purged", {}).get("unique_dates", 0)) < 2:
        raise CombinedRunError("Saved split report has insufficient purge dates.")

    # Every prediction key and actual label must match the persisted test block.
    expected_test = model_table[model_table["split"] == "test"]
    expected_test = expected_test[
        ["ticker", "target_session_date", "movement_label"]
    ].rename(columns={"movement_label": "actual_movement"})
    joined = predictions.merge(
        expected_test,
        on=["ticker", "target_session_date"],
        how="outer",
        suffixes=("_saved", "_table"),
        indicator=True,
        validate="one_to_one",
    )
    if not joined["_merge"].eq("both").all():
        raise CombinedRunError("Prediction rows do not match the test split.")
    if not joined["actual_movement_saved"].eq(
        joined["actual_movement_table"]
    ).all():
        raise CombinedRunError("Saved test labels differ from the model table.")

    # Recompute metrics directly from the saved prediction rows instead of
    # trusting the JSON summary written by the training phase.
    recomputed_metrics = classification_metrics(
        predictions["actual_movement"],
        predictions["predicted_movement"].to_numpy(),
    )
    recorded_metrics = summary.get("test_metrics")
    if not isinstance(recorded_metrics, Mapping):
        raise CombinedRunError("Saved test metrics are missing.")
    _assert_metric_mapping(recorded_metrics, recomputed_metrics, "test_metrics")

    recomputed_ticker_metrics = per_ticker_metrics(predictions)
    if summary.get("per_ticker_test_metrics") != recomputed_ticker_metrics:
        raise CombinedRunError("Per-ticker test metrics changed.")

    training_config = TrainingConfig(**summary.get("training_config", {}))
    recomputed_gates = evaluate_quality_gates(
        str(summary["quality_champion"]),
        list(summary["validation_ranking"]),
        dict(recorded_metrics),
        dict(summary["baseline_test_metrics"]),
        recomputed_ticker_metrics,
        training_config,
    )
    _assert_metric_mapping(
        dict(summary["quality_gates"]),
        recomputed_gates,
        "quality_gates",
    )

    model_bundle = joblib.load(outputs["model"])
    if model_bundle.get("label_order") != list(LABEL_ORDER):
        raise CombinedRunError("Saved movement label order changed.")
    if model_bundle.get("champion_name") != summary.get("quality_champion"):
        raise CombinedRunError("Model and metrics champion names differ.")
    if model_bundle.get("foundation_manifest_sha256") != (
        EXPECTED_FOUNDATION_MANIFEST_SHA256
    ):
        raise CombinedRunError("Saved model references a different foundation.")
    if model_bundle.get("text_features") != summary.get("text_features"):
        raise CombinedRunError("Model and metrics text feature lists differ.")
    if model_bundle.get("decision_policy") != summary.get("decision_policy"):
        raise CombinedRunError("Model and metrics decision policies differ.")
    decision_policy = summary.get("decision_policy")
    if not isinstance(decision_policy, Mapping):
        raise CombinedRunError("Saved decision policy is missing.")
    if decision_policy.get("ticker_logit_offsets") != {}:
        raise CombinedRunError(
            "Production ticker-specific decision offsets are forbidden."
        )
    global_offsets = decision_policy.get("global_logit_offsets")
    if not isinstance(global_offsets, Mapping):
        raise CombinedRunError("Production global decision offsets are missing.")
    if float(global_offsets.get("Down", 999.0)) != 0.0:
        raise CombinedRunError("The Down policy offset must remain zero.")
    allowed_offsets = _allowed_decision_global_offsets(training_config)
    for label in ("Flat", "Up"):
        if float(global_offsets.get(label, 999.0)) not in allowed_offsets:
            raise CombinedRunError(
                f"Production {label} policy offset is outside the fixed grid."
            )
    if decision_policy.get("fit_split") != "selection_oof":
        raise CombinedRunError(
            "Production decision policy was not fitted from selection OOF."
        )

    # Reload the champion and recompute the exact test rows. This catches any
    # divergence between the serialized model, frozen validation policy, and
    # saved label/probability evidence.
    expected_test_frame = model_table[model_table["split"] == "test"].copy()
    recomputed_predictions = _predict_output(
        model_bundle["pipeline"],
        expected_test_frame,
        list(summary["numeric_features"])
        + list(summary["categorical_features"])
        + list(summary["text_features"]),
        dict(summary["decision_policy"]),
    )
    comparable_columns = [
        "ticker",
        "target_session_date",
        "actual_movement",
        "predicted_movement",
        "prob_down",
        "prob_flat",
        "prob_up",
    ]
    saved_comparable = predictions[comparable_columns].reset_index(drop=True)
    fresh_comparable = recomputed_predictions[comparable_columns].reset_index(
        drop=True
    )
    if not saved_comparable[[
        "ticker",
        "target_session_date",
        "actual_movement",
        "predicted_movement",
    ]].equals(
        fresh_comparable[[
            "ticker",
            "target_session_date",
            "actual_movement",
            "predicted_movement",
        ]]
    ):
        raise CombinedRunError("Reloaded model predictions changed.")
    if not np.allclose(
        saved_comparable[["prob_down", "prob_flat", "prob_up"]],
        fresh_comparable[["prob_down", "prob_flat", "prob_up"]],
        atol=1e-10,
    ):
        raise CombinedRunError("Reloaded model probabilities changed.")
    importance_records = model_bundle.get("validation_global_importance")
    if not isinstance(importance_records, list):
        raise CombinedRunError("Validation-only global importance is missing.")
    _validate_global_importance_records(importance_records)

    # A joblib model is only considered reloadable in the same validated
    # runtime versions that created it.
    current_versions = runtime_versions()
    recorded_versions = summary.get("runtime_versions")
    if not isinstance(recorded_versions, Mapping):
        raise CombinedRunError("Runtime version evidence is missing.")
    for key in ("python", "numpy", "pandas", "scikit_learn", "joblib"):
        if recorded_versions.get(key) != current_versions.get(key):
            raise CombinedRunError(f"Runtime version changed for {key}.")
    return summary


def run_movement_phase(
    project_root: Path,
    replace_existing: bool,
    diagnostic_dir: Path | None = None,
) -> dict[str, Any]:
    """Search, diagnose, gate, persist, and verify the movement champion."""

    root = project_root.expanduser().resolve()
    runtime_report = _assert_compatible_openmp_runtime()
    outputs = resolve_outputs(root)
    protect_outputs(root, replace_existing)
    news, prices, foundation_manifest = load_foundation_frames(root)
    model_table = build_model_table(news, prices)
    split_table, split_report = assign_chronological_splits(model_table)
    numeric_features, categorical_features, text_features = feature_columns(
        split_table
    )
    result = train_and_evaluate(
        split_table,
        numeric_features,
        categorical_features,
        text_features,
        allow_failed_result=True,
    )
    result["openmp_runtime_report"] = runtime_report

    # Diagnostics are written outside the project before any quality failure is
    # raised. Project rollback can then restore its exact pre-strike state while
    # the Commander still receives the evidence needed for the next decision.
    diagnostic_path = write_external_diagnostics(
        diagnostic_dir,
        result,
        split_report,
    )
    if result["status"] != "passed":
        location = str(diagnostic_path) if diagnostic_path else "not requested"
        failures = (
            result.get("quality_gates", {}).get("failures")
            or result.get(
                "development_confirmation_gates",
                {},
            ).get("failures")
            or result.get("validation_gates", {}).get("failures")
            or ["Unknown quality failure"]
        )
        raise CombinedRunError(
            "Movement model did not pass unchanged quality gates. "
            f"Diagnostics: {location}. Failures: {' | '.join(failures)}"
        )

    model_bundle = {
        "pipeline": result["champion_pipeline"],
        "champion_name": result["champion_name"],
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "text_features": text_features,
        "validation_global_importance": result["global_importance"].to_dict(
            orient="records"
        ),
        "label_order": list(LABEL_ORDER),
        "foundation_manifest_sha256": EXPECTED_FOUNDATION_MANIFEST_SHA256,
        "random_seed": result["random_seed"],
        "runtime_versions": result["runtime_versions"],
        "openmp_runtime_report": result["openmp_runtime_report"],
        "champion_refit_convergence": result[
            "champion_refit_convergence"
        ],
        "decision_policy": result["decision_policy"],
    }
    write_joblib(outputs["model"], model_bundle)
    write_csv(outputs["model_table"], split_table)
    write_csv(outputs["test_predictions"], result["test_predictions"])

    # Checksums are recorded only after all three movement artifacts complete
    # atomic writes. Intelligence cannot start before these files re-verify.
    movement_artifacts = [
        artifact_entry(root, outputs[name])
        for name in ("model", "model_table", "test_predictions")
    ]
    summary = {
        "status": "trained_and_evaluated",
        "package_contract_version": PACKAGE_CONTRACT_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "quality_champion": result["champion_name"],
        "selection_split": "purged_expanding_development_selection",
        "development_confirmation_split": result[
            "development_confirmation_split"
        ],
        "development_confirmation_metrics": result[
            "development_confirmation_metrics"
        ],
        "baseline_development_confirmation_metrics": result[
            "baseline_development_confirmation_metrics"
        ],
        "development_confirmation_gates": result[
            "development_confirmation_gates"
        ],
        "development_confirmation_evaluation_count": result[
            "development_confirmation_evaluation_count"
        ],
        "development_confirmation_pristine": result[
            "development_confirmation_pristine"
        ],
        "development_confirmation_known_from_prior_run": result[
            "development_confirmation_known_from_prior_run"
        ],
        "development_confirmation_used_for_candidate_selection": (
            result[
                "development_confirmation_used_for_candidate_selection"
            ]
        ),
        "development_confirmation_tournament": result[
            "development_confirmation_tournament"
        ],
        "evaluation_protocol": result["evaluation_protocol"],
        "historical_audit_pristine": result[
            "historical_audit_pristine"
        ],
        "test_used_for_selection": False,
        "historical_audit_used_for_selection": False,
        "test_evaluation_count": result["test_evaluation_count"],
        "historical_audit_evaluation_count": result[
            "historical_audit_evaluation_count"
        ],
        "candidate_results": result["candidate_results"],
        "validation_ranking": result["validation_ranking"],
        "validation_gates": result["validation_gates"],
        "rolling_fold_reports": result["rolling_fold_reports"],
        "test_metrics": result["test_metrics"],
        "historical_audit_metrics": result[
            "historical_audit_metrics"
        ],
        "baseline_test_metrics": result["baseline_test_metrics"],
        "baseline_historical_audit_metrics": result[
            "baseline_historical_audit_metrics"
        ],
        "per_ticker_test_metrics": result["per_ticker_test_metrics"],
        "per_ticker_historical_audit_metrics": result[
            "per_ticker_historical_audit_metrics"
        ],
        "quality_gates": result["quality_gates"],
        "test_latency_ms_per_record": result["test_latency_ms_per_record"],
        "historical_audit_latency_ms_per_record": result[
            "historical_audit_latency_ms_per_record"
        ],
        "refit_seconds": result["refit_seconds"],
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "text_features": text_features,
        "split_report": split_report,
        "random_seed": result["random_seed"],
        "runtime_versions": result["runtime_versions"],
        "openmp_runtime_report": result["openmp_runtime_report"],
        "training_config": result["training_config"],
        "decision_policy": result["decision_policy"],
        "decision_policy_calibration": result[
            "decision_policy_calibration"
        ],
        "foundation_manifest_sha256": EXPECTED_FOUNDATION_MANIFEST_SHA256,
        "foundation_status": foundation_manifest.get("status"),
        "movement_artifacts": movement_artifacts,
        "movement_model_trained": True,
        "explainability_started": False,
        "deployment_changed": False,
        "external_diagnostic_summary": (
            str(diagnostic_path) if diagnostic_path else None
        ),
    }
    write_json(outputs["movement_metrics"], summary)
    verified = verify_movement_phase(root)
    print("STOCK MOVEMENT MODEL: PASSED", flush=True)
    print(f"Champion: {verified['quality_champion']}", flush=True)
    print(
        "Historical-audit macro F1: "
        f"{verified['test_metrics']['macro_f1']:.6f}",
        flush=True,
    )
    print(
        "Historical-audit weighted F1: "
        f"{verified['test_metrics']['weighted_f1']:.6f}",
        flush=True,
    )
    return verified


def _validate_intelligence_semantics(project_root: Path) -> None:
    """Recompute coverage, chronology, and licence claims from saved outputs."""

    root = project_root.expanduser().resolve()
    outputs = resolve_outputs(root)
    movement_summary = verify_movement_phase(root)
    predictions = pd.read_csv(outputs["test_predictions"])
    predictions["target_session_date"] = pd.to_datetime(
        predictions["target_session_date"],
        errors="coerce",
    ).dt.normalize()
    expected_ids = set(predictions["record_id"].astype(int))

    # Each output is checked for semantic coverage after its checksum passes.
    # This prevents an empty but correctly checksummed file from passing.
    global_frame = pd.read_csv(outputs["global_drivers"])
    if (
        global_frame.empty
        or global_frame["importance"].gt(0).sum() < MINIMUM_GLOBAL_DRIVERS
    ):
        raise CombinedRunError("Global driver evidence is empty or all zero.")

    local_frame = pd.read_csv(outputs["local_drivers"])
    local_counts = local_frame.groupby("record_id", observed=True).size()
    if set(local_counts.index.astype(int)) != expected_ids:
        raise CombinedRunError("Local explanation coverage is incomplete.")
    if local_counts.lt(MINIMUM_LOCAL_DRIVERS_PER_RECORD).any():
        raise CombinedRunError("A test record has too few local drivers.")
    if local_frame.groupby("record_id")["absolute_effect"].max().le(0).any():
        raise CombinedRunError("A test record has only zero local effects.")

    phrase_frame = pd.read_csv(outputs["sentiment_phrases"])
    phrase_required = {
        "sentiment_label",
        "association_score",
        "relevance_score",
        "reference_record_count",
        "hard_label_record_count",
        "effective_reference_weight",
        "probability_fallback_used",
        "selection_basis",
        "method",
    }
    phrase_missing = sorted(phrase_required - set(phrase_frame.columns))
    if phrase_missing:
        raise CombinedRunError(
            f"Sentiment phrase evidence is missing columns: {phrase_missing}"
        )
    phrase_counts = phrase_frame.groupby("sentiment_label", observed=True).size()
    if set(phrase_counts.index) != {"Bearish", "Neutral", "Bullish"}:
        raise CombinedRunError("Sentiment phrase class coverage is incomplete.")
    if phrase_counts.lt(3).any():
        raise CombinedRunError("A sentiment class has too few phrases.")

    # Probability fallback is allowed only as explicit reference-only evidence
    # when a hard sentiment class is absent. Saved weights and relevance must
    # remain finite and positive so placeholder rows cannot pass verification.
    numeric_phrase_columns = [
        "association_score",
        "relevance_score",
        "reference_record_count",
        "hard_label_record_count",
        "effective_reference_weight",
    ]
    phrase_numeric = phrase_frame[numeric_phrase_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )
    if not np.isfinite(phrase_numeric.to_numpy(dtype=float)).all():
        raise CombinedRunError("Sentiment phrase numeric evidence is invalid.")
    if phrase_numeric["reference_record_count"].le(0).any():
        raise CombinedRunError("Sentiment phrase reference counts are invalid.")
    if phrase_numeric["hard_label_record_count"].lt(0).any():
        raise CombinedRunError("Sentiment phrase hard-label counts are invalid.")
    if phrase_numeric["effective_reference_weight"].le(0).any():
        raise CombinedRunError("Sentiment phrase effective weights are invalid.")
    if phrase_numeric["relevance_score"].le(0).any():
        raise CombinedRunError("Sentiment phrase relevance evidence is invalid.")

    allowed_phrase_methods = {
        "reference_only_class_mean_tfidf_difference",
        "reference_only_probability_weighted_tfidf_difference",
    }
    if not set(phrase_frame["method"]).issubset(allowed_phrase_methods):
        raise CombinedRunError("Sentiment phrase method is not approved.")
    allowed_selection_basis = {
        "positive_tfidf_contrast",
        "probability_weighted_relevance",
    }
    if not set(phrase_frame["selection_basis"]).issubset(
        allowed_selection_basis
    ):
        raise CombinedRunError("Sentiment phrase selection basis is invalid.")
    fallback_flags = (
        phrase_frame["probability_fallback_used"]
        .astype(str)
        .str.strip()
        .str.lower()
    )
    if not set(fallback_flags).issubset({"true", "false"}):
        raise CombinedRunError("Sentiment phrase fallback flags are invalid.")
    fallback_rows = fallback_flags.eq("true")
    if fallback_rows.any():
        if not fallback_rows.all():
            raise CombinedRunError(
                "Sentiment probability fallback flags are inconsistent."
            )
        if set(phrase_frame["method"]) != {
            "reference_only_probability_weighted_tfidf_difference"
        }:
            raise CombinedRunError(
                "Sentiment probability fallback method is inconsistent."
            )
    else:
        if set(phrase_frame["method"]) != {
            "reference_only_class_mean_tfidf_difference"
        }:
            raise CombinedRunError("Hard-label sentiment phrase method changed.")
        if "probability_weighted_relevance" in set(
            phrase_frame["selection_basis"]
        ):
            raise CombinedRunError(
                "Hard-label phrases unexpectedly used probability fallback."
            )

    match_frame = pd.read_csv(outputs["historical_matches"])
    match_counts = match_frame.groupby("record_id", observed=True).size()
    if set(match_counts.index.astype(int)) != expected_ids:
        raise CombinedRunError("Historical match coverage is incomplete.")
    if match_counts.lt(MINIMUM_MATCHES_PER_RECORD).any():
        raise CombinedRunError("A test record has too few historical matches.")
    query_dates = pd.to_datetime(match_frame["query_session_date"], errors="coerce")
    historical_dates = pd.to_datetime(
        match_frame["historical_session_date"],
        errors="coerce",
    )
    if query_dates.isna().any() or historical_dates.isna().any():
        raise CombinedRunError("Historical match dates are invalid.")
    if (historical_dates >= query_dates).any():
        raise CombinedRunError("Historical matches include non-earlier evidence.")
    if set(match_frame["candidate_scope"]) != {
        "train_validation_reference_only"
    }:
        raise CombinedRunError("Historical match reference scope changed.")

    context_frame = pd.read_csv(outputs["company_context"])
    if context_frame["ticker"].duplicated().any():
        raise CombinedRunError("Company context is not one row per ticker.")
    expected_tickers = set(
        pd.read_csv(outputs["model_table"])["ticker"].astype(str).unique()
    )
    if set(context_frame["ticker"]) != expected_tickers:
        raise CombinedRunError("Company context ticker coverage is incomplete.")
    validation_end = pd.Timestamp(
        movement_summary["split_report"]["validation"]["end_date"]
    )
    context_end = pd.to_datetime(context_frame["last_event_date"], errors="coerce")
    if context_end.isna().any() or (context_end > validation_end).any():
        raise CombinedRunError("Company context contains test-period evidence.")

    scenario_frame = pd.read_csv(outputs["scenarios"])
    if scenario_frame["record_id"].duplicated().any():
        raise CombinedRunError("Scenario output is not one row per test record.")
    if set(scenario_frame["record_id"].astype(int)) != expected_ids:
        raise CombinedRunError("Scenario coverage is incomplete.")
    scenario_probabilities = scenario_frame[
        ["prob_down", "prob_flat", "prob_up"]
    ]
    if not np.allclose(scenario_probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise CombinedRunError("Scenario probabilities do not sum to one.")
    if (scenario_frame["downside_return"] > scenario_frame["upside_return"]).any():
        raise CombinedRunError("Scenario downside and upside values are reversed.")
    reference_end = pd.to_datetime(
        scenario_frame["reference_history_end_date"],
        errors="coerce",
    )
    scenario_dates = pd.to_datetime(
        scenario_frame["target_session_date"],
        errors="coerce",
    )
    if (reference_end >= scenario_dates).any():
        raise CombinedRunError("Scenario history is not strictly earlier.")

    provenance = load_json_object(outputs["provenance"], "provenance report")
    if provenance.get("status") != "provenance_verified":
        raise CombinedRunError("Provenance report is not verified.")
    licence = provenance.get("licence_boundary", {})
    if licence.get("raw_tiingo_values_publicly_redistributable") is not False:
        raise CombinedRunError("Tiingo redistribution boundary changed.")
    if provenance.get("deployment_changed") is not False:
        raise CombinedRunError("Provenance unexpectedly reports deployment changes.")

    # Scan every public CSV header. Derived returns are allowed, but raw Tiingo
    # OHLCV fields must not appear in movement or intelligence outputs.
    for name, file_path in outputs.items():
        if name in {"manifest", "provenance", "movement_metrics", "model"}:
            continue
        columns = set(pd.read_csv(file_path, nrows=0).columns)
        forbidden = columns & FORBIDDEN_PUBLIC_COLUMNS
        if forbidden:
            raise CombinedRunError(
                f"Restricted raw price columns found in {name}: {sorted(forbidden)}"
            )


def run_intelligence_phase(project_root: Path) -> dict[str, Any]:
    """Build intelligence only after movement verification passes."""

    # Intelligence imports are delayed until the movement model passes.
    # This keeps the training process free of unrelated native libraries.
    from financial_news_intelligence.intelligence.historical_intelligence import (
        company_context,
        earlier_only_matches,
        sentiment_phrases,
    )
    from financial_news_intelligence.intelligence.investment_scenarios import (
        build_scenarios,
    )
    from financial_news_intelligence.intelligence.movement_explainability import (
        global_drivers,
        local_perturbation_drivers,
    )
    from financial_news_intelligence.intelligence.provenance import (
        build_provenance_report,
    )

    root = project_root.expanduser().resolve()
    movement_summary = verify_movement_phase(root)
    outputs = resolve_outputs(root)
    news, _, foundation_manifest = load_foundation_frames(root)
    model_table = pd.read_csv(outputs["model_table"])
    model_table["target_session_date"] = pd.to_datetime(
        model_table["target_session_date"],
        errors="coerce",
    ).dt.normalize()
    predictions = pd.read_csv(outputs["test_predictions"])
    predictions["target_session_date"] = pd.to_datetime(
        predictions["target_session_date"],
        errors="coerce",
    ).dt.normalize()
    bundle = joblib.load(outputs["model"])
    champion = bundle["pipeline"]

    # Reference-only global intelligence prevents test-period evidence from
    # influencing phrases, company context, historical matches, or scenarios.
    reference_events = filter_events_by_split(
        news,
        model_table,
        {"train", "validation"},
    )
    query_events = filter_events_by_split(news, model_table, {"test"})
    expected_tickers = set(model_table["ticker"].astype(str).unique())

    importance_records = bundle.get("validation_global_importance")
    if not isinstance(importance_records, list):
        raise CombinedRunError("Validation-only global importance is missing.")
    global_frame = global_drivers(pd.DataFrame(importance_records))
    local_frame = local_perturbation_drivers(
        champion,
        model_table,
        predictions,
        bundle["numeric_features"],
        bundle["categorical_features"],
        bundle.get("text_features", []),
    )
    phrase_frame = sentiment_phrases(reference_events)
    match_frame = earlier_only_matches(
        reference_events,
        query_events,
        predictions,
    )
    context_frame = company_context(reference_events, expected_tickers)
    scenario_frame = build_scenarios(reference_events, predictions)
    provenance_payload = build_provenance_report(
        foundation_manifest,
        movement_summary,
    )

    write_csv(outputs["global_drivers"], global_frame)
    write_csv(outputs["local_drivers"], local_frame)
    write_csv(outputs["sentiment_phrases"], phrase_frame)
    write_csv(outputs["historical_matches"], match_frame)
    write_csv(outputs["company_context"], context_frame)
    write_csv(outputs["scenarios"], scenario_frame)
    write_json(outputs["provenance"], provenance_payload)

    artifact_names = [name for name in outputs if name != "manifest"]
    manifest_payload = {
        "status": "movement_and_intelligence_verified",
        "package_contract_version": PACKAGE_CONTRACT_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "foundation_manifest_sha256": EXPECTED_FOUNDATION_MANIFEST_SHA256,
        "quality_champion": movement_summary["quality_champion"],
        "test_metrics": movement_summary["test_metrics"],
        "quality_gates": movement_summary["quality_gates"],
        "test_used_for_selection": False,
        "explainability_started_after_model_pass": True,
        "global_intelligence_reference_scope": "train_validation_only",
        "sentiment_phrase_reference_scope": "train_validation_only",
        "sentiment_phrase_methods": sorted(set(phrase_frame["method"])),
        "sentiment_phrase_probability_fallback_used": bool(
            phrase_frame["probability_fallback_used"].any()
        ),
        "historical_retrieval_strictly_earlier_only": True,
        "historical_match_minimum_per_record": MINIMUM_MATCHES_PER_RECORD,
        "local_driver_minimum_per_record": MINIMUM_LOCAL_DRIVERS_PER_RECORD,
        "raw_tiingo_values_exported": False,
        "public_deployment_authorized": False,
        "deployment_changed": False,
        "artifacts": [
            artifact_entry(root, outputs[name]) for name in artifact_names
        ],
    }
    write_json(outputs["manifest"], manifest_payload)
    verify_manifest(root)
    _validate_intelligence_semantics(root)
    print("EXPLAINABILITY AND INTELLIGENCE: PASSED", flush=True)
    print(f"Intelligence artifacts: {len(artifact_names) - 4}", flush=True)
    return load_json_object(outputs["manifest"], "combined manifest")


def verify_all(project_root: Path) -> dict[str, Any]:
    """Verify final checksums plus all movement and intelligence semantics."""

    root = project_root.expanduser().resolve()
    manifest = verify_manifest(root)
    verify_movement_phase(root)
    _validate_intelligence_semantics(root)
    if manifest.get("quality_gates", {}).get("status") != "passed":
        raise CombinedRunError("Final manifest quality gates are not passed.")
    if manifest.get("global_intelligence_reference_scope") != (
        "train_validation_only"
    ):
        raise CombinedRunError("Global intelligence scope changed.")
    if manifest.get("raw_tiingo_values_exported") is not False:
        raise CombinedRunError("Final manifest reports raw Tiingo export.")
    return manifest


def parse_args() -> argparse.Namespace:
    """Parse the project root, phase, and controlled replacement flag."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument(
        "--phase",
        choices=("movement", "intelligence", "all", "verify"),
        default="all",
    )
    parser.add_argument("--replace-existing", action="store_true")
    parser.add_argument(
        "--diagnostic-dir",
        type=Path,
        help=(
            "Owner-only external directory that keeps licence-safe diagnostics "
            "after project rollback."
        ),
    )
    return parser.parse_args()


def main() -> int:
    """Run the requested phase and convert failures to a clear exit code."""

    args = parse_args()
    try:
        if args.phase in {"movement", "all"}:
            run_movement_phase(
                args.project_root,
                args.replace_existing,
                args.diagnostic_dir,
            )
        if args.phase in {"intelligence", "all"}:
            run_intelligence_phase(args.project_root)
        if args.phase == "verify":
            verify_all(args.project_root)
            print("MOVEMENT INTELLIGENCE VERIFICATION: PASSED", flush=True)
    except Exception as exc:  # noqa: BLE001 - CLI must report exact failure.
        print(
            f"MOVEMENT INTELLIGENCE FAILED: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
