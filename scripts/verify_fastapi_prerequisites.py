#!/usr/bin/env python3
"""Verify FastAPI prerequisite manifests and artifact checksums safely.

Purpose
-------
Confirm that the completed foundation, movement, explainability, intelligence,
and sentiment-selection phases still match the audited deployment contracts.
This verifier uses only the Python standard library. It deliberately avoids
NumPy, scikit-learn, joblib, PyTorch, and threadpoolctl so prerequisite checks
cannot load incompatible native OpenMP runtimes.

Inputs and outputs
------------------
The input is the existing project root. One optional owner-only JSON evidence
file records verified manifests, artifact counts, and deployment boundaries.
No project artifact is modified.

Failure behavior
----------------
Every mismatch fails closed. The verifier names what failed, where it failed,
why it failed, and the exact safe next step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


EXPECTED_MOVEMENT_CHAMPION = "stability_soft_vote_rf_sgd"
EXPECTED_SENTIMENT_DEPLOYMENT_MODEL = "distilbert"
MINIMUM_TEST_MACRO_F1 = 0.30
MINIMUM_TEST_WEIGHTED_F1 = 0.40
REQUIRED_MOVEMENT_ARTIFACTS = {
    "artifacts/models/stock_movement/champion_model.joblib",
    "data/processed/stock_movement_model_table.csv",
    "data/processed/stock_movement_test_predictions.csv",
    "reports/explainability/movement_global_drivers.csv",
    "reports/explainability/movement_local_drivers.csv",
    "reports/intelligence/sentiment_phrases.csv",
    "reports/intelligence/historical_matches.csv",
    "reports/intelligence/company_context.csv",
    "reports/intelligence/investment_scenarios.csv",
    "reports/intelligence/provenance_and_disclaimers.json",
}
REQUIRED_FOUNDATION_ARTIFACTS = {
    "data/processed/news_sentiment_evidence.csv",
    "data/processed/market_price_evidence.csv",
    "reports/qa/market_data_foundation_qa.json",
}


class PrerequisiteVerificationError(RuntimeError):
    """Raised when a required project prerequisite is missing or changed."""


def parse_args() -> argparse.Namespace:
    """Parse the project root and optional evidence path."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--evidence-file", type=Path)
    return parser.parse_args()


def safe_project_path(project_root: Path, relative_path: str) -> Path:
    """Resolve one manifest path and require it to remain in the project."""

    candidate = Path(relative_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise PrerequisiteVerificationError(
            f"Unsafe manifest path: {relative_path}"
        )
    resolved_root = project_root.expanduser().resolve()
    resolved_target = (resolved_root / candidate).resolve()
    try:
        resolved_target.relative_to(resolved_root)
    except ValueError as exc:
        raise PrerequisiteVerificationError(
            f"Manifest path escapes project root: {relative_path}"
        ) from exc
    return resolved_target


def sha256(file_path: Path) -> str:
    """Return the SHA-256 checksum for one safe regular file."""

    if not file_path.exists() or file_path.is_symlink() or not file_path.is_file():
        raise PrerequisiteVerificationError(
            f"Required file is missing or unsafe: {file_path}"
        )
    digest = hashlib.sha256()
    with file_path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(file_path: Path, description: str) -> dict[str, Any]:
    """Load one required JSON object after verifying it is a regular file."""

    sha256(file_path)
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PrerequisiteVerificationError(
            f"{description} is not valid JSON: {file_path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise PrerequisiteVerificationError(
            f"{description} must contain one JSON object: {file_path}"
        )
    return payload


def require_owner_only(file_path: Path) -> None:
    """Require no group or other permission bits on a private artifact."""

    mode = stat.S_IMODE(file_path.stat().st_mode)
    if mode & 0o077:
        raise PrerequisiteVerificationError(
            f"Artifact permissions are not owner-only: {file_path}: {oct(mode)}"
        )


def verify_inventory(
    project_root: Path,
    entries: Any,
    description: str,
) -> dict[str, Path]:
    """Verify one manifest inventory and return paths by recorded name."""

    if not isinstance(entries, list) or not entries:
        raise PrerequisiteVerificationError(
            f"{description} artifact inventory is missing or empty."
        )
    verified: dict[str, Path] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise PrerequisiteVerificationError(
                f"{description} contains a non-object artifact entry."
            )
        relative_path = entry.get("path")
        expected_checksum = entry.get("sha256")
        expected_size = entry.get("size_bytes")
        if not isinstance(relative_path, str) or not isinstance(
            expected_checksum, str
        ):
            raise PrerequisiteVerificationError(
                f"{description} contains an incomplete artifact entry."
            )
        file_path = safe_project_path(project_root, relative_path)
        actual_checksum = sha256(file_path)
        if actual_checksum != expected_checksum:
            raise PrerequisiteVerificationError(
                f"Artifact checksum changed: {relative_path}"
            )
        if isinstance(expected_size, int) and file_path.stat().st_size != expected_size:
            raise PrerequisiteVerificationError(
                f"Artifact size changed: {relative_path}"
            )
        require_owner_only(file_path)
        verified[relative_path] = file_path
    return verified


def finite_number(value: Any, field_name: str) -> float:
    """Return one finite number or fail with the exact manifest field."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PrerequisiteVerificationError(
            f"Manifest field is not numeric: {field_name}"
        )
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        raise PrerequisiteVerificationError(
            f"Manifest field is not finite: {field_name}"
        )
    return number


def verify_contracts(project_root: Path) -> dict[str, Any]:
    """Verify manifests, inventories, quality gates, and deployment boundaries."""

    foundation_path = safe_project_path(
        project_root,
        "artifacts/manifests/market_data_foundation_manifest.json",
    )
    movement_path = safe_project_path(
        project_root,
        "artifacts/manifests/movement_intelligence_manifest.json",
    )
    sentiment_path = safe_project_path(
        project_root,
        "artifacts/manifests/sentiment_model_champion.json",
    )

    foundation = load_json(foundation_path, "Market Data Foundation manifest")
    movement = load_json(movement_path, "Movement Intelligence manifest")
    sentiment = load_json(sentiment_path, "sentiment champion manifest")

    if foundation.get("status") != "foundation_verified":
        raise PrerequisiteVerificationError(
            "Market Data Foundation status is not foundation_verified."
        )
    readiness = foundation.get("readiness")
    if not isinstance(readiness, Mapping) or readiness.get(
        "ready_for_stock_movement_package"
    ) is not True:
        raise PrerequisiteVerificationError(
            "Market Data Foundation readiness is not approved for movement use."
        )
    if foundation.get("automatic_deployment_change") is not False:
        raise PrerequisiteVerificationError(
            "Market Data Foundation unexpectedly reports a deployment change."
        )

    if movement.get("status") != "movement_and_intelligence_verified":
        raise PrerequisiteVerificationError(
            "Movement and intelligence status is not verified."
        )
    if movement.get("quality_champion") != EXPECTED_MOVEMENT_CHAMPION:
        raise PrerequisiteVerificationError(
            "The verified movement champion changed."
        )
    if movement.get("deployment_changed") is not False:
        raise PrerequisiteVerificationError(
            "Movement Intelligence unexpectedly reports a deployment change."
        )
    if movement.get("public_deployment_authorized") is not False:
        raise PrerequisiteVerificationError(
            "Movement Intelligence unexpectedly authorizes public deployment."
        )
    if movement.get("raw_tiingo_values_exported") is not False:
        raise PrerequisiteVerificationError(
            "Movement Intelligence reports that raw Tiingo values were exported."
        )
    if movement.get("test_used_for_selection") is not False:
        raise PrerequisiteVerificationError(
            "Movement Intelligence reports historical-audit selection leakage."
        )
    expected_foundation_checksum = movement.get("foundation_manifest_sha256")
    if expected_foundation_checksum != sha256(foundation_path):
        raise PrerequisiteVerificationError(
            "Movement Intelligence references a different foundation manifest."
        )

    test_metrics = movement.get("test_metrics")
    if not isinstance(test_metrics, Mapping):
        raise PrerequisiteVerificationError(
            "Movement Intelligence test metrics are missing."
        )
    macro_f1 = finite_number(test_metrics.get("macro_f1"), "test_metrics.macro_f1")
    weighted_f1 = finite_number(
        test_metrics.get("weighted_f1"),
        "test_metrics.weighted_f1",
    )
    if macro_f1 < MINIMUM_TEST_MACRO_F1:
        raise PrerequisiteVerificationError(
            f"Historical-audit macro F1 is below {MINIMUM_TEST_MACRO_F1:.2f}."
        )
    if weighted_f1 < MINIMUM_TEST_WEIGHTED_F1:
        raise PrerequisiteVerificationError(
            f"Historical-audit weighted F1 is below {MINIMUM_TEST_WEIGHTED_F1:.2f}."
        )

    if sentiment.get("status") != "champion_selected":
        raise PrerequisiteVerificationError(
            "Sentiment champion selection status is not champion_selected."
        )
    if sentiment.get("recommended_deployment_model") != (
        EXPECTED_SENTIMENT_DEPLOYMENT_MODEL
    ):
        raise PrerequisiteVerificationError(
            "The recommended sentiment deployment model changed."
        )
    if sentiment.get("automatic_deployment_change") is not False:
        raise PrerequisiteVerificationError(
            "Sentiment comparison unexpectedly reports a deployment change."
        )

    foundation_files = verify_inventory(
        project_root,
        foundation.get("artifacts"),
        "Market Data Foundation",
    )
    movement_files = verify_inventory(
        project_root,
        movement.get("artifacts"),
        "Movement Intelligence",
    )
    missing_foundation = sorted(REQUIRED_FOUNDATION_ARTIFACTS - set(foundation_files))
    if missing_foundation:
        raise PrerequisiteVerificationError(
            f"Required foundation artifacts are missing: {missing_foundation}"
        )
    missing_movement = sorted(REQUIRED_MOVEMENT_ARTIFACTS - set(movement_files))
    if missing_movement:
        raise PrerequisiteVerificationError(
            "Required movement or intelligence artifacts are missing: "
            f"{missing_movement}"
        )

    return {
        "status": "fastapi_prerequisites_verified",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "foundation_manifest_sha256": sha256(foundation_path),
        "movement_manifest_sha256": sha256(movement_path),
        "sentiment_champion_manifest_sha256": sha256(sentiment_path),
        "foundation_artifact_count": len(foundation_files),
        "movement_artifact_count": len(movement_files),
        "movement_champion": EXPECTED_MOVEMENT_CHAMPION,
        "sentiment_deployment_model": EXPECTED_SENTIMENT_DEPLOYMENT_MODEL,
        "historical_audit_macro_f1": macro_f1,
        "historical_audit_weighted_f1": weighted_f1,
        "native_numerical_libraries_loaded": False,
        "deployment_changed": False,
        "public_deployment_authorized": False,
    }


def write_evidence(file_path: Path, payload: Mapping[str, Any]) -> None:
    """Write one owner-only JSON evidence file atomically."""

    destination = file_path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.chmod(temporary, 0o600)
        temporary.replace(destination)
        os.chmod(destination, 0o600)
    finally:
        if temporary.exists() and temporary.is_file():
            temporary.unlink()


def main() -> int:
    """Run read-only prerequisite verification with complete failure details."""

    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    print("STARTED: FastAPI prerequisite verification.", flush=True)
    try:
        if not project_root.exists() or not project_root.is_dir():
            raise PrerequisiteVerificationError(
                f"Project root does not exist: {project_root}"
            )
        evidence = verify_contracts(project_root)
        if args.evidence_file is not None:
            write_evidence(args.evidence_file, evidence)
            print(
                f"PASSED: Prerequisite evidence written to {args.evidence_file}.",
                flush=True,
            )
        print("FASTAPI PREREQUISITE VERIFICATION: PASSED", flush=True)
        return 0
    except Exception as exc:  # noqa: BLE001 - verifier must name exact failure.
        print("FAILED: FastAPI prerequisite verification failed.", file=sys.stderr)
        print("Location: FastAPI prerequisite verifier", file=sys.stderr)
        print(f"Reason: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(
            "Safe next step: Restore the last verified manifests and artifacts, "
            "then rerun this verifier.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
