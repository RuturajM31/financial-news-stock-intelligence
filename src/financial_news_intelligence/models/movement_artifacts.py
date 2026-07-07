"""Persist and verify movement-model and intelligence artifacts safely.

Purpose
-------
Centralize every output path, owner-only atomic write, checksum calculation,
and temporary-file cleanup used by the combined movement and intelligence
package.

Inputs and outputs
------------------
Inputs are validated pandas tables, JSON-ready mappings, and a fitted
scikit-learn model bundle. Outputs stay inside dedicated project directories
and are consumed later by the API, Streamlit, monitoring, and documentation
phases.

Safety boundaries
-----------------
The module rejects symlinks, paths outside the project, unsafe replacement
targets, and incomplete atomic writes. Raw Tiingo cache files are never copied
or inventoried here. Temporary files are removed after success, failure, or
interruption.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import joblib
import pandas as pd

# These are the only files that the combined package is allowed to create.
# Keeping the list in one place makes installation, rollback, and verification
# use the same contract instead of maintaining separate path lists.
OUTPUT_PATHS = {
    "model": Path("artifacts/models/stock_movement/champion_model.joblib"),
    "movement_metrics": Path("reports/metrics/stock_movement_metrics.json"),
    "model_table": Path("data/processed/stock_movement_model_table.csv"),
    "test_predictions": Path("data/processed/stock_movement_test_predictions.csv"),
    "global_drivers": Path("reports/explainability/movement_global_drivers.csv"),
    "local_drivers": Path("reports/explainability/movement_local_drivers.csv"),
    "sentiment_phrases": Path("reports/intelligence/sentiment_phrases.csv"),
    "historical_matches": Path("reports/intelligence/historical_matches.csv"),
    "company_context": Path("reports/intelligence/company_context.csv"),
    "scenarios": Path("reports/intelligence/investment_scenarios.csv"),
    "provenance": Path("reports/intelligence/provenance_and_disclaimers.json"),
    "manifest": Path("artifacts/manifests/movement_intelligence_manifest.json"),
}

# Atomic writers add one of these suffixes while a file is incomplete. The
# installer and rollback logic use the same markers to remove partial writes.
TEMPORARY_SUFFIX_MARKERS = (".tmp", ".strike_tmp")


class MovementArtifactError(RuntimeError):
    """Raised when output persistence or checksum verification fails."""


def sha256(file_path: Path) -> str:
    """Return one safe regular file's hexadecimal SHA-256 checksum."""

    if not file_path.exists() or file_path.is_symlink() or not file_path.is_file():
        raise MovementArtifactError(f"Missing or unsafe artifact: {file_path}")

    # Stream large artifacts in bounded chunks so model files do not need to be
    # loaded fully into memory just to calculate a checksum.
    digest = hashlib.sha256()
    with file_path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_outputs(project_root: Path) -> dict[str, Path]:
    """Resolve every controlled output below the project root."""

    root = project_root.expanduser().resolve()
    resolved: dict[str, Path] = {}
    for name, relative_path in OUTPUT_PATHS.items():
        target = (root / relative_path).resolve()

        # A configured path must remain inside the project. This prevents an
        # accidental absolute path or ``..`` component from writing elsewhere.
        if root not in target.parents:
            raise MovementArtifactError(
                f"Output escapes project root: {relative_path}"
            )
        resolved[name] = target
    return resolved


def cleanup_temporary_outputs(project_root: Path) -> list[str]:
    """Remove incomplete temporary files for controlled outputs only."""

    removed: list[str] = []
    root = project_root.expanduser().resolve()
    for output_path in resolve_outputs(root).values():
        parent = output_path.parent
        if not parent.exists() or parent.is_symlink() or not parent.is_dir():
            continue

        # Match only the exact output filename followed by an approved marker.
        # This avoids deleting unrelated user files from the same directory.
        for candidate in parent.glob(f"{output_path.name}.*"):
            if candidate.is_symlink() or not candidate.is_file():
                continue
            if not any(marker in candidate.name for marker in TEMPORARY_SUFFIX_MARKERS):
                continue
            candidate.unlink()
            removed.append(candidate.relative_to(root).as_posix())
    return sorted(removed)


def protect_outputs(project_root: Path, replace_existing: bool) -> None:
    """Protect existing controlled outputs unless replacement is explicit."""

    # Clear only stale temporary files before checking the final output paths.
    cleanup_temporary_outputs(project_root)

    for file_path in resolve_outputs(project_root).values():
        if not file_path.exists():
            continue
        if file_path.is_symlink() or not file_path.is_file():
            raise MovementArtifactError(f"Unsafe output target: {file_path}")
        if not replace_existing:
            raise MovementArtifactError(f"Output already exists: {file_path}")

        # The installer has already backed up controlled outputs. Deleting the
        # old file here prevents stale content from surviving a shorter rewrite.
        file_path.unlink()


def _temporary_path(file_path: Path) -> Path:
    """Return a process-specific temporary path beside the final output."""

    return file_path.with_name(f"{file_path.name}.strike_tmp.{os.getpid()}")


def _atomic_bytes(file_path: Path, content: bytes) -> None:
    """Write owner-only bytes atomically and clean partial files on failure."""

    file_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(file_path)
    if temporary.exists():
        if temporary.is_symlink() or not temporary.is_file():
            raise MovementArtifactError(f"Unsafe temporary target: {temporary}")
        temporary.unlink()

    try:
        temporary.write_bytes(content)
        os.chmod(temporary, 0o600)
        temporary.replace(file_path)
        os.chmod(file_path, 0o600)
    finally:
        # ``replace`` removes the temporary path after success. This branch
        # handles serialization errors, disk failures, and interrupted writes.
        if temporary.exists() and temporary.is_file() and not temporary.is_symlink():
            temporary.unlink()


def write_json(file_path: Path, payload: Mapping[str, Any]) -> None:
    """Write one owner-only JSON object atomically."""

    content = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
    _atomic_bytes(file_path, content)


def write_csv(file_path: Path, frame: pd.DataFrame) -> None:
    """Write one owner-only CSV atomically without an index."""

    _atomic_bytes(file_path, frame.to_csv(index=False).encode("utf-8"))


def write_joblib(file_path: Path, payload: Mapping[str, Any]) -> None:
    """Write one owner-only joblib model bundle atomically."""

    file_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(file_path)
    if temporary.exists():
        if temporary.is_symlink() or not temporary.is_file():
            raise MovementArtifactError(f"Unsafe temporary target: {temporary}")
        temporary.unlink()

    try:
        joblib.dump(dict(payload), temporary)
        os.chmod(temporary, 0o600)
        temporary.replace(file_path)
        os.chmod(file_path, 0o600)
    finally:
        if temporary.exists() and temporary.is_file() and not temporary.is_symlink():
            temporary.unlink()


def load_json_object(file_path: Path, description: str) -> dict[str, Any]:
    """Load one required UTF-8 JSON object from a safe regular file."""

    if not file_path.exists() or file_path.is_symlink() or not file_path.is_file():
        raise MovementArtifactError(f"Missing or unsafe {description}: {file_path}")
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MovementArtifactError(f"Invalid {description}: {exc}") from exc
    if not isinstance(payload, dict):
        raise MovementArtifactError(f"{description} must contain a JSON object.")
    return payload


def artifact_entry(project_root: Path, file_path: Path) -> dict[str, Any]:
    """Create one checksum, size, and permission entry for an artifact."""

    root = project_root.expanduser().resolve()
    resolved = file_path.resolve()
    if root not in resolved.parents:
        raise MovementArtifactError(f"Artifact is outside project: {file_path}")
    if resolved.stat().st_mode & 0o077:
        raise MovementArtifactError(f"Artifact is not owner-only: {resolved}")
    return {
        "path": resolved.relative_to(root).as_posix(),
        "sha256": sha256(resolved),
        "size_bytes": resolved.stat().st_size,
        "mode": oct(resolved.stat().st_mode & 0o777),
    }


def verify_inventory_entries(
    project_root: Path,
    entries: list[dict[str, Any]],
    expected_paths: set[str],
) -> None:
    """Recompute one artifact inventory and require an exact path set."""

    root = project_root.expanduser().resolve()
    recorded: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise MovementArtifactError("Invalid artifact manifest entry.")
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise MovementArtifactError(f"Unsafe artifact path: {relative}")
        file_path = root / relative
        if sha256(file_path) != entry.get("sha256"):
            raise MovementArtifactError(f"Artifact checksum changed: {relative}")
        if file_path.stat().st_size != entry.get("size_bytes"):
            raise MovementArtifactError(f"Artifact size changed: {relative}")
        if oct(file_path.stat().st_mode & 0o777) != entry.get("mode"):
            raise MovementArtifactError(f"Artifact mode changed: {relative}")
        if file_path.stat().st_mode & 0o077:
            raise MovementArtifactError(f"Artifact is not owner-only: {relative}")
        recorded.add(relative.as_posix())

    if recorded != expected_paths:
        raise MovementArtifactError(
            "Artifact inventory differs from the controlled output contract."
        )


def verify_manifest(project_root: Path) -> dict[str, Any]:
    """Recompute saved checksums and required final completion markers."""

    outputs = resolve_outputs(project_root)
    manifest = load_json_object(outputs["manifest"], "combined manifest")
    if manifest.get("status") != "movement_and_intelligence_verified":
        raise MovementArtifactError("Combined manifest is not verified.")
    if manifest.get("test_used_for_selection") is not False:
        raise MovementArtifactError("Test data was used for model selection.")
    if manifest.get("explainability_started_after_model_pass") is not True:
        raise MovementArtifactError("Explainability sequencing marker is missing.")
    if manifest.get("deployment_changed") is not False:
        raise MovementArtifactError("Deployment unexpectedly changed.")

    entries = manifest.get("artifacts")
    if not isinstance(entries, list):
        raise MovementArtifactError("Combined artifact inventory is missing.")
    expected = {
        path.relative_to(project_root.resolve()).as_posix()
        for name, path in outputs.items()
        if name != "manifest"
    }
    verify_inventory_entries(project_root, entries, expected)
    return manifest
