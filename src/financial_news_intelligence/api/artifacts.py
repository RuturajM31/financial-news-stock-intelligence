"""Verify model, manifest, and intelligence artifacts before API use.

Purpose
-------
The API may only load artifacts recorded by the completed sentiment,
foundation, and movement phases. Every path stays inside the project root and
every recorded SHA-256 checksum is recomputed before use.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .errors import ApiProblem
from .runtime_environment import environment_python


@dataclass(frozen=True)
class ArtifactPaths:
    """Verified artifact locations consumed by API services."""

    sentiment_model_directory: Path
    sentiment_python: Path
    movement_model: Path
    movement_model_table: Path
    movement_test_predictions: Path
    foundation_news: Path
    foundation_prices: Path
    global_drivers: Path
    sentiment_phrases: Path
    provenance: Path
    movement_manifest: Path
    foundation_manifest: Path
    sentiment_champion_manifest: Path


def sha256(file_path: Path) -> str:
    """Return the SHA-256 checksum for one safe regular file."""

    if not file_path.exists() or file_path.is_symlink() or not file_path.is_file():
        raise ApiProblem(
            503,
            "artifact_missing",
            "Artifact verification failed.",
            str(file_path),
            "A required artifact is missing, unsafe, or not a regular file.",
            "Restore the verified project artifact and rerun API verification.",
        )
    digest = hashlib.sha256()
    with file_path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_path(project_root: Path, relative_path: str) -> Path:
    """Resolve one relative artifact path below the project root."""

    candidate = Path(relative_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ApiProblem(
            503,
            "artifact_path_unsafe",
            "Artifact verification failed.",
            "Artifact manifest path",
            f"The recorded path is unsafe: {relative_path}",
            "Restore the verified manifest and rerun API verification.",
        )
    target = (project_root / candidate).resolve()
    if project_root != target and project_root not in target.parents:
        raise ApiProblem(
            503,
            "artifact_path_escape",
            "Artifact verification failed.",
            "Artifact manifest path",
            f"The path escapes the project root: {relative_path}",
            "Restore the verified manifest and rerun API verification.",
        )
    return target


def _load_json(file_path: Path, description: str) -> dict[str, Any]:
    """Load one required JSON object from a safe regular file."""

    sha256(file_path)
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ApiProblem(
            503,
            "manifest_invalid_json",
            "Artifact verification failed.",
            description,
            f"The JSON content is invalid: {exc}",
            "Restore the verified manifest and rerun API verification.",
        ) from exc
    if not isinstance(payload, dict):
        raise ApiProblem(
            503,
            "manifest_not_object",
            "Artifact verification failed.",
            description,
            "The manifest does not contain one JSON object.",
            "Restore the verified manifest and rerun API verification.",
        )
    return payload


def _verify_inventory(
    project_root: Path,
    entries: Any,
    description: str,
) -> dict[str, Path]:
    """Verify every manifest inventory entry and return paths by name."""

    if not isinstance(entries, list) or not entries:
        raise ApiProblem(
            503,
            "artifact_inventory_missing",
            "Artifact verification failed.",
            description,
            "The artifact inventory is missing or empty.",
            "Restore the verified manifest and rerun API verification.",
        )
    verified: dict[str, Path] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ApiProblem(
                503,
                "artifact_entry_invalid",
                "Artifact verification failed.",
                description,
                "An artifact entry is not a JSON object.",
                "Restore the verified manifest and rerun API verification.",
            )
        relative_path = entry.get("path")
        expected_checksum = entry.get("sha256")
        expected_size = entry.get("size_bytes")
        if not isinstance(relative_path, str) or not isinstance(
            expected_checksum, str
        ):
            raise ApiProblem(
                503,
                "artifact_entry_incomplete",
                "Artifact verification failed.",
                description,
                "An artifact entry is missing its path or checksum.",
                "Restore the verified manifest and rerun API verification.",
            )
        file_path = _safe_path(project_root, relative_path)
        actual_checksum = sha256(file_path)
        if actual_checksum != expected_checksum:
            raise ApiProblem(
                503,
                "artifact_checksum_changed",
                "Artifact verification failed.",
                relative_path,
                "The artifact checksum differs from the verified manifest.",
                "Restore the verified artifact and rerun API verification.",
            )
        if isinstance(expected_size, int) and file_path.stat().st_size != expected_size:
            raise ApiProblem(
                503,
                "artifact_size_changed",
                "Artifact verification failed.",
                relative_path,
                "The artifact size differs from the verified manifest.",
                "Restore the verified artifact and rerun API verification.",
            )
        if file_path.stat().st_mode & 0o077:
            raise ApiProblem(
                503,
                "artifact_permissions_unsafe",
                "Artifact verification failed.",
                relative_path,
                "The artifact is readable or writable by other operating-system users.",
                "Restore owner-only permissions and rerun API verification.",
            )
        verified[relative_path] = file_path
    return verified


class ArtifactRegistry:
    """Resolve and verify every artifact required by the API."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root.expanduser().resolve()

    def verify(self) -> ArtifactPaths:
        """Verify all recorded artifacts and return controlled paths."""

        foundation_manifest_path = _safe_path(
            self.project_root,
            "artifacts/manifests/market_data_foundation_manifest.json",
        )
        movement_manifest_path = _safe_path(
            self.project_root,
            "artifacts/manifests/movement_intelligence_manifest.json",
        )
        champion_manifest_path = _safe_path(
            self.project_root,
            "artifacts/manifests/sentiment_model_champion.json",
        )
        foundation = _load_json(foundation_manifest_path, "foundation manifest")
        movement = _load_json(movement_manifest_path, "movement manifest")
        champion = _load_json(champion_manifest_path, "sentiment champion manifest")

        if foundation.get("status") != "foundation_verified":
            raise ApiProblem(
                503,
                "foundation_not_verified",
                "API readiness failed.",
                "Market Data Foundation manifest",
                "The foundation status is not foundation_verified.",
                "Rerun the foundation verifier before starting FastAPI.",
            )
        if movement.get("status") != "movement_and_intelligence_verified":
            raise ApiProblem(
                503,
                "movement_not_verified",
                "API readiness failed.",
                "Movement Intelligence manifest",
                "Movement and intelligence status is not verified.",
                "Rerun independent movement verification before starting FastAPI.",
            )
        if movement.get("quality_champion") != "stability_soft_vote_rf_sgd":
            raise ApiProblem(
                503,
                "movement_champion_changed",
                "API readiness failed.",
                "Movement Intelligence manifest",
                "The verified movement champion name changed.",
                "Review the new model evidence before changing API deployment.",
            )
        if movement.get("deployment_changed") is not False:
            raise ApiProblem(
                503,
                "unexpected_deployment_change",
                "API readiness failed.",
                "Movement Intelligence manifest",
                "The movement package unexpectedly reports a deployment change.",
                "Stop and audit the movement artifacts before starting FastAPI.",
            )
        if movement.get("public_deployment_authorized") is not False:
            raise ApiProblem(
                503,
                "public_boundary_changed",
                "API readiness failed.",
                "Movement Intelligence manifest",
                "The public deployment boundary changed unexpectedly.",
                "Stop and complete a separate licensing review.",
            )
        if champion.get("recommended_deployment_model") != "distilbert":
            raise ApiProblem(
                503,
                "sentiment_champion_changed",
                "API readiness failed.",
                "Sentiment champion manifest",
                "DistilBERT is no longer the recorded deployment model.",
                "Review model-comparison evidence before changing API deployment.",
            )

        foundation_paths = _verify_inventory(
            self.project_root,
            foundation.get("artifacts"),
            "Foundation artifact inventory",
        )
        movement_paths = _verify_inventory(
            self.project_root,
            movement.get("artifacts"),
            "Movement artifact inventory",
        )

        comparison_path = _safe_path(
            self.project_root,
            "reports/metrics/sentiment_model_comparison.json",
        )
        comparison = _load_json(comparison_path, "sentiment comparison report")
        if comparison.get("deployment_champion") != "distilbert":
            raise ApiProblem(
                503,
                "sentiment_evidence_disagrees",
                "API readiness failed.",
                "Sentiment comparison report",
                "The comparison report and champion manifest disagree.",
                "Restore the verified sentiment comparison artifacts.",
            )
        models = comparison.get("models")
        selected = next(
            (
                item
                for item in models or []
                if isinstance(item, Mapping) and item.get("model_key") == "distilbert"
            ),
            None,
        )
        if not isinstance(selected, Mapping):
            raise ApiProblem(
                503,
                "sentiment_model_evidence_missing",
                "API readiness failed.",
                "Sentiment comparison report",
                "DistilBERT model evidence is missing.",
                "Restore the verified sentiment comparison report.",
            )
        model_directory_value = selected.get("final_model_directory")
        if not isinstance(model_directory_value, str):
            raise ApiProblem(
                503,
                "sentiment_model_path_missing",
                "API readiness failed.",
                "Sentiment comparison report",
                "The DistilBERT model directory is missing.",
                "Restore the verified sentiment comparison report.",
            )
        configured_model_directory = os.getenv("FNI_SENTIMENT_MODEL_DIRECTORY")
        if configured_model_directory:
            # Container and orchestrated deployments use a relative, project-owned
            # path because recorded training evidence contains the original local
            # workstation path. _safe_path rejects absolute paths and traversal.
            model_directory = _safe_path(
                self.project_root, configured_model_directory.strip()
            )
        else:
            model_directory = Path(model_directory_value).expanduser().resolve()
        if self.project_root not in model_directory.parents:
            raise ApiProblem(
                503,
                "sentiment_model_path_unsafe",
                "API readiness failed.",
                "DistilBERT model directory",
                "The model directory is outside the project root.",
                "Restore the verified local DistilBERT model directory.",
            )
        if not model_directory.exists() or not model_directory.is_dir():
            raise ApiProblem(
                503,
                "sentiment_model_directory_missing",
                "API readiness failed.",
                "DistilBERT model directory",
                "The verified model directory does not exist.",
                "Restore the DistilBERT artifacts and rerun API verification.",
            )

        source_evidence = champion.get("source_evidence")
        distilbert_evidence = (
            source_evidence.get("distilbert")
            if isinstance(source_evidence, Mapping)
            else None
        )
        expected_model_files = (
            distilbert_evidence.get("artifact_files")
            if isinstance(distilbert_evidence, Mapping)
            else None
        )
        if not isinstance(expected_model_files, list):
            raise ApiProblem(
                503,
                "sentiment_inventory_missing",
                "API readiness failed.",
                "Sentiment champion manifest",
                "The DistilBERT artifact inventory is missing.",
                "Restore the verified sentiment champion manifest.",
            )
        for entry in expected_model_files:
            if not isinstance(entry, Mapping) or not isinstance(entry.get("path"), str):
                raise ApiProblem(
                    503,
                    "sentiment_inventory_invalid",
                    "API readiness failed.",
                    "Sentiment champion manifest",
                    "A DistilBERT artifact entry is invalid.",
                    "Restore the verified sentiment champion manifest.",
                )
            file_path = (model_directory / str(entry["path"])).resolve()
            if model_directory not in file_path.parents:
                raise ApiProblem(
                    503,
                    "sentiment_artifact_path_unsafe",
                    "API readiness failed.",
                    "DistilBERT artifact inventory",
                    "A model file path escapes the verified model directory.",
                    "Restore the verified sentiment champion manifest.",
                )
            if sha256(file_path) != entry.get("sha256"):
                raise ApiProblem(
                    503,
                    "sentiment_artifact_checksum_changed",
                    "API readiness failed.",
                    str(entry["path"]),
                    "A DistilBERT artifact checksum changed.",
                    "Restore the verified DistilBERT model artifacts.",
                )
            if file_path.stat().st_size != entry.get("size_bytes"):
                raise ApiProblem(
                    503,
                    "sentiment_artifact_size_changed",
                    "API readiness failed.",
                    str(entry["path"]),
                    "A DistilBERT artifact size changed.",
                    "Restore the verified DistilBERT model artifacts.",
                )

        try:
            sentiment_python = environment_python(
                self.project_root,
                ".venv-distilbert",
            )
        except Exception as error:
            raise ApiProblem(
                503,
                "sentiment_runtime_missing",
                "API readiness failed.",
                ".venv-distilbert/bin/python",
                str(error),
                "Restore the verified transformer environment before starting FastAPI.",
            ) from error

        required_movement = {
            "artifacts/models/stock_movement/champion_model.joblib": "movement_model",
            "data/processed/stock_movement_model_table.csv": "movement_model_table",
            (
                "data/processed/stock_movement_test_predictions.csv"
            ): "movement_test_predictions",
            "reports/explainability/movement_global_drivers.csv": "global_drivers",
            "reports/intelligence/sentiment_phrases.csv": "sentiment_phrases",
            "reports/intelligence/provenance_and_disclaimers.json": "provenance",
        }
        missing = sorted(set(required_movement) - set(movement_paths))
        if missing:
            raise ApiProblem(
                503,
                "movement_artifacts_missing",
                "API readiness failed.",
                "Movement artifact inventory",
                f"Required API artifacts are missing: {missing}",
                "Restore the verified movement intelligence artifacts.",
            )

        return ArtifactPaths(
            sentiment_model_directory=model_directory,
            sentiment_python=sentiment_python,
            movement_model=movement_paths[
                "artifacts/models/stock_movement/champion_model.joblib"
            ],
            movement_model_table=movement_paths[
                "data/processed/stock_movement_model_table.csv"
            ],
            movement_test_predictions=movement_paths[
                "data/processed/stock_movement_test_predictions.csv"
            ],
            foundation_news=foundation_paths[
                "data/processed/news_sentiment_evidence.csv"
            ],
            foundation_prices=foundation_paths[
                "data/processed/market_price_evidence.csv"
            ],
            global_drivers=movement_paths[
                "reports/explainability/movement_global_drivers.csv"
            ],
            sentiment_phrases=movement_paths[
                "reports/intelligence/sentiment_phrases.csv"
            ],
            provenance=movement_paths[
                "reports/intelligence/provenance_and_disclaimers.json"
            ],
            movement_manifest=movement_manifest_path,
            foundation_manifest=foundation_manifest_path,
            sentiment_champion_manifest=champion_manifest_path,
        )
