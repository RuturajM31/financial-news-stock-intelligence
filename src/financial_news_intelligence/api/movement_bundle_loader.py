"""Load the verified movement bundle before the full movement runtime starts.

Purpose
-------
The audited movement model is stored as one joblib bundle. The real Mac
diagnostic proved that the bundle loads successfully through a
minimal joblib path with the filesystem-level pyarrow shim active. This module
preserves that proven path before the larger movement runtime is imported. It
validates saved versions from package metadata and does not import pyarrow.

Failure behavior
----------------
The loader fails closed. It does not repair, rewrite, or retrain the model. A
load failure names the artifact, exception type, safe exception detail, and the
exact next verification step.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import platform
import sys
from pathlib import Path
from typing import Any, Mapping

from .errors import ApiProblem


EXPECTED_LABEL_ORDER = ["Down", "Flat", "Up"]
REQUIRED_BUNDLE_KEYS = {
    "pipeline",
    "champion_name",
    "numeric_features",
    "categorical_features",
    "label_order",
    "runtime_versions",
    "decision_policy",
}
RUNTIME_DISTRIBUTIONS = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scikit_learn": "scikit-learn",
    "joblib": "joblib",
}
MAXIMUM_EXCEPTION_DETAIL_CHARACTERS = 400


def _safe_exception_detail(error: Exception) -> str:
    """Return bounded one-line exception detail without logging object data."""

    detail = " ".join(str(error).split())
    if not detail:
        return "No additional exception detail was provided."
    return detail[:MAXIMUM_EXCEPTION_DETAIL_CHARACTERS]


def _installed_runtime_versions() -> dict[str, str]:
    """Read versions and require every distribution to come from this environment.

    The project movement artifact was trained in ``.venv``. A base pyenv
    interpreter can expose packages with the same names but incompatible
    internal classes. Distribution locations are therefore checked before
    joblib deserialization starts.
    """

    versions = {"python": platform.python_version()}
    environment_root = Path(sys.prefix).expanduser().resolve()
    for contract_name, distribution_name in RUNTIME_DISTRIBUTIONS.items():
        try:
            distribution = importlib.metadata.distribution(distribution_name)
        except importlib.metadata.PackageNotFoundError as error:
            raise ApiProblem(
                503,
                "movement_runtime_dependency_missing",
                "Movement model loading failed.",
                f"Python distribution: {distribution_name}",
                "The dependency recorded by the movement model is not installed.",
                "Restore the verified project .venv and rerun FastAPI verification.",
            ) from error
        distribution_root = Path(distribution.locate_file("")).expanduser().resolve()
        if (
            distribution_root != environment_root
            and environment_root not in distribution_root.parents
        ):
            raise ApiProblem(
                503,
                "movement_runtime_dependency_outside_venv",
                "Movement model loading failed.",
                f"Python distribution: {distribution_name}",
                (
                    "The dependency was loaded from outside the active virtual "
                    f"environment. Environment: {environment_root}; "
                    f"distribution: {distribution_root}."
                ),
                (
                    "Start the worker with the project .venv/bin/python launcher "
                    "without resolving that symlink, then rerun verification."
                ),
            )
        versions[contract_name] = distribution.version
    return versions


def validate_movement_bundle(bundle: Any, model_path: Path) -> dict[str, Any]:
    """Validate bundle structure, class order, and training runtime versions."""

    if not isinstance(bundle, dict):
        raise ApiProblem(
            503,
            "movement_bundle_not_object",
            "Movement model loading failed.",
            str(model_path),
            "The joblib artifact did not contain one dictionary bundle.",
            "Restore the verified movement model artifact.",
        )

    missing_keys = sorted(REQUIRED_BUNDLE_KEYS - set(bundle))
    if missing_keys:
        raise ApiProblem(
            503,
            "movement_bundle_incomplete",
            "Movement model loading failed.",
            str(model_path),
            f"The model bundle is missing required keys: {missing_keys}.",
            "Restore the verified movement model artifact.",
        )

    if bundle.get("label_order") != EXPECTED_LABEL_ORDER:
        raise ApiProblem(
            503,
            "movement_label_order_changed",
            "Movement model loading failed.",
            str(model_path),
            "The saved movement class order is not Down, Flat, Up.",
            "Restore the verified movement model artifact.",
        )

    recorded_versions = bundle.get("runtime_versions")
    if not isinstance(recorded_versions, Mapping):
        raise ApiProblem(
            503,
            "movement_runtime_evidence_missing",
            "Movement model loading failed.",
            str(model_path),
            "The saved runtime-version evidence is missing or invalid.",
            "Restore the verified movement model artifact.",
        )

    installed_versions = _installed_runtime_versions()
    changed_versions = {
        name: {
            "expected": recorded_versions.get(name),
            "actual": installed_version,
        }
        for name, installed_version in installed_versions.items()
        if recorded_versions.get(name) != installed_version
    }
    if changed_versions:
        raise ApiProblem(
            503,
            "movement_runtime_changed",
            "Movement model loading failed.",
            "Main analytics Python environment",
            f"Runtime versions differ from training: {changed_versions}.",
            "Run FastAPI with the verified project .venv environment.",
        )

    return bundle


def load_verified_movement_bundle(model_path: Path) -> dict[str, Any]:
    """Load and validate the joblib bundle before the full movement runtime imports."""

    resolved_path = model_path.expanduser().resolve()
    if (
        not resolved_path.exists()
        or resolved_path.is_symlink()
        or not resolved_path.is_file()
    ):
        raise ApiProblem(
            503,
            "movement_model_missing",
            "Movement model loading failed.",
            str(resolved_path),
            "The verified joblib artifact is missing or is not a safe regular file.",
            "Restore the verified movement model artifact and rerun verification.",
        )

    # Validate package origin before importing joblib or deserializing the
    # scikit-learn pipeline. This catches a base-interpreter launch before an
    # incompatible private sklearn class is requested from the artifact.
    _installed_runtime_versions()
    try:
        joblib = importlib.import_module("joblib")
        bundle = joblib.load(resolved_path)
    except Exception as error:  # noqa: BLE001 - exact artifact stage is required.
        raise ApiProblem(
            503,
            "movement_model_load_failed",
            "Movement model loading failed.",
            str(resolved_path),
            (
                "The joblib artifact could not be loaded. "
                f"Exception: {type(error).__name__}: "
                f"{_safe_exception_detail(error)}"
            ),
            (
                "Run the read-only joblib diagnostic with the verified .venv. "
                "If it passes, keep bundle loading before pandas imports."
            ),
        ) from error

    return validate_movement_bundle(bundle, resolved_path)
