#!/usr/bin/env python3
"""Verify pinned FastAPI and model-runtime dependencies before installation.

Purpose
-------
Confirm that the existing project Python environments contain the exact
versions required by the audited project lock files. The main analytics
runtime and DistilBERT runtime are checked in separate processes so native
libraries are never mixed during verification.

Failure behavior
----------------
This script does not install or modify packages. It fails closed and reports
what failed, where it failed, why it failed, and the exact safe next step.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import io
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


MAIN_REQUIREMENTS: dict[str, tuple[str, str, str]] = {
    "numpy": ("1.26.4", "numpy", "project"),
    "pandas": ("2.1.4", "pandas", "project"),
    "scikit-learn": ("1.5.0", "sklearn", "project"),
    "joblib": ("1.4.2", "joblib", "project"),
    "pytest": ("8.3.4", "pytest", "project_or_base"),
    "fastapi": ("0.115.6", "fastapi", "project_or_base"),
    "starlette": ("0.41.3", "starlette", "project_or_base"),
    "pydantic": ("2.10.4", "pydantic", "project_or_base"),
    "uvicorn": ("0.34.0", "uvicorn", "project_or_base"),
    "requests": ("2.32.3", "requests", "project_or_base"),
    "beautifulsoup4": ("4.12.3", "bs4", "project_or_base"),
    "pypdf": ("5.1.0", "pypdf", "project_or_base"),
    "httpx": ("0.28.1", "httpx", "project_or_base"),
}
TRANSFORMER_REQUIREMENTS: dict[str, tuple[str, str, str]] = {
    "numpy": ("1.26.4", "numpy", "project"),
    "torch": ("2.2.2", "torch", "project"),
    "transformers": ("4.46.3", "transformers", "project"),
    "tokenizers": ("0.20.3", "tokenizers", "project"),
    "safetensors": ("0.8.0", "safetensors", "project"),
}


class DependencyVerificationError(RuntimeError):
    """Describe one missing, incompatible, or misplaced dependency."""


@dataclass(frozen=True)
class DependencyEvidence:
    """One verified distribution and its import origin."""

    distribution: str
    version: str
    module: str
    origin: str
    origin_scope: str


def parse_args() -> argparse.Namespace:
    """Parse one project root, environment name, profile, and evidence file."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--environment-name", required=True)
    parser.add_argument("--profile", required=True, choices=("main", "transformer"))
    parser.add_argument("--evidence-file", type=Path)
    return parser.parse_args()


def absolute_path(path: Path) -> Path:
    """Return an absolute path without dereferencing virtual-environment links."""

    return Path(os.path.abspath(os.fspath(path.expanduser())))


def is_within(candidate: Path, parent: Path) -> bool:
    """Return true when one resolved path is inside another resolved path."""

    try:
        candidate.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def require_environment(project_root: Path, environment_name: str) -> Path:
    """Require this process to run inside the named project environment."""

    expected = absolute_path(project_root / environment_name)
    observed = absolute_path(Path(sys.prefix))
    if not expected.exists() or not expected.is_dir():
        raise DependencyVerificationError(
            f"Expected environment directory is missing: {expected}"
        )
    try:
        matches = os.path.samefile(expected, observed)
    except OSError:
        matches = expected.resolve() == observed.resolve()
    if not matches:
        raise DependencyVerificationError(
            "The dependency verifier is not running inside the required "
            f"environment. Expected {expected}; observed {observed}."
        )
    return expected


def distribution_evidence(
    distribution_name: str,
    expected_version: str,
    module_name: str,
    origin_rule: str,
    environment_root: Path,
) -> DependencyEvidence:
    """Verify one exact version and its import origin without importing it."""

    try:
        actual_version = importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError as error:
        raise DependencyVerificationError(
            f"Required package is not installed: {distribution_name}=={expected_version}"
        ) from error
    if actual_version != expected_version:
        raise DependencyVerificationError(
            f"Package version mismatch for {distribution_name}. "
            f"Expected {expected_version}; found {actual_version}."
        )
    specification = importlib.util.find_spec(module_name)
    if specification is None or specification.origin is None:
        raise DependencyVerificationError(
            f"Import module could not be located: {module_name}"
        )
    origin = Path(specification.origin).resolve()
    base_root = Path(sys.base_prefix).resolve()
    in_project = is_within(origin, environment_root)
    in_base = is_within(origin, base_root)
    if origin_rule == "project" and not in_project:
        raise DependencyVerificationError(
            f"{distribution_name} is outside the project environment. "
            f"Origin: {origin}"
        )
    if origin_rule == "project_or_base" and not (in_project or in_base):
        raise DependencyVerificationError(
            f"{distribution_name} is outside the approved environment roots. "
            f"Origin: {origin}"
        )
    scope = "project_environment" if in_project else "base_environment"
    return DependencyEvidence(
        distribution=distribution_name,
        version=actual_version,
        module=module_name,
        origin=str(origin),
        origin_scope=scope,
    )


def verify_pdf_parser() -> None:
    """Create and read one in-memory PDF to prove PDF input is usable."""

    pypdf = importlib.import_module("pypdf")
    buffer = io.BytesIO()
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.write(buffer)
    buffer.seek(0)
    reader = pypdf.PdfReader(buffer)
    if len(reader.pages) != 1:
        raise DependencyVerificationError(
            "The pypdf functional probe did not return one page."
        )


def verify_transformer_imports() -> None:
    """Import transformer libraries only inside the transformer process."""

    for module_name in ("numpy", "torch", "transformers", "tokenizers", "safetensors"):
        importlib.import_module(module_name)


def write_evidence(file_path: Path, payload: Mapping[str, Any]) -> None:
    """Write owner-only JSON evidence atomically."""

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


def verify_dependencies(
    project_root: Path,
    environment_name: str,
    profile: str,
) -> dict[str, Any]:
    """Verify one isolated dependency profile and return auditable evidence."""

    environment_root = require_environment(project_root, environment_name)
    requirements = MAIN_REQUIREMENTS if profile == "main" else TRANSFORMER_REQUIREMENTS
    evidence = [
        distribution_evidence(name, version, module, rule, environment_root)
        for name, (version, module, rule) in requirements.items()
    ]
    if profile == "main":
        verify_pdf_parser()
    else:
        verify_transformer_imports()
    base_scoped = [item.distribution for item in evidence if item.origin_scope == "base_environment"]
    return {
        "status": "dependencies_verified",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "profile": profile,
        "environment_name": environment_name,
        "python_executable": sys.executable,
        "python_prefix": sys.prefix,
        "python_base_prefix": sys.base_prefix,
        "dependencies": [item.__dict__ for item in evidence],
        "base_environment_dependencies": base_scoped,
        "pdf_parser_functional_probe": profile == "main",
        "transformer_import_probe": profile == "transformer",
    }


def main() -> int:
    """Run dependency verification and print consistent status markers."""

    args = parse_args()
    root = args.project_root.expanduser().resolve()
    print(f"STARTED: {args.profile} dependency verification.", flush=True)
    try:
        evidence = verify_dependencies(root, args.environment_name, args.profile)
        base_scoped = evidence["base_environment_dependencies"]
        if base_scoped:
            print(
                "WARNING: Exact-version web dependencies are provided by the "
                "approved base Python environment: " + ", ".join(base_scoped),
                flush=True,
            )
        if args.evidence_file:
            write_evidence(args.evidence_file, evidence)
            print(f"PASSED: Dependency evidence written to {args.evidence_file}.")
        print(f"FASTAPI {args.profile.upper()} DEPENDENCY VERIFICATION: PASSED")
        return 0
    except Exception as error:  # noqa: BLE001 - must report exact dependency failure.
        print("FAILED: FastAPI dependency verification failed.", file=sys.stderr)
        print(f"Location: {args.environment_name} dependency profile", file=sys.stderr)
        print(f"Reason: {type(error).__name__}: {error}", file=sys.stderr)
        print(
            "Safe next step: Install the exact pinned package in the named "
            "project environment, then rerun this package.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
