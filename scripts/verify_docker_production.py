#!/usr/bin/env python3
"""Verify installed Docker production files and required local artifacts."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


REQUIRED_FILES = (
    ".dockerignore",
    "docker/Dockerfile.fastapi",
    "docker/Dockerfile.streamlit",
    "docker/compose.yaml",
    "docker/entrypoint-fastapi.sh",
    "docker/entrypoint-streamlit.sh",
    "docker/healthcheck.py",
    "docker/requirements-streamlit-container.txt",
    "docker/smoke_test.py",
    "docs/DOCKER_PRODUCTION_CONTRACT.md",
    "tests/test_docker_production.py",
)


def require_markers(path: Path, markers: tuple[str, ...]) -> None:
    text = path.read_text(encoding="utf-8")
    missing = [marker for marker in markers if marker not in text]
    if missing:
        raise RuntimeError(f"{path} is missing markers: {missing}")




def rendered_port_matches(entry: object, target: int, published: int) -> bool:
    """Accept Compose 2.x JSON port representations without weakening loopback policy.

    Older Docker Compose releases may serialize a long-syntax port mapping as
    either a dictionary or a normalized string. The source YAML is separately
    checked for ``host_ip: 127.0.0.1`` before this helper is used.
    """
    if isinstance(entry, dict):
        try:
            observed_target = int(str(entry.get("target", "-1")).split("/", 1)[0])
            observed_published = int(str(entry.get("published", "-1")).split("/", 1)[0])
        except ValueError:
            return False
        host_ip = entry.get("host_ip")
        return (
            observed_target == target
            and observed_published == published
            and host_ip in (None, "", "127.0.0.1")
        )

    if isinstance(entry, str):
        normalized = entry.strip()
        accepted = {
            f"127.0.0.1:{published}:{target}",
            f"127.0.0.1:{published}:{target}/tcp",
        }
        return normalized in accepted

    return False


def environment_port(
    environment: dict[str, str],
    variable: str,
    default: int,
) -> int:
    """Return a validated published port from the effective Compose environment.

    The installer deliberately chooses free ephemeral host ports and passes them
    through ``FNI_FASTAPI_PORT`` and ``FNI_STREAMLIT_PORT``. Verification must
    compare the rendered Compose configuration with those effective values, not
    with the YAML defaults.
    """
    raw_value = environment.get(variable, str(default)).strip()
    try:
        port = int(raw_value)
    except ValueError as error:
        raise RuntimeError(
            f"{variable} must be an integer TCP port; found {raw_value!r}."
        ) from error
    if not 1 <= port <= 65535:
        raise RuntimeError(
            f"{variable} must be between 1 and 65535; found {port}."
        )
    return port

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True, type=Path)
    args = parser.parse_args()
    root = args.project_root.expanduser().resolve()

    for relative in REQUIRED_FILES:
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"Required Docker file is missing or unsafe: {relative}")

    require_markers(
        root / "docker/Dockerfile.fastapi",
        (
            "FROM python:3.10.20-slim-bookworm",
            "download.pytorch.org/whl/cpu",
            "USER fni",
            "HEALTHCHECK",
        ),
    )
    require_markers(
        root / "docker/compose.yaml",
        (
            'target: 8000',
            'published: "${FNI_FASTAPI_PORT:-18000}"',
            'target: 8501',
            'published: "${FNI_STREAMLIT_PORT:-18501}"',
            'host_ip: "127.0.0.1"',
            "read_only: true",
            "no-new-privileges:true",
            "cap_drop:",
            "driver: bridge",
        ),
    )
    require_markers(
        root / ".dockerignore",
        ("data/private", ".venv-*", "**/secrets.toml", ".strike_backups"),
    )
    require_markers(
        root / "src/financial_news_intelligence/api/artifacts.py",
        ("FNI_SENTIMENT_MODEL_DIRECTORY", "configured_model_directory"),
    )

    manifest = json.loads(
        (root / "artifacts/manifests/sentiment_model_champion.json").read_text()
    )
    entries = manifest["source_evidence"]["distilbert"]["artifact_files"]
    model_root = root / "artifacts/models/distilbert_sentiment/final_model"
    for entry in entries:
        path = model_root / entry["path"]
        if not path.is_file() or path.stat().st_size != entry["size_bytes"]:
            raise RuntimeError(f"Required DistilBERT artifact is missing: {path}")

    required_runtime = (
        "artifacts/models/stock_movement/champion_model.joblib",
        "data/processed/stock_movement_model_table.csv",
        "data/processed/stock_movement_test_predictions.csv",
        "data/processed/news_sentiment_evidence.csv",
        "data/processed/market_price_evidence.csv",
    )
    for relative in required_runtime:
        if not (root / relative).is_file():
            raise RuntimeError(f"Required container runtime artifact is missing: {relative}")

    environment = os.environ.copy()
    environment.setdefault("FNI_API_KEY", "verification-only-key-1234567890")
    environment.setdefault("FNI_FASTAPI_PORT", "18000")
    environment.setdefault("FNI_STREAMLIT_PORT", "18501")
    environment.setdefault("FNI_DOCKER_IMAGE_TAG", "verification")
    completed = subprocess.run(
        ["docker", "compose", "-f", "docker/compose.yaml", "config", "--format", "json"],
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        if completed.stdout:
            print(completed.stdout)
        if completed.stderr:
            print(completed.stderr)
        raise RuntimeError("Docker Compose configuration validation failed.")

    rendered = json.loads(completed.stdout)
    expected_ports = {
        "fastapi": (
            8000,
            environment_port(environment, "FNI_FASTAPI_PORT", 18000),
        ),
        "streamlit": (
            8501,
            environment_port(environment, "FNI_STREAMLIT_PORT", 18501),
        ),
    }
    for service, (target, published) in expected_ports.items():
        entries = rendered.get("services", {}).get(service, {}).get("ports", [])
        if not any(
            rendered_port_matches(entry, target, published)
            for entry in entries
        ):
            raise RuntimeError(
                f"Rendered Compose configuration did not publish {service} "
                f"as 127.0.0.1:{published}:{target}."
            )

    print("DOCKER FILE INVENTORY: PASSED")
    print("DOCKER SECURITY CONTRACT: PASSED")
    print("DOCKER BUILD CONTEXT CONTRACT: PASSED")
    print("DOCKER RUNTIME ARTIFACT PREREQUISITES: PASSED")
    print("DOCKER COMPOSE CONFIGURATION: PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
