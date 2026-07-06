#!/usr/bin/env python3
"""Verify FastAPI endpoints, artifacts, and isolated model workers.

This verifier never imports NumPy, pandas, scikit-learn, or PyTorch into the
coordinator process. Each model family runs behind its own subprocess boundary.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    """Parse the project root and optional evidence output."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--evidence-file", type=Path)
    parser.add_argument("--skip-sentiment-probe", action="store_true")
    return parser.parse_args()


def write_evidence(file_path: Path, payload: dict[str, Any]) -> None:
    """Write owner-only verification evidence atomically."""

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
    """Run read-only verification and report every tested boundary."""

    args = parse_args()
    root = args.project_root.expanduser().resolve()
    from financial_news_intelligence.api.runtime_environment import (
        RuntimeEnvironmentError,
        environment_python,
        require_active_environment,
    )

    try:
        expected_python = environment_python(root, ".venv")
        require_active_environment(root, ".venv")
    except RuntimeEnvironmentError as error:
        print(f"FAILED: {error.what_failed}")
        print(f"Location: {error.where_failed}")
        print(f"Reason: {error.why_failed}")
        print(f"Safe next step: {error.safe_next_step}")
        return 1

    movement_client = None
    sentiment_client = None
    try:
        from financial_news_intelligence.api.app import create_app
        from financial_news_intelligence.api.artifacts import ArtifactRegistry
        from financial_news_intelligence.api.config import ApiSettings
        from financial_news_intelligence.api.movement_worker_client import (
            MovementWorkerClient,
        )
        from financial_news_intelligence.api.sentiment_worker_client import (
            SentimentWorkerClient,
        )

        print("STARTED: FastAPI verification.", flush=True)
        paths = ArtifactRegistry(root).verify()
        print("PASSED: Artifact checksums and deployment boundaries.", flush=True)

        movement_client = MovementWorkerClient(
            project_root=root,
            python_executable=expected_python,
            timeout_seconds=180,
        )
        movement_probe = movement_client.readiness()
        print("PASSED: Isolated movement worker and saved prediction.", flush=True)

        sentiment_probe: dict[str, Any] | None = None
        if args.skip_sentiment_probe:
            print("SKIPPED: Isolated DistilBERT inference probe.", flush=True)
        else:
            sentiment_client = SentimentWorkerClient(
                project_root=root,
                python_executable=paths.sentiment_python,
                model_directory=paths.sentiment_model_directory,
                timeout_seconds=120,
            )
            sentiment_probe = sentiment_client.predict(
                ["The company filed its quarterly financial report."]
            )[0]
            print("PASSED: Isolated DistilBERT inference probe.", flush=True)

        class OpenApiOnlyServices:
            """Minimal injected services; OpenAPI generation calls no endpoint."""

            def close(self) -> None:
                return None

        settings = ApiSettings(
            project_root=root,
            environment="test",
            require_api_key=False,
            trusted_hosts=("testserver", "localhost", "127.0.0.1"),
        )
        application = create_app(
            settings=settings,
            services=OpenApiOnlyServices(),
        )
        openapi = application.openapi()
        paths_found = set(openapi.get("paths", {}))
        required_paths = {
            "/health",
            "/ready",
            "/v1/sentiment/text",
            "/v1/sentiment/file",
            "/v1/sentiment/url",
            "/v1/movement/predict",
            "/v1/intelligence/historical",
            "/v1/explainability",
            "/v1/scenarios/analyze",
            "/v1/provenance",
        }
        missing_paths = sorted(required_paths - paths_found)
        if missing_paths:
            raise RuntimeError(f"OpenAPI paths are missing: {missing_paths}")
        print("PASSED: OpenAPI endpoint contract.", flush=True)

        evidence = {
            "status": "fastapi_verified",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "application_version": openapi.get("info", {}).get("version"),
            "endpoint_count": len(paths_found),
            "required_endpoints": sorted(required_paths),
            "movement_probe": movement_probe,
            "sentiment_probe": sentiment_probe,
            "movement_process_isolated": True,
            "sentiment_process_isolated": True,
            "web_process_loaded_native_model_libraries": False,
            "deployment_changed": False,
            "public_deployment_authorized": False,
            "limitations": [
                "Movement prediction supports verified historical-audit sessions only.",
                "The verified market evidence ends in 2020.",
                "Public deployment still requires licensing and security closure.",
            ],
        }
        if args.evidence_file:
            write_evidence(args.evidence_file, evidence)
            print(f"PASSED: Evidence written to {args.evidence_file}.", flush=True)
        print("FASTAPI APPLICATION VERIFICATION: PASSED", flush=True)
        return 0
    except Exception as error:  # noqa: BLE001 - verifier must name exact stage.
        print("FAILED: FastAPI application verification failed.", file=sys.stderr)
        print("Location: FastAPI application verifier", file=sys.stderr)
        print(f"Reason: {type(error).__name__}: {error}", file=sys.stderr)
        print(
            "Safe next step: Restore verified artifacts or correct the named "
            "worker boundary, then rerun verification.",
            file=sys.stderr,
        )
        return 1
    finally:
        if sentiment_client is not None:
            sentiment_client.close()
        if movement_client is not None:
            movement_client.close()


if __name__ == "__main__":
    raise SystemExit(main())
