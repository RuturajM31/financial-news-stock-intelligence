#!/usr/bin/env python3
"""Run movement and intelligence inference as a JSON-lines worker.

The worker is the only FastAPI component that imports NumPy, pandas, joblib,
and scikit-learn. If a native library crashes, the web process remains alive
and reports a structured failure.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping


def parse_args() -> argparse.Namespace:
    """Parse the verified project root."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True, type=Path)
    return parser.parse_args()


def json_safe(value: Any) -> Any:
    """Convert model-library scalar and date values into JSON-safe values."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("A worker result contains a non-finite number.")
        return value
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        return json_safe(value.item())
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def parse_datetime(value: Any) -> datetime:
    """Require one timezone-aware ISO 8601 timestamp."""

    if not isinstance(value, str):
        raise ValueError("published_at must be an ISO 8601 string.")
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("published_at must include a timezone.")
    return parsed


def failure_from_exception(error: Exception) -> dict[str, Any]:
    """Return a safe worker failure while preserving structured ApiProblem data."""

    from financial_news_intelligence.api.errors import ApiProblem
    from financial_news_intelligence.api.runtime_environment import (
        RuntimeEnvironmentError,
    )

    if isinstance(error, RuntimeEnvironmentError):
        return {
            "status": "FAILED",
            "status_code": 503,
            "error_code": "movement_runtime_environment_invalid",
            "what_failed": error.what_failed,
            "where_failed": error.where_failed,
            "why_failed": error.why_failed,
            "safe_next_step": error.safe_next_step,
        }
    if isinstance(error, ApiProblem):
        return {
            "status": "FAILED",
            "status_code": error.status_code,
            "error_code": error.error_code,
            "what_failed": error.what_failed,
            "where_failed": error.where_failed,
            "why_failed": error.why_failed,
            "safe_next_step": error.safe_next_step,
        }
    return {
        "status": "FAILED",
        "status_code": 503,
        "error_code": "movement_worker_internal_failure",
        "what_failed": "Movement and intelligence processing failed.",
        "where_failed": "Isolated movement worker",
        "why_failed": f"An unexpected {type(error).__name__} occurred.",
        "safe_next_step": "Run the FastAPI verifier and inspect verified artifacts.",
    }


def require_payload(request: Any) -> tuple[str, dict[str, Any]]:
    """Require one command and one object payload."""

    if not isinstance(request, dict):
        raise ValueError("The worker request must be one JSON object.")
    command = request.get("command")
    payload = request.get("payload", {})
    if not isinstance(command, str) or not command:
        raise ValueError("The worker command is missing.")
    if not isinstance(payload, dict):
        raise ValueError("The worker payload must be one JSON object.")
    return command, payload


def prediction_result(
    movement: Any,
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any], Any, Any]:
    """Build one prediction from validated worker inputs."""

    text = str(payload.get("text", "")).strip()
    ticker = str(payload.get("ticker", "")).strip().upper()
    sentiment = payload.get("sentiment")
    if not text or not ticker or not isinstance(sentiment, Mapping):
        raise ValueError("Prediction text, ticker, and sentiment are required.")
    prediction, frame, target_date = movement.predict(
        text,
        ticker,
        parse_datetime(payload.get("published_at")),
        sentiment,
    )
    return prediction, frame, target_date


def handle_request(
    command: str,
    payload: dict[str, Any],
    movement: Any,
    intelligence: Any,
) -> dict[str, Any]:
    """Execute one supported worker command without hidden side effects."""

    if command == "readiness":
        return {
            "movement_probe": movement.verify_saved_prediction(),
            "champion_model": movement.champion_name,
            "provenance_status": intelligence.provenance.get("status"),
        }
    if command == "provenance":
        return {"provenance": dict(intelligence.provenance)}

    prediction, frame, target_date = prediction_result(movement, payload)
    common = {
        "prediction": prediction,
        "target_session_date": target_date.date(),
        "champion_model": movement.champion_name,
    }
    if command == "predict":
        return common
    if command == "historical":
        result = intelligence.historical_matches(
            text=str(payload["text"]),
            ticker=str(payload["ticker"]).upper(),
            cutoff=target_date,
            limit=int(payload.get("limit", 5)),
            minimum_similarity=float(payload.get("minimum_similarity", 0.0)),
            sentiment_label=str(payload["sentiment"]["label"]),
        )
        return {**common, **result}
    if command == "explain":
        top_n = int(payload.get("top_n", 5))
        return {
            **common,
            "global_drivers": movement.global_drivers(top_n),
            "local_drivers": movement.local_drivers(
                frame,
                prediction["direction"],
                top_n,
            ),
        }
    if command == "scenario":
        request = payload.get("scenario_request")
        if not isinstance(request, Mapping):
            raise ValueError("scenario_request must be one object.")
        result = intelligence.scenario(
            ticker=str(payload["ticker"]).upper(),
            cutoff=target_date,
            probabilities={
                "Down": prediction["prob_down"],
                "Flat": prediction["prob_flat"],
                "Up": prediction["prob_up"],
            },
            request=request,
        )
        return {**common, **result}
    raise ValueError(f"Unsupported worker command: {command}")


def main() -> int:
    """Load verified artifacts once and serve requests until input closes."""

    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    try:
        from financial_news_intelligence.api.runtime_environment import (
            require_active_environment,
        )

        # Validate sys.prefix before importing joblib, pandas, or scikit-learn.
        # Resolving .venv/bin/python to its base interpreter previously loaded
        # the global scikit-learn package and broke joblib deserialization.
        require_active_environment(project_root, ".venv")

        from financial_news_intelligence.api.artifacts import ArtifactRegistry
        from financial_news_intelligence.api.movement_bundle_loader import (
            load_verified_movement_bundle,
        )

        paths = ArtifactRegistry(project_root).verify()

        # Load the exact owner-controlled joblib bundle before importing the
        # pandas-based runtime. The read-only diagnostic proved this ordering is
        # compatible with the verified Mac .venv, while the previous order failed
        # after pandas and scikit-learn had already entered the process.
        movement_bundle = load_verified_movement_bundle(paths.movement_model)

        from financial_news_intelligence.api.intelligence_runtime import (
            IntelligenceRuntime,
        )
        from financial_news_intelligence.api.movement_runtime import MovementRuntime

        movement = MovementRuntime(paths, movement_bundle)
        intelligence = IntelligenceRuntime(movement, paths)
    except Exception as error:  # noqa: BLE001 - startup failure must be structured.
        print(
            json.dumps(
                failure_from_exception(error),
                separators=(",", ":"),
            ),
            flush=True,
        )
        return 1

    for raw_line in sys.stdin:
        try:
            request = json.loads(raw_line)
            command, payload = require_payload(request)
            result = handle_request(command, payload, movement, intelligence)
            response = {"status": "PASSED", "result": json_safe(result)}
        except Exception as error:  # noqa: BLE001 - one request must not stop worker.
            response = failure_from_exception(error)
        print(json.dumps(response, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
