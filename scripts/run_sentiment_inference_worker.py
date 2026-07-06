#!/usr/bin/env python3
"""Run local DistilBERT inference as a JSON-lines worker.

Input comes from FastAPI through standard input. One line represents one batch
with a ``texts`` array. One JSON line is returned for each request. The worker
never logs article text, model logits, environment secrets, or API keys.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

LABEL_ORDER = ("Bearish", "Neutral", "Bullish")
MAX_BATCH_SIZE = 50
MAX_TOKEN_LENGTH = 128


def parse_args() -> argparse.Namespace:
    """Parse the verified local model directory."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--model-directory", required=True, type=Path)
    return parser.parse_args()


def failure(why_failed: str) -> dict[str, str]:
    """Return one safe worker failure without internal stack details."""

    return {
        "status": "FAILED",
        "what_failed": "DistilBERT sentiment inference failed.",
        "where_failed": "Isolated sentiment worker",
        "why_failed": why_failed,
        "safe_next_step": "Run the FastAPI verifier and inspect model artifacts.",
    }


def validate_texts(payload: Any) -> list[str]:
    """Require one bounded non-empty string batch."""

    if not isinstance(payload, dict) or not isinstance(payload.get("texts"), list):
        raise ValueError("The worker request must contain a texts array.")
    texts = [str(value).strip() for value in payload["texts"]]
    if not texts or len(texts) > MAX_BATCH_SIZE:
        raise ValueError(
            f"The worker batch must contain 1 to {MAX_BATCH_SIZE} articles."
        )
    if any(not value for value in texts):
        raise ValueError("The worker batch contains empty text.")
    return texts


def main() -> int:
    """Load the local model once and process requests until input closes."""

    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    model_directory = args.model_directory.expanduser().resolve()
    try:
        from financial_news_intelligence.api.runtime_environment import (
            require_active_environment,
        )

        require_active_environment(project_root, ".venv-distilbert")
    except Exception as error:  # noqa: BLE001 - worker must return safe JSON.
        print(json.dumps(failure(f"Runtime validation failed: {error}")))
        return 1
    if not model_directory.exists() or not model_directory.is_dir():
        print(json.dumps(failure("The verified model directory is missing.")))
        return 1
    try:
        import numpy as np
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_directory,
            local_files_only=True,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_directory,
            local_files_only=True,
        )
        model.eval()
    except Exception as exc:  # noqa: BLE001 - worker returns a safe error only.
        print(json.dumps(failure(f"Model loading failed: {type(exc).__name__}.")))
        return 1

    for raw_line in sys.stdin:
        try:
            payload = json.loads(raw_line)
            texts = validate_texts(payload)
            encoded = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=MAX_TOKEN_LENGTH,
                return_tensors="pt",
            )
            with torch.no_grad():
                logits = model(**encoded).logits
                probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
            if probabilities.shape != (len(texts), len(LABEL_ORDER)):
                raise ValueError(
                    f"Unexpected probability shape: {probabilities.shape}."
                )
            if not np.isfinite(probabilities).all():
                raise ValueError("Sentiment probabilities are not finite.")
            results: list[dict[str, Any]] = []
            for row in probabilities:
                if not np.isclose(row.sum(), 1.0, atol=1e-6):
                    raise ValueError("Sentiment probabilities do not sum to one.")
                position = int(row.argmax())
                results.append(
                    {
                        "label": LABEL_ORDER[position],
                        "confidence": float(row[position]),
                        "prob_bearish": float(row[0]),
                        "prob_neutral": float(row[1]),
                        "prob_bullish": float(row[2]),
                    }
                )
            print(
                json.dumps(
                    {"status": "PASSED", "results": results},
                    separators=(",", ":"),
                ),
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - one bad request must not stop worker.
            print(
                json.dumps(
                    failure(f"Request processing failed: {type(exc).__name__}."),
                    separators=(",", ":"),
                ),
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
