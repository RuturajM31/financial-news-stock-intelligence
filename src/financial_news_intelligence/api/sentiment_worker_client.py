"""Isolated DistilBERT worker client.

The FastAPI web process does not load native model libraries. DistilBERT runs
in the separate ``.venv-distilbert`` Python process, while movement and
intelligence use another isolated worker in the main analytics environment.
"""

from __future__ import annotations

import json
import os
import select
import subprocess
import threading
from pathlib import Path
from typing import Any

from .errors import ApiProblem
from .runtime_environment import absolute_launcher_path


class SentimentWorkerClient:
    """Manage one bounded JSON-lines DistilBERT subprocess."""

    def __init__(
        self,
        project_root: Path,
        python_executable: Path,
        model_directory: Path,
        timeout_seconds: int,
    ) -> None:
        self.project_root = project_root.expanduser().resolve()
        self.python_executable = absolute_launcher_path(python_executable)
        self.model_directory = model_directory
        self.timeout_seconds = timeout_seconds
        self._process: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()

    def _start(self) -> subprocess.Popen[str]:
        """Start the isolated worker only when the first request arrives."""

        if self._process is not None and self._process.poll() is None:
            return self._process
        worker_script = (
            self.project_root / "scripts/run_sentiment_inference_worker.py"
        ).resolve()
        if not worker_script.exists() or not worker_script.is_file():
            raise ApiProblem(
                503,
                "sentiment_worker_script_missing",
                "Sentiment prediction failed.",
                str(worker_script),
                "The isolated sentiment worker script is missing.",
                "Reinstall the verified FastAPI package and retry.",
            )
        environment = os.environ.copy()
        environment.update(
            {
                # The transformer environment may not install the project as a
                # package. The worker needs only the pure-standard-library
                # runtime boundary module before loading NumPy, PyTorch, and
                # Transformers.
                "PYTHONPATH": str(self.project_root / "src"),
                "PYTHONNOUSERSITE": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            }
        )
        self._process = subprocess.Popen(
            [
                str(self.python_executable),
                str(worker_script),
                "--project-root",
                str(self.project_root),
                "--model-directory",
                str(self.model_directory),
            ],
            cwd=self.project_root,
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        return self._process

    def predict(self, texts: list[str]) -> list[dict[str, Any]]:
        """Return validated three-class sentiment results for a bounded batch."""

        if not texts or any(not str(value).strip() for value in texts):
            raise ApiProblem(
                422,
                "sentiment_text_invalid",
                "Sentiment prediction failed.",
                "Sentiment request text",
                "At least one article is empty.",
                "Provide non-empty financial-news text and retry.",
            )
        request_payload = json.dumps(
            {"texts": texts},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        with self._lock:
            process = self._start()
            if process.stdin is None or process.stdout is None:
                self._terminate()
                raise ApiProblem(
                    503,
                    "sentiment_worker_pipe_missing",
                    "Sentiment prediction failed.",
                    "Isolated worker communication pipe",
                    "The worker did not expose its input and output streams.",
                    "Restart the FastAPI process and retry once.",
                )
            try:
                process.stdin.write(request_payload + "\n")
                process.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                self._terminate()
                raise ApiProblem(
                    503,
                    "sentiment_worker_write_failed",
                    "Sentiment prediction failed.",
                    "Isolated worker input",
                    "The DistilBERT worker stopped before reading the request.",
                    "Restart the FastAPI process and rerun API readiness.",
                ) from exc
            readable, _, _ = select.select(
                [process.stdout],
                [],
                [],
                self.timeout_seconds,
            )
            if not readable:
                self._terminate()
                raise ApiProblem(
                    504,
                    "sentiment_worker_timeout",
                    "Sentiment prediction failed.",
                    "Isolated DistilBERT worker",
                    f"The worker exceeded the {self.timeout_seconds}-second timeout.",
                    "Retry once with a shorter article; then inspect worker readiness.",
                )
            response_line = process.stdout.readline()
            if not response_line:
                exit_code = process.poll()
                self._terminate()
                raise ApiProblem(
                    503,
                    "sentiment_worker_stopped",
                    "Sentiment prediction failed.",
                    "Isolated DistilBERT worker",
                    f"The worker stopped without a response. Exit code: {exit_code}.",
                    "Run the FastAPI verifier to inspect the sentiment runtime.",
                )
        try:
            response = json.loads(response_line)
        except json.JSONDecodeError as exc:
            self._terminate()
            raise ApiProblem(
                503,
                "sentiment_worker_response_invalid",
                "Sentiment prediction failed.",
                "Isolated worker response",
                "The worker returned invalid JSON.",
                "Restart the API and run the FastAPI verifier.",
            ) from exc
        if not isinstance(response, dict) or response.get("status") != "PASSED":
            reason = (
                response.get("why_failed")
                if isinstance(response, dict)
                else "Unknown worker failure."
            )
            raise ApiProblem(
                503,
                "sentiment_worker_prediction_failed",
                "Sentiment prediction failed.",
                "Isolated DistilBERT worker",
                str(reason),
                (
                    "Run the FastAPI verifier and restore verified model "
                    "artifacts if needed."
                ),
            )
        results = response.get("results")
        if not isinstance(results, list) or len(results) != len(texts):
            raise ApiProblem(
                503,
                "sentiment_worker_count_changed",
                "Sentiment prediction failed.",
                "Isolated worker response",
                "The response count does not match the request count.",
                "Restart the API and run the FastAPI verifier.",
            )
        required_fields = {
            "label",
            "confidence",
            "prob_bearish",
            "prob_neutral",
            "prob_bullish",
        }
        validated: list[dict[str, Any]] = []
        for item in results:
            if not isinstance(item, dict) or required_fields - set(item):
                raise ApiProblem(
                    503,
                    "sentiment_worker_schema_changed",
                    "Sentiment prediction failed.",
                    "Isolated worker response",
                    "A sentiment result is missing required fields.",
                    "Restart the API and run the FastAPI verifier.",
                )
            probabilities = [
                float(item["prob_bearish"]),
                float(item["prob_neutral"]),
                float(item["prob_bullish"]),
            ]
            if any(value < 0.0 or value > 1.0 for value in probabilities):
                raise ApiProblem(
                    503,
                    "sentiment_worker_probability_invalid",
                    "Sentiment prediction failed.",
                    "Isolated worker response",
                    "A sentiment probability is outside the range zero to one.",
                    "Restart the API and run the FastAPI verifier.",
                )
            if abs(sum(probabilities) - 1.0) > 1e-6:
                raise ApiProblem(
                    503,
                    "sentiment_worker_probability_total_invalid",
                    "Sentiment prediction failed.",
                    "Isolated worker response",
                    "The sentiment probabilities do not total one.",
                    "Restart the API and run the FastAPI verifier.",
                )
            validated.append(dict(item))
        return validated

    def _terminate(self) -> None:
        """Stop a failed worker without affecting the FastAPI process."""

        process = self._process
        self._process = None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

    def close(self) -> None:
        """Stop the worker during application shutdown."""

        with self._lock:
            self._terminate()
