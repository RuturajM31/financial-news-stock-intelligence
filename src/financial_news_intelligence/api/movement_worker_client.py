"""Isolated movement and intelligence worker client.

The FastAPI process must remain available even if a native numerical library
fails. Movement prediction, historical retrieval, explainability, and scenario
analysis therefore run in one separate ``.venv`` subprocess. One request is one
JSON line. A worker crash becomes a structured HTTP 503 error instead of taking
down the web process.
"""

from __future__ import annotations

import json
import os
import select
import signal
import subprocess
import threading
from pathlib import Path
from typing import Any, Mapping

from .errors import ApiProblem
from .runtime_environment import absolute_launcher_path


NATIVE_THREAD_LIMITS = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
}


class MovementWorkerClient:
    """Manage one bounded JSON-lines movement/intelligence subprocess."""

    def __init__(
        self,
        project_root: Path,
        python_executable: Path,
        timeout_seconds: int,
    ) -> None:
        self.project_root = project_root.expanduser().resolve()
        self.python_executable = absolute_launcher_path(python_executable)
        self.timeout_seconds = timeout_seconds
        self._process: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()

    def _start(self) -> subprocess.Popen[str]:
        """Start the worker lazily and reuse it while it remains healthy."""

        if self._process is not None and self._process.poll() is None:
            return self._process
        worker_script = (
            self.project_root / "scripts/run_movement_intelligence_worker.py"
        ).resolve()
        if not worker_script.exists() or not worker_script.is_file():
            raise ApiProblem(
                503,
                "movement_worker_script_missing",
                "Movement and intelligence processing failed.",
                str(worker_script),
                "The isolated movement worker script is missing.",
                "Reinstall the verified FastAPI package and rerun readiness.",
            )
        environment = os.environ.copy()
        environment.update(
            {
                "PYTHONPATH": os.pathsep.join(
                    [
                        str(self.project_root / "runtime_shims"),
                        str(self.project_root / "src"),
                    ]
                ),
                "PYTHONNOUSERSITE": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONHASHSEED": "0",
            }
        )
        for name, value in NATIVE_THREAD_LIMITS.items():
            environment.setdefault(name, value)
        self._process = subprocess.Popen(
            [
                str(self.python_executable),
                str(worker_script),
                "--project-root",
                str(self.project_root),
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

    @staticmethod
    def _exit_reason(exit_code: int | None) -> str:
        """Describe one worker exit without exposing internal data."""

        if exit_code is None:
            return "The worker stopped before its exit code was available."
        if exit_code < 0:
            signal_number = -exit_code
            try:
                signal_name = signal.Signals(signal_number).name
            except ValueError:
                signal_name = f"signal_{signal_number}"
            return (
                f"The worker was terminated by {signal_name} "
                f"({signal_number})."
            )
        return f"The worker stopped with exit code {exit_code}."

    @classmethod
    def _stopped_reason(cls, process: subprocess.Popen[str]) -> str:
        """Wait briefly for an exited worker and return its exact reason."""

        exit_code = process.poll()
        if exit_code is None:
            try:
                exit_code = process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                exit_code = None
        return cls._exit_reason(exit_code)

    def _request(
        self,
        command: str,
        payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Send one request and require one valid success response."""

        request_payload = {"command": command, "payload": dict(payload or {})}
        request_line = json.dumps(
            request_payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        with self._lock:
            process = self._start()
            if process.stdin is None or process.stdout is None:
                self._terminate()
                raise ApiProblem(
                    503,
                    "movement_worker_pipe_missing",
                    "Movement and intelligence processing failed.",
                    "Isolated movement worker communication pipe",
                    "The worker did not expose its input and output streams.",
                    "Restart FastAPI and rerun readiness.",
                )
            try:
                process.stdin.write(request_line + "\n")
                process.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                exit_reason = self._stopped_reason(process)
                self._terminate()
                raise ApiProblem(
                    503,
                    "movement_worker_write_failed",
                    "Movement and intelligence processing failed.",
                    "Isolated movement worker input",
                    exit_reason,
                    "Restart FastAPI and run the FastAPI verifier.",
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
                    "movement_worker_timeout",
                    "Movement and intelligence processing failed.",
                    "Isolated movement worker",
                    f"The worker exceeded the {self.timeout_seconds}-second timeout.",
                    "Run the FastAPI verifier and inspect the movement runtime.",
                )
            response_line = process.stdout.readline()
            if not response_line:
                exit_reason = self._stopped_reason(process)
                self._terminate()
                raise ApiProblem(
                    503,
                    "movement_worker_stopped",
                    "Movement and intelligence processing failed.",
                    "Isolated movement worker",
                    exit_reason,
                    "Run the FastAPI verifier before retrying the request.",
                )
        try:
            response = json.loads(response_line)
        except json.JSONDecodeError as exc:
            self._terminate()
            raise ApiProblem(
                503,
                "movement_worker_response_invalid",
                "Movement and intelligence processing failed.",
                "Isolated movement worker response",
                "The worker returned invalid JSON.",
                "Restart FastAPI and run the FastAPI verifier.",
            ) from exc
        if not isinstance(response, dict):
            raise ApiProblem(
                503,
                "movement_worker_response_not_object",
                "Movement and intelligence processing failed.",
                "Isolated movement worker response",
                "The worker response is not one JSON object.",
                "Restart FastAPI and run the FastAPI verifier.",
            )
        if response.get("status") != "PASSED":
            raise ApiProblem(
                int(response.get("status_code", 503)),
                str(response.get("error_code", "movement_worker_failed")),
                str(
                    response.get(
                        "what_failed",
                        "Movement and intelligence processing failed.",
                    )
                ),
                str(response.get("where_failed", "Isolated movement worker")),
                str(response.get("why_failed", "Unknown worker failure.")),
                str(
                    response.get(
                        "safe_next_step",
                        "Run the FastAPI verifier and inspect verified artifacts.",
                    )
                ),
            )
        result = response.get("result")
        if not isinstance(result, dict):
            raise ApiProblem(
                503,
                "movement_worker_result_invalid",
                "Movement and intelligence processing failed.",
                "Isolated movement worker response",
                "The success response does not contain one result object.",
                "Restart FastAPI and run the FastAPI verifier.",
            )
        return result

    def readiness(self) -> dict[str, Any]:
        """Verify the saved champion prediction in the isolated worker."""

        return self._request("readiness")

    def predict(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Return one historical-session movement prediction."""

        return self._request("predict", payload)

    def historical(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Return strictly earlier historical intelligence."""

        return self._request("historical", payload)

    def explain(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Return global and local movement sensitivity evidence."""

        return self._request("explain", payload)

    def scenario(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Return user-controlled research scenario outcomes."""

        return self._request("scenario", payload)

    def provenance(self) -> dict[str, Any]:
        """Return the verified licence-safe provenance report."""

        return self._request("provenance")

    def _terminate(self) -> None:
        """Stop the worker without affecting the FastAPI process."""

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
