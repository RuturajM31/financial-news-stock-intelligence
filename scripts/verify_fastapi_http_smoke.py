#!/usr/bin/env python3
"""Start Uvicorn locally and verify real HTTP health and readiness responses."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


class HttpSmokeError(RuntimeError):
    """Describe one local server startup or HTTP contract failure."""


def parse_args() -> argparse.Namespace:
    """Parse the project root and evidence output file."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--evidence-file", required=True, type=Path)
    parser.add_argument("--startup-timeout-seconds", type=int, default=90)
    return parser.parse_args()


def free_loopback_port() -> int:
    """Reserve and release one local port for the short-lived smoke server."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def request_json(url: str, timeout_seconds: float = 5.0) -> tuple[int, dict[str, Any]]:
    """Request one local endpoint and require a JSON object response."""

    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        status = int(response.status)
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise HttpSmokeError(f"Endpoint returned a non-object JSON body: {url}")
    return status, payload


def validate_health(status: int, payload: dict[str, Any]) -> None:
    """Require the real health endpoint contract."""

    if status != 200 or payload.get("status") != "PASSED":
        raise HttpSmokeError(
            f"Health response failed. HTTP {status}; payload={payload}"
        )


def validate_readiness(status: int, payload: dict[str, Any]) -> None:
    """Require lightweight readiness without starting model inference."""

    components = payload.get("components")
    details = payload.get("details")
    if status != 200 or payload.get("status") != "PASSED":
        raise HttpSmokeError(
            f"Readiness response failed. HTTP {status}; payload={payload}"
        )
    if not isinstance(components, dict) or components.get("artifacts") != "PASSED":
        raise HttpSmokeError("Readiness did not verify artifact checksums.")
    if components.get("movement_worker") != "CONFIGURED":
        raise HttpSmokeError("Readiness unexpectedly started or failed movement.")
    if components.get("sentiment_worker") != "CONFIGURED":
        raise HttpSmokeError("Readiness unexpectedly started or failed sentiment.")
    if not isinstance(details, dict) or details.get("deep_probe_run") is not False:
        raise HttpSmokeError("Readiness did not remain lightweight.")


def main() -> int:
    """Start the actual server, call two endpoints, and stop it cleanly."""

    args = parse_args()
    root = args.project_root.expanduser().resolve()
    port = free_loopback_port()
    command = [
        sys.executable,
        str(root / "scripts/run_fastapi_application.py"),
        "--project-root",
        str(root),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--log-level",
        "warning",
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "FNI_PROJECT_ROOT": str(root),
            "FNI_API_ENVIRONMENT": "test",
            "FNI_REQUIRE_API_KEY": "false",
            "FNI_TRUSTED_HOSTS": "127.0.0.1,localhost",
            "FNI_DEEP_READINESS_PROBE": "false",
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    print("STARTED: Local Uvicorn HTTP smoke test.", flush=True)
    process = subprocess.Popen(
        command,
        cwd=root,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    server_output: list[str] = []
    evidence: dict[str, Any] = {"port": port, "health": None, "readiness": None}
    try:
        deadline = time.monotonic() + args.startup_timeout_seconds
        last_error = "Server did not answer yet."
        while time.monotonic() < deadline:
            if process.poll() is not None:
                if process.stdout is not None:
                    server_output.append(process.stdout.read())
                raise HttpSmokeError(
                    "Uvicorn stopped before health verification. "
                    f"Exit code: {process.returncode}. Output: {''.join(server_output)[-2000:]}"
                )
            try:
                health_status, health_payload = request_json(
                    f"http://127.0.0.1:{port}/health"
                )
                validate_health(health_status, health_payload)
                evidence["health"] = health_payload
                break
            except (OSError, urllib.error.URLError, json.JSONDecodeError, HttpSmokeError) as error:
                last_error = f"{type(error).__name__}: {error}"
                time.sleep(0.25)
        else:
            raise HttpSmokeError(
                "Uvicorn did not become healthy within the startup timeout. "
                f"Last error: {last_error}"
            )
        ready_status, ready_payload = request_json(
            f"http://127.0.0.1:{port}/ready"
        )
        validate_readiness(ready_status, ready_payload)
        evidence["readiness"] = ready_payload
        args.evidence_file.parent.mkdir(parents=True, exist_ok=True)
        args.evidence_file.write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.chmod(args.evidence_file, 0o600)
        print("PASSED: Real /health endpoint over HTTP.")
        print("PASSED: Lightweight /ready endpoint over HTTP.")
        print("FASTAPI HTTP SMOKE TEST: PASSED")
        return 0
    except Exception as error:  # noqa: BLE001 - must report exact smoke failure.
        print("FAILED: FastAPI HTTP smoke test failed.", file=sys.stderr)
        print("Location: Local Uvicorn server or HTTP endpoint", file=sys.stderr)
        print(f"Reason: {type(error).__name__}: {error}", file=sys.stderr)
        print(
            "Safe next step: Inspect the smoke-test evidence and server output, "
            "correct the named startup or endpoint issue, and rerun.",
            file=sys.stderr,
        )
        return 1
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        if process.stdout is not None:
            remaining = process.stdout.read()
            if remaining:
                server_output.append(remaining)


if __name__ == "__main__":
    raise SystemExit(main())
