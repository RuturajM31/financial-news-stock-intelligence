#!/usr/bin/env python3
"""Verify local Docker endpoints with bounded startup retries and no secrets."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable


ResponseValidator = Callable[[int, str], bool]


def fetch(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout_seconds: float = 20.0,
) -> tuple[int, str]:
    """Fetch one HTTP endpoint and return bounded text for validation."""

    request = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return response.status, response.read(5 * 1024 * 1024).decode(
                "utf-8", errors="replace"
            )
    except urllib.error.HTTPError as error:
        return error.code, error.read(1_000_000).decode("utf-8", errors="replace")


def wait_for_endpoint(
    url: str,
    *,
    label: str,
    validator: ResponseValidator,
    startup_timeout_seconds: float,
    headers: dict[str, str] | None = None,
) -> tuple[int, str]:
    """Retry transient host-publication failures until one endpoint is valid."""

    deadline = time.monotonic() + startup_timeout_seconds
    attempt = 0
    last_detail = "no response received"

    while True:
        attempt += 1
        try:
            status, body = (
                fetch(url) if headers is None else fetch(url, headers=headers)
            )
            if validator(status, body):
                return status, body
            last_detail = f"HTTP {status}; body prefix={body[:160]!r}"
        except (
            ConnectionError,
            TimeoutError,
            OSError,
            urllib.error.URLError,
        ) as error:
            last_detail = f"{type(error).__name__}: {error}"

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError(
                f"{label} was not ready within {startup_timeout_seconds:.0f}s; "
                f"last result: {last_detail}"
            )

        if attempt == 1 or attempt % 10 == 0:
            print(
                f"WAITING: {label} host endpoint is not ready "
                f"(attempt {attempt}; {last_detail})",
                flush=True,
            )
        time.sleep(min(1.0, remaining))


def valid_json_status(expected: str) -> ResponseValidator:
    """Build a validator for the API health/readiness response contract."""

    def validate(status: int, body: str) -> bool:
        if status != 200:
            return False
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            return False
        return isinstance(payload, dict) and payload.get("status") == expected

    return validate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fastapi-port", required=True, type=int)
    parser.add_argument("--streamlit-port", required=True, type=int)
    parser.add_argument("--startup-timeout-seconds", type=float, default=180.0)
    args = parser.parse_args()

    api_key = os.environ.get("FNI_API_KEY", "").strip()
    if len(api_key) < 24:
        raise RuntimeError(
            "FNI_API_KEY must contain at least 24 characters for authenticated smoke tests."
        )

    authenticated_headers = {"X-API-Key": api_key}
    api = f"http://127.0.0.1:{args.fastapi_port}"
    ui = f"http://127.0.0.1:{args.streamlit_port}"

    wait_for_endpoint(
        api + "/health",
        label="FastAPI health",
        validator=valid_json_status("PASSED"),
        startup_timeout_seconds=args.startup_timeout_seconds,
    )

    wait_for_endpoint(
        api + "/ready",
        label="FastAPI readiness",
        validator=valid_json_status("PASSED"),
        startup_timeout_seconds=args.startup_timeout_seconds,
    )

    wait_for_endpoint(
        api + "/metrics",
        label="Prometheus metrics",
        validator=lambda status, body: (
            status == 200
            and "fni_build_info" in body
            and "fni_http_requests_total" in body
        ),
        startup_timeout_seconds=args.startup_timeout_seconds,
        headers=authenticated_headers,
    )

    wait_for_endpoint(
        api + "/v1/provenance",
        label="Protected-route authentication",
        validator=lambda status, _body: status in {401, 403},
        startup_timeout_seconds=args.startup_timeout_seconds,
    )

    wait_for_endpoint(
        ui + "/_stcore/health",
        label="Streamlit health",
        validator=lambda status, body: status == 200 and "ok" in body.lower(),
        startup_timeout_seconds=args.startup_timeout_seconds,
    )

    wait_for_endpoint(
        ui + "/",
        label="Streamlit root page",
        validator=lambda status, body: (
            status == 200 and "streamlit" in body.lower()
        ),
        startup_timeout_seconds=args.startup_timeout_seconds,
    )

    print("FASTAPI HEALTH OVER DOCKER: PASSED")
    print("FASTAPI READINESS OVER DOCKER: PASSED")
    print("PROMETHEUS METRICS OVER DOCKER: PASSED")
    print("PROTECTED ROUTE AUTHENTICATION OVER DOCKER: PASSED")
    print("STREAMLIT HEALTH OVER DOCKER: PASSED")
    print("STREAMLIT ROOT OVER DOCKER: PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
