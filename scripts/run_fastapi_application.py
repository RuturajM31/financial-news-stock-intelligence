#!/usr/bin/env python3
"""Start the verified FastAPI application in the main analytics environment."""

from __future__ import annotations

import argparse
import ipaddress
import os
import sys
from pathlib import Path


NATIVE_THREAD_LIMITS = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
}


def configure_native_thread_limits() -> None:
    """Limit native numerical libraries before importing model modules.

    The project has previously loaded Intel and LLVM OpenMP runtimes together.
    Limiting native pools does not repair incompatible binary builds, but it
    reduces thread oversubscription and keeps local API inference deterministic.
    The later dependency-closure phase must standardize the binary runtime.
    """

    for variable_name, default_value in NATIVE_THREAD_LIMITS.items():
        os.environ.setdefault(variable_name, default_value)


def parse_args() -> argparse.Namespace:
    """Parse one project root, host, port, and log level."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument(
        "--log-level",
        choices=("critical", "error", "warning", "info", "debug"),
        default="info",
    )
    return parser.parse_args()


def main() -> int:
    """Validate the runtime and start one API worker."""

    args = parse_args()
    root = args.project_root.expanduser().resolve()
    if not 1 <= args.port <= 65_535:
        print("FAILED: FastAPI received an invalid TCP port.")
        print("Location: --port startup argument")
        print("Reason: A TCP port must be between 1 and 65535.")
        print("Safe next step: Choose an unused local port and run the command again.")
        return 1
    try:
        bind_address = ipaddress.ip_address(args.host)
    except ValueError:
        bind_address = None
    if args.host != "localhost" and (
        bind_address is None or not bind_address.is_loopback
    ):
        print("FAILED: FastAPI refused a non-loopback network binding.")
        print("Location: --host startup argument")
        print(
            "Reason: This phase authorizes local API use only; public or LAN "
            "binding is deferred to the container and deployment security phases."
        )
        print("Safe next step: Use --host 127.0.0.1 for this phase.")
        return 1
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
    os.environ["FNI_PROJECT_ROOT"] = str(root)
    configure_native_thread_limits()
    try:
        import uvicorn

        from financial_news_intelligence.api.app import create_app
        from financial_news_intelligence.api.config import ApiSettings

        settings = ApiSettings.from_environment(root)
        app = create_app(settings=settings)
    except Exception as exc:  # noqa: BLE001 - startup must show exact stage.
        print("FAILED: FastAPI configuration or application creation failed.")
        print("Location: FastAPI startup")
        print(f"Reason: {type(exc).__name__}: {exc}")
        print("Safe next step: Correct the named setting and run verification again.")
        return 1
    print("STARTED: FastAPI application is starting.", flush=True)
    print(f"Documentation: http://{args.host}:{args.port}/docs", flush=True)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        workers=1,
        access_log=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
