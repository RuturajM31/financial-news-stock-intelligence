#!/usr/bin/env python3
"""Build or verify the final SEC and Tiingo market-data foundation.

Purpose
-------
Expose one controlled command-line entry point for the fixed qualified window.
The command validates prerequisites, protects completed outputs, runs the
foundation, and prints a concise machine-reviewable result.

Inputs and outputs
------------------
The project root supplies the ticker reference, passed Tiingo qualification,
sentiment champion, and local model files. ``TIINGO_API_TOKEN`` is read only
from the environment. Successful execution creates the two compatibility CSVs,
rejection evidence, QA evidence, and the foundation manifest.

Safety
------
The fixed 2015-2020 contract cannot be widened through CLI parameters. Existing
outputs require ``--replace-existing``. ``--verify-only`` performs no network
request or write. The command never trains the movement model or changes
application deployment.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from financial_news_intelligence.data.market_data_foundation import (
    MANIFEST_OUTPUT,
    NEWS_OUTPUT,
    PRICE_OUTPUT,
    QA_OUTPUT,
    REJECTED_OUTPUT,
    FoundationConfig,
    FoundationError,
    build_foundation,
    verify_foundation,
)

CONTROLLED_OUTPUTS = (
    NEWS_OUTPUT,
    PRICE_OUTPUT,
    REJECTED_OUTPUT,
    QA_OUTPUT,
    MANIFEST_OUTPUT,
)


def protect_outputs(project_root: Path, replace_existing: bool) -> None:
    """Reject accidental replacement of completed controlled evidence."""

    existing = [
        project_root / relative_path
        for relative_path in CONTROLLED_OUTPUTS
        if (project_root / relative_path).exists()
    ]
    if existing and not replace_existing:
        formatted = "\n".join(f"- {path}" for path in existing)
        raise FoundationError(
            "Foundation outputs already exist. Use --verify-only or "
            f"--replace-existing.\n{formatted}"
        )
    if replace_existing:
        for file_path in existing:
            if file_path.is_symlink() or not file_path.is_file():
                raise FoundationError(
                    f"Refusing to replace unsafe output: {file_path}"
                )
            file_path.unlink()


def parser() -> argparse.ArgumentParser:
    """Create the intentionally small final command-line interface."""

    argument_parser = argparse.ArgumentParser(
        description="Build the qualified SEC and Tiingo foundation."
    )
    argument_parser.add_argument(
        "--project-root",
        type=Path,
        required=True,
        help="Financial News Stock Intelligence project root.",
    )
    argument_parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="Replace controlled outputs after strike backup.",
    )
    argument_parser.add_argument(
        "--refresh-sec-cache",
        action="store_true",
        help="Request fresh SEC responses instead of safe cached responses.",
    )
    argument_parser.add_argument(
        "--refresh-tiingo-cache",
        action="store_true",
        help="Request fresh Tiingo responses instead of private caches.",
    )
    argument_parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify saved outputs without requests or writes.",
    )
    return argument_parser


def main() -> int:
    """Execute the build or verification path and print final evidence."""

    arguments = parser().parse_args()
    project_root = arguments.project_root.expanduser().resolve()
    if not project_root.exists() or not project_root.is_dir():
        raise FoundationError(f"Project root does not exist: {project_root}")

    if arguments.verify_only:
        result = verify_foundation(project_root)
    else:
        api_token = os.environ.get("TIINGO_API_TOKEN", "").strip()
        if not api_token:
            raise FoundationError("TIINGO_API_TOKEN is empty.")
        protect_outputs(project_root, arguments.replace_existing)
        config = FoundationConfig(
            refresh_sec_cache=arguments.refresh_sec_cache,
            refresh_tiingo_cache=arguments.refresh_tiingo_cache,
        )
        result = build_foundation(
            project_root,
            api_token=api_token,
            config=config,
        )

    print("Status:", result["status"])
    print("Accepted SEC events:", result["accepted_articles"])
    print("Tiingo adjusted price rows:", result["primary_price_rows"])
    print("Movement classes:", result["movement_class_counts"])
    print("Manifest SHA-256:", result["manifest_sha256"])
    print("Movement model trained:", result["movement_model_trained"])
    print("Deployment changed:", result["deployment_changed"])
    print("MARKET DATA FOUNDATION: PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
