#!/usr/bin/env python3
"""Trace native libraries and fail before unsafe movement training.

Purpose
-------
Import the exact movement-training dependency path one step at a time, capture
``threadpoolctl`` evidence after each import, and reject a process that contains
more than one OpenMP runtime family.

Inputs
------
``--project-root`` identifies the existing project. ``--diagnostic-dir`` is an
owner-only external directory that survives installer rollback.

Outputs and downstream use
--------------------------
The script writes ``movement_openmp_import_trace.json``. The suite must pass this
preflight before focused tests or model training begin.

Limitations
-----------
The check diagnoses loaded libraries; it does not rewrite the Python environment
or hide a conflict with ``KMP_DUPLICATE_LIB_OK``.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

# Capture the shell-provided value before importing threadpoolctl. Some managed
# Python images set this variable during native-library discovery; only an
# inherited user or process setting is treated as an unsafe bypass request.
ORIGINAL_KMP_DUPLICATE_LIB_OK = os.environ.get("KMP_DUPLICATE_LIB_OK")

from threadpoolctl import threadpool_info

IMPORT_STEPS = (
    "numpy",
    "pandas",
    "sklearn",
    "financial_news_intelligence.models.movement_dataset",
    "financial_news_intelligence.models.movement_training",
)


class RuntimePreflightError(RuntimeError):
    """Raised when the movement process contains unsafe native runtimes."""


def classify_openmp_family(row: Mapping[str, Any]) -> str:
    """Normalize one threadpoolctl OpenMP record to a runtime family."""

    filepath = str(row.get("filepath") or "").lower()
    prefix = str(row.get("prefix") or "").lower()
    combined = f"{filepath} {prefix}"
    if "libiomp" in combined:
        return "intel"
    if "libomp" in combined:
        return "llvm"
    if "libgomp" in combined:
        return "gnu"
    return prefix or "unknown"


def runtime_status_from_pools(
    pools: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return normalized OpenMP evidence for one import snapshot."""

    openmp_rows: list[dict[str, Any]] = []
    families: set[str] = set()
    for row in pools:
        if str(row.get("user_api", "")).lower() != "openmp":
            continue
        family = classify_openmp_family(row)
        families.add(family)
        openmp_rows.append(
            {
                "family": family,
                "prefix": row.get("prefix"),
                "filepath": row.get("filepath"),
                "version": row.get("version"),
                "num_threads": row.get("num_threads"),
            }
        )
    return {
        "status": "compatible" if len(families) <= 1 else "conflict",
        "openmp_families": sorted(families),
        "openmp_libraries": openmp_rows,
        "threadpool_count": len(pools),
    }


def atomic_json(file_path: Path, payload: Mapping[str, Any]) -> None:
    """Write one owner-only JSON file atomically."""

    file_path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(file_path.parent, 0o700)
    temporary = file_path.with_name(
        f"{file_path.name}.strike_tmp.{os.getpid()}"
    )
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.chmod(temporary, 0o600)
        temporary.replace(file_path)
        os.chmod(file_path, 0o600)
    finally:
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()


def run_preflight(
    project_root: Path,
    diagnostic_dir: Path,
) -> dict[str, Any]:
    """Import movement dependencies and return the complete runtime trace."""

    root = project_root.expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise RuntimePreflightError(f"Unsafe project root: {root}")
    if str(ORIGINAL_KMP_DUPLICATE_LIB_OK or "").strip().lower() in {
        "1",
        "true",
        "yes",
    }:
        raise RuntimePreflightError(
            "KMP_DUPLICATE_LIB_OK is forbidden because it hides conflicts."
        )

    snapshots: list[dict[str, Any]] = []
    initial = runtime_status_from_pools(threadpool_info())
    snapshots.append({"after_import": "process_start", **initial})

    for module_name in IMPORT_STEPS:
        importlib.import_module(module_name)
        snapshot = runtime_status_from_pools(threadpool_info())
        snapshots.append({"after_import": module_name, **snapshot})

    final = snapshots[-1]
    report = {
        "status": final["status"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(root),
        "import_steps": list(IMPORT_STEPS),
        "snapshots": snapshots,
        "final_openmp_families": final["openmp_families"],
        "raw_market_data_included": False,
        "credentials_included": False,
    }
    output = diagnostic_dir.expanduser().resolve()
    if output.is_symlink():
        raise RuntimePreflightError(
            f"Diagnostic directory cannot be a symlink: {output}"
        )
    atomic_json(output / "movement_openmp_import_trace.json", report)
    return report


def parse_args() -> argparse.Namespace:
    """Parse project and external diagnostic paths."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--diagnostic-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    """Run the preflight and print a stable success marker."""

    args = parse_args()
    report = run_preflight(args.project_root, args.diagnostic_dir)
    if report["status"] != "compatible":
        families = ", ".join(report["final_openmp_families"])
        raise RuntimePreflightError(
            f"Incompatible OpenMP runtime families loaded: {families}"
        )
    print("MOVEMENT OPENMP PREFLIGHT: PASSED", flush=True)
    print(
        "OpenMP families: "
        + (", ".join(report["final_openmp_families"]) or "none"),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimePreflightError as exc:
        print(f"ERROR: {exc}", flush=True)
        raise SystemExit(1)
