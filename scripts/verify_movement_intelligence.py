#!/usr/bin/env python3
"""Independently verify movement or complete intelligence artifacts.

Purpose
-------
Provide a small command-line verifier that recomputes movement metrics,
probability constraints, split boundaries, artifact checksums, explanation
coverage, historical chronology, scenario coverage, provenance, and licence
boundaries from saved outputs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from run_movement_intelligence import verify_all, verify_movement_phase


def parse_args() -> argparse.Namespace:
    """Parse the project root and required verification phase."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument(
        "--phase",
        choices=("movement", "all"),
        required=True,
    )
    return parser.parse_args()


def main() -> int:
    """Run semantic verification for the selected completed phase."""

    # Verification is read-only. It never rewrites models or intelligence.
    args = parse_args()
    try:
        if args.phase == "movement":
            summary = verify_movement_phase(args.project_root)
            print("STOCK MOVEMENT VERIFICATION: PASSED")
            print(f"Champion: {summary['quality_champion']}")
            print(
                "Historical-audit macro F1: "
                f"{summary['test_metrics']['macro_f1']:.6f}"
            )
        else:
            manifest = verify_all(args.project_root)
            print("MOVEMENT INTELLIGENCE VERIFICATION: PASSED")
            print(f"Champion: {manifest['quality_champion']}")
    except Exception as exc:  # noqa: BLE001 - report the exact verifier failure.
        print(
            f"VERIFICATION FAILED: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
