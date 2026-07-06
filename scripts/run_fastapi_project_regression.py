#!/usr/bin/env python3
"""Run the existing isolated project regression before and after installation.

The project already contains a runtime-aware regression runner that executes
analytics and transformer tests in separate Python environments. This wrapper
verifies its command-line contract, runs it without guessing, retains the full
output, and requires its explicit success marker.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


SUCCESS_MARKER = "ISOLATED PROJECT REGRESSION: PASSED"


class RegressionGateError(RuntimeError):
    """Describe one missing or failed full-project regression gate."""


def parse_args() -> argparse.Namespace:
    """Parse the project root, phase, and evidence output path."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=("pre-run", "post-run"))
    parser.add_argument("--evidence-file", required=True, type=Path)
    return parser.parse_args()


def build_regression_command(
    python_executable: Path,
    runner: Path,
    project_root: Path,
    phase: str,
    help_text: str,
) -> list[str]:
    """Build only the audited command supported by the existing runner."""

    required_options = ("--project-root", "--phase")
    missing = [option for option in required_options if option not in help_text]
    if missing:
        raise RegressionGateError(
            "The existing regression runner does not expose the required "
            f"options: {missing}"
        )
    return [
        str(python_executable),
        str(runner),
        "--project-root",
        str(project_root),
        "--phase",
        phase,
    ]


def run_regression(
    python_executable: Path,
    project_root: Path,
    phase: str,
    evidence_file: Path,
) -> None:
    """Run the existing isolated regression and retain owner-only evidence."""

    runner = project_root / "scripts/run_isolated_project_regression.py"
    if not runner.is_file() or runner.is_symlink():
        raise RegressionGateError(
            f"The isolated project regression runner is missing or unsafe: {runner}"
        )
    help_result = subprocess.run(
        [str(python_executable), str(runner), "--help"],
        cwd=project_root,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if help_result.returncode != 0:
        raise RegressionGateError(
            "The isolated regression runner did not provide its help contract. "
            f"Exit code: {help_result.returncode}."
        )
    command = build_regression_command(
        python_executable,
        runner,
        project_root,
        phase,
        help_result.stdout + help_result.stderr,
    )
    process = subprocess.Popen(
        command,
        cwd=project_root,
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    lines: list[str] = []
    if process.stdout is None:
        raise RegressionGateError("The regression output stream is unavailable.")
    for line in process.stdout:
        lines.append(line)
        print(line, end="", flush=True)
    exit_code = process.wait()
    output = "".join(lines)
    evidence_file.parent.mkdir(parents=True, exist_ok=True)
    evidence_file.write_text(output, encoding="utf-8")
    os.chmod(evidence_file, 0o600)
    if exit_code != 0:
        raise RegressionGateError(
            f"The {phase} isolated project regression exited with code {exit_code}."
        )
    if SUCCESS_MARKER not in output:
        raise RegressionGateError(
            f"The {phase} regression did not print its required success marker."
        )


def main() -> int:
    """Run one regression phase and print a stable package marker."""

    args = parse_args()
    root = args.project_root.expanduser().resolve()
    print(f"STARTED: Full project regression ({args.phase}).", flush=True)
    try:
        run_regression(Path(sys.executable), root, args.phase, args.evidence_file)
        print(f"FASTAPI PROJECT REGRESSION GATE ({args.phase}): PASSED")
        return 0
    except Exception as error:  # noqa: BLE001 - gate must report exact failure.
        print("FAILED: Full project regression gate failed.", file=sys.stderr)
        print(f"Location: {args.phase} isolated project regression", file=sys.stderr)
        print(f"Reason: {type(error).__name__}: {error}", file=sys.stderr)
        print(
            "Safe next step: Correct the named test or regression-runner contract, "
            "then rerun the FastAPI strike.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
