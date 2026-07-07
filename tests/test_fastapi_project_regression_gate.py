"""Tests for the pre-install and post-install full-project regression gate."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from scripts import run_fastapi_project_regression as regression


def test_build_regression_command_requires_explicit_options(tmp_path: Path) -> None:
    """Prepare incomplete help, build the command, and check rejection."""

    with pytest.raises(regression.RegressionGateError, match="required options"):
        regression.build_regression_command(
            Path("python"),
            tmp_path / "runner.py",
            tmp_path,
            "pre-run",
            "usage: runner [--phase PHASE]",
        )


def test_build_regression_command_uses_project_and_phase(tmp_path: Path) -> None:
    """Prepare valid help, build the command, and check stable arguments."""

    command = regression.build_regression_command(
        Path("/env/bin/python"),
        tmp_path / "runner.py",
        tmp_path,
        "post-run",
        "options: --project-root ROOT --phase {pre-run,post-run}",
    )

    assert command[-4:] == ["--project-root", str(tmp_path), "--phase", "post-run"]


def test_run_regression_requires_existing_safe_runner(tmp_path: Path) -> None:
    """Prepare a missing runner, run the gate, and check safe failure."""

    with pytest.raises(regression.RegressionGateError, match="missing or unsafe"):
        regression.run_regression(
            Path("python"), tmp_path, "pre-run", tmp_path / "evidence.log"
        )


def test_run_regression_retains_success_evidence(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Prepare a fake runner, run the gate, and check success evidence."""

    scripts = tmp_path / "scripts"
    scripts.mkdir()
    runner = scripts / "run_isolated_project_regression.py"
    runner.write_text("# placeholder\n", encoding="utf-8")

    class Completed:
        """Provide the help response used by the command builder."""

        returncode = 0
        stdout = "--project-root ROOT --phase PHASE"
        stderr = ""

    class Stream:
        """Yield one successful regression marker."""

        def __iter__(self):
            return iter([regression.SUCCESS_MARKER + " (1 passed)\n"])

    class Process:
        """Provide a successful subprocess interface."""

        stdout = Stream()

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(regression.subprocess, "run", lambda *args, **kwargs: Completed())
    monkeypatch.setattr(regression.subprocess, "Popen", lambda *args, **kwargs: Process())
    evidence = tmp_path / "evidence.log"

    regression.run_regression(Path("python"), tmp_path, "pre-run", evidence)

    assert regression.SUCCESS_MARKER in evidence.read_text(encoding="utf-8")
    assert os.stat(evidence).st_mode & 0o077 == 0
