"""Tests for virtual-environment launcher and prefix boundaries."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from financial_news_intelligence.api.movement_worker_client import (
    MovementWorkerClient,
)
from financial_news_intelligence.api.runtime_environment import (
    RuntimeEnvironmentError,
    absolute_launcher_path,
    environment_python,
    require_active_environment,
)


def test_absolute_launcher_path_preserves_virtual_environment_symlink(
    tmp_path: Path,
) -> None:
    """Prepare a symlinked launcher, resolve its absolute path, and keep the symlink."""

    target = tmp_path / "base-python"
    launcher = tmp_path / ".venv" / "bin" / "python"
    launcher.parent.mkdir(parents=True)
    target.write_text("placeholder\n", encoding="utf-8")
    launcher.symlink_to(target)

    result = absolute_launcher_path(launcher)

    assert result == Path(os.path.abspath(launcher))
    assert result != launcher.resolve()


def test_environment_python_returns_project_launcher_without_dereferencing(
    tmp_path: Path,
) -> None:
    """Prepare one executable launcher, locate it, and preserve environment identity."""

    target = tmp_path / "shared-python"
    launcher = tmp_path / ".venv" / "bin" / "python"
    launcher.parent.mkdir(parents=True)
    target.write_text("#!/bin/sh\n", encoding="utf-8")
    target.chmod(0o755)
    launcher.symlink_to(target)

    result = environment_python(tmp_path, ".venv")

    assert result == Path(os.path.abspath(launcher))
    assert result != launcher.resolve()


def test_require_active_environment_rejects_wrong_prefix(tmp_path: Path) -> None:
    """Prepare two environments and check wrong-prefix rejection."""

    expected_prefix = tmp_path / ".venv"
    wrong_prefix = tmp_path / "global-python"
    expected_prefix.mkdir()
    wrong_prefix.mkdir()

    with pytest.raises(RuntimeEnvironmentError) as captured:
        require_active_environment(
            tmp_path,
            ".venv",
            actual_prefix=wrong_prefix,
        )

    assert captured.value.what_failed == "Python environment validation failed."
    assert str(wrong_prefix) in captured.value.where_failed
    assert "not running inside" in captured.value.why_failed
    assert str(expected_prefix / "bin" / "python") in captured.value.safe_next_step


def test_movement_worker_client_keeps_virtual_environment_launcher(
    tmp_path: Path,
) -> None:
    """Prepare a symlinked launcher, build the client, and prevent base-Python use."""

    target = tmp_path / "base-python"
    launcher = tmp_path / ".venv" / "bin" / "python"
    launcher.parent.mkdir(parents=True)
    target.write_text("#!/bin/sh\n", encoding="utf-8")
    target.chmod(0o755)
    launcher.symlink_to(target)

    client = MovementWorkerClient(
        project_root=tmp_path,
        python_executable=launcher,
        timeout_seconds=1,
    )

    assert client.python_executable == Path(os.path.abspath(launcher))
    assert client.python_executable != launcher.resolve()
