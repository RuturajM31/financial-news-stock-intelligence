"""Validate and preserve project virtual-environment boundaries.

Purpose
-------
Python virtual environments often expose ``bin/python`` as a symbolic link to
one shared base interpreter. Resolving that link before starting a subprocess
removes the virtual-environment context and can load packages from the wrong
site-packages directory. The movement artifact was created with the project
``.venv`` and must be loaded only from that environment.

Failure behavior
----------------
These helpers fail closed. They never activate, repair, or modify an
environment. They report what failed, where it failed, why it failed, and the
safe next step.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


class RuntimeEnvironmentError(RuntimeError):
    """Describe one invalid Python environment boundary."""

    def __init__(
        self,
        what_failed: str,
        where_failed: str,
        why_failed: str,
        safe_next_step: str,
    ) -> None:
        super().__init__(why_failed)
        self.what_failed = what_failed
        self.where_failed = where_failed
        self.why_failed = why_failed
        self.safe_next_step = safe_next_step


def absolute_launcher_path(file_path: Path) -> Path:
    """Return an absolute executable path without resolving symbolic links.

    The input comes from a project-owned virtual-environment launcher such as
    ``.venv/bin/python``. Preserving that launcher path is required because the
    interpreter uses it to set ``sys.prefix`` and select the correct
    site-packages directory.
    """

    return Path(os.path.abspath(os.fspath(file_path.expanduser())))


def environment_python(project_root: Path, environment_name: str) -> Path:
    """Return the executable launcher for one project virtual environment."""

    environment_launcher = absolute_launcher_path(
        project_root.expanduser() / environment_name / "bin" / "python"
    )
    if (
        not environment_launcher.exists()
        or not environment_launcher.is_file()
        or not os.access(environment_launcher, os.X_OK)
    ):
        raise RuntimeEnvironmentError(
            "Python environment validation failed.",
            str(environment_launcher),
            (
                "The required virtual-environment Python launcher is missing "
                "or not executable."
            ),
            f"Restore {environment_name} and rerun FastAPI verification.",
        )
    return environment_launcher


def require_active_environment(
    project_root: Path,
    environment_name: str,
    *,
    actual_prefix: Path | None = None,
) -> Path:
    """Require the current process to use the named project environment.

    ``sys.prefix`` is the reliable environment identity. Comparing resolved
    Python binaries is unsafe because virtual-environment launchers commonly
    point to the same base executable.
    """

    expected_prefix = absolute_launcher_path(
        project_root.expanduser() / environment_name
    )
    observed_prefix = absolute_launcher_path(actual_prefix or Path(sys.prefix))
    if not expected_prefix.exists() or not expected_prefix.is_dir():
        raise RuntimeEnvironmentError(
            "Python environment validation failed.",
            str(expected_prefix),
            "The expected virtual-environment directory does not exist.",
            f"Restore {environment_name} and rerun FastAPI verification.",
        )
    try:
        environment_matches = os.path.samefile(observed_prefix, expected_prefix)
    except OSError:
        environment_matches = observed_prefix.resolve() == expected_prefix.resolve()
    if not environment_matches:
        raise RuntimeEnvironmentError(
            "Python environment validation failed.",
            str(observed_prefix),
            (
                "The process is not running inside the required project "
                f"environment {expected_prefix}."
            ),
            (
                "Start the command with the project-owned launcher "
                f"{expected_prefix / 'bin' / 'python'} and rerun verification."
            ),
        )
    return expected_prefix
