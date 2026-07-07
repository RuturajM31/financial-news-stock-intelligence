#!/usr/bin/env python3
"""Run the installed Streamlit application-wide test suite predictably.

The script keeps the test command in the repository so later CI, containers,
and deployment checks can reuse the exact same test selection. It does not
start Streamlit, FastAPI, or a model worker; real startup belongs to Package 9.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

TEST_FILES = (
    "tests/test_streamlit_page_contracts.py",
    "tests/test_streamlit_navigation_branding.py",
    "tests/test_streamlit_api_security.py",
    "tests/test_streamlit_chart_explanations.py",
    "tests/test_streamlit_3d_accessibility.py",
    "tests/test_streamlit_report_downloads.py",
    "tests/test_streamlit_session_privacy.py",
    "tests/test_streamlit_provenance_redaction.py",
    "tests/test_streamlit_accessibility_contracts.py",
)


def validate_project_root(project_root: Path) -> None:
    """Fail closed unless the required application and test files exist."""

    required = (
        "app/streamlit_app.py",
        "app/pages/about_ruturaj.py",
        "app/services/api_client.py",
        "app/styles/premium_theme.css",
        *TEST_FILES,
    )
    missing = [relative for relative in required if not (project_root / relative).is_file()]
    if missing:
        raise FileNotFoundError("Missing Streamlit application test files: " + ", ".join(missing))


def run(project_root: Path) -> int:
    """Execute the fixed application test list with project-local imports."""

    validate_project_root(project_root)
    environment = dict(os.environ)
    environment["STREAMLIT_PROJECT_ROOT"] = str(project_root)
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(project_root) + (os.pathsep + existing if existing else "")
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        *TEST_FILES,
    ]
    completed = subprocess.run(command, cwd=project_root, env=environment, check=False)
    return completed.returncode


def main() -> int:
    """Parse the project root and print stable test-suite status markers."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True, type=Path)
    args = parser.parse_args()
    try:
        return_code = run(args.project_root.expanduser().resolve())
    except Exception as error:
        print(f"FAILED: Streamlit application test runner: {type(error).__name__}: {error}")
        return 1
    if return_code != 0:
        print("FAILED: STREAMLIT APPLICATION-WIDE TESTS")
        return return_code
    print("STREAMLIT APPLICATION-WIDE TESTS: PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
