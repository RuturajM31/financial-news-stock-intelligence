#!/usr/bin/env python3
"""Run Streamlit application tests and the existing isolated project regression."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

APPLICATION_SUCCESS = 'STREAMLIT APPLICATION-WIDE TESTS: PASSED'
PROJECT_SUCCESS = 'ISOLATED PROJECT REGRESSION: PASSED'


class RegressionClosureError(RuntimeError):
    """Describe a missing runner, failed command, or absent success marker."""


def main_python(project_root: Path) -> Path:
    """Return the main project interpreter without resolving its venv link."""

    return project_root / '.venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')


def _run_and_capture(command: list[str], project_root: Path, evidence_file: Path) -> str:
    """Run one command, stream its output, and retain owner-only evidence."""

    process = subprocess.Popen(
        command,
        cwd=project_root,
        env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='replace',
    )
    if process.stdout is None:
        raise RegressionClosureError('The regression output stream is unavailable.')
    lines: list[str] = []
    for line in process.stdout:
        lines.append(line)
        print(line, end='', flush=True)
    exit_code = process.wait()
    output = ''.join(lines)
    evidence_file.parent.mkdir(parents=True, exist_ok=True)
    evidence_file.write_text(output, encoding='utf-8')
    os.chmod(evidence_file, 0o600)
    if exit_code != 0:
        raise RegressionClosureError(f'Regression command exited with code {exit_code}.')
    return output


def _verify_help(interpreter: Path, runner: Path, required: tuple[str, ...]) -> None:
    """Require the audited command-line options before running a project script."""

    completed = subprocess.run(
        [str(interpreter), str(runner), '--help'],
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        check=False,
    )
    if completed.returncode != 0:
        raise RegressionClosureError(f'Runner help failed: {runner}')
    help_text = completed.stdout + completed.stderr
    missing = [option for option in required if option not in help_text]
    if missing:
        raise RegressionClosureError(f'Runner {runner.name} is missing options: {missing}')


def run_regressions(project_root: Path, phase: str, evidence_dir: Path) -> None:
    """Run application-wide and isolated project regressions for one phase."""

    interpreter = main_python(project_root)
    if not interpreter.is_file():
        raise FileNotFoundError(f'Main project interpreter is missing: {interpreter}')
    app_runner = project_root / 'scripts/run_streamlit_application_tests.py'
    project_runner = project_root / 'scripts/run_isolated_project_regression.py'
    for runner in (app_runner, project_runner):
        if not runner.is_file() or runner.is_symlink():
            raise FileNotFoundError(f'Regression runner is missing or unsafe: {runner}')
    _verify_help(interpreter, app_runner, ('--project-root',))
    _verify_help(interpreter, project_runner, ('--project-root', '--phase'))
    app_output = _run_and_capture(
        [str(interpreter), str(app_runner), '--project-root', str(project_root)],
        project_root,
        evidence_dir / f'{phase}_streamlit_application_tests.log',
    )
    if APPLICATION_SUCCESS not in app_output:
        raise RegressionClosureError('Application tests did not print their success marker.')
    project_output = _run_and_capture(
        [
            str(interpreter),
            str(project_runner),
            '--project-root',
            str(project_root),
            '--phase',
            phase,
        ],
        project_root,
        evidence_dir / f'{phase}_isolated_project_regression.log',
    )
    if PROJECT_SUCCESS not in project_output:
        raise RegressionClosureError('Isolated regression did not print its success marker.')


def main() -> int:
    """Parse one phase, run both regressions, and print closure markers."""

    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', required=True, type=Path)
    parser.add_argument('--phase', required=True, choices=('pre-install', 'post-install'))
    parser.add_argument('--evidence-dir', required=True, type=Path)
    args = parser.parse_args()
    print(f'STARTED: Streamlit and project regression ({args.phase}).', flush=True)
    try:
        run_regressions(
            args.project_root.expanduser().resolve(),
            args.phase,
            args.evidence_dir.expanduser().absolute(),
        )
    except Exception as error:
        print('FAILED: Streamlit or project regression failed.')
        print(f'Location: {args.phase} application-wide or isolated project regression')
        print(f'Reason: {type(error).__name__}: {error}')
        print('Safe next step: Correct the named failed test or runner contract, then rerun Package 9.')
        return 1
    print(f'STREAMLIT APPLICATION REGRESSION ({args.phase}): PASSED')
    print(f'FULL PROJECT REGRESSION ({args.phase}): PASSED')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
