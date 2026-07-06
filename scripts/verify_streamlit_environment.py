#!/usr/bin/env python3
"""Create or validate the isolated Streamlit environment without guessing.

Purpose
-------
Keep Streamlit, Plotly, pandas, and pyarrow separate from the analytics and
DistilBERT environments. The script creates ``.venv-streamlit`` only when it
is absent. An existing environment is inspected but never rewritten.

Failure behaviour
-----------------
A newly created environment is recorded in an owner-only state file so the
strike wrapper can remove it if a later verification step fails. Existing
environments remain untouched on both success and failure.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ENVIRONMENT_NAME = '.venv-streamlit'
MINIMUM_PYTHON = (3, 10)


def environment_python(project_root: Path) -> Path:
    """Return the venv interpreter path without resolving its symlink."""

    if os.name == 'nt':
        return project_root / ENVIRONMENT_NAME / 'Scripts' / 'python.exe'
    return project_root / ENVIRONMENT_NAME / 'bin' / 'python'


def _run_version(interpreter: Path) -> tuple[int, int, int]:
    """Read the exact Python version from one interpreter."""

    completed = subprocess.run(
        [str(interpreter), '-c', 'import json,sys; print(json.dumps(list(sys.version_info[:3])))'],
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            'The Streamlit environment Python could not run: '
            + (completed.stderr.strip() or completed.stdout.strip())
        )
    values = json.loads(completed.stdout.strip())
    if not isinstance(values, list) or len(values) != 3:
        raise RuntimeError('The Streamlit Python version response was invalid.')
    return tuple(int(value) for value in values)


def ensure_environment(project_root: Path, state_file: Path) -> dict[str, object]:
    """Create the environment when absent and return its rollback state."""

    environment_root = project_root / ENVIRONMENT_NAME
    created = False
    if environment_root.exists() and not environment_root.is_dir():
        raise FileExistsError(f'{environment_root} exists but is not a directory.')
    if not environment_root.exists():
        completed = subprocess.run(
            [sys.executable, '-m', 'venv', str(environment_root)],
            cwd=project_root,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                'Python could not create .venv-streamlit: '
                + (completed.stderr.strip() or completed.stdout.strip())
            )
        created = True
    interpreter = environment_python(project_root)
    if not interpreter.is_file():
        raise FileNotFoundError(f'Streamlit environment Python is missing: {interpreter}')
    version = _run_version(interpreter)
    if version < MINIMUM_PYTHON:
        raise RuntimeError(
            f'.venv-streamlit uses Python {version[0]}.{version[1]}.{version[2]}, '
            'but Python 3.10 or newer is required.'
        )
    state = {
        'created_by_this_run': created,
        'environment_root': str(environment_root),
        'python_path': str(interpreter),
        'python_version': '.'.join(str(part) for part in version),
    }
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(state, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    os.chmod(state_file, 0o600)
    return state


def main() -> int:
    """Parse arguments, create or inspect the environment, and print markers."""

    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', required=True, type=Path)
    parser.add_argument('--state-file', required=True, type=Path)
    args = parser.parse_args()
    try:
        state = ensure_environment(
            args.project_root.expanduser().resolve(),
            args.state_file.expanduser().absolute(),
        )
    except Exception as error:
        print('FAILED: Streamlit environment verification failed.')
        print('Location: .venv-streamlit creation or interpreter check')
        print(f'Reason: {type(error).__name__}: {error}')
        print('Safe next step: Correct the named Python or environment problem, then rerun Package 9.')
        return 1
    action = 'CREATED' if state['created_by_this_run'] else 'REUSED WITHOUT CHANGES'
    print(f'STREAMLIT ENVIRONMENT: PASSED ({action})')
    print(f"STREAMLIT PYTHON: {state['python_path']}")
    print(f"STREAMLIT PYTHON VERSION: {state['python_version']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
