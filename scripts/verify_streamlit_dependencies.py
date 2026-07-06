#!/usr/bin/env python3
"""Install exact UI wheels and diagnose each native import separately.

Purpose
-------
The Streamlit environment must remain separate from analytics and model
runtimes. Package 9.3 installs exact binary wheels only, then runs every native
import in its own isolated Python process. This prevents a silent combined
probe from hiding which library failed on an older Intel macOS system.

Failure behaviour
-----------------
Each probe records its name, return code, signal, standard output, and standard
error before the next probe starts. If a native extension terminates Python,
the user receives the exact failed stage and signal instead of a blank reason.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import signal
import subprocess
import tempfile
from pathlib import Path
from typing import Any

EXPECTED_RUNTIME_PINS = {
    'streamlit': '1.58.0',
    'plotly': '6.8.0',
    'numpy': '1.26.4',
    'pandas': '2.1.4',
    'pyarrow': '14.0.2',
}
FORBIDDEN_MODEL_PACKAGES = ('torch', 'transformers', 'sklearn', 'joblib')
RUNTIME_REQUIREMENTS_NAME = 'requirements-streamlit-runtime.txt'
PROBE_TIMEOUT_SECONDS = 60


def _environment_python(state: dict[str, object]) -> Path:
    """Return the recorded venv interpreter without resolving its symlink."""

    value = state.get('python_path')
    if not isinstance(value, str) or not value:
        raise ValueError('Environment state does not contain python_path.')
    return Path(value)


def _read_exact_pins(path: Path) -> dict[str, str]:
    """Read exact name-version pins while ignoring comments and blank lines."""

    pins: dict[str, str] = {}
    for raw_line in path.read_text(encoding='utf-8').splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#'):
            continue
        if '==' not in line or line.count('==') != 1:
            raise ValueError(f'Requirement must be an exact pin: {line}')
        name, version = (part.strip() for part in line.split('==', 1))
        if not name or not version:
            raise ValueError(f'Requirement is incomplete: {line}')
        normalized = name.lower().replace('_', '-')
        if normalized in pins:
            raise ValueError(f'Duplicate requirement: {normalized}')
        pins[normalized] = version
    return pins


def _clean_environment() -> dict[str, str]:
    """Return an environment without Python-path or dynamic-loader leakage."""

    blocked = {
        'PYTHONPATH',
        'PYTHONHOME',
        'VIRTUAL_ENV',
        'CONDA_PREFIX',
        '_OLD_VIRTUAL_PATH',
        'DYLD_INSERT_LIBRARIES',
        'DYLD_LIBRARY_PATH',
        'DYLD_FALLBACK_LIBRARY_PATH',
        'LD_PRELOAD',
        'LD_LIBRARY_PATH',
        'KMP_DUPLICATE_LIB_OK',
    }
    cleaned = {key: value for key, value in os.environ.items() if key not in blocked}
    cleaned.update(
        {
            'PIP_ONLY_BINARY': ':all:',
            'PYTHONFAULTHANDLER': '1',
            'PYTHONNOUSERSITE': '1',
            'PYTHONDONTWRITEBYTECODE': '1',
        }
    )
    return cleaned


def _run_pip(command: list[str], failure_message: str) -> str:
    """Run one pip command and raise a short error with its output tail."""

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        check=False,
        env=_clean_environment(),
    )
    output = completed.stdout + '\n' + completed.stderr
    if completed.returncode != 0:
        raise RuntimeError(f'{failure_message} Output tail: {output[-4000:]}')
    return output


def _install_when_new(
    interpreter: Path,
    requirements: Path,
    created_by_this_run: bool,
) -> dict[str, object]:
    """Download compatible wheels first, then install offline from the wheelhouse."""

    if not created_by_this_run:
        return {
            'wheel_only_install': True,
            'wheelhouse_file_count': 0,
            'wheel_files': [],
            'installation_mode': 'reused',
        }

    with tempfile.TemporaryDirectory(prefix='streamlit_wheelhouse_') as directory:
        wheelhouse = Path(directory)
        _run_pip(
            [
                str(interpreter),
                '-m',
                'pip',
                'download',
                '--disable-pip-version-check',
                '--no-input',
                '--only-binary=:all:',
                '--dest',
                str(wheelhouse),
                '--requirement',
                str(requirements),
            ],
            'Compatible binary wheels could not be downloaded.',
        )
        wheel_files = sorted(path for path in wheelhouse.iterdir() if path.is_file())
        if not wheel_files:
            raise RuntimeError('The binary wheel preflight downloaded no files.')
        non_wheels = [path.name for path in wheel_files if path.suffix != '.whl']
        if non_wheels:
            raise RuntimeError(f'Non-wheel files were downloaded: {non_wheels}')
        pyarrow_prefix = f"pyarrow-{EXPECTED_RUNTIME_PINS['pyarrow']}-"
        if not any(path.name.lower().startswith(pyarrow_prefix) for path in wheel_files):
            raise RuntimeError('The approved PyArrow wheel was not downloaded.')
        _run_pip(
            [
                str(interpreter),
                '-m',
                'pip',
                'install',
                '--disable-pip-version-check',
                '--no-input',
                '--no-user',
                '--no-index',
                '--find-links',
                str(wheelhouse),
                '--only-binary=:all:',
                '--requirement',
                str(requirements),
            ],
            'Offline installation from the verified wheelhouse failed.',
        )
        return {
            'wheel_only_install': True,
            'wheelhouse_file_count': len(wheel_files),
            'wheel_files': [path.name for path in wheel_files],
            'installation_mode': 'new_binary_wheelhouse',
        }


def _signal_name(returncode: int) -> str | None:
    """Translate a negative subprocess return code into a signal name."""

    if returncode >= 0:
        return None
    number = -returncode
    try:
        return signal.Signals(number).name
    except ValueError:
        return f'SIGNAL_{number}'


def _write_partial_evidence(path: Path, evidence: dict[str, Any]) -> None:
    """Persist diagnostic evidence even when a later native import fails."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    os.chmod(path, 0o600)


def _run_probe_stage(
    interpreter: Path,
    stage_name: str,
    code: str,
    evidence: dict[str, Any],
    evidence_file: Path,
) -> dict[str, Any]:
    """Run one isolated import stage and retain complete failure diagnostics."""

    with tempfile.TemporaryDirectory(prefix='streamlit_native_probe_') as directory:
        try:
            completed = subprocess.run(
                [str(interpreter), '-I', '-X', 'faulthandler', '-c', code],
                cwd=directory,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                check=False,
                timeout=PROBE_TIMEOUT_SECONDS,
                env=_clean_environment(),
            )
        except subprocess.TimeoutExpired as error:
            record = {
                'status': 'FAILED',
                'returncode': None,
                'signal': None,
                'stdout': (error.stdout or '')[-2000:],
                'stderr': (error.stderr or '')[-2000:],
                'timeout_seconds': PROBE_TIMEOUT_SECONDS,
            }
            evidence['probe_stages'][stage_name] = record
            evidence['failed_stage'] = stage_name
            _write_partial_evidence(evidence_file, evidence)
            raise RuntimeError(
                f"Native probe '{stage_name}' timed out after {PROBE_TIMEOUT_SECONDS} seconds."
            ) from error

    record = {
        'status': 'PASSED' if completed.returncode == 0 else 'FAILED',
        'returncode': completed.returncode,
        'signal': _signal_name(completed.returncode),
        'stdout': completed.stdout[-2000:],
        'stderr': completed.stderr[-4000:],
    }
    evidence['probe_stages'][stage_name] = record
    _write_partial_evidence(evidence_file, evidence)
    if completed.returncode != 0:
        signal_text = f", signal {record['signal']}" if record['signal'] else ''
        detail = completed.stderr.strip() or completed.stdout.strip() or 'No text was produced.'
        raise RuntimeError(
            f"Native probe '{stage_name}' failed with return code {completed.returncode}"
            f'{signal_text}. Output tail: {detail[-3000:]}'
        )
    payload = json.loads(completed.stdout.strip())
    if not isinstance(payload, dict):
        raise RuntimeError(f"Native probe '{stage_name}' did not return a JSON object.")
    return payload


def _probe(
    interpreter: Path,
    project_root: Path,
    evidence_file: Path,
) -> dict[str, object]:
    """Run staged imports, exact version checks, pip check, and Arrow conversion."""

    evidence: dict[str, Any] = {
        'platform': {
            'system': platform.system(),
            'machine': platform.machine(),
            'release': platform.release(),
        },
        'project_root_exists': project_root.is_dir(),
        'probe_stages': {},
    }
    stages = (
        (
            'numpy_import',
            "import json,numpy as value; print(json.dumps({'numpy': value.__version__}))",
        ),
        (
            'pandas_import',
            "import json,pandas as value; print(json.dumps({'pandas': value.__version__}))",
        ),
        (
            'pyarrow_import',
            "import json,pyarrow as value; print(json.dumps({'pyarrow': value.__version__}))",
        ),
        (
            'streamlit_import',
            "import json,streamlit as value; print(json.dumps({'streamlit': value.__version__}))",
        ),
        (
            'plotly_import',
            "import json,plotly as value; print(json.dumps({'plotly': value.__version__}))",
        ),
        (
            'pandas_pyarrow_round_trip',
            "import json,pandas as pd,pyarrow as pa; "
            "frame=pd.DataFrame({'label':['Down','Flat','Up'],'value':[0.2,0.3,0.5]}); "
            "table=pa.Table.from_pandas(frame,preserve_index=False); "
            "result=table.to_pandas(); "
            "print(json.dumps({'pyarrow_rows':table.num_rows,'round_trip_rows':len(result)}))",
        ),
        (
            'model_runtime_isolation',
            "import importlib.util,json; names=('torch','transformers','sklearn','joblib'); "
            "print(json.dumps({'forbidden_model_packages':"
            "{name:importlib.util.find_spec(name) is not None for name in names}}))",
        ),
    )
    for stage_name, code in stages:
        payload = _run_probe_stage(
            interpreter,
            stage_name,
            code,
            evidence,
            evidence_file,
        )
        evidence.update(payload)

    for name, expected in EXPECTED_RUNTIME_PINS.items():
        if evidence.get(name) != expected:
            raise RuntimeError(
                f'Installed {name} version {evidence.get(name)!r} does not match {expected!r}.'
            )
    if evidence.get('pyarrow_rows') != 3 or evidence.get('round_trip_rows') != 3:
        raise RuntimeError('The PyArrow dataframe round-trip returned the wrong row count.')
    forbidden = evidence.get('forbidden_model_packages')
    if not isinstance(forbidden, dict) or any(bool(value) for value in forbidden.values()):
        raise RuntimeError('A model runtime was found inside .venv-streamlit.')
    check = subprocess.run(
        [str(interpreter), '-I', '-m', 'pip', 'check'],
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        check=False,
        env=_clean_environment(),
    )
    if check.returncode != 0:
        detail = check.stdout.strip() or check.stderr.strip()
        raise RuntimeError('pip check found incompatible Streamlit dependencies: ' + detail)
    evidence['pip_check'] = 'PASSED'
    evidence['failed_stage'] = None
    return evidence


def verify_dependencies(
    project_root: Path,
    state_file: Path,
    evidence_file: Path,
) -> dict[str, object]:
    """Install when new, verify exact dependencies, and retain evidence."""

    state = json.loads(state_file.read_text(encoding='utf-8'))
    requirements = project_root / RUNTIME_REQUIREMENTS_NAME
    pins = _read_exact_pins(requirements)
    if pins != EXPECTED_RUNTIME_PINS:
        raise ValueError(f'Unexpected Streamlit runtime pins: {pins}')
    interpreter = _environment_python(state)
    installation = _install_when_new(
        interpreter,
        requirements,
        bool(state.get('created_by_this_run')),
    )
    initial: dict[str, Any] = {
        **installation,
        'runtime_pins': pins,
        'environment_created_by_this_run': bool(state.get('created_by_this_run')),
    }
    _write_partial_evidence(evidence_file, initial)
    evidence = _probe(interpreter, project_root, evidence_file)
    evidence.update(initial)
    _write_partial_evidence(evidence_file, evidence)
    return evidence


def main() -> int:
    """Parse arguments and print stable dependency markers."""

    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', required=True, type=Path)
    parser.add_argument('--state-file', required=True, type=Path)
    parser.add_argument('--evidence-file', required=True, type=Path)
    args = parser.parse_args()
    try:
        evidence = verify_dependencies(
            args.project_root.expanduser().resolve(),
            args.state_file.expanduser().absolute(),
            args.evidence_file.expanduser().absolute(),
        )
    except Exception as error:
        print('FAILED: Streamlit dependency verification failed.')
        print('Location: .venv-streamlit staged native import or wheel verification')
        print(f'Reason: {type(error).__name__}: {error}')
        print('Safe next step: Use the named failed stage and signal, then rerun Package 9.3.')
        return 1
    print('STREAMLIT BINARY WHEEL PREFLIGHT: PASSED')
    print('STREAMLIT STAGED NATIVE IMPORTS: PASSED')
    print('STREAMLIT DEPENDENCIES: PASSED')
    print('PYARROW DATAFRAME PROBE: PASSED')
    print('MODEL RUNTIME ISOLATION: PASSED')
    print(f"STREAMLIT VERSION: {evidence['streamlit']}")
    print(f"PLOTLY VERSION: {evidence['plotly']}")
    print(f"NUMPY VERSION: {evidence['numpy']}")
    print(f"PANDAS VERSION: {evidence['pandas']}")
    print(f"PYARROW VERSION: {evidence['pyarrow']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
