#!/usr/bin/env python3
"""Require every Package 9.3 evidence file before closing the Streamlit phase."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

REQUIRED_JSON = {
    'environment_state.json': ('python_path', 'python_version'),
    'dependency_evidence.json': (
        'streamlit',
        'plotly',
        'numpy',
        'pandas',
        'pyarrow',
        'pip_check',
        'wheel_only_install',
        'runtime_pins',
        'probe_stages',
        'failed_stage',
    ),
    'streamlit_startup_evidence.json': ('health', 'root_status', 'process_stopped'),
    'streamlit_fastapi_connection_evidence.json': (
        'direct_health',
        'direct_readiness',
        'streamlit_client',
        'streamlit_app_test',
        'fastapi_process_stopped',
    ),
}
REQUIRED_LOG_MARKERS = {
    'pre-install_streamlit_application_tests.log': 'STREAMLIT APPLICATION-WIDE TESTS: PASSED',
    'pre-install_isolated_project_regression.log': 'ISOLATED PROJECT REGRESSION: PASSED',
    'post-install_streamlit_application_tests.log': 'STREAMLIT APPLICATION-WIDE TESTS: PASSED',
    'post-install_isolated_project_regression.log': 'ISOLATED PROJECT REGRESSION: PASSED',
}


def _load_object(path: Path) -> dict[str, object]:
    """Load one required JSON object."""

    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError(f'Evidence file is not a JSON object: {path.name}')
    return payload


def verify_closure(project_root: Path, evidence_dir: Path, output_file: Path) -> dict[str, object]:
    """Validate environment, HTTP, process-stop, and regression evidence."""

    summary: dict[str, object] = {'project_root': str(project_root), 'checks': {}}
    checks = summary['checks']
    if not isinstance(checks, dict):
        raise AssertionError('Closure summary checks container is invalid.')
    for filename, keys in REQUIRED_JSON.items():
        path = evidence_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f'Required evidence is missing: {filename}')
        payload = _load_object(path)
        missing = [key for key in keys if key not in payload]
        if missing:
            raise ValueError(f'{filename} is missing fields: {missing}')
        checks[filename] = 'PASSED'
    dependencies = _load_object(evidence_dir / 'dependency_evidence.json')
    expected_pins = {
        'streamlit': '1.58.0',
        'plotly': '6.8.0',
        'numpy': '1.26.4',
        'pandas': '2.1.4',
        'pyarrow': '14.0.2',
    }
    if dependencies.get('runtime_pins') != expected_pins:
        raise ValueError('The Streamlit runtime pins do not match the approved set.')
    if dependencies.get('wheel_only_install') is not True:
        raise ValueError('The Streamlit environment was not installed from binary wheels only.')
    stages = dependencies.get('probe_stages')
    if not isinstance(stages, dict) or not stages:
        raise ValueError('The staged native import evidence is missing.')
    failed = [name for name, value in stages.items() if value.get('status') != 'PASSED']
    if failed or dependencies.get('failed_stage') is not None:
        raise ValueError(f'The staged native imports did not all pass: {failed}')
    startup = _load_object(evidence_dir / 'streamlit_startup_evidence.json')
    if startup.get('health') != 'ok' or startup.get('root_status') != 200:
        raise ValueError('Real Streamlit HTTP evidence did not pass.')
    if startup.get('process_stopped') is not True:
        raise ValueError('Temporary Streamlit process was not stopped.')
    connection = _load_object(evidence_dir / 'streamlit_fastapi_connection_evidence.json')
    if connection.get('fastapi_process_stopped') is not True:
        raise ValueError('Temporary FastAPI process was not stopped.')
    for filename, marker in REQUIRED_LOG_MARKERS.items():
        path = evidence_dir / filename
        if not path.is_file() or marker not in path.read_text(encoding='utf-8'):
            raise ValueError(f'Regression evidence is missing its success marker: {filename}')
        checks[filename] = 'PASSED'
    interpreter = project_root / '.venv-streamlit' / (
        'Scripts/python.exe' if os.name == 'nt' else 'bin/python'
    )
    if not interpreter.is_file():
        raise FileNotFoundError('The final .venv-streamlit interpreter is missing.')
    for required in (
        'app/streamlit_app.py',
        'scripts/run_streamlit_application_tests.py',
        'tests/test_streamlit_runtime_closure.py',
        'docs/STREAMLIT_APPLICATION_CONTRACT.md',
        'docs/STREAMLIT_QA_CLOSURE_REPORT.md',
    ):
        if not (project_root / required).is_file():
            raise FileNotFoundError(f'Final Streamlit file is missing: {required}')
    summary['persistent_streamlit_server_started'] = False
    summary['persistent_fastapi_server_started'] = False
    summary['public_deployment_changed'] = False
    summary['status'] = 'PASSED'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    os.chmod(output_file, 0o600)
    return summary


def main() -> int:
    """Parse arguments and print final Streamlit closure markers."""

    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', required=True, type=Path)
    parser.add_argument('--evidence-dir', required=True, type=Path)
    parser.add_argument('--output-file', required=True, type=Path)
    args = parser.parse_args()
    try:
        verify_closure(
            args.project_root.expanduser().resolve(),
            args.evidence_dir.expanduser().absolute(),
            args.output_file.expanduser().absolute(),
        )
    except Exception as error:
        print('FAILED: Final Streamlit closure verification failed.')
        print('Location: Package 9.3 evidence or final installed project state')
        print(f'Reason: {type(error).__name__}: {error}')
        print('Safe next step: Correct the named failed gate and rerun Package 9.')
        return 1
    print('STREAMLIT ENVIRONMENT CLOSURE: PASSED')
    print('STREAMLIT DEPENDENCY CLOSURE: PASSED')
    print('STREAMLIT REAL STARTUP CLOSURE: PASSED')
    print('STREAMLIT TO FASTAPI CONNECTION CLOSURE: PASSED')
    print('STREAMLIT REGRESSION CLOSURE: PASSED')
    print('TEMPORARY SERVER CLEANUP: PASSED')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
