#!/usr/bin/env python3
"""Verify the real Streamlit API client against a temporary FastAPI server."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


class ConnectionClosureError(RuntimeError):
    """Describe a temporary FastAPI or Streamlit-client connection failure."""


def free_loopback_port() -> int:
    """Reserve and release one local port for the temporary API."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(('127.0.0.1', 0))
        return int(sock.getsockname()[1])


def _request_json(url: str, timeout_seconds: float = 5.0) -> dict[str, object]:
    """Read one local JSON object."""

    request = urllib.request.Request(url, headers={'Accept': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        payload = json.loads(response.read().decode('utf-8'))
    if not isinstance(payload, dict):
        raise ConnectionClosureError(f'Endpoint did not return a JSON object: {url}')
    return payload


def _main_python(project_root: Path) -> Path:
    """Return the main analytics interpreter without resolving the venv link."""

    return project_root / '.venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')


def _streamlit_python(project_root: Path) -> Path:
    """Return the isolated Streamlit interpreter without resolving the venv link."""

    return project_root / '.venv-streamlit' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')


def _fastapi_environment(project_root: Path) -> dict[str, str]:
    """Build a clean server environment with the project source import path.

    The FastAPI runner is a script inside ``scripts/``. Running that script
    directly does not automatically add ``PROJECT_ROOT/src`` to Python's
    import path. The application package lives below that source folder, so
    the temporary server must receive the source root explicitly.
    """

    source_root = project_root / 'src'
    package_root = source_root / 'financial_news_intelligence'
    if not package_root.is_dir():
        raise FileNotFoundError(
            f'Financial News Intelligence source package is missing: {package_root}'
        )
    environment = dict(os.environ)
    for key in ('PYTHONPATH', 'PYTHONHOME', 'PYTHONSTARTUP'):
        environment.pop(key, None)
    environment.update(
        {
            'FNI_PROJECT_ROOT': str(project_root),
            'FNI_API_ENVIRONMENT': 'test',
            'FNI_REQUIRE_API_KEY': 'false',
            'FNI_TRUSTED_HOSTS': '127.0.0.1,localhost',
            'FNI_DEEP_READINESS_PROBE': 'false',
            'PYTHONNOUSERSITE': '1',
            'PYTHONDONTWRITEBYTECODE': '1',
            'PYTHONPATH': os.pathsep.join((str(source_root), str(project_root))),
        }
    )
    return environment


def _client_probe(project_root: Path, base_url: str) -> dict[str, object]:
    """Call health and readiness through the real Streamlit client module."""

    code = r"""
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
sys.path.insert(0, str(root))
os.environ['FNI_STREAMLIT_API_BASE_URL'] = sys.argv[2]
from app.services.api_client import FinancialNewsApiClient
from app.services.api_settings import get_api_client_settings

client = FinancialNewsApiClient(get_api_client_settings(environment=os.environ))
health = client.health()
readiness = client.readiness()
print(json.dumps({
    'health_status': health.status,
    'readiness_status': readiness.status,
    'readiness_components': dict(readiness.components),
}, sort_keys=True))
"""
    completed = subprocess.run(
        [str(_streamlit_python(project_root)), '-c', code, str(project_root), base_url],
        cwd=project_root,
        env={
            **os.environ,
            'PYTHONPATH': str(project_root),
            'PYTHONNOUSERSITE': '1',
            'PYTHONDONTWRITEBYTECODE': '1',
        },
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        check=False,
    )
    if completed.returncode != 0:
        raise ConnectionClosureError(
            'The real Streamlit API client probe failed: '
            + (completed.stderr.strip() or completed.stdout.strip())
        )
    payload = json.loads(completed.stdout.strip())
    if payload.get('health_status') != 'PASSED' or payload.get('readiness_status') != 'PASSED':
        raise ConnectionClosureError(f'Streamlit client returned failed status: {payload}')
    return payload


def _app_test_probe(project_root: Path, base_url: str) -> dict[str, object]:
    """Run the real Streamlit script with AppTest and require no exception."""

    code = r"""
import json
import os
import sys
from pathlib import Path
from streamlit.testing.v1 import AppTest

root = Path(sys.argv[1])
os.environ['FNI_STREAMLIT_API_BASE_URL'] = sys.argv[2]
sys.path.insert(0, str(root))
app = AppTest.from_file(str(root / 'app/streamlit_app.py'), default_timeout=60)
app.run()
exceptions = [str(item.value) for item in app.exception]
print(json.dumps({'exception_count': len(exceptions), 'exceptions': exceptions}, sort_keys=True))
"""
    completed = subprocess.run(
        [str(_streamlit_python(project_root)), '-c', code, str(project_root), base_url],
        cwd=project_root,
        env={
            **os.environ,
            'PYTHONPATH': str(project_root),
            'PYTHONNOUSERSITE': '1',
            'PYTHONDONTWRITEBYTECODE': '1',
        },
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        check=False,
    )
    if completed.returncode != 0:
        raise ConnectionClosureError(
            'Streamlit AppTest failed: '
            + (completed.stderr.strip() or completed.stdout.strip())
        )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    if payload.get('exception_count') != 0:
        raise ConnectionClosureError(f'Streamlit AppTest reported exceptions: {payload}')
    return payload


def verify_connection(project_root: Path, evidence_file: Path, timeout_seconds: int) -> dict[str, object]:
    """Start FastAPI, verify direct and Streamlit-client calls, then stop it."""

    main_python = _main_python(project_root)
    streamlit_python = _streamlit_python(project_root)
    for interpreter in (main_python, streamlit_python):
        if not interpreter.is_file():
            raise FileNotFoundError(f'Required interpreter is missing: {interpreter}')
    runner = project_root / 'scripts/run_fastapi_application.py'
    if not runner.is_file() or runner.is_symlink():
        raise FileNotFoundError(f'FastAPI runner is missing or unsafe: {runner}')
    port = free_loopback_port()
    base_url = f'http://127.0.0.1:{port}'
    command = [
        str(main_python),
        str(runner),
        '--project-root',
        str(project_root),
        '--host',
        '127.0.0.1',
        '--port',
        str(port),
        '--log-level',
        'warning',
    ]
    environment = _fastapi_environment(project_root)
    process = subprocess.Popen(
        command,
        cwd=project_root,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='replace',
    )
    output: list[str] = []
    evidence: dict[str, object] = {'port': port, 'base_url': base_url}
    try:
        deadline = time.monotonic() + timeout_seconds
        last_error = 'FastAPI did not answer yet.'
        while time.monotonic() < deadline:
            if process.poll() is not None:
                if process.stdout is not None:
                    output.append(process.stdout.read())
                raise ConnectionClosureError(
                    'FastAPI stopped before health verification. '
                    f'Exit code {process.returncode}. Output tail: {"".join(output)[-3000:]}'
                )
            try:
                health = _request_json(f'{base_url}/health')
                if health.get('status') == 'PASSED':
                    evidence['direct_health'] = health
                    break
                last_error = f'Unexpected health response: {health}'
            except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
                last_error = f'{type(error).__name__}: {error}'
            time.sleep(0.25)
        else:
            raise ConnectionClosureError(
                f'FastAPI did not become healthy within {timeout_seconds} seconds. Last error: {last_error}'
            )
        ready = _request_json(f'{base_url}/ready')
        if ready.get('status') != 'PASSED':
            raise ConnectionClosureError(f'FastAPI readiness failed: {ready}')
        evidence['direct_readiness'] = ready
        evidence['streamlit_client'] = _client_probe(project_root, base_url)
        evidence['streamlit_app_test'] = _app_test_probe(project_root, base_url)
        return evidence
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        if process.stdout is not None:
            remaining = process.stdout.read()
            if remaining:
                output.append(remaining)
        evidence['fastapi_process_stopped'] = process.poll() is not None
        evidence['server_output_tail'] = ''.join(output)[-3000:]
        evidence_file.parent.mkdir(parents=True, exist_ok=True)
        evidence_file.write_text(json.dumps(evidence, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        os.chmod(evidence_file, 0o600)
        if not evidence['fastapi_process_stopped']:
            raise ConnectionClosureError('The temporary FastAPI process did not stop.')


def main() -> int:
    """Parse arguments and verify the real local connection."""

    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', required=True, type=Path)
    parser.add_argument('--evidence-file', required=True, type=Path)
    parser.add_argument('--startup-timeout-seconds', type=int, default=120)
    args = parser.parse_args()
    print('STARTED: Real Streamlit-to-FastAPI connection test.', flush=True)
    try:
        evidence = verify_connection(
            args.project_root.expanduser().resolve(),
            args.evidence_file.expanduser().absolute(),
            args.startup_timeout_seconds,
        )
    except Exception as error:
        print('FAILED: Streamlit-to-FastAPI connection test failed.')
        print('Location: Temporary FastAPI server, Streamlit API client, or Streamlit AppTest')
        print(f'Reason: {type(error).__name__}: {error}')
        print('Safe next step: Inspect the connection evidence, correct the named issue, and rerun Package 9.')
        return 1
    if not evidence.get('fastapi_process_stopped'):
        print('FAILED: Temporary FastAPI server remained running.')
        return 1
    print('FASTAPI HEALTH OVER HTTP: PASSED')
    print('FASTAPI READINESS OVER HTTP: PASSED')
    print('STREAMLIT API CLIENT CONNECTION: PASSED')
    print('STREAMLIT APP RUNTIME TEST: PASSED')
    print('TEMPORARY FASTAPI SERVER STOPPED: TRUE')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
