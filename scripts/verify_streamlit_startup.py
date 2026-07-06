#!/usr/bin/env python3
"""Start the real Streamlit server locally, verify HTTP, and stop it cleanly."""

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


class StreamlitStartupError(RuntimeError):
    """Describe a real Streamlit startup or shutdown failure."""


def free_loopback_port() -> int:
    """Reserve and release one loopback port for the short-lived server."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(('127.0.0.1', 0))
        return int(sock.getsockname()[1])


def environment_python(project_root: Path) -> Path:
    """Return the Streamlit interpreter without resolving its venv symlink."""

    return project_root / '.venv-streamlit' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')


def _request_text(url: str, timeout_seconds: float = 5.0) -> tuple[int, str]:
    """Read one local HTTP response as UTF-8 text."""

    request = urllib.request.Request(url, headers={'Accept': 'text/html,*/*'})
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return int(response.status), response.read().decode('utf-8', errors='replace')


def verify_startup(project_root: Path, evidence_file: Path, timeout_seconds: int) -> dict[str, object]:
    """Run Streamlit, verify health and root HTML, and prove clean shutdown."""

    interpreter = environment_python(project_root)
    if not interpreter.is_file():
        raise FileNotFoundError(f'Streamlit interpreter is missing: {interpreter}')
    port = free_loopback_port()
    command = [
        str(interpreter),
        '-m',
        'streamlit',
        'run',
        str(project_root / 'app/streamlit_app.py'),
        '--server.headless=true',
        '--server.address=127.0.0.1',
        f'--server.port={port}',
        '--server.fileWatcherType=none',
        '--browser.gatherUsageStats=false',
    ]
    environment = {
        **os.environ,
        'PYTHONPATH': str(project_root),
        'PYTHONNOUSERSITE': '1',
        'PYTHONDONTWRITEBYTECODE': '1',
    }
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
    evidence: dict[str, object] = {'port': port, 'health': None, 'root_status': None}
    try:
        deadline = time.monotonic() + timeout_seconds
        last_error = 'The server did not answer yet.'
        while time.monotonic() < deadline:
            if process.poll() is not None:
                if process.stdout is not None:
                    output.append(process.stdout.read())
                raise StreamlitStartupError(
                    'Streamlit stopped before health verification. '
                    f'Exit code {process.returncode}. Output tail: {"".join(output)[-3000:]}'
                )
            try:
                health_status, health_body = _request_text(
                    f'http://127.0.0.1:{port}/_stcore/health'
                )
                if health_status == 200 and health_body.strip().lower() == 'ok':
                    evidence['health'] = health_body.strip()
                    break
                last_error = f'Unexpected health response: {health_status} {health_body!r}'
            except (OSError, urllib.error.URLError) as error:
                last_error = f'{type(error).__name__}: {error}'
            time.sleep(0.25)
        else:
            raise StreamlitStartupError(
                f'Streamlit did not become healthy within {timeout_seconds} seconds. Last error: {last_error}'
            )
        root_status, root_html = _request_text(f'http://127.0.0.1:{port}/')
        if root_status != 200 or '<html' not in root_html.lower():
            raise StreamlitStartupError('Streamlit root did not return an HTML page.')
        evidence['root_status'] = root_status
        evidence['root_contains_streamlit_shell'] = 'streamlit' in root_html.lower()
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
        evidence['process_stopped'] = process.poll() is not None
        evidence['output_tail'] = ''.join(output)[-3000:]
        evidence_file.parent.mkdir(parents=True, exist_ok=True)
        evidence_file.write_text(json.dumps(evidence, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        os.chmod(evidence_file, 0o600)
        if not evidence['process_stopped']:
            raise StreamlitStartupError('The temporary Streamlit process did not stop.')


def main() -> int:
    """Parse arguments and run the real local Streamlit smoke test."""

    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', required=True, type=Path)
    parser.add_argument('--evidence-file', required=True, type=Path)
    parser.add_argument('--startup-timeout-seconds', type=int, default=90)
    args = parser.parse_args()
    print('STARTED: Real local Streamlit startup test.', flush=True)
    try:
        evidence = verify_startup(
            args.project_root.expanduser().resolve(),
            args.evidence_file.expanduser().absolute(),
            args.startup_timeout_seconds,
        )
    except Exception as error:
        print('FAILED: Real Streamlit startup test failed.')
        print('Location: Temporary Streamlit server or local HTTP health endpoint')
        print(f'Reason: {type(error).__name__}: {error}')
        print('Safe next step: Inspect the startup evidence, correct the named issue, and rerun Package 9.')
        return 1
    if not evidence.get('process_stopped'):
        print('FAILED: Temporary Streamlit server remained running.')
        return 1
    print('REAL STREAMLIT STARTUP: PASSED')
    print('STREAMLIT HTTP HEALTH: PASSED')
    print('TEMPORARY STREAMLIT SERVER STOPPED: TRUE')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
