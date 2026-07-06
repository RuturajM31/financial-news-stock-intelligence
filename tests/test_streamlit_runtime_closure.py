"""Verify Package 9.3 runtime scripts, evidence rules, and safety boundaries."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(
    os.environ.get('STREAMLIT_PROJECT_ROOT', Path(__file__).resolve().parents[1])
).expanduser().resolve()
SOURCE_ROOT = Path(
    os.environ.get('STREAMLIT_CLOSURE_SOURCE_ROOT', PROJECT_ROOT)
).expanduser().resolve()


def _load(relative: str):
    """Load one installed Package 9.3 script by project-relative path."""

    path = SOURCE_ROOT / relative
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Could not load {relative}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_streamlit_runtime_requirements_are_exact_and_wheel_compatible() -> None:
    """Pin the compatible Streamlit, Plotly, NumPy, pandas, and PyArrow set."""

    dependency = _load('scripts/verify_streamlit_dependencies.py')
    pins = dependency._read_exact_pins(SOURCE_ROOT / 'requirements-streamlit-runtime.txt')
    assert pins == {
        'streamlit': '1.58.0',
        'plotly': '6.8.0',
        'numpy': '1.26.4',
        'pandas': '2.1.4',
        'pyarrow': '14.0.2',
    }


def test_dependency_install_forbids_source_builds() -> None:
    """Require a binary-only wheel preflight followed by offline installation."""

    text = (SOURCE_ROOT / 'scripts/verify_streamlit_dependencies.py').read_text(encoding='utf-8')
    for fragment in (
        "'download'",
        "'--only-binary=:all:'",
        "'--no-index'",
        "'--find-links'",
        "PIP_ONLY_BINARY",
    ):
        assert fragment in text
    assert "'pip',\n                'install'" in text
    assert 'pip wheel' not in text



def test_binary_wheel_install_downloads_then_installs_offline(monkeypatch, tmp_path: Path) -> None:
    """Exercise the two-stage wheelhouse path without making a network request."""

    dependency = _load('scripts/verify_streamlit_dependencies.py')
    interpreter = tmp_path / 'python'
    interpreter.write_text('', encoding='utf-8')
    requirements = tmp_path / 'requirements.txt'
    requirements.write_text('streamlit==1.58.0\n', encoding='utf-8')
    commands: list[list[str]] = []

    class Completed:
        """Provide the subprocess fields used by the installer helper."""

        returncode = 0
        stdout = ''
        stderr = ''

    def fake_run(command, **_kwargs):
        """Record pip commands and create one wheel after the download stage."""

        values = [str(value) for value in command]
        commands.append(values)
        if 'download' in values:
            destination = Path(values[values.index('--dest') + 1])
            (destination / 'streamlit-1.58.0-py3-none-any.whl').write_bytes(b'wheel')
            (destination / 'pyarrow-14.0.2-cp310-cp310-macosx_10_15_x86_64.whl').write_bytes(
                b'wheel'
            )
        return Completed()

    monkeypatch.setattr(dependency.subprocess, 'run', fake_run)
    evidence = dependency._install_when_new(interpreter, requirements, True)
    assert evidence['wheel_only_install'] is True
    assert evidence['wheelhouse_file_count'] == 2
    assert 'download' in commands[0]
    assert '--only-binary=:all:' in commands[0]
    assert 'install' in commands[1]
    assert '--no-index' in commands[1]
    assert '--find-links' in commands[1]


def test_existing_environment_skips_dependency_install(monkeypatch, tmp_path: Path) -> None:
    """Never change an environment that existed before the strike run."""

    dependency = _load('scripts/verify_streamlit_dependencies.py')

    def unexpected_run(*_args, **_kwargs):
        """Fail when a pip command is attempted for an existing environment."""

        raise AssertionError('pip must not run for an existing environment')

    monkeypatch.setattr(dependency.subprocess, 'run', unexpected_run)
    result = dependency._install_when_new(tmp_path / 'python', tmp_path / 'requirements.txt', False)
    assert result['installation_mode'] == 'reused'
    assert result['wheel_only_install'] is True


def test_runtime_pin_reader_rejects_non_exact_requirement(tmp_path: Path) -> None:
    """Reject ranges that could silently select an incompatible PyArrow release."""

    dependency = _load('scripts/verify_streamlit_dependencies.py')
    requirements = tmp_path / 'requirements.txt'
    requirements.write_text('pyarrow>=14\n', encoding='utf-8')
    try:
        dependency._read_exact_pins(requirements)
    except ValueError:
        pass
    else:
        raise AssertionError('A non-exact runtime requirement was accepted.')

def test_dependency_probe_uses_isolated_stages_and_faulthandler() -> None:
    """Identify the exact native import instead of hiding a process crash."""

    text = (SOURCE_ROOT / 'scripts/verify_streamlit_dependencies.py').read_text(
        encoding='utf-8'
    )
    for fragment in (
        "'numpy_import'",
        "'pandas_import'",
        "'pyarrow_import'",
        "'streamlit_import'",
        "'pandas_pyarrow_round_trip'",
        "'-I'",
        "'faulthandler'",
        'failed_stage',
    ):
        assert fragment in text


def test_clean_probe_environment_removes_path_and_loader_leakage(monkeypatch) -> None:
    """Keep project shims and dynamic-loader settings out of native probes."""

    dependency = _load('scripts/verify_streamlit_dependencies.py')
    monkeypatch.setenv('PYTHONPATH', '/unsafe/project/path')
    monkeypatch.setenv('DYLD_LIBRARY_PATH', '/unsafe/native/path')
    cleaned = dependency._clean_environment()
    assert 'PYTHONPATH' not in cleaned
    assert 'DYLD_LIBRARY_PATH' not in cleaned
    assert cleaned['PYTHONFAULTHANDLER'] == '1'
    assert cleaned['PYTHONNOUSERSITE'] == '1'


def test_negative_return_code_is_reported_as_a_signal() -> None:
    """Turn silent native termination into a readable signal name."""

    dependency = _load('scripts/verify_streamlit_dependencies.py')
    assert dependency._signal_name(-6) in {'SIGABRT', 'SIGNAL_6'}
    assert dependency._signal_name(1) is None


def test_environment_python_does_not_resolve_the_venv_symlink() -> None:
    """Protect the active venv after the earlier FastAPI symlink defect."""

    environment = _load('scripts/verify_streamlit_environment.py')
    expected = PROJECT_ROOT / '.venv-streamlit' / 'bin/python'
    assert environment.environment_python(PROJECT_ROOT) == expected


def test_startup_uses_loopback_and_headless_mode() -> None:
    """Keep the real startup test local and free of a browser launch."""

    text = (SOURCE_ROOT / 'scripts/verify_streamlit_startup.py').read_text(encoding='utf-8')
    assert '127.0.0.1' in text
    assert '--server.headless=true' in text
    assert '--server.fileWatcherType=none' in text


def test_connection_uses_real_project_clients_and_temporary_fastapi() -> None:
    """Require the actual FastAPI runner and Streamlit API client."""

    text = (SOURCE_ROOT / 'scripts/verify_streamlit_fastapi_connection.py').read_text(encoding='utf-8')
    assert 'scripts/run_fastapi_application.py' in text
    assert 'FinancialNewsApiClient' in text
    assert 'AppTest.from_file' in text
    assert 'process.terminate()' in text


def test_fastapi_environment_adds_project_source_path(monkeypatch) -> None:
    """Make the temporary FastAPI runner import the src-layout package."""

    connection = _load('scripts/verify_streamlit_fastapi_connection.py')
    monkeypatch.setenv('PYTHONPATH', '/unsafe/inherited/path')
    monkeypatch.setenv('PYTHONHOME', '/unsafe/inherited/home')
    environment = connection._fastapi_environment(PROJECT_ROOT)
    assert environment['PYTHONPATH'].split(os.pathsep) == [
        str(PROJECT_ROOT / 'src'),
        str(PROJECT_ROOT),
    ]
    assert 'PYTHONHOME' not in environment
    assert environment['FNI_PROJECT_ROOT'] == str(PROJECT_ROOT)


def test_fastapi_environment_rejects_missing_source_package(tmp_path: Path) -> None:
    """Fail before startup when the required src-layout package is absent."""

    connection = _load('scripts/verify_streamlit_fastapi_connection.py')
    try:
        connection._fastapi_environment(tmp_path)
    except FileNotFoundError as error:
        assert 'financial_news_intelligence' in str(error)
    else:
        raise AssertionError('A missing FastAPI source package was accepted.')


def test_fastapi_environment_supports_a_real_src_layout_import(tmp_path: Path) -> None:
    """Prove a directly executed runner can import the src-layout package."""

    connection = _load('scripts/verify_streamlit_fastapi_connection.py')
    package_root = tmp_path / 'src/financial_news_intelligence'
    package_root.mkdir(parents=True)
    (package_root / '__init__.py').write_text("VALUE = 'imported'\n", encoding='utf-8')
    scripts_root = tmp_path / 'scripts'
    scripts_root.mkdir()
    runner = scripts_root / 'run_fastapi_application.py'
    runner.write_text(
        'from financial_news_intelligence import VALUE\nprint(VALUE)\n',
        encoding='utf-8',
    )
    completed = subprocess.run(
        [sys.executable, str(runner)],
        cwd=tmp_path,
        env=connection._fastapi_environment(tmp_path),
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        check=False,
    )
    assert completed.returncode == 0
    assert completed.stdout.strip() == 'imported'


def test_regression_requires_both_success_markers() -> None:
    """Prevent final closure from accepting only the Streamlit tests."""

    regression = _load('scripts/run_streamlit_project_regression.py')
    assert regression.APPLICATION_SUCCESS == 'STREAMLIT APPLICATION-WIDE TESTS: PASSED'
    assert regression.PROJECT_SUCCESS == 'ISOLATED PROJECT REGRESSION: PASSED'


def test_final_closure_requires_pre_and_post_regression_logs() -> None:
    """Require both sides of the installation regression gate."""

    closure = _load('scripts/verify_streamlit_final_closure.py')
    assert set(closure.REQUIRED_LOG_MARKERS) == {
        'pre-install_streamlit_application_tests.log',
        'pre-install_isolated_project_regression.log',
        'post-install_streamlit_application_tests.log',
        'post-install_isolated_project_regression.log',
    }


def test_final_closure_requires_staged_native_import_evidence() -> None:
    """Prevent closure from accepting the old combined-probe evidence shape."""

    closure = _load('scripts/verify_streamlit_final_closure.py')
    assert 'probe_stages' in closure.REQUIRED_JSON['dependency_evidence.json']
    assert 'failed_stage' in closure.REQUIRED_JSON['dependency_evidence.json']


def test_final_closure_rejects_missing_evidence(tmp_path: Path) -> None:
    """Fail closed when any required runtime evidence file is absent."""

    closure = _load('scripts/verify_streamlit_final_closure.py')
    try:
        closure.verify_closure(PROJECT_ROOT, tmp_path, tmp_path / 'out.json')
    except FileNotFoundError:
        pass
    else:
        raise AssertionError('Missing closure evidence was accepted.')


def test_docs_keep_deployment_out_of_streamlit_closure() -> None:
    """Keep public deployment in its later dedicated phase."""

    contract = (SOURCE_ROOT / 'docs/STREAMLIT_APPLICATION_CONTRACT.md').read_text(encoding='utf-8')
    assert 'Public deployment is not changed' in contract
    assert 'Ruturaj Mokashi' in contract


def test_qa_report_lists_real_runtime_gates() -> None:
    """Require the closure report to name every real runtime check."""

    report = (SOURCE_ROOT / 'docs/STREAMLIT_QA_CLOSURE_REPORT.md').read_text(encoding='utf-8')
    for phrase in (
        'isolated environment',
        'binary-wheel',
        'pyarrow 14.0.2',
        'real Streamlit startup',
        'Streamlit-to-FastAPI',
        'pre-install',
        'post-install',
        'rollback',
    ):
        assert phrase in report


def test_no_runtime_evidence_contains_a_secret_value() -> None:
    """Keep test API keys and private values out of installed evidence code."""

    combined = '\n'.join(
        path.read_text(encoding='utf-8')
        for path in (SOURCE_ROOT / 'scripts').glob('*streamlit*.py')
    )
    assert 'sk-' not in combined
    assert '/Users/' not in combined
    assert 'BEGIN PRIVATE KEY' not in combined
