"""Tests that the web-layer imports stay free of native model libraries."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


FORBIDDEN_MODULES = ("numpy", "pandas", "sklearn", "torch", "transformers")


def run_clean_import(
    project_root: Path,
    module_name: str,
) -> subprocess.CompletedProcess[str]:
    """Import one module in a clean process and report forbidden imports."""

    code = (
        "import importlib,sys; "
        f"tracked={FORBIDDEN_MODULES!r}; "
        "before={name for name in tracked if name in sys.modules}; "
        f"importlib.import_module({module_name!r}); "
        "bad=[name for name in tracked if name in sys.modules and name not in before]; "
        "print(','.join(bad)); "
        "raise SystemExit(1 if bad else 0)"
    )
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(project_root / "src"),
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def test_app_import_does_not_load_native_model_libraries() -> None:
    """Prepare a clean process and check app imports stay model-free."""

    project_root = Path(__file__).resolve().parents[1]
    result = run_clean_import(
        project_root,
        "financial_news_intelligence.api.app",
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == ""


def test_services_import_does_not_load_native_model_libraries() -> None:
    """Prepare a clean process and check service imports stay model-free."""

    project_root = Path(__file__).resolve().parents[1]
    result = run_clean_import(
        project_root,
        "financial_news_intelligence.api.services",
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == ""


def test_movement_worker_names_sigsegv_exit_safely() -> None:
    """Prepare SIGSEGV, run exit formatting, and check the exact safe reason."""

    from financial_news_intelligence.api.movement_worker_client import (
        MovementWorkerClient,
    )

    result = MovementWorkerClient._exit_reason(-11)

    assert "SIGSEGV" in result
    assert "(11)" in result

def test_app_import_does_not_require_optional_file_parsers() -> None:
    """Prepare missing parsers, import the app, and check startup succeeds."""

    project_root = Path(__file__).resolve().parents[1]
    code = (
        "import sys; "
        "sys.modules['docx']=None; "
        "sys.modules['pypdf']=None; "
        "import financial_news_intelligence.api.app"
    )
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(project_root / "src"),
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr



def test_app_factory_does_not_require_python_multipart() -> None:
    """Prepare missing multipart modules, create the app, and check startup."""

    project_root = Path(__file__).resolve().parents[1]
    code = (
        "import sys; "
        "sys.modules['multipart']=None; "
        "sys.modules['python_multipart']=None; "
        "from pathlib import Path; "
        "from financial_news_intelligence.api.config import ApiSettings; "
        "from financial_news_intelligence.api.app import create_app; "
        "settings=ApiSettings(project_root=Path.cwd(), environment='test', "
        "api_key='a'*32, require_api_key=True, "
        "trusted_hosts=('testserver',)); "
        "app=create_app(settings=settings, services=object()); "
        "assert app.title"
    )
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(project_root / "src"),
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
