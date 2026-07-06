"""Tests for exact dependency versions, origins, and functional probes."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import verify_fastapi_dependencies as dependencies


def test_distribution_evidence_accepts_exact_project_package(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Prepare an exact project package, run validation, and check its scope."""

    environment_root = tmp_path / ".venv"
    module_file = environment_root / "lib/python3.10/site-packages/example/__init__.py"
    module_file.parent.mkdir(parents=True)
    module_file.write_text("", encoding="utf-8")
    monkeypatch.setattr(dependencies.importlib.metadata, "version", lambda _name: "1.2.3")
    monkeypatch.setattr(
        dependencies.importlib.util,
        "find_spec",
        lambda _name: SimpleNamespace(origin=str(module_file)),
    )

    result = dependencies.distribution_evidence(
        "example", "1.2.3", "example", "project", environment_root
    )

    assert result.version == "1.2.3"
    assert result.origin_scope == "project_environment"


def test_distribution_evidence_rejects_version_mismatch(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Prepare a wrong version, run validation, and check fail-closed rejection."""

    monkeypatch.setattr(dependencies.importlib.metadata, "version", lambda _name: "9.9.9")

    with pytest.raises(dependencies.DependencyVerificationError, match="version mismatch"):
        dependencies.distribution_evidence(
            "example", "1.2.3", "example", "project", tmp_path
        )


def test_distribution_evidence_rejects_native_package_outside_environment(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Prepare a global native package, run validation, and check rejection."""

    global_file = tmp_path / "global/example/__init__.py"
    global_file.parent.mkdir(parents=True)
    global_file.write_text("", encoding="utf-8")
    environment_root = tmp_path / ".venv"
    environment_root.mkdir()
    monkeypatch.setattr(dependencies.importlib.metadata, "version", lambda _name: "1.2.3")
    monkeypatch.setattr(
        dependencies.importlib.util,
        "find_spec",
        lambda _name: SimpleNamespace(origin=str(global_file)),
    )

    with pytest.raises(dependencies.DependencyVerificationError, match="outside"):
        dependencies.distribution_evidence(
            "example", "1.2.3", "example", "project", environment_root
        )


def test_require_environment_rejects_wrong_python_prefix(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Prepare a wrong prefix, run environment validation, and check rejection."""

    expected = tmp_path / ".venv"
    observed = tmp_path / "base"
    expected.mkdir()
    observed.mkdir()
    monkeypatch.setattr(dependencies.sys, "prefix", str(observed))

    with pytest.raises(dependencies.DependencyVerificationError, match="not running"):
        dependencies.require_environment(tmp_path, ".venv")


def test_verify_pdf_parser_reads_one_generated_page(monkeypatch: Any) -> None:
    """Prepare a fake PDF library, run the probe, and check one page is read."""

    class Writer:
        """Write deterministic bytes for the functional probe."""

        def add_blank_page(self, width: int, height: int) -> None:
            assert width == 72 and height == 72

        def write(self, buffer: Any) -> None:
            buffer.write(b"pdf")

    class Reader:
        """Expose one page through the expected pypdf interface."""

        def __init__(self, buffer: Any) -> None:
            assert buffer.read() == b"pdf"
            self.pages = [object()]

    fake = SimpleNamespace(PdfWriter=Writer, PdfReader=Reader)
    monkeypatch.setattr(dependencies.importlib, "import_module", lambda _name: fake)

    dependencies.verify_pdf_parser()


def test_write_evidence_uses_owner_only_permissions(tmp_path: Path) -> None:
    """Prepare evidence, write it, and check contents and private permissions."""

    output = tmp_path / "evidence.json"

    dependencies.write_evidence(output, {"status": "PASSED"})

    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "PASSED"
    assert os.stat(output).st_mode & 0o077 == 0


def test_main_profile_requires_pdf_and_http_server_packages() -> None:
    """Prepare the main contract, inspect it, and check required packages."""

    required = set(dependencies.MAIN_REQUIREMENTS)

    assert {"pypdf", "fastapi", "uvicorn", "starlette", "httpx"} <= required


def test_transformer_profile_keeps_model_packages_in_project_environment() -> None:
    """Prepare the transformer contract, inspect it, and check origin rules."""

    rules = {name: values[2] for name, values in dependencies.TRANSFORMER_REQUIREMENTS.items()}

    assert rules
    assert set(rules.values()) == {"project"}
