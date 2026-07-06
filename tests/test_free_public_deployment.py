"""Static tests for the free Streamlit Community Cloud deployment contract."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(os.environ.get("FNI_FREE_DEPLOY_TEST_ROOT", Path(__file__).resolve().parents[1])).resolve()


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_free_deployment_files_exist() -> None:
    for relative in (
        "docs/FREE_STREAMLIT_PUBLIC_DEPLOYMENT_CONTRACT.md",
        "deployment/streamlit-community-cloud/README.md",
        "deployment/streamlit-community-cloud/DEPLOYMENT_CHECKLIST.md",
        "deployment/streamlit-community-cloud/secrets.example.toml",
        "scripts/verify_free_public_deployment.py",
        "PUBLIC_DEPLOYMENT_PACKAGE_MANIFEST.json",
    ):
        target = ROOT / relative
        assert target.is_file(), relative
        assert not target.is_symlink(), relative


def test_contract_uses_streamlit_community_cloud_free_target() -> None:
    content = read("docs/FREE_STREAMLIT_PUBLIC_DEPLOYMENT_CONTRACT.md")
    assert "Streamlit Community Cloud" in content
    assert "free public app path only" in content
    assert "app/streamlit_app.py" in content
    assert "project-foundation-streamlit-closure" in content
    assert "paid Kubernetes cluster" in content
    assert "paid container registry" in content


def test_readme_has_exact_manual_app_settings() -> None:
    content = read("deployment/streamlit-community-cloud/README.md")
    assert "Repository: RuturajM31/financial-news-stock-intelligence" in content
    assert "Branch: project-foundation-streamlit-closure" in content
    assert "Main file path: app/streamlit_app.py" in content
    assert "Python 3.10" in content
    assert "free streamlit.app subdomain" in content


def test_checklist_keeps_external_url_as_final_evidence() -> None:
    content = read("deployment/streamlit-community-cloud/DEPLOYMENT_CHECKLIST.md")
    assert "Package 14.3 passed locally" in content
    assert "Git working tree is clean" in content
    assert "Deploy to a free `streamlit.app` subdomain" in content
    assert "Copy the final `https://*.streamlit.app` URL" in content


def test_no_paid_or_cluster_mutating_deployment_commands_are_introduced() -> None:
    combined = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [
            ROOT / "docs/FREE_STREAMLIT_PUBLIC_DEPLOYMENT_CONTRACT.md",
            ROOT / "deployment/streamlit-community-cloud/README.md",
            ROOT / "deployment/streamlit-community-cloud/DEPLOYMENT_CHECKLIST.md",
        ]
    )
    forbidden_commands = [
        r"\bdocker\s+push\b",
        r"\bhelm\s+upgrade\b",
        r"\bhelm\s+install\b",
        r"\bkubectl\s+apply\b",
        r"\bterraform\s+apply\b",
        r"\baws\s+eks\b",
        r"\bgcloud\s+container\b",
        r"\baz\s+aks\b",
    ]
    for pattern in forbidden_commands:
        assert not re.search(pattern, combined, flags=re.IGNORECASE), pattern


def test_secrets_example_contains_no_real_secret_material() -> None:
    content = read("deployment/streamlit-community-cloud/secrets.example.toml")
    assert "paste-only-in-streamlit-cloud-secrets-ui" in content
    assert "ghp_" not in content
    assert "github_pat_" not in content
    assert "BEGIN PRIVATE KEY" not in content
    assert "Do not copy real secrets into Git" in content


def test_manifest_declares_free_only_manual_public_deployment() -> None:
    manifest = json.loads(read("PUBLIC_DEPLOYMENT_PACKAGE_MANIFEST.json"))
    assert manifest["package"] == "Free Public Streamlit Deployment Strike Package 14.3"
    assert manifest["free_deployment_target"] == "Streamlit Community Cloud"
    assert manifest["paid_services_required"] is False
    assert manifest["public_deployment_prepared"] is True
    assert manifest["external_app_created"] is False
    assert manifest["git_operations_performed"] is False


def test_precondition_manifest_excludes_disposable_python_cache_artifacts() -> None:
    manifest = json.loads(read("PUBLIC_DEPLOYMENT_PACKAGE_MANIFEST.json"))
    preconditions = manifest["precondition_sha256"]
    assert preconditions, "precondition hashes must not be empty"
    for relative in preconditions:
        assert "__pycache__" not in Path(relative).parts, relative
        assert ".pytest_cache" not in Path(relative).parts, relative
        assert not relative.endswith(".pyc"), relative


def test_verifier_passes_against_current_layout(tmp_path: Path) -> None:
    simulated_project = tmp_path / "project"
    shutil.copytree(ROOT, simulated_project)
    (simulated_project / "app").mkdir(exist_ok=True)
    (simulated_project / "app/streamlit_app.py").write_text("import streamlit as st\nst.write('ok')\n", encoding="utf-8")
    (simulated_project / "requirements.txt").write_text("streamlit==1.58.0\n", encoding="utf-8")
    command = [sys.executable, str(simulated_project / "scripts/verify_free_public_deployment.py"), "--project-root", str(simulated_project)]
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "FREE PUBLIC STREAMLIT DEPLOYMENT SOURCE VERIFICATION: PASSED" in completed.stdout
