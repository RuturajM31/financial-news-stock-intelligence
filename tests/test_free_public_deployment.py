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
        "app/requirements.txt",
        "app/public_cloud_app.py",
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
    assert "Python 3.14-compatible" in content
    assert "free streamlit.app subdomain" in content


def test_checklist_keeps_external_url_as_final_evidence() -> None:
    content = read("deployment/streamlit-community-cloud/DEPLOYMENT_CHECKLIST.md")
    assert "Package 14.7 passed locally" in content
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
    assert manifest["package"] == "Free Public Streamlit Deployment Strike Package 14.7"
    assert manifest["free_deployment_target"] == "Streamlit Community Cloud"
    assert manifest["paid_services_required"] is False
    assert manifest["public_deployment_prepared"] is True
    assert manifest["external_app_created"] is False
    assert manifest["git_operations_performed"] is False


def test_precondition_manifest_excludes_disposable_python_cache_artifacts() -> None:
    manifest = json.loads(read("PUBLIC_DEPLOYMENT_PACKAGE_MANIFEST.json"))
    preconditions = manifest["precondition_sha256"]
    assert isinstance(preconditions, dict)
    for relative in preconditions:
        assert "__pycache__" not in Path(relative).parts, relative
        assert ".pytest_cache" not in Path(relative).parts, relative
        assert not relative.endswith(".pyc"), relative


def test_verifier_passes_against_current_layout(tmp_path: Path) -> None:
    simulated_project = tmp_path / "project"
    shutil.copytree(ROOT, simulated_project)
    (simulated_project / "app").mkdir(exist_ok=True)
    (simulated_project / "app/streamlit_app.py").write_text("from pathlib import Path\nPROJECT_ROOT = Path(__file__).resolve().parents[1]\nfrom app.public_cloud_app import render_public_streamlit_cloud_app, should_use_public_streamlit_cloud_app\nif should_use_public_streamlit_cloud_app(PROJECT_ROOT):\n    render_public_streamlit_cloud_app(PROJECT_ROOT)\n", encoding="utf-8")
    (simulated_project / "requirements.txt").write_text("streamlit==1.58.0\n", encoding="utf-8")
    command = [sys.executable, str(simulated_project / "scripts/verify_free_public_deployment.py"), "--project-root", str(simulated_project)]
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "FREE PUBLIC STREAMLIT DEPLOYMENT SOURCE VERIFICATION: PASSED" in completed.stdout



def parse_requirements(relative: str) -> set[str]:
    names: set[str] = set()
    for raw_line in read(relative).splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        names.add(re.split(r"[<>=!~;\[]", line, maxsplit=1)[0].strip().lower().replace("_", "-"))
    return names


def test_app_requirements_override_root_heavy_stack_for_streamlit_cloud() -> None:
    names = parse_requirements("app/requirements.txt")
    assert {"streamlit", "plotly", "numpy", "pandas", "pyarrow", "requests"}.issubset(names)
    forbidden = {
        "torch",
        "scipy",
        "transformers",
        "datasets",
        "accelerate",
        "peft",
        "sentence-transformers",
        "bert-score",
        "evaluate",
        "fastapi",
        "uvicorn",
        "scikit-learn",
    }
    assert names.isdisjoint(forbidden)


def test_docs_explain_python_310_and_app_requirements_recovery() -> None:
    docs = read("deployment/streamlit-community-cloud/README.md") + "\n" + read("docs/FREE_STREAMLIT_PUBLIC_DEPLOYMENT_CONTRACT.md")
    assert "app/requirements.txt" in docs
    assert "Python 3.10" in docs
    assert "Python 3.14" in docs
    assert "delete" in docs.lower() and "recreate" in docs.lower()


def test_app_requirements_are_one_package_per_line_and_python_314_safe() -> None:
    entries = [line.strip() for line in read("app/requirements.txt").splitlines() if line.strip() and not line.strip().startswith("#")]
    assert entries, "app/requirements.txt must contain public UI dependencies"
    for entry in entries:
        assert " " not in entry, f"requirement must be one package per line: {entry!r}"
    blocked_prefixes = ("numpy==1.", "pandas==2.1", "pyarrow==14.", "scipy==", "torch==")
    assert not [entry for entry in entries if entry.startswith(blocked_prefixes)]


def test_docs_explain_python_314_dependency_recovery() -> None:
    docs = read("deployment/streamlit-community-cloud/README.md") + "\n" + read("docs/FREE_STREAMLIT_PUBLIC_DEPLOYMENT_CONTRACT.md")
    assert "Python 3.14-compatible" in docs
    assert "app/requirements.txt" in docs
    assert "heavy training/API stack" in docs



def test_public_cloud_app_module_is_self_contained_and_free_safe() -> None:
    content = read("app/public_cloud_app.py")
    assert "def should_use_public_streamlit_cloud_app" in content
    assert "/mount/src/" in content
    assert "FNI_PUBLIC_STREAMLIT_MODE" in content
    assert "def render_public_streamlit_cloud_app" in content
    assert "Interactive sentiment and movement demo" in content
    assert "Public mode is active" in content
    assert "Top-end signal charts" in content
    assert "Forecast panels" in content
    assert "3D intelligence view" in content
    assert "Premium public intelligence demo" in content
    assert "Forecast and 3D visual closure" in content
    assert "FastAPI" in content
    forbidden = ("torch", "transformers", "sklearn", "scipy", "uvicorn")
    assert not any(term in content for term in forbidden)


def test_installer_patches_streamlit_entrypoint_for_cloud_only_demo_mode() -> None:
    """Verify the installed Streamlit entrypoint uses public Cloud demo mode safely."""

    entrypoint = (ROOT / "app" / "streamlit_app.py").read_text(encoding="utf-8")

    assert "should_use_public_streamlit_cloud_app(PROJECT_ROOT)" in entrypoint
    assert "render_public_streamlit_cloud_app(PROJECT_ROOT)" in entrypoint
    assert "from app.public_cloud_app import" in entrypoint


def test_verifier_requires_public_demo_bootstrap() -> None:
    verifier = read("scripts/verify_free_public_deployment.py")
    assert "verify_public_demo_mode" in verifier
    assert "PUBLIC STREAMLIT CLOUD PREMIUM UI FALLBACK: PASSED" in verifier
    assert "from app.public_cloud_app import" in verifier


def test_public_cloud_app_restores_premium_visual_sections() -> None:
    content = read("app/public_cloud_app.py")
    required = (
        "Top-end signal charts",
        "Forecast panels",
        "3D intelligence view",
        "go.Scatter3d",
        "go.Surface",
        "Forecast and 3D visual closure",
        "Premium public intelligence demo",
    )
    for phrase in required:
        assert phrase in content, phrase


def test_public_cloud_app_uses_premium_dark_theme_tokens() -> None:
    content = read("app/public_cloud_app.py")
    for phrase in ("--fni-bg", "--fni-panel", "--fni-cyan", "fni-hero", "fni-card", "fni-panel"):
        assert phrase in content, phrase
