#!/usr/bin/env python3
"""Verify the free public Streamlit deployment source contract offline.

The verifier intentionally avoids contacting Streamlit Community Cloud,
Docker registries, Kubernetes clusters, cloud vendors, or paid services. It
checks that the repository is ready for the free Streamlit Community Cloud
path and that the deployment package does not introduce paid-resource or
secret-handling risks.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REQUIRED_FILES = (
    "docs/FREE_STREAMLIT_PUBLIC_DEPLOYMENT_CONTRACT.md",
    "deployment/streamlit-community-cloud/README.md",
    "deployment/streamlit-community-cloud/DEPLOYMENT_CHECKLIST.md",
    "deployment/streamlit-community-cloud/secrets.example.toml",
    "PUBLIC_DEPLOYMENT_PACKAGE_MANIFEST.json",
    "app/requirements.txt",
)
REQUIRED_EXISTING_APP_FILES = (
    "app/streamlit_app.py",
    "requirements.txt",
)
PAID_OR_MUTATING_PATTERNS = (
    r"\bdocker\s+push\b",
    r"\bhelm\s+upgrade\b",
    r"\bhelm\s+install\b",
    r"\bkubectl\s+apply\b",
    r"\bkubectl\s+rollout\b",
    r"\bterraform\s+apply\b",
    r"\baws\s+eks\b",
    r"\bgcloud\s+container\b",
    r"\baz\s+aks\b",
    r"\bstripe\b",
    r"\bcredit card\b",
    r"\bpaid Kubernetes\b",
    r"\bpaid registry\b",
)
SECRET_PATTERNS = (
    r"(?i)aws_access_key_id\s*=",
    r"(?i)aws_secret_access_key\s*=",
    r"(?i)github_pat_[A-Za-z0-9_]+",
    r"(?i)ghp_[A-Za-z0-9_]{20,}",
    r"-----BEGIN (RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read(project_root: Path, relative: str) -> str:
    return (project_root / relative).read_text(encoding="utf-8")


def verify_inventory(project_root: Path) -> None:
    for relative in REQUIRED_FILES:
        target = project_root / relative
        require(target.is_file() and not target.is_symlink(), f"Missing deployment file: {relative}")
    for relative in REQUIRED_EXISTING_APP_FILES:
        target = project_root / relative
        require(target.is_file() and not target.is_symlink(), f"Missing Streamlit app prerequisite: {relative}")
    print("FREE DEPLOYMENT FILE INVENTORY: PASSED")


def verify_contract(project_root: Path) -> None:
    contract = read(project_root, "docs/FREE_STREAMLIT_PUBLIC_DEPLOYMENT_CONTRACT.md")
    for phrase in (
        "Streamlit Community Cloud",
        "free public app path only",
        "Branch: `project-foundation-streamlit-closure`",
        "Main file: `app/streamlit_app.py`",
        "Do not commit `.streamlit/secrets.toml`",
        "cannot click the Streamlit Community Cloud UI",
        "https://*.streamlit.app",
    ):
        require(phrase in contract, f"Deployment contract missing phrase: {phrase}")
    print("FREE STREAMLIT DEPLOYMENT CONTRACT: PASSED")


def verify_cloud_readme(project_root: Path) -> None:
    readme = read(project_root, "deployment/streamlit-community-cloud/README.md")
    for phrase in (
        "Repository: RuturajM31/financial-news-stock-intelligence",
        "Branch: project-foundation-streamlit-closure",
        "Main file path: app/streamlit_app.py",
        "Python 3.10",
        "free streamlit.app subdomain",
        "does not require a credit card",
    ):
        require(phrase in readme, f"Streamlit Cloud README missing phrase: {phrase}")
    print("STREAMLIT COMMUNITY CLOUD README CONTRACT: PASSED")


def verify_no_paid_or_mutating_commands(project_root: Path) -> None:
    files = [project_root / relative for relative in REQUIRED_FILES]
    files.extend(sorted((project_root / "deployment/streamlit-community-cloud").glob("*.md")))
    content = "\n".join(path.read_text(encoding="utf-8") for path in files if path.is_file())
    for pattern in PAID_OR_MUTATING_PATTERNS:
        # The contract may mention forbidden phrases only when explicitly rejecting them.
        matches = [m.group(0) for m in re.finditer(pattern, content, flags=re.IGNORECASE)]
        if matches and pattern not in (r"\bpaid Kubernetes\b", r"\bpaid registry\b", r"\bcredit card\b"):
            raise RuntimeError(f"Forbidden deployment mutation command found: {pattern}")
    require("Do not use paid Kubernetes hosting" in content, "Free-only Kubernetes boundary is missing.")
    require("Do not use paid Kubernetes hosting" in content or "paid Kubernetes cluster" in content, "Free-only paid-cluster boundary is missing.")
    print("FREE-ONLY NO-MUTATION CONTRACT: PASSED")


def verify_secret_hygiene(project_root: Path) -> None:
    for relative in REQUIRED_FILES:
        content = read(project_root, relative)
        for pattern in SECRET_PATTERNS:
            require(not re.search(pattern, content), f"Potential secret in {relative}: {pattern}")
    require(not (project_root / ".streamlit/secrets.toml").exists(), "Committed .streamlit/secrets.toml is forbidden.")
    print("FREE DEPLOYMENT SECRET HYGIENE: PASSED")


def verify_manifest(project_root: Path) -> None:
    manifest = json.loads(read(project_root, "PUBLIC_DEPLOYMENT_PACKAGE_MANIFEST.json"))
    require(manifest["package"] == "Free Public Streamlit Deployment Strike Package 14.4", "Unexpected manifest package name.")
    require(manifest["free_deployment_target"] == "Streamlit Community Cloud", "Unexpected free deployment target.")
    require(manifest["paid_services_required"] is False, "Manifest must reject paid services.")
    require(manifest["image_registry_changed"] is False, "Manifest must not change image registry.")
    require(manifest["kubernetes_resources_changed"] is False, "Manifest must not change Kubernetes resources.")
    require(manifest["git_operations_performed"] is False, "Manifest must not perform Git operations.")
    print("FREE PUBLIC DEPLOYMENT MANIFEST: PASSED")


def parse_requirements(project_root: Path, relative: str) -> list[str]:
    entries: list[str] = []
    for raw_line in read(project_root, relative).splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        entries.append(line)
    return entries


def package_name(requirement: str) -> str:
    return re.split(r"[<>=!~;\[]", requirement, maxsplit=1)[0].strip().lower().replace("_", "-")


def verify_streamlit_cloud_requirements(project_root: Path) -> None:
    requirements = parse_requirements(project_root, "app/requirements.txt")
    names = {package_name(item) for item in requirements}
    required = {"streamlit", "plotly", "numpy", "pandas", "pyarrow", "requests"}
    missing = sorted(required - names)
    require(not missing, "Streamlit Cloud app requirements missing: " + ", ".join(missing))
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
    present_forbidden = sorted(forbidden & names)
    require(not present_forbidden, "Streamlit Cloud app requirements include heavy/non-public-runtime packages: " + ", ".join(present_forbidden))
    content = read(project_root, "deployment/streamlit-community-cloud/README.md") + "\n" + read(project_root, "docs/FREE_STREAMLIT_PUBLIC_DEPLOYMENT_CONTRACT.md")
    require("app/requirements.txt" in content, "Deployment docs must name app/requirements.txt")
    require("Python 3.10" in content, "Deployment docs must require Python 3.10")
    print("STREAMLIT CLOUD DEPENDENCY ISOLATION: PASSED")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    require(project_root.is_dir() and not project_root.is_symlink(), f"Unsafe project root: {project_root}")
    verify_inventory(project_root)
    verify_contract(project_root)
    verify_cloud_readme(project_root)
    verify_streamlit_cloud_requirements(project_root)
    verify_no_paid_or_mutating_commands(project_root)
    verify_secret_hygiene(project_root)
    verify_manifest(project_root)
    print("FREE PUBLIC STREAMLIT DEPLOYMENT SOURCE VERIFICATION: PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
