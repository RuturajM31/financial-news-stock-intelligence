"""Tests for the CI/CD and security package source contract."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(os.environ.get("FNI_CICD_TEST_ROOT", Path(__file__).resolve().parents[1])).resolve()


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def load_yaml(relative: str) -> dict:
    data = yaml.safe_load(read(relative))
    assert isinstance(data, dict)
    return data


def triggers(workflow: dict) -> dict:
    value = workflow.get("on", workflow.get(True))
    assert isinstance(value, dict)
    return value


def test_required_ci_cd_files_exist() -> None:
    for relative in (
        ".github/workflows/ci.yml",
        ".github/workflows/codeql.yml",
        ".github/dependabot.yml",
        ".github/pull_request_template.md",
        "SECURITY.md",
        "docs/CI_CD_SECURITY_CONTRACT.md",
        "scripts/verify_ci_cd_security.py",
        "CI_CD_SECURITY_PACKAGE_MANIFEST.json",
    ):
        target = ROOT / relative
        assert target.is_file(), relative
        assert not target.is_symlink(), relative


def test_ci_workflow_uses_least_privilege_and_safe_triggers() -> None:
    workflow = load_yaml(".github/workflows/ci.yml")
    assert workflow["permissions"] == {"contents": "read"}
    assert "concurrency" in workflow
    trigger = triggers(workflow)
    assert {"push", "pull_request", "workflow_dispatch"}.issubset(trigger)
    assert "pull_request_target" not in trigger


def test_ci_workflow_runs_project_and_security_gates() -> None:
    content = read(".github/workflows/ci.yml")
    for token in (
        "python -m pip check",
        "python -m pytest -q",
        "tests/test_ci_cd_security.py",
        "python scripts/verify_ci_cd_security.py --project-root .",
        "python -m pip_audit -r requirements.txt --strict",
        "python -m bandit -q -r src app scripts -x tests",
        "python scripts/verify_docker_production.py --project-root .",
        "python scripts/verify_kubernetes_helm.py --project-root .",
    ):
        assert token in content


def test_all_ci_jobs_have_timeouts() -> None:
    jobs = load_yaml(".github/workflows/ci.yml")["jobs"]
    for name, job in jobs.items():
        assert isinstance(job.get("timeout-minutes"), int), name
        assert 1 <= job["timeout-minutes"] <= 60, name


def test_codeql_workflow_has_only_code_scanning_permissions() -> None:
    workflow = load_yaml(".github/workflows/codeql.yml")
    permissions = workflow["permissions"]
    assert permissions["contents"] == "read"
    assert permissions["security-events"] == "write"
    assert permissions["actions"] == "read"
    assert "packages" not in permissions
    trigger = triggers(workflow)
    assert "schedule" in trigger
    assert "pull_request_target" not in trigger


def test_dependabot_monitors_pip_and_github_actions_weekly() -> None:
    dependabot = load_yaml(".github/dependabot.yml")
    updates = dependabot["updates"]
    ecosystems = {(item["package-ecosystem"], item["directory"]) for item in updates}
    assert ("pip", "/") in ecosystems
    assert ("github-actions", "/") in ecosystems
    for item in updates:
        assert item["schedule"]["interval"] == "weekly"
        assert "security" in item["labels"]


def test_workflows_do_not_deploy_or_publish() -> None:
    content = "\n".join(path.read_text(encoding="utf-8") for path in (ROOT / ".github/workflows").glob("*.yml"))
    forbidden = [
        r"\bdocker\s+push\b",
        r"\bhelm\s+upgrade\b",
        r"\bhelm\s+install\b",
        r"\bkubectl\s+apply\b",
        r"\bkubectl\s+rollout\b",
        r"\bterraform\s+apply\b",
        r"\bscp\b",
        r"\brsync\b",
    ]
    for pattern in forbidden:
        assert not re.search(pattern, content), pattern


def test_workflows_do_not_consume_deployment_secrets() -> None:
    content = "\n".join(path.read_text(encoding="utf-8") for path in (ROOT / ".github/workflows").glob("*.yml"))
    assert "${{ secrets." not in content
    assert "pull_request_target" not in content


def test_contract_explicitly_defers_public_deployment() -> None:
    contract = read("docs/CI_CD_SECURITY_CONTRACT.md")
    assert "Public deployment remains deferred" in contract
    assert "does not publish Docker images" in contract
    assert "does not publish Docker images, push to an image registry" in contract


def test_security_policy_private_reporting_and_secret_rules() -> None:
    content = read("SECURITY.md")
    assert "Do not open a public issue" in content
    assert "must not be committed to Git" in content
    assert "project-foundation-streamlit-closure" in content


def test_pull_request_template_preserves_manual_security_checkpoints() -> None:
    content = read(".github/pull_request_template.md")
    for phrase in (
        "No secrets",
        "least-privilege permissions",
        "Public deployment was not changed",
        "Docker production contract remains unchanged",
        "Kubernetes/Helm production contract remains unchanged",
    ):
        assert phrase in content


def test_ci_cd_verifier_passes_against_current_layout() -> None:
    command = [sys.executable, str(ROOT / "scripts/verify_ci_cd_security.py"), "--project-root", str(ROOT)]
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "CI/CD AND SECURITY SOURCE VERIFICATION: PASSED" in completed.stdout


def test_manifest_declares_no_public_deployment_or_git_operations() -> None:
    manifest = load_yaml("CI_CD_SECURITY_PACKAGE_MANIFEST.json")
    assert manifest["package"] == "CI/CD and Security Strike Package 13.1"
    assert manifest["public_deployment_changed"] is False
    assert manifest["image_registry_changed"] is False
    assert manifest["git_operations_performed"] is False
