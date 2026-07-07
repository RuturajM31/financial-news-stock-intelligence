#!/usr/bin/env python3
"""Verify CI/CD and security source contracts without contacting external services.

The verifier reads workflow, Dependabot, documentation, and policy files from
the repository. It enforces least-privilege defaults, required validation
jobs, secret hygiene, and the explicit boundary that this stage cannot deploy
publicly or mutate an image registry or Kubernetes cluster.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import yaml

REQUIRED_FILES = {
    ".github/workflows/ci.yml",
    ".github/workflows/codeql.yml",
    ".github/dependabot.yml",
    ".github/pull_request_template.md",
    "SECURITY.md",
    "docs/CI_CD_SECURITY_CONTRACT.md",
    "CI_CD_SECURITY_PACKAGE_MANIFEST.json",
}
FORBIDDEN_DEPLOYMENT_PATTERNS = [
    r"\bdocker\s+push\b",
    r"\bhelm\s+upgrade\b",
    r"\bhelm\s+install\b",
    r"\bkubectl\s+apply\b",
    r"\bkubectl\s+rollout\b",
    r"\baws\s+eks\b",
    r"\bgcloud\s+container\b",
    r"\baz\s+aks\b",
    r"\bterraform\s+apply\b",
    r"\bscp\b",
    r"\brsync\b",
]
SECRET_PATTERNS = [
    r"(?i)aws_access_key_id\s*=",
    r"(?i)aws_secret_access_key\s*=",
    r"(?i)github_pat_[A-Za-z0-9_]+",
    r"(?i)ghp_[A-Za-z0-9_]{20,}",
    r"(?i)api[_-]?key\s*[:=]\s*['\"][A-Za-z0-9_\-]{16,}['\"]",
    r"(?i)secret\s*[:=]\s*['\"][A-Za-z0-9_\-]{16,}['\"]",
    r"-----BEGIN (RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----",
]


def require(condition: bool, message: str) -> None:
    """Raise one fail-closed error for a violated CI/CD contract."""
    if not condition:
        raise RuntimeError(message)


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file and require a mapping at the top level."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    require(isinstance(data, dict), f"{path} must contain a YAML mapping.")
    return data


def normalized_on(workflow: dict[str, Any]) -> dict[str, Any]:
    """Return the GitHub Actions trigger map despite YAML 1.1 parsing quirks."""
    trigger = workflow.get("on", workflow.get(True))
    require(isinstance(trigger, dict), "workflow trigger must be a mapping.")
    return trigger


def all_workflow_text(project_root: Path) -> str:
    """Concatenate workflow text for deployment and secret pattern scans."""
    return "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((project_root / ".github/workflows").glob("*.yml"))
    )


def verify_required_files(project_root: Path) -> None:
    """Ensure the package installed every expected CI/CD and security file."""
    for relative in sorted(REQUIRED_FILES):
        target = project_root / relative
        require(target.is_file() and not target.is_symlink(), f"Missing required CI/CD file: {relative}")
    print("CI/CD FILE INVENTORY: PASSED")


def verify_ci_workflow(project_root: Path) -> None:
    """Verify the main CI workflow covers tests, security, and source contracts."""
    workflow_path = project_root / ".github/workflows/ci.yml"
    workflow = load_yaml(workflow_path)
    require(workflow.get("permissions") == {"contents": "read"}, "CI workflow must default to contents:read only.")
    require("concurrency" in workflow, "CI workflow must define concurrency cancellation.")
    triggers = normalized_on(workflow)
    require("pull_request" in triggers and "push" in triggers and "workflow_dispatch" in triggers, "CI workflow triggers are incomplete.")
    require("pull_request_target" not in triggers, "CI workflow must not use pull_request_target.")
    jobs = workflow.get("jobs")
    require(isinstance(jobs, dict), "CI workflow must define jobs.")
    for required_job in ("focused-regression", "dependency-and-static-security", "deployment-contracts"):
        require(required_job in jobs, f"CI workflow missing job: {required_job}")
        timeout = jobs[required_job].get("timeout-minutes")
        require(isinstance(timeout, int) and 1 <= timeout <= 60, f"CI job {required_job} must define a bounded timeout.")
    content = workflow_path.read_text(encoding="utf-8")
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
        require(token in content, f"CI workflow missing validation command: {token}")
    print("CI WORKFLOW CONTRACT: PASSED")


def verify_codeql_workflow(project_root: Path) -> None:
    """Verify CodeQL has code-scanning permission and no deployment authority."""
    workflow_path = project_root / ".github/workflows/codeql.yml"
    workflow = load_yaml(workflow_path)
    permissions = workflow.get("permissions")
    require(isinstance(permissions, dict), "CodeQL workflow permissions must be explicit.")
    require(permissions.get("contents") == "read", "CodeQL workflow must have contents:read.")
    require(permissions.get("security-events") == "write", "CodeQL workflow must have security-events:write.")
    require("packages" not in permissions or permissions.get("packages") == "read", "CodeQL must not have package write permission.")
    triggers = normalized_on(workflow)
    require("schedule" in triggers and "workflow_dispatch" in triggers, "CodeQL workflow must support scheduled and manual analysis.")
    require("pull_request_target" not in triggers, "CodeQL workflow must not use pull_request_target.")
    content = workflow_path.read_text(encoding="utf-8")
    require("github/codeql-action/init@v3" in content, "CodeQL init action is missing.")
    require("github/codeql-action/analyze@v3" in content, "CodeQL analyze action is missing.")
    print("CODEQL WORKFLOW CONTRACT: PASSED")


def verify_dependabot(project_root: Path) -> None:
    """Verify dependency maintenance covers Python and GitHub Actions."""
    dependabot = load_yaml(project_root / ".github/dependabot.yml")
    updates = dependabot.get("updates")
    require(isinstance(updates, list), "Dependabot updates must be a list.")
    ecosystems = {(item.get("package-ecosystem"), item.get("directory")) for item in updates if isinstance(item, dict)}
    require(("pip", "/") in ecosystems, "Dependabot must monitor root Python dependencies.")
    require(("github-actions", "/") in ecosystems, "Dependabot must monitor GitHub Actions.")
    for item in updates:
        require(isinstance(item, dict), "Dependabot update entries must be mappings.")
        schedule = item.get("schedule")
        require(isinstance(schedule, dict) and schedule.get("interval") == "weekly", "Dependabot updates must be weekly.")
        labels = item.get("labels", [])
        require("security" in labels, "Dependabot updates must carry the security label.")
    print("DEPENDABOT CONTRACT: PASSED")


def verify_no_deployment_mutation(project_root: Path) -> None:
    """Reject public deployment, registry, and cluster mutation commands."""
    content = all_workflow_text(project_root)
    for pattern in FORBIDDEN_DEPLOYMENT_PATTERNS:
        require(not re.search(pattern, content), f"Forbidden deployment command in workflows: {pattern}")
    contract = (project_root / "docs/CI_CD_SECURITY_CONTRACT.md").read_text(encoding="utf-8")
    require("Public deployment remains deferred" in contract, "CI/CD contract must explicitly defer public deployment.")
    require("does not publish Docker images" in contract, "CI/CD contract must forbid image publishing.")
    print("NO-DEPLOYMENT-MUTATION CONTRACT: PASSED")


def verify_secret_hygiene(project_root: Path) -> None:
    """Scan installed CI/CD files for common committed secret forms."""
    files = [project_root / relative for relative in REQUIRED_FILES]
    files.extend(sorted((project_root / ".github/workflows").glob("*.yml")))
    for path in files:
        content = path.read_text(encoding="utf-8")
        for pattern in SECRET_PATTERNS:
            require(not re.search(pattern, content), f"Potential secret pattern in {path}: {pattern}")
    workflow_text = all_workflow_text(project_root)
    require("${{ secrets." not in workflow_text, "This stage must not consume GitHub secrets before deployment.")
    print("SECRET HYGIENE CONTRACT: PASSED")


def verify_documentation(project_root: Path) -> None:
    """Verify security policy and pull request template document the gates."""
    security = (project_root / "SECURITY.md").read_text(encoding="utf-8")
    require("Do not open a public issue" in security, "Security policy must direct private vulnerability reporting.")
    require("must not be committed to Git" in security, "Security policy must prohibit committing secrets.")
    template = (project_root / ".github/pull_request_template.md").read_text(encoding="utf-8")
    for phrase in ("No secrets", "Public deployment was not changed", "least-privilege permissions"):
        require(phrase in template, f"Pull request template missing checkpoint: {phrase}")
    print("SECURITY DOCUMENTATION CONTRACT: PASSED")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    require(project_root.is_dir() and not project_root.is_symlink(), f"Unsafe project root: {project_root}")
    verify_required_files(project_root)
    verify_ci_workflow(project_root)
    verify_codeql_workflow(project_root)
    verify_dependabot(project_root)
    verify_no_deployment_mutation(project_root)
    verify_secret_hygiene(project_root)
    verify_documentation(project_root)
    print("CI/CD AND SECURITY SOURCE VERIFICATION: PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
