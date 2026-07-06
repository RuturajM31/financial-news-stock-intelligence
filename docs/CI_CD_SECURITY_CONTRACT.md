# CI/CD and Security Contract

## Purpose

This contract adds repository-level CI/CD and security controls after the Docker and Kubernetes/Helm production packages are already validated. It is a gatekeeping stage, not a public deployment stage.

## Installed files

- `.github/workflows/ci.yml`
- `.github/workflows/codeql.yml`
- `.github/dependabot.yml`
- `.github/pull_request_template.md`
- `SECURITY.md`
- `scripts/verify_ci_cd_security.py`
- `tests/test_ci_cd_security.py`
- `CI_CD_SECURITY_PACKAGE_MANIFEST.json`

## CI validation boundaries

The workflow validates focused project tests, Docker source contracts, Kubernetes/Helm source contracts, the CI/CD security verifier, dependency health, Bandit static checks, and pip-audit dependency vulnerability checks.

## Security boundaries

The workflows use least-privilege default permissions. CodeQL is isolated to code scanning permissions. Dependabot is enabled for Python dependencies and GitHub Actions. The pull request template requires explicit confirmation that deployment remains unchanged unless the pull request is the deployment stage.

## Explicit non-goals

This package does not publish Docker images, push to an image registry, run `kubectl apply`, run `helm upgrade`, create cloud infrastructure, expose public ingress, or change any public deployment target. Public deployment remains deferred to the next deployment package.

## Rollback behavior

The installer backs up every overwritten file and records whether payload directories existed before installation. If any validation fails after project mutation, the installer restores all affected files and removes only directories created by this package.
