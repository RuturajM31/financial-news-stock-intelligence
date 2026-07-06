# Security Policy

This repository treats credentials, API keys, generated private data, model artifacts with sensitive provenance, and deployment secrets as non-source assets. They must not be committed to Git.

## Reporting

Report a suspected vulnerability through a private GitHub security advisory or by contacting the repository owner directly. Do not open a public issue containing exploit details, credentials, or private logs.

## Supported branch

The active hardening branch is `project-foundation-streamlit-closure` until final closure and deployment promotion are complete.

## CI/CD security baseline

The CI/CD stage validates source integrity, focused regression coverage, Python dependency health, static security checks, Docker source contracts, Kubernetes/Helm source contracts, and the repository security policy. It does not publish images, mutate a Kubernetes cluster, or change public deployment.

## Secret handling

Runtime secrets must be supplied through environment variables, GitHub encrypted secrets, Kubernetes Secrets, or local untracked files. Workflows in this stage must not print secret values, pass secret values as command-line arguments, or store secret values in repository files.
