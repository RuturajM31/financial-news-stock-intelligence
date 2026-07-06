# Docker Production Contract

## Purpose

Package 11.7 creates two reproducible local production images: a FastAPI image
that preserves model-process isolation and a separate Streamlit presentation
image. The package does not publish images, expose public endpoints, or change
Kubernetes, Helm, CI/CD, or cloud resources.

## Runtime boundaries

- FastAPI runs as one non-root process and starts movement and sentiment workers
  as separate subprocesses.
- The Linux image uses one consistent Python environment with a controlled
  `.venv-distilbert` alias, preserving process isolation without duplicating the
  large CPU-only PyTorch runtime.
- Streamlit communicates with FastAPI by Compose service name on a dedicated,
  project-scoped bridge network.
- Host ports use explicit long syntax and bind only to `127.0.0.1`; no LAN or
  public binding is authorized.
- The bridge network permits provider egress required by the application while
  remaining isolated by the unique Compose project name.
- API authentication remains mandatory in production mode.

## Artifact handling

The build context excludes private Tiingo cache, raw data, training checkpoints,
local virtual environments, secrets, logs, and strike backups. It includes only
the verified DistilBERT final model and movement/foundation artifacts required by
runtime manifests. Runtime artifacts are owner-readable only inside the image.

## Container security

Both services run as UID/GID 10001, use a read-only root filesystem, drop all
Linux capabilities, enable `no-new-privileges`, limit PIDs/CPU/memory, and use
bounded tmpfs storage. Health checks use the Python standard library and never
include API keys.

## Closure gates

The strike verifies package hashes, Git preconditions, Python tests, rendered
Compose port mappings, actual Docker PortBindings, required artifact inventory,
image builds, non-root/read-only security settings, service connectivity, API
health/readiness and metrics, Streamlit health, secret-free logs, cleanup, and
full project regression. Built images remain local; containers and networks are
removed.


## Dynamic local verification ports

The transactional closure runner may select free loopback ports instead of the
Compose defaults. Installed verification reads `FNI_FASTAPI_PORT` and
`FNI_STREAMLIT_PORT` from the effective environment and compares the rendered
configuration with those exact values. This does not change the required
`127.0.0.1` host binding.

## Authenticated metrics verification

Production mode protects `/metrics` with the same API-key policy as other
protected endpoints. The transactional smoke test receives the temporary key
through its process environment and sends it only as the `X-API-Key` request
header for the metrics request. The key is never supplied as a command-line
argument, printed, persisted, or included in run evidence.
