# FastAPI Application Contract V1.9

## Scope

The application exposes health, readiness, text/file/URL sentiment, movement prediction, historical intelligence, explainability, scenario analysis, and licence-safe provenance endpoints.

## Runtime separation

- The FastAPI web process must not load NumPy, pandas, scikit-learn, PyTorch, or Transformers.
- Movement and intelligence run in the project `.venv` worker.
- DistilBERT runs in `.venv-distilbert`.
- Worker failures become structured API errors.

## Dependency gates

The installer verifies exact pinned versions and import origins before changing project files. Native movement libraries must be project-local. Transformer libraries must be project-local. Locked web packages may use the project environment or its approved base environment, and their origins are recorded. PDF support requires a functional `pypdf` probe.

## Readiness

The public `/ready` endpoint is lightweight by default. It verifies checksums, paths, and worker scripts without starting model inference. Deep model verification is an installer-only gate.

## Regression and HTTP gates

The existing isolated project regression must pass before and after installation. A temporary loopback Uvicorn server must return passed JSON from `/health` and lightweight `/ready`.

## Security and data boundaries

- Local loopback binding only.
- API-key authentication for protected routes.
- Bounded request, upload, redirect, CSV-row, and text sizes.
- Fail-closed schemas and validation.
- No raw Tiingo values or tokens in responses or logs.
- No public deployment authorization.

## Failure and rollback

Any failed dependency, regression, model, HTTP, checksum, or semantic gate triggers automatic rollback. Existing files are restored and newly installed files are removed.
