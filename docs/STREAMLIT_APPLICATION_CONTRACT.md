# Streamlit Application Contract

**Portfolio owner:** Ruturaj Mokashi
**Project:** Financial News & Stock Movement Intelligence

## Product boundary

The Streamlit application is a thin, portfolio-quality interface over the
verified FastAPI service. It presents sentiment, forecasts, historical evidence,
explainability, model training results, model comparison, scenarios, provenance,
and reports. It does not load model artifacts or private provider data directly.

## User experience

Every important chart states what it shows, why it matters, and the practical
conclusion. Advanced and 3D charts include readable 2D alternatives. The design
uses plain words, accessible labels, clear uncertainty, and professional branding
for Ruturaj Mokashi.

## Runtime boundary

- `.venv-streamlit` contains UI dependencies only.
- The runtime set is exactly Streamlit 1.58.0, Plotly 6.8.0, NumPy 1.26.4,
  pandas 2.1.4, and PyArrow 14.0.2 plus their compatible dependencies.
- Dependencies are downloaded as binary wheels only. Source builds are forbidden.
- Native imports run separately so one failed library is named precisely.
- `.venv` remains the analytics and FastAPI environment.
- `.venv-distilbert` remains the sentiment worker environment.
- Streamlit communicates with FastAPI over HTTP.
- Tokens, local paths, private caches, and restricted market data are not shown.

## Completion boundary

The Streamlit phase closes only after binary-wheel dependency checks, the PyArrow
probe, application-wide tests, real Streamlit startup, a real Streamlit-to-FastAPI
connection, pre-install and post-install full project regressions, process cleanup,
and rollback controls pass.

Public deployment is not changed by Streamlit Package 9.3. Docker, monitoring,
Kubernetes, Helm, CI/CD, security hardening, and public deployment remain later
project phases.

## Temporary FastAPI import boundary

The closure check starts the real FastAPI runner with `PROJECT_ROOT/src` on Python's import path. This allows the runner to import the `financial_news_intelligence` source package without installing the project into the Streamlit environment.
