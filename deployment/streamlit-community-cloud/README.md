# Free public deployment — Streamlit Community Cloud

This directory documents the free public deployment path for the Financial News and Stock Movement Intelligence app.

## Selected free target

Use **Streamlit Community Cloud** for the public app. Do not use paid Kubernetes hosting, paid image registries, paid virtual machines, paid databases, or paid private app hosting for this stage.

## App settings

In Streamlit Community Cloud, create the app with these values:

```text
Repository: RuturajM31/financial-news-stock-intelligence
Branch: project-foundation-streamlit-closure
Main file path: app/streamlit_app.py
Python version: Python 3.10 when selectable; Python 3.14-compatible app dependencies when Streamlit Cloud forces a newer runtime in Advanced settings
App URL: choose a free streamlit.app subdomain
```

Community Cloud runs `streamlit run` from the repository root. Keep paths relative to the repository root and keep the entrypoint as `app/streamlit_app.py`.

## Dependency handling

Community Cloud installs dependency files from the app entrypoint directory before the repository root. This package adds `app/requirements.txt` so Streamlit Cloud installs only the free public UI dependencies instead of the full local analytics/training stack from root `requirements.txt`.

## Secrets handling

Never commit `.streamlit/secrets.toml`. Use Streamlit Community Cloud's app settings secrets field only if a future optional feature requires a secret. The free public deployment package does not require paid secrets infrastructure.

## What this package does not do

- It does not publish Docker images.
- It does not create a paid registry.
- It does not create a Kubernetes cluster.
- It does not run cluster apply commands or Helm release mutation commands.
- It does not require a credit card.
- It does not perform Git operations.


## Fix for Python 3.14 dependency failure

If Streamlit Cloud logs show Python 3.14 and errors for `torch`, `numpy`, or `scipy`, the app was created with the wrong Python/runtime dependency path. Delete the failed app and recreate it using Python 3.10 when selectable; Python 3.14-compatible app dependencies when Streamlit Cloud forces a newer runtime in Advanced settings, branch `project-foundation-streamlit-closure`, and main file `app/streamlit_app.py`.


Package 14.7 note: `app/requirements.txt` uses Python-version-flexible public-UI dependencies so Streamlit Community Cloud can resolve wheels even when the platform defaults to Python 3.14. The heavy training/API stack remains isolated in the root requirements and Docker/Kubernetes paths.
