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
Python version: Python 3.10 if available in Advanced settings
App URL: choose a free streamlit.app subdomain
```

Community Cloud runs `streamlit run` from the repository root. Keep paths relative to the repository root and keep the entrypoint as `app/streamlit_app.py`.

## Dependency handling

Community Cloud installs Python dependencies from a requirements file in the repository root or in the same directory as the app entrypoint. This project already has root requirements and validated Streamlit runtime requirements in source control.

## Secrets handling

Never commit `.streamlit/secrets.toml`. Use Streamlit Community Cloud's app settings secrets field only if a future optional feature requires a secret. The free public deployment package does not require paid secrets infrastructure.

## What this package does not do

- It does not publish Docker images.
- It does not create a paid registry.
- It does not create a Kubernetes cluster.
- It does not run cluster apply commands or Helm release mutation commands.
- It does not require a credit card.
- It does not perform Git operations.
