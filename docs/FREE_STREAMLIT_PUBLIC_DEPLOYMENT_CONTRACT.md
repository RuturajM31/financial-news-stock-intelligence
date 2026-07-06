# Free Streamlit Public Deployment Contract

## Purpose

Package 14.3 prepares the project for the free public deployment path using Streamlit Community Cloud. The package does not create paid cloud resources, publish Docker images, mutate Kubernetes resources, or perform Git operations.

## Free deployment target

- Platform: Streamlit Community Cloud
- Cost boundary: free public app path only
- Repository source: GitHub
- Branch: `project-foundation-streamlit-closure`
- Main file: `app/streamlit_app.py`
- Python setting: select Python 3.10 in Streamlit Community Cloud advanced settings if offered, matching the validated local runtime family.

## Free-only rules

This project must not require a paid Kubernetes cluster, paid container registry, paid VM, paid database, paid private Streamlit plan, paid managed secrets product, or credit-card backed deployment service for the public deployment stage. Docker and Kubernetes remain portability and production-readiness artifacts, not the free public hosting target.

## Secrets handling

Do not commit `.streamlit/secrets.toml`, API keys, cloud tokens, registry tokens, Tiingo tokens, GitHub personal access tokens, or service credentials. If a future optional public app feature needs a secret, paste it only into Streamlit Community Cloud's app secrets interface.

## Deployment boundary

This package prepares deployment instructions and source verification. It cannot click the Streamlit Community Cloud UI or create the external app URL by itself. The actual public URL is confirmed only after the app is created in Streamlit Community Cloud and the URL is pasted back for final verification.

## Closure gate

The stage is closed when local package validation passes, changes are committed and pushed, the app is created on Streamlit Community Cloud using the free public app flow, and the resulting `https://*.streamlit.app` URL is verified.
