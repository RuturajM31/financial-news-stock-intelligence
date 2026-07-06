# Streamlit Community Cloud deployment checklist

## Before creating the app

- [ ] Package 14.7 passed locally.
- [ ] Git working tree is clean.
- [ ] Package 14.7 changes are committed and pushed to `project-foundation-streamlit-closure`.
- [ ] Repository is available to the GitHub account connected to Streamlit Community Cloud.
- [ ] No `.streamlit/secrets.toml` file is committed.

## Create app

- [ ] Sign in to Streamlit Community Cloud with GitHub.
- [ ] Choose repository `RuturajM31/financial-news-stock-intelligence`.
- [ ] Choose branch `project-foundation-streamlit-closure`.
- [ ] Set main file path to `app/streamlit_app.py`.
- [ ] Select Python 3.10 when selectable; Python 3.14-compatible app dependencies when Streamlit Cloud forces a newer runtime in Advanced settings before first deploy.
- [ ] Leave secrets empty unless a future optional feature explicitly needs them.
- [ ] Confirm `app/requirements.txt` exists in the pushed branch.
- [ ] Deploy to a free `streamlit.app` subdomain.

## After creation

- [ ] Open the public app URL.
- [ ] Confirm the home page renders.
- [ ] Confirm no secret values are displayed.
- [ ] Copy the final `https://*.streamlit.app` URL into the closure evidence.


## Recovery from failed Python 3.14 build

- [ ] If the app was already created with Python 3.14, delete the failed app.
- [ ] Recreate the app and select Python 3.10 when selectable; Python 3.14-compatible app dependencies when Streamlit Cloud forces a newer runtime in Advanced settings before clicking Deploy.
- [ ] Confirm logs no longer install from root `requirements.txt` and no longer mention `torch==2.2.2`.


Package 14.7 note: `app/requirements.txt` uses Python-version-flexible public-UI dependencies so Streamlit Community Cloud can resolve wheels even when the platform defaults to Python 3.14. The heavy training/API stack remains isolated in the root requirements and Docker/Kubernetes paths.
