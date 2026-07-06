# Streamlit Community Cloud deployment checklist

## Before creating the app

- [ ] Package 14.4 passed locally.
- [ ] Git working tree is clean.
- [ ] Package 14.4 changes are committed and pushed to `project-foundation-streamlit-closure`.
- [ ] Repository is available to the GitHub account connected to Streamlit Community Cloud.
- [ ] No `.streamlit/secrets.toml` file is committed.

## Create app

- [ ] Sign in to Streamlit Community Cloud with GitHub.
- [ ] Choose repository `RuturajM31/financial-news-stock-intelligence`.
- [ ] Choose branch `project-foundation-streamlit-closure`.
- [ ] Set main file path to `app/streamlit_app.py`.
- [ ] Select Python 3.10 in Advanced settings before first deploy.
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
- [ ] Recreate the app and select Python 3.10 in Advanced settings before clicking Deploy.
- [ ] Confirm logs no longer install from root `requirements.txt` and no longer mention `torch==2.2.2`.
