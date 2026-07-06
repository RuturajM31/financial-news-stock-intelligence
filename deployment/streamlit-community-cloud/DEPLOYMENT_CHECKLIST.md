# Streamlit Community Cloud deployment checklist

## Before creating the app

- [ ] Package 14.3 passed locally.
- [ ] Git working tree is clean.
- [ ] Package 14.3 changes are committed and pushed to `project-foundation-streamlit-closure`.
- [ ] Repository is available to the GitHub account connected to Streamlit Community Cloud.
- [ ] No `.streamlit/secrets.toml` file is committed.

## Create app

- [ ] Sign in to Streamlit Community Cloud with GitHub.
- [ ] Choose repository `RuturajM31/financial-news-stock-intelligence`.
- [ ] Choose branch `project-foundation-streamlit-closure`.
- [ ] Set main file path to `app/streamlit_app.py`.
- [ ] Select Python 3.10 in Advanced settings if the option is available.
- [ ] Leave secrets empty unless a future optional feature explicitly needs them.
- [ ] Deploy to a free `streamlit.app` subdomain.

## After creation

- [ ] Open the public app URL.
- [ ] Confirm the home page renders.
- [ ] Confirm no secret values are displayed.
- [ ] Copy the final `https://*.streamlit.app` URL into the closure evidence.
