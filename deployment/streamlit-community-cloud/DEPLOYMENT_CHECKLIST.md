# Streamlit Community Cloud checklist

## Before deployment

- [ ] The four routes render locally in this order: Overview, Analyze Article, Model Results, About / Architecture.
- [ ] `app/streamlit_app.py` is the selected entry point.
- [ ] `app/requirements.txt` is present.
- [ ] The working tree is clean and the intended branch is pushed.
- [ ] No `.streamlit/secrets.toml`, model weights, local caches, logs, or virtual environments are staged.
- [ ] The final local model works, or a read-only Hugging Face `HF_TOKEN` is ready for Community Cloud.

## Create or update the app

- [ ] Select `RuturajM31/financial-news-stock-intelligence`.
- [ ] Select the reviewed branch.
- [ ] Set the main file path to `app/streamlit_app.py`.
- [ ] Confirm Streamlit installs `app/requirements.txt`.
- [ ] Add `HF_TOKEN` through the Secrets UI only when the deployed runtime needs the private model repository.

## After deployment

- [ ] Confirm the health/startup log has no import failures.
- [ ] Confirm all four public pages open.
- [ ] Confirm Analyze Article accepts the presentation sample.
- [ ] Confirm model analysis produces sentence evidence when Full BERT is available.
- [ ] Confirm no credentials or local paths are visible.
- [ ] Confirm the public boundary states that the app does not predict stock prices or returns.
