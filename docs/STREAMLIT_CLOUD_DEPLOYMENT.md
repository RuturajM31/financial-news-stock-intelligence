# Streamlit Community Cloud deployment

Application settings:

- Repository: RuturajM31/financial-news-stock-intelligence
- Branch: main
- Main file: app/streamlit_app.py
- Recommended Python: 3.11
- Dependency file: app/requirements.txt
- Private model: ruturajmokashi/financial-news-full-bert

Required secrets:

    HF_MODEL_REPO = "ruturajmokashi/financial-news-full-bert"
    HF_TOKEN = "hf_REPLACE_WITH_PRIVATE_READ_TOKEN"

Use a dedicated Hugging Face READ token. Never commit .streamlit/secrets.toml.

Overview, Model Results and About do not load Full BERT. Analyze Article loads it only after a requested analysis. A complete local artifact has priority locally. Community Cloud downloads and validates the private snapshot, retains the Hugging Face cache, loads the model once through st.cache_resource, and runs inference on CPU. A cold start includes roughly 419 MiB of model download and CPU initialization.

If download, memory or loading fails, the app reports that Full BERT is unavailable. It never substitutes lexical scoring as model output.

Public validation checklist:

1. Confirm the deployed commit matches main.
2. Open Overview, Model Results and About without loading the model.
3. Run the presentation sample in Analyze Article.
4. Confirm class scores, sentence evidence, semantic views and secondary lexical cues.
5. Run a pasted article and check mobile layout.
6. Confirm failures do not expose secrets.
7. Do not call deployment successful until public Full BERT inference completes.

Rollback through Streamlit Community Cloud app settings to a previously verified commit or branch configuration. Do not overwrite project-foundation-streamlit-closure.