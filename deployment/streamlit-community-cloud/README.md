# Streamlit Community Cloud deployment

This is the only supported deployment path for the public Financial News Stock Intelligence application.

## App settings

Create a Streamlit Community Cloud app with:

```text
Repository: RuturajM31/financial-news-stock-intelligence
Branch: the reviewed branch you intend to deploy
Main file path: app/streamlit_app.py
Requirements file: app/requirements.txt
```

For a pull-request or cleanup review, select that review branch. Use `main` only after the reviewed change is merged. The application entry point resolves the repository `src/` layout itself, so no custom `PYTHONPATH` setting is required.

## Full BERT access

The local final model is intentionally ignored by Git. When Community Cloud does not have a local final model, the application resolves the private Hugging Face model repository using a read-only `HF_TOKEN`.

Add the token only in Streamlit Community Cloud's Secrets settings:

```toml
HF_TOKEN = "read-only-token"
```

Do not commit `.streamlit/secrets.toml`, model weights, or an actual token. Without a local artifact or this token, the app still starts and explains that Full BERT analysis is unavailable when the user requests it.

## Verification after deployment

1. Open the app and confirm the sidebar order: Overview, Analyze Article, Model Results, About / Architecture.
2. Open Overview, Model Results, and About / Architecture.
3. In Analyze Article, load the presentation sample and request analysis.
4. Confirm the result displays Bearish, Neutral, or Bullish evidence and no stock-price prediction claim.

The app is a public financial-news sentiment demonstration. It is not investment advice.
