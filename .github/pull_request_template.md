## Change summary

-

## Validation checklist

- [ ] Relevant retained pytest tests passed locally or in CI.
- [ ] The four public routes still render in order.
- [ ] Full BERT labels, source priority, and article aggregation are unchanged unless this is an approved model change.
- [ ] Model-result metrics and displayed rounding are unchanged unless regenerated intentionally.
- [ ] No local model artifacts, checkpoints, caches, logs, or credentials are included.

## Security checklist

- [ ] No secrets, tokens, credentials, or private data were committed.
- [ ] New workflows use least-privilege permissions.
- [ ] Public deployment changes keep `app/streamlit_app.py` and `app/requirements.txt` as the Community Cloud path.
