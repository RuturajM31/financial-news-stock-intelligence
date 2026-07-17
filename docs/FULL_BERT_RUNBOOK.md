# Full BERT runbook

## Scope

This runbook covers the retained Full BERT and Financial PhraseBank workflow. The public Streamlit app uses the saved final model for inference; training helpers are separate from the public deployment path.

## Model contract

- Base model: `google-bert/bert-base-uncased`
- Labels and order: `Bearish`, `Neutral`, `Bullish`
- Maximum sequence length: 128 tokens
- Article result: arithmetic mean of sentence-level class probabilities
- Final local artifact: `artifacts/models/bert_sentiment/final_model/`
- Remote fallback: private Hugging Face repository `ruturajmokashi/financial-news-full-bert`

Model resolution is deliberate: local final artifact first, private repository with a read-only `HF_TOKEN` second, then the local Hugging Face cache. Inference does not silently fall back to lexical scoring.

## Environment

For training and the full retained test suite, install the development requirements:

```powershell
python -m venv .venv-bert
.\.venv-bert\Scripts\python.exe -m pip install --upgrade pip
.\.venv-bert\Scripts\python.exe -m pip install -r requirements-dev.txt
```

For the public Streamlit application, use `app/requirements.txt` instead. Streamlit Community Cloud installs that file from the app directory.

## Verify the saved model

Run the lightweight benchmark against the local final model:

```powershell
.\.venv-bert\Scripts\python.exe scripts\benchmark_full_bert_inference.py
```

The benchmark reads the existing model and writes no training data. Confirm the three representative sentences retain their expected Bearish, Neutral, and Bullish labels before deleting any local checkpoints.

## Run a controlled smoke workflow

`run_bert_smoke.py` creates an isolated balanced subset and dedicated smoke outputs. It never replaces the final Full BERT artifact unless explicitly asked to replace only its own smoke paths.

The smoke runner deliberately rejects an interpreter that exposes `scikit-learn`, because its isolated Transformer path avoids a known OpenMP conflict. Run it only from a dedicated Transformer-only environment that satisfies that preflight; do not use the public-app environment for this workflow.

```powershell
<transformer-python> scripts\run_bert_smoke.py
```

## Run the retained Full BERT workflow

`run_full_bert.py` owns the reproducible Full BERT training run and writes its output paths through the training configuration. Review its command-line help before starting a training job:

```powershell
.\.venv-bert\Scripts\python.exe scripts\run_full_bert.py --help
```

A completed run must save a model configuration, tokenizer files, model weights, current-run metrics, trainer history, benchmark evidence, and the current manifest. Do not overwrite the verified final model or current metrics without a deliberate reproduction plan.

## Financial PhraseBank contracts

The retained source and split modules preserve:

- 3,453 original Financial PhraseBank 75%-agreement sentences;
- five exact duplicates removed;
- 3,448 records after deduplication;
- deterministic 70/15/15 stratified train, validation, and test splits;
- the fixed Bearish/Neutral/Bullish label mapping.

Run the relevant contracts with:

```powershell
.\.venv-bert\Scripts\python.exe -m pytest tests\test_financial_phrasebank.py tests\test_financial_phrasebank_split.py
```

## Safety boundary

The model classifies financial-news language. It is not a stock-price prediction model and must not be presented as investment advice.
