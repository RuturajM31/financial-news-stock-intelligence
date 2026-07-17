# Codebase guide

## Purpose

This repository supports one public product: a four-page Streamlit portfolio application for financial-news sentiment analysis. The application presents Full BERT article sentiment, sentence evidence, separate lexical cues, verified experiment results, and architecture context.

## Start here

- `app/streamlit_app.py` is the only application entry point.
- `app/public_cloud_app.py` owns page configuration, the four-route sidebar, article input, extraction, lazy model loading, and visual presentation.
- `app/requirements.txt` is the Streamlit Community Cloud dependency file.

The public route order is fixed:

1. Overview
2. Analyze Article
3. Model Results
4. About / Architecture

## Core modules

| Area | Files | Responsibility |
|---|---|---|
| Full BERT inference | `src/financial_news_intelligence/models/full_bert_inference.py` | Resolves the model source, validates the snapshot, loads lazily, tokenizes sentences, and aggregates probabilities. |
| Evidence visualisation | `src/financial_news_intelligence/models/bert_visualization.py` | Produces pure data structures for sentence, token, and reader views. |
| Experiment evidence | `src/financial_news_intelligence/models/experiment_results.py` | Strictly validates and loads the metrics shown on Model Results. |
| Training wrapper | `src/financial_news_intelligence/models/bert_training.py` | Defines the retained Full BERT training configuration. |
| Shared training engine | `src/financial_news_intelligence/models/distilbert_training.py` | Provides the shared Transformer training engine used by the Full BERT wrapper. |
| PhraseBank acquisition | `src/financial_news_intelligence/data/financial_phrasebank.py` | Downloads, normalizes, and records the source dataset. |
| PhraseBank split | `src/financial_news_intelligence/data/financial_phrasebank_split.py` | Removes exact duplicates and creates reproducible stratified splits. |

`schemas/common.py` retains the three sentiment labels used by Financial PhraseBank preparation. The package exposes only this active schema surface.

## Evidence files

The public Model Results page reads only the retained evidence files:

- `reports/metrics/bert_sentiment_current_run_metrics.json`
- `reports/metrics/bert_sentiment_metrics.json`
- `reports/metrics/distilbert_sentiment_metrics.json`
- `reports/metrics/bert_lora_sentiment_metrics.json`
- `reports/metrics/bert_sentiment_current_run_history.json`
- `reports/metrics/bert_sentiment_current_run_benchmark.json`
- `artifacts/manifests/bert_sentiment_current_run_manifest.json`
- `artifacts/manifests/financial_phrasebank_acquisition_manifest.json`
- `artifacts/manifests/financial_phrasebank_split_manifest.json`

Do not replace one run's metrics with another run's values. `experiment_results.py` intentionally rejects incomplete or incorrectly labelled evidence.

## Tests

The retained suite covers the public route contract, Analyze Article state and extraction behavior, Full BERT source resolution and inference helpers, visualisation values, model-result evidence, and Financial PhraseBank contracts.

Run the suite with:

```powershell
.\.venv-bert\Scripts\python.exe -m pytest
```

The integration test that runs a sample article requires a complete local final model or a valid read-only `HF_TOKEN`. CI keeps its Full BERT checks offline and does not download the private model.

## Local artifacts

Local virtual environments, model weights, Hugging Face caches, checkpoints, logs, and generated test outputs are ignored. Keep the final model at `artifacts/models/bert_sentiment/final_model/` for local inference. Checkpoints are reproducible outputs and must not be committed.

## Change discipline

Keep the four routes, model-loading order, labels, article aggregation, widget keys, session-state keys, displayed metric rounding, and evidence boundaries stable unless a deliberate product change is approved.
