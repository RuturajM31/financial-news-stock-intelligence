# Codebase guide

## Start here

Read the public product in this order:

1. app/streamlit_app.py
2. app/public_cloud_app.py
3. src/financial_news_intelligence/models/full_bert_inference.py
4. src/financial_news_intelligence/models/experiment_results.py
5. src/financial_news_intelligence/models/bert_visualization.py
6. tests/test_full_bert_streamlit_integration.py and related focused tests

## Repository structure

- app contains the Streamlit entry point, four-page public interface, components and styling.
- src/financial_news_intelligence contains reusable data, model, API and service code.
- reports/metrics contains lightweight verified evaluation evidence.
- artifacts/manifests contains dataset, training and runtime provenance.
- tests contains unit, integration and Streamlit AppTests.
- scripts contains training, benchmarking and validation entry points.
- deployment, docker, helm and kubernetes document serving and MLOps paths.

## Application entry and pages

app/streamlit_app.py is the deployment entry point. It delegates the public release to render_public_streamlit_cloud_app in app/public_cloud_app.py.

The public router exposes:

- Overview: product purpose and verified summary.
- Analyze Article: article input, Full BERT inference and evidence.
- Model Results: current and historical experiment evidence.
- About / Architecture: runtime, training and deployment architecture.

Widget and session-state keys live in app/public_cloud_app.py. Their names are compatibility-sensitive.

## Article analysis flow

The Analyze Article page accepts a URL, pasted article or deterministic sample. Extraction and validation produce a headline and article body. At least 25 words are required before analysis.

A successful request performs these stages:

1. Resolve the article source.
2. Preserve the exact article text.
3. Split the headline and body into meaningful ordered sentences.
4. Load the cached Full BERT runtime.
5. Run batched sentence inference.
6. Average sentence class scores into the article result.
7. Render model evidence.
8. Render lexical cues separately.

Changing the source clears stale results before another analysis.

## Full BERT model loading

src/financial_news_intelligence/models/full_bert_inference.py owns model loading and inference.

A complete local directory at artifacts/models/bert_sentiment/final_model has first priority. When it is unavailable, the loader retrieves the private repository configured by HF_MODEL_REPO and HF_TOKEN. Environment variables and Streamlit secrets are supported.

The downloaded snapshot is validated before loading. Errors are sanitized and never include the token. Streamlit wraps the runtime with st.cache_resource so model weights load once per process.

CUDA is selected when available. CPU is the fallback used by constrained cloud hosts. The model is placed in evaluation mode.

## Sentence inference and article aggregation

The fixed label mapping is:

- 0 Bearish
- 1 Neutral
- 2 Bullish

Sentences are tokenized with dynamic batch padding and truncation at 128 tokens. Inference runs inside torch.inference_mode. Model logits are converted to three softmax class scores for each sentence.

The article score for each class is the arithmetic mean of that class score across all article sentences. The largest mean selects the article label. Softmax scores are model confidence signals, not guaranteed calibrated probabilities.

## Evidence generation

Sentence evidence comes directly from Full BERT outputs and final-layer sentence representations.

bert_visualization.py builds deterministic display data:

- PCA projects high-dimensional sentence representations into two dimensions.
- Contextual-token similarity describes representation proximity.
- Token similarity does not prove causal importance.

Lexical phrase and risk cues are rule-based secondary evidence. They do not alter Full BERT predictions or article aggregation.

## Experiment results

experiment_results.py reads verified JSON artifacts in explicit stages:

1. Read the JSON object.
2. Validate test metrics and evaluation sections.
3. Require Bearish, Neutral, Bullish label order.
4. Validate the three-by-three confusion matrix.
5. Reject negative counts.
6. Resolve classification-report metrics with the saved fallback.
7. Construct immutable ModelMetrics records.

Current reproduced metrics and historical metrics remain separate. Model Results reads the current manifest, trainer history, benchmark, classification report and confusion matrix without loading Full BERT.

## Deployment flow

Streamlit Community Cloud runs app/streamlit_app.py from main and installs app/requirements.txt. The private Hugging Face snapshot is downloaded only when Analyze Article first needs the model.

Overview, Model Results and About do not load model weights. See docs/STREAMLIT_CLOUD_DEPLOYMENT.md for secrets, cold-start behavior and public validation.

## Verified artifact locations

- reports/metrics/bert_sentiment_current_run_metrics.json
- reports/metrics/bert_sentiment_current_run_history.json
- reports/metrics/bert_sentiment_current_run_benchmark.json
- reports/metrics/bert_sentiment_metrics.json
- artifacts/manifests/bert_sentiment_current_run_manifest.json
- artifacts/manifests/financial_phrasebank_acquisition_manifest.json
- artifacts/manifests/financial_phrasebank_split_manifest.json

These are maintained evidence. Do not reformat or substitute their values.

## Local-only files

Never commit:

- .venv-bert
- artifacts/models/bert_sentiment
- Hugging Face caches
- .streamlit/secrets.toml
- raw or processed local datasets
- runtime logs, screenshots, patches and backups

## Key tests

- test_full_bert_model_source.py: local/remote source resolution and token safety.
- test_full_bert_inference.py: sentence splitting, aggregation prerequisites and class order.
- test_full_bert_streamlit_integration.py: routes, real sample analysis and stale-state behavior.
- test_experiment_results.py: metrics, matrix, history and provenance.
- test_bert_visualization.py: visualization-data calculations.
- test_public_sentiment_refinement.py: extraction, entity and evidence workflows.
- test_architecture_observatory.py: architecture content, interaction and navigation.

Run focused tests after each file change, then the compatible release suite.