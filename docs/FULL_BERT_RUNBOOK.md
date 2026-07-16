# Full BERT training and local inference

## Verified environment

- Python 3.12
- `torch 2.7.1+cu128` for the local NVIDIA RTX 5070 Laptop GPU
- `transformers 4.46.3`
- Base model: `google-bert/bert-base-uncased`
- Pinned base revision: `86b5e0934494bd15c9632b12f734a8a67f723594`
- Labels: `0 Bearish`, `1 Neutral`, `2 Bullish`
- Maximum sequence length: 128

The repository's deployment requirements remain unchanged. The separate
`.venv-bert` environment is used because the local Blackwell GPU requires a
newer CUDA-enabled PyTorch build than the deployment environment.

## Train

From the repository root in PowerShell:

```powershell
$env:PYTHONPATH = "src"
$env:HF_HUB_DISABLE_XET = "1"
$env:TOKENIZERS_PARALLELISM = "false"
.venv-bert\Scripts\python.exe scripts\run_full_bert.py
```

The runner validates dataset checksums, uses the fixed train/validation/test
split, evaluates the held-out test split once, and writes the final model to:

`artifacts/models/bert_sentiment/final_model`

Current-run metrics are written separately from the historical benchmark:

- `reports/metrics/bert_sentiment_current_run_metrics.json`
- `reports/metrics/bert_sentiment_current_run_history.json`
- `artifacts/manifests/bert_sentiment_current_run_manifest.json`

## Validate inference

```powershell
$env:PYTHONPATH = "src"
.venv-bert\Scripts\python.exe scripts\benchmark_full_bert_inference.py
```

The benchmark checks all three labels and writes measured timing and memory to
`reports/metrics/bert_sentiment_current_run_benchmark.json`.

## Run the public application

```powershell
$env:PYTHONPATH = "src"
.venv-bert\Scripts\python.exe -m streamlit run app\public_cloud_app.py --server.port 8502
```

The app caches one model runtime per process, batches article sentences, and
falls back to CPU automatically when CUDA is unavailable. It produces article
sentiment and sentence evidence only; it does not predict prices or returns.

## Required artifact files

- `config.json`
- `model.safetensors`
- `tokenizer.json`
- `tokenizer_config.json`
- `vocab.txt`

If any file is missing, the Analyze Article page shows a clear model-artifact
error and does not reuse an older result.
