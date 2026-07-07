# BERT Smoke Training Package

## Purpose

This package adds a controlled one-epoch smoke experiment for the approved
full-BERT financial-sentiment benchmark.

The smoke run answers one question before the expensive full benchmark:

> Can the real BERT checkpoint, verified dataset splits, shared training
> engine, evaluation logic, and artifact-saving path complete end to end?

Smoke metrics are pipeline evidence only. They are not used to select the
champion model.

## Installed files

1. `scripts/run_bert_smoke.py`
2. `tests/test_bert_smoke_runner.py`
3. `scripts/run_bert_smoke_test_package.sh`
4. `README_BERT_SMOKE_PACKAGE.md`
5. `BERT_SMOKE_PACKAGE_MANIFEST.json`

The installer does not overwrite the verified BERT wrapper, shared DistilBERT
engine, existing training tests, completed DistilBERT model, or future full
BERT output locations.

## Real-data sample

The runner creates deterministic balanced samples while preserving every
original JSONL field:

| Split | Records per class | Total records |
|---|---:|---:|
| Train | 9 | 27 |
| Validation | 3 | 9 |
| Test | 3 | 9 |

The experiment uses one epoch and the same model ID, tokenizer, labels, class
weighting, training engine, and evaluation logic as the full BERT benchmark.

## Dedicated outputs

- `data/interim/bert_smoke/`
- `artifacts/models/bert_smoke/`
- `reports/metrics/bert_smoke_metrics.json`
- `artifacts/manifests/bert_smoke_manifest.json`

Existing files at these paths are protected. They are replaced only when the
operator explicitly supplies `--replace-existing`.

## Environment safety

Run the experiment with `.venv-distilbert`. The runner refuses an environment
where scikit-learn is visible because the project has already verified that
PyTorch and scikit-learn can load conflicting OpenMP runtimes on the target
Intel macOS machine.

## Complete verification sequence

The strike runner performs:

1. package checksum verification;
2. project prerequisite checks;
3. Python and Bash syntax checks;
4. focused smoke-runner tests;
5. BERT wrapper regression;
6. DistilBERT training-module regression;
7. main project regression;
8. real one-epoch BERT smoke training;
9. saved manifest, metrics, confusion matrix, and final-model checks.

## Success markers

The real training stage must print:

```text
BERT SMOKE TRAINING: PASSED
```

The complete package must end with:

```text
BERT SMOKE STRIKE PACKAGE: PASSED
```

## Network and runtime

The first execution may download `google-bert/bert-base-uncased`. The package
uses only the zero-cost Hugging Face checkpoint and local CPU compute. Runtime
depends on model cache, network speed, and the target machine.


## macOS shell compatibility

The real-training branch uses explicit commands for normal and replacement
runs. It does not expand an empty Bash array under ``set -u``, which keeps the
launcher safe on the Bash 3.2 version shipped with older macOS releases.
