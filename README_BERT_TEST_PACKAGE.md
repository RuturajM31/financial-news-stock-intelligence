# BERT Wrapper Verification Package

## Purpose

This project document explains the focused verification installed by the BERT
full-strike package. The verification runs before any BERT smoke training or
full training begins.

## Installed components

- `src/financial_news_intelligence/models/bert_training.py`
- `tests/test_bert_training.py`
- `scripts/run_bert_test_package.sh`
- `PACKAGE_MANIFEST.json`

The existing shared engine remains protected at:

- `src/financial_news_intelligence/models/distilbert_training.py`

The strike installer validates that the shared engine exposes the approved
experiment-identity contract. It does not overwrite that verified file.

## What is tested

The seven focused BERT tests verify:

1. the approved BERT experiment identity;
2. the unchanged DistilBERT baseline identity;
3. equality of all inherited settings except approved BERT overrides;
4. exact and isolated BERT artifact destinations;
5. rejection of unsupported model IDs;
6. unchanged delegation of a supplied configuration;
7. automatic creation of the default BERT configuration.

The runner then executes the focused DistilBERT regression and the remaining
main-project regression in their correct Python environments.

## Run again after installation

From the repository root:

```bash
bash scripts/run_bert_test_package.sh
```

A successful run ends with:

```text
FINAL BERT TEST PACKAGE: PASSED
```

## Safety boundary

This verification does not download BERT, tokenize the full dataset, train a
model, create checkpoints, or modify trained-model artifacts. Real model
behavior remains unverified until the later BERT smoke and full-training runs.
