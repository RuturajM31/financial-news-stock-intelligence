#!/usr/bin/env bash
#
# Run full BERT, LoRA, and final sentiment-model comparison in one sequence.
#
# Purpose
# -------
# Validate all new files, protect the existing DistilBERT and BERT wrapper,
# train full BERT, train BERT-LoRA, compare all three experiments, and repeat
# regression checks after artifact creation.
#
# Inputs
# ------
# - Existing verified project source and Financial PhraseBank splits.
# - Main environment at .venv/bin/python.
# - Isolated Transformer environment at .venv-distilbert/bin/python.
# - Completed DistilBERT metrics, model, and manifest.
#
# Outputs
# -------
# Full-BERT artifacts, LoRA artifacts, comparison JSON, champion manifest, and
# final success markers. The script does not change API or Streamlit code.
#
# Safety
# ------
# This script uses a child Bash process and avoids nounset so shell options
# cannot leak into the user's interactive terminal or trigger macOS Bash 3.2
# empty-array failures.

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MAIN_PYTHON="${PROJECT_ROOT}/.venv/bin/python"
TRANSFORMER_PYTHON="${PROJECT_ROOT}/.venv-distilbert/bin/python"

cd "${PROJECT_ROOT}"

if [[ ! -x "${MAIN_PYTHON}" ]]; then
    echo "ERROR: Main Python is not executable: ${MAIN_PYTHON}" >&2
    exit 1
fi

if [[ ! -x "${TRANSFORMER_PYTHON}" ]]; then
    echo "ERROR: Transformer Python is not executable: ${TRANSFORMER_PYTHON}" >&2
    exit 1
fi

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM="false"

printf '\n========================================\n'
printf '1. PYTHON AND BASH SYNTAX VALIDATION\n'
printf '========================================\n'

"${TRANSFORMER_PYTHON}" -m py_compile \
    src/financial_news_intelligence/models/distilbert_training.py \
    src/financial_news_intelligence/models/bert_training.py \
    src/financial_news_intelligence/models/lora_training.py \
    src/financial_news_intelligence/models/sentiment_comparison.py \
    scripts/run_full_bert.py \
    scripts/run_lora_training.py \
    scripts/run_sentiment_comparison.py \
    tests/test_distilbert_training.py \
    tests/test_bert_training.py \
    tests/test_full_bert_runner.py \
    tests/test_lora_training.py \
    tests/test_sentiment_comparison.py

bash -n scripts/run_sentiment_benchmark_suite.sh
printf 'SYNTAX VALIDATION: PASSED\n'

printf '\n========================================\n'
printf '2. NEW BENCHMARK TESTS\n'
printf '========================================\n'

"${TRANSFORMER_PYTHON}" -m pytest \
    tests/test_full_bert_runner.py \
    tests/test_lora_training.py \
    tests/test_sentiment_comparison.py \
    -q

printf '\n========================================\n'
printf '3. EXISTING TRANSFORMER REGRESSION\n'
printf '========================================\n'

"${TRANSFORMER_PYTHON}" -m pytest \
    tests/test_bert_training.py \
    tests/test_distilbert_training.py \
    -q

printf '\n========================================\n'
printf '4. MAIN PROJECT REGRESSION\n'
printf '========================================\n'

"${MAIN_PYTHON}" -m pytest tests \
    --ignore=tests/test_distilbert_training.py \
    --ignore=tests/test_bert_training.py \
    --ignore=tests/test_bert_smoke_runner.py \
    --ignore=tests/test_full_bert_runner.py \
    --ignore=tests/test_lora_training.py \
    --ignore=tests/test_sentiment_comparison.py \
    -q

printf '\n========================================\n'
printf '5. FULL BERT TRAINING AND VERIFICATION\n'
printf '========================================\n'

"${TRANSFORMER_PYTHON}" scripts/run_full_bert.py

printf '\n========================================\n'
printf '6. BERT LORA TRAINING AND VERIFICATION\n'
printf '========================================\n'

"${TRANSFORMER_PYTHON}" scripts/run_lora_training.py

printf '\n========================================\n'
printf '7. FINAL SENTIMENT MODEL COMPARISON\n'
printf '========================================\n'

"${TRANSFORMER_PYTHON}" scripts/run_sentiment_comparison.py

printf '\n========================================\n'
printf '8. POST-TRAINING EVIDENCE VERIFICATION\n'
printf '========================================\n'

"${TRANSFORMER_PYTHON}" scripts/run_full_bert.py --verify-only
"${TRANSFORMER_PYTHON}" scripts/run_lora_training.py --verify-only
"${TRANSFORMER_PYTHON}" scripts/run_sentiment_comparison.py --verify-only

printf '\n========================================\n'
printf '9. POST-TRAINING REGRESSION\n'
printf '========================================\n'

"${TRANSFORMER_PYTHON}" -m pytest \
    tests/test_full_bert_runner.py \
    tests/test_lora_training.py \
    tests/test_sentiment_comparison.py \
    tests/test_bert_training.py \
    tests/test_distilbert_training.py \
    -q

"${MAIN_PYTHON}" -m pytest tests \
    --ignore=tests/test_distilbert_training.py \
    --ignore=tests/test_bert_training.py \
    --ignore=tests/test_bert_smoke_runner.py \
    --ignore=tests/test_full_bert_runner.py \
    --ignore=tests/test_lora_training.py \
    --ignore=tests/test_sentiment_comparison.py \
    -q

printf '\n========================================\n'
printf 'SENTIMENT BENCHMARK SUITE: PASSED\n'
printf 'FULL BERT: COMPLETED\n'
printf 'BERT LORA: COMPLETED\n'
printf 'MODEL COMPARISON: COMPLETED\n'
printf 'APPLICATION DEPLOYMENT CHANGED: FALSE\n'
printf '========================================\n'
