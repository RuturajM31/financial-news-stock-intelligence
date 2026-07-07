#!/usr/bin/env bash
#
# Run the complete BERT-wrapper verification package.
#
# Inputs
# ------
# - Existing project source under src/.
# - Main Python environment at .venv/.
# - Isolated Transformer environment at .venv-distilbert/.
#
# Processing
# ----------
# 1. Validate required files and environments.
# 2. Compile the BERT, DistilBERT, and focused test modules.
# 3. Run BERT and DistilBERT tests in the isolated environment.
# 4. Run the remaining project regression in the main environment.
#
# Output
# ------
# The final success marker appears only when every command exits successfully.
# This script does not download or train BERT.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

MAIN_PYTHON="${PROJECT_ROOT}/.venv/bin/python"
TRANSFORMER_PYTHON="${PROJECT_ROOT}/.venv-distilbert/bin/python"

required_files=(
    "src/financial_news_intelligence/models/distilbert_training.py"
    "src/financial_news_intelligence/models/bert_training.py"
    "tests/test_distilbert_training.py"
    "tests/test_bert_training.py"
)

for required_file in "${required_files[@]}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "ERROR: Required file not found: ${required_file}" >&2
        exit 1
    fi
done

if [[ ! -x "${MAIN_PYTHON}" ]]; then
    echo "ERROR: Main Python not found: ${MAIN_PYTHON}" >&2
    exit 1
fi

if [[ ! -x "${TRANSFORMER_PYTHON}" ]]; then
    echo "ERROR: Transformer Python not found: ${TRANSFORMER_PYTHON}" >&2
    exit 1
fi

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM="false"

echo
echo "========================================"
echo "1. PYTHON SYNTAX VALIDATION"
echo "========================================"

"${TRANSFORMER_PYTHON}" -m py_compile "${required_files[@]}"

echo "SYNTAX VALIDATION: PASSED"

echo
echo "========================================"
echo "2. FOCUSED BERT WRAPPER TESTS"
echo "========================================"

"${TRANSFORMER_PYTHON}" -m pytest tests/test_bert_training.py -q

echo
echo "========================================"
echo "3. FOCUSED DISTILBERT REGRESSION"
echo "========================================"

"${TRANSFORMER_PYTHON}" -m pytest tests/test_distilbert_training.py -q

echo
echo "========================================"
echo "4. MAIN PROJECT REGRESSION"
echo "========================================"

"${MAIN_PYTHON}" -m pytest \
    --ignore=tests/test_distilbert_training.py \
    --ignore=tests/test_bert_training.py \
    -q

echo
echo "========================================"
echo "FINAL BERT TEST PACKAGE: PASSED"
echo "========================================"
