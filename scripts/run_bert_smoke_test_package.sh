#!/usr/bin/env bash
#
# Run the complete BERT smoke-training verification package.
#
# Purpose
# -------
# Validate the new smoke runner, protect existing BERT and DistilBERT code,
# run the full regression matrix, and optionally execute the real one-epoch
# BERT smoke experiment in the isolated Transformer environment.
#
# Inputs
# ------
# - Existing project source under src/.
# - Main Python environment at .venv/.
# - Isolated Transformer environment at .venv-distilbert/.
# - Verified Financial PhraseBank JSONL split files.
#
# Output
# ------
# A final success marker appears only when every requested command exits
# successfully. Real model download and training happen only with
# --run-training.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

MAIN_PYTHON="${PROJECT_ROOT}/.venv/bin/python"
TRANSFORMER_PYTHON="${PROJECT_ROOT}/.venv-distilbert/bin/python"
RUN_TRAINING="false"
REPLACE_EXISTING="false"

for argument in "$@"; do
    case "${argument}" in
        --run-training)
            RUN_TRAINING="true"
            ;;
        --replace-existing)
            REPLACE_EXISTING="true"
            ;;
        *)
            echo "ERROR: Unknown argument: ${argument}" >&2
            exit 2
            ;;
    esac
done

required_files=(
    "src/financial_news_intelligence/models/distilbert_training.py"
    "src/financial_news_intelligence/models/bert_training.py"
    "tests/test_distilbert_training.py"
    "tests/test_bert_training.py"
    "tests/test_bert_smoke_runner.py"
    "scripts/run_bert_smoke.py"
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
echo "2. FOCUSED BERT SMOKE-RUNNER TESTS"
echo "========================================"

"${TRANSFORMER_PYTHON}" -m pytest tests/test_bert_smoke_runner.py -q

echo
echo "========================================"
echo "3. FOCUSED BERT WRAPPER REGRESSION"
echo "========================================"

"${TRANSFORMER_PYTHON}" -m pytest tests/test_bert_training.py -q

echo
echo "========================================"
echo "4. FOCUSED DISTILBERT REGRESSION"
echo "========================================"

"${TRANSFORMER_PYTHON}" -m pytest tests/test_distilbert_training.py -q

echo
echo "========================================"
echo "5. MAIN PROJECT REGRESSION"
echo "========================================"

"${MAIN_PYTHON}" -m pytest tests \
    --ignore=tests/test_distilbert_training.py \
    --ignore=tests/test_bert_training.py \
    --ignore=tests/test_bert_smoke_runner.py \
    -q

if [[ "${RUN_TRAINING}" == "true" ]]; then
    echo
    echo "========================================"
    echo "6. REAL BERT SMOKE TRAINING"
    echo "========================================"

    # Bash 3.2 on macOS treats an empty array expansion as an unbound
    # variable when ``set -u`` is active. Use explicit command branches so
    # the normal smoke run passes no extra argument and replacement remains
    # opt-in and visible.
    if [[ "${REPLACE_EXISTING}" == "true" ]]; then
        "${TRANSFORMER_PYTHON}" scripts/run_bert_smoke.py \
            --replace-existing
    else
        "${TRANSFORMER_PYTHON}" scripts/run_bert_smoke.py
    fi
else
    echo
    echo "REAL BERT SMOKE TRAINING: NOT REQUESTED"
fi

echo
echo "========================================"
echo "FINAL BERT SMOKE TEST PACKAGE: PASSED"
echo "========================================"
