#!/usr/bin/env bash
# Run the final foundation as one auditable validation sequence.
#
# Inputs:
#   $1 is the project root. TIINGO_API_TOKEN must already exist in the
#   environment and is never printed.
#
# Outputs:
#   The script runs focused tests, full regressions, the live foundation build,
#   a read-back verification pass, and final regressions. Any failure returns a
#   non-zero exit code so the strike installer can restore controlled files.

set -euo pipefail

PROJECT_ROOT="${1:?Project root is required.}"
MAIN_PYTHON="$PROJECT_ROOT/.venv/bin/python"
TRANSFORMER_PYTHON="$PROJECT_ROOT/.venv-distilbert/bin/python"

if [[ -z "${TIINGO_API_TOKEN:-}" ]]; then
    echo "ERROR: TIINGO_API_TOKEN is empty." >&2
    exit 1
fi

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PROJECT_ROOT/src"

printf '\n========================================\n'
printf '1. FOUNDATION SYNTAX AND FOCUSED TESTS\n'
printf '========================================\n'
"$TRANSFORMER_PYTHON" -m py_compile \
  "$PROJECT_ROOT/src/financial_news_intelligence/data/"foundation_*.py \
  "$PROJECT_ROOT/src/financial_news_intelligence/data/market_data_foundation.py" \
  "$PROJECT_ROOT/scripts/run_market_data_foundation.py"
"$TRANSFORMER_PYTHON" -m pytest -q \
  "$PROJECT_ROOT/tests/test_market_data_foundation.py"

printf '\n========================================\n'
printf '2. PRE-BUILD MAIN PROJECT REGRESSION\n'
printf '========================================\n'
"$MAIN_PYTHON" -m pytest -q "$PROJECT_ROOT/tests"

printf '\n========================================\n'
printf '3. LIVE QUALIFIED FOUNDATION BUILD\n'
printf '========================================\n'
"$TRANSFORMER_PYTHON" \
  "$PROJECT_ROOT/scripts/run_market_data_foundation.py" \
  --project-root "$PROJECT_ROOT" \
  --replace-existing

printf '\n========================================\n'
printf '4. POST-BUILD EVIDENCE VERIFICATION\n'
printf '========================================\n'
"$TRANSFORMER_PYTHON" \
  "$PROJECT_ROOT/scripts/run_market_data_foundation.py" \
  --project-root "$PROJECT_ROOT" \
  --verify-only

printf '\n========================================\n'
printf '5. POST-BUILD PROJECT REGRESSION\n'
printf '========================================\n'
"$MAIN_PYTHON" -m pytest -q "$PROJECT_ROOT/tests"

printf '\n========================================\n'
printf 'MARKET DATA FOUNDATION SUITE: PASSED\n'
printf 'STOCK MOVEMENT MODEL TRAINED: FALSE\n'
printf 'APPLICATION DEPLOYMENT CHANGED: FALSE\n'
printf '========================================\n'
