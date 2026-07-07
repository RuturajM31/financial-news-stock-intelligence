#!/usr/bin/env bash
# Run isolated regression, movement modeling, and gated intelligence in order.
set -euo pipefail

PROJECT_ROOT="${1:?Usage: run_movement_intelligence_suite.sh PROJECT_ROOT}"
MAIN_PYTHON="$PROJECT_ROOT/.venv/bin/python"
DIAGNOSTIC_DIR="${MOVEMENT_DIAGNOSTIC_DIR:-$PROJECT_ROOT/reports/diagnostics/movement_run}"

# Diagnostics can be outside the project transaction. They survive rollback and
# contain only derived metrics, labels, probabilities, and runtime metadata.
mkdir -p "$DIAGNOSTIC_DIR"
chmod 700 "$DIAGNOSTIC_DIR"

# Thread limits reduce oversubscription but do not hide incompatible runtimes.
# Every relevant process also checks or rejects mixed OpenMP families directly.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=42
export PYTHONPATH="$PROJECT_ROOT/src"

printf '\n========================================\n'
printf '1. MOVEMENT NATIVE-RUNTIME PREFLIGHT\n'
printf '========================================\n'
"$MAIN_PYTHON" \
  "$PROJECT_ROOT/scripts/check_movement_runtime.py" \
  --project-root "$PROJECT_ROOT" \
  --diagnostic-dir "$DIAGNOSTIC_DIR"

printf '\n========================================\n'
printf '2. MOVEMENT-INTELLIGENCE SYNTAX AND FOCUSED TESTS\n'
printf '========================================\n'
"$MAIN_PYTHON" -m compileall -q \
  "$PROJECT_ROOT/src/financial_news_intelligence/models" \
  "$PROJECT_ROOT/src/financial_news_intelligence/intelligence" \
  "$PROJECT_ROOT/scripts/check_movement_runtime.py" \
  "$PROJECT_ROOT/scripts/run_isolated_project_regression.py" \
  "$PROJECT_ROOT/scripts/run_movement_intelligence.py" \
  "$PROJECT_ROOT/scripts/verify_movement_intelligence.py"
"$MAIN_PYTHON" -m pytest -q \
  "$PROJECT_ROOT/tests/test_movement_pipeline.py" \
  "$PROJECT_ROOT/tests/test_intelligence_pipeline.py" \
  "$PROJECT_ROOT/tests/test_runtime_isolation.py"

printf '\n========================================\n'
printf '3. PRE-RUN ISOLATED PROJECT REGRESSION\n'
printf '========================================\n'
"$MAIN_PYTHON" \
  "$PROJECT_ROOT/scripts/run_isolated_project_regression.py" \
  --project-root "$PROJECT_ROOT" \
  --phase pre-run

printf '\n========================================\n'
printf '4. MOVEMENT MODEL TRAINING AND QUALITY GATES\n'
printf '========================================\n'
"$MAIN_PYTHON" \
  "$PROJECT_ROOT/scripts/run_movement_intelligence.py" \
  --project-root "$PROJECT_ROOT" \
  --phase movement \
  --replace-existing \
  --diagnostic-dir "$DIAGNOSTIC_DIR"

printf '\n========================================\n'
printf '5. INDEPENDENT MOVEMENT VERIFICATION\n'
printf '========================================\n'
"$MAIN_PYTHON" \
  "$PROJECT_ROOT/scripts/verify_movement_intelligence.py" \
  --project-root "$PROJECT_ROOT" \
  --phase movement

printf '\n========================================\n'
printf '6. EXPLAINABILITY AND INTELLIGENCE\n'
printf '========================================\n'
"$MAIN_PYTHON" \
  "$PROJECT_ROOT/scripts/run_movement_intelligence.py" \
  --project-root "$PROJECT_ROOT" \
  --phase intelligence

printf '\n========================================\n'
printf '7. FINAL SEMANTIC AND CHECKSUM VERIFICATION\n'
printf '========================================\n'
"$MAIN_PYTHON" \
  "$PROJECT_ROOT/scripts/verify_movement_intelligence.py" \
  --project-root "$PROJECT_ROOT" \
  --phase all

printf '\n========================================\n'
printf '8. POST-RUN ISOLATED PROJECT REGRESSION\n'
printf '========================================\n'
"$MAIN_PYTHON" \
  "$PROJECT_ROOT/scripts/run_isolated_project_regression.py" \
  --project-root "$PROJECT_ROOT" \
  --phase post-run

printf '\n========================================\n'
printf 'MOVEMENT INTELLIGENCE SUITE: PASSED\n'
printf 'DIAGNOSTIC EVIDENCE: %s\n' "$DIAGNOSTIC_DIR"
printf 'STOCK MOVEMENT MODEL: COMPLETED AND QUALITY-GATED\n'
printf 'EXPLAINABILITY AND INTELLIGENCE: COMPLETED\n'
printf 'APPLICATION DEPLOYMENT CHANGED: FALSE\n'
printf '========================================\n'
