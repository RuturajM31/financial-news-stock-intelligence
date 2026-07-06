#!/usr/bin/env bash
set -euo pipefail

# This helper is intentionally separate from the strike installer because it
# changes a real cluster. Run it only during the later deployment stage.
NAMESPACE="${1:-financial-news-intelligence}"
SECRET_NAME="${2:-financial-news-intelligence-api}"

if [[ -z "${FNI_API_KEY:-}" || ${#FNI_API_KEY} -lt 24 ]]; then
  echo "ERROR: FNI_API_KEY must contain at least 24 characters." >&2
  exit 1
fi

# Generate each object locally first. The API key is read from stdin so it is
# not exposed in shell history or in the kubectl process argument list.
kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml   | kubectl apply -f -
printf '%s' "$FNI_API_KEY"   | kubectl -n "$NAMESPACE" create secret generic "$SECRET_NAME"       --from-file=api-key=/dev/stdin --dry-run=client -o yaml   | kubectl apply -f -

unset FNI_API_KEY
echo "API KEY SECRET: APPLIED"
