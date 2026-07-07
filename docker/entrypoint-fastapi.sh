#!/usr/bin/env sh
set -eu

umask 077

if [ -z "${FNI_API_KEY:-}" ] || [ "${#FNI_API_KEY}" -lt 24 ]; then
    echo "FAILED: FNI_API_KEY must contain at least 24 characters." >&2
    exit 1
fi

mkdir -p /tmp/huggingface /tmp/fni-home

exec /opt/fni/project/.venv/bin/python -m uvicorn \
    financial_news_intelligence.api.app:create_app \
    --factory \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --no-access-log \
    --log-level info
