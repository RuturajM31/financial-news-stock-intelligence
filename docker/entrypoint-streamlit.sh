#!/usr/bin/env sh
set -eu

umask 077

if [ -z "${FNI_API_KEY:-}" ] || [ "${#FNI_API_KEY}" -lt 24 ]; then
    echo "FAILED: FNI_API_KEY must contain at least 24 characters." >&2
    exit 1
fi

mkdir -p "${HOME}" "${HOME}/.streamlit"

exec /opt/fni/project/.venv-streamlit/bin/python -m streamlit run \
    /opt/fni/project/app/streamlit_app.py \
    --server.address=0.0.0.0 \
    --server.port=8501 \
    --server.headless=true \
    --server.enableCORS=true \
    --server.enableXsrfProtection=true \
    --browser.gatherUsageStats=false
