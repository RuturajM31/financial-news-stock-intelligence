"""Secret-safe structured logging for API lifecycle and requests."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Mapping


SENSITIVE_MARKERS = (
    "authorization",
    "api_key",
    "apikey",
    "password",
    "secret",
    "tiingo",
    "token",
)


def redact_mapping(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a shallow copy with secret-looking fields removed."""

    redacted: dict[str, Any] = {}
    for key, value in payload.items():
        normalized_key = str(key).lower()
        if any(marker in normalized_key for marker in SENSITIVE_MARKERS):
            redacted[str(key)] = "[REDACTED]"
        else:
            redacted[str(key)] = value
    return redacted


class JsonFormatter(logging.Formatter):
    """Format one log event as stable JSON without request content."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "marker": getattr(record, "marker", "WARNING"),
            "logger": record.name,
            "message": record.getMessage(),
        }
        for field_name in (
            "request_id",
            "method",
            "path",
            "status_code",
            "duration_ms",
            "client_ip",
        ):
            value = getattr(record, field_name, None)
            if value is not None:
                payload[field_name] = value
        return json.dumps(redact_mapping(payload), sort_keys=True)


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure the project API logger once and return it."""

    logger = logging.getLogger("financial_news_intelligence.api")
    logger.setLevel(level)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    return logger
