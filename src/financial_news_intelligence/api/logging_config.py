"""Secret-safe structured logging shared by API and monitoring code.

Purpose
-------
Provide one JSON logging contract for the FastAPI process and reusable service
operations. The formatter records operational metadata only. Request bodies,
article text, uploaded bytes, API keys, and provider tokens are never added.

Inputs and downstream use
-------------------------
Callers bind a validated request identifier, obtain a named logger, and attach
only allow-listed fields through ``extra``. Container, Kubernetes, and CI/CD
stages can consume the stable JSON records without changing application code.
"""

from __future__ import annotations

import contextvars
import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Mapping


SENSITIVE_MARKERS = (
    "authorization",
    "api_key",
    "apikey",
    "credential",
    "password",
    "private_key",
    "secret",
    "tiingo",
    "token",
)
_ALLOWED_RECORD_FIELDS = (
    "request_id",
    "method",
    "path",
    "status_code",
    "duration_ms",
    "client_ip",
    "operation",
    "outcome",
    "component",
    "count",
)
_REQUEST_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "fni_request_id",
    default=None,
)
_SECRET_VALUE_PATTERNS = (
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"\bghp_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(
        r"(?i)(api[_-]?key|token|password|secret|authorization)"
        r"\s*[:=]\s*[^\s,;]+"
    ),
)


def _sanitize_text(value: str, maximum_length: int = 1_000) -> str:
    """Remove token-shaped values and bound one log-safe string."""

    sanitized = value.replace("\r", "\\r").replace("\n", "\\n")
    for pattern in _SECRET_VALUE_PATTERNS:
        sanitized = pattern.sub("[REDACTED]", sanitized)
    if len(sanitized) > maximum_length:
        sanitized = sanitized[: maximum_length - 3] + "..."
    return sanitized


def redact_value(value: Any, *, depth: int = 0) -> Any:
    """Return a recursively redacted JSON-compatible representation.

    Recursion and collection sizes are bounded so malformed diagnostic objects
    cannot turn logging into an unbounded memory operation.
    """

    if depth > 4:
        return "[MAX_DEPTH]"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _sanitize_text(value)
    if isinstance(value, Mapping):
        redacted: dict[str, Any] = {}
        for index, (key, nested_value) in enumerate(value.items()):
            if index >= 50:
                redacted["[TRUNCATED]"] = True
                break
            safe_key = _sanitize_text(str(key), maximum_length=100)
            normalized = safe_key.lower()
            if any(marker in normalized for marker in SENSITIVE_MARKERS):
                redacted[safe_key] = "[REDACTED]"
            else:
                redacted[safe_key] = redact_value(nested_value, depth=depth + 1)
        return redacted
    if isinstance(value, (list, tuple, set, frozenset)):
        items = list(value)[:50]
        return [redact_value(item, depth=depth + 1) for item in items]
    return _sanitize_text(type(value).__name__)


def redact_mapping(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a recursively redacted mapping with stable string keys."""

    result = redact_value(payload)
    if not isinstance(result, dict):
        return {"value": result}
    return result


def bind_request_context(request_id: str) -> contextvars.Token[str | None]:
    """Bind one validated request ID for logs emitted during this request."""

    return _REQUEST_ID.set(request_id)


def reset_request_context(token: contextvars.Token[str | None]) -> None:
    """Restore the previous request context after request completion."""

    _REQUEST_ID.reset(token)


def get_logger(name: str) -> logging.Logger:
    """Return a logger below the centrally configured project namespace."""

    return logging.getLogger(name)


class JsonFormatter(logging.Formatter):
    """Format one event as stable, bounded, secret-safe JSON."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "marker": getattr(record, "marker", "INFO"),
            "logger": record.name,
            "message": _sanitize_text(record.getMessage()),
        }
        request_id = getattr(record, "request_id", None) or _REQUEST_ID.get()
        if request_id is not None:
            payload["request_id"] = request_id
        for field_name in _ALLOWED_RECORD_FIELDS:
            if field_name == "request_id":
                continue
            value = getattr(record, field_name, None)
            if value is not None:
                payload[field_name] = value
        if record.exc_info and record.exc_info[0] is not None:
            payload["exception_type"] = record.exc_info[0].__name__
        return json.dumps(redact_mapping(payload), sort_keys=True)


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure the whole project namespace once and return the API logger."""

    project_logger = logging.getLogger("financial_news_intelligence")
    project_logger.setLevel(level)
    project_logger.propagate = False
    if not project_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        project_logger.addHandler(handler)
    else:
        for handler in project_logger.handlers:
            handler.setFormatter(JsonFormatter())
    return logging.getLogger("financial_news_intelligence.api")
