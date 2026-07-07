"""Focused contracts for centralized logging and Prometheus monitoring."""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from financial_news_intelligence.api.app import create_app
from financial_news_intelligence.api.config import ApiSettings
from financial_news_intelligence.api.logging_config import (
    JsonFormatter,
    bind_request_context,
    redact_mapping,
    reset_request_context,
)
from financial_news_intelligence.api.monitoring import (
    get_metrics_registry,
    observe_operation,
    reset_metrics_registry,
)


class FakeServices:
    """Provide only the service methods needed by monitoring route tests."""

    def close(self) -> None:
        return None

    def readiness(self, run_deep_probe: bool = False) -> dict[str, Any]:
        return {
            "components": {"artifacts": "PASSED"},
            "details": {"deep_probe_run": run_deep_probe},
        }


def _client(tmp_path: Path) -> TestClient:
    settings = ApiSettings(
        project_root=tmp_path,
        environment="test",
        api_key="m" * 32,
        require_api_key=True,
        trusted_hosts=("testserver",),
    )
    return TestClient(create_app(settings=settings, services=FakeServices()))


def test_recursive_redaction_removes_keys_and_token_shaped_values() -> None:
    """Redact nested credentials even when a caller supplies unsafe metadata."""

    redacted = redact_mapping(
        {
            "safe": "kept",
            "nested": {"api_key": "top-secret", "note": "token=abcdef123456"},
            "authorization": "Bearer abcdefghijklmnop",
        }
    )

    rendered = json.dumps(redacted)
    assert redacted["safe"] == "kept"
    assert "top-secret" not in rendered
    assert "abcdefghijklmnop" not in rendered
    assert "abcdef123456" not in rendered
    assert rendered.count("[REDACTED]") >= 3


def test_json_formatter_uses_bound_request_context_without_traceback_data() -> None:
    """Attach correlation IDs while excluding exception messages and secrets."""

    token = bind_request_context("request-12345678")
    try:
        record = logging.LogRecord(
            name="financial_news_intelligence.api.test",
            level=logging.ERROR,
            pathname=__file__,
            lineno=1,
            msg="authorization=Bearer abcdefghijklmnop",
            args=(),
            exc_info=None,
        )
        payload = json.loads(JsonFormatter().format(record))
    finally:
        reset_request_context(token)

    assert payload["request_id"] == "request-12345678"
    assert "abcdefghijklmnop" not in payload["message"]
    assert payload["message"] == "[REDACTED]"


def test_metrics_endpoint_requires_api_key_and_exports_aggregate_data(
    tmp_path: Path,
) -> None:
    """Protect metrics and expose only aggregate process information."""

    reset_metrics_registry()
    with _client(tmp_path) as client:
        denied = client.get("/metrics")
        accepted = client.get("/metrics", headers={"X-API-Key": "m" * 32})

    assert denied.status_code == 401
    assert accepted.status_code == 200
    assert "fni_build_info" in accepted.text
    assert "fni_http_request_duration_seconds" in accepted.text
    assert "mmmmmmmm" not in accepted.text
    assert "request_id" not in accepted.text


def test_http_metrics_use_fixed_route_templates(tmp_path: Path) -> None:
    """Prevent unknown user paths from creating unbounded metric labels."""

    registry = reset_metrics_registry()
    with _client(tmp_path) as client:
        response = client.get("/unknown/user/value")
    assert response.status_code == 404

    rendered = registry.render()
    assert 'path="unmatched"' in rendered
    assert "/unknown/user/value" not in rendered


def test_operation_decorator_records_success_and_failure() -> None:
    """Measure both operation outcomes without swallowing exceptions."""

    registry = reset_metrics_registry()

    @observe_operation("focused_success")
    def passed() -> int:
        return 7

    @observe_operation("focused_failure")
    def failed() -> None:
        raise RuntimeError("expected")

    assert passed() == 7
    with pytest.raises(RuntimeError, match="expected"):
        failed()

    rendered = registry.render()
    assert 'operation="focused_success",outcome="passed"' in rendered
    assert 'operation="focused_failure",outcome="failed"' in rendered


def test_formatter_output_is_one_json_line() -> None:
    """Keep logs compatible with container and Kubernetes collectors."""

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("financial_news_intelligence.focused")
    logger.handlers = [handler]
    logger.propagate = False
    logger.setLevel(logging.INFO)

    logger.info("Monitoring package test.", extra={"marker": "PASSED"})
    output = stream.getvalue()

    assert output.count("\n") == 1
    payload = json.loads(output)
    assert payload["marker"] == "PASSED"
