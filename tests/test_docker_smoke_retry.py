"""Behavior tests for Docker host-publication retry handling."""

from __future__ import annotations

import importlib.util
import urllib.error
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "docker/smoke_test.py"


def load_module():
    spec = importlib.util.spec_from_file_location("docker_smoke_test", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_wait_for_endpoint_retries_connection_refusal(monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_module()
    attempts = {"count": 0}

    def fake_fetch(_url: str):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise urllib.error.URLError(ConnectionRefusedError(61, "Connection refused"))
        return 200, '{"status":"PASSED"}'

    monkeypatch.setattr(module, "fetch", fake_fetch)
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)

    status, body = module.wait_for_endpoint(
        "http://127.0.0.1:18000/health",
        label="FastAPI health",
        validator=module.valid_json_status("PASSED"),
        startup_timeout_seconds=5,
    )

    assert attempts["count"] == 2
    assert status == 200
    assert "PASSED" in body


def test_wait_for_endpoint_rejects_invalid_payload_until_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_module()
    responses = iter(
        [
            (200, "not-json"),
            (200, '{"status":"STARTING"}'),
            (200, '{"status":"PASSED"}'),
        ]
    )

    monkeypatch.setattr(module, "fetch", lambda _url: next(responses))
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)

    status, _body = module.wait_for_endpoint(
        "http://127.0.0.1:18000/health",
        label="FastAPI health",
        validator=module.valid_json_status("PASSED"),
        startup_timeout_seconds=5,
    )

    assert status == 200

def test_wait_for_endpoint_forwards_authentication_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_module()
    captured: dict[str, object] = {}

    def fake_fetch(_url: str, *, headers=None):
        captured["headers"] = headers
        return 200, "fni_build_info 1\nfni_http_requests_total 1\n"

    monkeypatch.setattr(module, "fetch", fake_fetch)

    status, _body = module.wait_for_endpoint(
        "http://127.0.0.1:18000/metrics",
        label="Prometheus metrics",
        validator=lambda response_status, body: (
            response_status == 200 and "fni_build_info" in body
        ),
        startup_timeout_seconds=5,
        headers={"X-API-Key": "temporary-test-key-1234567890"},
    )

    assert status == 200
    assert captured["headers"] == {
        "X-API-Key": "temporary-test-key-1234567890"
    }

