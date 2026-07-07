"""Tests for the real local HTTP smoke-test contract."""

from __future__ import annotations

import socket

import pytest

from scripts import verify_fastapi_http_smoke as smoke


def test_free_loopback_port_returns_bindable_port() -> None:
    """Prepare a local socket, request a port, and check it can be rebound."""

    port = smoke.free_loopback_port()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", port))


def test_validate_health_accepts_passed_response() -> None:
    """Prepare a passed health response, validate it, and check no failure."""

    smoke.validate_health(200, {"status": "PASSED"})


def test_validate_health_rejects_failed_response() -> None:
    """Prepare a failed health response, validate it, and check rejection."""

    with pytest.raises(smoke.HttpSmokeError, match="Health response failed"):
        smoke.validate_health(503, {"status": "FAILED"})


def test_validate_readiness_requires_lightweight_worker_state() -> None:
    """Prepare worker PASSED states, validate readiness, and reject deep probes."""

    payload = {
        "status": "PASSED",
        "components": {
            "artifacts": "PASSED",
            "movement_worker": "PASSED",
            "sentiment_worker": "PASSED",
        },
        "details": {"deep_probe_run": True},
    }

    with pytest.raises(smoke.HttpSmokeError, match="movement"):
        smoke.validate_readiness(200, payload)
