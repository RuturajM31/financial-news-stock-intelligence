"""Verify FastAPI connection safety without making a network request."""

from __future__ import annotations

import json
from typing import Any

import pytest

from tests.streamlit_test_support import import_project_module, python_files


class FakeResponse:
    """Provide the small urllib response surface used by the API client."""

    def __init__(self, payload: dict[str, Any]) -> None:
        """Store one small JSON body and a safe response identifier."""

        self._body = json.dumps(payload).encode("utf-8")
        self.headers = {"X-Request-ID": "safe-request-id"}

    def __enter__(self) -> "FakeResponse":
        """Return this response from a context manager."""

        return self

    def __exit__(self, *_: Any) -> None:
        """Close the fake context without suppressing exceptions."""

        return None

    def read(self, limit: int) -> bytes:
        """Return the bounded response body used by the client."""

        return self._body[:limit]


def _settings(api_key: str | None = "a" * 32) -> Any:
    """Build one valid client-settings object for isolated tests."""

    settings_module = import_project_module("app.services.api_settings")
    return settings_module.ApiClientSettings(
        base_url="http://127.0.0.1:8000",
        api_key=api_key,
        timeout_seconds=2.0,
        maximum_response_bytes=4096,
        maximum_upload_bytes=128,
    )


def test_settings_reject_credentials_inside_the_api_address() -> None:
    """Do not allow passwords or user names inside the backend URL."""

    settings_module = import_project_module("app.services.api_settings")
    with pytest.raises(ValueError, match="must not contain credentials"):
        settings_module.ApiClientSettings(
            base_url="http://user:pass@localhost:8000",
        ).validate()


def test_settings_reject_short_api_keys() -> None:
    """Require the minimum key length before protected requests are sent."""

    settings_module = import_project_module("app.services.api_settings")
    with pytest.raises(ValueError, match="at least"):
        settings_module.ApiClientSettings(
            base_url="http://localhost:8000",
            api_key="short",
        ).validate()


def test_client_rejects_traversal_and_query_paths() -> None:
    """Keep route construction limited to fixed application paths."""

    client_module = import_project_module("app.services.api_client")
    client = client_module.FinancialNewsApiClient(_settings())
    with pytest.raises(ValueError):
        client._build_url("/v1/../secret")
    with pytest.raises(ValueError):
        client._build_url("/health?token=value")


def test_public_request_does_not_send_the_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep credentials away from public health and readiness routes."""

    client_module = import_project_module("app.services.api_client")
    captured: list[Any] = []

    def fake_urlopen(request: Any, timeout: float) -> FakeResponse:
        """Capture one request and return safe JSON."""

        captured.append(request)
        assert timeout == 2.0
        return FakeResponse({"status": "PASSED"})

    monkeypatch.setattr(client_module, "urlopen", fake_urlopen)
    client = client_module.FinancialNewsApiClient(_settings())
    client._request("GET", "/health")
    assert "X-api-key" not in captured[0].headers


def test_protected_request_sends_the_key_only_in_a_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Require protected requests to use the approved header boundary."""

    client_module = import_project_module("app.services.api_client")
    captured: list[Any] = []

    def fake_urlopen(request: Any, timeout: float) -> FakeResponse:
        """Capture one protected request and return safe JSON."""

        captured.append(request)
        return FakeResponse({"status": "PASSED"})

    monkeypatch.setattr(client_module, "urlopen", fake_urlopen)
    client = client_module.FinancialNewsApiClient(_settings())
    client._request("POST", "/v1/sentiment/text", json_payload={"text": "news"}, protected=True)
    request = captured[0]
    assert request.headers["X-api-key"] == "a" * 32
    assert "a" * 32 not in request.full_url


def test_missing_key_fails_before_a_protected_network_call() -> None:
    """Fail locally rather than sending an unauthenticated analysis request."""

    client_module = import_project_module("app.services.api_client")
    client = client_module.FinancialNewsApiClient(_settings(api_key=None))
    with pytest.raises(client_module.StreamlitApiError) as captured:
        client.sentiment_text("Valid financial news")
    assert captured.value.problem.error_code == "streamlit_api_key_missing"


def test_file_upload_rejects_paths_and_unsupported_types() -> None:
    """Accept only a base filename with an approved extension."""

    client_module = import_project_module("app.services.api_client")
    client = client_module.FinancialNewsApiClient(_settings())
    with pytest.raises(ValueError, match="folder path"):
        client.sentiment_file("../secret.txt", b"content")
    with pytest.raises(ValueError, match="Only TXT"):
        client.sentiment_file("payload.exe", b"content")


def test_reserved_headers_cannot_be_overridden(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent callers from replacing authentication and request headers."""

    client_module = import_project_module("app.services.api_client")
    client = client_module.FinancialNewsApiClient(_settings())
    with pytest.raises(ValueError, match="controlled"):
        client._request("GET", "/health", extra_headers={"X-API-Key": "override"})


def test_application_source_has_no_direct_model_runtime_imports() -> None:
    """Keep PyTorch, Transformers, scikit-learn, and joblib outside Streamlit."""

    forbidden = (
        "import torch",
        "from torch",
        "import transformers",
        "from transformers",
        "import sklearn",
        "from sklearn",
        "import joblib",
    )
    for path in python_files():
        lowered = path.read_text(encoding="utf-8").lower()
        assert not any(fragment in lowered for fragment in forbidden), path


def test_application_source_has_no_hard_coded_local_user_path() -> None:
    """Keep developer machine paths out of public source and messages."""

    for path in python_files():
        text = path.read_text(encoding="utf-8")
        assert "/Users/" not in text
        assert "\\Users\\" not in text
