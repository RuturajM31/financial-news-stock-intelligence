"""Build safe Streamlit settings for the verified FastAPI backend.

Purpose
-------
Keep the API address, time limits, response limits, upload limits, and optional
API key in one validated object. The browser layer must not spread these values
through page code or expose them in visible messages.

Inputs and downstream use
-------------------------
Values may come from environment variables or Streamlit secrets. The API client
uses the returned settings for every request. No model path, provider cache, or
private market-data value is accepted here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping
from urllib.parse import urlsplit, urlunsplit


DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_TIMEOUT_SECONDS = 8.0
DEFAULT_MAXIMUM_RESPONSE_BYTES = 5 * 1024 * 1024
DEFAULT_MAXIMUM_UPLOAD_BYTES = 5 * 1024 * 1024
MINIMUM_API_KEY_CHARACTERS = 24


def _read_secret_value(
    secrets: Mapping[str, Any] | None,
    key: str,
) -> str | None:
    """Read one optional text value from supported Streamlit secret layouts.

    Supported layouts are a top-level key or a nested ``fastapi`` section. A
    non-text value is rejected because silently converting it could hide a bad
    deployment setting.
    """

    if secrets is None:
        return None

    nested = secrets.get("fastapi")
    candidates: list[Any] = []
    if isinstance(nested, Mapping):
        candidates.append(nested.get(key))
    candidates.append(secrets.get(key))

    for value in candidates:
        if value is None:
            continue
        if not isinstance(value, str):
            raise ValueError(f"Streamlit secret {key!r} must be text.")
        stripped = value.strip()
        return stripped or None
    return None


def _read_positive_float(
    environment: Mapping[str, str],
    name: str,
    default: float,
) -> float:
    """Read one positive floating-point setting from the environment."""

    raw_value = environment.get(name)
    if raw_value is None:
        return default
    try:
        value = float(raw_value)
    except ValueError as error:
        raise ValueError(f"{name} must be a number.") from error
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero.")
    return value


def _read_positive_integer(
    environment: Mapping[str, str],
    name: str,
    default: int,
) -> int:
    """Read one positive whole-number setting from the environment."""

    raw_value = environment.get(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError as error:
        raise ValueError(f"{name} must be a whole number.") from error
    if value < 1:
        raise ValueError(f"{name} must be greater than zero.")
    return value


@dataclass(frozen=True)
class ApiClientSettings:
    """Store validated connection settings for the Streamlit API client.

    Attributes:
        base_url: HTTP or HTTPS address of the verified FastAPI service.
        api_key: Optional key used only for protected routes.
        timeout_seconds: Maximum duration for one API request.
        maximum_response_bytes: Largest JSON response accepted by the UI.
        maximum_upload_bytes: Largest file body sent by the UI.
    """

    base_url: str
    api_key: str | None = None
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    maximum_response_bytes: int = DEFAULT_MAXIMUM_RESPONSE_BYTES
    maximum_upload_bytes: int = DEFAULT_MAXIMUM_UPLOAD_BYTES

    def validate(self) -> None:
        """Reject unsafe, incomplete, or internally inconsistent settings."""

        parsed = urlsplit(self.base_url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("The FastAPI address must use HTTP or HTTPS.")
        if not parsed.hostname:
            raise ValueError("The FastAPI address must include a host name.")
        if parsed.username or parsed.password:
            raise ValueError("The FastAPI address must not contain credentials.")
        if parsed.query or parsed.fragment:
            raise ValueError("The FastAPI address must not contain a query or fragment.")
        if self.timeout_seconds <= 0:
            raise ValueError("The FastAPI timeout must be greater than zero.")
        if self.maximum_response_bytes < 1:
            raise ValueError("The maximum API response size must be positive.")
        if self.maximum_upload_bytes < 1:
            raise ValueError("The maximum upload size must be positive.")
        if self.api_key and len(self.api_key) < MINIMUM_API_KEY_CHARACTERS:
            raise ValueError(
                "The FastAPI key must contain at least "
                f"{MINIMUM_API_KEY_CHARACTERS} characters."
            )

    @property
    def normalized_base_url(self) -> str:
        """Return a stable address without a trailing slash or credentials."""

        parsed = urlsplit(self.base_url)
        path = parsed.path.rstrip("/")
        return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def get_api_client_settings(
    secrets: Mapping[str, Any] | None = None,
    environment: Mapping[str, str] | None = None,
) -> ApiClientSettings:
    """Build validated settings from environment values and private secrets.

    Environment values take priority so controlled runtime deployments can
    override local secret files. The API key remains optional because health
    and readiness routes are public; protected analysis methods fail locally
    when the key is missing.
    """

    active_environment = environment if environment is not None else os.environ
    base_url = (
        active_environment.get("FNI_STREAMLIT_API_BASE_URL")
        or _read_secret_value(secrets, "api_base_url")
        or DEFAULT_API_BASE_URL
    )
    api_key = (
        active_environment.get("FNI_API_KEY")
        or _read_secret_value(secrets, "api_key")
    )
    settings = ApiClientSettings(
        base_url=base_url.strip(),
        api_key=api_key.strip() if api_key else None,
        timeout_seconds=_read_positive_float(
            active_environment,
            "FNI_STREAMLIT_API_TIMEOUT_SECONDS",
            DEFAULT_TIMEOUT_SECONDS,
        ),
        maximum_response_bytes=_read_positive_integer(
            active_environment,
            "FNI_STREAMLIT_MAX_RESPONSE_BYTES",
            DEFAULT_MAXIMUM_RESPONSE_BYTES,
        ),
        maximum_upload_bytes=_read_positive_integer(
            active_environment,
            "FNI_STREAMLIT_MAX_UPLOAD_BYTES",
            DEFAULT_MAXIMUM_UPLOAD_BYTES,
        ),
    )
    settings.validate()
    return settings
