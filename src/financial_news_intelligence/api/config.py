"""Environment-backed FastAPI settings with fail-closed validation.

Purpose
-------
Keep every request limit, timeout, security rule, and project path in one
explicit object. Named settings avoid unexplained numbers inside route code.

Inputs and downstream use
-------------------------
Values come from environment variables or direct test construction. The app,
file extraction, rate limiter, model workers, and startup scripts all consume
the same validated settings object.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ABSOLUTE_MAX_TEXT_CHARACTERS = 20_000
DEFAULT_MAX_TEXT_CHARACTERS = ABSOLUTE_MAX_TEXT_CHARACTERS
DEFAULT_MAX_UPLOAD_BYTES = 5 * 1024 * 1024
DEFAULT_MAX_CSV_ROWS = 50
DEFAULT_RATE_LIMIT_REQUESTS = 30
DEFAULT_RATE_LIMIT_WINDOW_SECONDS = 60
DEFAULT_SENTIMENT_WORKER_TIMEOUT_SECONDS = 120
DEFAULT_MOVEMENT_WORKER_TIMEOUT_SECONDS = 180
DEFAULT_URL_TIMEOUT_SECONDS = 20
DEFAULT_URL_MAX_REDIRECTS = 3
DEFAULT_HISTORICAL_MATCH_LIMIT = 5
DEFAULT_DEEP_READINESS_PROBE = False
MINIMUM_API_KEY_CHARACTERS = 24


def _environment_boolean(name: str, default: bool) -> bool:
    """Parse one strict true-or-false environment value."""

    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"{name} must be one of true, false, 1, 0, yes, no, on, or off."
    )


def _positive_integer(name: str, default: int) -> int:
    """Parse one positive integer from the environment."""

    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if value < 1:
        raise ValueError(f"{name} must be greater than zero.")
    return value


@dataclass(frozen=True)
class ApiSettings:
    """Validated runtime settings for one FastAPI process."""

    project_root: Path
    environment: str = "development"
    api_key: str | None = None
    require_api_key: bool = True
    trusted_hosts: tuple[str, ...] = ("127.0.0.1", "localhost", "testserver")
    max_text_characters: int = DEFAULT_MAX_TEXT_CHARACTERS
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES
    max_csv_rows: int = DEFAULT_MAX_CSV_ROWS
    rate_limit_requests: int = DEFAULT_RATE_LIMIT_REQUESTS
    rate_limit_window_seconds: int = DEFAULT_RATE_LIMIT_WINDOW_SECONDS
    sentiment_worker_timeout_seconds: int = (
        DEFAULT_SENTIMENT_WORKER_TIMEOUT_SECONDS
    )
    movement_worker_timeout_seconds: int = (
        DEFAULT_MOVEMENT_WORKER_TIMEOUT_SECONDS
    )
    url_timeout_seconds: int = DEFAULT_URL_TIMEOUT_SECONDS
    url_max_redirects: int = DEFAULT_URL_MAX_REDIRECTS
    historical_match_limit: int = DEFAULT_HISTORICAL_MATCH_LIMIT
    deep_readiness_probe: bool = DEFAULT_DEEP_READINESS_PROBE

    def __post_init__(self) -> None:
        """Reject unsafe or internally inconsistent configuration."""

        resolved_root = self.project_root.expanduser().resolve()
        object.__setattr__(self, "project_root", resolved_root)

        if not resolved_root.exists() or not resolved_root.is_dir():
            raise ValueError(
                f"Project root does not exist or is not a directory: {resolved_root}"
            )
        normalized_environment = self.environment.strip().lower()
        if normalized_environment not in {"development", "test", "production"}:
            raise ValueError(
                "environment must be development, test, or production."
            )
        object.__setattr__(self, "environment", normalized_environment)

        if normalized_environment == "production" and not self.require_api_key:
            raise ValueError(
                "API-key authentication cannot be disabled in production."
            )
        if normalized_environment == "production" and "*" in self.trusted_hosts:
            raise ValueError(
                "Wildcard trusted hosts are not allowed in production."
            )

        if self.require_api_key:
            if not self.api_key:
                raise ValueError(
                    "FNI_API_KEY is required when API-key authentication is enabled."
                )
            if len(self.api_key) < MINIMUM_API_KEY_CHARACTERS:
                raise ValueError(
                    "FNI_API_KEY must contain at least "
                    f"{MINIMUM_API_KEY_CHARACTERS} characters."
                )
        if not self.trusted_hosts:
            raise ValueError("At least one trusted host is required.")
        integer_fields = {
            "max_text_characters": self.max_text_characters,
            "max_upload_bytes": self.max_upload_bytes,
            "max_csv_rows": self.max_csv_rows,
            "rate_limit_requests": self.rate_limit_requests,
            "rate_limit_window_seconds": self.rate_limit_window_seconds,
            "sentiment_worker_timeout_seconds": (
                self.sentiment_worker_timeout_seconds
            ),
            "movement_worker_timeout_seconds": (
                self.movement_worker_timeout_seconds
            ),
            "url_timeout_seconds": self.url_timeout_seconds,
            "url_max_redirects": self.url_max_redirects,
            "historical_match_limit": self.historical_match_limit,
        }
        invalid = [name for name, value in integer_fields.items() if value < 1]
        if invalid:
            raise ValueError(
                "Positive integer settings are invalid: " + ", ".join(invalid)
            )
        if self.max_text_characters > ABSOLUTE_MAX_TEXT_CHARACTERS:
            raise ValueError(
                "max_text_characters cannot exceed the hard request-schema limit "
                f"of {ABSOLUTE_MAX_TEXT_CHARACTERS}."
            )

    @classmethod
    def from_environment(cls, project_root: Path | None = None) -> "ApiSettings":
        """Build settings from documented environment variables."""

        configured_root = project_root or Path(
            os.getenv(
                "FNI_PROJECT_ROOT",
                "/Users/ruturajmokashi/Projects/financial-news-stock-intelligence",
            )
        )
        trusted_hosts = tuple(
            host.strip()
            for host in os.getenv(
                "FNI_TRUSTED_HOSTS",
                "127.0.0.1,localhost",
            ).split(",")
            if host.strip()
        )
        return cls(
            project_root=configured_root,
            environment=os.getenv("FNI_API_ENVIRONMENT", "development"),
            api_key=os.getenv("FNI_API_KEY"),
            require_api_key=_environment_boolean(
                "FNI_REQUIRE_API_KEY",
                True,
            ),
            trusted_hosts=trusted_hosts,
            max_text_characters=_positive_integer(
                "FNI_MAX_TEXT_CHARACTERS",
                DEFAULT_MAX_TEXT_CHARACTERS,
            ),
            max_upload_bytes=_positive_integer(
                "FNI_MAX_UPLOAD_BYTES",
                DEFAULT_MAX_UPLOAD_BYTES,
            ),
            max_csv_rows=_positive_integer(
                "FNI_MAX_CSV_ROWS",
                DEFAULT_MAX_CSV_ROWS,
            ),
            rate_limit_requests=_positive_integer(
                "FNI_RATE_LIMIT_REQUESTS",
                DEFAULT_RATE_LIMIT_REQUESTS,
            ),
            rate_limit_window_seconds=_positive_integer(
                "FNI_RATE_LIMIT_WINDOW_SECONDS",
                DEFAULT_RATE_LIMIT_WINDOW_SECONDS,
            ),
            sentiment_worker_timeout_seconds=_positive_integer(
                "FNI_SENTIMENT_WORKER_TIMEOUT_SECONDS",
                DEFAULT_SENTIMENT_WORKER_TIMEOUT_SECONDS,
            ),
            movement_worker_timeout_seconds=_positive_integer(
                "FNI_MOVEMENT_WORKER_TIMEOUT_SECONDS",
                DEFAULT_MOVEMENT_WORKER_TIMEOUT_SECONDS,
            ),
            url_timeout_seconds=_positive_integer(
                "FNI_URL_TIMEOUT_SECONDS",
                DEFAULT_URL_TIMEOUT_SECONDS,
            ),
            url_max_redirects=_positive_integer(
                "FNI_URL_MAX_REDIRECTS",
                DEFAULT_URL_MAX_REDIRECTS,
            ),
            historical_match_limit=_positive_integer(
                "FNI_HISTORICAL_MATCH_LIMIT",
                DEFAULT_HISTORICAL_MATCH_LIMIT,
            ),
            deep_readiness_probe=_environment_boolean(
                "FNI_DEEP_READINESS_PROBE",
                DEFAULT_DEEP_READINESS_PROBE,
            ),
        )
