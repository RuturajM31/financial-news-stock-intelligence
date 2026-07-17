"""API-key authentication and request identity helpers."""

from __future__ import annotations

import hmac
import re
import uuid
from typing import Callable

from fastapi import Header, Request

from .config import ApiSettings
from .errors import ApiProblem


REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{8,64}$")


def request_id_from_header(value: str | None) -> str:
    """Use a safe caller request ID or create a random UUID."""

    if value and REQUEST_ID_PATTERN.fullmatch(value):
        return value
    return str(uuid.uuid4())


def api_key_dependency(settings: ApiSettings) -> Callable[..., None]:
    """Create the FastAPI dependency that enforces the configured API key."""

    def require_api_key(
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    ) -> None:
        if not settings.require_api_key:
            return
        # Constant-time comparison avoids revealing matching key prefixes.
        if not x_api_key or not hmac.compare_digest(
            x_api_key,
            settings.api_key or "",
        ):
            raise ApiProblem(
                status_code=401,
                error_code="authentication_failed",
                what_failed="API authentication failed.",
                where_failed="X-API-Key request header",
                why_failed="The API key is missing or does not match.",
                safe_next_step=(
                    "Send the configured FNI_API_KEY value in the X-API-Key "
                    "header. Do not place the key in the URL."
                ),
            )

    return require_api_key


def client_identity(request: Request) -> str:
    """Return one stable limiter key without trusting forwarding headers."""

    if request.client is None or not request.client.host:
        return "unknown-client"
    return request.client.host
