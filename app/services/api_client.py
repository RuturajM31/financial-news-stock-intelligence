"""Call the verified FastAPI service without loading model libraries.

Purpose
-------
Keep all HTTP work outside Streamlit page code. This client sends bounded JSON
or file requests, adds safe request identifiers, validates response sizes, and
converts network or API failures into the same plain four-part error format used
throughout the product.

Security and runtime boundary
-----------------------------
The module uses only Python's standard library. It never imports model packages,
reads provider caches, logs the API key, or places credentials in a URL.
"""

from __future__ import annotations

import json
import socket
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.services.api_contracts import (
    ApiProblemDetails,
    HealthStatus,
    ReadinessStatus,
    parse_health_response,
    parse_problem_payload,
    parse_readiness_response,
    validate_success_payload,
)
from app.services.api_settings import ApiClientSettings


_ALLOWED_UPLOAD_SUFFIXES = frozenset({".txt", ".pdf", ".docx", ".csv"})
_JSON_CONTENT_TYPE = "application/json"
_BINARY_CONTENT_TYPE = "application/octet-stream"


@dataclass(frozen=True)
class ApiResponse:
    """Store one decoded response together with its safe request identifier."""

    payload: dict[str, Any]
    request_id: str | None


class StreamlitApiError(RuntimeError):
    """Carry one safe client-facing API failure without a raw traceback."""

    def __init__(self, problem: ApiProblemDetails) -> None:
        """Store one already-sanitized API problem for later rendering."""

        super().__init__(problem.why_failed)
        self.problem = problem


class FinancialNewsApiClient:
    """Provide typed methods for every verified FastAPI route.

    The client is intentionally stateless. Streamlit session state stores only
    small, licence-safe result summaries and connection checks.
    """

    def __init__(self, settings: ApiClientSettings) -> None:
        """Validate and retain one immutable API configuration."""

        settings.validate()
        self.settings = settings

    def _build_url(self, path: str) -> str:
        """Join one fixed API path to the configured base address safely."""

        if not path.startswith("/"):
            raise ValueError("API paths must start with '/'.")
        if ".." in path or "?" in path or "#" in path:
            raise ValueError("API paths must not contain traversal or query text.")
        return f"{self.settings.normalized_base_url}{path}"

    def _require_api_key(self) -> str:
        """Return the configured key or fail before a protected request."""

        if not self.settings.api_key:
            raise StreamlitApiError(
                ApiProblemDetails(
                    error_code="streamlit_api_key_missing",
                    what_failed="The protected API request was not sent.",
                    where_failed="Streamlit private API settings",
                    why_failed="The FastAPI key is not configured for Streamlit.",
                    safe_next_step=(
                        "Add the approved API key to Streamlit secrets or the "
                        "FNI_API_KEY environment variable, then try again."
                    ),
                )
            )
        return self.settings.api_key

    def _decode_json_response(
        self,
        response: Any,
        request_id: str,
    ) -> ApiResponse:
        """Read one bounded UTF-8 JSON object and capture its response ID."""

        raw_body = response.read(self.settings.maximum_response_bytes + 1)
        if len(raw_body) > self.settings.maximum_response_bytes:
            raise StreamlitApiError(
                ApiProblemDetails(
                    error_code="api_response_too_large",
                    what_failed="The API response could not be accepted.",
                    where_failed="Streamlit API response reader",
                    why_failed="The response exceeded the approved size limit.",
                    safe_next_step=(
                        "Reduce the requested result size or inspect the FastAPI "
                        "response contract before trying again."
                    ),
                    request_id=request_id,
                )
            )
        try:
            decoded = raw_body.decode("utf-8")
            payload = json.loads(decoded)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise StreamlitApiError(
                ApiProblemDetails(
                    error_code="invalid_api_json",
                    what_failed="The API response could not be read.",
                    where_failed="Streamlit JSON response parser",
                    why_failed="FastAPI returned invalid UTF-8 JSON.",
                    safe_next_step=(
                        "Check the FastAPI service logs using the request ID, "
                        "then correct the response before retrying."
                    ),
                    request_id=request_id,
                )
            ) from error
        if not isinstance(payload, dict):
            raise StreamlitApiError(
                ApiProblemDetails(
                    error_code="invalid_api_shape",
                    what_failed="The API response could not be accepted.",
                    where_failed="Streamlit API response validation",
                    why_failed="FastAPI returned JSON that was not an object.",
                    safe_next_step=(
                        "Restore the verified FastAPI response contract and retry."
                    ),
                    request_id=request_id,
                )
            )
        response_id = response.headers.get("X-Request-ID") or request_id
        return ApiResponse(payload=payload, request_id=response_id)

    def _read_http_error(
        self,
        error: HTTPError,
        request_id: str,
    ) -> StreamlitApiError:
        """Convert one HTTP error into the approved safe error structure."""

        response_id = error.headers.get("X-Request-ID") or request_id
        raw_body = error.read(self.settings.maximum_response_bytes + 1)
        payload: Mapping[str, Any] | None = None
        if len(raw_body) <= self.settings.maximum_response_bytes:
            try:
                decoded = json.loads(raw_body.decode("utf-8"))
                if isinstance(decoded, Mapping):
                    payload = decoded
            except (UnicodeDecodeError, json.JSONDecodeError):
                payload = None
        problem = parse_problem_payload(
            payload=payload,
            fallback_code=f"http_{error.code}",
            fallback_location="FastAPI HTTP response",
            fallback_reason=f"FastAPI returned HTTP status {error.code}.",
            fallback_next_step=(
                "Check the service status and request settings, then try again."
            ),
            request_id=response_id,
        )
        return StreamlitApiError(problem)

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_payload: Mapping[str, Any] | None = None,
        binary_payload: bytes | None = None,
        extra_headers: Mapping[str, str] | None = None,
        protected: bool = False,
    ) -> ApiResponse:
        """Send one bounded request and return a validated JSON object.

        Exactly one body type may be supplied. The API key is sent only in the
        header of protected routes and is never included in an exception string.
        """

        if json_payload is not None and binary_payload is not None:
            raise ValueError("Only one request body type may be supplied.")

        request_id = str(uuid.uuid4())
        headers = {
            "Accept": _JSON_CONTENT_TYPE,
            "X-Request-ID": request_id,
        }
        body: bytes | None = None
        if json_payload is not None:
            body = json.dumps(
                json_payload,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
            headers["Content-Type"] = _JSON_CONTENT_TYPE
        elif binary_payload is not None:
            body = binary_payload
            headers["Content-Type"] = _BINARY_CONTENT_TYPE

        if protected:
            headers["X-API-Key"] = self._require_api_key()
        if extra_headers:
            reserved_headers = {"x-api-key", "x-request-id", "content-type"}
            for name, value in extra_headers.items():
                if name.lower() in reserved_headers:
                    raise ValueError(f"Header {name!r} is controlled by the API client.")
                if (
                    "\r" in name
                    or "\n" in name
                    or "\r" in value
                    or "\n" in value
                ):
                    raise ValueError("API headers must not contain line breaks.")
                headers[name] = value

        request = Request(
            self._build_url(path),
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with urlopen(request, timeout=self.settings.timeout_seconds) as response:
                return self._decode_json_response(response, request_id)
        except HTTPError as error:
            raise self._read_http_error(error, request_id) from error
        except (TimeoutError, socket.timeout) as error:
            raise StreamlitApiError(
                ApiProblemDetails(
                    error_code="api_timeout",
                    what_failed="The API request timed out.",
                    where_failed="Streamlit to FastAPI connection",
                    why_failed=(
                        "FastAPI did not answer within the approved time limit."
                    ),
                    safe_next_step=(
                        "Check whether FastAPI and its workers are running, then "
                        "try the request once more."
                    ),
                    request_id=request_id,
                )
            ) from error
        except URLError as error:
            if isinstance(error.reason, (TimeoutError, socket.timeout)):
                problem = ApiProblemDetails(
                    error_code="api_timeout",
                    what_failed="The API request timed out.",
                    where_failed="Streamlit to FastAPI connection",
                    why_failed=(
                        "FastAPI did not answer within the approved time limit."
                    ),
                    safe_next_step=(
                        "Check whether FastAPI and its workers are running, then "
                        "try the request once more."
                    ),
                    request_id=request_id,
                )
            else:
                problem = ApiProblemDetails(
                    error_code="api_unreachable",
                    what_failed="The API connection failed.",
                    where_failed="Streamlit to FastAPI connection",
                    why_failed=(
                        "The configured FastAPI service could not be reached."
                    ),
                    safe_next_step=(
                        "Start the verified FastAPI service or correct its private "
                        "address, then check the connection again."
                    ),
                    request_id=request_id,
                )
            raise StreamlitApiError(problem) from error

    def health(self) -> HealthStatus:
        """Read the public process-health route without an API key."""

        response = self._request("GET", "/health")
        try:
            return parse_health_response(response.payload, response.request_id)
        except ValueError as error:
            raise StreamlitApiError(
                ApiProblemDetails(
                    error_code="health_contract_failed",
                    what_failed="The FastAPI health result could not be accepted.",
                    where_failed="Streamlit health response validation",
                    why_failed=str(error),
                    safe_next_step=(
                        "Restore the verified /health response fields and retry."
                    ),
                    request_id=response.request_id,
                )
            ) from error

    def readiness(self) -> ReadinessStatus:
        """Read the public lightweight readiness route without model loading."""

        response = self._request("GET", "/ready")
        try:
            return parse_readiness_response(response.payload, response.request_id)
        except ValueError as error:
            raise StreamlitApiError(
                ApiProblemDetails(
                    error_code="readiness_contract_failed",
                    what_failed="The FastAPI readiness result could not be accepted.",
                    where_failed="Streamlit readiness response validation",
                    why_failed=str(error),
                    safe_next_step=(
                        "Restore the verified /ready response fields and retry."
                    ),
                    request_id=response.request_id,
                )
            ) from error

    def _protected_json(
        self,
        path: str,
        payload: Mapping[str, Any] | None = None,
        method: str = "POST",
    ) -> dict[str, Any]:
        """Send one protected request and require a passed response status."""

        response = self._request(
            method,
            path,
            json_payload=payload,
            protected=True,
        )
        try:
            return validate_success_payload(response.payload, path)
        except ValueError as error:
            raise StreamlitApiError(
                ApiProblemDetails(
                    error_code="analysis_contract_failed",
                    what_failed="The analysis response could not be accepted.",
                    where_failed=f"Streamlit validation for {path}",
                    why_failed=str(error),
                    safe_next_step=(
                        "Restore the verified endpoint response and retry."
                    ),
                    request_id=response.request_id,
                )
            ) from error

    def sentiment_text(self, text: str) -> dict[str, Any]:
        """Request DistilBERT sentiment for one pasted article."""

        return self._protected_json("/v1/sentiment/text", {"text": text})

    def sentiment_url(self, url: str) -> dict[str, Any]:
        """Request sentiment for one public HTTP or HTTPS article URL."""

        return self._protected_json("/v1/sentiment/url", {"url": url})

    def sentiment_file(
        self,
        filename: str,
        content: bytes,
        csv_text_column: str | None = None,
    ) -> dict[str, Any]:
        """Send one approved file using FastAPI's raw-binary upload contract."""

        safe_name = Path(filename).name
        suffix = Path(safe_name).suffix.lower()
        if not safe_name or safe_name != filename:
            raise ValueError("The upload filename must not contain a folder path.")
        if suffix not in _ALLOWED_UPLOAD_SUFFIXES:
            raise ValueError("Only TXT, PDF, DOCX, and CSV files are supported.")
        if not content:
            raise ValueError("The upload file must not be empty.")
        if len(content) > self.settings.maximum_upload_bytes:
            raise ValueError("The upload file exceeds the approved size limit.")
        headers = {"X-Filename": safe_name}
        if csv_text_column:
            headers["X-CSV-Text-Column"] = csv_text_column.strip()
        response = self._request(
            "POST",
            "/v1/sentiment/file",
            binary_payload=content,
            extra_headers=headers,
            protected=True,
        )
        try:
            return validate_success_payload(
                response.payload,
                "/v1/sentiment/file",
            )
        except ValueError as error:
            raise StreamlitApiError(
                ApiProblemDetails(
                    error_code="file_analysis_contract_failed",
                    what_failed="The file-analysis response could not be accepted.",
                    where_failed="Streamlit file response validation",
                    why_failed=str(error),
                    safe_next_step=(
                        "Restore the verified file response contract and retry."
                    ),
                    request_id=response.request_id,
                )
            ) from error

    def movement_prediction(
        self,
        text: str,
        ticker: str,
        published_at: str,
    ) -> dict[str, Any]:
        """Request one Down, Flat, or Up movement prediction."""

        return self._protected_json(
            "/v1/movement/predict",
            {"text": text, "ticker": ticker, "published_at": published_at},
        )

    def historical_intelligence(
        self,
        text: str,
        ticker: str,
        published_at: str,
        limit: int = 5,
        minimum_similarity: float = 0.0,
    ) -> dict[str, Any]:
        """Request strictly earlier historical matches for one event."""

        return self._protected_json(
            "/v1/intelligence/historical",
            {
                "text": text,
                "ticker": ticker,
                "published_at": published_at,
                "limit": limit,
                "minimum_similarity": minimum_similarity,
            },
        )

    def explainability(
        self,
        text: str,
        ticker: str,
        published_at: str,
        top_n: int = 5,
    ) -> dict[str, Any]:
        """Request global and local model-driver evidence."""

        return self._protected_json(
            "/v1/explainability",
            {
                "text": text,
                "ticker": ticker,
                "published_at": published_at,
                "top_n": top_n,
            },
        )

    def scenario_analysis(
        self,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Request a research-only investment scenario calculation."""

        return self._protected_json("/v1/scenarios/analyze", payload)

    def provenance(self) -> dict[str, Any]:
        """Request licence-safe source and model provenance."""

        return self._protected_json("/v1/provenance", method="GET")
