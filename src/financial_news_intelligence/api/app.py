"""FastAPI application factory and route definitions."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, AsyncIterator

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse

from .config import ApiSettings
from .errors import ApiProblem, problem_response
from .extraction import clean_text, extract_public_url, extract_uploaded_bytes
from .logging_config import (
    bind_request_context,
    configure_logging,
    reset_request_context,
)
from .monitoring import get_metrics_registry, route_template
from .rate_limit import FixedWindowRateLimiter
from .schemas import (
    BatchSentimentResponse,
    ExplainabilityRequest,
    ExplainabilityResponse,
    HealthResponse,
    HistoricalIntelligenceRequest,
    HistoricalIntelligenceResponse,
    MovementPredictionRequest,
    MovementPredictionResponse,
    ProvenanceResponse,
    ReadinessResponse,
    ScenarioAnalysisRequest,
    ScenarioAnalysisResponse,
    SentimentPredictionResponse,
    TextSentimentRequest,
    UrlSentimentRequest,
)
from .upload_request import read_upload_request
from .security import (
    api_key_dependency,
    client_identity,
    request_id_from_header,
)
if TYPE_CHECKING:
    from .services import ApiServices


APPLICATION_VERSION = "1.1.0"
SERVICE_NAME = "financial-news-stock-intelligence-api"
FILE_UPLOAD_REQUEST_BODY = {
    "required": True,
    "content": {
        "multipart/form-data": {
            "schema": {
                "type": "object",
                "required": ["file"],
                "properties": {
                    "file": {
                        "type": "string",
                        "format": "binary",
                        "description": "One TXT, PDF, DOCX, or CSV file.",
                    },
                    "csv_text_column": {
                        "type": "string",
                        "description": (
                            "Required only for CSV input. This is the exact "
                            "column containing article text."
                        ),
                    },
                },
                "additionalProperties": False,
            }
        },
        "application/octet-stream": {
            "schema": {
                "type": "string",
                "format": "binary",
                "description": (
                    "Raw file bytes. Send the filename in X-Filename and, "
                    "for CSV, the column name in X-CSV-Text-Column."
                ),
            }
        },
    },
}


def create_app(
    settings: ApiSettings | None = None,
    services: "ApiServices | Any | None" = None,
    rate_limiter: FixedWindowRateLimiter | None = None,
) -> FastAPI:
    """Create one configured application without module-import side effects."""

    active_settings = settings or ApiSettings.from_environment()
    if services is None:
        # Import service orchestration only when the real application needs it.
        # API schema tests can inject fake services without loading numerical
        # libraries into the FastAPI process.
        from .services import ApiServices

        active_services: Any = ApiServices(active_settings)
    else:
        active_services = services
    limiter = rate_limiter or FixedWindowRateLimiter(
        active_settings.rate_limit_requests,
        active_settings.rate_limit_window_seconds,
    )
    logger = configure_logging()
    registry = get_metrics_registry()
    registry.configure(service=SERVICE_NAME, version=APPLICATION_VERSION)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        registry.set_ready(True)
        logger.info(
            "FastAPI application lifecycle started.",
            extra={"marker": "STARTED", "component": "fastapi"},
        )
        try:
            yield
        finally:
            registry.set_ready(False)
            active_services.close()
            logger.info(
                "FastAPI application lifecycle stopped.",
                extra={"marker": "PASSED", "component": "fastapi"},
            )

    app = FastAPI(
        title="Financial News Stock Intelligence API",
        description=(
            "Verified sentiment, historical movement, explainability, and "
            "research-scenario endpoints."
        ),
        version=APPLICATION_VERSION,
        lifespan=lifespan,
    )
    app.state.settings = active_settings
    app.state.services = active_services
    app.state.rate_limiter = limiter
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=list(active_settings.trusted_hosts),
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next: Any) -> Any:
        request_id = request_id_from_header(request.headers.get("X-Request-ID"))
        request.state.request_id = request_id
        context_token = bind_request_context(request_id)
        registry.request_started()
        started = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["X-Request-ID"] = request_id
            return response
        except Exception:
            duration_seconds = max(0.0, time.perf_counter() - started)
            logger.exception(
                "API request raised an unhandled exception.",
                extra={
                    "marker": "FAILED",
                    "request_id": request_id,
                    "method": request.method,
                    "path": route_template(request.scope),
                    "duration_ms": round(duration_seconds * 1000, 3),
                    "client_ip": client_identity(request),
                },
            )
            raise
        finally:
            duration_seconds = max(0.0, time.perf_counter() - started)
            safe_path = route_template(request.scope)
            registry.observe_http(
                method=request.method,
                path=safe_path,
                status_code=status_code,
                duration_seconds=duration_seconds,
            )
            registry.request_finished()
            marker = "PASSED" if status_code < 400 else "FAILED"
            logger.info(
                "API request completed.",
                extra={
                    "marker": marker,
                    "request_id": request_id,
                    "method": request.method,
                    "path": safe_path,
                    "status_code": status_code,
                    "duration_ms": round(duration_seconds * 1000, 3),
                    "client_ip": client_identity(request),
                },
            )
            reset_request_context(context_token)

    @app.exception_handler(ApiProblem)
    async def api_problem_handler(
        request: Request,
        problem: ApiProblem,
    ) -> JSONResponse:
        return problem_response(problem, getattr(request.state, "request_id", None))

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request,
        error: RequestValidationError,
    ) -> JSONResponse:
        safe_errors = [
            {
                "location": [str(value) for value in item.get("loc", ())],
                "message": item.get("msg", "Invalid value."),
                "type": item.get("type", "validation_error"),
            }
            for item in error.errors()
        ]
        problem = ApiProblem(
            422,
            "request_validation_failed",
            "Request validation failed.",
            "FastAPI request schema",
            f"One or more request fields are invalid: {safe_errors}",
            "Correct the named fields and submit the request again.",
        )
        return problem_response(problem, getattr(request.state, "request_id", None))

    @app.exception_handler(Exception)
    async def unexpected_error_handler(
        request: Request,
        error: Exception,
    ) -> JSONResponse:
        problem = ApiProblem(
            500,
            "internal_error",
            "The API request failed.",
            "FastAPI application",
            f"An unexpected {type(error).__name__} occurred.",
            (
                "Use the request ID to inspect server logs; do not repeat the "
                "request rapidly."
            ),
        )
        return problem_response(problem, getattr(request.state, "request_id", None))

    require_api_key = api_key_dependency(active_settings)

    def enforce_rate_limit(request: Request) -> None:
        key = f"{client_identity(request)}:{request.url.path}"
        limiter.check(key)

    protected = [Depends(require_api_key), Depends(enforce_rate_limit)]

    @app.get(
        "/metrics",
        dependencies=[Depends(require_api_key)],
        include_in_schema=False,
        tags=["service"],
    )
    def metrics() -> PlainTextResponse:
        """Expose aggregate Prometheus metrics without request content."""

        return PlainTextResponse(
            registry.render(),
            media_type="text/plain; version=0.0.4",
        )

    @app.get("/health", response_model=HealthResponse, tags=["service"])
    def health() -> HealthResponse:
        return HealthResponse(
            status="PASSED",
            service=SERVICE_NAME,
            version=APPLICATION_VERSION,
        )

    @app.get("/ready", response_model=ReadinessResponse, tags=["service"])
    def ready() -> ReadinessResponse | JSONResponse:
        try:
            result = active_services.readiness(
                run_deep_probe=active_settings.deep_readiness_probe
            )
        except ApiProblem as problem:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "FAILED",
                    "components": {"readiness": "FAILED"},
                    "details": {
                        "error_code": problem.error_code,
                        "what_failed": problem.what_failed,
                        "where_failed": problem.where_failed,
                        "why_failed": problem.why_failed,
                        "safe_next_step": problem.safe_next_step,
                    },
                },
            )
        return ReadinessResponse(status="PASSED", **result)

    @app.post(
        "/v1/sentiment/text",
        response_model=SentimentPredictionResponse,
        dependencies=protected,
        tags=["sentiment"],
    )
    def sentiment_text(payload: TextSentimentRequest) -> SentimentPredictionResponse:
        text = clean_text(payload.text, active_settings.max_text_characters)
        return active_services.predict_sentiment([text], "text")[0]

    @app.post(
        "/v1/sentiment/url",
        response_model=SentimentPredictionResponse,
        dependencies=protected,
        tags=["sentiment"],
    )
    def sentiment_url(payload: UrlSentimentRequest) -> SentimentPredictionResponse:
        text = extract_public_url(
            str(payload.url),
            timeout_seconds=active_settings.url_timeout_seconds,
            maximum_bytes=active_settings.max_upload_bytes,
            maximum_characters=active_settings.max_text_characters,
            maximum_redirects=active_settings.url_max_redirects,
        )
        return active_services.predict_sentiment([text], "url")[0]

    @app.post(
        "/v1/sentiment/file",
        response_model=BatchSentimentResponse,
        dependencies=protected,
        tags=["sentiment"],
        openapi_extra={"requestBody": FILE_UPLOAD_REQUEST_BODY},
    )
    async def sentiment_file(request: Request) -> BatchSentimentResponse:
        upload = await read_upload_request(
            request,
            maximum_file_bytes=active_settings.max_upload_bytes,
        )
        texts = extract_uploaded_bytes(
            filename=upload.filename,
            content=upload.content,
            maximum_bytes=active_settings.max_upload_bytes,
            maximum_characters=active_settings.max_text_characters,
            csv_text_column=upload.csv_text_column,
            maximum_csv_rows=active_settings.max_csv_rows,
        )
        source_type = upload.filename.rsplit(".", 1)[-1].lower()
        results = active_services.predict_sentiment(texts, source_type)
        return BatchSentimentResponse(
            source_type=source_type,
            result_count=len(results),
            results=results,
        )

    @app.post(
        "/v1/movement/predict",
        response_model=MovementPredictionResponse,
        dependencies=protected,
        tags=["movement"],
    )
    def movement_predict(
        payload: MovementPredictionRequest,
    ) -> MovementPredictionResponse:
        text = clean_text(payload.text, active_settings.max_text_characters)
        return active_services.predict_movement(
            text,
            payload.ticker,
            payload.published_at,
        )

    @app.post(
        "/v1/intelligence/historical",
        response_model=HistoricalIntelligenceResponse,
        dependencies=protected,
        tags=["intelligence"],
    )
    def historical_intelligence(
        payload: HistoricalIntelligenceRequest,
    ) -> HistoricalIntelligenceResponse:
        if payload.limit > active_settings.historical_match_limit:
            raise ApiProblem(
                422,
                "historical_limit_exceeded",
                "Historical intelligence request failed.",
                "Historical match limit",
                f"The request asked for {payload.limit} matches; the configured "
                f"limit is {active_settings.historical_match_limit}.",
                "Lower the limit and submit the request again.",
            )
        text = clean_text(payload.text, active_settings.max_text_characters)
        return active_services.historical(
            text,
            payload.ticker,
            payload.published_at,
            payload.limit,
            payload.minimum_similarity,
        )

    @app.post(
        "/v1/explainability",
        response_model=ExplainabilityResponse,
        dependencies=protected,
        tags=["intelligence"],
    )
    def explainability(payload: ExplainabilityRequest) -> ExplainabilityResponse:
        text = clean_text(payload.text, active_settings.max_text_characters)
        return active_services.explain(
            text,
            payload.ticker,
            payload.published_at,
            payload.top_n,
        )

    @app.post(
        "/v1/scenarios/analyze",
        response_model=ScenarioAnalysisResponse,
        dependencies=protected,
        tags=["intelligence"],
    )
    def scenario_analysis(
        payload: ScenarioAnalysisRequest,
    ) -> ScenarioAnalysisResponse:
        clean_text(payload.text, active_settings.max_text_characters)
        return active_services.scenario(payload)

    @app.get(
        "/v1/provenance",
        response_model=ProvenanceResponse,
        dependencies=protected,
        tags=["intelligence"],
    )
    def provenance() -> ProvenanceResponse:
        return ProvenanceResponse(provenance=active_services.provenance())

    return app
