"""API tests for health, authentication, schemas, and safe failures."""

from datetime import date
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from financial_news_intelligence.api.app import create_app
from financial_news_intelligence.api.config import ApiSettings
from financial_news_intelligence.api.schemas import (
    DriverEvidence,
    ExplainabilityResponse,
    HistoricalEventMatch,
    HistoricalIntelligenceResponse,
    MovementPredictionResponse,
    MovementProbabilitiesResponse,
    ScenarioAnalysisResponse,
    ScenarioOutcomeResponse,
    SentimentPredictionResponse,
    SentimentProbabilitiesResponse,
)


class FakeServices:
    """Deterministic services that avoid loading model binaries in API tests."""

    def close(self) -> None:
        return None

    def readiness(self, run_deep_probe: bool = False) -> dict[str, Any]:
        return {
            "components": {
                "artifacts": "PASSED",
                "movement_worker": "PASSED" if run_deep_probe else "CONFIGURED",
                "sentiment_worker": "PASSED" if run_deep_probe else "CONFIGURED",
            },
            "details": {},
        }

    def predict_sentiment(
        self,
        texts: list[str],
        source_type: str,
    ) -> list[SentimentPredictionResponse]:
        return [
            SentimentPredictionResponse(
                label="Bullish",
                confidence=0.7,
                probabilities=SentimentProbabilitiesResponse(
                    bearish=0.1,
                    neutral=0.2,
                    bullish=0.7,
                ),
                source_type=source_type,
            )
            for _ in texts
        ]

    def _movement_response(self) -> MovementPredictionResponse:
        """Return one deterministic movement result for route tests."""

        return MovementPredictionResponse(
            ticker="AAPL",
            target_session_date=date(2019, 1, 3),
            direction="Up",
            confidence=0.60,
            probabilities=MovementProbabilitiesResponse(
                down=0.20,
                flat=0.20,
                up=0.60,
            ),
            sentiment=self.predict_sentiment(["text"], "text")[0],
            champion_model="stability_soft_vote_rf_sgd",
        )

    def predict_movement(
        self,
        text: str,
        ticker: str,
        published_at: Any,
        source_type: str = "text",
    ) -> MovementPredictionResponse:
        """Return one deterministic movement response."""

        return self._movement_response()

    def historical(
        self,
        text: str,
        ticker: str,
        published_at: Any,
        limit: int,
        minimum_similarity: float,
    ) -> HistoricalIntelligenceResponse:
        """Return one deterministic earlier-event match."""

        return HistoricalIntelligenceResponse(
            ticker=ticker,
            query_target_session_date=date(2019, 1, 3),
            matches=[
                HistoricalEventMatch(
                    article_id="article-1",
                    ticker=ticker,
                    target_session_date=date(2018, 12, 20),
                    source_url="https://www.sec.gov/example",
                    sentiment_label="Bullish",
                    movement_label="Up",
                    reaction_return_percent=1.2,
                    similarity_score=0.8,
                )
            ],
            important_phrases=["revenue growth"],
            limitations=["Historical evidence is not a guarantee."],
        )

    def explain(
        self,
        text: str,
        ticker: str,
        published_at: Any,
        top_n: int,
    ) -> ExplainabilityResponse:
        """Return deterministic global and local driver evidence."""

        driver = DriverEvidence(
            rank=1,
            feature="net_sentiment_mean",
            importance=0.2,
            method="test_method",
            interpretation="Sensitivity only.",
        )
        return ExplainabilityResponse(
            prediction=self._movement_response(),
            global_drivers=[driver],
            local_drivers=[
                DriverEvidence(
                    rank=1,
                    feature="event_text",
                    probability_effect=0.1,
                    absolute_effect=0.1,
                    direction="supports_prediction",
                    method="test_method",
                    interpretation="Sensitivity only.",
                )
            ],
        )

    def scenario(self, request: Any) -> ScenarioAnalysisResponse:
        """Return deterministic low, base, and high portfolio outcomes."""

        outcomes = [
            ScenarioOutcomeResponse(
                scenario=name,
                historical_return_percent=value,
                shares_purchased=10.0,
                cash_balance=0.0,
                estimated_tax=0.0,
                net_final_value=1000.0 + value,
                gain_loss=value,
                gain_loss_percent=value / 10.0,
            )
            for name, value in (("low", -10.0), ("base", 5.0), ("high", 20.0))
        ]
        return ScenarioAnalysisResponse(
            prediction=self._movement_response(),
            evidence_count=25,
            evidence_end_date=date(2018, 12, 20),
            outcomes=outcomes,
            method="Test historical method.",
        )

    def provenance(self) -> dict[str, Any]:
        return {"status": "provenance_verified"}


def make_client(tmp_path: Path) -> TestClient:
    """Create one authenticated test application."""

    settings = ApiSettings(
        project_root=tmp_path,
        environment="test",
        api_key="a" * 32,
        require_api_key=True,
        trusted_hosts=("testserver",),
        rate_limit_requests=20,
    )
    return TestClient(create_app(settings=settings, services=FakeServices()))


def test_health_endpoint_does_not_require_authentication(tmp_path: Path) -> None:
    """Prepare the app, run health, and check process status."""

    with make_client(tmp_path) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "PASSED"


def test_prediction_endpoint_rejects_missing_api_key(tmp_path: Path) -> None:
    """Prepare an unauthenticated request, run it, and check safe rejection."""

    with make_client(tmp_path) as client:
        response = client.post(
            "/v1/sentiment/text",
            json={"text": "Revenue increased."},
        )

    assert response.status_code == 401
    payload = response.json()
    assert payload["error_code"] == "authentication_failed"
    assert payload["what_failed"]
    assert payload["where_failed"]
    assert payload["why_failed"]
    assert payload["safe_next_step"]


def test_text_sentiment_endpoint_returns_explicit_probabilities(
    tmp_path: Path,
) -> None:
    """Prepare valid text, run prediction, and check the response contract."""

    with make_client(tmp_path) as client:
        response = client.post(
            "/v1/sentiment/text",
            headers={"X-API-Key": "a" * 32},
            json={"text": "Revenue increased."},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["label"] == "Bullish"
    assert payload["probabilities"] == {
        "bearish": 0.1,
        "neutral": 0.2,
        "bullish": 0.7,
    }
    assert "X-Request-ID" in response.headers


def test_unknown_request_field_is_rejected(tmp_path: Path) -> None:
    """Prepare a misspelled field, run validation, and check fail-closed schema."""

    with make_client(tmp_path) as client:
        response = client.post(
            "/v1/sentiment/text",
            headers={"X-API-Key": "a" * 32},
            json={"text": "Revenue increased.", "guess": True},
        )

    assert response.status_code == 422
    assert response.json()["error_code"] == "request_validation_failed"


def test_readiness_endpoint_reports_all_components(tmp_path: Path) -> None:
    """Prepare the app, run readiness, and check component status."""

    with make_client(tmp_path) as client:
        response = client.get("/ready")

    assert response.status_code == 200
    assert response.json()["components"]["movement_worker"] == "CONFIGURED"
    assert response.json()["components"]["sentiment_worker"] == "CONFIGURED"


def test_txt_file_endpoint_returns_one_prediction(tmp_path: Path) -> None:
    """Prepare a TXT upload, run file sentiment, and check batch output."""

    with make_client(tmp_path) as client:
        response = client.post(
            "/v1/sentiment/file",
            headers={"X-API-Key": "a" * 32},
            files={"file": ("article.txt", b"Revenue increased.", "text/plain")},
        )

    assert response.status_code == 200
    assert response.json()["result_count"] == 1


def test_file_endpoint_rejects_unsupported_extension(tmp_path: Path) -> None:
    """Prepare an unsupported upload, run extraction, and check safe error."""

    with make_client(tmp_path) as client:
        response = client.post(
            "/v1/sentiment/file",
            headers={"X-API-Key": "a" * 32},
            files={"file": ("article.exe", b"bad", "application/octet-stream")},
        )

    assert response.status_code == 415
    assert response.json()["error_code"] == "unsupported_file_type"


def test_movement_endpoint_returns_three_probabilities(tmp_path: Path) -> None:
    """Prepare historical input, run movement, and check class probabilities."""

    with make_client(tmp_path) as client:
        response = client.post(
            "/v1/movement/predict",
            headers={"X-API-Key": "a" * 32},
            json={
                "text": "Apple filed a quarterly report.",
                "ticker": "aapl",
                "published_at": "2019-01-02T12:00:00+00:00",
            },
        )

    assert response.status_code == 200
    assert response.json()["probabilities"] == {
        "down": 0.2,
        "flat": 0.2,
        "up": 0.6,
    }


def test_historical_endpoint_returns_earlier_event(tmp_path: Path) -> None:
    """Prepare historical input, run retrieval, and check earlier evidence."""

    with make_client(tmp_path) as client:
        response = client.post(
            "/v1/intelligence/historical",
            headers={"X-API-Key": "a" * 32},
            json={
                "text": "Apple filed a quarterly report.",
                "ticker": "AAPL",
                "published_at": "2019-01-02T12:00:00+00:00",
                "limit": 5,
                "minimum_similarity": 0.0,
            },
        )

    assert response.status_code == 200
    assert response.json()["matches"][0]["target_session_date"] == "2018-12-20"


def test_explainability_endpoint_returns_global_and_local_drivers(
    tmp_path: Path,
) -> None:
    """Prepare historical input, run explanation, and check both driver scopes."""

    with make_client(tmp_path) as client:
        response = client.post(
            "/v1/explainability",
            headers={"X-API-Key": "a" * 32},
            json={
                "text": "Apple filed a quarterly report.",
                "ticker": "AAPL",
                "published_at": "2019-01-02T12:00:00+00:00",
                "top_n": 5,
            },
        )

    assert response.status_code == 200
    assert len(response.json()["global_drivers"]) == 1
    assert len(response.json()["local_drivers"]) == 1


def test_scenario_endpoint_returns_three_outcomes(tmp_path: Path) -> None:
    """Prepare portfolio assumptions, run scenarios, and check three outcomes."""

    with make_client(tmp_path) as client:
        response = client.post(
            "/v1/scenarios/analyze",
            headers={"X-API-Key": "a" * 32},
            json={
                "text": "Apple filed a quarterly report.",
                "ticker": "AAPL",
                "published_at": "2019-01-02T12:00:00+00:00",
                "investment_amount": 1000.0,
                "share_price": 100.0,
            },
        )

    assert response.status_code == 200
    assert [item["scenario"] for item in response.json()["outcomes"]] == [
        "low",
        "base",
        "high",
    ]


def test_provenance_endpoint_returns_verified_status(tmp_path: Path) -> None:
    """Prepare the app, run provenance, and check verified evidence status."""

    with make_client(tmp_path) as client:
        response = client.get(
            "/v1/provenance",
            headers={"X-API-Key": "a" * 32},
        )

    assert response.status_code == 200
    assert response.json()["provenance"]["status"] == "provenance_verified"


def test_url_sentiment_endpoint_uses_bounded_extracted_text(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Prepare a safe extractor result, run URL sentiment, and check source type."""

    monkeypatch.setattr(
        "financial_news_intelligence.api.app.extract_public_url",
        lambda *_args, **_kwargs: "Revenue increased.",
    )

    with make_client(tmp_path) as client:
        response = client.post(
            "/v1/sentiment/url",
            headers={"X-API-Key": "a" * 32},
            json={"url": "https://example.com/article"},
        )

    assert response.status_code == 200
    assert response.json()["source_type"] == "url"


def test_historical_endpoint_rejects_limit_above_configuration(
    tmp_path: Path,
) -> None:
    """Prepare an excessive match count, run the route, and check rejection."""

    with make_client(tmp_path) as client:
        response = client.post(
            "/v1/intelligence/historical",
            headers={"X-API-Key": "a" * 32},
            json={
                "text": "Apple filed a quarterly report.",
                "ticker": "AAPL",
                "published_at": "2019-01-02T12:00:00+00:00",
                "limit": 6,
                "minimum_similarity": 0.0,
            },
        )

    assert response.status_code == 422
    assert response.json()["error_code"] == "historical_limit_exceeded"


def test_raw_body_file_endpoint_returns_one_prediction(tmp_path: Path) -> None:
    """Prepare raw TXT bytes, run file sentiment, and check batch output."""

    with make_client(tmp_path) as client:
        response = client.post(
            "/v1/sentiment/file",
            headers={
                "X-API-Key": "a" * 32,
                "X-Filename": "article.txt",
                "Content-Type": "text/plain",
            },
            content=b"Revenue increased.",
        )

    assert response.status_code == 200
    assert response.json()["result_count"] == 1
    assert response.json()["source_type"] == "txt"


def test_file_endpoint_openapi_describes_multipart_and_raw_inputs(
    tmp_path: Path,
) -> None:
    """Prepare OpenAPI, read the file route, and check both input contracts."""

    with make_client(tmp_path) as client:
        document = client.get("/openapi.json").json()

    content = document["paths"]["/v1/sentiment/file"]["post"]["requestBody"][
        "content"
    ]
    assert "multipart/form-data" in content
    assert "application/octet-stream" in content
    multipart_schema = content["multipart/form-data"]["schema"]
    assert multipart_schema["required"] == ["file"]
    assert multipart_schema["additionalProperties"] is False
