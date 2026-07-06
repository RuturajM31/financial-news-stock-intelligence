"""Application service orchestration without native-library imports.

The FastAPI process performs request validation and response construction.
DistilBERT sentiment runs in ``.venv-distilbert``. Movement prediction and all
NumPy/pandas/scikit-learn intelligence run in one separate ``.venv`` worker.
This boundary prevents a native crash from terminating the web process.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping

from .artifacts import ArtifactPaths, ArtifactRegistry
from .config import ApiSettings
from .movement_worker_client import MovementWorkerClient
from .monitoring import observe_operation
from .schemas import (
    DriverEvidence,
    ExplainabilityResponse,
    HistoricalEventMatch,
    HistoricalIntelligenceResponse,
    MovementPredictionResponse,
    MovementProbabilitiesResponse,
    ScenarioAnalysisRequest,
    ScenarioAnalysisResponse,
    ScenarioOutcomeResponse,
    SentimentPredictionResponse,
    SentimentProbabilitiesResponse,
)
from .sentiment_worker_client import SentimentWorkerClient


class ApiServices:
    """Lazy verified services shared by all requests in one web process."""

    def __init__(self, settings: ApiSettings) -> None:
        self.settings = settings
        self._paths: ArtifactPaths | None = None
        self._movement: MovementWorkerClient | None = None
        self._sentiment: SentimentWorkerClient | None = None

    def _artifact_paths(self) -> ArtifactPaths:
        """Verify recorded checksums before either model worker starts."""

        if self._paths is None:
            self._paths = ArtifactRegistry(self.settings.project_root).verify()
        return self._paths

    def _movement_client(self) -> MovementWorkerClient:
        """Start the isolated movement worker only when first required."""

        if self._movement is None:
            self._artifact_paths()
            self._movement = MovementWorkerClient(
                project_root=self.settings.project_root,
                python_executable=self.settings.project_root / ".venv/bin/python",
                timeout_seconds=self.settings.movement_worker_timeout_seconds,
            )
        return self._movement

    def _sentiment_client(self) -> SentimentWorkerClient:
        """Start the isolated DistilBERT worker only when first required."""

        if self._sentiment is None:
            paths = self._artifact_paths()
            self._sentiment = SentimentWorkerClient(
                project_root=self.settings.project_root,
                python_executable=paths.sentiment_python,
                model_directory=paths.sentiment_model_directory,
                timeout_seconds=self.settings.sentiment_worker_timeout_seconds,
            )
        return self._sentiment

    @staticmethod
    def _sentiment_response(
        raw: Mapping[str, Any],
        source_type: str,
    ) -> SentimentPredictionResponse:
        """Convert one worker result into the public response schema."""

        return SentimentPredictionResponse(
            label=raw["label"],
            confidence=float(raw["confidence"]),
            probabilities=SentimentProbabilitiesResponse(
                bearish=float(raw["prob_bearish"]),
                neutral=float(raw["prob_neutral"]),
                bullish=float(raw["prob_bullish"]),
            ),
            source_type=source_type,
            warnings=[
                "Sentiment describes the supplied text and is not an investment signal."
            ],
        )

    @observe_operation("sentiment_prediction")
    def predict_sentiment(
        self,
        texts: list[str],
        source_type: str,
    ) -> list[SentimentPredictionResponse]:
        """Return one explicit three-class result per accepted text."""

        raw_results = self._sentiment_client().predict(texts)
        return [
            self._sentiment_response(raw, source_type) for raw in raw_results
        ]

    @staticmethod
    def _movement_response(
        raw: Mapping[str, Any],
        ticker: str,
        sentiment: SentimentPredictionResponse,
    ) -> MovementPredictionResponse:
        """Convert one isolated-worker prediction into the public schema."""

        prediction = raw["prediction"]
        return MovementPredictionResponse(
            ticker=ticker,
            target_session_date=raw["target_session_date"],
            direction=prediction["direction"],
            confidence=float(prediction["confidence"]),
            probabilities=MovementProbabilitiesResponse(
                down=float(prediction["prob_down"]),
                flat=float(prediction["prob_flat"]),
                up=float(prediction["prob_up"]),
            ),
            sentiment=sentiment,
            champion_model=str(raw["champion_model"]),
            warnings=[
                "The verified price and model evidence ends in 2020.",
                "This endpoint supports historical research sessions only.",
            ],
        )

    def _worker_payload(
        self,
        text: str,
        ticker: str,
        published_at: datetime,
        sentiment_raw: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Build the stable worker request shared by intelligence operations."""

        return {
            "text": text,
            "ticker": ticker,
            "published_at": published_at.isoformat(),
            "sentiment": dict(sentiment_raw),
        }

    @observe_operation("movement_prediction")
    def predict_movement(
        self,
        text: str,
        ticker: str,
        published_at: datetime,
        source_type: str = "text",
    ) -> MovementPredictionResponse:
        """Combine isolated sentiment and movement predictions."""

        sentiment_raw = self._sentiment_client().predict([text])[0]
        sentiment = self._sentiment_response(sentiment_raw, source_type)
        raw = self._movement_client().predict(
            self._worker_payload(text, ticker, published_at, sentiment_raw)
        )
        return self._movement_response(raw, ticker, sentiment)

    @observe_operation("historical_intelligence")
    def historical(
        self,
        text: str,
        ticker: str,
        published_at: datetime,
        limit: int,
        minimum_similarity: float,
    ) -> HistoricalIntelligenceResponse:
        """Return strictly earlier same-ticker reference events."""

        sentiment_raw = self._sentiment_client().predict([text])[0]
        sentiment = self._sentiment_response(sentiment_raw, "text")
        payload = self._worker_payload(text, ticker, published_at, sentiment_raw)
        payload.update({"limit": limit, "minimum_similarity": minimum_similarity})
        raw = self._movement_client().historical(payload)
        prediction = self._movement_response(raw, ticker, sentiment)
        return HistoricalIntelligenceResponse(
            ticker=ticker,
            query_target_session_date=prediction.target_session_date,
            matches=[HistoricalEventMatch(**item) for item in raw["matches"]],
            important_phrases=list(raw.get("important_phrases", [])),
            limitations=[
                (
                    "Similarity measures word overlap and does not prove the "
                    "events are equivalent."
                ),
                "Historical reactions do not guarantee future returns.",
            ],
        )

    @observe_operation("movement_explainability")
    def explain(
        self,
        text: str,
        ticker: str,
        published_at: datetime,
        top_n: int,
    ) -> ExplainabilityResponse:
        """Return global and local sensitivity evidence from the worker."""

        sentiment_raw = self._sentiment_client().predict([text])[0]
        sentiment = self._sentiment_response(sentiment_raw, "text")
        payload = self._worker_payload(text, ticker, published_at, sentiment_raw)
        payload["top_n"] = top_n
        raw = self._movement_client().explain(payload)
        return ExplainabilityResponse(
            prediction=self._movement_response(raw, ticker, sentiment),
            global_drivers=[DriverEvidence(**item) for item in raw["global_drivers"]],
            local_drivers=[DriverEvidence(**item) for item in raw["local_drivers"]],
        )

    @observe_operation("scenario_analysis")
    def scenario(self, request: ScenarioAnalysisRequest) -> ScenarioAnalysisResponse:
        """Return transparent user-controlled investment scenarios."""

        sentiment_raw = self._sentiment_client().predict([request.text])[0]
        sentiment = self._sentiment_response(sentiment_raw, "text")
        payload = self._worker_payload(
            request.text,
            request.ticker,
            request.published_at,
            sentiment_raw,
        )
        payload["scenario_request"] = {
            "investment_amount": request.investment_amount,
            "share_price": request.share_price,
            "currency": request.currency,
            "allow_fractional_shares": request.allow_fractional_shares,
            "share_precision": request.share_precision,
            "entry_fee": request.entry_fee,
            "exit_fee": request.exit_fee,
            "tax_rate_percent": request.tax_rate_percent,
        }
        raw = self._movement_client().scenario(payload)
        return ScenarioAnalysisResponse(
            prediction=self._movement_response(raw, request.ticker, sentiment),
            evidence_count=int(raw["evidence_count"]),
            evidence_end_date=raw["evidence_end_date"],
            class_median_fallbacks=list(raw.get("class_median_fallbacks", [])),
            outcomes=[ScenarioOutcomeResponse(**item) for item in raw["outcomes"]],
            method=str(raw["method"]),
        )

    @observe_operation("provenance_lookup")
    def provenance(self) -> dict[str, Any]:
        """Return licence-safe provenance without exposing private price rows."""

        return dict(self._movement_client().provenance()["provenance"])

    @observe_operation("readiness_check")
    def readiness(self, run_deep_probe: bool = False) -> dict[str, Any]:
        """Return cached artifact readiness and optional isolated worker probes.

        The public readiness endpoint uses the lightweight path by default. One
        call verifies recorded artifact checksums and worker launch locations,
        but it does not load DistilBERT, pandas, scikit-learn, or the movement
        model. Deep probes remain available to the installation verifier.
        """

        paths = self._artifact_paths()
        movement_script = (
            self.settings.project_root
            / "scripts"
            / "run_movement_intelligence_worker.py"
        )
        sentiment_script = (
            self.settings.project_root
            / "scripts"
            / "run_sentiment_inference_worker.py"
        )
        for script_path, description in (
            (movement_script, "movement worker script"),
            (sentiment_script, "sentiment worker script"),
        ):
            if not script_path.is_file() or script_path.is_symlink():
                from .errors import ApiProblem

                raise ApiProblem(
                    503,
                    "worker_script_missing",
                    "Readiness verification failed.",
                    str(script_path),
                    f"The required {description} is missing or unsafe.",
                    "Reinstall the verified FastAPI package and rerun readiness.",
                )

        components = {
            "artifacts": "PASSED",
            "movement_worker": "CONFIGURED",
            "sentiment_worker": "CONFIGURED",
        }
        details: dict[str, Any] = {
            "sentiment_model": paths.sentiment_model_directory.name,
            "native_model_processes_isolated": True,
            "deep_probe_run": False,
        }
        if run_deep_probe:
            movement_probe = self._movement_client().readiness()
            sentiment_probe = self.predict_sentiment(
                ["The company filed its quarterly report."],
                "readiness_probe",
            )[0]
            components["movement_worker"] = "PASSED"
            components["sentiment_worker"] = "PASSED"
            details["movement_probe"] = movement_probe
            details["sentiment_probe_label"] = sentiment_probe.label.value
            details["deep_probe_run"] = True
        return {"components": components, "details": details}

    def close(self) -> None:
        """Stop both isolated workers during application shutdown."""

        if self._sentiment is not None:
            self._sentiment.close()
        if self._movement is not None:
            self._movement.close()
