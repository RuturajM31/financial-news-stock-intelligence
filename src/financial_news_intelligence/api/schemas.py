"""Explicit request and response schemas for every FastAPI endpoint."""

from __future__ import annotations

import re
from datetime import date, datetime
from enum import Enum
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    field_validator,
    model_validator,
)


class SentimentLabel(str, Enum):
    """Supported financial-sentiment labels."""

    BEARISH = "Bearish"
    NEUTRAL = "Neutral"
    BULLISH = "Bullish"


class MovementLabel(str, Enum):
    """Supported next-session movement labels."""

    DOWN = "Down"
    FLAT = "Flat"
    UP = "Up"


class ProjectSchema(BaseModel):
    """Reject unknown fields and normalize surrounding whitespace."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )


TICKER_PATTERN = re.compile(r"^[A-Z][A-Z0-9.\-]{0,14}$")
MANDATORY_DISCLAIMER = (
    "For educational and research use only. This output is not financial, "
    "investment, legal, or tax advice. Markets involve risk, and users must "
    "perform independent due diligence."
)


class TextSentimentRequest(ProjectSchema):
    """One pasted financial-news text request."""

    text: str = Field(min_length=1, max_length=20_000)


class UrlSentimentRequest(ProjectSchema):
    """One public HTTP or HTTPS article URL."""

    url: HttpUrl


class SentimentProbabilitiesResponse(ProjectSchema):
    """Bearish, Neutral, and Bullish probabilities."""

    bearish: float = Field(ge=0.0, le=1.0)
    neutral: float = Field(ge=0.0, le=1.0)
    bullish: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_total(self) -> "SentimentProbabilitiesResponse":
        total = self.bearish + self.neutral + self.bullish
        if abs(total - 1.0) > 1e-6:
            raise ValueError("Sentiment probabilities must total 1.0.")
        return self


class SentimentPredictionResponse(ProjectSchema):
    """One DistilBERT sentiment prediction."""

    status: str = "PASSED"
    label: SentimentLabel
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: SentimentProbabilitiesResponse
    deployment_model: str = "distilbert"
    source_type: str
    warnings: list[str] = Field(default_factory=list)


class BatchSentimentResponse(ProjectSchema):
    """Sentiment predictions for one file or CSV batch."""

    status: str = "PASSED"
    source_type: str
    result_count: int = Field(ge=1)
    results: list[SentimentPredictionResponse]


class MovementPredictionRequest(ProjectSchema):
    """Historical-research movement request using verified session evidence."""

    text: str = Field(min_length=1, max_length=20_000)
    ticker: str
    published_at: datetime

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, value: str) -> str:
        normalized = value.strip().upper()
        if not TICKER_PATTERN.fullmatch(normalized):
            raise ValueError("Ticker contains unsupported characters.")
        return normalized

    @field_validator("published_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("published_at must include a timezone.")
        return value


class MovementProbabilitiesResponse(ProjectSchema):
    """Down, Flat, and Up probabilities."""

    down: float = Field(ge=0.0, le=1.0)
    flat: float = Field(ge=0.0, le=1.0)
    up: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_total(self) -> "MovementProbabilitiesResponse":
        total = self.down + self.flat + self.up
        if abs(total - 1.0) > 1e-6:
            raise ValueError("Movement probabilities must total 1.0.")
        return self


class MovementPredictionResponse(ProjectSchema):
    """One leakage-safe historical-session movement prediction."""

    status: str = "PASSED"
    ticker: str
    target_session_date: date
    direction: MovementLabel
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: MovementProbabilitiesResponse
    sentiment: SentimentPredictionResponse
    champion_model: str
    research_mode: str = "verified_historical_audit_sessions_only"
    warnings: list[str] = Field(default_factory=list)
    disclaimer: str = MANDATORY_DISCLAIMER


class HistoricalIntelligenceRequest(MovementPredictionRequest):
    """Text and cutoff used to retrieve strictly earlier SEC events."""

    limit: int = Field(default=5, ge=1, le=20)
    minimum_similarity: float = Field(default=0.0, ge=0.0, le=1.0)


class HistoricalEventMatch(ProjectSchema):
    """One earlier same-ticker SEC event and observed reaction."""

    article_id: str
    ticker: str
    target_session_date: date
    source_url: str
    sentiment_label: SentimentLabel
    movement_label: MovementLabel
    reaction_return_percent: float
    similarity_score: float = Field(ge=0.0, le=1.0)


class HistoricalIntelligenceResponse(ProjectSchema):
    """Earlier-only matches and reference-period phrase evidence."""

    status: str = "PASSED"
    ticker: str
    query_target_session_date: date
    matches: list[HistoricalEventMatch]
    important_phrases: list[str] = Field(default_factory=list)
    reference_scope: str = "train_validation_and_strictly_earlier_only"
    limitations: list[str]
    disclaimer: str = MANDATORY_DISCLAIMER


class DriverEvidence(ProjectSchema):
    """One global or local movement-model driver."""

    rank: int = Field(ge=1)
    feature: str
    importance: float | None = None
    probability_effect: float | None = None
    absolute_effect: float | None = None
    direction: str | None = None
    method: str
    interpretation: str


class ExplainabilityRequest(MovementPredictionRequest):
    """Movement request plus a requested local-driver count."""

    top_n: int = Field(default=5, ge=1, le=20)


class ExplainabilityResponse(ProjectSchema):
    """Global and local sensitivity evidence for one prediction."""

    status: str = "PASSED"
    prediction: MovementPredictionResponse
    global_drivers: list[DriverEvidence]
    local_drivers: list[DriverEvidence]
    reference_scope: str = "train_validation_only"
    limitation: str = (
        "Driver values describe model sensitivity and do not prove causality."
    )


class ScenarioAnalysisRequest(MovementPredictionRequest):
    """Movement request plus user-controlled portfolio assumptions."""

    investment_amount: float = Field(gt=0)
    share_price: float = Field(gt=0)
    currency: str = "EUR"
    allow_fractional_shares: bool = True
    share_precision: int = Field(default=6, ge=0, le=8)
    entry_fee: float = Field(default=0, ge=0)
    exit_fee: float = Field(default=0, ge=0)
    tax_rate_percent: float | None = Field(default=None, ge=0, le=100)

    @field_validator("currency")
    @classmethod
    def normalize_currency(cls, value: str) -> str:
        normalized = value.strip().upper()
        if not re.fullmatch(r"[A-Z]{3}", normalized):
            raise ValueError("currency must be a three-letter code.")
        return normalized

    @model_validator(mode="after")
    def validate_available_capital(self) -> "ScenarioAnalysisRequest":
        if self.entry_fee >= self.investment_amount:
            raise ValueError("entry_fee must be below investment_amount.")
        return self


class ScenarioOutcomeResponse(ProjectSchema):
    """Portfolio value under one historical-return scenario."""

    scenario: str
    historical_return_percent: float
    shares_purchased: float = Field(ge=0)
    cash_balance: float = Field(ge=0)
    estimated_tax: float = Field(ge=0)
    net_final_value: float = Field(ge=0)
    gain_loss: float
    gain_loss_percent: float


class ScenarioAnalysisResponse(ProjectSchema):
    """Downside, base, and upside research scenarios."""

    status: str = "PASSED"
    prediction: MovementPredictionResponse
    evidence_count: int = Field(ge=1)
    evidence_end_date: date
    class_median_fallbacks: list[str] = Field(default_factory=list)
    outcomes: list[ScenarioOutcomeResponse]
    method: str
    disclaimer: str = MANDATORY_DISCLAIMER


class HealthResponse(ProjectSchema):
    """Process health without model loading."""

    status: str
    service: str
    version: str


class ReadinessResponse(ProjectSchema):
    """Artifact and model readiness for traffic."""

    status: str
    components: dict[str, str]
    details: dict[str, Any] = Field(default_factory=dict)


class ProvenanceResponse(ProjectSchema):
    """Licence-safe project provenance and deployment boundary."""

    status: str = "PASSED"
    provenance: dict[str, Any]
