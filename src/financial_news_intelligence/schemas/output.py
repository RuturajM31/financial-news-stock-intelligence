"""Schemas for sentiment, forecasts, explanations, and final results."""

from datetime import date, datetime, timezone
from math import isclose

from pydantic import Field, HttpUrl, model_validator

from .common import MovementLabel, ProjectSchema, SentimentLabel
from .context import ArticleContext, FinancialContext


class SentimentProbabilities(ProjectSchema):
    """Probabilities for Bearish, Neutral, and Bullish sentiment."""

    bearish: float = Field(ge=0.0, le=1.0)
    neutral: float = Field(ge=0.0, le=1.0)
    bullish: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_total(self) -> "SentimentProbabilities":
        # All sentiment probabilities must form one complete result.
        total = self.bearish + self.neutral + self.bullish

        if not isclose(total, 1.0, abs_tol=1e-4):
            raise ValueError("Sentiment probabilities must total 1.0.")

        return self


class MovementProbabilities(ProjectSchema):
    """Probabilities for Down, Flat, and Up movement."""

    down: float = Field(ge=0.0, le=1.0)
    flat: float = Field(ge=0.0, le=1.0)
    up: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_total(self) -> "MovementProbabilities":
        # All movement probabilities must form one complete result.
        total = self.down + self.flat + self.up

        if not isclose(total, 1.0, abs_tol=1e-4):
            raise ValueError("Movement probabilities must total 1.0.")

        return self


class SentimentResult(ProjectSchema):
    """Complete financial-sentiment result."""

    label: SentimentLabel
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: SentimentProbabilities


class HistoricalReturnRange(ProjectSchema):
    """Historical return range from similar past articles."""

    lower_percent: float | None = None
    median_percent: float | None = None
    upper_percent: float | None = None
    sample_size: int = Field(default=0, ge=0)


class ForecastResult(ProjectSchema):
    """
    Dedicated next-trading-session forecast.

    Output: Direction, probabilities, confidence, historical range,
    and the main factors that influenced the estimate.
    """

    forecast_for: date | None = None
    direction: MovementLabel
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: MovementProbabilities

    # Show the historical range instead of claiming an exact future price.
    historical_return_range: HistoricalReturnRange | None = None

    # Explain the main article signals behind the forecast.
    key_drivers: list[str] = Field(default_factory=list)


class ImportantPhrase(ProjectSchema):
    """One article phrase that influenced the model result."""

    text: str = Field(min_length=1)
    importance_score: float = Field(ge=0.0, le=1.0)


class SimilarArticle(ProjectSchema):
    """One similar historical article used for comparison."""

    title: str = Field(min_length=1)
    published_at: datetime | None = None
    ticker: str | None = None
    similarity_score: float = Field(ge=0.0, le=1.0)

    sentiment: SentimentLabel | None = None
    next_session_movement: MovementLabel | None = None
    source_url: HttpUrl | None = None


class ModelReference(ProjectSchema):
    """Model identity stored for transparency and reproducibility."""

    task: str
    model_name: str
    model_version: str | None = None


class IntelligenceResult(ProjectSchema):
    """
    Complete result returned by FastAPI and displayed by Streamlit.

    Input:  Context, predictions, explanations, ratios, and history.
    Output: One structured financial-intelligence response.
    """

    # Record when the analysis result was created.
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    article_context: ArticleContext
    sentiment: SentimentResult
    forecast: ForecastResult

    important_phrases: list[ImportantPhrase] = Field(default_factory=list)
    key_entities: list[str] = Field(default_factory=list)
    similar_articles: list[SimilarArticle] = Field(default_factory=list)

    financial_context: FinancialContext | None = None
    models_used: list[ModelReference] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    disclaimer: str
