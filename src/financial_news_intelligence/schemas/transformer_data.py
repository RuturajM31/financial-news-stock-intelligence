"""Define the article format used for Transformer model training."""

from datetime import datetime

from pydantic import Field, field_validator, model_validator

from financial_news_intelligence.schemas.common import (
    ProjectSchema,
    SentimentLabel,
)
from financial_news_intelligence.schemas.training_data import (
    DatasetSplit,
)


# ============================================================
# 1. ONE ARTICLE CARD
# ============================================================

class TransformerExample(ProjectSchema):
    """
    Store one article that DistilBERT or BERT will read later.

    Main model inputs:
    - text: the article content;
    - label: Bullish, Neutral, or Bearish.
    """

    # Unique name for this exact article.
    article_id: str = Field(min_length=1)

    # Articles about the same real-world event share this value.
    event_id: str = Field(min_length=1)

    # Financial-news text read by the Transformer.
    text: str = Field(min_length=20)

    # Correct answer used to teach the Transformer.
    label: SentimentLabel

    # Publication time used for chronological splitting.
    published_at: datetime

    # Optional company details used for later analysis.
    company: str | None = None
    ticker: str | None = None

    # Link to the verified source information created earlier.
    source_id: str = Field(min_length=1)

    # Empty at first. Added by the splitting function later.
    split: DatasetSplit | None = None

    @field_validator("published_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        """Require a timezone so articles are ordered correctly."""

        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(
                "published_at must include a timezone."
            )

        return value


# ============================================================
# 2. DATASET SPLIT SETTINGS
# ============================================================

class TransformerSplitConfig(ProjectSchema):
    """
    Store how the articles will be divided.

    Oldest 70%:
        training

    Next 15%:
        validation

    Newest 15%:
        testing
    """

    train_ratio: float = Field(
        default=0.70,
        gt=0,
        lt=1,
    )

    validation_ratio: float = Field(
        default=0.15,
        gt=0,
        lt=1,
    )

    test_ratio: float = Field(
        default=0.15,
        gt=0,
        lt=1,
    )

    @model_validator(mode="after")
    def check_total(self) -> "TransformerSplitConfig":
        """Confirm that the three ratios total 100%."""

        total = (
            self.train_ratio
            + self.validation_ratio
            + self.test_ratio
        )

        if round(total, 6) != 1.0:
            raise ValueError(
                "Dataset split ratios must total 1.0."
            )

        return self
