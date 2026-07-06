"""Shared labels and validation rules for project schemas."""

from enum import Enum

from pydantic import BaseModel, ConfigDict


class SourceType(str, Enum):
    """Ways an article can enter the system."""

    TEXT = "text"
    TXT = "txt"
    PDF = "pdf"
    DOCX = "docx"
    CSV = "csv"
    URL = "url"


class SentimentLabel(str, Enum):
    """Possible financial-sentiment results."""

    BEARISH = "Bearish"
    NEUTRAL = "Neutral"
    BULLISH = "Bullish"


class MovementLabel(str, Enum):
    """Possible next-session movement results."""

    DOWN = "Down"
    FLAT = "Flat"
    UP = "Up"


class MarketSession(str, Enum):
    """Publication timing relative to the stock market."""

    BEFORE_MARKET = "Before Market"
    DURING_MARKET = "During Market"
    AFTER_MARKET = "After Market"
    WEEKEND = "Weekend"
    HOLIDAY = "Holiday"
    UNKNOWN = "Unknown"


class ProjectSchema(BaseModel):
    """Base validation rules inherited by every project schema."""

    # Reject unknown fields so spelling mistakes are caught immediately.
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )
