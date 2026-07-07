"""Stable settings shared across the project."""

from typing import Final


PROJECT_NAME: Final[str] = "Financial News and Stock Movement Intelligence"
RANDOM_SEED: Final[int] = 42

# Labels returned by the sentiment model.
SENTIMENT_LABELS: Final[tuple[str, ...]] = (
    "Bearish",
    "Neutral",
    "Bullish",
)

# Labels returned by the movement model.
MOVEMENT_LABELS: Final[tuple[str, ...]] = (
    "Down",
    "Flat",
    "Up",
)

# Input types accepted by the application.
SUPPORTED_INPUT_TYPES: Final[tuple[str, ...]] = (
    "text",
    "txt",
    "pdf",
    "docx",
    "csv",
    "url",
)

DISCLAIMER: Final[str] = (
    "Historical statistical intelligence only. "
    "Not financial advice or a guaranteed prediction."
)
