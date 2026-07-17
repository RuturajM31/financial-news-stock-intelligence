"""Sentiment labels shared by Financial PhraseBank preparation and training."""

from enum import Enum


class SentimentLabel(str, Enum):
    """Possible financial-sentiment results."""

    BEARISH = "Bearish"
    NEUTRAL = "Neutral"
    BULLISH = "Bullish"