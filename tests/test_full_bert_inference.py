from __future__ import annotations

from types import SimpleNamespace

import pytest

from financial_news_intelligence.models.full_bert_inference import (
    LABEL_ORDER,
    analyze_article,
    split_article_sentences,
)


def test_sentence_split_keeps_headline_and_filters_boilerplate() -> None:
    article_sentences = split_article_sentences(
        "Company raises guidance",
        "Revenue rose sharply during the quarter. Accept cookies to continue. Demand remained strong across regions.",
    )
    expected_sentences = [
        "Company raises guidance",
        "Revenue rose sharply during the quarter.",
        "Demand remained strong across regions.",
    ]
    assert article_sentences == expected_sentences


def test_article_requires_meaningful_content() -> None:
    mock_runtime = SimpleNamespace()
    with pytest.raises(ValueError, match="meaningful sentence"):
        analyze_article(mock_runtime, "", "Too short")


def test_verified_label_order_is_stable() -> None:
    assert LABEL_ORDER == ("Bearish", "Neutral", "Bullish")
