from __future__ import annotations

from types import SimpleNamespace

import pytest

from financial_news_intelligence.models.full_bert_inference import (
    LABEL_ORDER,
    analyze_article,
    split_article_sentences,
)


def test_sentence_split_keeps_headline_and_filters_boilerplate() -> None:
    sentences = split_article_sentences(
        "Company raises guidance",
        "Revenue rose sharply during the quarter. Accept cookies to continue. Demand remained strong across regions.",
    )
    assert sentences == [
        "Company raises guidance",
        "Revenue rose sharply during the quarter.",
        "Demand remained strong across regions.",
    ]


def test_article_requires_meaningful_content() -> None:
    with pytest.raises(ValueError, match="meaningful sentence"):
        analyze_article(SimpleNamespace(), "", "Too short")


def test_verified_label_order_is_stable() -> None:
    assert LABEL_ORDER == ("Bearish", "Neutral", "Bullish")
