from __future__ import annotations

from types import SimpleNamespace

import pytest

from financial_news_intelligence.models.bert_visualization import (
    circular_positions,
    cosine_links,
    merge_wordpieces,
    reader_segments,
    token_landscape,
)
from financial_news_intelligence.models.full_bert_inference import deterministic_pca


def _sentence(text: str, label: str, confidence: float = .8):
    scores = {"Bearish": .1, "Neutral": .1, "Bullish": .1}
    scores[label] = confidence
    return SimpleNamespace(text=text, label=label, confidence=confidence, probabilities=scores)


def test_wordpiece_merging_excludes_special_and_preserves_words() -> None:
    assert merge_wordpieces(["[CLS]", "space", "##x", "share", "price", ".", "[SEP]"]) == [
        "spacex", "share", "price", "."
    ]


def test_token_landscape_uses_only_supplied_tokens_and_excludes_punctuation() -> None:
    predictions = [_sentence("SpaceX demand increased", "Bullish")]
    rows = token_landscape(predictions, [["[CLS]", "space", "##x", "demand", "increased", ".", "[SEP]"]])
    assert {row.token.lower() for row in rows} == {"spacex", "demand", "increased"}
    assert all(row.token not in {"[CLS]", "[SEP]", "."} for row in rows)


def test_reader_segments_round_trip_exact_text() -> None:
    text = "Headline\n\nRevenue rose sharply. Risk remained elevated."
    predictions = [
        _sentence("Headline", "Neutral"),
        _sentence("Revenue rose sharply.", "Bullish"),
        _sentence("Risk remained elevated.", "Bearish"),
    ]
    segments = reader_segments(text, predictions)
    assert "".join(segment.text for segment in segments) == text
    assert [segment.sentence_class for segment in segments if segment.sentence_class] == [
        "Neutral", "Bullish", "Bearish"
    ]


def test_pca_projection_is_deterministic_and_complete() -> None:
    embeddings = [(1.0, 2.0, 0.0), (2.0, 0.0, 1.0), (0.0, 1.0, 2.0)]
    first = deterministic_pca(embeddings)
    second = deterministic_pca(embeddings)
    assert first == pytest.approx(second)
    assert len(first) == len(embeddings)


def test_contextual_network_and_layout_are_deterministic() -> None:
    nodes = [
        SimpleNamespace(text="revenue", embedding=(1.0, 0.0)),
        SimpleNamespace(text="growth", embedding=(.9, .1)),
        SimpleNamespace(text="risk", embedding=(0.0, 1.0)),
    ]
    assert cosine_links(nodes) == cosine_links(nodes)
    assert circular_positions(3) == circular_positions(3)
