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


def _sentence(
    sentence_text: str,
    sentiment_label: str,
    confidence_score: float = .8,
):
    sentence_probabilities = {"Bearish": .1, "Neutral": .1, "Bullish": .1}
    sentence_probabilities[sentiment_label] = confidence_score
    return SimpleNamespace(
        text=sentence_text,
        label=sentiment_label,
        confidence=confidence_score,
        probabilities=sentence_probabilities,
    )


def test_wordpiece_merging_excludes_special_and_preserves_words() -> None:
    assert merge_wordpieces(["[CLS]", "space", "##x", "share", "price", ".", "[SEP]"]) == [
        "spacex", "share", "price", "."
    ]


def test_token_landscape_uses_only_supplied_tokens_and_excludes_punctuation() -> None:
    sentence_predictions = [_sentence("SpaceX demand increased", "Bullish")]
    token_rows = token_landscape(
        sentence_predictions,
        [["[CLS]", "space", "##x", "demand", "increased", ".", "[SEP]"]],
    )
    assert {token_row.token.lower() for token_row in token_rows} == {
        "spacex",
        "demand",
        "increased",
    }
    assert all(
        token_row.token not in {"[CLS]", "[SEP]", "."}
        for token_row in token_rows
    )


def test_reader_segments_round_trip_exact_text() -> None:
    article_text = "Headline\n\nRevenue rose sharply. Risk remained elevated."
    sentence_predictions = [
        _sentence("Headline", "Neutral"),
        _sentence("Revenue rose sharply.", "Bullish"),
        _sentence("Risk remained elevated.", "Bearish"),
    ]
    rendered_segments = reader_segments(article_text, sentence_predictions)
    assert "".join(segment.text for segment in rendered_segments) == article_text
    assert [
        segment.sentence_class
        for segment in rendered_segments
        if segment.sentence_class
    ] == [
        "Neutral",
        "Bullish",
        "Bearish",
    ]


def test_pca_projection_is_deterministic_and_complete() -> None:
    sentence_embeddings = [
        (1.0, 2.0, 0.0),
        (2.0, 0.0, 1.0),
        (0.0, 1.0, 2.0),
    ]
    first_projection = deterministic_pca(sentence_embeddings)
    second_projection = deterministic_pca(sentence_embeddings)
    assert first_projection == pytest.approx(second_projection)
    assert len(first_projection) == len(sentence_embeddings)


def test_contextual_network_and_layout_are_deterministic() -> None:
    token_nodes = [
        SimpleNamespace(text="revenue", embedding=(1.0, 0.0)),
        SimpleNamespace(text="growth", embedding=(.9, .1)),
        SimpleNamespace(text="risk", embedding=(0.0, 1.0)),
    ]
    assert cosine_links(token_nodes) == cosine_links(token_nodes)
    assert circular_positions(3) == circular_positions(3)
