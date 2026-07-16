"""Deterministic presentation helpers for real Full BERT article outputs."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
import re
from typing import Any, Iterable

from .full_bert_inference import LABEL_ORDER


STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "for",
    "from", "had", "has", "have", "he", "her", "his", "in", "into", "is",
    "it", "its", "of", "on", "or", "our", "said", "she", "that", "the",
    "their", "they", "this", "to", "was", "were", "will", "with", "would",
}
SPECIAL_TOKENS = {"[CLS]", "[SEP]", "[PAD]", "[UNK]", "[MASK]"}


@dataclass(frozen=True)
class TokenLandscapeItem:
    token: str
    count: int
    dominant_class: str
    class_counts: dict[str, int]
    example_sentence: str


@dataclass(frozen=True)
class ReaderSegment:
    text: str
    sentence_index: int | None
    sentence_class: str | None
    score: float


def merge_wordpieces(tokens: Iterable[str]) -> list[str]:
    """Merge `##` fragments without treating token frequency as importance."""

    merged: list[str] = []
    for token in tokens:
        if token in SPECIAL_TOKENS or not token:
            continue
        if token.startswith("##") and merged:
            merged[-1] += token[2:]
        else:
            merged.append(token)
    return merged


def token_landscape(
    sentence_predictions: Iterable[Any],
    tokenized_sentences: Iterable[list[str]],
) -> list[TokenLandscapeItem]:
    """Build token counts and class distributions from submitted article text."""

    token_counts: Counter[str] = Counter()
    token_class_counts: dict[str, Counter[str]] = defaultdict(Counter)
    example_sentences: dict[str, str] = {}
    display_tokens: dict[str, str] = {}

    for sentence_prediction, wordpiece_tokens in zip(
        sentence_predictions,
        tokenized_sentences,
    ):
        # Preserve the article's original capitalization for display.
        original_words = {
            re.sub(r"[^a-z0-9]", "", word.lower()): word.strip(
                ".,;:!?()[]{}\"'"
            )
            for word in sentence_prediction.text.split()
        }

        for merged_token in merge_wordpieces(wordpiece_tokens):
            normalized_token = re.sub(
                r"[^a-z0-9-]",
                "",
                merged_token.lower(),
            )
            token_is_usable = (
                normalized_token
                and normalized_token not in STOP_WORDS
                and re.search(r"[a-z0-9]", normalized_token)
            )
            if not token_is_usable:
                continue

            token_counts[normalized_token] += 1
            token_class_counts[normalized_token][sentence_prediction.label] += 1
            example_sentences.setdefault(
                normalized_token,
                sentence_prediction.text,
            )
            display_tokens.setdefault(
                normalized_token,
                original_words.get(normalized_token, normalized_token),
            )

    landscape_items: list[TokenLandscapeItem] = []

    for normalized_token, token_count in token_counts.most_common(36):
        class_distribution = {
            class_label: token_class_counts[normalized_token][class_label]
            for class_label in LABEL_ORDER
        }
        dominant_class = max(
            LABEL_ORDER,
            key=lambda class_label: (
                class_distribution[class_label],
                -LABEL_ORDER.index(class_label),
            ),
        )
        landscape_items.append(
            TokenLandscapeItem(
                display_tokens[normalized_token],
                token_count,
                dominant_class,
                class_distribution,
                example_sentences[normalized_token],
            )
        )

    return landscape_items

def reader_segments(text: str, sentence_predictions: Iterable[Any]) -> list[ReaderSegment]:
    """Address analyzed sentences while preserving every original character."""

    segments: list[ReaderSegment] = []
    cursor = 0
    for index, prediction in enumerate(sentence_predictions):
        start = text.find(prediction.text, cursor)
        if start < 0:
            start = text.lower().find(prediction.text.lower(), cursor)
        if start < 0:
            continue
        if start > cursor:
            segments.append(ReaderSegment(text[cursor:start], None, None, 0.0))
        end = start + len(prediction.text)
        segments.append(ReaderSegment(
            text[start:end], index, prediction.label, float(prediction.confidence)
        ))
        cursor = end
    if cursor < len(text):
        segments.append(ReaderSegment(text[cursor:], None, None, 0.0))
    return segments


def cosine_links(
    tokens: Iterable[Any],
    *,
    maximum_nodes: int = 18,
    maximum_links: int = 28,
) -> tuple[list[Any], list[tuple[int, int, float]]]:
    """Return the strongest positive contextual-token similarities.

    Similarity describes proximity between model representations. It is not
    causal evidence and does not measure a token's influence on the prediction.
    """

    # NumPy stays lazy because this helper is rendered only on demand.
    import numpy as np

    contextual_tokens = [
        token
        for token in tokens
        if token.text.lower() not in STOP_WORDS
        and re.search(r"[A-Za-z0-9]", token.text)
    ][:maximum_nodes]

    if len(contextual_tokens) < 2:
        return contextual_tokens, []

    token_embedding_matrix = np.asarray(
        [token.embedding for token in contextual_tokens],
        dtype=np.float64,
    )
    token_embedding_norms = np.linalg.norm(
        token_embedding_matrix,
        axis=1,
        keepdims=True,
    )
    normalized_token_embeddings = token_embedding_matrix / np.maximum(
        token_embedding_norms,
        1e-12,
    )
    token_similarity_matrix = (
        normalized_token_embeddings @ normalized_token_embeddings.T
    )

    candidate_links = [
        (
            source_index,
            target_index,
            float(token_similarity_matrix[source_index, target_index]),
        )
        for source_index in range(len(contextual_tokens))
        for target_index in range(source_index + 1, len(contextual_tokens))
        if math.isfinite(
            token_similarity_matrix[source_index, target_index]
        )
        and token_similarity_matrix[source_index, target_index] > 0
    ]
    candidate_links.sort(
        key=lambda candidate_link: (
            -candidate_link[2],
            candidate_link[0],
            candidate_link[1],
        )
    )
    strongest_links = candidate_links[:maximum_links]

    return contextual_tokens, strongest_links

def circular_positions(count: int) -> list[tuple[float, float]]:
    """Return stable positions for a compact contextual-token network."""

    if count <= 0:
        return []
    return [
        (math.cos(2 * math.pi * index / count), math.sin(2 * math.pi * index / count))
        for index in range(count)
    ]
