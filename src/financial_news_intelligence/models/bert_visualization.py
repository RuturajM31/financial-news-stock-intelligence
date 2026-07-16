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
    """Build submitted-token counts and sentence-class distributions."""

    counts: Counter[str] = Counter()
    class_counts: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[str, str] = {}
    display_case: dict[str, str] = {}
    for prediction, pieces in zip(sentence_predictions, tokenized_sentences):
        source_words = {
            re.sub(r"[^a-z0-9]", "", word.lower()): word.strip(".,;:!?()[]{}\"'")
            for word in prediction.text.split()
        }
        for token in merge_wordpieces(pieces):
            normalized = re.sub(r"[^a-z0-9-]", "", token.lower())
            if not normalized or normalized in STOP_WORDS or not re.search(r"[a-z0-9]", normalized):
                continue
            counts[normalized] += 1
            class_counts[normalized][prediction.label] += 1
            examples.setdefault(normalized, prediction.text)
            display_case.setdefault(normalized, source_words.get(normalized, normalized))
    rows = []
    for token, count in counts.most_common(36):
        distribution = {label: class_counts[token][label] for label in LABEL_ORDER}
        dominant = max(LABEL_ORDER, key=lambda label: (distribution[label], -LABEL_ORDER.index(label)))
        rows.append(TokenLandscapeItem(display_case[token], count, dominant, distribution, examples[token]))
    return rows


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


def cosine_links(tokens: Iterable[Any], *, maximum_nodes: int = 18, maximum_links: int = 28) -> tuple[list[Any], list[tuple[int, int, float]]]:
    """Return deterministic strongest positive contextual-token similarities."""

    import numpy as np

    nodes = [token for token in tokens if token.text.lower() not in STOP_WORDS and re.search(r"[A-Za-z0-9]", token.text)][:maximum_nodes]
    if len(nodes) < 2:
        return nodes, []
    matrix = np.asarray([node.embedding for node in nodes], dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized = matrix / np.maximum(norms, 1e-12)
    similarities = normalized @ normalized.T
    candidates = [
        (i, j, float(similarities[i, j]))
        for i in range(len(nodes)) for j in range(i + 1, len(nodes))
        if math.isfinite(similarities[i, j]) and similarities[i, j] > 0
    ]
    candidates.sort(key=lambda item: (-item[2], item[0], item[1]))
    selected = candidates[:maximum_links]
    return nodes, selected


def circular_positions(count: int) -> list[tuple[float, float]]:
    """Return stable positions for a compact contextual-token network."""

    if count <= 0:
        return []
    return [
        (math.cos(2 * math.pi * index / count), math.sin(2 * math.pi * index / count))
        for index in range(count)
    ]
