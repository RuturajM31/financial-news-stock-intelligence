"""Reusable sentence and article inference for the trained Full BERT model."""



from __future__ import annotations

import os

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re
import time
from typing import Any, Callable, Iterable, Mapping


LABEL_ORDER = ("Bearish", "Neutral", "Bullish")
MAX_SEQUENCE_LENGTH = 128
DEFAULT_REMOTE_MODEL_REPO = "ruturajmokashi/financial-news-full-bert"
DEFAULT_MODEL_DIR = (
    Path(__file__).resolve().parents[3]
    / "artifacts" / "models" / "bert_sentiment" / "final_model"
)


@dataclass(frozen=True)
class SentencePrediction:
    text: str
    label: str
    probabilities: dict[str, float]
    confidence: float
    embedding: tuple[float, ...] = ()


@dataclass(frozen=True)
class ArticlePrediction:
    label: str
    probabilities: dict[str, float]
    confidence: float
    sentences: tuple[SentencePrediction, ...]
    strongest_by_label: dict[str, tuple[SentencePrediction, ...]]
    inference_seconds: float
    device: str


@dataclass(frozen=True)
class ContextualToken:
    text: str
    embedding: tuple[float, ...]


@dataclass(frozen=True)
class BertRuntime:
    tokenizer: Any
    model: Any
    device: str
    model_dir: str


def validate_model_snapshot(model_dir: str | Path) -> tuple[str, ...]:
    path = Path(model_dir)
    missing = [n for n in ("config.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt") if not (path / n).is_file()]
    if not any((path / n).is_file() for n in ("model.safetensors", "pytorch_model.bin")):
        missing.append("model.safetensors or pytorch_model.bin")
    return tuple(missing)


def _optional_streamlit_secrets() -> Mapping[str, Any]:
    try:
        import streamlit as st
        return st.secrets.to_dict()
    except Exception:
        return {}


def _setting(name: str, environ: Mapping[str, str], secrets: Mapping[str, Any]) -> str:
    value = environ.get(name) or secrets.get(name, "")
    return str(value).strip() if value is not None else ""


def resolve_model_directory(
    local_model_dir: str | Path = DEFAULT_MODEL_DIR, *,
    environ: Mapping[str, str] | None = None,
    secrets: Mapping[str, Any] | None = None,
    snapshot_download: Callable[..., str] | None = None,
) -> Path:
    """Prefer complete local artifacts, otherwise retrieve a private Hub snapshot."""
    local_path = Path(local_model_dir).expanduser().resolve()
    if not validate_model_snapshot(local_path):
        return local_path
    env = os.environ if environ is None else environ
    secret_values = _optional_streamlit_secrets() if secrets is None else secrets
    repo_id = _setting("HF_MODEL_REPO", env, secret_values) or DEFAULT_REMOTE_MODEL_REPO
    token = _setting("HF_TOKEN", env, secret_values)
    if not token:
        raise RuntimeError("Full BERT is unavailable because HF_TOKEN is not configured for the private model repository.")
    try:
        if snapshot_download is None:
            from huggingface_hub import snapshot_download as hub_download
            snapshot_download = hub_download
        downloaded = snapshot_download(
            repo_id=repo_id, token=token,
            allow_patterns=["config.json", "model.safetensors", "pytorch_model.bin", "tokenizer.json", "tokenizer_config.json", "vocab.txt", "special_tokens_map.json"],
        )
    except Exception as error:
        raise RuntimeError("Full BERT could not be retrieved. Check the private repository, READ token, and deployment network access.") from error
    snapshot = Path(downloaded).expanduser().resolve()
    missing = validate_model_snapshot(snapshot)
    if missing:
        raise RuntimeError("The downloaded Full BERT snapshot is incomplete. Missing: " + ", ".join(missing))
    return snapshot


@lru_cache(maxsize=4)
def load_bert_runtime(model_dir: str | Path | None = None, device: str | None = None) -> BertRuntime:
    """Load and cache Full BERT from the local artifact or private Hub snapshot."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    path = resolve_model_directory(DEFAULT_MODEL_DIR if model_dir is None else model_dir)
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True, low_cpu_mem_usage=True)
    actual_mapping = tuple(model.config.id2label[index] for index in range(len(LABEL_ORDER)))
    if actual_mapping != LABEL_ORDER:
        raise ValueError(f"Unexpected model label order: {actual_mapping!r}")
    model.to(resolved_device)
    model.eval()
    return BertRuntime(tokenizer, model, resolved_device, str(path))

def split_article_sentences(headline: str, body: str) -> list[str]:
    """Keep the headline and meaningful article sentences in their original order."""

    candidates: list[str] = []
    if headline.strip():
        candidates.append(re.sub(r"\s+", " ", headline).strip())
    for part in re.split(r"(?<=[.!?])\s+|[\r\n]+", body):
        sentence = re.sub(r"\s+", " ", part).strip(" \t-•")
        if len(sentence.split()) < 4 or len(sentence) < 24:
            continue
        lowered = sentence.lower()
        if any(fragment in lowered for fragment in (
            "accept cookies", "privacy policy", "all rights reserved", "subscribe to our newsletter",
        )):
            continue
        candidates.append(sentence)
    return candidates


def _chunks(items: list[str], batch_size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def predict_sentences(runtime: BertRuntime, sentences: list[str], *, batch_size: int = 16) -> tuple[SentencePrediction, ...]:
    """Predict ordered sentences and retain final-layer CLS representations."""

    import torch

    predictions: list[SentencePrediction] = []
    for batch in _chunks(sentences, batch_size):
        encoded = runtime.tokenizer(
            batch, padding=True, truncation=True,
            max_length=MAX_SEQUENCE_LENGTH, return_tensors="pt",
        )
        encoded = {key: value.to(runtime.device) for key, value in encoded.items()}
        with torch.inference_mode():
            output = runtime.model(**encoded, output_hidden_states=True, return_dict=True)
            probabilities = torch.softmax(output.logits, dim=-1).cpu().tolist()
            cls_embeddings = output.hidden_states[-1][:, 0, :].float().cpu().tolist()
        for sentence, scores, embedding in zip(batch, probabilities, cls_embeddings):
            score_map = {label: float(scores[index]) for index, label in enumerate(LABEL_ORDER)}
            label = max(LABEL_ORDER, key=score_map.get)
            predictions.append(SentencePrediction(
                sentence, label, score_map, score_map[label], tuple(float(value) for value in embedding)
            ))
    return tuple(predictions)


def analyze_article(runtime: BertRuntime, headline: str, body: str, *, batch_size: int = 16) -> ArticlePrediction:
    """Aggregate arithmetic-mean sentence softmax scores into the article result."""

    sentences = split_article_sentences(headline, body)
    if not sentences:
        raise ValueError("The article does not contain a meaningful sentence to analyze.")
    started = time.perf_counter()
    sentence_predictions = predict_sentences(runtime, sentences, batch_size=batch_size)
    means = {
        label: sum(item.probabilities[label] for item in sentence_predictions) / len(sentence_predictions)
        for label in LABEL_ORDER
    }
    label = max(LABEL_ORDER, key=means.get)
    strongest = {
        current: tuple(sorted(
            sentence_predictions, key=lambda item: item.probabilities[current], reverse=True
        )[:3])
        for current in LABEL_ORDER
    }
    return ArticlePrediction(
        label, means, means[label], sentence_predictions, strongest,
        time.perf_counter() - started, runtime.device,
    )


def wordpiece_tokens(runtime: BertRuntime, text: str, *, include_special_tokens: bool = True) -> list[str]:
    """Return readable ordered WordPiece inputs for one sentence."""

    encoded = runtime.tokenizer(
        text, truncation=True, max_length=MAX_SEQUENCE_LENGTH,
        add_special_tokens=include_special_tokens,
    )
    return runtime.tokenizer.convert_ids_to_tokens(encoded["input_ids"])


def deterministic_pca(embeddings: list[tuple[float, ...]]) -> list[tuple[float, float]]:
    """Project sentence representations with deterministic PCA and canonical signs."""

    import numpy as np
    from sklearn.decomposition import PCA

    if not embeddings:
        return []
    if len(embeddings) == 1:
        return [(0.0, 0.0)]
    matrix = np.asarray(embeddings, dtype=np.float64)
    components = min(2, matrix.shape[0], matrix.shape[1])
    pca = PCA(n_components=components, svd_solver="full")
    projected = pca.fit_transform(matrix)
    for column in range(components):
        loading = pca.components_[column]
        pivot = int(np.argmax(np.abs(loading)))
        if loading[pivot] < 0:
            projected[:, column] *= -1
    if components == 1:
        projected = np.column_stack([projected[:, 0], np.zeros(len(projected))])
    return [(float(x), float(y)) for x, y in projected[:, :2]]


def contextual_tokens(runtime: BertRuntime, text: str) -> tuple[ContextualToken, ...]:
    """Compute merged final-layer contextual token representations for one sentence."""

    import torch

    encoded = runtime.tokenizer(
        text, truncation=True, max_length=MAX_SEQUENCE_LENGTH,
        return_tensors="pt", return_offsets_mapping=True,
    )
    offsets = encoded.pop("offset_mapping")[0].tolist()
    model_inputs = {key: value.to(runtime.device) for key, value in encoded.items()}
    with torch.inference_mode():
        output = runtime.model(**model_inputs, output_hidden_states=True, return_dict=True)
    vectors = output.hidden_states[-1][0].float().cpu()
    tokens: list[ContextualToken] = []
    current_text = ""
    current_vectors: list[Any] = []
    previous_end = -1

    def flush() -> None:
        nonlocal current_text, current_vectors
        readable = current_text.strip()
        if readable and re.search(r"[A-Za-z0-9]", readable):
            vector = torch.stack(current_vectors).mean(dim=0).tolist()
            tokens.append(ContextualToken(readable, tuple(float(value) for value in vector)))
        current_text, current_vectors = "", []

    for (start, end), vector in zip(offsets, vectors):
        if start == end:
            continue
        piece = text[start:end]
        if current_vectors and start > previous_end:
            flush()
        current_text += piece
        current_vectors.append(vector)
        previous_end = end
    flush()
    return tuple(tokens)