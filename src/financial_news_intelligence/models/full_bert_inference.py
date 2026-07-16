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
    """Return required inference artifacts missing from a model directory."""

    model_directory = Path(model_dir)
    required_tokenizer_files = (
        "config.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt",
    )
    missing_artifacts = [
        artifact_name
        for artifact_name in required_tokenizer_files
        if not (model_directory / artifact_name).is_file()
    ]
    supported_weight_files = ("model.safetensors", "pytorch_model.bin")
    model_weights_are_available = any(
        (model_directory / weight_file).is_file()
        for weight_file in supported_weight_files
    )
    if not model_weights_are_available:
        missing_artifacts.append("model.safetensors or pytorch_model.bin")
    return tuple(missing_artifacts)


def _optional_streamlit_secrets() -> Mapping[str, Any]:
    """Read Streamlit secrets when available without requiring Streamlit."""

    try:
        # Keep Streamlit lazy so scripts and non-analysis pages stay lightweight.
        import streamlit as st

        return st.secrets.to_dict()
    except Exception:
        # This boundary intentionally treats non-Streamlit contexts as no secrets.
        return {}


def _setting(
    name: str,
    environ: Mapping[str, str],
    secrets: Mapping[str, Any],
) -> str:
    """Resolve one setting with environment variables taking priority."""

    raw_setting = environ.get(name) or secrets.get(name, "")
    if raw_setting is None:
        return ""
    return str(raw_setting).strip()


def resolve_model_directory(
    local_model_dir: str | Path = DEFAULT_MODEL_DIR,
    *,
    environ: Mapping[str, str] | None = None,
    secrets: Mapping[str, Any] | None = None,
    snapshot_download: Callable[..., str] | None = None,
) -> Path:
    """Prefer a complete local model, otherwise retrieve the private snapshot."""

    # Local artifacts avoid a download and preserve the development workflow.
    local_model_directory = Path(local_model_dir).expanduser().resolve()
    if not validate_model_snapshot(local_model_directory):
        return local_model_directory

    environment_settings = os.environ if environ is None else environ
    secret_settings = (
        _optional_streamlit_secrets() if secrets is None else secrets
    )
    remote_model_repository = (
        _setting("HF_MODEL_REPO", environment_settings, secret_settings)
        or DEFAULT_REMOTE_MODEL_REPO
    )
    hugging_face_token = _setting(
        "HF_TOKEN",
        environment_settings,
        secret_settings,
    )
    if not hugging_face_token:
        raise RuntimeError(
            "Full BERT is unavailable because HF_TOKEN is not configured "
            "for the private model repository."
        )

    try:
        if snapshot_download is None:
            # Hugging Face Hub stays lazy because the local model has priority.
            from huggingface_hub import snapshot_download as hub_download

            snapshot_download = hub_download

        downloaded_model_directory = snapshot_download(
            repo_id=remote_model_repository,
            token=hugging_face_token,
            allow_patterns=[
                "config.json", "model.safetensors", "pytorch_model.bin",
                "tokenizer.json", "tokenizer_config.json", "vocab.txt",
                "special_tokens_map.json",
            ],
        )
    except Exception as error:
        # Report actionable context without exposing the private token.
        raise RuntimeError(
            "Full BERT could not be retrieved. Check the private repository, "
            "READ token, and deployment network access."
        ) from error

    downloaded_model_snapshot = (
        Path(downloaded_model_directory).expanduser().resolve()
    )
    missing_artifacts = validate_model_snapshot(downloaded_model_snapshot)
    if missing_artifacts:
        missing_list = ", ".join(missing_artifacts)
        raise RuntimeError(
            "The downloaded Full BERT snapshot is incomplete. Missing: "
            + missing_list
        )
    return downloaded_model_snapshot


@lru_cache(maxsize=4)
def load_bert_runtime(
    model_dir: str | Path | None = None,
    device: str | None = None,
) -> BertRuntime:
    """Load and cache Full BERT from the local artifact or private snapshot."""

    # Heavy imports stay lazy so pages without inference do not load PyTorch.
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    requested_model_directory = (
        DEFAULT_MODEL_DIR if model_dir is None else model_dir
    )
    model_source = resolve_model_directory(requested_model_directory)
    resolved_device = device or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        local_files_only=True,
        use_fast=True,
    )
    full_bert_model = AutoModelForSequenceClassification.from_pretrained(
        model_source,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    actual_label_order = tuple(
        full_bert_model.config.id2label[class_index]
        for class_index in range(len(LABEL_ORDER))
    )
    if actual_label_order != LABEL_ORDER:
        raise ValueError(
            f"Unexpected model label order: {actual_label_order!r}"
        )

    full_bert_model.to(resolved_device)
    # Evaluation mode disables training-only behavior such as dropout.
    full_bert_model.eval()
    return BertRuntime(
        tokenizer,
        full_bert_model,
        resolved_device,
        str(model_source),
    )

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


def predict_sentences(
    runtime: BertRuntime,
    sentences: list[str],
    *,
    batch_size: int = 16,
) -> tuple[SentencePrediction, ...]:
    """Predict ordered sentences and retain final-layer CLS representations."""

    # PyTorch remains lazy because inference is requested only on Analyze.
    import torch

    sentence_predictions: list[SentencePrediction] = []

    for sentence_batch in _chunks(sentences, batch_size):
        # Dynamic padding limits each batch to its longest sentence. Truncation
        # enforces the model's verified maximum sequence length of 128 tokens.
        tokenized_sentence_batch = runtime.tokenizer(
            sentence_batch,
            padding=True,
            truncation=True,
            max_length=MAX_SEQUENCE_LENGTH,
            return_tensors="pt",
        )
        device_sentence_batch = {
            input_name: input_tensor.to(runtime.device)
            for input_name, input_tensor in tokenized_sentence_batch.items()
        }

        # Inference mode avoids gradient tensors and preserves evaluation output.
        with torch.inference_mode():
            model_output = runtime.model(
                **device_sentence_batch,
                output_hidden_states=True,
                return_dict=True,
            )
            sentence_class_scores = torch.softmax(
                model_output.logits,
                dim=-1,
            ).cpu().tolist()
            sentence_embeddings = (
                model_output.hidden_states[-1][:, 0, :]
                .float()
                .cpu()
                .tolist()
            )

        for sentence, class_scores, sentence_embedding in zip(
            sentence_batch,
            sentence_class_scores,
            sentence_embeddings,
        ):
            class_score_map = {
                class_label: float(class_scores[class_index])
                for class_index, class_label in enumerate(LABEL_ORDER)
            }
            predicted_sentiment = max(
                LABEL_ORDER,
                key=class_score_map.get,
            )
            sentence_predictions.append(
                SentencePrediction(
                    sentence,
                    predicted_sentiment,
                    class_score_map,
                    class_score_map[predicted_sentiment],
                    tuple(float(value) for value in sentence_embedding),
                )
            )

    return tuple(sentence_predictions)


def analyze_article(
    runtime: BertRuntime,
    headline: str,
    body: str,
    *,
    batch_size: int = 16,
) -> ArticlePrediction:
    """Average sentence class scores into one article-level prediction."""

    article_sentences = split_article_sentences(headline, body)
    if not article_sentences:
        raise ValueError(
            "The article does not contain a meaningful sentence to analyze."
        )

    inference_started_at = time.perf_counter()
    sentence_predictions = predict_sentences(
        runtime,
        article_sentences,
        batch_size=batch_size,
    )

    # Full BERT alone determines the result. Lexical cues are rendered
    # separately and never enter this arithmetic-mean aggregation.
    article_class_scores = {
        class_label: sum(
            prediction.probabilities[class_label]
            for prediction in sentence_predictions
        )
        / len(sentence_predictions)
        for class_label in LABEL_ORDER
    }
    predicted_sentiment = max(
        LABEL_ORDER,
        key=article_class_scores.get,
    )
    strongest_sentences_by_label = {
        class_label: tuple(
            sorted(
                sentence_predictions,
                key=lambda prediction: prediction.probabilities[class_label],
                reverse=True,
            )[:3]
        )
        for class_label in LABEL_ORDER
    }

    return ArticlePrediction(
        predicted_sentiment,
        article_class_scores,
        article_class_scores[predicted_sentiment],
        sentence_predictions,
        strongest_sentences_by_label,
        time.perf_counter() - inference_started_at,
        runtime.device,
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