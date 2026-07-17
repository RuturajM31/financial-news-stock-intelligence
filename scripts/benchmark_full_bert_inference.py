"""Validate and benchmark the saved Full BERT sentiment artifact."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

import psutil
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from financial_news_intelligence.models.full_bert_inference import (  # noqa: E402
    LABEL_ORDER,
    analyze_article,
    load_bert_runtime,
    predict_sentences,
)


SENTENCES = [
    "The company reported record revenue and raised its full-year guidance.",
    "The firm issued a profit warning after demand declined sharply.",
    "The company will publish its quarterly results next Thursday.",
]


def main() -> None:
    """Benchmark the saved Full BERT runtime and write the current-run report."""

    selected_device = "cuda" if torch.cuda.is_available() else "cpu"
    current_process = psutil.Process(os.getpid())
    initial_rss_bytes = current_process.memory_info().rss
    if selected_device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    load_start_time = time.perf_counter()
    full_bert_runtime = load_bert_runtime(device=selected_device)
    model_load_seconds = time.perf_counter() - load_start_time
    sentence_predictions = predict_sentences(
        full_bert_runtime,
        SENTENCES,
        batch_size=3,
    )
    if selected_device == "cuda":
        torch.cuda.synchronize()
    warm_start_time = time.perf_counter()
    predict_sentences(full_bert_runtime, SENTENCES, batch_size=3)
    if selected_device == "cuda":
        torch.cuda.synchronize()
    warm_batch_seconds = time.perf_counter() - warm_start_time
    article_result = analyze_article(
        full_bert_runtime,
        SENTENCES[0],
        " ".join(SENTENCES[1:]),
        batch_size=3,
    )
    benchmark_payload = {
        "model_directory": full_bert_runtime.model_dir,
        "model_artifact_size_bytes": sum(
            artifact_file.stat().st_size
            for artifact_file in Path(full_bert_runtime.model_dir).iterdir()
            if artifact_file.is_file()
        ),
        "label_order": list(LABEL_ORDER),
        "device": selected_device,
        "device_name": (
            torch.cuda.get_device_name(0)
            if selected_device == "cuda"
            else "CPU"
        ),
        "model_load_seconds": model_load_seconds,
        "warm_batch_seconds": warm_batch_seconds,
        "warm_sentence_milliseconds": (
            warm_batch_seconds * 1000 / len(SENTENCES)
        ),
        "process_rss_increase_bytes": (
            current_process.memory_info().rss - initial_rss_bytes
        ),
        "cuda_peak_allocated_bytes": (
            torch.cuda.max_memory_allocated()
            if selected_device == "cuda"
            else None
        ),
        "predictions": [
            {
                "text": sentence_prediction.text,
                "predicted_label": sentence_prediction.label,
                "probabilities": sentence_prediction.probabilities,
            }
            for sentence_prediction in sentence_predictions
        ],
        "article_aggregation_smoke_test": {
            "predicted_label": article_result.label,
            "sentence_count": len(article_result.sentences),
        },
    }
    output_path = (
        ROOT / "reports" / "metrics" / "bert_sentiment_current_run_benchmark.json"
    )
    output_path.write_text(
        json.dumps(benchmark_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(benchmark_payload, indent=2))


if __name__ == "__main__":
    main()
