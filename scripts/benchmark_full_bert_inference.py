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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    process = psutil.Process(os.getpid())
    rss_before = process.memory_info().rss
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    runtime = load_bert_runtime(device=device)
    load_seconds = time.perf_counter() - load_started
    predictions = predict_sentences(runtime, SENTENCES, batch_size=3)
    if device == "cuda":
        torch.cuda.synchronize()
    warm_started = time.perf_counter()
    predict_sentences(runtime, SENTENCES, batch_size=3)
    if device == "cuda":
        torch.cuda.synchronize()
    warm_seconds = time.perf_counter() - warm_started
    article = analyze_article(runtime, SENTENCES[0], " ".join(SENTENCES[1:]), batch_size=3)
    payload = {
        "model_directory": runtime.model_dir,
        "model_artifact_size_bytes": sum(item.stat().st_size for item in Path(runtime.model_dir).iterdir() if item.is_file()),
        "label_order": list(LABEL_ORDER),
        "device": device,
        "device_name": torch.cuda.get_device_name(0) if device == "cuda" else "CPU",
        "model_load_seconds": load_seconds,
        "warm_batch_seconds": warm_seconds,
        "warm_sentence_milliseconds": warm_seconds * 1000 / len(SENTENCES),
        "process_rss_increase_bytes": process.memory_info().rss - rss_before,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
        "predictions": [
            {
                "text": item.text,
                "predicted_label": item.label,
                "probabilities": item.probabilities,
            }
            for item in predictions
        ],
        "article_aggregation_smoke_test": {
            "predicted_label": article.label,
            "sentence_count": len(article.sentences),
        },
    }
    output = ROOT / "reports" / "metrics" / "bert_sentiment_current_run_benchmark.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
