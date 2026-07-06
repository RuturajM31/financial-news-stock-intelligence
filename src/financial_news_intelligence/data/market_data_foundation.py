"""Build the final verified SEC and Tiingo market-data foundation.

Purpose
-------
Orchestrate the already-qualified, authenticated data path required before the
stock-movement model may run. The pipeline combines official SEC disclosure
events, private Tiingo adjusted daily prices, the existing local sentiment
champion, strictly future market-session mapping, fixed movement labels, QA,
and checksummed manifests.

Inputs and provenance
---------------------
1. ``data/reference/company_tickers.csv`` supplies curated company names.
2. ``reports/provider_qualification/tiingo/`` supplies the passed ten-ticker
   technical qualification summary.
3. SEC EDGAR supplies official filing acceptance timestamps and document URLs.
4. Tiingo EOD supplies authenticated internal-use adjusted OHLCV evidence.
5. The existing local deployment champion supplies sentiment probabilities.
6. ``TIINGO_API_TOKEN`` is read from the process environment only.

Dataset grain, joins, and formulas
----------------------------------
- Event grain: one unique ticker and SEC document URL.
- Price grain: one ticker and trading session.
- Final news evidence grain: one SEC event mapped to one target session.
- Join key: canonical ticker plus the first session open strictly after the SEC
  acceptance timestamp.
- Reaction return: ``target_close / previous_close - 1`` using adjusted closes.
- Movement label: Down below -0.5%, Flat inside +/-0.5%, Up above +0.5%.

Outputs and downstream use
--------------------------
The compatibility outputs consumed by the stock-movement package are:
- ``data/processed/news_sentiment_evidence.csv``;
- ``data/processed/market_price_evidence.csv``.

Rejected rows, QA evidence, and a checksum manifest are also produced. Raw
Tiingo responses remain under ``data/private/tiingo_eod`` with owner-only
permissions and are excluded from public outputs.

Safety, leakage, and deployment boundaries
------------------------------------------
No ticker, timestamp, source, price, company fact, or market session is guessed.
The package refuses to run without the passed ten-ticker qualification summary.
Historical Tiingo response checksums must match the qualified responses. Event
observations must precede target-session opens. Yahoo/yfinance remains optional
secondary-only and is not used as sole authority. The package never trains the
movement model and never changes FastAPI, Streamlit, Docker, Kubernetes, or
public deployment files.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from financial_news_intelligence.data.foundation_market_sessions import (
    map_articles_to_sessions,
)
from financial_news_intelligence.data.foundation_sec_events import (
    collect_sec_events,
)
from financial_news_intelligence.data.foundation_tiingo_prices import (
    collect_tiingo_prices,
)
from financial_news_intelligence.data.foundation_ticker_resolution import (
    load_ticker_reference,
)

LABEL_ORDER = ("Down", "Flat", "Up")
SENTIMENT_ORDER = ("Bearish", "Neutral", "Bullish")
REQUIRED_TICKERS = (
    "NVDA",
    "AAPL",
    "MSFT",
    "AMZN",
    "GOOGL",
    "META",
    "TSLA",
    "NFLX",
    "AMD",
    "INTC",
)

NEWS_OUTPUT = Path("data/processed/news_sentiment_evidence.csv")
PRICE_OUTPUT = Path("data/processed/market_price_evidence.csv")
REJECTED_OUTPUT = Path(
    "data/processed/market_data_foundation_rejected_rows.csv"
)
QA_OUTPUT = Path("reports/qa/market_data_foundation_qa.json")
MANIFEST_OUTPUT = Path(
    "artifacts/manifests/market_data_foundation_manifest.json"
)
QUALIFICATION_SUMMARY = Path(
    "reports/provider_qualification/tiingo/"
    "tiingo_eod_qualification_summary.json"
)
SEC_CACHE_DIRECTORY = Path("data/cache/market_data_foundation/sec")
TIINGO_CACHE_DIRECTORY = Path("data/private/tiingo_eod")


class FoundationError(RuntimeError):
    """Raised when foundation evidence fails provenance or readiness gates."""


@dataclass(frozen=True)
class FoundationConfig:
    """Store the fixed, already-qualified historical experiment contract."""

    event_start_date: date = date(2015, 2, 2)
    event_end_date: date = date(2020, 3, 31)
    price_start_date: date = date(2015, 1, 1)
    price_end_date: date = date(2020, 4, 1)
    minimum_price_rows_per_ticker: int = 1000
    flat_threshold: float = 0.005
    minimum_articles: int = 300
    minimum_sessions: int = 300
    sentiment_batch_size: int = 16
    refresh_sec_cache: bool = False
    refresh_tiingo_cache: bool = False


def sha256(file_path: Path) -> str:
    """Return one safe regular file's hexadecimal SHA-256 checksum."""

    if not file_path.exists() or file_path.is_symlink() or not file_path.is_file():
        raise FoundationError(f"Missing or unsafe checksum source: {file_path}")
    digest = hashlib.sha256()
    with file_path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _project_path(project_root: Path, relative_path: Path) -> Path:
    """Resolve one project-controlled path without allowing traversal."""

    root = project_root.resolve()
    target = (root / relative_path).resolve()
    if target != root and root not in target.parents:
        raise FoundationError(f"Path escapes project root: {relative_path}")
    return target


def _load_json(file_path: Path, description: str) -> dict[str, Any]:
    """Load one required UTF-8 JSON object from a safe regular file."""

    if not file_path.exists() or file_path.is_symlink() or not file_path.is_file():
        raise FoundationError(f"Missing or unsafe {description}: {file_path}")
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FoundationError(f"Invalid {description}: {exc}") from exc
    if not isinstance(payload, dict):
        raise FoundationError(f"{description} must contain a JSON object.")
    return payload


def _sentiment_model_contract(
    project_root: Path,
) -> tuple[Path, tuple[str, ...], dict[str, Any]]:
    """Resolve the saved deployment champion without changing deployment."""

    champion = _load_json(
        _project_path(
            project_root,
            Path("artifacts/manifests/sentiment_model_champion.json"),
        ),
        "sentiment champion manifest",
    )
    comparison = _load_json(
        _project_path(
            project_root,
            Path("reports/metrics/sentiment_model_comparison.json"),
        ),
        "sentiment comparison report",
    )
    model_key = champion.get("recommended_deployment_model")
    if not isinstance(model_key, str) or not model_key:
        raise FoundationError("Champion manifest has no deployment model key.")
    if comparison.get("deployment_champion") != model_key:
        raise FoundationError("Sentiment champion files disagree.")

    models = comparison.get("models")
    if not isinstance(models, list):
        raise FoundationError("Sentiment comparison has no model evidence.")
    selected = next(
        (
            model
            for model in models
            if isinstance(model, dict) and model.get("model_key") == model_key
        ),
        None,
    )
    if not isinstance(selected, dict):
        raise FoundationError("Deployment model evidence is missing.")

    directory_value = selected.get("final_model_directory")
    if not isinstance(directory_value, str) or not directory_value:
        raise FoundationError("Deployment model directory is missing.")
    model_directory = Path(directory_value).expanduser().resolve()
    root = project_root.resolve()
    if root not in model_directory.parents:
        raise FoundationError("Deployment model directory is outside project.")
    if (
        not model_directory.exists()
        or model_directory.is_symlink()
        or not model_directory.is_dir()
    ):
        raise FoundationError(
            f"Deployment model directory is missing or unsafe: {model_directory}"
        )
    if comparison.get("label_order") != list(SENTIMENT_ORDER):
        raise FoundationError("Sentiment label order changed.")
    return model_directory, SENTIMENT_ORDER, selected


def predict_sentiment_local(
    texts: Sequence[str],
    model_directory: Path,
    label_order: Sequence[str] = SENTIMENT_ORDER,
    batch_size: int = 16,
) -> pd.DataFrame:
    """Run deterministic local sentiment inference in bounded batches."""

    if not texts:
        raise FoundationError("Sentiment inference requires text.")
    if batch_size < 1:
        raise FoundationError("Sentiment batch size must be positive.")
    if tuple(label_order) != SENTIMENT_ORDER:
        raise FoundationError("Unexpected sentiment label order.")

    try:
        import torch
        from transformers import AutoModelForSequenceClassification
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise FoundationError(
            "Local sentiment inference dependencies are unavailable."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(
        model_directory,
        local_files_only=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_directory,
        local_files_only=True,
    )
    model.eval()

    probability_parts: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = [str(value) for value in texts[start : start + batch_size]]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            logits = model(**encoded).logits
            probability_parts.append(torch.softmax(logits, dim=-1).cpu().numpy())

    values = np.vstack(probability_parts)
    if values.shape != (len(texts), len(SENTIMENT_ORDER)):
        raise FoundationError(f"Unexpected sentiment output shape: {values.shape}")
    positions = values.argmax(axis=1)
    return pd.DataFrame(
        {
            "prob_bearish": values[:, 0],
            "prob_neutral": values[:, 1],
            "prob_bullish": values[:, 2],
            "sentiment_label": [SENTIMENT_ORDER[index] for index in positions],
        }
    )


def add_movement_labels(
    mapped: pd.DataFrame,
    flat_threshold: float,
) -> pd.DataFrame:
    """Add fixed Down, Flat, and Up labels from verified reaction returns."""

    if not 0.0 < flat_threshold < 0.25:
        raise FoundationError("Flat threshold must be between 0 and 0.25.")
    if "reaction_return" not in mapped.columns or mapped.empty:
        raise FoundationError("Mapped evidence has no reaction returns.")
    result = mapped.copy()
    result["movement_label"] = np.select(
        [
            result["reaction_return"].lt(-flat_threshold),
            result["reaction_return"].gt(flat_threshold),
        ],
        ["Down", "Up"],
        default="Flat",
    )
    return result


def _readiness_report(
    evidence: pd.DataFrame,
    minimum_articles: int,
    minimum_sessions: int,
) -> dict[str, Any]:
    """Prove that purged chronological model splits can contain all classes."""

    if len(evidence) < minimum_articles:
        raise FoundationError(
            f"Foundation needs {minimum_articles} events; accepted {len(evidence)}."
        )
    session_frame = evidence[
        ["ticker", "target_session_date", "movement_label"]
    ].drop_duplicates()
    conflicts = session_frame.groupby(
        ["ticker", "target_session_date"],
        observed=True,
    )["movement_label"].nunique()
    if conflicts.gt(1).any():
        raise FoundationError("One ticker-session has conflicting labels.")

    dates = np.array(
        sorted(pd.to_datetime(session_frame["target_session_date"]).unique())
    )
    if len(dates) < minimum_sessions:
        raise FoundationError(
            f"Foundation needs {minimum_sessions} sessions; accepted {len(dates)}."
        )

    train_end = max(1, int(len(dates) * 0.70))
    validation_size = max(1, int(len(dates) * 0.15))
    validation_start = train_end + 1
    validation_end = validation_start + validation_size
    test_start = validation_end + 1
    if test_start >= len(dates):
        raise FoundationError("Purged readiness leaves no test block.")

    boundaries = {
        "train": dates[:train_end],
        "validation": dates[validation_start:validation_end],
        "test": dates[test_start:],
    }
    split_report: dict[str, Any] = {}
    for name, split_dates in boundaries.items():
        labels = session_frame[
            pd.to_datetime(session_frame["target_session_date"]).isin(split_dates)
        ]["movement_label"]
        class_counts = {
            label: int((labels == label).sum()) for label in LABEL_ORDER
        }
        if any(value == 0 for value in class_counts.values()):
            raise FoundationError(
                f"{name} readiness lacks a movement class: {class_counts}"
            )
        split_report[name] = {
            "session_dates": int(len(split_dates)),
            "class_counts": class_counts,
            "start": str(pd.Timestamp(split_dates[0]).date()),
            "end": str(pd.Timestamp(split_dates[-1]).date()),
        }

    return {
        "ready_for_stock_movement_package": True,
        "article_records": int(len(evidence)),
        "unique_session_dates": int(len(dates)),
        "movement_class_counts": {
            label: int((evidence["movement_label"] == label).sum())
            for label in LABEL_ORDER
        },
        "simulated_purged_splits": split_report,
    }


def _atomic_csv(file_path: Path, frame: pd.DataFrame) -> None:
    """Write one owner-only CSV atomically."""

    file_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = file_path.with_suffix(file_path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.chmod(temporary, 0o600)
    temporary.replace(file_path)


def _atomic_json(file_path: Path, payload: Mapping[str, Any]) -> None:
    """Write one owner-only JSON object atomically."""

    file_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = file_path.with_suffix(file_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    os.chmod(temporary, 0o600)
    temporary.replace(file_path)


def _artifact_entry(project_root: Path, relative_path: Path) -> dict[str, Any]:
    """Create one manifest entry for a completed output file."""

    file_path = _project_path(project_root, relative_path)
    return {
        "path": relative_path.as_posix(),
        "sha256": sha256(file_path),
        "size_bytes": file_path.stat().st_size,
        "mode": oct(file_path.stat().st_mode & 0o777),
    }


def _private_cache_inventory(project_root: Path) -> list[dict[str, Any]]:
    """Inventory private Tiingo caches without reading values into reports."""

    cache_directory = _project_path(project_root, TIINGO_CACHE_DIRECTORY)
    entries: list[dict[str, Any]] = []
    for file_path in sorted(cache_directory.glob("*.json")):
        if file_path.is_symlink() or not file_path.is_file():
            raise FoundationError(f"Unsafe private cache file: {file_path}")
        if file_path.stat().st_mode & 0o077:
            raise FoundationError(f"Private cache is not owner-only: {file_path}")
        entries.append(
            {
                "path": file_path.relative_to(project_root).as_posix(),
                "sha256": sha256(file_path),
                "size_bytes": file_path.stat().st_size,
                "mode": oct(file_path.stat().st_mode & 0o777),
            }
        )
    if len(entries) != len(REQUIRED_TICKERS):
        raise FoundationError(
            "Private Tiingo cache must contain exactly ten ticker responses."
        )
    return entries


def build_foundation(
    project_root: Path,
    api_token: str,
    config: FoundationConfig | None = None,
    sentiment_predictor: Callable[..., pd.DataFrame] | None = None,
    sec_opener: Callable[..., Any] | None = None,
    tiingo_opener: Callable[..., Any] | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Build all outputs only after every provenance and leakage gate passes."""

    root = project_root.resolve()
    active = config or FoundationConfig()
    if active.event_start_date >= active.event_end_date:
        raise FoundationError("Event start date must precede event end date.")
    if active.price_start_date >= active.price_end_date:
        raise FoundationError("Price start date must precede price end date.")
    if active.event_start_date < active.price_start_date:
        raise FoundationError("Event window cannot start before price history.")
    if active.event_end_date >= active.price_end_date:
        raise FoundationError("Price history must extend beyond event history.")

    reference = load_ticker_reference(
        _project_path(root, Path("data/reference/company_tickers.csv"))
    )
    qualification_path = _project_path(root, QUALIFICATION_SUMMARY)
    qualification = _load_json(
        qualification_path,
        "Tiingo qualification summary",
    )
    progress = progress_callback or (lambda message: print(message, flush=True))

    start_utc = datetime.combine(
        active.event_start_date,
        time.min,
        tzinfo=timezone.utc,
    )
    end_utc = datetime.combine(
        active.event_end_date,
        time.max,
        tzinfo=timezone.utc,
    )
    progress("EVENT SOURCE: SEC EDGAR official company disclosures")
    sec_kwargs: dict[str, Any] = {}
    if sec_opener is not None:
        sec_kwargs["opener"] = sec_opener
    news, news_rejected, sec_requests = collect_sec_events(
        reference,
        REQUIRED_TICKERS,
        start_utc,
        end_utc,
        _project_path(root, SEC_CACHE_DIRECTORY),
        refresh_cache=active.refresh_sec_cache,
        progress_callback=progress,
        **sec_kwargs,
    )
    progress(f"SEC EVENTS READY: {len(news)} verified rows")

    progress("PRICE SOURCE: authenticated Tiingo EOD internal-use data")
    tiingo_kwargs: dict[str, Any] = {}
    if tiingo_opener is not None:
        tiingo_kwargs["opener"] = tiingo_opener
    prices, tiingo_requests = collect_tiingo_prices(
        qualification,
        REQUIRED_TICKERS,
        active.price_start_date,
        active.price_end_date,
        active.minimum_price_rows_per_ticker,
        api_token,
        _project_path(root, TIINGO_CACHE_DIRECTORY),
        refresh_cache=active.refresh_tiingo_cache,
        progress_callback=progress,
        **tiingo_kwargs,
    )
    progress(f"TIINGO PRICES READY: {len(prices)} ticker-session rows")

    mapped, session_rejected = map_articles_to_sessions(news, prices)
    model_directory, sentiment_labels, model_evidence = (
        _sentiment_model_contract(root)
    )
    predictor = sentiment_predictor or predict_sentiment_local
    progress(f"SENTIMENT INFERENCE: {len(mapped)} mapped events")
    sentiment = predictor(
        mapped["text"].astype(str).tolist(),
        model_directory=model_directory,
        label_order=sentiment_labels,
        batch_size=active.sentiment_batch_size,
    )
    if len(sentiment) != len(mapped):
        raise FoundationError("Sentiment row count differs from mapped events.")

    evidence = pd.concat(
        [mapped.reset_index(drop=True), sentiment.reset_index(drop=True)],
        axis=1,
    )
    evidence = add_movement_labels(evidence, active.flat_threshold)
    probability_columns = [
        "prob_bearish",
        "prob_neutral",
        "prob_bullish",
    ]
    if not np.allclose(evidence[probability_columns].sum(axis=1), 1.0):
        raise FoundationError("Sentiment probabilities do not sum to one.")
    if evidence["article_id"].duplicated().any():
        raise FoundationError("Final article IDs are not unique.")
    if (evidence["published_at_utc"] >= evidence["session_open_utc"]).any():
        raise FoundationError("Future leakage exists in final evidence.")

    readiness = _readiness_report(
        evidence,
        active.minimum_articles,
        active.minimum_sessions,
    )
    news_columns = [
        "article_id",
        "ticker",
        "company",
        "published_at_utc",
        "timestamp_type",
        "text",
        "headline",
        "source_name",
        "source_url",
        "news_provider",
        "provider_role",
        "verification_status",
        "ticker_resolution_method",
        "matched_alias",
        "language",
        "source_country",
        "sentiment_label",
        "prob_bearish",
        "prob_neutral",
        "prob_bullish",
        "target_session_date",
        "previous_session_date",
        "session_open_utc",
        "session_close_utc",
        "previous_close",
        "target_open",
        "target_close",
        "target_volume",
        "reaction_return",
        "movement_label",
        "hours_to_session_open",
        "provenance_note",
    ]
    news_output = evidence[
        [column for column in news_columns if column in evidence.columns]
    ].sort_values(["published_at_utc", "ticker", "article_id"])
    price_output = prices.sort_values(["ticker", "session_date"])

    rejected_parts: list[pd.DataFrame] = []
    for stage, frame in (
        ("sec_ingestion", news_rejected),
        ("session_mapping", session_rejected),
    ):
        if frame.empty:
            continue
        part = frame.copy()
        part.insert(0, "stage", stage)
        rejected_parts.append(part)
    rejected_output = (
        pd.concat(rejected_parts, ignore_index=True, sort=False)
        if rejected_parts
        else pd.DataFrame(columns=["stage", "rejection_reason"])
    )

    news_path = _project_path(root, NEWS_OUTPUT)
    price_path = _project_path(root, PRICE_OUTPUT)
    rejected_path = _project_path(root, REJECTED_OUTPUT)
    qa_path = _project_path(root, QA_OUTPUT)
    manifest_path = _project_path(root, MANIFEST_OUTPUT)
    _atomic_csv(news_path, news_output)
    _atomic_csv(price_path, price_output)
    _atomic_csv(rejected_path, rejected_output)

    private_cache = _private_cache_inventory(root)
    generated_at = datetime.now(timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )
    qa_payload = {
        "status": "foundation_completed",
        "generated_at_utc": generated_at,
        "config": {
            **asdict(active),
            "event_start_date": active.event_start_date.isoformat(),
            "event_end_date": active.event_end_date.isoformat(),
            "price_start_date": active.price_start_date.isoformat(),
            "price_end_date": active.price_end_date.isoformat(),
        },
        "records": {
            "accepted_events": int(len(news_output)),
            "price_rows": int(len(price_output)),
            "rejected_rows": int(len(rejected_output)),
            "ticker_count": int(price_output["ticker"].nunique()),
        },
        "readiness": readiness,
        "sentiment": {
            "deployment_model_key": model_evidence.get("model_key"),
            "model_directory": str(model_directory),
            "label_order": list(sentiment_labels),
        },
        "qualification": {
            "path": QUALIFICATION_SUMMARY.as_posix(),
            "sha256": sha256(qualification_path),
            "all_tickers_passed": qualification.get("all_tickers_passed"),
        },
        "licence_boundary": {
            "tiingo_classification": "internal_use_only",
            "raw_values_publicly_redistributable": False,
            "raw_values_exposed_by_application": False,
            "private_cache_count": len(private_cache),
        },
        "sec_requests": sec_requests,
        "tiingo_requests": tiingo_requests,
        "yahoo_yfinance_role": "optional_secondary_only_not_executed",
        "movement_model_trained": False,
        "deployment_changed": False,
    }
    _atomic_json(qa_path, qa_payload)

    output_paths = [NEWS_OUTPUT, PRICE_OUTPUT, REJECTED_OUTPUT, QA_OUTPUT]
    manifest_payload = {
        "status": "foundation_verified",
        "package_contract_version": "8.0.0",
        "generated_at_utc": generated_at,
        "config": qa_payload["config"],
        "readiness": readiness,
        "source_contract": {
            "events": "SEC EDGAR official company disclosures",
            "prices": "Tiingo EOD authenticated internal-use primary",
            "secondary_prices": "Yahoo/yfinance optional secondary only",
            "sentiment": "existing local deployment champion",
        },
        "qualification": qa_payload["qualification"],
        "private_cache_artifacts": private_cache,
        "artifacts": [
            _artifact_entry(root, relative_path)
            for relative_path in output_paths
        ],
        "automatic_deployment_change": False,
        "movement_model_trained": False,
    }
    _atomic_json(manifest_path, manifest_payload)
    return verify_foundation(root)


def verify_foundation(project_root: Path) -> dict[str, Any]:
    """Recompute checksums and revalidate the saved foundation contract."""

    root = project_root.resolve()
    manifest_path = _project_path(root, MANIFEST_OUTPUT)
    manifest = _load_json(manifest_path, "market-data foundation manifest")
    if manifest.get("status") != "foundation_verified":
        raise FoundationError("Foundation manifest is not verified.")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 4:
        raise FoundationError("Foundation manifest must list four artifacts.")
    for entry in artifacts:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("path"), str):
            raise FoundationError("Foundation artifact entry is invalid.")
        relative_path = Path(str(entry["path"]))
        file_path = _project_path(root, relative_path)
        if sha256(file_path) != entry.get("sha256"):
            raise FoundationError(f"Checksum mismatch: {relative_path}")
        if file_path.stat().st_size != entry.get("size_bytes"):
            raise FoundationError(f"Size mismatch: {relative_path}")
        if file_path.stat().st_mode & 0o077:
            raise FoundationError(f"Output is not owner-only: {relative_path}")

    news = pd.read_csv(_project_path(root, NEWS_OUTPUT))
    prices = pd.read_csv(_project_path(root, PRICE_OUTPUT))
    required_news = {
        "article_id",
        "ticker",
        "published_at_utc",
        "text",
        "source_name",
        "source_url",
        "verification_status",
        "prob_bearish",
        "prob_neutral",
        "prob_bullish",
        "target_session_date",
        "session_open_utc",
        "reaction_return",
        "movement_label",
    }
    required_prices = {
        "ticker",
        "session_date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "source_provider",
        "cross_check_provider",
        "verification_status",
    }
    if required_news - set(news.columns):
        raise FoundationError("Saved news evidence is missing required columns.")
    if required_prices - set(prices.columns):
        raise FoundationError("Saved price evidence is missing required columns.")
    if news.empty or prices.empty:
        raise FoundationError("Saved foundation evidence is empty.")
    if news["article_id"].duplicated().any():
        raise FoundationError("Saved article IDs are not unique.")
    if prices.duplicated(["ticker", "session_date"]).any():
        raise FoundationError("Saved price grain is not ticker-session unique.")
    if set(prices["ticker"]) != set(REQUIRED_TICKERS):
        raise FoundationError("Saved prices do not cover all ten tickers.")
    if set(prices["source_provider"]) != {"tiingo_eod"}:
        raise FoundationError("Saved price provider is not Tiingo EOD.")

    probabilities = news[
        ["prob_bearish", "prob_neutral", "prob_bullish"]
    ].apply(pd.to_numeric, errors="coerce")
    if probabilities.isna().any().any():
        raise FoundationError("Saved sentiment probabilities are invalid.")
    if not np.allclose(probabilities.sum(axis=1), 1.0):
        raise FoundationError("Saved sentiment probabilities do not sum to one.")
    observed = pd.to_datetime(news["published_at_utc"], utc=True, errors="coerce")
    session_open = pd.to_datetime(news["session_open_utc"], utc=True, errors="coerce")
    if observed.isna().any() or session_open.isna().any():
        raise FoundationError("Saved timestamp evidence is invalid.")
    if (observed >= session_open).any():
        raise FoundationError("Saved evidence contains future leakage.")
    if set(news["movement_label"]) != set(LABEL_ORDER):
        raise FoundationError("Saved evidence must contain all movement classes.")

    readiness = manifest.get("readiness")
    if not isinstance(readiness, Mapping) or not readiness.get(
        "ready_for_stock_movement_package"
    ):
        raise FoundationError("Foundation is not ready for movement modeling.")
    private_cache = manifest.get("private_cache_artifacts")
    if not isinstance(private_cache, list) or len(private_cache) != 10:
        raise FoundationError("Private Tiingo cache inventory is incomplete.")
    for entry in private_cache:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("path"), str):
            raise FoundationError("Private cache manifest entry is invalid.")
        file_path = _project_path(root, Path(str(entry["path"])))
        if sha256(file_path) != entry.get("sha256"):
            raise FoundationError(f"Private cache checksum mismatch: {file_path}")
        if file_path.stat().st_mode & 0o077:
            raise FoundationError(f"Private cache is not owner-only: {file_path}")

    return {
        "status": "foundation_verified",
        "accepted_articles": int(len(news)),
        "primary_price_rows": int(len(prices)),
        "movement_class_counts": {
            label: int((news["movement_label"] == label).sum())
            for label in LABEL_ORDER
        },
        "manifest_sha256": sha256(manifest_path),
        "deployment_changed": False,
        "movement_model_trained": False,
    }
