"""Prepare leakage-safe session-level evidence for movement modeling.

Purpose
-------
Load the exact verified Market Data Foundation v8 artifacts, recompute every
recorded checksum, aggregate SEC events to one ticker-session row, add only
market features known before the target session, and create chronological
train, validation, and historical-audit blocks separated by purged dates.

Inputs and provenance
---------------------
The foundation manifest is the source of truth for file paths and checksums.
The news file contains one verified SEC event mapped to one future session.
The price file contains one Tiingo adjusted daily row per ticker and session.

Dataset grain, joins, and formulas
----------------------------------
- Model grain: one canonical ticker and target-session date.
- Event aggregation key: ``ticker + target_session_date``.
- Price-feature join key: ``ticker + target_session_date``.
- Every market predictor is shifted by one full session.
- Split dates are chronological and one date is purged at each boundary.

Outputs and downstream use
--------------------------
The module returns an in-memory model table, approved feature lists, split
metadata, and train/validation-only SEC event evidence used by intelligence.
No raw Tiingo cache file is copied or exposed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

LABEL_ORDER = ("Down", "Flat", "Up")
SENTIMENT_ORDER = ("Bearish", "Neutral", "Bullish")
EXPECTED_FOUNDATION_MANIFEST_SHA256 = (
    "510152d22c1d65631a2b6e6f4a7ec3c4c03b542b109d48c78400d73d9f025da2"
)
FOUNDATION_MANIFEST = Path(
    "artifacts/manifests/market_data_foundation_manifest.json"
)
EXPECTED_FOUNDATION_ARTIFACTS = {
    "data/processed/news_sentiment_evidence.csv",
    "data/processed/market_price_evidence.csv",
    "data/processed/market_data_foundation_rejected_rows.csv",
    "reports/qa/market_data_foundation_qa.json",
}

# Target and target-adjacent columns are never eligible as model inputs.
FORBIDDEN_FEATURES = {
    "article_id",
    "published_at_utc",
    "session_open_utc",
    "session_close_utc",
    "previous_session_date",
    "previous_close",
    "target_open",
    "target_close",
    "target_volume",
    "reaction_return",
    "movement_label",
    "sentiment_label",
    "source_url",
    "headline",
    "text",
}


class MovementDatasetError(RuntimeError):
    """Raised when foundation evidence or split logic is unsafe."""


@dataclass(frozen=True)
class SplitConfig:
    """Store deterministic chronological split and purge rules."""

    train_ratio: float = 0.70
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    purge_dates: int = 1
    minimum_unique_dates: int = 300
    minimum_rows_per_split: int = 30
    include_ticker: bool = True


def sha256(file_path: Path) -> str:
    """Return one safe regular file's hexadecimal SHA-256 checksum."""

    if not file_path.exists() or file_path.is_symlink() or not file_path.is_file():
        raise MovementDatasetError(f"Missing or unsafe checksum source: {file_path}")
    digest = hashlib.sha256()
    with file_path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_project_path(project_root: Path, relative_path: Path) -> Path:
    """Resolve one path below the project root without traversal."""

    root = project_root.expanduser().resolve()
    target = (root / relative_path).resolve()
    if root not in target.parents:
        raise MovementDatasetError(f"Path escapes project root: {relative_path}")
    return target


def _load_json(file_path: Path, description: str) -> dict[str, Any]:
    """Load one required UTF-8 JSON object from a regular file."""

    if not file_path.exists() or file_path.is_symlink() or not file_path.is_file():
        raise MovementDatasetError(f"Missing or unsafe {description}: {file_path}")
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MovementDatasetError(f"Invalid {description}: {exc}") from exc
    if not isinstance(payload, dict):
        raise MovementDatasetError(f"{description} must contain a JSON object.")
    return payload


def _require_columns(
    frame: pd.DataFrame,
    required_columns: set[str],
    description: str,
) -> None:
    """Require an exact minimum schema before calculations begin."""

    missing = sorted(required_columns - set(frame.columns))
    if missing:
        raise MovementDatasetError(
            f"{description} is missing required columns: {missing}"
        )


def verify_foundation(project_root: Path) -> dict[str, Any]:
    """Verify the exact completed v8 foundation and every artifact checksum."""

    root = project_root.expanduser().resolve()
    manifest_path = _safe_project_path(root, FOUNDATION_MANIFEST)
    actual_manifest_sha = sha256(manifest_path)
    if actual_manifest_sha != EXPECTED_FOUNDATION_MANIFEST_SHA256:
        raise MovementDatasetError(
            "Foundation manifest checksum mismatch. "
            f"Expected {EXPECTED_FOUNDATION_MANIFEST_SHA256}; "
            f"found {actual_manifest_sha}."
        )

    manifest = _load_json(manifest_path, "foundation manifest")
    if manifest.get("status") != "foundation_verified":
        raise MovementDatasetError("Foundation status is not verified.")
    if manifest.get("movement_model_trained") is not False:
        raise MovementDatasetError("Foundation unexpectedly reports model training.")
    if manifest.get("automatic_deployment_change") is not False:
        raise MovementDatasetError("Foundation unexpectedly changed deployment.")

    readiness = manifest.get("readiness")
    if not isinstance(readiness, Mapping):
        raise MovementDatasetError("Foundation readiness evidence is missing.")
    if readiness.get("ready_for_stock_movement_package") is not True:
        raise MovementDatasetError("Foundation is not ready for movement modeling.")
    if int(readiness.get("unique_session_dates", 0)) < 300:
        raise MovementDatasetError("Foundation has fewer than 300 session dates.")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise MovementDatasetError("Foundation artifact inventory is missing.")
    recorded_paths: set[str] = set()
    for entry in artifacts:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("path"), str):
            raise MovementDatasetError("Foundation artifact entry is invalid.")
        relative_path = Path(str(entry["path"]))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise MovementDatasetError(f"Unsafe foundation path: {relative_path}")
        file_path = _safe_project_path(root, relative_path)
        if sha256(file_path) != entry.get("sha256"):
            raise MovementDatasetError(
                f"Foundation artifact checksum changed: {relative_path}"
            )
        if file_path.stat().st_size != entry.get("size_bytes"):
            raise MovementDatasetError(
                f"Foundation artifact size changed: {relative_path}"
            )
        if file_path.stat().st_mode & 0o077:
            raise MovementDatasetError(
                f"Foundation artifact is not owner-only: {relative_path}"
            )
        recorded_paths.add(relative_path.as_posix())
    if recorded_paths != EXPECTED_FOUNDATION_ARTIFACTS:
        raise MovementDatasetError("Foundation artifact path set changed.")

    return manifest


def load_foundation_frames(
    project_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load verified foundation event and price tables after checksum checks."""

    root = project_root.expanduser().resolve()
    manifest = verify_foundation(root)
    news_path = _safe_project_path(
        root,
        Path("data/processed/news_sentiment_evidence.csv"),
    )
    price_path = _safe_project_path(
        root,
        Path("data/processed/market_price_evidence.csv"),
    )
    news = pd.read_csv(news_path)
    prices = pd.read_csv(price_path)
    if news.empty or prices.empty:
        raise MovementDatasetError("Verified foundation evidence is empty.")
    return news, prices, manifest


def aggregate_events(news: pd.DataFrame) -> pd.DataFrame:
    """Aggregate verified SEC events to one ticker-target-session row."""

    required = {
        "article_id",
        "ticker",
        "company",
        "published_at_utc",
        "text",
        "source_name",
        "source_url",
        "sentiment_label",
        "prob_bearish",
        "prob_neutral",
        "prob_bullish",
        "target_session_date",
        "session_open_utc",
        "reaction_return",
        "movement_label",
    }
    _require_columns(news, required, "Foundation event evidence")
    frame = news.copy()

    # Normalize identifiers and timestamps before grouping. Invalid values fail
    # closed because silently dropping them could change class balance or dates.
    frame["ticker"] = frame["ticker"].astype(str).str.upper().str.strip()
    frame["published_at_utc"] = pd.to_datetime(
        frame["published_at_utc"], utc=True, errors="coerce"
    )
    frame["session_open_utc"] = pd.to_datetime(
        frame["session_open_utc"], utc=True, errors="coerce"
    )
    frame["target_session_date"] = pd.to_datetime(
        frame["target_session_date"], errors="coerce"
    ).dt.normalize()
    frame["reaction_return"] = pd.to_numeric(
        frame["reaction_return"], errors="coerce"
    )
    timestamp_columns = [
        "published_at_utc",
        "session_open_utc",
        "target_session_date",
        "reaction_return",
    ]
    if frame[timestamp_columns].isna().any().any():
        raise MovementDatasetError("Foundation event values are invalid.")
    if frame["article_id"].duplicated().any():
        raise MovementDatasetError("Foundation article IDs are not unique.")
    if (frame["published_at_utc"] >= frame["session_open_utc"]).any():
        raise MovementDatasetError("Foundation event mapping contains future leakage.")
    if set(frame["movement_label"]) != set(LABEL_ORDER):
        raise MovementDatasetError("Foundation must contain all movement labels.")
    if not set(frame["sentiment_label"]).issubset(SENTIMENT_ORDER):
        raise MovementDatasetError("Foundation contains an unknown sentiment label.")

    probability_columns = ["prob_bearish", "prob_neutral", "prob_bullish"]
    frame[probability_columns] = frame[probability_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )
    if frame[probability_columns].isna().any().any():
        raise MovementDatasetError("Sentiment probabilities contain missing values.")
    if not np.allclose(frame[probability_columns].sum(axis=1), 1.0, atol=1e-6):
        raise MovementDatasetError("Sentiment probabilities do not sum to one.")

    # These derived event features use only text and timestamps available before
    # the mapped target-session open.
    frame["text"] = frame["text"].fillna("").astype(str)
    frame["text_length"] = frame["text"].str.len()
    frame["net_sentiment"] = frame["prob_bullish"] - frame["prob_bearish"]
    clipped = np.clip(frame[probability_columns].to_numpy(float), 1e-12, 1.0)
    frame["sentiment_entropy"] = -(
        clipped * np.log(clipped)
    ).sum(axis=1) / np.log(3.0)

    # Confidence and probability spread describe how decisive the sentiment
    # champion was before the target market session. They use only the three
    # saved sentiment probabilities and therefore cannot include future prices.
    frame["sentiment_confidence"] = frame[probability_columns].max(axis=1)
    frame["sentiment_probability_spread"] = (
        frame[probability_columns].max(axis=1)
        - frame[probability_columns].min(axis=1)
    )
    frame["hours_to_open"] = (
        frame["session_open_utc"] - frame["published_at_utc"]
    ).dt.total_seconds() / 3600.0

    group = frame.groupby(["ticker", "target_session_date"], observed=True)
    if group["movement_label"].nunique().gt(1).any():
        raise MovementDatasetError("One ticker-session has conflicting labels.")
    if group["reaction_return"].nunique().gt(1).any():
        raise MovementDatasetError("One ticker-session has conflicting returns.")
    if group["company"].nunique().gt(1).any():
        raise MovementDatasetError("One ticker-session has conflicting company names.")

    # One row now represents all SEC events mapped to one ticker and target day.
    table = group.agg(
        company=("company", "first"),
        article_count=("article_id", "count"),
        unique_source_count=("source_name", "nunique"),
        text_length_mean=("text_length", "mean"),
        text_length_max=("text_length", "max"),
        event_text=(
            "text",
            lambda values: " | ".join(
                sorted(
                    {
                        str(value).strip()
                        for value in values
                        if str(value).strip()
                    }
                )
            ),
        ),
        prob_bearish_mean=("prob_bearish", "mean"),
        prob_neutral_mean=("prob_neutral", "mean"),
        prob_bullish_mean=("prob_bullish", "mean"),
        net_sentiment_mean=("net_sentiment", "mean"),
        net_sentiment_std=("net_sentiment", "std"),
        sentiment_entropy_mean=("sentiment_entropy", "mean"),
        sentiment_confidence_mean=("sentiment_confidence", "mean"),
        sentiment_confidence_max=("sentiment_confidence", "max"),
        sentiment_probability_spread_mean=(
            "sentiment_probability_spread",
            "mean",
        ),
        bearish_event_share=(
            "sentiment_label",
            lambda values: float((values == "Bearish").mean()),
        ),
        neutral_event_share=(
            "sentiment_label",
            lambda values: float((values == "Neutral").mean()),
        ),
        bullish_event_share=(
            "sentiment_label",
            lambda values: float((values == "Bullish").mean()),
        ),
        hours_to_open_min=("hours_to_open", "min"),
        hours_to_open_mean=("hours_to_open", "mean"),
        hours_to_open_max=("hours_to_open", "max"),
        reaction_return=("reaction_return", "first"),
        movement_label=("movement_label", "first"),
    ).reset_index()
    table["net_sentiment_std"] = table["net_sentiment_std"].fillna(0.0)

    # Log scaling and a simple multi-event marker preserve ordering while
    # reducing the influence of unusually busy filing days.
    table["article_count_log1p"] = np.log1p(table["article_count"])
    table["is_multi_event_day"] = table["article_count"].gt(1).astype(int)
    if table["event_text"].eq("").any():
        raise MovementDatasetError("An aggregated ticker-session has no text.")
    return table


def build_prior_price_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Create adjusted-price predictors shifted one complete session."""

    required = {"ticker", "session_date", "close", "volume", "source_provider"}
    _require_columns(prices, required, "Foundation price evidence")
    frame = prices.copy()
    frame["ticker"] = frame["ticker"].astype(str).str.upper().str.strip()
    frame["session_date"] = pd.to_datetime(
        frame["session_date"], errors="coerce"
    ).dt.normalize()
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce")
    if frame[["session_date", "close", "volume"]].isna().any().any():
        raise MovementDatasetError("Foundation price values are invalid.")
    if frame.duplicated(["ticker", "session_date"]).any():
        raise MovementDatasetError("Price grain is not one ticker-session row.")
    if set(frame["source_provider"].astype(str).str.lower()) != {"tiingo_eod"}:
        raise MovementDatasetError("Foundation primary price provider changed.")
    if (frame["close"] <= 0).any() or (frame["volume"] < 0).any():
        raise MovementDatasetError("Foundation prices contain invalid values.")

    parts: list[pd.DataFrame] = []
    for _, ticker_frame in frame.groupby("ticker", observed=True):
        ticker_frame = ticker_frame.sort_values("session_date").copy()
        daily_return = ticker_frame["close"].pct_change()
        rolling_volume_mean = ticker_frame["volume"].rolling(
            20,
            min_periods=5,
        ).mean()
        rolling_volume_std = ticker_frame["volume"].rolling(
            20,
            min_periods=5,
        ).std(ddof=0)

        # Every market feature is shifted by one complete session. The value at
        # the target date therefore depends only on prices observed before the
        # market opens for the event reaction being predicted.
        ticker_frame["prior_return_1d"] = daily_return.shift(1)
        for window in (2, 3, 5, 10, 20, 60):
            ticker_frame[f"prior_return_{window}d"] = (
                ticker_frame["close"].pct_change(window).shift(1)
            )
        for window in (5, 10, 20, 60):
            ticker_frame[f"prior_volatility_{window}d"] = (
                daily_return.rolling(
                    window,
                    min_periods=max(3, window // 4),
                ).std(ddof=0).shift(1)
                * np.sqrt(252.0)
            )

        ticker_frame["prior_volume_zscore_20d"] = (
            (ticker_frame["volume"] - rolling_volume_mean)
            / rolling_volume_std.replace(0.0, np.nan)
        ).shift(1)
        ticker_frame["prior_volume_change_1d"] = (
            ticker_frame["volume"].pct_change().replace([np.inf, -np.inf], np.nan)
            .shift(1)
        )
        ticker_frame["prior_volume_change_5d"] = (
            ticker_frame["volume"].pct_change(5).replace([np.inf, -np.inf], np.nan)
            .shift(1)
        )
        ticker_frame["prior_log_volume"] = np.log1p(
            ticker_frame["volume"]
        ).shift(1)

        rolling_high = ticker_frame["close"].rolling(20, min_periods=5).max()
        rolling_low = ticker_frame["close"].rolling(20, min_periods=5).min()
        range_width = (rolling_high - rolling_low).replace(0.0, np.nan)
        ticker_frame["prior_close_position_20d"] = (
            (ticker_frame["close"] - rolling_low) / range_width
        ).shift(1)
        ticker_frame["prior_drawdown_20d"] = (
            ticker_frame["close"] / rolling_high - 1.0
        ).shift(1)
        ticker_frame["prior_momentum_5_20"] = (
            ticker_frame["close"].pct_change(5)
            - ticker_frame["close"].pct_change(20)
        ).shift(1)
        ticker_frame["prior_trend_strength_20d"] = (
            ticker_frame["close"].pct_change(20)
            / daily_return.rolling(20, min_periods=5).std(ddof=0).replace(0.0, np.nan)
        ).shift(1)

        selected_columns = [
            "ticker",
            "session_date",
            "prior_return_1d",
            "prior_return_2d",
            "prior_return_3d",
            "prior_return_5d",
            "prior_return_10d",
            "prior_return_20d",
            "prior_return_60d",
            "prior_volatility_5d",
            "prior_volatility_10d",
            "prior_volatility_20d",
            "prior_volatility_60d",
            "prior_volume_zscore_20d",
            "prior_volume_change_1d",
            "prior_volume_change_5d",
            "prior_log_volume",
            "prior_close_position_20d",
            "prior_drawdown_20d",
            "prior_momentum_5_20",
            "prior_trend_strength_20d",
        ]
        parts.append(ticker_frame[selected_columns])
    price_features = pd.concat(parts, ignore_index=True)

    # Cross-sectional market-regime features are calculated only from already
    # shifted ticker features. They describe broad market conditions known
    # before the target session without exposing any target-session price.
    market_regime = price_features.groupby("session_date", observed=True).agg(
        prior_market_return_1d=("prior_return_1d", "mean"),
        prior_market_return_5d=("prior_return_5d", "mean"),
        prior_market_return_20d=("prior_return_20d", "mean"),
        prior_market_volatility_20d=("prior_volatility_20d", "mean"),
        prior_market_breadth_1d=(
            "prior_return_1d",
            lambda values: float((values > 0).mean()),
        ),
    ).reset_index()
    price_features = price_features.merge(
        market_regime,
        on="session_date",
        how="left",
        validate="many_to_one",
    )
    price_features["prior_relative_return_1d"] = (
        price_features["prior_return_1d"]
        - price_features["prior_market_return_1d"]
    )
    price_features["prior_relative_return_5d"] = (
        price_features["prior_return_5d"]
        - price_features["prior_market_return_5d"]
    )
    price_features["prior_relative_return_20d"] = (
        price_features["prior_return_20d"]
        - price_features["prior_market_return_20d"]
    )
    return price_features


def add_prior_event_history_features(event_table: pd.DataFrame) -> pd.DataFrame:
    """Add ticker-event history known before each target session.

    One row still represents one ticker and target session. Every rolling
    statistic is shifted by one event session, so the current reaction label
    and return never enter their own predictors.
    """

    parts: list[pd.DataFrame] = []
    for _, ticker_frame in event_table.groupby("ticker", observed=True):
        ordered = ticker_frame.sort_values("target_session_date").copy()
        prior_return = ordered["reaction_return"].shift(1)
        prior_label = ordered["movement_label"].shift(1)
        for window in (5, 20):
            ordered[f"prior_event_return_mean_{window}"] = (
                prior_return.rolling(window, min_periods=2).mean()
            )
            ordered[f"prior_event_return_std_{window}"] = (
                prior_return.rolling(window, min_periods=2).std(ddof=0)
            )
        ordered["prior_event_count_20"] = (
            prior_return.rolling(20, min_periods=1).count()
        )
        for label in LABEL_ORDER:
            label_indicator = prior_label.eq(label).where(prior_label.notna())
            ordered[f"prior_{label.lower()}_share_20"] = (
                label_indicator.astype(float)
                .rolling(20, min_periods=2)
                .mean()
            )
        parts.append(ordered)
    return pd.concat(parts, ignore_index=True)


def build_model_table(news: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Join session aggregates to strictly lagged market predictors."""

    event_table = add_prior_event_history_features(aggregate_events(news))
    price_features = build_prior_price_features(prices)
    table = event_table.merge(
        price_features,
        left_on=["ticker", "target_session_date"],
        right_on=["ticker", "session_date"],
        how="left",
        validate="one_to_one",
    ).drop(columns=["session_date"])

    # Calendar features are known before trading begins and contain no target
    # price values. They can help capture regular day-of-week or seasonal effects.
    table["session_day_of_week"] = table["target_session_date"].dt.dayofweek
    table["session_day_of_month"] = table["target_session_date"].dt.day
    table["session_month"] = table["target_session_date"].dt.month
    table["session_quarter"] = table["target_session_date"].dt.quarter
    table["session_is_month_end"] = (
        table["target_session_date"].dt.is_month_end.astype(int)
    )
    if table.empty:
        raise MovementDatasetError("Model table is empty after the price join.")
    if table.duplicated(["ticker", "target_session_date"]).any():
        raise MovementDatasetError("Model table grain is not ticker-session unique.")
    return table.sort_values(
        ["target_session_date", "ticker"]
    ).reset_index(drop=True)


def assign_chronological_splits(
    table: pd.DataFrame,
    config: SplitConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create purged date blocks and require all classes in every split."""

    active = config or SplitConfig()
    ratios = np.array(
        [active.train_ratio, active.validation_ratio, active.test_ratio],
        dtype=float,
    )
    if (ratios <= 0).any() or not np.isclose(ratios.sum(), 1.0):
        raise MovementDatasetError("Split ratios must be positive and sum to one.")
    if active.purge_dates < 1:
        raise MovementDatasetError("At least one purged date is required.")

    frame = table.copy()
    frame["target_session_date"] = pd.to_datetime(
        frame["target_session_date"], errors="coerce"
    ).dt.normalize()
    if frame["target_session_date"].isna().any():
        raise MovementDatasetError("Model table contains invalid target dates.")
    unique_dates = np.array(sorted(frame["target_session_date"].unique()))
    if len(unique_dates) < active.minimum_unique_dates:
        raise MovementDatasetError(
            f"Need {active.minimum_unique_dates} dates; found {len(unique_dates)}."
        )

    train_end = max(1, int(len(unique_dates) * active.train_ratio))
    validation_size = max(1, int(len(unique_dates) * active.validation_ratio))
    validation_start = train_end + active.purge_dates
    validation_end = validation_start + validation_size
    test_start = validation_end + active.purge_dates
    if test_start >= len(unique_dates):
        raise MovementDatasetError("Purged boundaries leave no test block.")

    split_dates = {
        "train": unique_dates[:train_end],
        "validation": unique_dates[validation_start:validation_end],
        "test": unique_dates[test_start:],
    }
    frame["split"] = "purged"
    report: dict[str, Any] = {}
    for split_name, dates in split_dates.items():
        frame.loc[frame["target_session_date"].isin(dates), "split"] = split_name
        split_rows = frame[frame["target_session_date"].isin(dates)]
        class_counts = {
            label: int((split_rows["movement_label"] == label).sum())
            for label in LABEL_ORDER
        }
        if len(split_rows) < active.minimum_rows_per_split:
            raise MovementDatasetError(
                f"{split_name} has only {len(split_rows)} rows."
            )
        if any(count == 0 for count in class_counts.values()):
            raise MovementDatasetError(
                f"{split_name} lacks a movement class: {class_counts}"
            )
        report[split_name] = {
            "start_date": str(pd.Timestamp(dates[0]).date()),
            "end_date": str(pd.Timestamp(dates[-1]).date()),
            "unique_dates": int(len(dates)),
            "rows": int(len(split_rows)),
            "class_counts": class_counts,
            "tickers": sorted(split_rows["ticker"].astype(str).unique()),
        }

    purged_rows = frame[frame["split"] == "purged"]
    report["purged"] = {
        "unique_dates": int(purged_rows["target_session_date"].nunique()),
        "rows": int(len(purged_rows)),
        "dates": [
            str(pd.Timestamp(value).date())
            for value in sorted(purged_rows["target_session_date"].unique())
        ],
    }

    # These explicit comparisons make accidental date overlap fail before model
    # fitting rather than relying only on how the arrays were sliced above.
    if not (
        pd.Timestamp(report["train"]["end_date"])
        < pd.Timestamp(report["validation"]["start_date"])
        < pd.Timestamp(report["validation"]["end_date"])
        < pd.Timestamp(report["test"]["start_date"])
    ):
        raise MovementDatasetError("Chronological split boundaries overlap.")

    return frame[frame["split"] != "purged"].reset_index(drop=True), report


def feature_columns(
    table: pd.DataFrame,
    config: SplitConfig | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """Return approved numeric, categorical, and filing-text predictors."""

    active = config or SplitConfig()
    excluded = FORBIDDEN_FEATURES | {
        "company",
        "target_session_date",
        "split",
    }
    categorical = ["ticker"] if active.include_ticker else []
    text_features = ["event_text"]
    numeric = sorted(
        column
        for column in table.columns
        if column not in excluded
        and column not in categorical
        and column not in text_features
        and pd.api.types.is_numeric_dtype(table[column])
    )
    selected = set(numeric + categorical + text_features)
    if not numeric:
        raise MovementDatasetError("No numeric movement features were selected.")
    if not set(text_features).issubset(table.columns):
        raise MovementDatasetError("Aggregated filing text is unavailable.")
    if table["event_text"].fillna("").astype(str).str.strip().eq("").any():
        raise MovementDatasetError("Aggregated filing text contains empty values.")
    if selected & FORBIDDEN_FEATURES:
        raise MovementDatasetError("Forbidden target evidence entered features.")
    return numeric, categorical, text_features


def filter_events_by_split(
    event_frame: pd.DataFrame,
    model_table: pd.DataFrame,
    allowed_splits: set[str],
) -> pd.DataFrame:
    """Return events whose ticker-session key belongs to approved splits."""

    if not allowed_splits or not allowed_splits.issubset(
        {"train", "validation", "test"}
    ):
        raise MovementDatasetError("Allowed split names are invalid.")
    _require_columns(
        event_frame,
        {"ticker", "target_session_date"},
        "Event evidence",
    )
    _require_columns(
        model_table,
        {"ticker", "target_session_date", "split"},
        "Model table",
    )

    events = event_frame.copy()
    events["ticker"] = events["ticker"].astype(str).str.upper().str.strip()
    events["target_session_date"] = pd.to_datetime(
        events["target_session_date"], errors="coerce"
    ).dt.normalize()
    keys = model_table[model_table["split"].isin(allowed_splits)][
        ["ticker", "target_session_date"]
    ].drop_duplicates()
    keys["target_session_date"] = pd.to_datetime(
        keys["target_session_date"], errors="coerce"
    ).dt.normalize()

    # An inner join is intentional. It prevents purged or test-period events
    # from entering global intelligence when only reference evidence is allowed.
    filtered = events.merge(
        keys,
        on=["ticker", "target_session_date"],
        how="inner",
        validate="many_to_one",
    )
    if filtered.empty:
        raise MovementDatasetError(
            f"No events remain for allowed splits: {sorted(allowed_splits)}"
        )
    return filtered.sort_values(
        ["target_session_date", "ticker", "article_id"]
    ).reset_index(drop=True)
