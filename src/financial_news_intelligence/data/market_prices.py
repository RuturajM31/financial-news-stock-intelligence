"""Download, verify, cache, and trace historical stock prices."""

import csv
import io
import json
import math
import re
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlencode

import requests

from financial_news_intelligence.data.provenance import (
    assess_source_url,
    assert_usage_allowed,
    build_provenance,
)
from financial_news_intelligence.paths import PROJECT_ROOT
from financial_news_intelligence.schemas.market_data import (
    MarketPriceHistory,
    PriceBar,
    PriceCrossCheck,
)
from financial_news_intelligence.schemas.provenance import (
    DataPurpose,
    VerificationStatus,
)


# ============================================================
# 1. PROVIDER AND CACHE SETTINGS
# ============================================================

STOOQ_DOWNLOAD_URL = "https://stooq.com/q/d/l/"

DEFAULT_PRICE_CACHE_DIR = (
    PROJECT_ROOT / "data" / "interim" / "market_prices"
)

DEFAULT_TIMEOUT_SECONDS = 20
DEFAULT_CROSS_CHECK_TOLERANCE_PCT = 0.50


# ============================================================
# 2. DOMAIN-SPECIFIC ERRORS
# ============================================================

class MarketPriceError(ValueError):
    """Base error for invalid or unavailable market-price data."""


class PriceCrossCheckError(MarketPriceError):
    """Raised when provider prices fail independent verification."""


# ============================================================
# 3. TICKER AND URL PREPARATION
# ============================================================

def normalize_us_ticker(ticker: str) -> str:
    """Normalize and validate a US stock ticker."""

    normalized_ticker = ticker.strip().upper()

    if not re.fullmatch(
        r"[A-Z][A-Z0-9.\-]{0,14}",
        normalized_ticker,
    ):
        raise ValueError("Ticker contains unsupported characters.")

    return normalized_ticker


def build_stooq_download_url(
    ticker: str,
    start_date: date,
    end_date: date,
) -> str:
    """
    Build a Stooq daily-price CSV request.

    Periods in class-share tickers are converted to hyphens for the
    provider symbol, while the project keeps the canonical ticker.
    """

    normalized_ticker = normalize_us_ticker(ticker)

    if start_date > end_date:
        raise ValueError("start_date cannot be after end_date.")

    provider_symbol = (
        normalized_ticker.lower().replace(".", "-") + ".us"
    )

    query_string = urlencode(
        {
            "s": provider_symbol,
            "d1": start_date.strftime("%Y%m%d"),
            "d2": end_date.strftime("%Y%m%d"),
            "i": "d",
        }
    )

    return f"{STOOQ_DOWNLOAD_URL}?{query_string}"


# ============================================================
# 4. CSV PARSING
# ============================================================

def parse_stooq_csv(
    raw_csv: str,
    ticker: str,
) -> tuple[PriceBar, ...]:
    """
    Convert the provider CSV into validated daily price bars.

    Required columns:
    Date, Open, High, Low, Close, Volume
    """

    if not raw_csv.strip():
        raise MarketPriceError("Price response is empty.")

    normalized_ticker = normalize_us_ticker(ticker)

    csv_reader = csv.DictReader(
        io.StringIO(raw_csv.lstrip("\ufeff"))
    )

    original_columns = csv_reader.fieldnames or []

    column_lookup = {
        column.strip().lower(): column
        for column in original_columns
    }

    required_columns = {
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }

    missing_columns = required_columns.difference(column_lookup)

    if missing_columns:
        raise MarketPriceError(
            "Price CSV is missing columns: "
            + ", ".join(sorted(missing_columns))
        )

    adjusted_close_column = None

    for candidate in (
        "adjusted_close",
        "adjusted close",
        "adj close",
        "adjclose",
    ):
        if candidate in column_lookup:
            adjusted_close_column = column_lookup[candidate]
            break

    price_bars: list[PriceBar] = []

    for row_number, row in enumerate(csv_reader, start=2):
        # Ignore fully empty lines at the bottom of a provider response.
        if not any(str(value or "").strip() for value in row.values()):
            continue

        try:
            adjusted_close = None

            if adjusted_close_column:
                adjusted_value = str(
                    row.get(adjusted_close_column, "")
                ).strip()

                if adjusted_value:
                    adjusted_close = float(adjusted_value)

            price_bar = PriceBar(
                ticker=normalized_ticker,
                session_date=date.fromisoformat(
                    str(row[column_lookup["date"]]).strip()
                ),
                open_price=float(
                    row[column_lookup["open"]]
                ),
                high_price=float(
                    row[column_lookup["high"]]
                ),
                low_price=float(
                    row[column_lookup["low"]]
                ),
                close_price=float(
                    row[column_lookup["close"]]
                ),
                adjusted_close=adjusted_close,
                volume=int(
                    float(row[column_lookup["volume"]])
                ),
            )

        except Exception as error:
            raise MarketPriceError(
                f"Invalid price data on CSV row {row_number}: {error}"
            ) from error

        price_bars.append(price_bar)

    if not price_bars:
        raise MarketPriceError(
            "Price response contains no usable records."
        )

    return tuple(
        sorted(
            price_bars,
            key=lambda price_bar: price_bar.session_date,
        )
    )


# ============================================================
# 5. INDEPENDENT PRICE CROSS-CHECK
# ============================================================

def verify_price_cross_check(
    *,
    price_bars: tuple[PriceBar, ...],
    reference_closes: dict[date, float],
    cross_check_source_url: str,
    checked_at: datetime,
    tolerance_pct: float = DEFAULT_CROSS_CHECK_TOLERANCE_PCT,
) -> PriceCrossCheck:
    """
    Compare provider closes with sampled primary-source closes.

    The reference values must be collected from the supplied traceable
    primary-source URL. At least one common session is required.
    """

    if tolerance_pct <= 0:
        raise ValueError("tolerance_pct must be greater than zero.")

    if checked_at.tzinfo is None or checked_at.utcoffset() is None:
        raise ValueError("checked_at must be timezone-aware.")

    source_assessment = assess_source_url(
        cross_check_source_url
    )

    if (
        source_assessment.verification_status
        != VerificationStatus.VERIFIED_PRIMARY
    ):
        raise PriceCrossCheckError(
            "Cross-check source must be verified primary."
        )

    provider_closes = {
        price_bar.session_date: price_bar.close_price
        for price_bar in price_bars
    }

    differences: list[float] = []

    for session_date, reference_close in reference_closes.items():
        if (
            not math.isfinite(reference_close)
            or reference_close <= 0
        ):
            raise PriceCrossCheckError(
                "Reference close prices must be positive and finite."
            )

        provider_close = provider_closes.get(session_date)

        if provider_close is None:
            continue

        difference_pct = (
            abs(provider_close - reference_close)
            / reference_close
            * 100
        )

        differences.append(difference_pct)

    if not differences:
        raise PriceCrossCheckError(
            "No common sessions were available for cross-checking."
        )

    maximum_difference_pct = max(differences)

    if maximum_difference_pct > tolerance_pct:
        raise PriceCrossCheckError(
            "Price cross-check failed: maximum difference "
            f"{maximum_difference_pct:.6f}% exceeds "
            f"{tolerance_pct:.6f}%."
        )

    return PriceCrossCheck(
        source_id=source_assessment.source_id,
        source_name=source_assessment.source_name,
        source_url=cross_check_source_url,
        verification_status=(
            source_assessment.verification_status
        ),
        checked_at=checked_at,
        matched_sessions=len(differences),
        maximum_difference_pct=maximum_difference_pct,
        tolerance_pct=tolerance_pct,
        passed=True,
    )


# ============================================================
# 6. RAW RESPONSE AND METADATA CACHE
# ============================================================

def cache_verified_price_history(
    *,
    history: MarketPriceHistory,
    raw_csv: str,
    cache_dir: Path = DEFAULT_PRICE_CACHE_DIR,
) -> tuple[Path, Path]:
    """
    Save the untouched provider response and validated metadata.

    The checksum fragment in the filename makes each cached payload
    distinguishable and auditable.
    """

    cache_dir.mkdir(parents=True, exist_ok=True)

    checksum_fragment = (
        history.provenance.checksum_sha256[:12]
    )

    file_stem = (
        f"{history.provenance.source_id}_"
        f"{history.ticker}_"
        f"{history.start_date:%Y%m%d}_"
        f"{history.end_date:%Y%m%d}_"
        f"{checksum_fragment}"
    )

    raw_cache_path = cache_dir / f"{file_stem}.csv"
    metadata_cache_path = cache_dir / f"{file_stem}.json"

    raw_cache_path.write_text(
        raw_csv,
        encoding="utf-8",
    )

    metadata_cache_path.write_text(
        history.model_dump_json(indent=2),
        encoding="utf-8",
    )

    return raw_cache_path, metadata_cache_path


# ============================================================
# 7. COMPLETE VERIFIED DOWNLOAD
# ============================================================

def fetch_verified_price_history(
    *,
    ticker: str,
    start_date: date,
    end_date: date,
    reference_closes: dict[date, float],
    cross_check_source_url: str,
    purpose: DataPurpose = DataPurpose.TRAINING,
    retrieved_at: datetime | None = None,
    checked_at: datetime | None = None,
    tolerance_pct: float = DEFAULT_CROSS_CHECK_TOLERANCE_PCT,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    cache_dir: Path = DEFAULT_PRICE_CACHE_DIR,
    request_get: Callable[..., Any] | None = None,
) -> MarketPriceHistory:
    """
    Download and accept price history only after verification.

    Flow:
    URL → raw CSV → validated bars → primary-source comparison
    → provenance → protected-use gate → cache.
    """

    normalized_ticker = normalize_us_ticker(ticker)

    if start_date > end_date:
        raise ValueError("start_date cannot be after end_date.")

    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive.")

    if retrieved_at is None:
        retrieved_at = datetime.now(timezone.utc)

    if checked_at is None:
        checked_at = retrieved_at

    if retrieved_at.tzinfo is None:
        raise ValueError("retrieved_at must be timezone-aware.")

    download_url = build_stooq_download_url(
        normalized_ticker,
        start_date,
        end_date,
    )

    if request_get is None:
        request_get = requests.get

    try:
        response = request_get(
            download_url,
            timeout=timeout_seconds,
            headers={
                "User-Agent": (
                    "FinancialNewsStockIntelligence/0.1"
                )
            },
        )

        response.raise_for_status()

    except Exception as error:
        raise MarketPriceError(
            f"Could not download market prices: {error}"
        ) from error

    raw_csv = response.text

    all_price_bars = parse_stooq_csv(
        raw_csv,
        normalized_ticker,
    )

    price_bars = tuple(
        price_bar
        for price_bar in all_price_bars
        if start_date <= price_bar.session_date <= end_date
    )

    if not price_bars:
        raise MarketPriceError(
            "No price records fall inside the requested range."
        )

    cross_check = verify_price_cross_check(
        price_bars=price_bars,
        reference_closes=reference_closes,
        cross_check_source_url=cross_check_source_url,
        checked_at=checked_at,
        tolerance_pct=tolerance_pct,
    )

    provenance = build_provenance(
        source_url=download_url,
        raw_payload=raw_csv,
        retrieved_at=retrieved_at,
        as_of_date=price_bars[-1].session_date,
        raw_record_count=len(price_bars),
        cross_checked=True,
        cross_check_source_url=cross_check.source_url,
    )

    # This gate blocks secondary data unless cross-check evidence exists.
    assert_usage_allowed(
        provenance,
        purpose,
    )

    history = MarketPriceHistory(
        ticker=normalized_ticker,
        start_date=start_date,
        end_date=end_date,
        bars=price_bars,
        provenance=provenance,
        cross_check=cross_check,
    )

    cache_verified_price_history(
        history=history,
        raw_csv=raw_csv,
        cache_dir=cache_dir,
    )

    return history
