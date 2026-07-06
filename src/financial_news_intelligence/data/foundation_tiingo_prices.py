"""Collect private adjusted Tiingo EOD prices after technical qualification.

Purpose
-------
Download the exact qualified 2015-01-01 through 2020-04-01 daily window for
all ten project tickers, validate every row, preserve private cached responses,
and create a model-compatible adjusted OHLCV table.

Inputs and source variables
---------------------------
- ``TIINGO_API_TOKEN`` supplied only through the process environment;
- the licence-safe ten-ticker qualification summary already produced locally;
- authenticated Tiingo metadata-free historical price responses;
- an optional private cache under ``data/private/tiingo_eod``.

Grain, formulas, and joins
--------------------------
One output row represents one canonical ticker and one trading session. Model
columns use Tiingo's split/dividend-adjusted values: ``adjOpen``, ``adjHigh``,
``adjLow``, ``adjClose``, and ``adjVolume``. The canonical ticker plus session
date is the downstream join key. Raw fields remain in the private response
cache and are not copied into public reports.

Provenance, security, and licence boundary
------------------------------------------
The API token is sent in an Authorization header, never in the URL, logs, or
saved files. Raw JSON caches use owner-only permissions and a generated nested
``.gitignore``. Historical response checksums must match the previously passed
qualification summary for the same provider symbol and date window. Tiingo is
an authenticated internal-use primary source; raw provider values must not be
redistributed or exposed by the future public application. Yahoo/yfinance may
be added later only as an optional secondary cross-check, never as the sole
source.

Fallback and failure behaviour
------------------------------
A validated private cache is preferred on reruns. Live requests use visible
progress, two-second pacing, bounded retries, and Retry-After handling. Any
missing ticker, checksum drift, duplicate date, invalid adjusted value, or
licence-contract mismatch fails closed before processed evidence is written.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

API_BASE_URL = "https://api.tiingo.com/tiingo/daily"
REQUIRED_FIELDS = {
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "adjOpen",
    "adjHigh",
    "adjLow",
    "adjClose",
    "adjVolume",
    "divCash",
    "splitFactor",
}
POSITIVE_FIELDS = {
    "open",
    "high",
    "low",
    "close",
    "adjOpen",
    "adjHigh",
    "adjLow",
    "adjClose",
    "splitFactor",
}
NON_NEGATIVE_FIELDS = {"volume", "adjVolume", "divCash"}
ProgressCallback = Callable[[str], None]


class TiingoPriceError(RuntimeError):
    """Raised when private Tiingo price evidence violates the contract."""


@dataclass(frozen=True)
class RetryPolicy:
    """Store bounded retry and timeout settings for authenticated requests."""

    maximum_attempts: int = 3
    base_delay_seconds: float = 3.0
    maximum_delay_seconds: float = 30.0
    timeout_seconds: float = 30.0


@dataclass
class RequestPacer:
    """Space live requests below the qualified free-plan hourly limit."""

    minimum_interval_seconds: float = 2.0
    clock: Callable[[], float] = time.monotonic
    sleeper: Callable[[float], None] = time.sleep
    last_request_started_at: float | None = None

    def wait(self) -> None:
        """Sleep only for the remaining required request interval."""

        if self.last_request_started_at is None:
            self.last_request_started_at = self.clock()
            return
        elapsed = self.clock() - self.last_request_started_at
        remaining = self.minimum_interval_seconds - elapsed
        if remaining > 0:
            self.sleeper(remaining)
        self.last_request_started_at = self.clock()


@dataclass(frozen=True)
class PriceRequestEvidence:
    """Record one non-secret Tiingo request and cache outcome."""

    canonical_ticker: str
    provider_symbol: str
    request_url: str
    response_origin: str
    cache_path: str
    response_sha256: str
    qualification_sha256: str
    checksum_matches_qualification: bool
    response_bytes: int
    accepted_rows: int
    observed_start_date: str
    observed_end_date: str
    attempts: int
    retry_delays_seconds: tuple[float, ...]


def _progress(callback: ProgressCallback | None, message: str) -> None:
    """Emit one immediately visible progress line when configured."""

    if callback is not None:
        callback(message)


def _canonical_json_bytes(payload: Any) -> bytes:
    """Serialize JSON deterministically before computing response checksums."""

    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _sha256_bytes(content: bytes) -> str:
    """Return one lowercase SHA-256 digest."""

    return hashlib.sha256(content).hexdigest()


def _parse_date(value: Any, field_name: str) -> date:
    """Parse one Tiingo ISO timestamp into a date."""

    text = str(value or "").strip()[:10]
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise TiingoPriceError(
            f"{field_name} is not a valid ISO date: {value!r}"
        ) from exc


def _finite_number(value: Any) -> bool:
    """Return True only for finite non-boolean numeric values."""

    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _retry_after_seconds(value: str | None) -> float | None:
    """Parse Retry-After seconds or an HTTP date."""

    if value is None or not value.strip():
        return None
    try:
        return max(0.0, float(value.strip()))
    except ValueError:
        pass
    try:
        retry_at = parsedate_to_datetime(value.strip())
    except (TypeError, ValueError, OverflowError):
        return None
    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=timezone.utc)
    return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())


def build_prices_url(
    provider_symbol: str,
    start_date: date,
    end_date: date,
) -> str:
    """Build the documented Tiingo EOD historical-prices endpoint."""

    return (
        f"{API_BASE_URL}/{provider_symbol}/prices"
        f"?startDate={start_date.isoformat()}"
        f"&endDate={end_date.isoformat()}"
        "&format=json"
    )


def _read_private_cache(cache_path: Path) -> tuple[list[dict[str, Any]], bytes]:
    """Read one owner-only validated private JSON cache."""

    if cache_path.is_symlink() or not cache_path.is_file():
        raise TiingoPriceError(f"Unsafe Tiingo cache path: {cache_path}")
    if cache_path.stat().st_mode & 0o077:
        raise TiingoPriceError(
            f"Tiingo cache permissions are not owner-only: {cache_path}"
        )
    content = cache_path.read_bytes()
    try:
        payload = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TiingoPriceError(f"Invalid Tiingo cache JSON: {cache_path}") from exc
    if not isinstance(payload, list):
        raise TiingoPriceError("Tiingo price response must contain a JSON list.")
    rows = [dict(row) for row in payload if isinstance(row, Mapping)]
    if len(rows) != len(payload):
        raise TiingoPriceError("Tiingo price response contains non-object rows.")
    return rows, _canonical_json_bytes(payload)


def _write_private_cache(cache_path: Path, payload: Sequence[Mapping[str, Any]]) -> None:
    """Write raw Tiingo JSON atomically with owner-only permissions."""

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    ignore_path = cache_path.parent / ".gitignore"
    if not ignore_path.exists():
        ignore_path.write_text(
            "# Private Tiingo responses: internal use only.\n*\n!.gitignore\n",
            encoding="utf-8",
        )
    content = json.dumps(
        list(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    temporary.write_bytes(content)
    os.chmod(temporary, 0o600)
    temporary.replace(cache_path)


def fetch_prices_with_cache(
    provider_symbol: str,
    start_date: date,
    end_date: date,
    api_token: str,
    cache_path: Path,
    *,
    refresh_cache: bool = False,
    retry_policy: RetryPolicy | None = None,
    pacer: RequestPacer | None = None,
    opener: Callable[..., Any] = urlopen,
    sleeper: Callable[[float], None] = time.sleep,
    progress_callback: ProgressCallback | None = None,
) -> tuple[list[dict[str, Any]], str, int, tuple[float, ...]]:
    """Return Tiingo rows from a private cache or bounded live request."""

    if not api_token.strip():
        raise TiingoPriceError("TIINGO_API_TOKEN is empty.")
    if cache_path.exists() and not refresh_cache:
        rows, _ = _read_private_cache(cache_path)
        _progress(
            progress_callback,
            f"[{provider_symbol}] using verified private cache",
        )
        return rows, "private_cache", 0, ()

    policy = retry_policy or RetryPolicy()
    active_pacer = pacer or RequestPacer()
    request_url = build_prices_url(provider_symbol, start_date, end_date)
    delays: list[float] = []
    last_error: Exception | None = None

    for attempt in range(1, policy.maximum_attempts + 1):
        active_pacer.wait()
        _progress(
            progress_callback,
            f"[{provider_symbol}] request attempt "
            f"{attempt}/{policy.maximum_attempts}",
        )
        request = Request(
            request_url,
            headers={
                "Authorization": f"Token {api_token}",
                "Accept": "application/json",
                "User-Agent": "financial-news-stock-intelligence/tiingo-foundation",
            },
        )
        try:
            with opener(request, timeout=policy.timeout_seconds) as response:
                raw_content = response.read()
            payload = json.loads(raw_content.decode("utf-8"))
            if not isinstance(payload, list):
                raise TiingoPriceError(
                    "Tiingo price response must contain a JSON list."
                )
            rows = [dict(row) for row in payload if isinstance(row, Mapping)]
            if len(rows) != len(payload):
                raise TiingoPriceError(
                    "Tiingo price response contains non-object rows."
                )
            _write_private_cache(cache_path, rows)
            return rows, "live", attempt, tuple(delays)
        except HTTPError as exc:
            last_error = exc
            if exc.code in {401, 403}:
                raise TiingoPriceError(
                    f"Tiingo authentication failed with HTTP {exc.code}."
                ) from exc
            provider_delay = _retry_after_seconds(
                exc.headers.get("Retry-After") if exc.headers else None
            )
            fallback_delay = policy.base_delay_seconds * (2 ** (attempt - 1))
            delay = provider_delay if provider_delay is not None else fallback_delay
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            delay = policy.base_delay_seconds * (2 ** (attempt - 1))
        except TiingoPriceError as exc:
            last_error = exc
            delay = policy.base_delay_seconds * (2 ** (attempt - 1))

        if attempt < policy.maximum_attempts:
            bounded = min(delay, policy.maximum_delay_seconds)
            delays.append(bounded)
            _progress(
                progress_callback,
                f"[{provider_symbol}] retrying after {bounded:.1f} seconds",
            )
            sleeper(bounded)

    raise TiingoPriceError(
        f"Tiingo request failed for {provider_symbol}: "
        f"{type(last_error).__name__}: {last_error}"
    ) from last_error


def validate_price_rows(
    rows: Sequence[Mapping[str, Any]],
    provider_symbol: str,
    start_date: date,
    end_date: date,
    minimum_rows: int,
) -> tuple[date, date]:
    """Validate schema, date order, uniqueness, and adjusted numeric evidence."""

    if len(rows) < minimum_rows:
        raise TiingoPriceError(
            f"{provider_symbol} returned {len(rows)} rows; "
            f"minimum is {minimum_rows}."
        )
    dates: list[date] = []
    seen: set[date] = set()
    for index, row in enumerate(rows):
        missing = REQUIRED_FIELDS - set(row)
        if missing:
            raise TiingoPriceError(
                f"{provider_symbol} row {index} is missing fields: "
                f"{sorted(missing)}"
            )
        observed_date = _parse_date(row["date"], f"{provider_symbol}.date")
        if observed_date in seen:
            raise TiingoPriceError(
                f"{provider_symbol} contains duplicate date {observed_date}."
            )
        seen.add(observed_date)
        dates.append(observed_date)
        for field in POSITIVE_FIELDS:
            value = row.get(field)
            if not _finite_number(value) or float(value) <= 0:
                raise TiingoPriceError(
                    f"{provider_symbol} row {index} has invalid {field}."
                )
        for field in NON_NEGATIVE_FIELDS:
            value = row.get(field)
            if not _finite_number(value) or float(value) < 0:
                raise TiingoPriceError(
                    f"{provider_symbol} row {index} has invalid {field}."
                )
    if any(later <= earlier for earlier, later in zip(dates, dates[1:])):
        raise TiingoPriceError(
            f"{provider_symbol} dates are not strictly increasing."
        )
    if dates[0] > start_date + pd.Timedelta(days=7):
        raise TiingoPriceError(
            f"{provider_symbol} history starts too late: {dates[0]}."
        )
    if dates[-1] < end_date - pd.Timedelta(days=7):
        raise TiingoPriceError(
            f"{provider_symbol} history ends too early: {dates[-1]}."
        )
    return dates[0], dates[-1]


def validate_qualification_summary(
    summary: Mapping[str, Any],
    start_date: date,
    end_date: date,
    required_tickers: tuple[str, ...],
) -> dict[str, Mapping[str, Any]]:
    """Validate the passed local qualification summary and return ticker rows."""

    if summary.get("all_tickers_passed") is not True:
        raise TiingoPriceError("Tiingo qualification did not pass all tickers.")
    if summary.get("requested_start_date") != start_date.isoformat():
        raise TiingoPriceError("Tiingo qualification start date differs.")
    if summary.get("requested_end_date") != end_date.isoformat():
        raise TiingoPriceError("Tiingo qualification end date differs.")
    licence = summary.get("licence")
    if not isinstance(licence, Mapping) or (
        licence.get("classification") != "internal_use_only"
    ):
        raise TiingoPriceError("Tiingo licence boundary is missing or changed.")
    results = summary.get("results")
    if not isinstance(results, list):
        raise TiingoPriceError("Tiingo qualification results are missing.")
    by_ticker: dict[str, Mapping[str, Any]] = {}
    for row in results:
        if not isinstance(row, Mapping) or row.get("passed") is not True:
            raise TiingoPriceError("Tiingo qualification result is invalid.")
        ticker = str(row.get("canonical_ticker") or "").upper()
        by_ticker[ticker] = row
    if set(by_ticker) != set(required_tickers):
        raise TiingoPriceError(
            "Qualified ticker set differs from the project contract."
        )
    return by_ticker


def rows_to_adjusted_frame(
    rows: Sequence[Mapping[str, Any]],
    canonical_ticker: str,
    provider_symbol: str,
    fetched_at_utc: str,
) -> pd.DataFrame:
    """Convert validated Tiingo rows into model-compatible adjusted OHLCV."""

    output = pd.DataFrame(
        {
            "ticker": canonical_ticker,
            "session_date": [str(row["date"])[:10] for row in rows],
            "open": [float(row["adjOpen"]) for row in rows],
            "high": [float(row["adjHigh"]) for row in rows],
            "low": [float(row["adjLow"]) for row in rows],
            "close": [float(row["adjClose"]) for row in rows],
            "volume": [float(row["adjVolume"]) for row in rows],
            "dividend_cash": [float(row["divCash"]) for row in rows],
            "split_factor": [float(row["splitFactor"]) for row in rows],
            "source_provider": "tiingo_eod",
            "provider_symbol": provider_symbol,
            "cross_check_provider": "tiingo_ten_ticker_qualification",
            "cross_check_status": "qualified_exact_window_checksum_match",
            "verification_status": "primary_verified",
            "source_url": build_prices_url(
                provider_symbol,
                _parse_date(rows[0]["date"], "first date"),
                _parse_date(rows[-1]["date"], "last date"),
            ),
            "price_provider_role": "authenticated_internal_primary",
            "fetched_at_utc": fetched_at_utc,
            "price_provenance_note": (
                "Adjusted Tiingo EOD values for internal model development. "
                "Raw provider values must not be redistributed."
            ),
        }
    )
    return output


def collect_tiingo_prices(
    qualification_summary: Mapping[str, Any],
    required_tickers: tuple[str, ...],
    start_date: date,
    end_date: date,
    minimum_rows: int,
    api_token: str,
    cache_directory: Path,
    *,
    refresh_cache: bool = False,
    opener: Callable[..., Any] = urlopen,
    sleeper: Callable[[float], None] = time.sleep,
    progress_callback: ProgressCallback | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Collect every qualified ticker and require exact checksum continuity."""

    qualification_rows = validate_qualification_summary(
        qualification_summary,
        start_date,
        end_date,
        required_tickers,
    )
    pacer = RequestPacer(sleeper=sleeper)
    frames: list[pd.DataFrame] = []
    evidence: list[dict[str, Any]] = []

    for index, ticker in enumerate(required_tickers, start=1):
        qualified = qualification_rows[ticker]
        provider_symbol = str(qualified["provider_symbol"]).upper()
        expected_checksum = str(qualified["prices_sha256"])
        cache_path = (
            cache_directory
            / f"{ticker}_{start_date.isoformat()}_{end_date.isoformat()}.json"
        )
        _progress(
            progress_callback,
            f"[TIINGO {index}/{len(required_tickers)}] {ticker}",
        )
        rows, origin, attempts, delays = fetch_prices_with_cache(
            provider_symbol,
            start_date,
            end_date,
            api_token,
            cache_path,
            refresh_cache=refresh_cache,
            pacer=pacer,
            opener=opener,
            sleeper=sleeper,
            progress_callback=progress_callback,
        )
        observed_start, observed_end = validate_price_rows(
            rows,
            provider_symbol,
            start_date,
            end_date,
            minimum_rows,
        )
        canonical_bytes = _canonical_json_bytes(rows)
        observed_checksum = _sha256_bytes(canonical_bytes)
        if observed_checksum != expected_checksum:
            raise TiingoPriceError(
                f"{ticker} historical checksum differs from the passed "
                "qualification response."
            )
        fetched_at = datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        )
        frames.append(
            rows_to_adjusted_frame(
                rows,
                ticker,
                provider_symbol,
                fetched_at,
            )
        )
        evidence.append(
            asdict(
                PriceRequestEvidence(
                    canonical_ticker=ticker,
                    provider_symbol=provider_symbol,
                    request_url=build_prices_url(
                        provider_symbol,
                        start_date,
                        end_date,
                    ),
                    response_origin=origin,
                    cache_path=str(cache_path),
                    response_sha256=observed_checksum,
                    qualification_sha256=expected_checksum,
                    checksum_matches_qualification=True,
                    response_bytes=len(canonical_bytes),
                    accepted_rows=len(rows),
                    observed_start_date=observed_start.isoformat(),
                    observed_end_date=observed_end.isoformat(),
                    attempts=attempts,
                    retry_delays_seconds=delays,
                )
            )
        )
        _progress(
            progress_callback,
            f"[{ticker}] accepted {len(rows)} adjusted sessions",
        )

    prices = pd.concat(frames, ignore_index=True)
    if prices.duplicated(["ticker", "session_date"]).any():
        raise TiingoPriceError("Tiingo output grain is not ticker-session unique.")
    if set(prices["ticker"]) != set(required_tickers):
        raise TiingoPriceError("Tiingo output does not contain every ticker.")
    return prices.sort_values(["ticker", "session_date"]), evidence
