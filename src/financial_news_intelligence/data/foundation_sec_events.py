"""Collect official SEC company-disclosure events for movement modeling.

Purpose
-------
Build a dated, ticker-linked event table from SEC EDGAR metadata. The module
uses official ticker-to-CIK, recent-submission, and historical-submission
endpoints, bounded retries, local checksummed caches, and explicit rejection
evidence.

Inputs and source variables
---------------------------
- the curated company/ticker reference loaded by the orchestrator;
- a UTC event window;
- SEC company-ticker, recent-submission, and historical-submission JSON;
- optional verified caches under ``data/cache/market_data_foundation/sec``.

Grain, joins, and downstream use
--------------------------------
One accepted row represents one unique ``ticker + SEC document URL`` event.
The official SEC acceptance timestamp is the event clock. Ticker resolution
uses the SEC's official ticker-to-CIK mapping and therefore does not guess from
free text. The resulting rows feed local sentiment inference and future market
session mapping.

Assumptions, limitations, and fallbacks
---------------------------------------
Ownership and holder-reporting forms are excluded because they are not issuer
operating disclosures. Filing descriptions are normalized metadata, not copied
article bodies. Cached responses are reused only after JSON validation. Live
requests use a documented user agent, short timeouts, bounded retries, and a
consecutive-failure circuit breaker. Historical files are selected only when
the SEC-declared date range overlaps the model window. No synthetic ticker,
timestamp, URL, or company fact is created.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

SEC_TICKER_ENDPOINT = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_TEMPLATE = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
SEC_HISTORICAL_TEMPLATE = "https://data.sec.gov/submissions/{file_name}"
SEC_ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"
DEFAULT_SEC_USER_AGENT = os.environ.get(
    "SEC_USER_AGENT",
    "financial-news-stock-intelligence/1.0 ruturajmokashi@gmail.com",
)

# These filings describe insider or holder ownership rather than issuer events.
EXCLUDED_FORMS = {
    "3",
    "3/A",
    "4",
    "4/A",
    "5",
    "5/A",
    "13F-HR",
    "13F-HR/A",
    "SC 13D",
    "SC 13D/A",
    "SC 13G",
    "SC 13G/A",
}

ProgressCallback = Callable[[str], None]


class SecEventError(RuntimeError):
    """Raised when official SEC event evidence cannot satisfy the contract."""


@dataclass(frozen=True)
class FetchResult:
    """Return response bytes with cache and retry provenance."""

    content: bytes
    origin: str
    attempts: int
    retry_delays_seconds: tuple[float, ...]


@dataclass(frozen=True)
class RequestEvidence:
    """Record one SEC request without storing secrets or response values."""

    provider: str
    ticker: str
    request_url: str
    requested_at_utc: str
    status: str
    origin: str
    cache_path: str
    response_sha256: str
    response_bytes: int
    parsed_records: int
    accepted_records: int
    rejected_records: int
    attempts: int
    retry_delays_seconds: tuple[float, ...]
    error_type: str
    error_message: str


def _progress(callback: ProgressCallback | None, message: str) -> None:
    """Print one immediately visible progress message when requested."""

    if callback is not None:
        callback(message)


def _sha256_bytes(content: bytes) -> str:
    """Return a hexadecimal SHA-256 digest for provider response evidence."""

    return hashlib.sha256(content).hexdigest()


def _utc_text(value: datetime) -> str:
    """Serialize one timezone-aware datetime in stable UTC ISO format."""

    if value.tzinfo is None:
        raise SecEventError("SEC request evidence requires timezone-aware time.")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_timestamp(value: object) -> pd.Timestamp | None:
    """Parse SEC compact or ISO timestamp text as UTC."""

    text = str(value or "").strip()
    if not text:
        return None
    for date_format in ("%Y%m%d%H%M%S", "%Y%m%dT%H%M%SZ"):
        try:
            parsed = datetime.strptime(text, date_format).replace(
                tzinfo=timezone.utc
            )
            return pd.Timestamp(parsed)
        except ValueError:
            continue
    parsed = pd.to_datetime(text, utc=True, errors="coerce")
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed)


def _retry_after_seconds(error: HTTPError) -> float | None:
    """Parse Retry-After seconds or HTTP date from one SEC response."""

    value = error.headers.get("Retry-After") if error.headers else None
    if not value:
        return None
    text = str(value).strip()
    try:
        return max(0.0, float(text))
    except ValueError:
        pass
    try:
        retry_at = parsedate_to_datetime(text)
    except (TypeError, ValueError, OverflowError):
        return None
    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=timezone.utc)
    return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())


def _retryable(error: Exception) -> bool:
    """Return whether a failed SEC request may succeed after a short wait."""

    if isinstance(error, HTTPError):
        return error.code == 429 or 500 <= error.code <= 599
    return isinstance(error, (URLError, TimeoutError, SecEventError))


def _retry_delay(error: Exception, attempt: int) -> float:
    """Choose one bounded deterministic retry delay."""

    if isinstance(error, HTTPError) and error.code == 429:
        provider_delay = _retry_after_seconds(error)
        if provider_delay is not None:
            return min(10.0, max(1.0, provider_delay))
    return min(10.0, 2.0 * (2 ** (attempt - 1)))


def fetch_json_with_cache(
    request_url: str,
    cache_path: Path,
    *,
    refresh_cache: bool = False,
    attempts: int = 2,
    timeout_seconds: int = 20,
    opener: Callable[..., Any] = urlopen,
    sleeper: Callable[[float], None] = time.sleep,
    progress_callback: ProgressCallback | None = None,
    progress_label: str = "SEC",
) -> FetchResult:
    """Fetch one SEC JSON object with safe cache reuse and bounded retries."""

    if attempts < 1:
        raise SecEventError("SEC attempts must be at least one.")
    if timeout_seconds < 1:
        raise SecEventError("SEC timeout must be positive.")

    if cache_path.exists() and not refresh_cache:
        if cache_path.is_symlink() or not cache_path.is_file():
            raise SecEventError(f"Unsafe SEC cache path: {cache_path}")
        content = cache_path.read_bytes()
        payload = json.loads(content.decode("utf-8"))
        if not isinstance(payload, dict):
            raise SecEventError("Cached SEC response must contain a JSON object.")
        _progress(progress_callback, f"{progress_label}: using verified cache")
        return FetchResult(content, "cache", 0, ())

    last_error: Exception | None = None
    delays: list[float] = []
    headers = {
        "User-Agent": DEFAULT_SEC_USER_AGENT,
        "Accept": "application/json",
    }

    for attempt in range(1, attempts + 1):
        _progress(
            progress_callback,
            f"{progress_label}: request attempt {attempt}/{attempts}",
        )
        request = Request(request_url, headers=headers)
        try:
            with opener(request, timeout=timeout_seconds) as response:
                content = response.read()
            if not content:
                raise SecEventError("SEC returned an empty response.")
            payload = json.loads(content.decode("utf-8"))
            if not isinstance(payload, dict):
                raise SecEventError("SEC response must contain a JSON object.")

            # Cache only after full JSON validation, then replace atomically.
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
            temporary.write_bytes(content)
            os.chmod(temporary, 0o600)
            temporary.replace(cache_path)
            _progress(progress_callback, f"{progress_label}: response cached")
            return FetchResult(content, "live", attempt, tuple(delays))
        except Exception as exc:  # noqa: BLE001 - retained as QA evidence.
            last_error = exc
            if attempt >= attempts or not _retryable(exc):
                break
            delay = _retry_delay(exc, attempt)
            delays.append(delay)
            _progress(
                progress_callback,
                f"{progress_label}: retrying after {delay:.1f} seconds",
            )
            sleeper(delay)

    raise SecEventError(
        f"SEC request failed after {attempts} attempts: "
        f"{type(last_error).__name__}: {last_error}"
    ) from last_error


def parse_ticker_mapping(content: bytes) -> dict[str, int]:
    """Parse the official SEC ticker file into uppercase ticker-to-CIK values."""

    try:
        payload = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SecEventError(f"Invalid SEC ticker JSON: {exc}") from exc

    rows = payload.values() if isinstance(payload, dict) else payload
    if not isinstance(rows, (list, tuple, dict_values_type())):
        raise SecEventError("SEC ticker response must be a JSON collection.")

    mapping: dict[str, int] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        ticker = str(row.get("ticker") or "").strip().upper()
        cik_value = row.get("cik_str") or row.get("cik")
        try:
            cik = int(cik_value)
        except (TypeError, ValueError):
            continue
        if ticker and cik > 0:
            mapping[ticker] = cik
    if not mapping:
        raise SecEventError("SEC ticker response contained no valid mappings.")
    return mapping


def dict_values_type() -> type:
    """Return the runtime type used by ``dict.values()`` for validation."""

    return type({}.values())


def _column_rows(columns: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Convert one SEC compact column collection into row dictionaries.

    The recent object and every historical submissions file use the same
    column-oriented structure. Keeping one conversion function prevents the
    historical path from silently diverging from the recent path.
    """

    accessions = columns.get("accessionNumber")
    if not isinstance(accessions, list):
        return []

    rows: list[dict[str, Any]] = []
    for index in range(len(accessions)):
        row: dict[str, Any] = {}
        for key, values in columns.items():
            if isinstance(values, list) and index < len(values):
                row[key] = values[index]
        rows.append(row)
    return rows


def _submission_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return rows from a recent response or a historical submissions file."""

    filings = payload.get("filings")
    recent = filings.get("recent") if isinstance(filings, Mapping) else None
    if isinstance(recent, Mapping):
        return _column_rows(recent)

    # Historical files referenced by ``filings.files`` contain the compact
    # columns at the JSON root rather than under ``filings.recent``.
    return _column_rows(payload)


def _parse_date(value: object) -> pd.Timestamp | None:
    """Parse one SEC date-range value as a normalized UTC timestamp."""

    parsed = pd.to_datetime(str(value or "").strip(), utc=True, errors="coerce")
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed)


def parse_historical_file_references(
    content: bytes,
    start_utc: datetime,
    end_utc: datetime,
) -> list[dict[str, str]]:
    """Select SEC historical files whose declared dates overlap the window.

    The SEC main submissions response contains ``filings.files`` when older
    history is stored in additional JSON files. Each accepted reference keeps
    the official file name and declared date bounds for request provenance.
    """

    try:
        payload = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SecEventError(f"Invalid SEC submissions JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise SecEventError("SEC submissions response must contain an object.")

    filings = payload.get("filings")
    files = filings.get("files") if isinstance(filings, Mapping) else None
    if not isinstance(files, list):
        return []

    requested_start = pd.Timestamp(start_utc)
    requested_end = pd.Timestamp(end_utc)
    selected: list[dict[str, str]] = []
    for item in files:
        if not isinstance(item, Mapping):
            continue
        file_name = str(item.get("name") or "").strip()
        filing_from = _parse_date(item.get("filingFrom"))
        filing_to = _parse_date(item.get("filingTo"))

        # Reject path traversal and unexpected file types before constructing
        # a provider URL or local cache path.
        if (
            not file_name
            or Path(file_name).name != file_name
            or not file_name.startswith("CIK")
            or not file_name.endswith(".json")
        ):
            continue
        if filing_from is None or filing_to is None:
            continue
        if filing_from >= requested_end or filing_to < requested_start:
            continue
        selected.append(
            {
                "name": file_name,
                "filing_from": filing_from.strftime("%Y-%m-%d"),
                "filing_to": filing_to.strftime("%Y-%m-%d"),
            }
        )

    return sorted(
        selected,
        key=lambda value: (value["filing_from"], value["name"]),
    )


def parse_submissions(
    content: bytes,
    company: str,
    ticker: str,
    cik: int,
    start_utc: datetime,
    end_utc: datetime,
    request_url: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse official SEC submissions into accepted and rejected event rows."""

    try:
        payload = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SecEventError(f"Invalid SEC submissions JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise SecEventError("SEC submissions response must contain an object.")

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for item in _submission_rows(payload):
        form = str(item.get("form") or "").strip().upper()
        acceptance = _parse_timestamp(item.get("acceptanceDateTime"))
        accession = str(item.get("accessionNumber") or "").strip()
        document = str(item.get("primaryDocument") or "").strip()
        description = str(item.get("primaryDocDescription") or "").strip()

        reason: str | None = None
        if form in EXCLUDED_FORMS:
            reason = "excluded_ownership_or_holder_form"
        elif not form:
            reason = "missing_form"
        elif acceptance is None:
            reason = "missing_acceptance_timestamp"
        elif not (pd.Timestamp(start_utc) <= acceptance < pd.Timestamp(end_utc)):
            reason = "outside_requested_window"
        elif not accession or not document:
            reason = "missing_accession_or_primary_document"

        if reason is not None:
            rejected.append(
                {
                    "query_company": company,
                    "query_ticker": ticker,
                    "form": form,
                    "request_url": request_url,
                    "rejection_reason": reason,
                }
            )
            continue

        compact_accession = accession.replace("-", "")
        source_url = (
            f"{SEC_ARCHIVES_BASE}/{cik}/{compact_accession}/{document}"
        )
        normalized_description = description or f"SEC Form {form}"
        headline = f"{company}: {normalized_description}"
        article_id = hashlib.sha256(
            f"{ticker}|{source_url}".encode("utf-8")
        ).hexdigest()[:24]
        accepted.append(
            {
                "article_id": article_id,
                "ticker": ticker,
                "company": company,
                "published_at_utc": acceptance.isoformat().replace(
                    "+00:00", "Z"
                ),
                "timestamp_type": "sec_acceptance_datetime",
                "text": headline,
                "headline": headline,
                "source_name": "U.S. Securities and Exchange Commission",
                "source_url": source_url,
                "news_provider": "sec_edgar_submissions_api",
                "provider_role": "primary_company_disclosure",
                "verification_status": "primary_verified",
                "ticker_resolution_method": "sec_official_ticker_to_cik",
                "matched_alias": ticker,
                "language": "English",
                "source_country": "United States",
                "request_url": request_url,
                "provenance_note": (
                    "Timestamp, accession, form, and document URL are official "
                    "SEC metadata. Text is a normalized metadata descriptor."
                ),
            }
        )

    return pd.DataFrame(accepted), pd.DataFrame(rejected)


def _append_request_evidence(
    requests: list[dict[str, Any]],
    *,
    provider: str,
    ticker: str,
    request_url: str,
    cache_path: Path,
    fetched: FetchResult,
    accepted_count: int,
    rejected_count: int,
) -> None:
    """Append one completed SEC request record with checksum provenance."""

    requests.append(
        asdict(
            RequestEvidence(
                provider=provider,
                ticker=ticker,
                request_url=request_url,
                requested_at_utc=_utc_text(datetime.now(timezone.utc)),
                status="completed",
                origin=fetched.origin,
                cache_path=str(cache_path),
                response_sha256=_sha256_bytes(fetched.content),
                response_bytes=len(fetched.content),
                parsed_records=accepted_count + rejected_count,
                accepted_records=accepted_count,
                rejected_records=rejected_count,
                attempts=fetched.attempts,
                retry_delays_seconds=fetched.retry_delays_seconds,
                error_type="",
                error_message="",
            )
        )
    )


def _pace_before_live_request(
    cache_path: Path,
    refresh_cache: bool,
    last_live_request: float | None,
    sleeper: Callable[[float], None],
) -> float | None:
    """Respect SEC fair-access pacing only when a network request is needed."""

    needs_live = refresh_cache or not cache_path.exists()
    if not needs_live:
        return last_live_request
    if last_live_request is not None:
        remaining = 0.2 - (time.monotonic() - last_live_request)
        if remaining > 0:
            sleeper(remaining)
    return time.monotonic()


def collect_sec_events(
    reference: pd.DataFrame,
    required_tickers: tuple[str, ...],
    start_utc: datetime,
    end_utc: datetime,
    cache_directory: Path,
    *,
    refresh_cache: bool = False,
    opener: Callable[..., Any] = urlopen,
    sleeper: Callable[[float], None] = time.sleep,
    progress_callback: ProgressCallback | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    """Collect recent and historical SEC events for every qualified ticker.

    The main CIK response is always parsed first. Its ``filings.files`` list is
    then followed for every SEC-declared file overlapping the 2015--2020 model
    window. This is required because the main response contains only the most
    recent year or 1,000 filings and can otherwise yield zero historical rows.
    """

    selected = reference[reference["ticker"].isin(required_tickers)].copy()
    observed = tuple(selected["ticker"].astype(str))
    if set(observed) != set(required_tickers):
        missing = sorted(set(required_tickers) - set(observed))
        raise SecEventError(
            f"Ticker reference is missing qualified tickers: {missing}"
        )

    ticker_cache = cache_directory / "company_tickers.json"
    ticker_fetch = fetch_json_with_cache(
        SEC_TICKER_ENDPOINT,
        ticker_cache,
        refresh_cache=refresh_cache,
        opener=opener,
        sleeper=sleeper,
        progress_callback=progress_callback,
        progress_label="[SEC ticker map]",
    )
    cik_mapping = parse_ticker_mapping(ticker_fetch.content)
    requests: list[dict[str, Any]] = []
    _append_request_evidence(
        requests,
        provider="sec_company_ticker_file",
        ticker="ALL",
        request_url=SEC_TICKER_ENDPOINT,
        cache_path=ticker_cache,
        fetched=ticker_fetch,
        accepted_count=len(cik_mapping),
        rejected_count=0,
    )

    accepted_parts: list[pd.DataFrame] = []
    rejected_parts: list[pd.DataFrame] = []
    consecutive_failures = 0
    last_live_request: float | None = None

    for index, row in enumerate(selected.itertuples(index=False), start=1):
        ticker = str(row.ticker)
        company = str(row.company)
        label = f"[SEC {index}/{len(selected)}] {ticker}"
        cik = cik_mapping.get(ticker)
        if cik is None:
            raise SecEventError(f"SEC ticker map has no CIK for {ticker}.")

        request_url = SEC_SUBMISSIONS_TEMPLATE.format(cik=cik)
        cache_path = cache_directory / f"CIK{cik:010d}.json"
        last_live_request = _pace_before_live_request(
            cache_path,
            refresh_cache,
            last_live_request,
            sleeper,
        )

        try:
            main_fetch = fetch_json_with_cache(
                request_url,
                cache_path,
                refresh_cache=refresh_cache,
                opener=opener,
                sleeper=sleeper,
                progress_callback=progress_callback,
                progress_label=label,
            )
            recent_accepted, recent_rejected = parse_submissions(
                main_fetch.content,
                company,
                ticker,
                cik,
                start_utc,
                end_utc,
                request_url,
            )
            if not recent_accepted.empty:
                accepted_parts.append(recent_accepted)
            if not recent_rejected.empty:
                rejected_parts.append(recent_rejected)
            _append_request_evidence(
                requests,
                provider="sec_edgar_submissions_api",
                ticker=ticker,
                request_url=request_url,
                cache_path=cache_path,
                fetched=main_fetch,
                accepted_count=len(recent_accepted),
                rejected_count=len(recent_rejected),
            )

            historical_files = parse_historical_file_references(
                main_fetch.content,
                start_utc,
                end_utc,
            )
            _progress(
                progress_callback,
                f"{label}: {len(historical_files)} historical files overlap window",
            )
            ticker_accepted = len(recent_accepted)

            for history_index, file_info in enumerate(
                historical_files,
                start=1,
            ):
                file_name = file_info["name"]
                historical_url = SEC_HISTORICAL_TEMPLATE.format(
                    file_name=file_name
                )
                historical_cache = (
                    cache_directory / "historical" / file_name
                )
                history_label = (
                    f"{label} history {history_index}/{len(historical_files)}"
                )
                last_live_request = _pace_before_live_request(
                    historical_cache,
                    refresh_cache,
                    last_live_request,
                    sleeper,
                )
                historical_fetch = fetch_json_with_cache(
                    historical_url,
                    historical_cache,
                    refresh_cache=refresh_cache,
                    opener=opener,
                    sleeper=sleeper,
                    progress_callback=progress_callback,
                    progress_label=history_label,
                )
                historical_accepted, historical_rejected = parse_submissions(
                    historical_fetch.content,
                    company,
                    ticker,
                    cik,
                    start_utc,
                    end_utc,
                    historical_url,
                )
                if not historical_accepted.empty:
                    accepted_parts.append(historical_accepted)
                if not historical_rejected.empty:
                    rejected_parts.append(historical_rejected)
                ticker_accepted += len(historical_accepted)
                _append_request_evidence(
                    requests,
                    provider="sec_edgar_historical_submissions_api",
                    ticker=ticker,
                    request_url=historical_url,
                    cache_path=historical_cache,
                    fetched=historical_fetch,
                    accepted_count=len(historical_accepted),
                    rejected_count=len(historical_rejected),
                )
                _progress(
                    progress_callback,
                    f"{history_label}: accepted {len(historical_accepted)} disclosures",
                )

            consecutive_failures = 0
            _progress(
                progress_callback,
                f"{label}: accepted {ticker_accepted} total disclosures",
            )
        except Exception as exc:  # noqa: BLE001 - retained as QA evidence.
            consecutive_failures += 1
            requests.append(
                asdict(
                    RequestEvidence(
                        provider="sec_edgar_submissions_api",
                        ticker=ticker,
                        request_url=request_url,
                        requested_at_utc=_utc_text(datetime.now(timezone.utc)),
                        status="failed",
                        origin="none",
                        cache_path=str(cache_path),
                        response_sha256="",
                        response_bytes=0,
                        parsed_records=0,
                        accepted_records=0,
                        rejected_records=0,
                        attempts=2,
                        retry_delays_seconds=(),
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )
                )
            )
            _progress(progress_callback, f"{label}: failed: {exc}")
            if consecutive_failures >= 2:
                raise SecEventError(
                    "SEC circuit breaker opened after two consecutive failures."
                ) from exc

    if not accepted_parts:
        raise SecEventError("No verified SEC disclosures were accepted.")

    accepted = pd.concat(accepted_parts, ignore_index=True)
    accepted["published_at_utc"] = pd.to_datetime(
        accepted["published_at_utc"], utc=True
    )
    accepted = (
        accepted.sort_values(
            ["ticker", "source_url", "published_at_utc", "article_id"]
        )
        .drop_duplicates(["ticker", "source_url"], keep="first")
        .sort_values(["published_at_utc", "ticker", "article_id"])
        .reset_index(drop=True)
    )
    accepted["published_at_utc"] = accepted["published_at_utc"].dt.strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    # Every qualified ticker must contribute real SEC evidence. This prevents a
    # large event total from hiding historical-pagination failures for several
    # companies and protects the later movement model from ticker-selection bias.
    accepted_tickers = set(accepted["ticker"].astype(str))
    missing_event_tickers = sorted(set(required_tickers) - accepted_tickers)
    if missing_event_tickers:
        raise SecEventError(
            "SEC historical collection produced no in-window disclosures for: "
            f"{missing_event_tickers}"
        )

    rejected = (
        pd.concat(rejected_parts, ignore_index=True, sort=False)
        if rejected_parts
        else pd.DataFrame(columns=["query_ticker", "rejection_reason"])
    )
    return accepted, rejected, requests
