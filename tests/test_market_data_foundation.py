"""Focused tests for the final SEC and Tiingo foundation contract.

Every test follows Prepare data -> Run function -> Check result. Network calls,
provider responses, model inference, and project artifacts use deterministic
fixtures so the tests remain fast, free, and reproducible.
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.error import HTTPError

import numpy as np
import pandas as pd
import pytest

from financial_news_intelligence.data import market_data_foundation
from financial_news_intelligence.data.foundation_market_sessions import (
    add_session_timestamps,
    map_articles_to_sessions,
)
from financial_news_intelligence.data.foundation_sec_events import (
    SEC_HISTORICAL_TEMPLATE,
    SEC_SUBMISSIONS_TEMPLATE,
    SEC_TICKER_ENDPOINT,
    SecEventError,
    collect_sec_events,
    fetch_json_with_cache,
    parse_historical_file_references,
    parse_submissions,
    parse_ticker_mapping,
)
from financial_news_intelligence.data.foundation_tiingo_prices import (
    RequestPacer,
    TiingoPriceError,
    build_prices_url,
    collect_tiingo_prices,
    rows_to_adjusted_frame,
    validate_price_rows,
    validate_qualification_summary,
)
from financial_news_intelligence.data.foundation_ticker_resolution import (
    load_ticker_reference,
    resolve_title,
)
from financial_news_intelligence.data.market_data_foundation import (
    FoundationConfig,
    FoundationError,
    REQUIRED_TICKERS,
    _readiness_report,
    add_movement_labels,
)


def _price_rows(count: int = 1370) -> list[dict[str, object]]:
    """Create strictly ordered complete Tiingo rows for focused tests."""

    dates = pd.bdate_range("2015-01-02", periods=count)
    rows: list[dict[str, object]] = []
    for index, observed in enumerate(dates):
        value = 100.0 + index / 10.0
        rows.append(
            {
                "date": observed.strftime("%Y-%m-%dT00:00:00.000Z"),
                "open": value,
                "high": value + 1.0,
                "low": value - 1.0,
                "close": value + 0.5,
                "volume": 1000 + index,
                "adjOpen": value,
                "adjHigh": value + 1.0,
                "adjLow": value - 1.0,
                "adjClose": value + 0.5,
                "adjVolume": 1000 + index,
                "divCash": 0.0,
                "splitFactor": 1.0,
            }
        )
    return rows


def _qualification(rows: list[dict[str, object]]) -> dict[str, object]:
    """Create one ten-ticker passed qualification summary."""

    canonical = json.dumps(
        rows,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    import hashlib

    checksum = hashlib.sha256(canonical).hexdigest()
    return {
        "all_tickers_passed": True,
        "requested_start_date": "2015-01-01",
        "requested_end_date": "2020-04-01",
        "licence": {"classification": "internal_use_only"},
        "results": [
            {
                "canonical_ticker": ticker,
                "provider_symbol": ticker,
                "prices_sha256": checksum,
                "passed": True,
            }
            for ticker in REQUIRED_TICKERS
        ],
    }


def _reference(tmp_path: Path) -> Path:
    """Write the exact curated ten-ticker reference."""

    path = tmp_path / "company_tickers.csv"
    pd.DataFrame(
        [
            {
                "company": f"{ticker} Corporation",
                "ticker": ticker,
                "aliases": ticker,
            }
            for ticker in REQUIRED_TICKERS
        ]
    ).to_csv(path, index=False)
    return path


def test_load_ticker_reference_preserves_exact_tickers(tmp_path: Path) -> None:
    """Accept a unique curated reference without inventing symbols."""

    # Prepare data.
    path = _reference(tmp_path)
    # Run function.
    frame = load_ticker_reference(path)
    # Check result.
    assert set(frame["ticker"]) == set(REQUIRED_TICKERS)


def test_resolve_title_rejects_ambiguous_aliases(tmp_path: Path) -> None:
    """Reject a title matching more than one curated company."""

    # Prepare data.
    path = tmp_path / "reference.csv"
    pd.DataFrame(
        [
            {"company": "Alpha", "ticker": "AAA", "aliases": "Cloud"},
            {"company": "Beta", "ticker": "BBB", "aliases": "Cloud"},
        ]
    ).to_csv(path, index=False)
    reference = load_ticker_reference(path)
    # Run function.
    result = resolve_title("Cloud results", reference)
    # Check result.
    assert result.status == "rejected"


def test_parse_ticker_mapping_accepts_official_shape() -> None:
    """Parse SEC ticker-to-CIK rows from the official object structure."""

    # Prepare data.
    content = json.dumps(
        {"0": {"ticker": "AAPL", "cik_str": 320193}}
    ).encode()
    # Run function.
    result = parse_ticker_mapping(content)
    # Check result.
    assert result == {"AAPL": 320193}


def test_parse_ticker_mapping_rejects_empty_payload() -> None:
    """Fail closed when SEC returns no valid ticker mapping."""

    # Prepare data.
    content = b"{}"
    # Run function.
    # Check result.
    with pytest.raises(SecEventError):
        parse_ticker_mapping(content)


def test_parse_submissions_accepts_in_window_disclosure() -> None:
    """Create one verified event from official SEC filing metadata."""

    # Prepare data.
    payload = {
        "filings": {
            "recent": {
                "accessionNumber": ["0001-20-000001"],
                "form": ["8-K"],
                "acceptanceDateTime": ["20190304120000"],
                "primaryDocument": ["event.htm"],
                "primaryDocDescription": ["Current report"],
            }
        }
    }
    # Run function.
    accepted, rejected = parse_submissions(
        json.dumps(payload).encode(),
        "Apple Inc.",
        "AAPL",
        320193,
        datetime(2015, 2, 2, tzinfo=timezone.utc),
        datetime(2020, 4, 1, tzinfo=timezone.utc),
        "https://example.test/sec",
    )
    # Check result.
    assert len(accepted) == 1 and rejected.empty
    assert accepted.iloc[0]["timestamp_type"] == "sec_acceptance_datetime"


def test_parse_historical_references_selects_overlapping_files() -> None:
    """Follow only SEC-declared historical files overlapping the model window."""

    # Prepare data.
    payload = {
        "filings": {
            "files": [
                {
                    "name": "CIK0000320193-submissions-001.json",
                    "filingFrom": "2010-01-01",
                    "filingTo": "2019-12-31",
                },
                {
                    "name": "CIK0000320193-submissions-002.json",
                    "filingFrom": "1999-01-01",
                    "filingTo": "2009-12-31",
                },
            ]
        }
    }
    # Run function.
    result = parse_historical_file_references(
        json.dumps(payload).encode(),
        datetime(2015, 1, 1, tzinfo=timezone.utc),
        datetime(2020, 4, 1, tzinfo=timezone.utc),
    )
    # Check result.
    assert [item["name"] for item in result] == [
        "CIK0000320193-submissions-001.json"
    ]


def test_parse_historical_submissions_accepts_root_columns() -> None:
    """Parse the compact root columns used by SEC historical JSON files."""

    # Prepare data.
    payload = {
        "accessionNumber": ["0000320193-19-000001"],
        "form": ["8-K"],
        "acceptanceDateTime": ["20190304120000"],
        "primaryDocument": ["event.htm"],
        "primaryDocDescription": ["Current report"],
    }
    # Run function.
    accepted, rejected = parse_submissions(
        json.dumps(payload).encode(),
        "Apple Inc.",
        "AAPL",
        320193,
        datetime(2015, 1, 1, tzinfo=timezone.utc),
        datetime(2020, 4, 1, tzinfo=timezone.utc),
        "https://data.sec.gov/submissions/history.json",
    )
    # Check result.
    assert len(accepted) == 1 and rejected.empty
    assert accepted.iloc[0]["ticker"] == "AAPL"


def test_historical_reference_rejects_unsafe_file_name() -> None:
    """Reject SEC file references that could escape the cache directory."""

    # Prepare data.
    payload = {
        "filings": {
            "files": [
                {
                    "name": "../unsafe.json",
                    "filingFrom": "2015-01-01",
                    "filingTo": "2019-12-31",
                }
            ]
        }
    }
    # Run function.
    result = parse_historical_file_references(
        json.dumps(payload).encode(),
        datetime(2015, 1, 1, tzinfo=timezone.utc),
        datetime(2020, 4, 1, tzinfo=timezone.utc),
    )
    # Check result.
    assert result == []


def test_collect_sec_events_follows_historical_files(tmp_path: Path) -> None:
    """Fetch historical SEC pagination when recent rows miss the model window."""

    # Prepare data.
    reference = pd.DataFrame(
        [{"company": "Apple Inc.", "ticker": "AAPL", "aliases": "Apple"}]
    )
    historical_name = "CIK0000320193-submissions-001.json"
    main_url = SEC_SUBMISSIONS_TEMPLATE.format(cik=320193)
    historical_url = SEC_HISTORICAL_TEMPLATE.format(file_name=historical_name)
    responses = {
        SEC_TICKER_ENDPOINT: {"0": {"ticker": "AAPL", "cik_str": 320193}},
        main_url: {
            "filings": {
                "recent": {
                    "accessionNumber": ["0000320193-24-000001"],
                    "form": ["8-K"],
                    "acceptanceDateTime": ["20240101120000"],
                    "primaryDocument": ["recent.htm"],
                    "primaryDocDescription": ["Recent report"],
                },
                "files": [
                    {
                        "name": historical_name,
                        "filingFrom": "2010-01-01",
                        "filingTo": "2019-12-31",
                    }
                ],
            }
        },
        historical_url: {
            "accessionNumber": ["0000320193-19-000001"],
            "form": ["8-K"],
            "acceptanceDateTime": ["20190304120000"],
            "primaryDocument": ["event.htm"],
            "primaryDocDescription": ["Current report"],
        },
    }

    class Response:
        """Provide the context-manager interface expected by urllib."""

        def __init__(self, payload: object) -> None:
            """Encode one deterministic JSON payload for the fake response."""

            self.content = json.dumps(payload).encode()

        def __enter__(self) -> "Response":
            """Return this fake response to the urllib context manager."""

            return self

        def __exit__(self, *_args: object) -> None:
            """Close the fake context without suppressing exceptions."""

            return None

        def read(self) -> bytes:
            """Return the prepared response bytes."""

            return self.content

    def opener(request: object, **_kwargs: object) -> Response:
        """Return deterministic SEC fixtures by request URL."""

        return Response(responses[request.full_url])

    # Run function.
    accepted, _rejected, requests = collect_sec_events(
        reference,
        ("AAPL",),
        datetime(2015, 1, 1, tzinfo=timezone.utc),
        datetime(2020, 4, 1, tzinfo=timezone.utc),
        tmp_path / "sec",
        opener=opener,
        sleeper=lambda _seconds: None,
    )
    # Check result.
    assert len(accepted) == 1
    assert accepted.iloc[0]["ticker"] == "AAPL"
    assert any(
        item["provider"] == "sec_edgar_historical_submissions_api"
        for item in requests
    )
    assert (tmp_path / "sec/historical" / historical_name).exists()


def test_parse_submissions_rejects_ownership_form() -> None:
    """Exclude insider ownership filings from issuer-event evidence."""

    # Prepare data.
    payload = {
        "filings": {
            "recent": {
                "accessionNumber": ["0001-20-000001"],
                "form": ["4"],
                "acceptanceDateTime": ["20190304120000"],
                "primaryDocument": ["owner.htm"],
                "primaryDocDescription": ["Ownership"],
            }
        }
    }
    # Run function.
    accepted, rejected = parse_submissions(
        json.dumps(payload).encode(),
        "Apple Inc.",
        "AAPL",
        320193,
        datetime(2015, 2, 2, tzinfo=timezone.utc),
        datetime(2020, 4, 1, tzinfo=timezone.utc),
        "https://example.test/sec",
    )
    # Check result.
    assert accepted.empty
    assert rejected.iloc[0]["rejection_reason"].startswith("excluded_")


def test_sec_cache_reuse_avoids_network(tmp_path: Path) -> None:
    """Reuse a valid SEC cache without calling the opener."""

    # Prepare data.
    cache = tmp_path / "sec.json"
    cache.write_text('{"ok": true}', encoding="utf-8")
    called = False

    def opener(*_args: object, **_kwargs: object) -> object:
        """Record an unexpected network call."""

        nonlocal called
        called = True
        raise AssertionError("network should not run")

    # Run function.
    result = fetch_json_with_cache(
        "https://example.test",
        cache,
        opener=opener,
    )
    # Check result.
    assert result.origin == "cache" and called is False


def test_build_prices_url_contains_fixed_window() -> None:
    """Build a token-free documented Tiingo historical URL."""

    # Prepare data.
    start = date(2015, 1, 1)
    end = date(2020, 4, 1)
    # Run function.
    url = build_prices_url("AAPL", start, end)
    # Check result.
    assert "startDate=2015-01-01" in url
    assert "endDate=2020-04-01" in url
    assert "token" not in url.lower()


def test_request_pacer_waits_only_remaining_interval() -> None:
    """Space requests without adding unnecessary delay."""

    # Prepare data.
    times = iter([10.0, 11.0, 12.0])
    sleeps: list[float] = []
    pacer = RequestPacer(
        minimum_interval_seconds=2.0,
        clock=lambda: next(times),
        sleeper=sleeps.append,
    )
    # Run function.
    pacer.wait()
    pacer.wait()
    # Check result.
    assert sleeps == [1.0]


def test_validate_price_rows_accepts_complete_adjusted_history() -> None:
    """Accept ordered unique rows with all raw and adjusted fields."""

    # Prepare data.
    rows = _price_rows()
    # Run function.
    observed_start, observed_end = validate_price_rows(
        rows,
        "AAPL",
        date(2015, 1, 1),
        date(2018, 11, 1),
        1000,
    )
    # Check result.
    assert observed_start.isoformat() == "2015-01-02"
    assert observed_end > observed_start


def test_validate_price_rows_rejects_duplicate_date() -> None:
    """Reject duplicate sessions because downstream joins become ambiguous."""

    # Prepare data.
    rows = _price_rows()
    rows[1]["date"] = rows[0]["date"]
    # Run function.
    # Check result.
    with pytest.raises(TiingoPriceError):
        validate_price_rows(
            rows,
            "AAPL",
            date(2015, 1, 1),
            date(2018, 11, 1),
            1000,
        )


def test_validate_price_rows_rejects_invalid_adjusted_close() -> None:
    """Reject non-positive adjusted values before model use."""

    # Prepare data.
    rows = _price_rows()
    rows[10]["adjClose"] = 0.0
    # Run function.
    # Check result.
    with pytest.raises(TiingoPriceError):
        validate_price_rows(
            rows,
            "AAPL",
            date(2015, 1, 1),
            date(2018, 11, 1),
            1000,
        )


def test_validate_qualification_requires_internal_use_boundary() -> None:
    """Reject a summary that removes the Tiingo licence boundary."""

    # Prepare data.
    rows = _price_rows()
    summary = _qualification(rows)
    summary["licence"] = {"classification": "redistributable"}
    # Run function.
    # Check result.
    with pytest.raises(TiingoPriceError):
        validate_qualification_summary(
            summary,
            date(2015, 1, 1),
            date(2020, 4, 1),
            REQUIRED_TICKERS,
        )


def test_validate_qualification_requires_exact_ticker_set() -> None:
    """Reject missing or unexpected qualification tickers."""

    # Prepare data.
    summary = _qualification(_price_rows())
    summary["results"] = summary["results"][:-1]
    # Run function.
    # Check result.
    with pytest.raises(TiingoPriceError):
        validate_qualification_summary(
            summary,
            date(2015, 1, 1),
            date(2020, 4, 1),
            REQUIRED_TICKERS,
        )


def test_rows_to_adjusted_frame_uses_adjusted_values() -> None:
    """Use adjusted OHLCV instead of raw provider values."""

    # Prepare data.
    rows = _price_rows(3)
    rows[0]["open"] = 999.0
    rows[0]["adjOpen"] = 50.0
    # Run function.
    frame = rows_to_adjusted_frame(
        rows,
        "AAPL",
        "AAPL",
        "2026-07-03T00:00:00Z",
    )
    # Check result.
    assert frame.iloc[0]["open"] == 50.0
    assert frame.iloc[0]["source_provider"] == "tiingo_eod"


def test_collect_prices_rejects_checksum_drift(tmp_path: Path) -> None:
    """Fail when live history differs from the passed qualification response."""

    # Prepare data.
    rows = _price_rows()
    summary = _qualification(rows)
    summary["results"][0]["prices_sha256"] = "0" * 64
    cache = tmp_path / "private"
    cache.mkdir()
    first = cache / "NVDA_2015-01-01_2020-04-01.json"
    first.write_text(json.dumps(rows), encoding="utf-8")
    os.chmod(first, 0o600)
    # Run function.
    # Check result.
    with pytest.raises(TiingoPriceError, match="checksum differs"):
        collect_tiingo_prices(
            summary,
            REQUIRED_TICKERS,
            date(2015, 1, 1),
            date(2020, 4, 1),
            1000,
            "secret",
            cache,
        )


def test_add_session_timestamps_handles_dst() -> None:
    """Convert New York session clocks to correct UTC offsets."""

    # Prepare data.
    prices = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "session_date": "2019-01-02",
                "open": 1.0,
                "close": 1.1,
                "volume": 100,
            },
            {
                "ticker": "AAPL",
                "session_date": "2019-07-02",
                "open": 1.0,
                "close": 1.1,
                "volume": 100,
            },
        ]
    )
    # Run function.
    result = add_session_timestamps(prices)
    # Check result.
    assert result.iloc[0]["session_open_utc"].hour == 14
    assert result.iloc[1]["session_open_utc"].hour == 13


def test_map_articles_uses_strictly_future_open() -> None:
    """Map an event at market open to the following session."""

    # Prepare data.
    news = pd.DataFrame(
        [
            {
                "article_id": "a1",
                "ticker": "AAPL",
                "published_at_utc": "2019-01-02T14:30:00Z",
                "text": "event",
                "source_name": "SEC",
                "source_url": "https://example.test/a1",
            }
        ]
    )
    prices = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "session_date": "2019-01-02",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 1000,
            },
            {
                "ticker": "AAPL",
                "session_date": "2019-01-03",
                "open": 101.0,
                "high": 102.0,
                "low": 100.0,
                "close": 102.0,
                "volume": 1100,
            },
        ]
    )
    # Run function.
    mapped, rejected = map_articles_to_sessions(news, prices)
    # Check result.
    assert rejected.empty
    assert str(mapped.iloc[0]["target_session_date"]) == "2019-01-03"


def test_add_movement_labels_uses_fixed_threshold() -> None:
    """Apply the documented symmetric 0.5 percent label threshold."""

    # Prepare data.
    frame = pd.DataFrame({"reaction_return": [-0.006, 0.0, 0.006]})
    # Run function.
    result = add_movement_labels(frame, 0.005)
    # Check result.
    assert result["movement_label"].tolist() == ["Down", "Flat", "Up"]


def test_readiness_requires_all_classes_in_each_split() -> None:
    """Reject evidence that cannot support three-class chronological splits."""

    # Prepare data.
    dates = pd.date_range("2018-01-01", periods=310, freq="D")
    frame = pd.DataFrame(
        {
            "ticker": "AAPL",
            "target_session_date": dates,
            "movement_label": "Flat",
        }
    )
    # Run function.
    # Check result.
    with pytest.raises(FoundationError):
        _readiness_report(frame, 300, 300)


def test_readiness_accepts_balanced_chronological_blocks() -> None:
    """Accept evidence with every class across purged chronological blocks."""

    # Prepare data.
    dates = pd.date_range("2018-01-01", periods=330, freq="D")
    labels = ["Down", "Flat", "Up"] * 110
    frame = pd.DataFrame(
        {
            "ticker": "AAPL",
            "target_session_date": dates,
            "movement_label": labels,
        }
    )
    # Run function.
    result = _readiness_report(frame, 300, 300)
    # Check result.
    assert result["ready_for_stock_movement_package"] is True


def test_foundation_config_uses_qualified_fixed_dates() -> None:
    """Keep the final experiment inside the qualified provider window."""

    # Prepare data.
    config = FoundationConfig()
    # Run function.
    values = (
        config.event_start_date,
        config.event_end_date,
        config.price_start_date,
        config.price_end_date,
    )
    # Check result.
    assert values == (
        date(2015, 2, 2),
        date(2020, 3, 31),
        date(2015, 1, 1),
        date(2020, 4, 1),
    )


def test_required_tickers_match_qualification_contract() -> None:
    """Preserve the exact ten tickers proven by the live qualification."""

    # Prepare data.
    expected = {
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
    }
    # Run function.
    observed = set(REQUIRED_TICKERS)
    # Check result.
    assert observed == expected and len(REQUIRED_TICKERS) == 10


def test_private_output_writer_uses_owner_only_permissions(tmp_path: Path) -> None:
    """Protect processed evidence from other operating-system users."""

    # Prepare data.
    path = tmp_path / "evidence.csv"
    frame = pd.DataFrame({"value": [1]})
    # Run function.
    market_data_foundation._atomic_csv(path, frame)
    # Check result.
    assert path.stat().st_mode & 0o077 == 0


def test_token_is_never_part_of_tiingo_url() -> None:
    """Keep the secret token out of URLs and persisted provenance."""

    # Prepare data.
    token = "super-secret-token"
    # Run function.
    url = build_prices_url("AAPL", date(2015, 1, 1), date(2020, 4, 1))
    # Check result.
    assert token not in url and "token" not in url.lower()


def test_http_authentication_failure_is_not_retried(tmp_path: Path) -> None:
    """Fail immediately on invalid Tiingo credentials."""

    # Prepare data.
    from financial_news_intelligence.data.foundation_tiingo_prices import (
        fetch_prices_with_cache,
    )

    headers = {"Retry-After": "1"}

    def opener(*_args: object, **_kwargs: object) -> object:
        """Raise a deterministic authentication error."""

        raise HTTPError("https://example.test", 401, "bad token", headers, None)

    # Run function.
    # Check result.
    with pytest.raises(TiingoPriceError, match="authentication failed"):
        fetch_prices_with_cache(
            "AAPL",
            date(2015, 1, 1),
            date(2020, 4, 1),
            "secret",
            tmp_path / "aapl.json",
            opener=opener,
        )


def test_build_and_verify_foundation_with_synthetic_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the complete orchestrator, serialization, and read-back gates."""

    # Prepare data.
    project_root = tmp_path / "project"
    reference_directory = project_root / "data/reference"
    reference_directory.mkdir(parents=True)
    _reference(reference_directory)
    qualification_path = (
        project_root
        / "reports/provider_qualification/tiingo/"
        / "tiingo_eod_qualification_summary.json"
    )
    qualification_path.parent.mkdir(parents=True)
    qualification_path.write_text(
        json.dumps(_qualification(_price_rows())),
        encoding="utf-8",
    )
    model_directory = project_root / "artifacts/models/fake/final_model"
    model_directory.mkdir(parents=True)
    private_directory = project_root / "data/private/tiingo_eod"
    private_directory.mkdir(parents=True)
    for ticker in REQUIRED_TICKERS:
        cache = private_directory / f"{ticker}.json"
        cache.write_text("[]", encoding="utf-8")
        os.chmod(cache, 0o600)

    dates = pd.bdate_range("2018-01-02", periods=42)
    price_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    for ticker in REQUIRED_TICKERS:
        close = 100.0
        for position, observed in enumerate(dates):
            if position:
                return_value = (-0.01, 0.0, 0.01)[position % 3]
                close *= 1.0 + return_value
            price_rows.append(
                {
                    "ticker": ticker,
                    "session_date": observed.strftime("%Y-%m-%d"),
                    "open": close,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "volume": 1000 + position,
                    "source_provider": "tiingo_eod",
                    "provider_symbol": ticker,
                    "cross_check_provider": "tiingo_ten_ticker_qualification",
                    "cross_check_status": "qualified_exact_window_checksum_match",
                    "verification_status": "primary_verified",
                    "source_url": "https://api.tiingo.com/example",
                    "price_provider_role": "authenticated_internal_primary",
                    "fetched_at_utc": "2026-07-03T00:00:00Z",
                    "price_provenance_note": "internal test evidence",
                }
            )
            if position > 0:
                prior_day = observed - pd.Timedelta(days=1)
                event_rows.append(
                    {
                        "article_id": f"{ticker}-{position}",
                        "ticker": ticker,
                        "company": f"{ticker} Corporation",
                        "published_at_utc": prior_day.strftime(
                            "%Y-%m-%dT20:00:00Z"
                        ),
                        "timestamp_type": "sec_acceptance_datetime",
                        "text": f"{ticker} SEC event {position}",
                        "headline": f"{ticker} SEC event {position}",
                        "source_name": "SEC",
                        "source_url": f"https://sec.test/{ticker}/{position}",
                        "news_provider": "sec_edgar_submissions_api",
                        "provider_role": "primary_company_disclosure",
                        "verification_status": "primary_verified",
                        "ticker_resolution_method": "sec_official_ticker_to_cik",
                        "matched_alias": ticker,
                        "language": "English",
                        "source_country": "United States",
                        "request_url": "https://sec.test/submissions",
                        "provenance_note": "official test metadata",
                    }
                )

    def fake_sec(*_args: object, **_kwargs: object):
        """Return deterministic official-event fixtures."""

        return pd.DataFrame(event_rows), pd.DataFrame(), []

    def fake_prices(*_args: object, **_kwargs: object):
        """Return deterministic adjusted-price fixtures."""

        return pd.DataFrame(price_rows), []

    def fake_predictor(texts: list[str], **_kwargs: object) -> pd.DataFrame:
        """Return deterministic three-class sentiment probabilities."""

        count = len(texts)
        return pd.DataFrame(
            {
                "prob_bearish": np.full(count, 0.2),
                "prob_neutral": np.full(count, 0.3),
                "prob_bullish": np.full(count, 0.5),
                "sentiment_label": ["Bullish"] * count,
            }
        )

    monkeypatch.setattr(market_data_foundation, "collect_sec_events", fake_sec)
    monkeypatch.setattr(
        market_data_foundation,
        "collect_tiingo_prices",
        fake_prices,
    )
    monkeypatch.setattr(
        market_data_foundation,
        "_sentiment_model_contract",
        lambda _root: (
            model_directory,
            market_data_foundation.SENTIMENT_ORDER,
            {"model_key": "distilbert"},
        ),
    )
    config = FoundationConfig(minimum_articles=300, minimum_sessions=30)

    # Run function.
    result = market_data_foundation.build_foundation(
        project_root,
        api_token="secret",
        config=config,
        sentiment_predictor=fake_predictor,
    )
    verified = market_data_foundation.verify_foundation(project_root)

    # Check result.
    assert result["status"] == "foundation_verified"
    assert verified["primary_price_rows"] == len(price_rows)
    assert (project_root / market_data_foundation.NEWS_OUTPUT).exists()
    assert (project_root / market_data_foundation.MANIFEST_OUTPUT).exists()
