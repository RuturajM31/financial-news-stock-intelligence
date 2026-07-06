"""Map verified articles to strictly future trading sessions.

Purpose
-------
Convert UTC article observation times into leakage-safe target sessions using
the actual session dates present in the qualified Tiingo EOD price table. The module
does not infer weekends or holidays from a generic weekday calendar.

Inputs and grain
----------------
News input grain is one accepted article. Price input grain is one ticker and
trading session. The supported default exchange is the US equity market in
``America/New_York`` with a 09:30 local open and 16:00 local close.

Mapping rule
------------
Each article maps to the first available session whose open timestamp is
strictly later than ``published_at_utc``. An article observed exactly at market
open therefore maps to the following session. The entry price is the previous
session close and the reaction price is the mapped target-session close.

Formula and downstream use
--------------------------
``reaction_return = target_close / previous_close - 1``. The mapped article
rows feed the final movement-label and sentiment evidence table.

Assumptions and limitations
---------------------------
The current project reference contains US equities. Non-US tickers must supply
an explicit timezone and future extension for market-open rules. Half-day
closes do not affect the target-open mapping because actual Tiingo session dates
remain authoritative.
"""

from __future__ import annotations

from datetime import time
from zoneinfo import ZoneInfo

import pandas as pd


class MarketSessionError(ValueError):
    """Raised when articles cannot be mapped without time leakage."""


def _clock(value: str) -> time:
    """Parse one documented ``HH:MM`` market clock value."""

    try:
        hour_text, minute_text = value.split(":", 1)
        return time(int(hour_text), int(minute_text))
    except (AttributeError, TypeError, ValueError) as exc:
        raise MarketSessionError(f"Invalid market clock value: {value}") from exc


def add_session_timestamps(
    prices: pd.DataFrame,
    timezone_name: str = "America/New_York",
    open_local: str = "09:30",
    close_local: str = "16:00",
) -> pd.DataFrame:
    """Attach UTC open and close timestamps to verified price sessions."""

    required = {"ticker", "session_date", "open", "close", "volume"}
    missing = sorted(required - set(prices.columns))
    if missing or prices.empty:
        raise MarketSessionError(f"Price evidence is incomplete: {missing}")

    zone = ZoneInfo(timezone_name)
    open_time = _clock(open_local)
    close_time = _clock(close_local)
    result = prices.copy()
    result["session_date"] = pd.to_datetime(
        result["session_date"],
        errors="coerce",
    ).dt.date
    if result["session_date"].isna().any():
        raise MarketSessionError("Price evidence contains invalid session dates.")

    result["session_open_utc"] = [
        pd.Timestamp.combine(session_date, open_time)
        .tz_localize(zone)
        .tz_convert("UTC")
        for session_date in result["session_date"]
    ]
    result["session_close_utc"] = [
        pd.Timestamp.combine(session_date, close_time)
        .tz_localize(zone)
        .tz_convert("UTC")
        for session_date in result["session_date"]
    ]
    return result.sort_values(["ticker", "session_open_utc"]).reset_index(
        drop=True
    )


def map_articles_to_sessions(
    news: pd.DataFrame,
    prices: pd.DataFrame,
    timezone_name: str = "America/New_York",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Map articles to future sessions and preserve unmapped rejection rows."""

    news_required = {
        "article_id",
        "ticker",
        "published_at_utc",
        "text",
        "source_name",
        "source_url",
    }
    missing = sorted(news_required - set(news.columns))
    if missing or news.empty:
        raise MarketSessionError(f"News evidence is incomplete: {missing}")

    news_frame = news.copy()
    news_frame["ticker"] = news_frame["ticker"].astype(str).str.upper()
    news_frame["published_at_utc"] = pd.to_datetime(
        news_frame["published_at_utc"],
        utc=True,
        errors="coerce",
    )
    if news_frame["published_at_utc"].isna().any():
        raise MarketSessionError("News evidence contains invalid timestamps.")

    price_frame = add_session_timestamps(prices, timezone_name=timezone_name)
    accepted_parts: list[pd.DataFrame] = []
    rejected_rows: list[dict[str, object]] = []

    for ticker, ticker_news in news_frame.groupby("ticker", observed=True):
        ticker_prices = price_frame[price_frame["ticker"] == ticker].copy()
        if ticker_prices.empty:
            for article in ticker_news.itertuples(index=False):
                rejected_rows.append(
                    {
                        "article_id": article.article_id,
                        "ticker": ticker,
                        "source_url": article.source_url,
                        "rejection_reason": "no_primary_price_history",
                    }
                )
            continue

        # Previous close is shifted inside the ticker group so it can never
        # accidentally borrow a different company's session.
        ticker_prices["previous_close"] = ticker_prices["close"].shift(1)
        ticker_prices["previous_session_date"] = ticker_prices[
            "session_date"
        ].shift(1)
        usable_prices = ticker_prices.dropna(
            subset=["previous_close", "previous_session_date"]
        ).copy()
        if usable_prices.empty:
            continue

        mapped = pd.merge_asof(
            ticker_news.sort_values("published_at_utc"),
            usable_prices.sort_values("session_open_utc"),
            left_on="published_at_utc",
            right_on="session_open_utc",
            direction="forward",
            allow_exact_matches=False,
            suffixes=("_news", "_price"),
        )

        rejected = mapped[mapped["session_date"].isna()].copy()
        for article in rejected.itertuples(index=False):
            rejected_rows.append(
                {
                    "article_id": article.article_id,
                    "ticker": ticker,
                    "source_url": article.source_url,
                    "rejection_reason": "no_future_primary_session",
                }
            )

        accepted = mapped.dropna(subset=["session_date", "close"]).copy()
        if accepted.empty:
            continue
        accepted["ticker"] = ticker
        accepted["reaction_return"] = (
            accepted["close"] / accepted["previous_close"] - 1.0
        )
        accepted["hours_to_session_open"] = (
            accepted["session_open_utc"] - accepted["published_at_utc"]
        ).dt.total_seconds() / 3600.0
        accepted_parts.append(accepted)

    if not accepted_parts:
        raise MarketSessionError("No article maps to a future primary session.")

    result = pd.concat(accepted_parts, ignore_index=True)
    if (result["published_at_utc"] >= result["session_open_utc"]).any():
        raise MarketSessionError("Future leakage detected in session mapping.")
    if result["hours_to_session_open"].le(0).any():
        raise MarketSessionError("Mapped sessions must open after observation.")

    result = result.rename(
        columns={
            "session_date": "target_session_date",
            "open": "target_open",
            "high": "target_high",
            "low": "target_low",
            "close": "target_close",
            "volume": "target_volume",
            "source_url_news": "source_url",
            "source_url_price": "price_source_url",
            "verification_status_news": "verification_status",
            "verification_status_price": "price_verification_status",
        }
    )
    result = result.sort_values(
        ["target_session_date", "ticker", "published_at_utc", "article_id"]
    ).reset_index(drop=True)
    return result, pd.DataFrame(rejected_rows)
