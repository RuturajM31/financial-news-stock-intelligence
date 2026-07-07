"""Map a news publication time to the correct US trading session."""

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import exchange_calendars as xcals

from financial_news_intelligence.schemas import (
    MarketSession,
    TradingContext,
)


# ============================================================
# 1. MARKET SETTINGS
# ============================================================

DEFAULT_CALENDAR = "XNYS"
DEFAULT_SOURCE_TIMEZONE = "UTC"
MARKET_TIMEZONE = "America/New_York"


# ============================================================
# 2. PREPARE THE PUBLICATION TIME
# ============================================================

def prepare_publication_time(
    published_at: datetime,
    source_timezone: str = DEFAULT_SOURCE_TIMEZONE,
) -> datetime:
    """
    Make the publication time timezone-aware.

    Input:  Article publication datetime and its source timezone.
    Output: A timezone-aware datetime.
    Next:   Converted into New York market time.
    """

    if not isinstance(published_at, datetime):
        raise TypeError("published_at must be a datetime.")

    try:
        source_zone = ZoneInfo(source_timezone)
    except Exception as error:
        raise ValueError(
            f"Invalid source timezone: {source_timezone}"
        ) from error

    # A naive datetime has no timezone, so attach the supplied source zone.
    if published_at.tzinfo is None:
        return published_at.replace(tzinfo=source_zone)

    return published_at


# ============================================================
# 3. LOAD A CALENDAR AROUND THE ARTICLE DATE
# ============================================================

def load_market_calendar(
    published_at: datetime,
    calendar_code: str = DEFAULT_CALENDAR,
):
    """
    Load enough exchange-calendar history around the article date.

    One year before and two years after provide room for previous
    and next-session lookups.
    """

    start_year = published_at.year - 1
    end_year = published_at.year + 2

    try:
        return xcals.get_calendar(
            calendar_code,
            start=f"{start_year}-01-01",
            end=f"{end_year}-12-31",
        )
    except Exception as error:
        raise ValueError(
            f"Could not load market calendar: {calendar_code}"
        ) from error


# ============================================================
# 4. MAP PUBLICATION TIME TO A MARKET SESSION
# ============================================================

def map_trading_session(
    published_at: datetime,
    source_timezone: str = DEFAULT_SOURCE_TIMEZONE,
    calendar_code: str = DEFAULT_CALENDAR,
) -> TradingContext:
    """
    Classify when the article appeared and select its target session.

    Mapping rules:
    - Before market: use the same day's session.
    - During market: use the following trading session.
    - After market: use the following trading session.
    - Weekend or holiday: use the next available session.

    Output is later used to calculate the correct stock-return window.
    """

    aware_time = prepare_publication_time(
        published_at,
        source_timezone,
    )

    market_zone = ZoneInfo(MARKET_TIMEZONE)

    # Convert the article timestamp into New York exchange time.
    market_time = aware_time.astimezone(market_zone)
    market_date = market_time.date()

    calendar = load_market_calendar(
        market_time,
        calendar_code,
    )

    # --------------------------------------------------------
    # Weekend or exchange holiday
    # --------------------------------------------------------

    if not calendar.is_session(market_date):
        target_session = calendar.date_to_session(
            market_date,
            direction="next",
        )

        if market_date.weekday() >= 5:
            session_type = MarketSession.WEEKEND
        else:
            session_type = MarketSession.HOLIDAY

        return TradingContext(
            published_at=aware_time,
            timezone=MARKET_TIMEZONE,
            market_session=session_type,
            next_trading_session=target_session.date(),
        )

    # --------------------------------------------------------
    # Valid trading day
    # --------------------------------------------------------

    current_session = calendar.date_to_session(market_date)

    session_open = calendar.session_open(
        current_session
    ).to_pydatetime()

    session_close = calendar.session_close(
        current_session
    ).to_pydatetime()

    publication_utc = market_time.astimezone(timezone.utc)

    # News released before opening can affect today's full session.
    if publication_utc < session_open:
        session_type = MarketSession.BEFORE_MARKET
        target_session = current_session

    # News released during trading is mapped to the next full session.
    elif publication_utc < session_close:
        session_type = MarketSession.DURING_MARKET
        target_session = calendar.next_session(current_session)

    # News released at or after closing maps to the next session.
    else:
        session_type = MarketSession.AFTER_MARKET
        target_session = calendar.next_session(current_session)

    return TradingContext(
        published_at=aware_time,
        timezone=MARKET_TIMEZONE,
        market_session=session_type,
        next_trading_session=target_session.date(),
    )
