"""Tests for mapping financial news to the correct trading session."""

from datetime import date, datetime, timezone

import pytest

from financial_news_intelligence.data.market_session import (
    map_trading_session,
    prepare_publication_time,
)
from financial_news_intelligence.schemas import MarketSession


# ============================================================
# 1. BEFORE-MARKET NEWS
# ============================================================

def test_before_market_uses_same_trading_day() -> None:
    """News before market opening should affect the same session."""

    # 13:00 UTC is 08:00 New York time in January.
    published_at = datetime(
        2024,
        1,
        16,
        13,
        0,
        tzinfo=timezone.utc,
    )

    result = map_trading_session(published_at)

    assert result.market_session == MarketSession.BEFORE_MARKET
    assert result.next_trading_session == date(2024, 1, 16)


# ============================================================
# 2. DURING-MARKET NEWS
# ============================================================

def test_during_market_uses_next_full_session() -> None:
    """Intraday news should map to the next complete trading session."""

    # 15:00 UTC is 10:00 New York time.
    published_at = datetime(
        2024,
        1,
        16,
        15,
        0,
        tzinfo=timezone.utc,
    )

    result = map_trading_session(published_at)

    assert result.market_session == MarketSession.DURING_MARKET
    assert result.next_trading_session == date(2024, 1, 17)


# ============================================================
# 3. AFTER-MARKET NEWS
# ============================================================

def test_after_market_uses_next_trading_day() -> None:
    """News after market closing should use the next session."""

    # 22:00 UTC is 17:00 New York time.
    published_at = datetime(
        2024,
        1,
        16,
        22,
        0,
        tzinfo=timezone.utc,
    )

    result = map_trading_session(published_at)

    assert result.market_session == MarketSession.AFTER_MARKET
    assert result.next_trading_session == date(2024, 1, 17)


# ============================================================
# 4. WEEKEND NEWS
# ============================================================

def test_weekend_uses_next_monday_session() -> None:
    """Saturday news should map to the next available session."""

    published_at = datetime(
        2024,
        1,
        20,
        12,
        0,
        tzinfo=timezone.utc,
    )

    result = map_trading_session(published_at)

    assert result.market_session == MarketSession.WEEKEND
    assert result.next_trading_session == date(2024, 1, 22)


# ============================================================
# 5. MARKET-HOLIDAY NEWS
# ============================================================

def test_market_holiday_uses_next_session() -> None:
    """Holiday news should skip the closed exchange day."""

    # 15 January 2024 was Martin Luther King Jr. Day.
    published_at = datetime(
        2024,
        1,
        15,
        12,
        0,
        tzinfo=timezone.utc,
    )

    result = map_trading_session(published_at)

    assert result.market_session == MarketSession.HOLIDAY
    assert result.next_trading_session == date(2024, 1, 16)


# ============================================================
# 6. EARLY-CLOSING SESSION
# ============================================================

def test_early_close_is_respected() -> None:
    """News after an early close should map to the next session."""

    # The NYSE closed early on 29 November 2024.
    published_at = datetime(
        2024,
        11,
        29,
        18,
        30,
        tzinfo=timezone.utc,
    )

    result = map_trading_session(published_at)

    assert result.market_session == MarketSession.AFTER_MARKET
    assert result.next_trading_session == date(2024, 12, 2)


# ============================================================
# 7. NAIVE DATETIME WITH SOURCE TIMEZONE
# ============================================================

def test_naive_datetime_uses_supplied_timezone() -> None:
    """A timezone-free timestamp should use its stated source zone."""

    # 14:00 Berlin time equals 13:00 UTC in January.
    published_at = datetime(2024, 1, 16, 14, 0)

    result = map_trading_session(
        published_at,
        source_timezone="Europe/Berlin",
    )

    assert result.published_at.tzinfo is not None
    assert result.market_session == MarketSession.BEFORE_MARKET
    assert result.next_trading_session == date(2024, 1, 16)


# ============================================================
# 8. INVALID TIMEZONE
# ============================================================

def test_invalid_source_timezone_is_rejected() -> None:
    """An unknown timezone should produce a clear error."""

    published_at = datetime(2024, 1, 16, 14, 0)

    with pytest.raises(ValueError, match="Invalid source timezone"):
        prepare_publication_time(
            published_at,
            source_timezone="Planet/Mars",
        )


# ============================================================
# 9. INVALID PUBLICATION VALUE
# ============================================================

def test_non_datetime_value_is_rejected() -> None:
    """Publication time must be a real datetime object."""

    with pytest.raises(TypeError, match="must be a datetime"):
        prepare_publication_time("2024-01-16")  # type: ignore[arg-type]
