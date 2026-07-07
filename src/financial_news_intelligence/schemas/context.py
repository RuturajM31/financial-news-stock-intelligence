"""Schemas for company, market, and financial context."""

from datetime import date, datetime

from pydantic import Field, HttpUrl

from .common import MarketSession, ProjectSchema


class CompanyContext(ProjectSchema):
    """
    Company information detected from the article.

    Input:  Clean article text.
    Output: Detected company, ticker, and confidence.
    Next:   Used by stock-price and forecasting services.
    """

    # Company and ticker are detected automatically.
    company: str | None = None
    ticker: str | None = None

    # Show how certain the detection service is.
    detection_confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )


class TradingContext(ProjectSchema):
    """
    Trading-session information connected to the article.

    Input:  Publication timestamp.
    Output: Market timing and next valid trading session.
    Next:   Used to calculate the correct future stock return.
    """

    published_at: datetime | None = None
    timezone: str | None = None

    # Classify when the article appeared relative to the market.
    market_session: MarketSession = MarketSession.UNKNOWN

    # Weekend, holiday, and after-market news map to the next session.
    next_trading_session: date | None = None


class ArticleContext(ProjectSchema):
    """
    Complete context discovered for one article.

    Output: Company information plus trading-session information.
    """

    company: CompanyContext
    trading: TradingContext


class FinancialContext(ProjectSchema):
    """
    Point-in-time financial information for the detected company.

    Rule: Values must be available before the article publication time.
    """

    # Record when and where the financial data came from.
    as_of_date: date | None = None
    source_name: str | None = None
    source_url: HttpUrl | None = None

    # Valuation ratios.
    pe_ratio: float | None = None
    pb_ratio: float | None = None
    dividend_yield: float | None = None

    # Financial-strength ratios.
    debt_to_equity: float | None = None
    current_ratio: float | None = None

    # Profitability ratios.
    return_on_equity: float | None = None
    profit_margin: float | None = None
    operating_margin: float | None = None

    # Growth indicators.
    revenue_growth: float | None = None
    eps_growth: float | None = None

    # Supporting company value.
    market_cap: float | None = None
