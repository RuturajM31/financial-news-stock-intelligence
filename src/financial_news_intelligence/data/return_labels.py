"""Create historical Up, Flat, or Down labels from verified prices."""

import math
from datetime import date

from financial_news_intelligence.data.provenance import (
    assert_usage_allowed,
)
from financial_news_intelligence.schemas.common import MovementLabel
from financial_news_intelligence.schemas.market_data import (
    MarketPriceHistory,
    ReturnLabel,
)
from financial_news_intelligence.schemas.provenance import DataPurpose


# ============================================================
# 1. RETURN CALCULATION
# ============================================================

def calculate_open_to_close_return(
    open_price: float,
    close_price: float,
) -> float:
    """Calculate the target session's percentage price movement."""

    for price_name, price_value in (
        ("open_price", open_price),
        ("close_price", close_price),
    ):
        if not math.isfinite(price_value) or price_value <= 0:
            raise ValueError(
                f"{price_name} must be finite and greater than zero."
            )

    return round(
        (close_price - open_price)
        / open_price
        * 100,
        6,
    )


# ============================================================
# 2. MOVEMENT CLASSIFICATION
# ============================================================

def classify_return_direction(
    return_pct: float,
    flat_threshold_pct: float,
) -> MovementLabel:
    """
    Convert a numerical return into Up, Flat, or Down.

    Returns inside the inclusive positive/negative threshold are Flat.
    """

    if not math.isfinite(return_pct):
        raise ValueError("return_pct must be finite.")

    if (
        not math.isfinite(flat_threshold_pct)
        or flat_threshold_pct < 0
    ):
        raise ValueError(
            "flat_threshold_pct must be finite and non-negative."
        )

    if return_pct > flat_threshold_pct:
        return MovementLabel.UP

    if return_pct < -flat_threshold_pct:
        return MovementLabel.DOWN

    return MovementLabel.FLAT


# ============================================================
# 3. TARGET-SESSION LABEL
# ============================================================

def create_return_label(
    *,
    price_history: MarketPriceHistory,
    target_session: date,
    flat_threshold_pct: float,
    purpose: DataPurpose = DataPurpose.TRAINING,
) -> ReturnLabel:
    """
    Label the verified session selected by the article-time mapper.

    Input:
    - verified price history;
    - article target session;
    - configurable Flat threshold.

    Output:
    - traceable Up, Flat, or Down historical outcome.
    """

    # Recheck permission before derived labels enter protected pipelines.
    assert_usage_allowed(
        price_history.provenance,
        purpose,
    )

    target_bar = next(
        (
            price_bar
            for price_bar in price_history.bars
            if price_bar.session_date == target_session
        ),
        None,
    )

    if target_bar is None:
        raise LookupError(
            "Target trading session is absent from price history: "
            f"{target_session}"
        )

    return_pct = calculate_open_to_close_return(
        target_bar.open_price,
        target_bar.close_price,
    )

    direction = classify_return_direction(
        return_pct,
        flat_threshold_pct,
    )

    return ReturnLabel(
        ticker=price_history.ticker,
        target_session=target_session,
        open_price=target_bar.open_price,
        close_price=target_bar.close_price,
        return_pct=return_pct,
        direction=direction,
        flat_threshold_pct=flat_threshold_pct,
        price_source_id=price_history.provenance.source_id,
        price_checksum_sha256=(
            price_history.provenance.checksum_sha256
        ),
    )
