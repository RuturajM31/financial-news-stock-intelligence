"""Convert historical reactions into user investment scenarios."""

from __future__ import annotations

import math
from datetime import datetime, timezone

from financial_news_intelligence.schemas.historical_intelligence import (
    ReactionCohort,
)
from financial_news_intelligence.schemas.investment import (
    InvestmentOutcome,
    InvestmentRequest,
    InvestmentScenarioResult,
    ScenarioLevel,
)


# ============================================================
# 1. USER-FACING SAFETY TEXT
# ============================================================

INVESTMENT_SCENARIO_DISCLAIMER = (
    "These values are statistical scenarios derived from historically "
    "similar news reactions. They are not guaranteed returns, a price "
    "forecast, or financial advice. Actual prices, liquidity, spreads, "
    "fees, taxes, and market conditions may produce different results."
)


# ============================================================
# 2. DOMAIN-SPECIFIC ERROR
# ============================================================

class InsufficientInvestmentAmountError(ValueError):
    """Raised when the available amount cannot purchase any shares."""


# ============================================================
# 3. SHARE PURCHASE CALCULATION
# ============================================================

def calculate_share_purchase(
    request: InvestmentRequest,
) -> tuple[float, float, float]:
    """
    Calculate shares, share cost, and uninvested cash.

    Entry fees are deducted before shares are purchased. Fractional
    shares are rounded down so the calculation never overspends.
    """

    available_capital = (
        request.investment_amount - request.entry_fee
    )
    raw_shares = available_capital / request.share_price

    if request.allow_fractional_shares:
        precision_factor = 10 ** request.share_precision
        shares_purchased = (
            math.floor(raw_shares * precision_factor)
            / precision_factor
        )
    else:
        shares_purchased = float(math.floor(raw_shares))

    if shares_purchased <= 0:
        raise InsufficientInvestmentAmountError(
            "Available capital cannot purchase one permitted share unit."
        )

    share_cost = shares_purchased * request.share_price
    cash_balance = available_capital - share_cost

    # Small floating-point remnants are normalized for clean reporting.
    if abs(cash_balance) < 0.00000001:
        cash_balance = 0.0

    return (
        shares_purchased,
        share_cost,
        cash_balance,
    )


# ============================================================
# 4. ONE RETURN SCENARIO
# ============================================================

def calculate_investment_outcome(
    *,
    request: InvestmentRequest,
    historical_return_pct: float,
    scenario: ScenarioLevel,
) -> InvestmentOutcome:
    """
    Apply one historical return to the user's purchased shares.

    Formula:
    final stock value = share cost × (1 + historical return / 100)
    net value = stock value + cash - exit fee - estimated tax
    """

    if (
        not math.isfinite(historical_return_pct)
        or historical_return_pct < -100
    ):
        raise ValueError(
            "historical_return_pct must be finite and at least -100%."
        )

    (
        shares_purchased,
        share_cost,
        cash_balance,
    ) = calculate_share_purchase(request)

    final_stock_value = share_cost * (
        1 + historical_return_pct / 100
    )
    gross_final_value = final_stock_value + cash_balance

    value_after_exit_fee = max(
        gross_final_value - request.exit_fee,
        0.0,
    )

    pre_tax_gain = (
        value_after_exit_fee - request.investment_amount
    )

    if request.tax_rate_pct is None:
        estimated_tax = 0.0
    else:
        # Taxes are estimated only on a positive scenario gain.
        estimated_tax = max(pre_tax_gain, 0.0) * (
            request.tax_rate_pct / 100
        )

    net_final_value = max(
        value_after_exit_fee - estimated_tax,
        0.0,
    )
    gain_loss = net_final_value - request.investment_amount
    gain_loss_pct = (
        gain_loss / request.investment_amount * 100
    )

    return InvestmentOutcome(
        scenario=scenario,
        historical_return_pct=historical_return_pct,
        initial_investment=request.investment_amount,
        share_price=request.share_price,
        shares_purchased=shares_purchased,
        share_cost=share_cost,
        cash_balance=cash_balance,
        entry_fee=request.entry_fee,
        exit_fee=request.exit_fee,
        estimated_tax=estimated_tax,
        gross_final_value=gross_final_value,
        net_final_value=net_final_value,
        gain_loss=gain_loss,
        gain_loss_pct=gain_loss_pct,
    )


# ============================================================
# 5. COMPLETE LOW, BASE, AND HIGH RESULT
# ============================================================

def build_investment_scenario_result(
    *,
    request: InvestmentRequest,
    cohort: ReactionCohort,
    generated_at: datetime | None = None,
) -> InvestmentScenarioResult:
    """
    Convert one historical reaction cohort into user-specific outcomes.

    The calculator does not invent a future return. It uses the cohort's
    lower quantile, median, and upper quantile as scenario inputs.
    """

    if generated_at is None:
        generated_at = datetime.now(timezone.utc)

    if generated_at.tzinfo is None or generated_at.utcoffset() is None:
        raise ValueError("generated_at must be timezone-aware.")

    low_outcome = calculate_investment_outcome(
        request=request,
        historical_return_pct=cohort.low_return_pct,
        scenario=ScenarioLevel.LOW,
    )
    base_outcome = calculate_investment_outcome(
        request=request,
        historical_return_pct=cohort.median_return_pct,
        scenario=ScenarioLevel.BASE,
    )
    high_outcome = calculate_investment_outcome(
        request=request,
        historical_return_pct=cohort.high_return_pct,
        scenario=ScenarioLevel.HIGH,
    )

    if request.tax_rate_pct is None:
        tax_assumption = (
            "Taxes are excluded because no tax rate was supplied."
        )
    else:
        tax_assumption = (
            "Estimated tax is applied only to positive scenario gains "
            f"using {request.tax_rate_pct:.2f}%."
        )

    share_assumption = (
        "Fractional shares are allowed and rounded down to "
        f"{request.share_precision} decimal places."
        if request.allow_fractional_shares
        else "Only whole shares are purchased; remaining capital stays as cash."
    )

    return InvestmentScenarioResult(
        generated_at=generated_at,
        ticker=cohort.ticker,
        sentiment_label=cohort.sentiment_label,
        request=request,
        historical_sample_size=cohort.sample_size,
        historical_evidence_checksum_sha256=(
            cohort.evidence_checksum_sha256
        ),
        low=low_outcome,
        base=base_outcome,
        high=high_outcome,
        assumptions=(
            "Low, base, and high returns come from the historical cohort's "
            "lower quantile, median, and upper quantile.",
            "The supplied share price is treated as the purchase price.",
            "Entry and exit fees are fixed currency amounts.",
            share_assumption,
            tax_assumption,
        ),
        disclaimer=INVESTMENT_SCENARIO_DISCLAIMER,
    )
