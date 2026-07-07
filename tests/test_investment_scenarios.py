"""Tests for the evidence-based investment scenario calculator."""

from datetime import datetime, timezone

import pytest

from financial_news_intelligence.schemas.common import SentimentLabel
from financial_news_intelligence.schemas.historical_intelligence import (
    ReactionCohort,
)
from financial_news_intelligence.schemas.investment import (
    InvestmentRequest,
    ScenarioLevel,
)
from financial_news_intelligence.services.investment_scenarios import (
    InsufficientInvestmentAmountError,
    build_investment_scenario_result,
    calculate_investment_outcome,
    calculate_share_purchase,
)


# ============================================================
# 1. TEST DATA HELPER
# ============================================================

def make_cohort() -> ReactionCohort:
    """Create one deterministic bullish historical evidence cohort."""

    return ReactionCohort(
        query_article_id="f" * 64,
        ticker="AAPL",
        sentiment_label=SentimentLabel.BULLISH,
        cutoff_at=datetime(
            2024,
            2,
            1,
            tzinfo=timezone.utc,
        ),
        minimum_similarity=0.70,
        minimum_sample_size=5,
        sample_size=5,
        lower_quantile=0.10,
        upper_quantile=0.90,
        low_return_pct=-2.0,
        median_return_pct=1.0,
        high_return_pct=4.0,
        matched_article_ids=tuple(
            f"{index + 1:064x}"[-64:]
            for index in range(5)
        ),
        matched_returns_pct=(-2.0, -1.0, 1.0, 2.0, 4.0),
        matched_similarity_scores=(0.95, 0.90, 0.85, 0.80, 0.75),
        latest_evidence_published_at=datetime(
            2024,
            1,
            31,
            tzinfo=timezone.utc,
        ),
        evidence_checksum_sha256="a" * 64,
    )


# ============================================================
# 2. FRACTIONAL SHARE PURCHASE
# ============================================================

def test_fractional_shares_use_available_capital() -> None:
    """Entry fees should be removed before fractional shares are bought."""

    request = InvestmentRequest(
        investment_amount=1000.0,
        share_price=190.0,
        entry_fee=5.0,
        allow_fractional_shares=True,
        share_precision=4,
    )

    shares, share_cost, cash = calculate_share_purchase(request)

    assert shares == 5.2368
    assert share_cost + cash == pytest.approx(995.0)
    assert cash >= 0


# ============================================================
# 3. WHOLE SHARES AND CASH BALANCE
# ============================================================

def test_whole_share_mode_keeps_unspent_cash() -> None:
    """Whole-share purchases should not silently spend leftover cash."""

    request = InvestmentRequest(
        investment_amount=1000.0,
        share_price=190.0,
        allow_fractional_shares=False,
    )

    shares, share_cost, cash = calculate_share_purchase(request)

    assert shares == 5.0
    assert share_cost == 950.0
    assert cash == 50.0


# ============================================================
# 4. FEES AND POSITIVE TAX
# ============================================================

def test_fees_and_tax_reduce_positive_scenario_value() -> None:
    """Positive gains should include supplied fees and estimated tax."""

    request = InvestmentRequest(
        investment_amount=1000.0,
        share_price=100.0,
        entry_fee=10.0,
        exit_fee=5.0,
        tax_rate_pct=25.0,
    )

    outcome = calculate_investment_outcome(
        request=request,
        historical_return_pct=10.0,
        scenario=ScenarioLevel.HIGH,
    )

    assert outcome.gross_final_value == pytest.approx(1089.0)
    assert outcome.estimated_tax == pytest.approx(21.0)
    assert outcome.net_final_value == pytest.approx(1063.0)
    assert outcome.gain_loss == pytest.approx(63.0)


# ============================================================
# 5. LOSSES ARE NOT TAXED
# ============================================================

def test_negative_scenario_has_no_estimated_tax() -> None:
    """The calculator should not estimate tax on a scenario loss."""

    request = InvestmentRequest(
        investment_amount=1000.0,
        share_price=100.0,
        tax_rate_pct=25.0,
    )

    outcome = calculate_investment_outcome(
        request=request,
        historical_return_pct=-10.0,
        scenario=ScenarioLevel.LOW,
    )

    assert outcome.estimated_tax == 0.0
    assert outcome.net_final_value == pytest.approx(900.0)
    assert outcome.gain_loss == pytest.approx(-100.0)


# ============================================================
# 6. INSUFFICIENT CAPITAL
# ============================================================

def test_whole_share_mode_rejects_unaffordable_stock() -> None:
    """A whole-share request should fail when no share can be purchased."""

    request = InvestmentRequest(
        investment_amount=50.0,
        share_price=100.0,
        allow_fractional_shares=False,
    )

    with pytest.raises(InsufficientInvestmentAmountError):
        calculate_share_purchase(request)


# ============================================================
# 7. COMPLETE EVIDENCE-BASED RESULT
# ============================================================

def test_complete_result_preserves_evidence_and_disclaimer() -> None:
    """The final result should expose evidence, assumptions, and risk text."""

    request = InvestmentRequest(
        investment_amount=1000.0,
        share_price=100.0,
        currency="eur",
    )

    result = build_investment_scenario_result(
        request=request,
        cohort=make_cohort(),
        generated_at=datetime(
            2024,
            2,
            1,
            12,
            0,
            tzinfo=timezone.utc,
        ),
    )

    assert result.ticker == "AAPL"
    assert result.request.currency == "EUR"
    assert result.historical_sample_size == 5
    assert result.historical_evidence_checksum_sha256 == "a" * 64
    assert result.low.net_final_value == pytest.approx(980.0)
    assert result.base.net_final_value == pytest.approx(1010.0)
    assert result.high.net_final_value == pytest.approx(1040.0)
    assert "not guaranteed returns" in result.disclaimer
    assert any("Taxes are excluded" in item for item in result.assumptions)
