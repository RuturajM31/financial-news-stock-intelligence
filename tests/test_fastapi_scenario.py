"""Tests for transparent investment-scenario calculations."""

import pytest

from financial_news_intelligence.api.errors import ApiProblem
from financial_news_intelligence.api.intelligence_runtime import IntelligenceRuntime


def test_scenario_formula_applies_fees_and_positive_tax() -> None:
    """Prepare assumptions, run one outcome, and check each named formula."""

    result = IntelligenceRuntime._calculate_outcome(
        request={
            "investment_amount": 1000.0,
            "share_price": 100.0,
            "allow_fractional_shares": True,
            "share_precision": 6,
            "entry_fee": 10.0,
            "exit_fee": 5.0,
            "tax_rate_percent": 25.0,
        },
        historical_return_percent=10.0,
        scenario="high",
    )

    assert result["shares_purchased"] == 9.9
    assert result["cash_balance"] == pytest.approx(0.0)
    assert result["estimated_tax"] == pytest.approx(21.0)
    assert result["net_final_value"] == pytest.approx(1063.0)
    assert result["gain_loss"] == pytest.approx(63.0)


def test_scenario_rejects_unaffordable_whole_share() -> None:
    """Prepare insufficient capital, run calculation, and check safe failure."""

    with pytest.raises(ApiProblem) as captured:
        IntelligenceRuntime._calculate_outcome(
            request={
                "investment_amount": 50.0,
                "share_price": 100.0,
                "allow_fractional_shares": False,
                "share_precision": 0,
                "entry_fee": 0.0,
                "exit_fee": 0.0,
                "tax_rate_percent": None,
            },
            historical_return_percent=5.0,
            scenario="base",
        )

    assert captured.value.error_code == "investment_amount_insufficient"
