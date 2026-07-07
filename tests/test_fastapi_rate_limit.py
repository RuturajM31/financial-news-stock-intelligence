"""Tests for deterministic in-process rate limiting."""

import pytest

from financial_news_intelligence.api.errors import ApiProblem
from financial_news_intelligence.api.rate_limit import FixedWindowRateLimiter


def test_rate_limiter_rejects_request_after_configured_limit() -> None:
    """Prepare one window, run three checks, and check the third is rejected."""

    limiter = FixedWindowRateLimiter(maximum_requests=2, window_seconds=60)
    limiter.check("client:endpoint", now=10.0)
    limiter.check("client:endpoint", now=11.0)

    with pytest.raises(ApiProblem) as captured:
        limiter.check("client:endpoint", now=12.0)

    assert captured.value.status_code == 429
    assert captured.value.error_code == "rate_limit_exceeded"
