"""Tests for company-name and ticker detection."""

from pathlib import Path

import pytest

from financial_news_intelligence.data.company_ticker import (
    detect_company_ticker,
)


# ============================================================
# 1. EXACT COMPANY MATCH
# ============================================================

def test_detects_exact_company_name() -> None:
    """A full company name should return its ticker."""

    # Prepare article text.
    article_text = "Microsoft Corporation announced stronger earnings."

    # Run the detector.
    result = detect_company_ticker(article_text)

    # Check the detected company.
    assert result.company == "Microsoft Corporation"
    assert result.ticker == "MSFT"
    assert result.detection_confidence == 1.0


# ============================================================
# 2. COMPANY ALIAS MATCH
# ============================================================

def test_detects_company_alias() -> None:
    """A common company alias should map to the official company."""

    article_text = "Google introduced a new artificial intelligence model."

    result = detect_company_ticker(article_text)

    assert result.company == "Alphabet Inc."
    assert result.ticker == "GOOGL"
    assert result.detection_confidence == 1.0


# ============================================================
# 3. CASE-INSENSITIVE MATCH
# ============================================================

def test_matching_is_case_insensitive() -> None:
    """Lowercase company names should still be detected."""

    article_text = "tesla reported higher vehicle deliveries."

    result = detect_company_ticker(article_text)

    assert result.company == "Tesla Inc."
    assert result.ticker == "TSLA"


# ============================================================
# 4. UNKNOWN COMPANY
# ============================================================

def test_returns_empty_context_for_unknown_company() -> None:
    """Unknown companies should not produce invented tickers."""

    article_text = "The regional business reported quarterly results."

    result = detect_company_ticker(article_text)

    assert result.company is None
    assert result.ticker is None
    assert result.detection_confidence == 0.0


# ============================================================
# 5. EMPTY ARTICLE
# ============================================================

def test_rejects_empty_article_text() -> None:
    """Empty article text cannot be analysed."""

    with pytest.raises(ValueError, match="cannot be empty"):
        detect_company_ticker("   ")


# ============================================================
# 6. MISSING REFERENCE FILE
# ============================================================

def test_rejects_missing_mapping_file(tmp_path: Path) -> None:
    """A missing company reference file should raise a clear error."""

    missing_file = tmp_path / "missing_company_tickers.csv"

    with pytest.raises(FileNotFoundError):
        detect_company_ticker(
            "Apple announced new products.",
            mapping_path=missing_file,
        )
