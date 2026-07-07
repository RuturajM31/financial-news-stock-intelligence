"""Detect company names and ticker symbols in financial-news text."""

import csv
import re
from pathlib import Path

from rapidfuzz import fuzz

from financial_news_intelligence.paths import REFERENCE_DATA_DIR
from financial_news_intelligence.schemas import CompanyContext


# ============================================================
# 1. DEFAULT REFERENCE FILE
# ============================================================

# Use the permanent company-to-ticker mapping stored in the project.
DEFAULT_MAPPING_PATH = REFERENCE_DATA_DIR / "company_tickers.csv"


# ============================================================
# 2. LOAD COMPANY AND TICKER RECORDS
# ============================================================

def load_company_tickers(
    mapping_path: Path = DEFAULT_MAPPING_PATH,
) -> list[dict[str, object]]:
    """
    Load company names, tickers, and aliases.

    Input:  Company-ticker CSV file.
    Output: List of company records.
    Next:   Used by the company-detection function.
    """

    # Stop early when the reference file is missing.
    if not mapping_path.exists():
        raise FileNotFoundError(
            f"Company-ticker mapping not found: {mapping_path}"
        )

    company_records: list[dict[str, object]] = []

    # Read every company row from the reference CSV.
    with mapping_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        # Confirm that the required CSV columns exist.
        required_columns = {"company", "ticker", "aliases"}
        available_columns = set(reader.fieldnames or [])

        if not required_columns.issubset(available_columns):
            raise ValueError(
                "Company-ticker CSV must contain: "
                "company, ticker, aliases"
            )

        for row in reader:
            # Split aliases stored as: NVIDIA|Nvidia|NVIDIA Corp
            aliases = [
                alias.strip()
                for alias in row["aliases"].split("|")
                if alias.strip()
            ]

            # Include the official company name as another searchable name.
            searchable_names = [
                row["company"].strip(),
                *aliases,
            ]

            company_records.append(
                {
                    "company": row["company"].strip(),
                    "ticker": row["ticker"].strip().upper(),
                    "searchable_names": searchable_names,
                }
            )

    return company_records


# ============================================================
# 3. EXACT COMPANY MATCHING
# ============================================================

def find_exact_match(
    article_text: str,
    company_records: list[dict[str, object]],
) -> CompanyContext | None:
    """Find a complete company name or alias inside the article."""

    for record in company_records:
        searchable_names = record["searchable_names"]

        for company_name in searchable_names:
            # Word boundaries prevent "Meta" matching inside "metadata".
            pattern = rf"\b{re.escape(str(company_name))}\b"

            if re.search(pattern, article_text, flags=re.IGNORECASE):
                return CompanyContext(
                    company=str(record["company"]),
                    ticker=str(record["ticker"]),
                    detection_confidence=1.0,
                )

    return None


# ============================================================
# 4. FUZZY FALLBACK MATCHING
# ============================================================

def find_fuzzy_match(
    article_text: str,
    company_records: list[dict[str, object]],
    minimum_score: float = 90.0,
) -> CompanyContext | None:
    """
    Find a likely company when the article contains a small spelling variation.

    Input:  Article text and company reference records.
    Output: Best company match above the minimum score.
    """

    best_record: dict[str, object] | None = None
    best_score = 0.0

    for record in company_records:
        for company_name in record["searchable_names"]:
            # Compare each company name against the full article text.
            score = fuzz.partial_ratio(
                str(company_name).lower(),
                article_text.lower(),
            )

            if score > best_score:
                best_score = score
                best_record = record

    # Reject weak matches instead of inventing a company.
    if best_record is None or best_score < minimum_score:
        return None

    return CompanyContext(
        company=str(best_record["company"]),
        ticker=str(best_record["ticker"]),
        detection_confidence=round(best_score / 100, 4),
    )


# ============================================================
# 5. MAIN DETECTION FUNCTION
# ============================================================

def detect_company_ticker(
    article_text: str,
    mapping_path: Path = DEFAULT_MAPPING_PATH,
) -> CompanyContext:
    """
    Detect the main company and ticker from article text.

    Input:  Clean financial-news article text.
    Output: Company name, ticker, and detection confidence.
    Next:   Used by stock-price, ratio, and forecast services.
    """

    # Reject empty article text before searching.
    if not article_text.strip():
        raise ValueError("Article text cannot be empty.")

    company_records = load_company_tickers(mapping_path)

    # Use exact matching first because it is the most reliable.
    exact_match = find_exact_match(
        article_text,
        company_records,
    )

    if exact_match is not None:
        return exact_match

    # Use fuzzy matching only when no exact company name was found.
    fuzzy_match = find_fuzzy_match(
        article_text,
        company_records,
    )

    if fuzzy_match is not None:
        return fuzzy_match

    # Return an honest empty result when no company can be identified.
    return CompanyContext(
        company=None,
        ticker=None,
        detection_confidence=0.0,
    )
