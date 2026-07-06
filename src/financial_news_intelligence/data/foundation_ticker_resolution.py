"""Resolve company names in news titles to approved ticker symbols.

Purpose
-------
Read the project's curated company-to-ticker reference and provide one strict,
deterministic resolver for the market-data foundation. The resolver never
invents a ticker and never uses a web search to guess a company.

Inputs and grain
----------------
``data/reference/company_tickers.csv`` has one company per row. Required
columns are ``company``, ``ticker``, and ``aliases``. Aliases are separated by
``|``. Optional ``nasdaq_symbol``, ``stooq_symbol``, and ``exchange_timezone``
columns may override
provider and market-session defaults.

Logic and joins
---------------
Titles are normalized to lowercase alphanumeric tokens. A candidate ticker is
accepted only when at least one complete approved alias appears in the title.
Exactly one ticker must match; ambiguous or unmatched titles are rejected. The
returned join key is the uppercase ticker.

Outputs and downstream use
--------------------------
The accepted reference frame feeds GDELT query construction, Stooq symbol
selection, yfinance cross-checks, and article-level provenance. Rejection
reasons are written to the foundation QA evidence by the orchestrator.

Assumptions and limitations
---------------------------
The reference file is authoritative for this project but must be maintained by
the project owner. Short aliases such as one-character stock symbols are not
used as text matches because they create false positives. Existing comments and
reference values are preserved; this module reads but does not rewrite them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


class TickerResolutionError(ValueError):
    """Raised when the curated ticker reference is missing or ambiguous."""


@dataclass(frozen=True)
class TickerMatch:
    """Describe one deterministic title-to-ticker resolution result."""

    ticker: str | None
    company: str | None
    matched_alias: str | None
    method: str
    status: str
    reason: str | None


def normalize_text(value: object) -> str:
    """Return lowercase words separated by single spaces.

    Punctuation is removed so ``NVIDIA's`` and ``NVIDIA`` match consistently.
    The function intentionally keeps numbers because many company names and
    products contain meaningful digits.
    """

    text = str(value or "").lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    return " ".join(tokens)


def _alias_values(company: str, aliases: object) -> tuple[str, ...]:
    """Build one deduplicated alias tuple, including the legal company name."""

    raw_values = [company]
    raw_values.extend(str(aliases or "").split("|"))

    # Longest aliases are checked first so a specific legal name wins over a
    # shorter brand alias when both occur in the same title.
    normalized = {
        normalize_text(value)
        for value in raw_values
        if normalize_text(value)
    }
    return tuple(sorted(normalized, key=lambda value: (-len(value), value)))


def load_ticker_reference(file_path: Path) -> pd.DataFrame:
    """Load and validate the curated company-to-ticker reference.

    The output grain is one ticker row. ``alias_values`` is a tuple used only
    in memory; it is not written back into the user's source CSV.
    """

    if not file_path.exists() or file_path.is_symlink() or not file_path.is_file():
        raise TickerResolutionError(
            f"Missing or unsafe company ticker reference: {file_path}"
        )

    frame = pd.read_csv(file_path)
    required = {"company", "ticker", "aliases"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise TickerResolutionError(
            f"Company ticker reference is missing columns: {missing}"
        )
    if frame.empty:
        raise TickerResolutionError("Company ticker reference is empty.")

    result = frame.copy()
    result["company"] = result["company"].astype(str).str.strip()
    result["ticker"] = result["ticker"].astype(str).str.strip().str.upper()
    result["aliases"] = result["aliases"].fillna("").astype(str).str.strip()

    if result["company"].eq("").any() or result["ticker"].eq("").any():
        raise TickerResolutionError("Company and ticker values must be non-empty.")
    if result["ticker"].duplicated().any():
        duplicates = sorted(result.loc[result["ticker"].duplicated(), "ticker"])
        raise TickerResolutionError(f"Duplicate ticker rows: {duplicates}")

    result["alias_values"] = [
        _alias_values(company, aliases)
        for company, aliases in zip(result["company"], result["aliases"])
    ]

    # Nasdaq uses the canonical exchange ticker for the curated US equities.
    # A reference-level override remains available for future symbol changes.
    if "nasdaq_symbol" not in result.columns:
        result["nasdaq_symbol"] = result["ticker"]
    else:
        result["nasdaq_symbol"] = (
            result["nasdaq_symbol"].fillna("").astype(str).str.strip().str.upper()
        )
        result.loc[result["nasdaq_symbol"].eq(""), "nasdaq_symbol"] = (
            result["ticker"]
        )

    # Stooq uses lower-case market suffixes for US shares. A reference-level
    # override is preferred because non-US symbols need exchange-specific IDs.
    if "stooq_symbol" not in result.columns:
        result["stooq_symbol"] = result["ticker"].str.lower() + ".us"
    else:
        derived = result["ticker"].str.lower() + ".us"
        result["stooq_symbol"] = (
            result["stooq_symbol"].fillna("").astype(str).str.strip()
        )
        result.loc[result["stooq_symbol"].eq(""), "stooq_symbol"] = derived

    # The current project reference contains US equities. An optional timezone
    # column keeps the code extensible without silently guessing other markets.
    if "exchange_timezone" not in result.columns:
        result["exchange_timezone"] = "America/New_York"
    else:
        result["exchange_timezone"] = (
            result["exchange_timezone"]
            .fillna("America/New_York")
            .astype(str)
            .str.strip()
        )

    return result.reset_index(drop=True)


def resolve_title(title: object, reference: pd.DataFrame) -> TickerMatch:
    """Resolve one title only when exactly one approved ticker matches.

    The title match uses token boundaries around each normalized alias. Aliases
    shorter than three characters are ignored unless they contain a digit,
    because common words and one-letter symbols create unacceptable ambiguity.
    """

    normalized_title = normalize_text(title)
    if not normalized_title:
        return TickerMatch(
            ticker=None,
            company=None,
            matched_alias=None,
            method="curated_alias_exact_phrase",
            status="rejected",
            reason="empty_title",
        )

    candidates: list[tuple[str, str, str]] = []
    padded_title = f" {normalized_title} "

    for row in reference.itertuples(index=False):
        for alias in row.alias_values:
            if len(alias) < 3 and not any(character.isdigit() for character in alias):
                continue
            if f" {alias} " in padded_title:
                candidates.append((row.ticker, row.company, alias))
                break

    unique_tickers = {ticker for ticker, _, _ in candidates}
    if not candidates:
        return TickerMatch(
            ticker=None,
            company=None,
            matched_alias=None,
            method="curated_alias_exact_phrase",
            status="rejected",
            reason="no_approved_alias_match",
        )
    if len(unique_tickers) != 1:
        return TickerMatch(
            ticker=None,
            company=None,
            matched_alias=None,
            method="curated_alias_exact_phrase",
            status="rejected",
            reason="ambiguous_alias_match",
        )

    ticker, company, alias = candidates[0]
    return TickerMatch(
        ticker=ticker,
        company=company,
        matched_alias=alias,
        method="curated_alias_exact_phrase",
        status="accepted",
        reason=None,
    )
