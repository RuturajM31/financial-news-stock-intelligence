"""Render the public Financial News Sentiment Analyzer in Streamlit.

The module contains the four routed pages and coordinates article extraction,
Full BERT inference, visualizations, model results, and architecture views.
"""

from __future__ import annotations

# Standard library
import html
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Third-party libraries
import streamlit as st

try:
    import requests
except Exception:  # pragma: no cover - public fallback
    requests = None  # type: ignore[assignment]

try:
    import plotly.graph_objects as go
except Exception:  # pragma: no cover - public fallback
    go = None  # type: ignore[assignment]


current_file_path = Path(__file__).resolve()
application_directory = current_file_path.parent
PROJECT_ROOT = application_directory.parent
source_directory = PROJECT_ROOT / "src"

# The repository uses a src layout, so local packages need src on Python's path.
if str(source_directory) not in os.sys.path:
    os.sys.path.insert(0, str(source_directory))

# Local project imports. Their heavy model dependencies load lazily at runtime.
from financial_news_intelligence.models.full_bert_inference import (
    LABEL_ORDER as BERT_LABEL_ORDER,
    analyze_article as analyze_article_with_bert,
    contextual_tokens,
    deterministic_pca,
    load_bert_runtime,
    wordpiece_tokens,
)
from financial_news_intelligence.models.bert_visualization import (
    circular_positions,
    cosine_links,
    reader_segments,
    token_landscape,
)
from financial_news_intelligence.models.experiment_results import (
    ExperimentDataError,
    diagnostic_findings,
    error_flows,
    leaderboard,
    load_experiment_lab_data,
    normalize_confusion,
)

# Built-in sample used by presentation and article-analysis workflows.
_BASE_EXAMPLE = (
    "Nvidia raises outlook as data-center demand reaches a record. "
    "Nvidia reported record quarterly data-center revenue after demand for its artificial-intelligence chips remained strong. "
    "Management raised its full-year guidance and said cloud customers continued to expand computing capacity. "
    "The company also reported improving gross margins and a healthy order pipeline across several regions. "
    "Executives said new products were shipping on schedule and expected production capacity to increase during the next quarter. "
    "However, management warned that supply constraints, export controls, and rising competition could pressure future margins. "
    "The company also noted uncertainty around regulatory approvals in some international markets. "
    "Analysts said the stronger outlook was encouraging, while cautioning that the share price already reflected high expectations. "
    "Nvidia plans to provide another operational update when it publishes its next quarterly results."
)

# Lexical cue lists support transparent evidence; they do not affect Full BERT.
_POSITIVE_TERMS = {
    "record",
    "growth",
    "strong",
    "stronger",
    "beat",
    "beats",
    "profit",
    "profits",
    "revenue",
    "demand",
    "surge",
    "upgrade",
    "bullish",
    "positive",
    "margin expansion",
    "bookings",
    "cloud",
    "ai",
    "data center",
    "partnership",
    "guidance raised",
}

_NEGATIVE_TERMS = {
    "miss",
    "weak",
    "decline",
    "loss",
    "slowdown",
    "downgrade",
    "bearish",
    "negative",
    "pressure",
    "lawsuit",
    "probe",
    "recall",
    "cut",
    "guidance cut",
    "warning",
}

_RISK_TERMS = {
    "risk",
    "supply",
    "constraints",
    "export",
    "controls",
    "competition",
    "margin",
    "uncertainty",
    "regulation",
    "inventory",
    "volatility",
    "macro",
    "china",
    "policy",
}

# Ordered aliases support deterministic company and ticker detection.
_COMPANY_ENTITIES = (
    {"company": "SpaceX", "ticker": "", "private": True, "aliases": ("spacex", "space exploration technologies")},
    {"company": "NVIDIA Corporation", "ticker": "NVDA", "private": False, "aliases": ("nvidia", "nvda")},
    {"company": "Micron Technology", "ticker": "MU", "private": False, "aliases": ("micron", "micron technology", "mu")},
    {"company": "Apple Inc.", "ticker": "AAPL", "private": False, "aliases": ("apple", "apple inc", "aapl")},
    {"company": "Microsoft Corporation", "ticker": "MSFT", "private": False, "aliases": ("microsoft", "microsoft corporation", "msft")},
    {"company": "Tesla Inc.", "ticker": "TSLA", "private": False, "aliases": ("tesla", "tesla inc", "tsla")},
    {"company": "Amazon.com Inc.", "ticker": "AMZN", "private": False, "aliases": ("amazon", "amazon.com", "amzn")},
    {"company": "Meta Platforms", "ticker": "META", "private": False, "aliases": ("meta platforms", "meta", "facebook")},
    {"company": "Alphabet Inc.", "ticker": "GOOGL", "private": False, "aliases": ("alphabet", "google", "googl")},
    {"company": "Advanced Micro Devices", "ticker": "AMD", "private": False, "aliases": ("advanced micro devices", "amd")},
    {"company": "Intel Corporation", "ticker": "INTC", "private": False, "aliases": ("intel", "intel corporation", "intc")},
)


@dataclass(frozen=True)
class ArticleSignal:
    """Store lexical article signals used by the public visualizations.

    These rule-based values remain separate from the Full BERT prediction.
    """

    ticker: str
    company: str
    headline: str
    source: str
    label: str
    confidence: float
    sentiment_score: float
    movement_up: float
    movement_flat: float
    movement_down: float
    risk_score: float
    positive_hits: list[str]
    negative_hits: list[str]
    risk_hits: list[str]



def _safe(text: Any) -> str:
    """Convert a value to text and make it safe for custom HTML."""

    # Escape user-provided text before inserting it into rendered HTML.
    raw_text = str(text)
    escaped_text = html.escape(raw_text, quote=True)
    return escaped_text


def _clean_text(text: str | None) -> str:
    """Decode and normalize text onto one line for scoring."""

    if text is None:
        return ""

    decoded_text = html.unescape(html.unescape(str(text)))
    cleaned_text = re.sub(r"\\[nrt]", " ", decoded_text)
    cleaned_text = re.sub(r"\\+(?=['\"])", "", cleaned_text)
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
    return cleaned_text


def _clean_article_text(text: str | None) -> str:
    """Decode entities while preserving meaningful article paragraphs."""

    if text is None:
        return ""

    decoded_text = html.unescape(html.unescape(str(text)))
    normalized_text = decoded_text.replace("\r\n", "\n").replace("\r", "\n")
    normalized_text = re.sub(r"\\[nrt]", " ", normalized_text)
    normalized_text = re.sub(r"\\+(?=['\"])", "", normalized_text)

    cleaned_paragraphs = []
    for raw_paragraph in re.split(r"\n\s*\n|(?<=\.)\s{3,}", normalized_text):
        cleaned_paragraph = re.sub(r"[ \t]+", " ", raw_paragraph)
        cleaned_paragraph = re.sub(r"\n+", " ", cleaned_paragraph).strip()
        if cleaned_paragraph:
            cleaned_paragraphs.append(cleaned_paragraph)

    return "\n\n".join(cleaned_paragraphs)


def _term_count(text: str, term: str) -> int:
    """Count case-insensitive matches of one complete term or phrase."""

    # Letter-and-number boundaries prevent matches inside larger words.
    term_pattern = rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])"
    matching_terms = re.findall(term_pattern, text, re.IGNORECASE)
    return len(matching_terms)


def _hits(text: str, terms: set[str]) -> list[str]:
    """Return configured terms that appear as complete matches in text."""

    matching_terms = [term for term in terms if _term_count(text, term) > 0]
    return sorted(matching_terms)


def _sentiment_interpretation(score: float) -> str:
    """Select the displayed sentiment label from the configured thresholds."""

    if score <= -0.15:
        return "Bearish"
    if score >= 0.15:
        return "Bullish"
    return "Neutral / mixed"


def _infer_company(headline: str, body: str) -> tuple[str, str]:
    """Rank known entities using headline, article position, name, and ticker."""

    headline_text = _clean_text(headline)
    body_text = _clean_text(body)
    lead_text = body_text[:500]
    remaining_text = body_text[500:]
    combined_text = f"{headline_text} {body_text}"

    ranked_entities: list[tuple[int, int, dict[str, Any]]] = []
    for entity_order, entity in enumerate(_COMPANY_ENTITIES):
        entity_score = 0
        for alias in entity["aliases"]:
            entity_score += _term_count(headline_text, alias) * 8
            entity_score += _term_count(lead_text, alias) * 5
            entity_score += _term_count(remaining_text, alias)

        if _term_count(combined_text, entity["company"]) > 0:
            entity_score += 3

        ticker_symbol = entity["ticker"]
        if ticker_symbol and _term_count(combined_text, ticker_symbol) > 0:
            entity_score += 3

        ranked_entities.append((entity_score, -entity_order, entity))

    best_score, _entity_order, best_entity = max(
        ranked_entities,
        key=lambda item: (item[0], item[1]),
    )
    if best_score < 5:
        return "", "Company not confidently detected"
    if best_entity["private"]:
        return "Private company", best_entity["company"]
    return best_entity["ticker"], best_entity["company"]




def _is_boilerplate_fragment(text: str) -> bool:
    """Identify short navigation or website text that is not article content."""

    normalized_text = text.lower().strip(" .:|")
    boilerplate_phrases = (
        "skip to content", "accept cookies", "cookie policy", "privacy policy",
        "sign up", "subscribe", "newsletter", "follow us", "share this",
        "related articles", "recommended for you", "all rights reserved",
        "advertisement", "read more", "menu", "home", "contact us",
    )
    if any(phrase in normalized_text for phrase in boilerplate_phrases):
        return True

    fragment_words = normalized_text.split()
    has_sentence_ending = re.search(r"[.!?]$", text.strip())
    return len(fragment_words) < 6 and not has_sentence_ending


def _extract_article_content(url: str) -> tuple[str, str, str]:
    """Download a page and return its cleaned headline, body, and method."""

    article_url = _clean_text(url)
    if not article_url:
        raise ValueError("Add an article link first.")
    if not article_url.startswith(("http://", "https://")):
        raise ValueError("Article link must start with http:// or https://")
    if requests is None:
        raise RuntimeError("requests is unavailable in this public runtime")

    # BeautifulSoup stays optional so the public app can explain a missing parser.
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise RuntimeError("BeautifulSoup is unavailable in this public runtime") from exc

    response = requests.get(
        article_url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml",
        },
        timeout=12,
    )
    response.raise_for_status()

    page_html = response.text or ""
    if len(page_html.strip()) < 200:
        raise ValueError("The website returned too little readable content.")

    page = BeautifulSoup(page_html, "html.parser")

    # Remove page furniture before searching for the article itself.
    for unwanted_element in page.select(
        "script,style,noscript,nav,footer,header,aside,form,dialog,svg"
    ):
        unwanted_element.decompose()

    boilerplate_attribute_pattern = re.compile(
        r"cookie|consent|newsletter|subscribe|related|recommend|share|social|"
        r"footer|navigation|navbar|menu|breadcrumb|advert|promo|modal|popup",
        re.IGNORECASE,
    )
    boilerplate_elements = page.find_all(
        attrs={"class": boilerplate_attribute_pattern}
    ) + page.find_all(attrs={"id": boilerplate_attribute_pattern})
    for boilerplate_element in boilerplate_elements:
        boilerplate_element.decompose()

    # Prefer the visible page heading, then social metadata, then the HTML title.
    headline_element = page.find("h1")
    if headline_element is None:
        headline_element = page.find("meta", attrs={"property": "og:title"})
    if headline_element is None:
        headline_element = page.find("title")

    if headline_element is None:
        headline = ""
    elif headline_element.name == "meta":
        headline = _clean_text(headline_element.get("content", ""))
    else:
        headline = _clean_text(headline_element.get_text(" ", strip=True))

    # Search increasingly broad containers while preserving the original order.
    content_containers = [
        page.find("article"),
        page.find("main"),
        page.find(attrs={"itemprop": "articleBody"}),
        page.find(
            class_=re.compile(
                r"article[-_ ]?(body|content)|story[-_ ]?(body|content)|entry[-_ ]?content",
                re.IGNORECASE,
            )
        ),
    ]
    article_container = next(
        (element for element in content_containers if element is not None),
        page.body or page,
    )
    paragraph_elements = article_container.find_all("p")
    if not paragraph_elements:
        paragraph_elements = page.find_all("p")

    article_paragraphs = []
    seen_paragraph_keys = set()
    normalized_headline_key = re.sub(r"\W+", "", headline.lower())

    for paragraph_element in paragraph_elements:
        paragraph_text = _clean_article_text(
            paragraph_element.get_text(" ", strip=True)
        )
        paragraph_key = re.sub(r"\W+", "", paragraph_text.lower())
        if (
            not paragraph_text
            or not paragraph_key
            or paragraph_key == normalized_headline_key
            or paragraph_key in seen_paragraph_keys
        ):
            continue
        if _is_boilerplate_fragment(paragraph_text):
            continue

        seen_paragraph_keys.add(paragraph_key)
        article_paragraphs.append(paragraph_text)

    article_body = "\n\n".join(article_paragraphs)
    if len(article_body.split()) < 25:
        raise ValueError(
            "The website prevented reliable article extraction or exposed too little article text."
        )

    return headline, article_body[:20000].strip(), "Structured article extraction"




def _score_article(
    text: str,
    source: str,
    *,
    headline_text: str = "",
    body_text: str = "",
) -> ArticleSignal:
    """Calculate deterministic lexical sentiment, movement, and risk signals."""

    cleaned_text = _clean_text(text)
    if not cleaned_text:
        raise ValueError("Article text is required for analysis.")

    entity_headline = headline_text or cleaned_text.split(".")[0]
    entity_body = body_text or cleaned_text
    ticker_symbol, company_name = _infer_company(entity_headline, entity_body)

    # Lexical cues explain article language but do not influence Full BERT.
    positive_hits = _hits(cleaned_text, _POSITIVE_TERMS)
    negative_hits = _hits(cleaned_text, _NEGATIVE_TERMS)
    risk_hits = _hits(cleaned_text, _RISK_TERMS)

    positive_count = len(positive_hits)
    negative_count = len(negative_hits)
    risk_count = len(risk_hits)

    # Positive and negative phrase counts produce a bounded sentiment score.
    sentiment_score = max(
        -1.0,
        min(1.0, ((positive_count * 1.2) - (negative_count * 1.05)) / 6.0),
    )
    risk_score = max(
        0.05,
        min(0.95, (risk_count * 0.13) + (negative_count * 0.05)),
    )

    # Convert the lexical signals into three normalized movement scenarios.
    movement_up = 0.36 + sentiment_score * 0.28 - risk_score * 0.05
    movement_down = 0.22 - sentiment_score * 0.18 + risk_score * 0.12
    movement_flat = 1.0 - movement_up - movement_down
    movement_scores = [
        max(0.06, movement_up),
        max(0.06, movement_flat),
        max(0.06, movement_down),
    ]
    movement_total = sum(movement_scores)
    movement_up, movement_flat, movement_down = [
        movement_score / movement_total
        for movement_score in movement_scores
    ]

    confidence = min(
        0.94,
        max(
            0.58,
            0.62
            + abs(sentiment_score) * 0.22
            + min(len(cleaned_text), 4500) / 30000,
        ),
    )
    display_headline = (
        _clean_text(headline_text)
        or cleaned_text.split(".")[0].strip()[:210]
    )

    return ArticleSignal(
        ticker=ticker_symbol,
        company=company_name,
        headline=display_headline,
        source=source,
        label=_sentiment_interpretation(sentiment_score),
        confidence=confidence,
        sentiment_score=sentiment_score,
        movement_up=movement_up,
        movement_flat=movement_flat,
        movement_down=movement_down,
        risk_score=risk_score,
        positive_hits=positive_hits,
        negative_hits=negative_hits,
        risk_hits=risk_hits,
    )




def _plotly_config() -> dict[str, bool]:
    """Return shared responsive settings for Plotly charts."""

    return {"displayModeBar": False, "responsive": True}


def _apply_theme() -> None:
    """Configure Streamlit and inject shared CSS for every public page."""

    # This wrapper applies presentation only; it does not render a route or load a model.
    st.set_page_config(
        page_title="Financial News Sentiment Analyzer",
        page_icon="📰",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # These selectors must stay synchronized with the HTML classes emitted by page renderers.
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap');

        html, body, [class*="css"] {
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }

        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at 8% 3%, rgba(29, 78, 216, .16), transparent 30rem),
                radial-gradient(circle at 86% 10%, rgba(126, 34, 206, .16), transparent 34rem),
                linear-gradient(180deg, #040914 0%, #07101f 52%, #040914 100%) !important;
            color: #f8fafc;
        }

        [data-testid="stHeader"] {
            background: rgba(4, 9, 20, .82) !important;
            border-bottom: 1px solid rgba(148, 163, 184, .10) !important;
            backdrop-filter: blur(20px);
        }

        [data-testid="stSidebar"] {
            background:
                radial-gradient(circle at 0% 0%, rgba(37, 99, 235, .18), transparent 18rem),
                linear-gradient(180deg, #030712 0%, #07101f 58%, #030712 100%) !important;
            border-right: 1px solid rgba(59, 130, 246, .20) !important;
        }

        [data-testid="stSidebar"] * {
            color: #dbeafe;
        }

        .block-container {
            padding-top: 1.1rem !important;
            padding-bottom: 1.2rem !important;
            max-width: 1500px !important;
        }

        .brand {
            padding: 1.1rem 1rem 1.2rem 1rem;
            border-bottom: 1px solid rgba(148, 163, 184, .13);
            margin-bottom: 1rem;
        }
        .brand-row {
            display: flex;
            align-items: center;
            gap: .8rem;
        }
        .brand-icon {
            width: 34px;
            height: 34px;
            border-radius: 12px;
            display: grid;
            place-items: center;
            background: linear-gradient(135deg, rgba(59,130,246,.28), rgba(139,92,246,.30));
            border: 1px solid rgba(96,165,250,.28);
            color: #a5b4fc;
            font-weight: 900;
        }
        .brand-title {
            font-size: 1rem;
            font-weight: 900;
            color: #ffffff;
            line-height: 1.15;
        }
        .brand-subtitle {
            font-size: .76rem;
            color: #cbd5e1;
            margin-top: .16rem;
        }
        .nav-title {
            color: #94a3b8;
            font-size: .70rem;
            font-weight: 800;
            letter-spacing: .08em;
            text-transform: uppercase;
            margin: .8rem 0 .55rem .15rem;
        }
        .nav-item {
            padding: .62rem .72rem;
            border-radius: 11px;
            color: #dbeafe;
            font-size: .84rem;
            font-weight: 650;
            margin-bottom: .25rem;
            display: flex;
            gap: .58rem;
            align-items: center;
        }
        .nav-item.active {
            background: linear-gradient(135deg, rgba(37,99,235,.95), rgba(29,78,216,.75));
            color: white;
            box-shadow: 0 12px 30px rgba(37,99,235,.20);
        }
        .side-card {
            margin-top: 1.2rem;
            padding: .88rem;
            border-radius: 12px;
            background: rgba(15, 23, 42, .66);
            border: 1px solid rgba(148, 163, 184, .12);
        }
        .status-green {
            color: #4ade80;
            font-size: .78rem;
            font-weight: 800;
        }

        .topbar {
            display: grid;
            grid-template-columns: minmax(0, 1fr) auto;
            align-items: center;
            gap: 1rem;
            margin-bottom: 1rem;
        }
        .title-wrap {
            display: flex;
            align-items: center;
            gap: .8rem;
        }
        .title-icon {
            width: 38px;
            height: 38px;
            display: grid;
            place-items: center;
            border-radius: 13px;
            color: #a78bfa;
            background: linear-gradient(135deg, rgba(59,130,246,.17), rgba(139,92,246,.20));
            border: 1px solid rgba(129,140,248,.28);
            font-weight: 900;
        }
        .page-title {
            font-size: 1.58rem;
            line-height: 1.05;
            font-weight: 950;
            color: white;
            letter-spacing: -.04em;
        }
        .page-subtitle {
            margin-top: .32rem;
            color: #cbd5e1;
            font-size: .82rem;
        }
        .chip-row {
            display: flex;
            gap: .55rem;
            flex-wrap: wrap;
            justify-content: flex-end;
        }
        .chip {
            border-radius: 9px;
            padding: .56rem .74rem;
            font-size: .78rem;
            font-weight: 850;
            color: #ecfeff;
            background: rgba(15,23,42,.70);
            border: 1px solid rgba(96,165,250,.20);
        }
        .chip.green {border-color: rgba(34,197,94,.28); color: #86efac;}
        .chip.blue {border-color: rgba(59,130,246,.30); color: #7dd3fc;}
        .chip.purple {border-color: rgba(168,85,247,.30); color: #d8b4fe;}
        .chip.cyan {border-color: rgba(20,184,166,.30); color: #67e8f9;}

        .card {
            border-radius: 13px;
            background:
                radial-gradient(circle at 100% 0%, rgba(59,130,246,.07), transparent 18rem),
                linear-gradient(145deg, rgba(15, 23, 42, .92), rgba(8, 13, 28, .94));
            border: 1px solid rgba(148, 163, 184, .14);
            box-shadow: 0 18px 48px rgba(0,0,0,.22);
        }
        .article-strip {
            padding: 1rem 1.05rem;
            margin-bottom: .7rem;
            display: grid;
            grid-template-columns: 190px minmax(0,1fr) 170px 170px 110px;
            gap: 1rem;
            align-items: center;
        }
        .brand-stock {
            display: flex;
            gap: .75rem;
            align-items: center;
        }
        .stock-logo {
            width: 44px;
            height: 44px;
            border-radius: 10px;
            display: grid;
            place-items: center;
            font-weight: 950;
            color: #022c22;
            background: linear-gradient(135deg, #84cc16, #22c55e);
        }
        .tiny-label {
            color: #94a3b8;
            font-size: .68rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: .04em;
        }
        .strong {
            color: white;
            font-size: 1rem;
            font-weight: 900;
        }
        .muted {
            color: #cbd5e1;
            font-size: .78rem;
        }
        .view-btn {
            padding: .65rem .7rem;
            border-radius: 9px;
            background: rgba(15, 23, 42, .85);
            border: 1px solid rgba(148, 163, 184, .15);
            color: white;
            font-weight: 850;
            font-size: .74rem;
            text-align: center;
        }

        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: .62rem;
            margin-bottom: .62rem;
        }
        .kpi {
            min-height: 98px;
            padding: .92rem;
            position: relative;
            overflow: hidden;
        }
        .kpi.purple {border-color: rgba(168,85,247,.35);}
        .kpi.teal {border-color: rgba(20,184,166,.35);}
        .kpi.green {border-color: rgba(34,197,94,.35);}
        .kpi.orange {border-color: rgba(249,115,22,.35);}
        .kpi.violet {border-color: rgba(139,92,246,.35);}
        .kpi-title {
            color: #cbd5e1;
            font-size: .70rem;
            font-weight: 850;
            margin-bottom: .28rem;
        }
        .kpi-value {
            color: white;
            font-size: 1.55rem;
            font-weight: 950;
            letter-spacing: -.05em;
        }
        .kpi-sub {
            color: #dbeafe;
            font-size: .76rem;
            margin-top: .12rem;
        }
        .spark {
            position: absolute;
            right: .8rem;
            bottom: .6rem;
            width: 74px;
            height: 28px;
            opacity: .95;
        }

        .two-col {
            display: grid;
            grid-template-columns: 1.05fr .95fr;
            gap: .62rem;
            margin-bottom: .62rem;
        }
        .insight {
            padding: 1.05rem 1.15rem;
            border-color: rgba(34,211,238,.38);
            background:
                radial-gradient(circle at 100% 100%, rgba(20,184,166,.18), transparent 20rem),
                linear-gradient(145deg, rgba(8,47,73,.55), rgba(8,13,28,.96));
        }
        .panel-title {
            color: #7dd3fc;
            font-size: .86rem;
            font-weight: 950;
            margin-bottom: .62rem;
        }
        .insight-text {
            color: white;
            font-size: 1.05rem;
            line-height: 1.48;
            font-weight: 700;
        }
        .green-text {color:#4ade80;font-weight:950;}
        .analysis-list {
            padding: 1.05rem 1.15rem;
        }
        .analysis-list ul {
            margin: 0;
            padding-left: 1.1rem;
            color: #e5e7eb;
            font-size: .84rem;
            line-height: 1.62;
        }

        .chart-grid {
            display: grid;
            grid-template-columns: .82fr .82fr 1.05fr 1.15fr;
            gap: .62rem;
            margin-bottom: .62rem;
        }
        .chart-card {
            padding: .85rem;
            min-height: 198px;
        }
        .chart-title {
            color: white;
            font-size: .82rem;
            font-weight: 900;
            margin-bottom: .4rem;
        }

        .workflow {
            padding: .9rem;
            margin-bottom: .62rem;
        }
        .workflow-grid {
            display: grid;
            grid-template-columns: repeat(7, minmax(0, 1fr));
            gap: .48rem;
        }
        .step {
            border-radius: 10px;
            padding: .62rem;
            border: 1px solid rgba(96,165,250,.16);
            background: rgba(15,23,42,.58);
            min-height: 74px;
        }
        .step-num {
            color: #c4b5fd;
            font-size: .68rem;
            font-weight: 900;
        }
        .step-text {
            color: #e2e8f0;
            font-size: .68rem;
            line-height: 1.35;
            margin-top: .22rem;
        }

        .bottom-grid {
            display: grid;
            grid-template-columns: 1fr 1fr 1.2fr;
            gap: .62rem;
        }
        .tag-panel {
            padding: .9rem;
        }
        .tag-title {
            font-size: .86rem;
            font-weight: 950;
            margin-bottom: .65rem;
        }
        .tag-title.good {color:#4ade80;}
        .tag-title.warn {color:#f59e0b;}
        .tag-title.info {color:#38bdf8;}
        .tag {
            display: inline-block;
            padding: .42rem .62rem;
            border-radius: 999px;
            background: rgba(148,163,184,.10);
            color: #cbd5e1;
            font-size: .72rem;
            font-weight: 750;
            margin: 0 .32rem .32rem 0;
        }

        .input-card {
            padding: 1.1rem;
            margin-bottom: .8rem;
        }
        .stTextInput > div > div,
        .stTextArea textarea,
        .stFileUploader > div {
            background: rgba(2, 6, 23, .88) !important;
            border: 1px solid rgba(59,130,246,.32) !important;
            border-radius: 13px !important;
        }
        .stButton button {
            border-radius: 13px !important;
            min-height: 3rem !important;
            font-weight: 900 !important;
        }
        .stFormSubmitButton button[kind="primary"] {
            background: linear-gradient(135deg, #0ea5e9, #7c3aed) !important;
        }
        .info-strip {
            padding: .9rem 1rem;
            border-radius: 12px;
            margin-top: .75rem;
            color: #93c5fd;
            background: linear-gradient(135deg, rgba(29,78,216,.28), rgba(88,28,135,.20));
            border: 1px solid rgba(59,130,246,.24);
            font-weight: 650;
        }
        .footer {
            color: #64748b;
            font-size: .70rem;
            margin-top: .65rem;
        }

        @media (max-width: 1200px) {
            .article-strip, .kpi-grid, .two-col, .chart-grid, .workflow-grid, .bottom-grid {
                grid-template-columns: 1fr;
            }
            .topbar {
                grid-template-columns: 1fr;
            }
            .chip-row {
                justify-content: flex-start;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_sidebar() -> str:
    """Render the stable four-page public navigation and return its selection."""

    # Stable route labels preserve navigation state across Streamlit reruns.
    page_options = [
        "Overview",
        "Analyze Article",
        "Model Results",
        "About / Architecture",
    ]

    st.sidebar.markdown(
        """
        <div class="brand">
          <div class="brand-row">
            <div class="brand-icon">↗</div>
            <div>
              <div class="brand-title">Financial News Sentiment Analyzer</div>
              <div class="brand-subtitle">Transformer experimentation and article-level financial sentiment analysis</div>
            </div>
          </div>
        </div>
        <div class="nav-title">Pages</div>
        """,
        unsafe_allow_html=True,
    )

    selected_page = st.sidebar.radio(
        "Pages",
        page_options,
        index=0,
        key="public_dashboard_page",
        label_visibility="collapsed",
    )

    st.sidebar.markdown(
        """
        <div class="side-card">
          <div class="tiny-label">Scope</div>
          <div class="status-green">● Financial-news text analysis</div>
        </div>
        <div style="position:fixed;bottom:1rem;color:#64748b;font-size:.72rem;">© 2025 Ruturaj Mokashi</div>
        """,
        unsafe_allow_html=True,
    )

    return selected_page








































































def _render_premium_sentiment_styles() -> None:
    """Inject unchanged shared CSS used by all four focused public pages."""

    # unsafe_allow_html is required because Streamlit receives the existing style element directly.
    st.markdown(
        """
        <style>
        :root{--fs-surface:rgba(15,32,56,.94);--fs-border:rgba(125,211,252,.18);--fs-cyan:#22d3ee;--fs-muted:#a9b8cc;--fs-radius:18px}
        [data-testid="stMainBlockContainer"]{max-width:1220px!important;padding-top:2rem!important;padding-bottom:3rem!important}
        [data-testid="stSidebar"]{min-width:220px!important;max-width:220px!important}
        [data-testid="stSidebar"] [role="radiogroup"] label{border-radius:12px;padding:.58rem .7rem;margin:.12rem 0;border:1px solid transparent}
        [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked){color:#ecfeff;background:rgba(8,145,178,.22);border-color:rgba(34,211,238,.32);box-shadow:inset 3px 0 #22d3ee}
        .fs-head{margin:0 0 1.8rem}.fs-eye{color:var(--fs-cyan);font-size:.78rem;font-weight:850;letter-spacing:.13em;text-transform:uppercase}
        .fs-title{color:#f8fafc;font-size:clamp(2.15rem,4vw,2.55rem);line-height:1.08;letter-spacing:-.045em;margin:.4rem 0 .65rem}
        .fs-copy{color:#c7d3e3;font-size:1rem;line-height:1.65;max-width:760px;margin:0}
        .fs-section{margin:2rem 0 .85rem}.fs-section h2{color:#f8fafc;font-size:1.48rem;letter-spacing:-.025em;margin:.25rem 0}.fs-section p{color:var(--fs-muted);font-size:.96rem;margin:0}
        .fs-card{background:radial-gradient(circle at 100% 0%,rgba(34,211,238,.08),transparent 14rem),linear-gradient(145deg,rgba(19,40,68,.96),var(--fs-surface));border:1px solid var(--fs-border);border-radius:var(--fs-radius);box-shadow:0 18px 42px rgba(0,0,0,.22);padding:1.25rem;overflow:hidden}
        .fs-hero{display:grid;grid-template-columns:minmax(0,3fr) minmax(320px,2fr);gap:1.25rem;margin-bottom:1rem}.fs-hero-copy{padding:1.75rem;display:flex;flex-direction:column;justify-content:center}
        .fs-preview-label{color:#a5f3fc;font-size:.75rem;font-weight:800;letter-spacing:.09em;text-transform:uppercase}
        .fs-preview-row{display:flex;justify-content:space-between;gap:1rem;padding:.72rem 0;border-bottom:1px solid rgba(148,163,184,.12)}.fs-preview-row:last-child{border:0}.fs-preview-row span{color:var(--fs-muted);font-size:.86rem}.fs-preview-row strong{color:#f8fafc;text-align:right}
        .fs-grid3,.fs-kpis{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:1rem;margin:1rem 0 2rem}.fs-kpi{min-height:118px}.fs-kpi strong{display:block;color:#f8fafc;font-size:1.75rem;letter-spacing:-.04em;margin:.35rem 0}.fs-kpi span{color:var(--fs-muted);font-size:.9rem}
        .fs-flow{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:.8rem}.fs-step{min-height:120px}.fs-icon{width:36px;height:36px;display:grid;place-items:center;color:#cffafe;border-radius:11px;background:rgba(8,145,178,.22);border:1px solid rgba(34,211,238,.25);margin-bottom:.7rem}.fs-step strong{color:#f8fafc;display:block}.fs-step span{color:var(--fs-muted);font-size:.86rem}
        .fs-note{margin-top:1.3rem;padding:1rem 1.15rem;border-radius:var(--fs-radius);color:#cbd5e1;font-size:.94rem;border:1px solid rgba(148,163,184,.16);background:rgba(15,23,42,.58)}
        .fs-empty{min-height:300px;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;color:var(--fs-muted)}.fs-empty-icon{width:52px;height:52px;display:grid;place-items:center;border-radius:16px;color:#67e8f9;background:rgba(8,145,178,.16);border:1px solid rgba(34,211,238,.22);font-size:1.35rem;margin-bottom:.8rem}
        .fs-meta,.fs-tech{display:flex;flex-wrap:wrap;gap:.5rem;margin:.65rem 0}.fs-chip,.fs-tech span{padding:.38rem .64rem;border-radius:999px;color:#cbd5e1;background:rgba(15,23,42,.76);border:1px solid rgba(148,163,184,.16);font-size:.8rem}
        .fs-results{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:.85rem;margin:.9rem 0 1.8rem}.fs-result{min-height:112px}.fs-result span{color:var(--fs-muted);font-size:.82rem}.fs-result strong{color:#f8fafc;display:block;font-size:1.25rem;letter-spacing:-.03em;margin-top:.45rem}
        .fs-highlight{color:#dce8f6;font-size:.96rem;line-height:1.8;white-space:pre-wrap}.fs-pos{background:rgba(52,211,153,.18);color:#a7f3d0;padding:.08rem .18rem;border-radius:4px}.fs-neg{background:rgba(251,113,133,.18);color:#fecdd3;padding:.08rem .18rem;border-radius:4px}.fs-risk{background:rgba(251,191,36,.18);color:#fde68a;padding:.08rem .18rem;border-radius:4px}
        .fs-legend{display:flex;gap:.75rem;flex-wrap:wrap;margin:.3rem 0 1rem}.fs-legend span{font-size:.8rem;color:var(--fs-muted)}
        .fs-winner{border-color:rgba(34,211,238,.42);background:radial-gradient(circle at 90% 0%,rgba(34,211,238,.18),transparent 18rem),linear-gradient(145deg,rgba(17,48,76,.98),rgba(12,27,48,.96))}.fs-badge{display:inline-flex;padding:.28rem .58rem;border-radius:999px;color:#cffafe;background:rgba(8,145,178,.24);border:1px solid rgba(34,211,238,.3);font-size:.74rem;font-weight:800}
        .fs-classes{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:.85rem}.fs-class{text-align:center;min-height:105px}.fs-class strong{color:#f8fafc;font-size:1.05rem;display:block;margin:.45rem 0}
        .fs-lane{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:.65rem;margin:.8rem 0 2rem}.fs-node{min-height:126px;text-align:center}.fs-node .fs-icon{margin:0 auto .65rem}.fs-node strong{color:#f8fafc;display:block;font-size:.9rem}.fs-node span{color:var(--fs-muted);font-size:.77rem;line-height:1.4;display:block;margin-top:.3rem}
        .fs-limits{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.6rem}.fs-limit{padding:.8rem;border-radius:12px;color:#cbd5e1;background:rgba(15,23,42,.55);border:1px solid rgba(148,163,184,.13)}
        .stButton>button{min-height:44px;border-radius:12px;font-weight:750}[data-testid="stMetric"]{background:linear-gradient(145deg,rgba(19,40,68,.92),rgba(14,29,52,.92));border:1px solid var(--fs-border);border-radius:var(--fs-radius);padding:1rem}
        [data-testid="stCaptionContainer"] p{font-size:.88rem;line-height:1.5}
        @media(max-width:900px){[data-testid="stMainBlockContainer"]{padding-left:1rem!important;padding-right:1rem!important}.fs-hero,.fs-grid3,.fs-kpis,.fs-flow,.fs-results,.fs-classes,.fs-lane,.fs-limits{grid-template-columns:1fr}.fs-title{font-size:2rem}}
                .fs-evidence-flow{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:.55rem;margin:.8rem 0 1rem}.fs-evidence-flow .fs-step{min-height:92px;padding:.85rem}
        .fs-cloud{display:flex;flex-wrap:wrap;align-items:center;justify-content:center;gap:.7rem;padding:1.2rem;min-height:180px}.fs-cloud-term{display:inline-flex;padding:.38rem .65rem;border-radius:11px;border:1px solid currentColor;line-height:1.15;cursor:help;background:rgba(15,23,42,.62)}
        .fs-reader{max-width:860px;margin:0 auto;color:#dce8f6;font-size:1rem;line-height:1.9}.fs-reader p{margin:0 0 1.15rem}.fs-reader-head{display:grid;grid-template-columns:2fr 1fr;gap:1rem;margin-bottom:1rem}
        .fs-reader-meta{position:sticky;top:.5rem;z-index:2;background:rgba(15,32,56,.96);padding:.75rem;border-radius:12px;border:1px solid var(--fs-border)}
        .fs-details{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.7rem;margin-top:1rem}.fs-detail{padding:.9rem}.fs-detail strong{color:#f8fafc;display:block;margin:.25rem 0}.fs-detail p{color:var(--fs-muted);font-size:.82rem;line-height:1.45;margin:0}
        mark.fs-pos,mark.fs-neg,mark.fs-risk{padding:.08rem .2rem;border-radius:5px}
        .fs-divider{height:1px;background:linear-gradient(90deg,transparent,rgba(34,211,238,.25),transparent);margin:2rem 0}
        @media(max-width:900px){.fs-evidence-flow,.fs-reader-head,.fs-details{grid-template-columns:1fr}.fs-reader-meta{position:static}.fs-cloud{justify-content:flex-start}}        .fs-result-separator{display:flex;align-items:center;gap:1rem;margin:2rem 0 .8rem;color:#67e8f9;font-size:.72rem;font-weight:900;letter-spacing:.16em}.fs-result-separator:before,.fs-result-separator:after{content:"";height:1px;flex:1;background:linear-gradient(90deg,transparent,rgba(34,211,238,.34),transparent)}
        .fs-bert-hero{display:grid;grid-template-columns:minmax(0,1.1fr) minmax(360px,.9fr);gap:1.35rem;align-items:center}.fs-bert-hero h2{font-size:2.5rem;color:#f8fafc;margin:.35rem 0}.fs-bert-hero p{color:var(--fs-muted);line-height:1.65;max-width:640px}.fs-result-facts{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.65rem}.fs-result-facts span{padding:.7rem;border-radius:11px;background:rgba(7,18,36,.64);color:var(--fs-muted);font-size:.72rem}.fs-result-facts strong{display:block;color:#f8fafc;font-size:.9rem;margin-top:.2rem}
        .fs-score-cards{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:.75rem;margin:.25rem 0 .8rem}.fs-score-card{border-top:3px solid;background:rgba(15,32,56,.82);border-radius:11px;padding:.75rem 1rem}.fs-score-card span{color:var(--fs-muted);font-size:.78rem}.fs-score-card strong{display:block;color:#f8fafc;font-size:1.2rem;margin-top:.18rem}
        .fs-mini-score{display:grid;grid-template-columns:62px 1fr 52px;align-items:center;gap:.5rem;margin:.55rem 0;font-size:.73rem;color:var(--fs-muted)}.fs-mini-score>div{height:7px;background:rgba(148,163,184,.16);border-radius:10px;overflow:hidden}.fs-mini-score i{display:block;height:100%;border-radius:10px}.fs-mini-score strong{text-align:right;color:#dbeafe}
        .fs-selected{margin-bottom:.7rem}.fs-selected p{color:#f8fafc;line-height:1.65;font-size:1rem}.fs-token-row{display:flex;flex-wrap:wrap;gap:.42rem;padding:.8rem;border:1px solid var(--fs-border);border-radius:12px;background:rgba(7,18,36,.72)}.fs-token-chip{display:inline-flex;padding:.3rem .52rem;border-radius:8px;background:rgba(34,211,238,.1);border:1px solid rgba(34,211,238,.24);color:#bae6fd;font-family:"IBM Plex Mono",monospace;font-size:.76rem}
        .fs-token-landscape{display:flex;flex-wrap:wrap;align-items:center;justify-content:center;gap:.58rem;min-height:145px}.fs-landscape-token{display:inline-flex;align-items:baseline;gap:.22rem;padding:.34rem .56rem;border:1px solid currentColor;border-radius:9px;background:rgba(7,18,36,.72);line-height:1.1}.fs-landscape-token small{font-size:.58em;opacity:.68}
        .fs-sticky-legend{position:sticky;top:.5rem;z-index:8;display:flex;flex-wrap:wrap;align-items:center;gap:.48rem;padding:.65rem .8rem;margin:.5rem 0 .9rem;background:rgba(7,18,36,.96);border:1px solid var(--fs-border);border-radius:11px;font-size:.74rem}.fs-sticky-legend strong{color:#f8fafc;margin-right:.2rem}.fs-sticky-legend span,.fs-sticky-legend mark{padding:.18rem .38rem;border-radius:6px}
        .fs-dual-reader{max-width:none;font-size:1.05rem;line-height:1.85;white-space:normal}.fs-bert-sentence{display:inline;border-left:3px solid transparent;padding:.08rem .16rem;margin:0 .03rem;border-radius:4px;transition:opacity .18s ease,background .18s ease}.fs-bert-bearish{border-left-color:#fb7185;background:rgba(251,113,133,.10)}.fs-bert-neutral{border-left-color:#22d3ee;background:rgba(34,211,238,.08)}.fs-bert-bullish{border-left-color:#34d399;background:rgba(52,211,153,.09)}.fs-dim{opacity:.28}.fs-navigator{position:sticky;top:4.25rem}.fs-navigator>strong{display:block;color:#f8fafc;font-size:1.25rem;margin:.3rem 0 .6rem}
        [data-testid="stTabs"] [data-baseweb="tab-list"]{gap:.25rem;background:rgba(7,18,36,.66);padding:.32rem;border:1px solid var(--fs-border);border-radius:13px;overflow-x:auto}[data-testid="stTabs"] button[data-baseweb="tab"]{border-radius:9px;padding:.58rem .8rem;white-space:nowrap}[data-testid="stTabs"] button[aria-selected="true"]{background:rgba(34,211,238,.13);color:#cffafe}
        @media(max-width:900px){.fs-bert-hero{grid-template-columns:1fr}.fs-result-facts{grid-template-columns:repeat(2,minmax(0,1fr))}.fs-score-cards{grid-template-columns:1fr}.fs-navigator{position:static}.fs-sticky-legend{position:static}}
        @media(max-width:430px){.fs-bert-hero h2{font-size:1.85rem}.fs-result-facts{grid-template-columns:1fr}.fs-mini-score{grid-template-columns:54px 1fr 45px}.fs-token-chip{font-size:.68rem}.fs-dual-reader{font-size:1rem;line-height:1.72}[data-testid="stTabs"] button[data-baseweb="tab"]{padding:.48rem .62rem;font-size:.78rem}}        .ml-badge{display:inline-flex;padding:.2rem .45rem;border-radius:999px;font-size:.62rem;font-weight:900;letter-spacing:.08em;border:1px solid rgba(34,211,238,.35);color:#67e8f9;background:rgba(34,211,238,.1);vertical-align:middle}.ml-historical{color:#c4b5fd;border-color:rgba(167,139,250,.38);background:rgba(167,139,250,.11)}.ml-runtime{color:#fde68a;border-color:rgba(251,191,36,.35);background:rgba(251,191,36,.1)}
        .ml-champion{display:grid;grid-template-columns:minmax(0,.8fr) minmax(520px,1.2fr);gap:1.35rem;align-items:center;margin-bottom:.85rem}.ml-champion h2{color:#f8fafc;margin:.6rem 0 .35rem;font-size:1.65rem}.ml-champion p{color:var(--fs-muted);line-height:1.6}.ml-champion-metrics{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:.65rem}.ml-champion-metrics>div{padding:.75rem;border-radius:11px;background:rgba(7,18,36,.7);border:1px solid rgba(148,163,184,.13)}.ml-champion-metrics span{display:block;color:var(--fs-muted);font-size:.7rem}.ml-champion-metrics strong{display:block;color:#f8fafc;font-size:1.08rem;margin-top:.25rem}
        .ml-summary-strip{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:.55rem;margin:.7rem 0 1.1rem}.ml-summary-strip span{padding:.62rem .72rem;text-align:center;border-radius:10px;border:1px solid var(--fs-border);background:rgba(15,32,56,.72);color:#cbd5e1;font-size:.75rem;font-weight:750}
        .ml-timeline{display:grid;grid-template-columns:repeat(8,minmax(0,1fr));gap:.45rem;margin:.3rem 0 1rem}.ml-timeline div{position:relative;padding:.75rem .55rem;border-radius:10px;background:rgba(15,32,56,.76);border:1px solid var(--fs-border);min-height:82px}.ml-timeline span{display:block;color:#22d3ee;font-size:.65rem;font-weight:900}.ml-timeline strong{display:block;color:#f8fafc;font-size:.74rem;line-height:1.35;margin-top:.35rem}
        .ml-resource-grid,.ml-runtime-grid,.ml-diagnostic-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.65rem}.ml-resource-grid .fs-card,.ml-runtime-grid .fs-card,.ml-diagnostic-grid .fs-card{padding:.85rem;min-height:88px}.ml-resource-grid span,.ml-runtime-grid span,.ml-diagnostic-grid span{display:block;color:var(--fs-muted);font-size:.7rem}.ml-resource-grid strong,.ml-runtime-grid strong,.ml-diagnostic-grid strong{display:block;color:#f8fafc;font-size:.92rem;margin-top:.35rem;overflow-wrap:anywhere}
        .ml-architecture-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:.8rem}.ml-architecture-grid strong{display:block;color:#f8fafc;font-size:1.02rem;margin:.55rem 0}.ml-architecture-grid span{display:block;color:var(--fs-muted);font-size:.78rem;line-height:1.45;padding:.2rem 0;border-bottom:1px solid rgba(148,163,184,.09)}
        @media(max-width:1050px){.ml-champion{grid-template-columns:1fr}.ml-summary-strip{grid-template-columns:repeat(2,minmax(0,1fr))}.ml-timeline{grid-template-columns:repeat(4,minmax(0,1fr))}}
        @media(max-width:600px){.ml-champion-metrics,.ml-resource-grid,.ml-runtime-grid,.ml-diagnostic-grid,.ml-architecture-grid{grid-template-columns:1fr}.ml-summary-strip{grid-template-columns:1fr}.ml-timeline{grid-template-columns:repeat(2,minmax(0,1fr))}.ml-champion h2{font-size:1.35rem}}        .ov-eyebrow{color:#67e8f9;font-size:.69rem;font-weight:900;letter-spacing:.18em;text-transform:uppercase}.ov-hero{position:relative;display:grid;grid-template-columns:minmax(0,1.08fr) minmax(430px,.92fr);gap:2.3rem;align-items:center;padding:2.1rem 0 1.2rem;isolation:isolate}.ov-hero:before{content:"";position:absolute;inset:-4rem -20vw auto 38%;height:24rem;background:radial-gradient(ellipse,rgba(8,145,178,.11),transparent 66%);z-index:-1;pointer-events:none}.ov-hero-copy h1{max-width:760px;margin:.65rem 0 1rem;color:#f8fafc;font-size:clamp(2.55rem,5vw,5rem);line-height:.98;letter-spacing:-.06em}.ov-hero-copy p{max-width:680px;color:#a8bad0;font-size:1.05rem;line-height:1.7}.ov-status{display:flex;flex-wrap:wrap;gap:.45rem;margin-top:1.3rem}.ov-status span{padding:.32rem .55rem;border-left:2px solid #22d3ee;background:rgba(15,32,56,.55);color:#cbd5e1;font-size:.69rem;font-weight:750}
        .ov-canvas{position:relative;min-height:370px;padding:1.15rem;border:1px solid rgba(34,211,238,.24);border-radius:22px;background:linear-gradient(155deg,rgba(15,32,56,.92),rgba(5,14,28,.97));box-shadow:0 25px 70px rgba(0,0,0,.28);overflow:hidden}.ov-canvas:before{content:"";position:absolute;inset:0;background-image:linear-gradient(rgba(34,211,238,.045) 1px,transparent 1px),linear-gradient(90deg,rgba(34,211,238,.045) 1px,transparent 1px);background-size:28px 28px;mask-image:linear-gradient(to bottom,black,transparent)}.ov-canvas>*{position:relative}.ov-canvas-head{display:flex;justify-content:space-between;gap:1rem;padding-bottom:.85rem;border-bottom:1px solid rgba(148,163,184,.13);font-size:.65rem;letter-spacing:.12em;color:#7dd3fc}.ov-canvas-head strong{color:#86efac}.ov-canvas-flow{display:grid;grid-template-columns:.7fr 42px 1.05fr 42px 1fr;align-items:center;gap:.4rem;min-height:210px}.ov-input-stack{display:flex;flex-direction:column;gap:.42rem;align-items:center}.ov-input-stack i{display:block;width:76%;height:20px;border-radius:5px;border:1px solid rgba(148,163,184,.24);background:linear-gradient(90deg,rgba(34,211,238,.2),rgba(15,32,56,.7))}.ov-input-stack i:nth-child(2){width:92%}.ov-input-stack small,.ov-output-stack small{color:#91a4ba;font-size:.63rem}.ov-energy-line{height:2px;background:linear-gradient(90deg,transparent,#22d3ee,transparent);animation:ovFlow 3.2s ease-in-out infinite}.ov-model-core{display:flex;min-height:112px;flex-direction:column;align-items:center;justify-content:center;border-radius:50%;border:1px solid rgba(34,211,238,.45);background:radial-gradient(circle,rgba(34,211,238,.17),rgba(8,20,38,.95) 66%);text-align:center;box-shadow:inset 0 0 25px rgba(34,211,238,.08)}.ov-model-core small{color:#67e8f9;font-size:.57rem;letter-spacing:.15em}.ov-model-core strong{color:#f8fafc;font-size:1.16rem;margin:.15rem 0}.ov-model-core span{color:#91a4ba;font-size:.62rem}.ov-output-stack{display:flex;flex-direction:column;gap:.5rem}.ov-output-stack>div{position:relative;height:24px;border-radius:6px;background:rgba(148,163,184,.1);overflow:hidden}.ov-output-stack b{display:block;height:100%;opacity:.7}.ov-output-stack span{position:absolute;inset:4px auto auto 7px;color:#f8fafc;font-size:.64rem;font-weight:800}.ov-canvas-route{display:grid;grid-template-columns:repeat(4,1fr);gap:.4rem}.ov-canvas-route span{padding:.48rem .35rem;text-align:center;border-top:1px solid rgba(34,211,238,.28);color:#b8c8da;font-size:.62rem}.ov-canvas-caption{margin-top:.65rem;color:#6f859e;font-size:.62rem;text-align:center}.ov-energy-line:nth-of-type(2){animation-delay:.8s}@keyframes ovFlow{0%,100%{opacity:.35;transform:scaleX(.8)}50%{opacity:1;transform:scaleX(1)}}
        .ov-metric-ribbon{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));margin:1.15rem 0 2rem;border-top:1px solid rgba(34,211,238,.22);border-bottom:1px solid rgba(34,211,238,.22);background:rgba(7,18,36,.48)}.ov-metric-ribbon>div{padding:.8rem 1rem;border-right:1px solid rgba(148,163,184,.12)}.ov-metric-ribbon>div:last-child{border-right:0}.ov-metric-ribbon strong{display:block;color:#f8fafc;font-size:1.25rem;letter-spacing:-.03em}.ov-metric-ribbon span{display:block;color:#b8c8da;font-size:.7rem;margin:.12rem 0}.ov-metric-ribbon small{color:#22d3ee;font-size:.52rem;letter-spacing:.1em;font-weight:900}
        .ov-section-intro{display:flex;align-items:end;justify-content:space-between;gap:1rem;margin-bottom:.85rem}.ov-section-intro h2{color:#f8fafc;font-size:1.65rem;letter-spacing:-.035em;margin:.2rem 0}.ov-engine-section{padding:1rem 0 1.6rem}.ov-engine-flow{display:grid;grid-template-columns:1.25fr 18px 1fr 18px 1fr 18px 1.2fr 18px 1fr 18px 1.25fr;align-items:stretch}.ov-engine-flow>i{align-self:center;height:1px;background:linear-gradient(90deg,rgba(34,211,238,.2),#22d3ee,rgba(34,211,238,.2))}.ov-stage{padding:.72rem;border-top:2px solid rgba(34,211,238,.4);background:linear-gradient(180deg,rgba(15,32,56,.72),rgba(7,18,36,.42));min-height:92px}.ov-stage-model{border-top-color:#34d399;background:linear-gradient(180deg,rgba(52,211,153,.1),rgba(7,18,36,.52))}.ov-stage strong{display:block;color:#f8fafc;font-size:.78rem;line-height:1.35}.ov-stage span{display:block;color:#8297ae;font-size:.65rem;line-height:1.45;margin-top:.35rem}
        .ov-capabilities{padding:1.2rem 0}.ov-mosaic{display:grid;grid-template-columns:repeat(12,minmax(0,1fr));gap:.8rem}.ov-feature{position:relative;padding:1.15rem;border:1px solid rgba(148,163,184,.16);border-radius:16px;background:linear-gradient(145deg,rgba(17,37,62,.9),rgba(7,18,36,.92));overflow:hidden}.ov-feature small{color:#67e8f9;font-size:.57rem;letter-spacing:.12em;font-weight:900}.ov-feature h3{color:#f8fafc;font-size:1.2rem;margin:.35rem 0}.ov-feature p,.ov-feature>span{color:#91a4ba;font-size:.75rem;line-height:1.55}.ov-feature-main{grid-column:span 7;display:grid;grid-template-columns:1.1fr .9fr;gap:1rem;min-height:190px}.ov-feature-semantic{grid-column:span 5;min-height:190px}.ov-feature-lexical{grid-column:span 5}.ov-feature-experiments{grid-column:span 7;display:grid;grid-template-columns:.6fr 1.4fr;gap:1rem}.ov-mini-heatmap{padding:.65rem;border-radius:10px;background:rgba(5,14,28,.75)}.ov-mini-heatmap>div{display:grid;grid-template-columns:24px repeat(3,1fr);gap:.28rem;margin:.3rem 0}.ov-mini-heatmap i{height:22px;border-radius:3px;background:rgba(148,163,184,.09)}.ov-mini-heatmap .bear{background:rgba(251,113,133,.72)}.ov-mini-heatmap .neutral{background:rgba(34,211,238,.72)}.ov-mini-heatmap .bull{background:rgba(52,211,153,.72)}.ov-mini-heatmap span{color:#9fb2c8;font-size:.58rem}.ov-mini-heatmap footer{display:grid;grid-template-columns:repeat(3,1fr);padding-left:24px;text-align:center}.ov-constellation{position:relative;height:72px;margin-top:.4rem}.ov-constellation i{position:absolute;width:10px;height:10px;border-radius:50%;background:#22d3ee;box-shadow:0 0 0 5px rgba(34,211,238,.08)}.ov-constellation i:nth-child(1){left:8%;top:48%}.ov-constellation i:nth-child(2){left:31%;top:15%;background:#34d399}.ov-constellation i:nth-child(3){left:54%;top:55%;background:#fb7185}.ov-constellation i:nth-child(4){left:72%;top:20%}.ov-constellation i:nth-child(5){left:90%;top:62%;background:#34d399}.ov-constellation b{position:absolute;height:1px;background:rgba(34,211,238,.25);transform-origin:left}.ov-constellation b:nth-of-type(1){left:9%;top:53%;width:25%;transform:rotate(-24deg)}.ov-constellation b:nth-of-type(2){left:32%;top:22%;width:26%;transform:rotate(28deg)}.ov-constellation b:nth-of-type(3){left:55%;top:60%;width:38%;transform:rotate(-18deg)}.ov-feature-lexical mark{display:inline-flex;margin:.15rem;padding:.22rem .38rem;border-radius:6px;background:rgba(52,211,153,.14);color:#86efac}.ov-feature-lexical mark.negative{background:rgba(251,113,133,.14);color:#fda4af}.ov-feature-lexical mark.risk{background:rgba(251,191,36,.14);color:#fde68a}.ov-model-strip{display:grid;grid-template-columns:repeat(3,1fr);gap:.45rem}.ov-model-strip>div{padding:.65rem;border-left:2px solid #22d3ee;background:rgba(5,14,28,.55)}.ov-model-strip strong,.ov-model-strip span,.ov-model-strip small{display:block}.ov-model-strip strong{color:#f8fafc;font-size:.82rem}.ov-model-strip span{color:#b8c8da;font-size:.62rem;margin-top:.18rem}.ov-model-strip small{color:#67e8f9;font-size:.48rem;margin-top:.32rem}
        .ov-proof{padding:1.2rem 0}.ov-dossier{display:grid;grid-template-columns:1fr 1.2fr 1fr 1.05fr;border:1px solid rgba(148,163,184,.16);border-radius:16px;overflow:hidden}.ov-dossier article{padding:1rem;border-right:1px solid rgba(148,163,184,.13);background:rgba(7,18,36,.62)}.ov-dossier article:last-child{border-right:0}.ov-dossier small,.ov-dossier strong,.ov-dossier span{display:block}.ov-dossier small{color:#22d3ee;font-size:.58rem;letter-spacing:.15em;font-weight:900}.ov-dossier strong{color:#f8fafc;font-size:.82rem;margin:.48rem 0;overflow-wrap:anywhere}.ov-dossier span{color:#91a4ba;font-size:.65rem;line-height:1.45;margin:.2rem 0}
        .ov-story{display:grid;grid-template-columns:.72fr 1.28fr;gap:1rem;align-items:stretch;padding:1.2rem 0}.ov-story-copy{padding:1.1rem 1.1rem 1.1rem 0}.ov-story-copy h2{color:#f8fafc;font-size:1.65rem;margin:.35rem 0}.ov-story-copy p{color:#91a4ba;font-size:.78rem;line-height:1.6}.ov-story-flow{display:grid;grid-template-columns:repeat(3,1fr);gap:.45rem}.ov-story-flow>div{padding:.7rem;border-radius:10px;background:rgba(15,32,56,.72);border:1px solid rgba(148,163,184,.13)}.ov-story-flow small,.ov-story-flow strong{display:block}.ov-story-flow small{color:#67e8f9;font-size:.5rem;letter-spacing:.1em}.ov-story-flow strong{color:#dce8f6;font-size:.68rem;line-height:1.45;margin-top:.28rem}.ov-story-scores span{display:flex!important;align-items:center;gap:.3rem;color:#b8c8da;font-size:.58rem;margin-top:.25rem}.ov-story-scores i{display:inline-block;height:5px;min-width:12px;border-radius:5px}
        .ov-closing{display:grid;grid-template-columns:1.25fr .75fr;gap:1.5rem;align-items:end;margin:1.35rem 0 .6rem;padding:1.4rem 0 1rem;border-top:1px solid rgba(34,211,238,.26)}.ov-closing h2{max-width:780px;color:#f8fafc;font-size:1.8rem;line-height:1.12;letter-spacing:-.04em;margin:.35rem 0}.ov-closing p{color:#91a4ba;font-size:.78rem}.ov-closing aside{padding-left:1rem;border-left:2px solid rgba(34,211,238,.35)}.ov-closing aside strong,.ov-closing aside span{display:block}.ov-closing aside strong{color:#dce8f6;font-size:.72rem}.ov-closing aside span{color:#71869f;font-size:.62rem;line-height:1.5;margin-top:.35rem}
        @media(max-width:1050px){.ov-hero{grid-template-columns:1fr}.ov-canvas{min-height:330px}.ov-engine-flow{grid-template-columns:1fr 12px 1fr 12px 1fr}.ov-engine-flow>i:nth-of-type(n+3){display:none}.ov-engine-flow .ov-stage:nth-of-type(n+4){margin-top:.5rem}.ov-feature-main,.ov-feature-semantic,.ov-feature-lexical,.ov-feature-experiments{grid-column:span 12}.ov-dossier{grid-template-columns:repeat(2,1fr)}.ov-dossier article:nth-child(2){border-right:0}.ov-dossier article:nth-child(-n+2){border-bottom:1px solid rgba(148,163,184,.13)}}
        @media(max-width:650px){.ov-hero{padding-top:1rem;gap:1.2rem}.ov-hero-copy h1{font-size:2.35rem}.ov-canvas{min-height:390px}.ov-canvas-flow{grid-template-columns:1fr;gap:.35rem;padding-top:.6rem}.ov-input-stack{flex-direction:row}.ov-input-stack i{width:42px!important}.ov-energy-line{width:2px;height:18px;justify-self:center}.ov-model-core{width:130px;min-height:100px;justify-self:center}.ov-canvas-route{grid-template-columns:repeat(2,1fr)}.ov-metric-ribbon{grid-template-columns:1fr 1fr}.ov-metric-ribbon>div:nth-child(2n){border-right:0}.ov-metric-ribbon>div:last-child{grid-column:span 2}.ov-engine-flow{display:flex;flex-direction:column;gap:.35rem}.ov-engine-flow>i{display:block!important;width:1px;height:12px;align-self:center}.ov-stage{min-height:auto}.ov-feature-main,.ov-feature-experiments{grid-template-columns:1fr}.ov-model-strip{grid-template-columns:1fr}.ov-dossier{grid-template-columns:1fr}.ov-dossier article{border-right:0;border-bottom:1px solid rgba(148,163,184,.13)}.ov-story{grid-template-columns:1fr}.ov-story-flow{grid-template-columns:1fr 1fr}.ov-closing{grid-template-columns:1fr}.ov-closing aside{padding-left:.75rem}.ov-section-intro h2,.ov-story-copy h2{font-size:1.4rem}}
        @media(max-width:410px){.ov-hero-copy h1{font-size:2.05rem}.ov-metric-ribbon{grid-template-columns:1fr}.ov-metric-ribbon>div,.ov-metric-ribbon>div:nth-child(2n){border-right:0}.ov-metric-ribbon>div:last-child{grid-column:auto}.ov-story-flow{grid-template-columns:1fr}.ov-canvas-head{flex-direction:column}.ov-status span{font-size:.62rem}}
        @media(prefers-reduced-motion:reduce){.ov-energy-line{animation:none}}        .ar-eyebrow{color:#67e8f9;font-family:"IBM Plex Mono",monospace;font-size:.66rem;font-weight:900;letter-spacing:.16em}.ar-hero{position:relative;padding:1.7rem 0 1rem;border-bottom:1px solid rgba(34,211,238,.19)}.ar-hero:before{content:"";position:absolute;inset:-2rem -10vw 0 48%;background-image:linear-gradient(rgba(34,211,238,.05) 1px,transparent 1px),linear-gradient(90deg,rgba(34,211,238,.05) 1px,transparent 1px);background-size:25px 25px;mask-image:linear-gradient(90deg,transparent,black,transparent);pointer-events:none}.ar-hero h1{position:relative;max-width:900px;color:#f8fafc;font-size:clamp(2.5rem,5vw,4.7rem);line-height:1;letter-spacing:-.06em;margin:.55rem 0}.ar-hero p{position:relative;max-width:790px;color:#9eb1c7;font-size:.98rem;line-height:1.65}.ar-hero-status{position:relative;display:flex;flex-wrap:wrap;gap:.45rem;margin-top:1rem}.ar-hero-status span{padding:.34rem .55rem;border-left:2px solid #34d399;background:rgba(52,211,153,.07);color:#bbf7d0;font-size:.68rem;font-weight:750}.ar-hero-status .planned{border-left-color:#fbbf24;background:rgba(251,191,36,.07);color:#fde68a}
        .ar-command-bar{display:grid;grid-template-columns:repeat(7,minmax(0,1fr));border:1px solid rgba(148,163,184,.14);border-radius:12px;overflow:hidden;margin:.85rem 0 1.25rem;background:rgba(5,14,28,.72)}.ar-command-bar>div{padding:.65rem;border-right:1px solid rgba(148,163,184,.12)}.ar-command-bar>div:last-child{border-right:0}.ar-command-bar small,.ar-command-bar strong{display:block}.ar-command-bar small{color:#748aa2;font-family:"IBM Plex Mono",monospace;font-size:.48rem;letter-spacing:.08em}.ar-command-bar strong{font-size:.7rem;margin-top:.25rem}.ar-ok{color:#86efac}.ar-local{color:#67e8f9}.ar-planned{color:#fde68a}
        .ar-observatory{position:relative;padding:1.15rem;border:1px solid rgba(34,211,238,.22);border-radius:18px;background-color:rgba(5,14,28,.9);background-image:linear-gradient(rgba(34,211,238,.045) 1px,transparent 1px),linear-gradient(90deg,rgba(34,211,238,.045) 1px,transparent 1px);background-size:24px 24px;overflow:hidden}.ar-observatory-head,.ar-section-head{display:flex;align-items:end;justify-content:space-between;gap:1rem}.ar-observatory-head h2,.ar-section-head h2{color:#f8fafc;font-size:1.5rem;letter-spacing:-.035em;margin:.28rem 0}.ar-observatory-head>span,.ar-section-head>span{color:#7890aa;font-family:"IBM Plex Mono",monospace;font-size:.6rem}.ar-bridge-layout{display:grid;grid-template-columns:1fr .72fr 1fr;gap:.85rem;align-items:center;margin-top:1rem}.ar-lane{min-height:160px;padding:1rem;border:1px solid rgba(148,163,184,.15);background:rgba(15,32,56,.72)}.ar-lane>small{color:#67e8f9;font-family:"IBM Plex Mono",monospace;font-size:.57rem;font-weight:900}.ar-lane p{color:#b4c4d7;font-size:.72rem;line-height:1.55}.ar-experiment-tags,.ar-output-tags{display:flex;flex-wrap:wrap;gap:.35rem}.ar-experiment-tags span,.ar-output-tags span{padding:.28rem .4rem;border-radius:5px;background:rgba(34,211,238,.08);border:1px solid rgba(34,211,238,.18);color:#bae6fd;font-size:.58rem}.ar-artifact-bridge{position:relative;display:flex;min-height:210px;flex-direction:column;align-items:center;justify-content:center;padding:1rem;border:1px solid rgba(52,211,153,.4);background:radial-gradient(circle,rgba(52,211,153,.13),rgba(7,18,36,.94) 68%);text-align:center;clip-path:polygon(8% 0,92% 0,100% 12%,100% 88%,92% 100%,8% 100%,0 88%,0 12%)}.ar-artifact-bridge:before,.ar-artifact-bridge:after{content:"";position:absolute;top:50%;width:28px;height:1px;background:#22d3ee}.ar-artifact-bridge:before{right:100%}.ar-artifact-bridge:after{left:100%}.ar-artifact-bridge small{color:#86efac;font-family:"IBM Plex Mono",monospace;font-size:.55rem}.ar-artifact-bridge strong{color:#f8fafc;font-size:.85rem;margin:.5rem 0;overflow-wrap:anywhere}.ar-artifact-bridge span{color:#91a4ba;font-size:.62rem;margin:.12rem}.ar-artifact-bridge b{margin-top:.6rem;padding:.28rem .45rem;border-radius:5px;background:rgba(52,211,153,.12);color:#86efac;font-size:.56rem;letter-spacing:.08em}
        .ar-mode-surface{margin:1rem 0 1.25rem;padding:.9rem;border-top:1px solid rgba(34,211,238,.2);border-bottom:1px solid rgba(34,211,238,.2);background:linear-gradient(90deg,rgba(7,18,36,.35),rgba(15,32,56,.62),rgba(7,18,36,.35))}.ar-mode-head{display:flex;justify-content:space-between;align-items:center;gap:1rem;margin-bottom:.65rem}.ar-mode-head span{color:#22d3ee;font-family:"IBM Plex Mono",monospace;font-size:.56rem;letter-spacing:.1em}.ar-mode-head strong{color:#f8fafc;font-size:.95rem}.ar-flow{display:flex;align-items:stretch;gap:.25rem;flex-wrap:wrap}.ar-node{flex:1 1 105px;min-width:0;padding:.65rem;border-top:2px solid rgba(34,211,238,.42);background:rgba(7,18,36,.82);min-height:102px}.ar-node small,.ar-node strong,.ar-node span{display:block}.ar-node small{color:#67e8f9;font-family:"IBM Plex Mono",monospace;font-size:.46rem;line-height:1.3}.ar-node strong{color:#f8fafc;font-size:.7rem;line-height:1.35;margin:.35rem 0}.ar-node span{color:#8196ae;font-size:.58rem;line-height:1.45}.ar-connector{flex:0 0 11px;align-self:center;height:1px;background:linear-gradient(90deg,rgba(34,211,238,.2),#22d3ee);animation:arPacket 3.5s ease-in-out infinite}.ar-detail-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:.45rem;margin-top:.7rem}.ar-detail-grid>div{padding:.65rem;border-left:2px solid rgba(34,211,238,.35);background:rgba(15,32,56,.48)}.ar-detail-grid small,.ar-detail-grid strong,.ar-detail-grid span{display:block}.ar-detail-grid small{color:#67e8f9;font-size:.5rem}.ar-detail-grid strong{color:#f8fafc;font-size:.7rem;margin:.28rem 0}.ar-detail-grid span{color:#8fa4bb;font-size:.59rem;line-height:1.45}.ar-topology{padding:.75rem;margin:.45rem 0;border-left:3px solid #34d399;background:rgba(52,211,153,.04)}.ar-topology.ar-target{border-left-color:#fbbf24;background:rgba(251,191,36,.04)}.ar-topology>p{color:#869bb3;font-size:.62rem;line-height:1.5;margin:.65rem 0 0}
        .ar-telemetry,.ar-lineage,.ar-contract,.ar-stack{padding:1rem 0}.ar-telemetry-grid{display:grid;grid-template-columns:repeat(4,1fr);margin-top:.7rem;border:1px solid rgba(148,163,184,.14);background:rgba(5,14,28,.7)}.ar-telemetry-grid>div{padding:.65rem;border-right:1px solid rgba(148,163,184,.1);border-bottom:1px solid rgba(148,163,184,.1);min-height:72px}.ar-telemetry-grid small,.ar-telemetry-grid strong{display:block}.ar-telemetry-grid small{color:#5eead4;font-family:"IBM Plex Mono",monospace;font-size:.48rem}.ar-telemetry-grid strong{color:#dce8f6;font-family:"IBM Plex Mono",monospace;font-size:.69rem;margin-top:.3rem;overflow-wrap:anywhere}
        .ar-lineage-flow{display:grid;grid-template-columns:repeat(8,1fr);gap:.35rem;margin-top:.7rem}.ar-lineage-flow>div{position:relative;padding:.62rem;border-top:2px solid rgba(34,211,238,.35);background:rgba(15,32,56,.56);min-height:100px}.ar-lineage-flow>div:not(:last-child):after{content:"";position:absolute;left:100%;top:18px;width:.35rem;height:1px;background:#22d3ee}.ar-lineage-flow small,.ar-lineage-flow strong,.ar-lineage-flow span{display:block}.ar-lineage-flow small{color:#67e8f9;font-size:.46rem}.ar-lineage-flow strong{color:#f8fafc;font-size:.62rem;line-height:1.35;margin:.3rem 0}.ar-lineage-flow span{color:#8196ae;font-size:.52rem;line-height:1.4}
        .ar-pillar-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:.55rem;margin-top:.7rem}.ar-pillar-grid article{padding:.75rem;border-left:2px solid rgba(34,211,238,.4);background:rgba(7,18,36,.62)}.ar-pillar-grid small{display:block;color:#67e8f9;font-size:.52rem;font-weight:900;letter-spacing:.08em;margin-bottom:.45rem}.ar-pillar-grid span{display:block;color:#9eb1c7;font-size:.59rem;line-height:1.4;padding:.22rem 0;border-bottom:1px solid rgba(148,163,184,.08)}
        .ar-stack-layers{margin-top:.7rem;border-left:1px solid rgba(34,211,238,.35)}.ar-stack-layers>div{display:grid;grid-template-columns:.72fr 1.35fr .75fr;gap:.7rem;align-items:center;padding:.55rem .7rem;border-bottom:1px solid rgba(148,163,184,.1);background:linear-gradient(90deg,rgba(34,211,238,.045),transparent)}.ar-stack-layers small{color:#67e8f9;font-family:"IBM Plex Mono",monospace;font-size:.5rem}.ar-stack-layers strong{color:#dce8f6;font-size:.68rem}.ar-stack-layers span{justify-self:end;padding:.2rem .35rem;border-radius:4px;font-size:.52rem}.ar-layer-active{color:#86efac;background:rgba(52,211,153,.1)}.ar-layer-experiment{color:#c4b5fd;background:rgba(167,139,250,.1)}.ar-layer-local{color:#67e8f9;background:rgba(34,211,238,.1)}.ar-layer-target{color:#fde68a;background:rgba(251,191,36,.1)}
        .ar-boundary{display:grid;grid-template-columns:1fr 2px 1fr;gap:1rem;margin:1.1rem 0;padding:1rem 0;border-top:1px solid rgba(148,163,184,.14);border-bottom:1px solid rgba(148,163,184,.14)}.ar-boundary>i{background:linear-gradient(#34d399,#fb7185)}.ar-boundary small{display:block;color:#67e8f9;font-size:.56rem;font-weight:900;margin-bottom:.4rem}.ar-boundary>div:last-child small{color:#fda4af}.ar-boundary span{display:block;color:#9eb1c7;font-size:.62rem;padding:.18rem 0}.ar-ownership{display:grid;grid-template-columns:1fr 1fr;gap:1rem;align-items:end;padding:.8rem 0}.ar-ownership h2{color:#f8fafc;font-size:1.25rem;margin:.3rem 0}.ar-ownership p{color:#8fa4bb;font-size:.65rem;line-height:1.55}.ar-closing{display:grid;grid-template-columns:1.2fr .8fr;gap:1.4rem;align-items:end;padding:1.15rem 0 .65rem;border-top:1px solid rgba(34,211,238,.22)}.ar-closing h2{color:#f8fafc;font-size:1.7rem;line-height:1.1;letter-spacing:-.04em;margin:.35rem 0}.ar-closing p,.ar-closing aside span{color:#879cb4;font-size:.68rem;line-height:1.5}.ar-closing aside{padding-left:.8rem;border-left:2px solid rgba(34,211,238,.35)}.ar-closing aside strong,.ar-closing aside span{display:block}.ar-closing aside strong{color:#dce8f6;font-size:.68rem;margin-bottom:.35rem}@keyframes arPacket{0%,100%{opacity:.25}50%{opacity:1;box-shadow:0 0 8px rgba(34,211,238,.35)}}
        @media(max-width:1050px){.ar-command-bar{grid-template-columns:repeat(4,1fr)}.ar-bridge-layout{grid-template-columns:1fr}.ar-artifact-bridge{width:min(430px,100%);justify-self:center;min-height:165px}.ar-artifact-bridge:before,.ar-artifact-bridge:after{display:none}.ar-telemetry-grid{grid-template-columns:repeat(3,1fr)}.ar-lineage-flow{grid-template-columns:repeat(4,1fr)}.ar-pillar-grid{grid-template-columns:repeat(2,1fr)}}
        @media(max-width:650px){.ar-hero h1{font-size:2.35rem}.ar-command-bar{grid-template-columns:repeat(2,1fr)}.ar-observatory-head,.ar-section-head,.ar-mode-head{align-items:flex-start;flex-direction:column}.ar-flow{display:grid;grid-template-columns:1fr}.ar-connector{width:1px;height:12px;justify-self:center}.ar-node{min-height:auto}.ar-detail-grid,.ar-telemetry-grid,.ar-pillar-grid{grid-template-columns:1fr}.ar-lineage-flow{grid-template-columns:repeat(2,1fr)}.ar-stack-layers>div{grid-template-columns:1fr}.ar-stack-layers span{justify-self:start}.ar-boundary{grid-template-columns:1fr}.ar-boundary>i{height:2px}.ar-ownership,.ar-closing{grid-template-columns:1fr}.ar-closing aside{padding-left:.65rem}}
        @media(max-width:410px){.ar-command-bar,.ar-lineage-flow{grid-template-columns:1fr}.ar-hero h1{font-size:2.05rem}.ar-artifact-bridge{clip-path:none;border-radius:12px}.ar-telemetry-grid strong{font-size:.64rem}}
        @media(prefers-reduced-motion:reduce){.ar-connector{animation:none}}        /* Compact Architecture Observatory */
        .ar-hero{padding:1rem 0 .55rem!important}.ar-hero:before{opacity:.55}.ar-hero h1{font-size:clamp(3rem,4.6vw,3.75rem)!important;line-height:1.02!important;margin:.35rem 0!important}.ar-hero p{font-size:1rem!important;line-height:1.5!important;margin:.2rem 0!important}.ar-hero-status{display:none!important}
        .ar-command-bar{margin:.55rem 0 .8rem!important}.ar-command-bar>div{padding:.55rem .62rem!important}.ar-command-bar small{font-size:.68rem!important}.ar-command-bar strong{font-size:.86rem!important}
        .ar-observatory{padding:.9rem!important}.ar-observatory-head h2,.ar-section-head h2{font-size:1.75rem!important}.ar-observatory-head>span,.ar-section-head>span,.ar-mode-head span{font-size:.7rem!important}.ar-system-map{display:grid;grid-template-columns:1fr 34px 1.08fr 34px 1fr;align-items:center;margin-top:.7rem}.ar-system-side,.ar-central-artifact{padding:.85rem;background:rgba(7,18,36,.88);text-align:center}.ar-system-side small,.ar-system-side strong,.ar-system-side span,.ar-central-artifact small,.ar-central-artifact strong,.ar-central-artifact span,.ar-central-artifact b{display:block}.ar-system-side small,.ar-central-artifact small{font-size:.7rem;color:#67e8f9}.ar-system-side strong,.ar-central-artifact strong{font-size:.94rem;color:#f8fafc;margin:.3rem 0}.ar-system-side span,.ar-central-artifact span{font-size:.78rem;color:#9eb1c7;line-height:1.45}.ar-central-artifact{border:1px solid rgba(34,211,238,.65);box-shadow:0 0 28px rgba(34,211,238,.1)}.ar-central-artifact b,.ar-artifact-dock b{font-size:.7rem;color:#86efac;margin-top:.35rem}.ar-map-link{height:2px;background:#22d3ee;position:relative}.ar-map-link:after{content:"";position:absolute;right:0;top:-4px;border-left:7px solid #22d3ee;border-top:5px solid transparent;border-bottom:5px solid transparent}
        .ar-blueprint{margin:.65rem 0;padding:.85rem;border-radius:14px;background-color:rgba(5,14,28,.88);background-image:linear-gradient(rgba(34,211,238,.05) 1px,transparent 1px),linear-gradient(90deg,rgba(34,211,238,.05) 1px,transparent 1px);background-size:22px 22px;border:1px solid rgba(34,211,238,.18)}.ar-mode-head{margin-bottom:.55rem!important}.ar-mode-head small{font-size:.72rem!important;color:#67e8f9}.ar-mode-head strong{font-size:1.05rem!important}.ar-flow{flex-wrap:nowrap!important;gap:0!important}.ar-node{min-height:98px!important;padding:.6rem!important;border:1px solid rgba(34,211,238,.18)!important;border-top:2px solid #22d3ee!important}.ar-node small{font-size:.62rem!important}.ar-node strong{font-size:.83rem!important}.ar-node span{font-size:.72rem!important;color:#9eb1c7!important}.ar-connector{flex-basis:18px!important;position:relative!important;animation:arPacket 3s ease-in-out infinite}.ar-connector:after{content:"";position:absolute;right:0;top:-4px;border-left:6px solid #22d3ee;border-top:4px solid transparent;border-bottom:4px solid transparent}.ar-artifact-dock{width:48%;margin:.55rem auto 0;padding:.55rem;text-align:center;border:1px solid rgba(34,211,238,.45);background:rgba(7,18,36,.94)}.ar-artifact-dock small,.ar-artifact-dock strong,.ar-artifact-dock span{display:block}.ar-artifact-dock small{font-size:.68rem;color:#67e8f9}.ar-artifact-dock strong{font-size:.9rem;color:#fff}.ar-artifact-dock span{font-size:.72rem;color:#a5b8cc}.ar-branch{display:grid;grid-template-columns:1fr auto 1fr;align-items:center;margin:.35rem 0}.ar-branch>i{height:2px;background:#22d3ee}.ar-branch>div{display:flex;gap:.4rem;padding:.4rem}.ar-branch span{padding:.4rem .6rem;border:1px solid rgba(34,211,238,.4);background:#071224;color:#dff7ff;font-size:.72rem}.ar-training-proof{display:grid;grid-template-columns:1fr 1fr;gap:.5rem;margin-top:.5rem}.ar-training-proof>div{padding:.55rem;border-left:3px solid #22d3ee;background:rgba(15,32,56,.8)}.ar-training-proof small,.ar-training-proof strong{display:block;font-size:.72rem}.ar-training-proof strong{font-size:.86rem;color:#f8fafc}.ar-deployment-blueprint{display:grid;grid-template-columns:1fr 1fr;gap:.7rem}.ar-topology{margin:0!important}.ar-topology p{font-size:.75rem!important}.ar-inspector{display:grid;grid-template-columns:repeat(4,1fr);gap:.45rem;margin:.55rem 0 .8rem}.ar-inspector>div{padding:.65rem;border-left:3px solid #22d3ee;background:rgba(15,32,56,.72)}.ar-inspector small,.ar-inspector strong{display:block}.ar-inspector small{font-size:.68rem;color:#67e8f9}.ar-inspector strong{font-size:.78rem;color:#dce8f6;line-height:1.45}
        .ar-evidence{padding:.65rem 0}.ar-instrument{margin-top:.55rem;padding:.7rem;border-radius:14px;background:linear-gradient(135deg,rgba(7,18,36,.96),rgba(15,32,56,.8));border:1px solid rgba(34,211,238,.2)}.ar-instrument-primary{display:grid;grid-template-columns:.7fr .7fr 2fr;gap:.5rem}.ar-instrument-primary>div{padding:.55rem;border-bottom:2px solid #34d399}.ar-instrument-primary small,.ar-instrument-primary strong,.ar-gauges small,.ar-gauges strong{display:block}.ar-instrument-primary small,.ar-gauges small{font-size:.68rem;color:#67e8f9}.ar-instrument-primary strong{font-size:.9rem;color:#f8fafc}.ar-gauges{display:grid;grid-template-columns:repeat(4,1fr);gap:.5rem;margin-top:.55rem}.ar-gauges>div{padding:.65rem;background:rgba(34,211,238,.05);border-top:2px solid #22d3ee}.ar-gauges strong{font-size:1.15rem;color:#f8fafc;margin-top:.2rem}.ar-technical-row{display:flex;flex-wrap:wrap;gap:.45rem 1rem;margin-top:.55rem;font-size:.76rem;color:#9eb1c7}.ar-technical-row b{color:#e5f4ff}
        .ar-lineage{padding:.65rem 0}.ar-lineage-track{display:flex;align-items:stretch;margin-top:.5rem}.ar-lineage-track>div{flex:1;padding:.55rem;background:rgba(15,32,56,.65);border-top:2px solid #22d3ee}.ar-lineage-track>i{flex:0 0 15px;align-self:center;height:2px;background:#22d3ee}.ar-lineage-track small,.ar-lineage-track strong,.ar-lineage-track span{display:block}.ar-lineage-track small{font-size:.64rem;color:#67e8f9}.ar-lineage-track strong{font-size:.75rem;color:#f8fafc;line-height:1.35;margin:.25rem 0}.ar-lineage-track span{font-size:.62rem;color:#8fa4bb}
        .ar-guarantees{padding:.65rem 0;display:grid;grid-template-columns:1fr 1fr;gap:.25rem .75rem}.ar-guarantees .ar-section-head{grid-column:1/-1}.ar-guarantees>div:not(.ar-section-head){display:grid;grid-template-columns:32px .75fr 1.25fr;align-items:center;gap:.55rem;padding:.48rem;border-bottom:1px solid rgba(148,163,184,.12)}.ar-guarantees b{color:#22d3ee;font-size:.8rem}.ar-guarantees strong{color:#f8fafc;font-size:.82rem}.ar-guarantees span{color:#9eb1c7;font-size:.75rem;line-height:1.4}
        .ar-boundary{margin:.7rem 0!important;padding:.7rem!important;background:rgba(7,18,36,.55)}.ar-boundary small{font-size:.72rem!important}.ar-boundary span{font-size:.82rem!important;padding:.22rem 0!important}.ar-closing{padding:.75rem 0 .3rem!important}.ar-closing h2{font-size:1.65rem!important}.ar-closing p,.ar-closing aside{font-size:.8rem!important}
        @media(max-width:900px){.ar-system-map{grid-template-columns:1fr}.ar-map-link{width:2px;height:18px;justify-self:center}.ar-map-link:after{right:-4px;top:auto;bottom:0;border-left:5px solid transparent;border-right:5px solid transparent;border-top:7px solid #22d3ee}.ar-flow{display:grid!important;grid-template-columns:1fr!important}.ar-connector{width:2px!important;height:14px!important;justify-self:center}.ar-connector:after{right:-4px;top:auto;bottom:0;border-left:5px solid transparent;border-right:5px solid transparent;border-top:7px solid #22d3ee}.ar-deployment-blueprint,.ar-guarantees{grid-template-columns:1fr}.ar-guarantees .ar-section-head{grid-column:auto}.ar-lineage-track{display:grid;grid-template-columns:1fr}.ar-lineage-track>i{width:2px;height:12px;justify-self:center}.ar-artifact-dock{width:100%}}
        @media(max-width:650px){.ar-hero h1{font-size:2.45rem!important}.ar-command-bar{grid-template-columns:repeat(2,1fr)!important}.ar-inspector,.ar-gauges,.ar-instrument-primary{grid-template-columns:1fr 1fr}.ar-lineage-track{grid-template-columns:1fr}.ar-guarantees>div:not(.ar-section-head){grid-template-columns:30px 1fr}.ar-guarantees span{grid-column:2}.ar-closing{grid-template-columns:1fr!important}}
        @media(max-width:410px){.ar-command-bar,.ar-inspector,.ar-gauges,.ar-instrument-primary{grid-template-columns:1fr!important}.ar-hero h1{font-size:2.15rem!important}.ar-node strong{font-size:.88rem!important}}
        @media(prefers-reduced-motion:reduce){.ar-connector{animation:none!important}}</style>
        """,
        unsafe_allow_html=True,
    )


def _section_heading(eyebrow: str, title: str, copy: str = "") -> None:
    """Render a reusable escaped heading for a Streamlit page section."""

    section_description = f"<p>{_safe(copy)}</p>" if copy else ""
    section_html = (
        f'<div class="fs-section"><div class="fs-eye">{_safe(eyebrow)}</div>'
        f"<h2>{_safe(title)}</h2>{section_description}</div>"
    )
    st.markdown(section_html, unsafe_allow_html=True)


def _scope_note(text: str) -> None:
    """Render escaped explanatory text in the shared note style."""

    note_html = f'<div class="fs-note">{_safe(text)}</div>'
    st.markdown(note_html, unsafe_allow_html=True)


def _plotly_layout(height: int = 390, **overrides: Any) -> dict[str, Any]:
    """Return the shared Plotly layout with optional chart-specific settings."""

    chart_layout: dict[str, Any] = {
        "height": height,
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": "Inter, sans-serif", "color": "#dbeafe", "size": 13},
        "margin": {"l": 70, "r": 34, "t": 55, "b": 55},
        "hoverlabel": {"bgcolor": "#0f2038", "font_color": "#f8fafc"},
        "showlegend": False,
    }
    chart_layout.update(overrides)
    return chart_layout


@st.cache_resource(show_spinner=False)
def _load_public_bert_runtime():
    """Cache the expensive Full BERT runtime across Streamlit script reruns.

    Real loading and local-versus-remote resolution remain delegated to the
    shared Full BERT module.
    """

    return load_bert_runtime()







def _initialize_sentiment_analyzer_state() -> None:
    """Initialize Analyze Article state without replacing values from reruns."""

    # Streamlit reruns this file after interactions, so existing user state must survive.
    state_defaults = {
        "an_url": "",
        "an_input_choice": "Article link",
        "an_manual_headline": "",
        "an_manual_body": "",
        "an_loaded_headline": "",
        "an_loaded_body": "",
        "an_source_type": "No article added",
        "an_source_url": "",
        "an_extraction_status": "Add an article link, paste an article, or use the sample.",
        "an_extraction_method": "Not started",
        "an_results_generated": False,
        "an_result_signature": "",
        "an_result_signal": None,
        "an_result_bert": None,
        "an_result_error": "",
        "an_selected_sentence": 0,
        "an_evidence_class": "",
        "an_token_network": None,
        "an_token_network_signature": "",
    }
    for state_key, default_value in state_defaults.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = default_value


def _clear_sentiment_results() -> None:
    """Remove stale analysis while preserving the user's current article input."""

    # Source callbacks invalidate derived results before Streamlit renders again.
    st.session_state.an_results_generated = False
    st.session_state.an_result_signature = ""
    st.session_state.an_result_signal = None
    st.session_state.an_result_bert = None
    st.session_state.an_result_error = ""
    st.session_state.an_selected_sentence = 0
    st.session_state.an_evidence_class = ""
    st.session_state.an_token_network = None
    st.session_state.an_token_network_signature = ""


def _on_sentiment_source_change() -> None:
    """Activate the selected source and clear results from the previous source."""

    source_choice = st.session_state.get("an_input_choice", "Article link")
    if source_choice == "Article link":
        _on_sentiment_url_change()
    elif source_choice == "Paste article":
        _on_sentiment_manual_change()
    else:
        st.session_state.an_loaded_headline = ""
        st.session_state.an_loaded_body = ""
        st.session_state.an_source_url = ""
        st.session_state.an_source_type = "No article added"
        st.session_state.an_extraction_method = "Not started"
        st.session_state.an_extraction_status = "Select 'Load presentation sample' to add the sample article."
        _clear_sentiment_results()

def _on_sentiment_url_change() -> None:
    """Treat a changed URL as a new, not-yet-extracted source."""

    st.session_state.an_loaded_headline = ""
    st.session_state.an_loaded_body = ""
    cleaned_url = _clean_text(st.session_state.get("an_url", ""))
    st.session_state.an_source_url = cleaned_url
    st.session_state.an_source_type = "Article link" if st.session_state.an_source_url else "No article added"
    st.session_state.an_extraction_method = "Not started"
    st.session_state.an_extraction_status = (
        "Article link added. Select 'Get article text' to load it."
        if st.session_state.an_source_url
        else "Add an article link, paste an article, or use the sample."
    )
    _clear_sentiment_results()


def _on_sentiment_manual_change() -> None:
    """Make edited manual text the active article and invalidate old results."""

    manual_headline = _clean_text(st.session_state.get("an_manual_headline", ""))
    manual_body = _clean_article_text(st.session_state.get("an_manual_body", ""))
    st.session_state.an_loaded_headline = manual_headline
    st.session_state.an_loaded_body = manual_body
    st.session_state.an_source_url = ""
    st.session_state.an_source_type = "Pasted article" if manual_headline or manual_body else "No article added"
    st.session_state.an_extraction_method = "Manual entry" if manual_headline or manual_body else "Not started"
    article_word_count = len(f"{manual_headline} {manual_body}".split())
    st.session_state.an_extraction_status = (
        "Pasted article is ready."
        if article_word_count >= 25
        else "Add at least 25 words before analyzing."
        if article_word_count
        else "Add an article link, paste an article, or use the sample."
    )
    _clear_sentiment_results()


def _load_sentiment_sample() -> None:
    """Load the built-in article and invalidate results from the prior source."""

    sample_text = _clean_text(_BASE_EXAMPLE)
    first_sentence, _, remaining_text = sample_text.partition(".")
    st.session_state.an_loaded_headline = first_sentence.strip()
    st.session_state.an_loaded_body = remaining_text.strip() or sample_text
    st.session_state.an_source_type = "Built-in sample"
    st.session_state.an_source_url = ""
    st.session_state.an_extraction_method = "Built-in sample"
    st.session_state.an_extraction_status = "Sample article is ready."
    _clear_sentiment_results()


def _navigate_to_analyzer(load_sample: bool = False) -> None:
    """Open Analyze Article and optionally prepare its built-in sample."""

    _initialize_sentiment_analyzer_state()
    if load_sample:
        _load_sentiment_sample()
    st.session_state.public_dashboard_page = "Analyze Article"
    if load_sample:
        st.session_state.an_input_choice = "Presentation sample"

def _sentiment_article_signature(headline: str, body: str, source: str) -> str:
    """Hash normalized article content and source for stale-result protection."""

    import hashlib

    normalized_article = f"{_clean_text(headline)}\n{_clean_text(body)}\n{_clean_text(source)}"
    return hashlib.sha256(normalized_article.encode("utf-8")).hexdigest()


def _navigate_public_page(page: str) -> None:
    """Navigate between the four existing public routes without side effects."""

    st.session_state.public_dashboard_page = page


def _render_sentiment_overview_page() -> None:
    """Render the Overview from saved experiment evidence and static product HTML."""

    _initialize_sentiment_analyzer_state()
    try:
        experiment_data = load_experiment_lab_data(PROJECT_ROOT)
    except ExperimentDataError as exc:
        st.error(f"Verified product metrics could not be loaded: {exc}")
        return
    current_metrics = experiment_data.current
    full_bert_historical = experiment_data.historical["Full BERT historical"]
    distilbert_metrics = experiment_data.historical["DistilBERT historical"]
    lora_metrics = experiment_data.historical["BERT-LoRA historical"]
    warm_sentence_latency = experiment_data.benchmark.get("warm_sentence_milliseconds")
    training_config = experiment_data.manifest.get("configuration", {})
    parameter_count = experiment_data.manifest.get("parameter_counts", {}).get("total_parameters")

    st.markdown(
        f'''
        <section class="ov-hero">
          <div class="ov-hero-copy">
            <div class="ov-eyebrow">FINANCIAL NLP · TRANSFORMER INTELLIGENCE</div>
            <h1>Financial news, transformed into decision-ready evidence</h1>
            <p>A fine-tuned Full BERT system that converts long-form financial reporting into article-level sentiment, sentence evidence and inspectable language signals.</p>
            <div class="ov-status"><span>Full BERT</span><span>Three sentiment classes</span><span>Sentence-level evidence</span><span>Local inference</span></div>
          </div>
          <div class="ov-canvas" aria-label="Transformer Intelligence Canvas">
            <div class="ov-canvas-head"><span>TRANSFORMER INTELLIGENCE CANVAS</span><strong>Local inference ready</strong></div>
            <div class="ov-canvas-flow">
              <div class="ov-input-stack"><i></i><i></i><i></i><small>Financial article</small></div>
              <div class="ov-energy-line"></div>
              <div class="ov-model-core"><small>FINE-TUNED</small><strong>Full BERT</strong><span>Sentence segmentation</span></div>
              <div class="ov-energy-line"></div>
              <div class="ov-output-stack"><div><b style="width:24%;background:#fb7185"></b><span>Bearish</span></div><div><b style="width:18%;background:#22d3ee"></b><span>Neutral</span></div><div><b style="width:58%;background:#34d399"></b><span>Bullish</span></div><small>Interface preview · class scores</small></div>
            </div>
            <div class="ov-canvas-route"><span>Sentence scores</span><span>Semantic map</span><span>Token landscape</span><span>Lexical cues</span></div>
            <div class="ov-canvas-caption">Financial article → sentence units → three-class scores → inspectable evidence</div>
          </div>
        </section>
        ''',
        unsafe_allow_html=True,
    )
    primary_action, secondary_action, _hero_spacer = st.columns([1.05, 1.05, 2.9], gap="small")
    with primary_action:
        st.button("Analyze an article", key="overview_analyze", type="primary", width="stretch", on_click=_navigate_public_page, args=("Analyze Article",))
    with secondary_action:
        st.button("Explore model results", key="overview_models", width="stretch", on_click=_navigate_public_page, args=("Model Results",))

    st.markdown(
        f'''
        <section class="ov-metric-ribbon">
          <div title="Current reproduced run"><strong>{current_metrics.accuracy * 100:.2f}%</strong><span>Current test accuracy</span><small>CURRENT RUN</small></div>
          <div title="Current reproduced run"><strong>{current_metrics.macro_f1:.4f}</strong><span>Current macro-F1</span><small>CURRENT RUN</small></div>
          <div title="Verified Financial PhraseBank dataset"><strong>3,448</strong><span>Deduplicated sentences</span><small>VERIFIED DATASET</small></div>
          <div title="Current runtime benchmark"><strong>{warm_sentence_latency:.2f} ms</strong><span>Warm GPU inference / sentence</span><small>RUNTIME BENCHMARK</small></div>
          <div title="Verified historical experiments"><strong>3</strong><span>Transformer experiments</span><small>VERIFIED EXPERIMENTS</small></div>
        </section>
        ''', unsafe_allow_html=True,
    )

    st.markdown(
        '''
        <section class="ov-engine-section">
          <div class="ov-section-intro"><div class="ov-eyebrow">SYSTEM FLOW</div><h2>Inside the intelligence engine</h2></div>
          <div class="ov-engine-flow">
            <div class="ov-stage ov-stage-wide"><strong>Article URL or pasted text</strong><span>URL, pasted article or deterministic sample</span></div>
            <i></i><div class="ov-stage"><strong>Clean extraction</strong><span>Readable content with boilerplate removed</span></div>
            <i></i><div class="ov-stage"><strong>Sentence segmentation</strong><span>Ordered units for batched inference</span></div>
            <i></i><div class="ov-stage ov-stage-model"><strong>Fine-tuned Full BERT</strong><span>Bearish, Neutral and Bullish sentence scores</span></div>
            <i></i><div class="ov-stage"><strong>Class-score aggregation</strong><span>Arithmetic mean across sentence scores</span></div>
            <i></i><div class="ov-stage ov-stage-wide"><strong>Evidence workspace</strong><span>Heatmaps, semantic projection, tokens and lexical cues</span></div>
          </div>
        </section>
        ''', unsafe_allow_html=True,
    )

    st.markdown(
        f'''
        <section class="ov-capabilities">
          <div class="ov-section-intro"><div class="ov-eyebrow">CAPABILITY SYSTEM</div><h2>Evidence at multiple levels</h2></div>
          <div class="ov-mosaic">
            <article class="ov-feature ov-feature-main">
              <div><small>PRIMARY MODEL VIEW</small><h3>Transformer sentiment intelligence</h3><p>Article-level class scores, sentence predictions, transparent mean aggregation and the strongest supporting evidence sentences.</p></div>
              <div class="ov-mini-heatmap"><small>INTERFACE PREVIEW</small><div><span>S1</span><i class="bear"></i><i></i><i></i></div><div><span>S2</span><i></i><i class="neutral"></i><i></i></div><div><span>S3</span><i></i><i></i><i class="bull"></i></div><footer><span>Bearish</span><span>Neutral</span><span>Bullish</span></footer></div>
            </article>
            <article class="ov-feature ov-feature-semantic"><small>REPRESENTATION VIEW · NOT CAUSAL EXPLANATION</small><h3>Semantic evidence</h3><p>Deterministic PCA sentence projection, contextual token similarity, WordPiece inspection and token landscape.</p><div class="ov-constellation"><i></i><i></i><i></i><i></i><i></i><b></b><b></b><b></b></div></article>
            <article class="ov-feature ov-feature-lexical"><small>SECONDARY RULE-BASED EVIDENCE</small><h3>Lexical and risk cues</h3><p><mark>positive phrase matches</mark> <mark class="negative">negative phrase matches</mark> <mark class="risk">risk-related language</mark></p><span>Contextual contribution analysis supplements—but does not determine—the Full BERT result.</span></article>
            <article class="ov-feature ov-feature-experiments">
              <div><small>VERIFIED EXPERIMENTATION</small><h3>Verified Transformer experimentation</h3></div>
              <div class="ov-model-strip"><div><strong>Full BERT</strong><span>{current_metrics.accuracy * 100:.2f}% current accuracy</span><span>{current_metrics.macro_f1:.4f} current macro-F1</span><small>CURRENT · PERFORMANCE CHAMPION</small></div><div><strong>DistilBERT</strong><span>{distilbert_metrics.accuracy * 100:.2f}% historical accuracy</span><span>{distilbert_metrics.macro_f1:.4f} historical macro-F1</span><small>HISTORICAL · EFFICIENT ALTERNATIVE</small></div><div><strong>BERT-LoRA</strong><span>{lora_metrics.accuracy * 100:.2f}% historical accuracy</span><span>{lora_metrics.macro_f1:.4f} historical macro-F1</span><small>HISTORICAL · PARAMETER-EFFICIENT</small></div></div>
            </article>
          </div>
        </section>
        ''', unsafe_allow_html=True,
    )

    st.markdown(
        f'''
        <section class="ov-proof">
          <div class="ov-section-intro"><div class="ov-eyebrow">TECHNICAL DOSSIER</div><h2>Proof, not promise</h2></div>
          <div class="ov-dossier">
            <article><small>DATA</small><strong>Financial PhraseBank</strong><span>sentences_75agree</span><span>3,453 original records</span><span>3,448 after deduplication</span><span>5 duplicates removed</span></article>
            <article><small>MODEL</small><strong>google-bert/bert-base-uncased</strong><span>{parameter_count / 1_000_000:.1f}M parameters</span><span>Three sentiment classes</span><span>Maximum sequence length: {training_config.get("max_length", 128)}</span></article>
            <article><small>EVALUATION</small><strong>{current_metrics.accuracy * 100:.2f}% reproduced accuracy</strong><span>{current_metrics.macro_f1:.4f} reproduced macro-F1</span><span>Fixed held-out test split</span><span>Interactive error analysis</span></article>
            <article><small>ENGINEERING</small><strong>Cached local inference</strong><span>Batched sentence processing</span><span>GPU and CPU support</span><span>Deterministic semantic visualisations</span><span>Tested state and extraction workflows</span></article>
          </div>
        </section>
        ''', unsafe_allow_html=True,
    )

    st.markdown(
        '''
        <section class="ov-story">
          <div class="ov-story-copy"><div class="ov-eyebrow">ILLUSTRATIVE INTERFACE PREVIEW</div><h2>From article to evidence</h2><p>The deterministic presentation article shows how one source becomes an inspectable evidence workspace. Values below preview the interface and are not a live prediction.</p></div>
          <div class="ov-story-flow">
            <div><small>ARTICLE HEADLINE</small><strong>Nvidia raises outlook as data-center demand reaches a record</strong></div>
            <div><small>ARTICLE STRUCTURE</small><strong>9 ordered sentences</strong></div>
            <div class="ov-story-scores"><small>THREE-CLASS VIEW</small><span><i style="width:24%;background:#fb7185"></i>Bearish</span><span><i style="width:18%;background:#22d3ee"></i>Neutral</span><span><i style="width:58%;background:#34d399"></i>Bullish</span></div>
            <div><small>DIRECTIONAL SENTENCE</small><strong>Management raised its full-year guidance.</strong></div>
            <div><small>SEMANTIC / TOKEN PREVIEW</small><strong>revenue · demand · guidance · margins</strong></div>
            <div><small>LEXICAL RISK CUE</small><strong>export controls</strong></div>
          </div>
        </section>
        ''', unsafe_allow_html=True,
    )
    story_action, _story_spacer = st.columns([1.15, 4.85])
    with story_action:
        st.button("Open this analysis", key="overview_open_sample", width="stretch", on_click=_navigate_to_analyzer, kwargs={"load_sample": True})

    st.markdown(
        '''
        <section class="ov-closing">
          <div><div class="ov-eyebrow">FINANCIAL-LANGUAGE INTELLIGENCE</div><h2>Move from headline-level intuition to inspectable financial-language evidence</h2><p>Analyze an article, examine sentence-level Transformer output, and review the verified experiments behind the system.</p></div>
          <aside><strong>Designed and developed by Ruturaj Mokashi</strong><span>This application analyses financial-news language. It does not predict stock prices, investment returns or future market outcomes.</span></aside>
        </section>
        ''', unsafe_allow_html=True,
    )
    analyze_action, models_action, architecture_action, _closing_spacer = st.columns([1, 1, 1, 1.7], gap="small")
    with analyze_action:
        st.button("Analyze financial news", key="overview_close_analyze", type="primary", width="stretch", on_click=_navigate_public_page, args=("Analyze Article",))
    with models_action:
        st.button("Explore model performance", key="overview_close_models", width="stretch", on_click=_navigate_public_page, args=("Model Results",))
    with architecture_action:
        st.button("View system architecture", key="overview_close_arch", width="stretch", on_click=_navigate_public_page, args=("About / Architecture",))

def _article_domain(url: str) -> str:
    """Return a display-friendly domain from an article URL."""

    if not url:
        return ""
    try:
        from urllib.parse import urlparse

        article_domain = urlparse(url).netloc.removeprefix("www.")
        return article_domain
    except Exception:
        return ""


def _supporting_sentence(text: str, phrase: str) -> str:
    """Return the first submitted sentence containing a complete phrase match."""

    candidate_sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    for candidate_sentence in candidate_sentences:
        if _term_count(candidate_sentence, phrase):
            return candidate_sentence.strip()[:300]
    return "Matched in the submitted article."


def _contextual_phrase(text: str, term: str) -> str:
    """Return a concise two-to-five-word phrase around a configured match."""

    supporting_sentence = _supporting_sentence(text, term)
    term_pattern = rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])"
    term_match = re.search(term_pattern, supporting_sentence, re.IGNORECASE)
    if not term_match:
        return term

    stop_words = {
        "a", "an", "and", "as", "at", "but", "by", "for", "from",
        "in", "is", "of", "on", "or", "the", "to", "was", "were", "with",
    }
    word_pattern = r"[A-Za-z0-9][A-Za-z0-9'’.-]*"
    words_before = [
        word
        for word in re.findall(word_pattern, supporting_sentence[:term_match.start()])
        if word.lower() not in stop_words
    ]
    words_after = [
        word
        for word in re.findall(word_pattern, supporting_sentence[term_match.end():])
        if word.lower() not in stop_words
    ]
    matched_words = re.findall(word_pattern, term_match.group(0))

    selected_words = words_before[-1:] + matched_words + words_after[:1]
    if len(selected_words) < 2 and len(words_after) > 1:
        selected_words += words_after[1:2]
    if len(selected_words) < 2 and len(words_before) > 1:
        selected_words = words_before[-2:-1] + selected_words

    evidence_phrase = " ".join(selected_words[:5]).strip(" ,.;:-")
    return evidence_phrase or term


def _evidence_contributions(
    text: str,
    signal: ArticleSignal,
) -> list[dict[str, Any]]:
    """Prepare display rows from the unchanged lexical sentiment weights."""

    weighted_terms = [
        *[(term, "Positive", 1.2 / 6.0) for term in signal.positive_hits],
        *[(term, "Negative", -(1.05 / 6.0)) for term in signal.negative_hits],
    ]
    evidence_rows = []

    for term, category, contribution_score in weighted_terms:
        evidence_rows.append(
            {
                "term": term,
                "phrase": _contextual_phrase(text, term),
                "category": category,
                "occurrences": _term_count(text, term),
                "individual_weight": contribution_score,
                "contribution": contribution_score,
                "context": _supporting_sentence(text, term),
            }
        )

    return sorted(
        evidence_rows,
        key=lambda evidence: (-abs(evidence["contribution"]), evidence["term"]),
    )[:10]


def _evidence_cloud_items(
    text: str,
    signal: ArticleSignal,
) -> list[dict[str, Any]]:
    """Prepare sentiment and risk phrases for the lexical evidence cloud."""

    evidence_items = _evidence_contributions(text, signal)
    for term in signal.risk_hits:
        evidence_items.append(
            {
                "term": term,
                "phrase": _contextual_phrase(text, term),
                "category": "Risk-related",
                "occurrences": _term_count(text, term),
                "individual_weight": 0.13,
                "contribution": 0.13,
                "context": _supporting_sentence(text, term),
            }
        )

    return sorted(
        evidence_items,
        key=lambda evidence: (
            -abs(evidence["contribution"]),
            -evidence["occurrences"],
            evidence["term"],
        ),
    )


def _risk_theme_rows(text: str, signal: ArticleSignal) -> list[dict[str, Any]]:
    """Group matched risk terms into the themes displayed by the risk chart."""

    risk_theme_terms = {
        "Uncertainty": {"uncertainty", "macro"},
        "Regulation and policy": {"regulation", "policy", "export", "controls", "china"},
        "Competition": {"competition"},
        "Volatility": {"volatility"},
        "Operational pressure": {"risk", "supply", "constraints", "inventory", "margin"},
    }
    risk_theme_rows = []

    for risk_theme, theme_terms in risk_theme_terms.items():
        matched_terms = sorted(set(signal.risk_hits).intersection(theme_terms))
        if not matched_terms:
            continue

        contextual_phrases = [
            _contextual_phrase(text, term)
            for term in matched_terms
        ]
        risk_theme_rows.append(
            {
                "theme": risk_theme,
                "contextual_label": contextual_phrases[0],
                "terms": ", ".join(matched_terms),
                "count": sum(_term_count(text, term) for term in matched_terms),
                "contribution": len(matched_terms) * 0.13,
            }
        )

    return sorted(
        risk_theme_rows,
        key=lambda risk_theme: (risk_theme["contribution"], risk_theme["count"]),
        reverse=True,
    )




def _render_sentiment_spectrum(signal: ArticleSignal) -> None:
    """Render the rule-based article sentiment score on its lexical spectrum."""

    import plotly.graph_objects as go

    sentiment_score = signal.sentiment_score
    sentiment_label = _sentiment_interpretation(sentiment_score)
    sentiment_color = (
        "#34d399"
        if sentiment_label == "Bullish"
        else "#fb7185"
        if sentiment_label == "Bearish"
        else "#22d3ee"
    )
    chart_figure = go.Figure()
    sentiment_regions = [
        (-1, -.15, "rgba(251,113,133,.18)"),
        (-.15, .15, "rgba(34,211,238,.12)"),
        (.15, 1, "rgba(52,211,153,.18)"),
    ]
    for region_start, region_end, fill_color in sentiment_regions:
        chart_figure.add_vrect(
            x0=region_start,
            x1=region_end,
            fillcolor=fill_color,
            line_width=0,
            layer="below",
        )

    chart_figure.add_trace(
        go.Scatter(
            x=[sentiment_score],
            y=[0],
            mode="markers",
            marker={
                "size": 24,
                "color": sentiment_color,
                "line": {"width": 3, "color": "#f8fafc"},
            },
            customdata=[[sentiment_label]],
            hovertemplate="<b>%{customdata[0]}</b><br>Sentiment score: %{x:+.3f}<extra></extra>",
        )
    )
    chart_figure.add_annotation(
        x=sentiment_score,
        y=.26,
        text=f"{sentiment_label} · {sentiment_score:+.2f}",
        showarrow=True,
        arrowcolor=sentiment_color,
        font={"color": "#f8fafc", "size": 14},
        bgcolor="rgba(15,32,56,.92)",
        bordercolor=sentiment_color,
        borderpad=7,
    )
    chart_figure.update_layout(
        **_plotly_layout(height=260, margin={"l": 38, "r": 38, "t": 45, "b": 55}),
        xaxis={
            "range": [-1.04, 1.04],
            "tickvals": [-1, -.15, 0, .15, 1],
            "ticktext": ["Bearish", "Bearish edge", "Neutral", "Bullish edge", "Bullish"],
            "gridcolor": "rgba(148,163,184,.12)",
            "zerolinecolor": "rgba(226,232,240,.45)",
            "fixedrange": True,
        },
        yaxis={"visible": False, "range": [-.35, .55], "fixedrange": True},
    )
    st.plotly_chart(chart_figure, width="stretch", config=_plotly_config())


def _render_evidence_cloud(text: str, signal: ArticleSignal) -> None:
    """Render matched lexical phrases as deterministic HTML evidence chips."""

    evidence_items = _evidence_cloud_items(text, signal)
    if not evidence_items:
        st.markdown('<div class="fs-card fs-empty" style="min-height:180px"><div class="fs-empty-icon">≋</div><strong>No configured evidence was detected.</strong><p>The submitted article contains no matched positive, negative, or risk terms.</p></div>', unsafe_allow_html=True)
        return

    evidence_colors = {
        "Positive": "#34d399",
        "Negative": "#fb7185",
        "Risk-related": "#fbbf24",
    }
    evidence_chips = []
    for evidence_item in evidence_items:
        font_size = min(
            30,
            16
            + int(abs(evidence_item["contribution"]) * 42)
            + min(evidence_item["occurrences"], 3),
        )
        tooltip = (
            f'Matched term: {evidence_item["term"]} | Category: {evidence_item["category"]} | '
            f'Contribution: {evidence_item["contribution"]:+.3f} | Occurrences: {evidence_item["occurrences"]} | '
            f'Context: {evidence_item["context"]}'
        )
        evidence_chips.append(
            f'<span class="fs-cloud-term" style="color:{evidence_colors[evidence_item["category"]]};font-size:{font_size}px" '
            f'title="{_safe(tooltip)}">{_safe(evidence_item["phrase"])}</span>'
        )

    container_class = (
        "fs-card fs-cloud"
        if len(evidence_items) >= 3
        else "fs-card fs-token-row"
    )
    cloud_html = f'<div class="{container_class}">' + "".join(evidence_chips) + "</div>"
    st.markdown(cloud_html, unsafe_allow_html=True)
    if len(evidence_items) < 3:
        st.caption("Only the genuine configured phrases detected in this article are shown.")


def _render_evidence_chart(text: str, signal: ArticleSignal) -> None:
    """Render lexical sentiment contributions as a horizontal bar chart."""

    evidence_rows = _evidence_contributions(text, signal)
    if not evidence_rows:
        st.markdown('<div class="fs-card fs-empty" style="min-height:190px"><div class="fs-empty-icon">≋</div><strong>No configured sentiment phrases were detected.</strong></div>', unsafe_allow_html=True)
        return

    import plotly.graph_objects as go

    displayed_evidence = list(reversed(evidence_rows))
    chart_figure = go.Figure(
        go.Bar(
            x=[evidence["contribution"] for evidence in displayed_evidence],
            y=[evidence["phrase"] for evidence in displayed_evidence],
            orientation="h",
            marker_color=[
                "#34d399" if evidence["contribution"] > 0 else "#fb7185"
                for evidence in displayed_evidence
            ],
            customdata=[
                [
                    evidence["term"],
                    evidence["category"],
                    evidence["occurrences"],
                    evidence["individual_weight"],
                    evidence["context"],
                ]
                for evidence in displayed_evidence
            ],
            hovertemplate=(
                "<b>%{y}</b><br>Configured term: %{customdata[0]}<br>Category: %{customdata[1]}<br>"
                "Occurrences: %{customdata[2]}<br>Individual weight: %{customdata[3]:+.3f}<br>"
                "Total contribution: %{x:+.3f}<br>Supporting sentence: %{customdata[4]}<extra></extra>"
            ),
        )
    )
    chart_figure.add_vline(x=0, line_color="rgba(226,232,240,.6)", line_width=2)
    chart_figure.update_layout(
        **_plotly_layout(
            height=max(320, 58 * len(displayed_evidence) + 110),
            margin={"l": 210, "r": 45, "t": 35, "b": 58},
        ),
        xaxis={
            "title": "Contribution to rule-based article score",
            "gridcolor": "rgba(148,163,184,.12)",
            "zeroline": False,
        },
        yaxis={"title": "", "automargin": True, "tickfont": {"size": 12}},
    )
    st.plotly_chart(chart_figure, width="stretch", config=_plotly_config())
    st.caption("Bars show how configured matched language contributed to the rule-based article score.")


def _render_risk_gauge(signal: ArticleSignal) -> None:
    """Render the heuristic lexical risk score as a compact bullet gauge."""

    import plotly.graph_objects as go

    risk_score = signal.risk_score * 100
    chart_figure = go.Figure()
    risk_regions = [
        (0, 33, "rgba(52,211,153,.24)"),
        (33, 66, "rgba(251,191,36,.24)"),
        (66, 100, "rgba(251,113,133,.24)"),
    ]
    for region_start, region_end, fill_color in risk_regions:
        chart_figure.add_shape(
            type="rect",
            x0=region_start,
            x1=region_end,
            y0=-.13,
            y1=.13,
            fillcolor=fill_color,
            line_width=0,
        )

    chart_figure.add_trace(
        go.Scatter(
            x=[risk_score],
            y=[0],
            mode="markers+text",
            text=[f"{risk_score:.0f}%"],
            textposition="top center",
            marker={
                "symbol": "line-ns",
                "size": 34,
                "color": "#f8fafc",
                "line": {"width": 4, "color": "#fbbf24"},
            },
            hovertemplate="Heuristic article-risk indicator: %{x:.1f}%<extra></extra>",
        )
    )
    chart_figure.update_layout(
        **_plotly_layout(height=190, margin={"l": 28, "r": 28, "t": 38, "b": 50}),
        xaxis={
            "range": [0, 100],
            "tickvals": [16.5, 49.5, 83],
            "ticktext": ["Low", "Moderate", "Elevated"],
            "fixedrange": True,
            "showgrid": False,
            "zeroline": False,
        },
        yaxis={"visible": False, "range": [-.28, .36], "fixedrange": True},
    )
    st.plotly_chart(chart_figure, width="stretch", config=_plotly_config())


def _render_risk_theme_chart(text: str, signal: ArticleSignal) -> None:
    """Render grouped lexical risk themes as a horizontal bar chart."""

    risk_theme_rows = _risk_theme_rows(text, signal)
    if not risk_theme_rows:
        st.markdown('<div class="fs-card fs-empty" style="min-height:190px"><div class="fs-empty-icon">✓</div><strong>No configured risk themes were detected.</strong></div>', unsafe_allow_html=True)
        return

    import plotly.graph_objects as go

    displayed_themes = list(reversed(risk_theme_rows))
    chart_figure = go.Figure(
        go.Bar(
            x=[theme["contribution"] for theme in displayed_themes],
            y=[theme["contextual_label"] for theme in displayed_themes],
            orientation="h",
            marker_color="#fbbf24",
            customdata=[
                [theme["theme"], theme["terms"], theme["count"]]
                for theme in displayed_themes
            ],
            hovertemplate="<b>%{y}</b><br>Theme: %{customdata[0]}<br>Matched terms: %{customdata[1]}<br>Occurrences: %{customdata[2]}<br>Risk-language contribution: %{x:.2f}<extra></extra>",
        )
    )
    chart_figure.update_layout(
        **_plotly_layout(height=max(280, 52 * len(displayed_themes) + 105)),
        xaxis={
            "title": "Contribution to heuristic risk score",
            "gridcolor": "rgba(148,163,184,.12)",
        },
        yaxis={"title": "", "automargin": True},
    )
    st.plotly_chart(chart_figure, width="stretch", config=_plotly_config())

_BERT_COLORS = {"Bearish": "#fb7185", "Neutral": "#22d3ee", "Bullish": "#34d399"}


def _render_bert_score_stack(result: Any) -> None:
    """Render article-level Full BERT scores in fixed class order."""
    import plotly.graph_objects as go

    fig = go.Figure()
    hover = "<br>".join(f"{label}: {result.probabilities[label]:.2%}" for label in BERT_LABEL_ORDER)
    for label in BERT_LABEL_ORDER:
        value = result.probabilities[label] * 100
        fig.add_trace(go.Bar(
            x=[value], y=["Article"], name=label, orientation="h",
            marker={"color": _BERT_COLORS[label]}, text=[f"{label} {value:.1f}%"],
            textposition="inside", insidetextanchor="middle",
            customdata=[[hover]], hovertemplate="%{customdata[0]}<extra></extra>",
        ))
    fig.update_layout(
        **_plotly_layout(height=170, showlegend=False, margin={"l": 8, "r": 8, "t": 20, "b": 25}, barmode="stack"),
        xaxis={"range": [0, 100], "visible": False, "fixedrange": True},
        yaxis={"visible": False, "fixedrange": True}, uniformtext_minsize=11,
    )
    st.plotly_chart(fig, width="stretch", config=_plotly_config(), key="an_score_stack")


def _render_sentence_heatmap(result: Any) -> None:
    """Render sentence-level Full BERT scores as a class heatmap."""
    import plotly.graph_objects as go

    # Rows represent article sentences; columns follow the three model classes.
    heatmap_values = [[sentence.probabilities[label] * 100 for label in BERT_LABEL_ORDER] for sentence in result.sentences]
    hover_text = [[
        f"Sentence {index + 1}<br>{_safe(sentence.text)}<br>"
        + "<br>".join(f"{label}: {sentence.probabilities[label]:.2%}" for label in BERT_LABEL_ORDER)
        + f"<br>Predicted class: {sentence.label}"
        for _ in BERT_LABEL_ORDER
    ] for index, sentence in enumerate(result.sentences)]
    chart_figure = go.Figure(go.Heatmap(
        z=heatmap_values, x=list(BERT_LABEL_ORDER), y=[f"S{index + 1}" for index in range(len(heatmap_values))],
        customdata=hover_text, colorscale=[[0, "#0b1628"], [.45, "#155e75"], [1, "#e2e8f0"]],
        zmin=0, zmax=100, colorbar={"title": "Score %", "thickness": 12},
        hovertemplate="%{customdata}<extra></extra>",
    ))
    for row, sentence in enumerate(result.sentences):
        column = BERT_LABEL_ORDER.index(sentence.label)
        chart_figure.add_shape(type="rect", x0=column - .48, x1=column + .48, y0=row - .48, y1=row + .48,
                      line={"color": _BERT_COLORS[sentence.label], "width": 3}, fillcolor="rgba(0,0,0,0)")
    chart_figure.update_layout(
        **_plotly_layout(height=max(310, 42 * len(heatmap_values) + 120), margin={"l": 54, "r": 35, "t": 25, "b": 55}),
        xaxis={"side": "top", "fixedrange": True}, yaxis={"autorange": "reversed", "fixedrange": True},
    )
    st.plotly_chart(chart_figure, width="stretch", config=_plotly_config(), key="an_sentence_heatmap")


def _render_sentiment_journey(result: Any) -> None:
    """Render sentence sentiment in reading order and store chart selections."""
    import plotly.graph_objects as go

    chart_figure = go.Figure()
    for sentence_index, sentence in enumerate(result.sentences):
        opacity = .38 + .62 * sentence.confidence
        hover_text = (
            f"Sentence {sentence_index + 1}<br>{_safe(sentence.text)}<br>"
            + "<br>".join(f"{label}: {sentence.probabilities[label]:.2%}" for label in BERT_LABEL_ORDER)
        )
        chart_figure.add_trace(go.Bar(
            x=[1], y=["Journey"], orientation="h", name=f"Sentence {sentence_index + 1}",
            marker={"color": _BERT_COLORS[sentence.label], "opacity": opacity,
                    "line": {"color": "#f8fafc" if sentence_index == st.session_state.an_selected_sentence else "rgba(15,23,42,.55)", "width": 2}},
            customdata=[[hover_text, sentence_index]], hovertemplate="%{customdata[0]}<extra></extra>", showlegend=False,
        ))
    chart_figure.update_layout(
        **_plotly_layout(height=135, margin={"l": 8, "r": 8, "t": 10, "b": 20}, barmode="stack"),
        xaxis={"visible": False, "fixedrange": True}, yaxis={"visible": False, "fixedrange": True},
    )
    selection_event = st.plotly_chart(
        chart_figure, width="stretch", config=_plotly_config(), key="an_sentiment_journey",
        on_select="rerun", selection_mode="points",
    )
    try:
        selected_points = selection_event.selection.points
        if selected_points:
            st.session_state.an_selected_sentence = int(selected_points[0]["customdata"][1])
    except (AttributeError, KeyError, TypeError, ValueError):
        pass


def _score_bars(sentence: Any) -> str:
    """Return HTML bars for one sentence's Full BERT class scores."""
    bars = []
    for label in BERT_LABEL_ORDER:
        value = sentence.probabilities[label] * 100
        bars.append(
            f'<div class="fs-mini-score"><span>{label}</span><div><i style="width:{value:.2f}%;background:{_BERT_COLORS[label]}"></i></div><strong>{value:.1f}%</strong></div>'
        )
    return "".join(bars)


def _render_selected_sentence(result: Any, key_prefix: str) -> None:
    """Render the selected sentence and its exact Full BERT input tokens."""
    index = min(max(int(st.session_state.an_selected_sentence), 0), len(result.sentences) - 1)
    st.session_state.an_selected_sentence = index
    sentence = result.sentences[index]
    st.markdown(
        f'<div class="fs-card fs-selected"><div class="fs-preview-label">SELECTED SENTENCE · {index + 1}</div>'
        f'<p>{_safe(sentence.text)}</p><span class="fs-chip" style="border-color:{_BERT_COLORS[sentence.label]}">'
        f'{sentence.label} · {sentence.confidence:.1%}</span>{_score_bars(sentence)}</div>', unsafe_allow_html=True,
    )
    tokens = wordpiece_tokens(_load_public_bert_runtime(), sentence.text, include_special_tokens=True)
    chips = "".join(f'<span class="fs-token-chip">{_safe(token)}</span>' for token in tokens)
    st.markdown(f'<div class="fs-token-row">{chips}</div>', unsafe_allow_html=True)
    st.caption("These are the tokens supplied to Full BERT. They are not token-importance scores.")


def _render_evidence_cards(result: Any) -> None:
    """Rank sentence evidence by class and update the selected sentence."""
    selected_class = st.segmented_control(
        "Evidence class", list(BERT_LABEL_ORDER), key="an_evidence_class", width="stretch",
    ) or result.label
    ranked = sorted(
        enumerate(result.sentences), key=lambda pair: pair[1].probabilities[selected_class], reverse=True
    )[:4]
    for position, (index, sentence) in enumerate(ranked):
        with st.container(border=True):
            st.markdown(f"**Sentence {index + 1} · {sentence.label} · {sentence.probabilities[selected_class]:.1%} {selected_class} score**")
            st.write(sentence.text)
            st.markdown(_score_bars(sentence), unsafe_allow_html=True)
            if st.button("Select sentence", key=f"an_select_evidence_{selected_class}_{index}_{position}", width="stretch"):
                st.session_state.an_selected_sentence = index
                st.rerun()


def _render_semantic_map(result: Any) -> None:
    """Project Full BERT sentence embeddings into a selectable PCA map."""
    import plotly.graph_objects as go

    sentence_embeddings = [sentence.embedding for sentence in result.sentences]
    # PCA is a two-dimensional visual summary, not proof of cause or market impact.
    projected_coordinates = deterministic_pca(sentence_embeddings)
    chart_figure = go.Figure()
    for sentiment_label in BERT_LABEL_ORDER:
        sentence_indices = [index for index, sentence in enumerate(result.sentences) if sentence.label == sentiment_label]
        if not sentence_indices:
            continue
        chart_figure.add_trace(go.Scatter(
            x=[projected_coordinates[index][0] for index in sentence_indices], y=[projected_coordinates[index][1] for index in sentence_indices],
            mode="markers+text", name=sentiment_label, text=[f"S{index + 1}" for index in sentence_indices], textposition="top center",
            marker={"color": _BERT_COLORS[sentiment_label], "size": [10 + 18 * result.sentences[index].confidence for index in sentence_indices],
                    "line": {"color": "#f8fafc", "width": 1.5}},
            customdata=[[index, result.sentences[index].text,
                         result.sentences[index].probabilities["Bearish"],
                         result.sentences[index].probabilities["Neutral"],
                         result.sentences[index].probabilities["Bullish"]] for index in sentence_indices],
            hovertemplate="Sentence %{customdata[0]}<br>%{customdata[1]}<br>Bearish: %{customdata[2]:.2%}<br>Neutral: %{customdata[3]:.2%}<br>Bullish: %{customdata[4]:.2%}<extra></extra>",
        ))
    chart_figure.update_layout(**_plotly_layout(height=410, showlegend=True), xaxis={"title": "PCA component 1", "zeroline": False}, yaxis={"title": "PCA component 2", "zeroline": False})
    selection_event = st.plotly_chart(chart_figure, width="stretch", config=_plotly_config(), key="an_semantic_map", on_select="rerun", selection_mode="points")
    try:
        selected_points = selection_event.selection.points
        if selected_points:
            st.session_state.an_selected_sentence = int(selected_points[0]["customdata"][0])
    except (AttributeError, KeyError, TypeError, ValueError):
        pass
    st.caption("Nearby points have more similar Full BERT sentence representations in this two-dimensional projection. The projection is not a complete explanation of model reasoning.")


def _render_token_landscape(result: Any) -> None:
    """Render WordPiece frequency across the strongest evidence sentences."""
    evidence_sentences = sorted(result.sentences, key=lambda sentence: sentence.probabilities[result.label], reverse=True)[:6]
    bert_runtime = _load_public_bert_runtime()
    token_labels = [wordpiece_tokens(bert_runtime, sentence.text, include_special_tokens=False) for sentence in evidence_sentences]
    token_items = token_landscape(evidence_sentences, token_labels)
    _section_heading("TOKEN LANDSCAPE", "Token landscape — frequency within BERT evidence sentences")
    if not token_items:
        st.info("No meaningful article tokens are available for this view.")
        return
    maximum_frequency = max(token_item.count for token_item in token_items)
    token_chips = []
    for token_item in token_items:
        token_size = 13 + 15 * token_item.count / maximum_frequency
        class_distribution = ", ".join(f"{label}: {token_item.class_counts[label]}" for label in BERT_LABEL_ORDER)
        tooltip = f"Token: {token_item.token} | Occurrences: {token_item.count} | Sentence classes: {class_distribution} | Example: {token_item.example_sentence}"
        token_chips.append(f'<span class="fs-landscape-token" style="font-size:{token_size:.1f}px;color:{_BERT_COLORS[token_item.dominant_class]}" title="{_safe(tooltip)}">{_safe(token_item.token)} <small>{token_item.count}</small></span>')
    st.markdown('<div class="fs-card fs-token-landscape">' + "".join(token_chips) + "</div>", unsafe_allow_html=True)
    st.caption("Token size represents frequency within selected evidence sentences, not model importance.")


def _render_contextual_network(result: Any, signature: str) -> None:
    """Build and reuse a requested contextual-token similarity network."""
    import plotly.graph_objects as go

    index = min(max(int(st.session_state.an_selected_sentence), 0), len(result.sentences) - 1)
    sentence = result.sentences[index]
    token_network_signature = f"{signature}:{index}"
    # Contextual embedding extraction stays lazy because it is relatively expensive.
    if st.button("Build contextual token view", key="an_build_token_network", width="stretch"):
        with st.spinner("Building contextual token view..."):
            nodes, links = cosine_links(contextual_tokens(_load_public_bert_runtime(), sentence.text))
        st.session_state.an_token_network = (nodes, links)
        st.session_state.an_token_network_signature = token_network_signature
    if st.session_state.an_token_network_signature != token_network_signature or not st.session_state.an_token_network:
        st.caption("Build this view when needed. It uses the selected sentence's final-layer contextual token representations.")
        return
    nodes, links = st.session_state.an_token_network
    positions = circular_positions(len(nodes))
    fig = go.Figure()
    for source, target, similarity in links:
        fig.add_trace(go.Scatter(x=[positions[source][0], positions[target][0]], y=[positions[source][1], positions[target][1]], mode="lines", line={"color": f"rgba(34,211,238,{max(.12, similarity * .55):.3f})", "width": 1 + similarity * 2}, hoverinfo="skip", showlegend=False))
    neighbours = {index: [] for index in range(len(nodes))}
    for source, target, similarity in links:
        neighbours[source].append((nodes[target].text, similarity))
        neighbours[target].append((nodes[source].text, similarity))
    fig.add_trace(go.Scatter(
        x=[point[0] for point in positions], y=[point[1] for point in positions], mode="markers+text",
        text=[node.text for node in nodes], textposition="top center", marker={"size": 18, "color": "#22d3ee", "line": {"color": "#f8fafc", "width": 1}},
        customdata=[[node.text, "<br>".join(f"{name}: {score:.3f}" for name, score in sorted(neighbours[index], key=lambda pair: -pair[1])[:4])] for index, node in enumerate(nodes)],
        hovertemplate="<b>%{customdata[0]}</b><br>Strongest contextual neighbours:<br>%{customdata[1]}<extra></extra>", showlegend=False,
    ))
    fig.update_layout(**_plotly_layout(height=390, margin={"l": 20, "r": 20, "t": 25, "b": 20}), xaxis={"visible": False, "fixedrange": True}, yaxis={"visible": False, "fixedrange": True})
    st.plotly_chart(fig, width="stretch", config=_plotly_config(), key="an_token_network_chart")
    st.caption("Connections represent similarity between contextual token representations. They do not prove causal importance.")


def _lexical_markup_exact(text: str, signal: ArticleSignal, selected_filter: str) -> str:
    """Escape source text and add separate lexical-evidence highlights."""
    categories = {term: ("Positive", "fs-pos", 1.2 / 6.0) for term in signal.positive_hits}
    categories.update({term: ("Negative", "fs-neg", -(1.05 / 6.0)) for term in signal.negative_hits})
    categories.update({term: ("Risk-related", "fs-risk", .13) for term in signal.risk_hits})
    if not categories:
        return _safe(text)
    # Match longer phrases first so shorter overlapping terms cannot replace them.
    phrase_pattern = re.compile("|".join(rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])" for term in sorted(categories, key=len, reverse=True)), re.IGNORECASE)
    pieces, cursor = [], 0
    for match in phrase_pattern.finditer(text):
        pieces.append(_safe(text[cursor:match.start()]))
        configured = next(term for term in categories if term.lower() == match.group(0).lower())
        category, css_class, contribution = categories[configured]
        dim = selected_filter.startswith("Lexical ") and selected_filter != f"Lexical {category.replace('-related', 'Risk')}"
        pieces.append(f'<mark class="{css_class}{" fs-dim" if dim else ""}" title="Matched phrase: {_safe(match.group(0))} | Category: {category} | Configured term: {_safe(configured)} | Contribution: {contribution:+.3f}">{_safe(match.group(0))}</mark>')
        cursor = match.end()
    pieces.append(_safe(text[cursor:]))
    return "".join(pieces)


def _render_article_reader(result: Any, signal: ArticleSignal, headline: str, body: str, word_count: int) -> None:
    """Render Full BERT sentence styling with separate lexical highlights."""
    _section_heading("ARTICLE READER", "Highlighted Article Evidence Explorer", "Explore Full BERT sentence classifications and lexical phrase cues while preserving the exact article wording.")
    reader_filter = st.segmented_control(
        "Evidence filter", ["All evidence", "BERT Bearish", "BERT Neutral", "BERT Bullish", "Lexical Positive", "Lexical Negative", "Lexical Risk"],
        default="All evidence", key="an_reader_filter", width="stretch",
    ) or "All evidence"
    st.markdown('<div class="fs-sticky-legend"><strong>Full BERT sentence class</strong><span class="fs-bert-bearish">Bearish</span><span class="fs-bert-neutral">Neutral</span><span class="fs-bert-bullish">Bullish</span><strong>Lexical phrase evidence</strong><mark class="fs-pos">Positive</mark><mark class="fs-neg">Negative</mark><mark class="fs-risk">Risk-related</mark></div>', unsafe_allow_html=True)
    reader_text = headline + ("\n\n" if headline and body else "") + body
    article_segments = reader_segments(reader_text, result.sentences)
    rendered = []
    for segment in article_segments:
        content = _lexical_markup_exact(segment.text, signal, reader_filter)
        if segment.sentence_class:
            bert_filter = reader_filter.startswith("BERT ")
            dim = bert_filter and reader_filter != f"BERT {segment.sentence_class}"
            rendered.append(f'<span id="bert-sentence-{segment.sentence_index + 1}" class="fs-bert-sentence fs-bert-{segment.sentence_class.lower()}{" fs-dim" if dim else ""}" title="Sentence {segment.sentence_index + 1} | {segment.sentence_class} | Dominant class score: {segment.score:.2%}">{content}</span>')
        else:
            rendered.append(content)
    left, right = st.columns([1.7, 1], gap="large")
    with left:
        with st.expander("Expand full article", expanded=word_count <= 180):
            st.markdown('<article class="fs-reader fs-dual-reader">' + "".join(rendered).replace("\n\n", "</p><p>").replace("\n", "<br>") + "</article>", unsafe_allow_html=True)
    with right:
        st.markdown(f'<div class="fs-card fs-navigator"><div class="fs-preview-label">EVIDENCE NAVIGATOR</div><strong>{_safe(result.label)} article</strong><div class="fs-meta"><span class="fs-chip">{_safe(signal.company)} · {_safe(signal.ticker)}</span><span class="fs-chip">{word_count} words</span><span class="fs-chip">{len(result.sentences)} sentences</span></div></div>', unsafe_allow_html=True)
        for label in BERT_LABEL_ORDER:
            with st.expander(f"Top {label} sentences"):
                for sentence in result.strongest_by_label[label]:
                    index = result.sentences.index(sentence)
                    st.markdown(f"**S{index + 1} · {sentence.probabilities[label]:.1%}** — {sentence.text}")
        with st.expander("Lexical phrases"):
            st.write("Positive: " + (", ".join(signal.positive_hits) or "None detected"))
            st.write("Negative: " + (", ".join(signal.negative_hits) or "None detected"))
            st.write("Risk-related: " + (", ".join(signal.risk_hits) or "None detected"))

def _render_sentiment_analyze_page() -> None:
    """Coordinate Analyze Article inputs, inference, state, and evidence views."""

    _initialize_sentiment_analyzer_state()
    st.markdown(
        '<div class="fs-head"><div class="fs-eye">ARTICLE WORKSPACE</div>'
        '<h1 class="fs-title">Analyze financial-news language</h1>'
        '<p class="fs-copy">Choose one source, review the exact text, then create an article-level sentiment result.</p></div>',
        unsafe_allow_html=True,
    )
    input_panel, preview_panel = st.columns([1.02, .98], gap="large")
    with input_panel:
        source_choice = st.segmented_control(
            "Article source", ["Article link", "Paste article", "Presentation sample"],
            key="an_input_choice", width="stretch", on_change=_on_sentiment_source_change,
        )
        if source_choice == "Article link":
            st.text_input("Article link", key="an_url", placeholder="https://example.com/financial-news-article", on_change=_on_sentiment_url_change)
            if st.button("Get article text", key="an_get_article_text", width="stretch"):
                article_url = _clean_text(st.session_state.an_url)
                _clear_sentiment_results()
                st.session_state.an_loaded_headline = ""
                st.session_state.an_loaded_body = ""
                st.session_state.an_source_url = article_url
                st.session_state.an_source_type = "Article link" if article_url else "No article added"
                if not article_url:
                    st.session_state.an_extraction_status = "Add an article link first."
                    st.session_state.an_extraction_method = "Not started"
                else:
                    try:
                        with st.spinner("Loading article text..."):
                            extracted_headline, extracted_body, extraction_method = _extract_article_content(article_url)
                        st.session_state.an_loaded_headline = extracted_headline
                        st.session_state.an_loaded_body = extracted_body
                        st.session_state.an_extraction_method = extraction_method
                        st.session_state.an_extraction_status = "Article text was loaded successfully."
                    except Exception as extraction_error:
                        error_name = type(extraction_error).__name__.lower()
                        if "timeout" in error_name:
                            failure_reason = "The website took too long to respond."
                        elif "http" in error_name or "connection" in error_name:
                            failure_reason = "The website returned a network or access error."
                        elif "runtime" in error_name:
                            failure_reason = "Article extraction is unavailable in this runtime."
                        else:
                            failure_reason = "We could not find enough readable article text on this page."
                        st.session_state.an_extraction_method = "Automatic extraction failed"
                        st.session_state.an_extraction_status = f"{failure_reason} Paste the article text instead."
        elif source_choice == "Paste article":
            st.text_input("Article headline (optional)", key="an_manual_headline", on_change=_on_sentiment_manual_change)
            st.text_area("Article body", key="an_manual_body", height=205, placeholder="Paste at least 25 words of financial-news text.", on_change=_on_sentiment_manual_change)
        else:
            st.write("Use the built-in financial-news example.")
            st.button("Load presentation sample", key="an_load_sample", on_click=_load_sentiment_sample, width="stretch")

        article_headline = _clean_text(st.session_state.an_loaded_headline)
        article_body = _clean_text(st.session_state.an_loaded_body)
        article_text = _clean_text(f"{article_headline}. {article_body}")
        article_word_count = len(article_text.split())
        content_ready = article_word_count >= 25
        article_signature = _sentiment_article_signature(
            article_headline, article_body, st.session_state.an_source_type,
        )
        # A signature mismatch means the displayed results belong to older input.
        if st.session_state.an_results_generated and st.session_state.an_result_signature != article_signature:
            _clear_sentiment_results()

        st.markdown(
            f'<div class="fs-note">{_safe(st.session_state.an_extraction_status)} · {article_word_count} words</div>',
            unsafe_allow_html=True,
        )
        if article_text and not content_ready:
            st.caption("At least 25 words are required.")
        if st.button("Analyze with Full BERT", key="an_analyze_article", type="primary", width="stretch", disabled=not content_ready):
            article_signal = _score_article(article_text, st.session_state.an_source_type, headline_text=article_headline, body_text=article_body)
            try:
                # Load Full BERT only after the user requests analysis.
                with st.status("Preparing Full BERT", expanded=True) as model_status:
                    model_status.write("Retrieving model artifact")
                    bert_runtime = _load_public_bert_runtime()
                    model_status.write("Loading model")
                    bert_result = analyze_article_with_bert(bert_runtime, article_headline, article_body)
                    model_status.update(label="Full BERT ready", state="complete", expanded=False)
            except Exception as inference_error:
                st.session_state.an_result_error = f"The trained BERT model could not run: {inference_error}"
                st.session_state.an_result_bert = None
                st.session_state.an_results_generated = False
            else:
                st.session_state.an_result_signal = article_signal
                st.session_state.an_result_bert = bert_result
                st.session_state.an_result_error = ""
                st.session_state.an_result_signature = article_signature
                st.session_state.an_results_generated = True
        if st.session_state.an_result_error:
            st.error(st.session_state.an_result_error)

    with preview_panel:
        if not article_headline and not article_body:
            st.markdown(
                '<div class="fs-card fs-empty"><div class="fs-empty-icon">▤</div>'
                '<strong>Article preview</strong><p>Add a link, paste text, or load the sample. '
                'The exact submitted content will appear here.</p></div>',
                unsafe_allow_html=True,
            )
        else:
            ticker, company = _infer_company(article_headline, article_body)
            domain = _article_domain(st.session_state.an_source_url)
            article_preview = article_body[:480] + ("…" if len(article_body) > 480 else "")
            domain_chip = f'<span class="fs-chip">{_safe(domain)}</span>' if domain else ""
            st.markdown(
                f'<div class="fs-card"><div class="fs-preview-label">ARTICLE PREVIEW</div>'
                f'<h3 style="color:#f8fafc">{_safe(article_headline or "No separate headline supplied")}</h3>'
                f'<div class="fs-meta"><span class="fs-chip">{_safe(st.session_state.an_source_type)}</span>'
                f'{domain_chip}<span class="fs-chip">{_safe(company)} · {_safe(ticker)}</span>'
                f'<span class="fs-chip">{article_word_count} words</span></div>'
                f'<p class="fs-copy" style="font-size:.94rem">{_safe(article_preview)}</p></div>',
                unsafe_allow_html=True,
            )

    result_ready = (
        st.session_state.an_results_generated
        and st.session_state.an_result_signature == article_signature
        and st.session_state.an_result_signal is not None
        and st.session_state.an_result_bert is not None
    )
    if not result_ready:
        return

    article_signal = st.session_state.an_result_signal
    bert_result = st.session_state.an_result_bert
    if not st.session_state.an_evidence_class:
        st.session_state.an_evidence_class = bert_result.label
    st.markdown('<div class="fs-result-separator"><span>FULL BERT ANALYSIS</span></div>', unsafe_allow_html=True)
    summary_tab, evidence_tab, semantic_tab, lexical_tab, reader_tab = st.tabs([
        "Summary", "Sentence Evidence", "Semantic & Token View", "Lexical Cues", "Article Reader",
    ])

    with summary_tab:
        strength = "Strong" if bert_result.confidence >= .75 else "Moderate" if bert_result.confidence >= .55 else "Slight"
        entity = f"{article_signal.company} · {article_signal.ticker}" if article_signal.ticker else f"{article_signal.company} · Private company"
        st.markdown(
            f'<div class="fs-card fs-bert-hero"><div><div class="fs-eye">FULL BERT ARTICLE RESULT</div>'
            f'<h2>{_safe(bert_result.label)}</h2><p>{strength} model preference for the dominant class. '
            f'This describes article language and has no investment implication.</p></div>'
            f'<div class="fs-result-facts"><span>Dominant class score<strong>{bert_result.confidence:.2%}</strong></span>'
            f'<span>Company / ticker<strong>{_safe(entity)}</strong></span><span>Sentences<strong>{len(bert_result.sentences)}</strong></span>'
            f'<span>Inference time<strong>{bert_result.inference_seconds * 1000:.1f} ms</strong></span>'
            f'<span>Model<strong>Full BERT</strong></span><span>Device<strong>{_safe(bert_result.device.upper())}</strong></span></div></div>',
            unsafe_allow_html=True,
        )
        _render_bert_score_stack(bert_result)
        score_cards = "".join(
            f'<div class="fs-score-card" style="border-top-color:{_BERT_COLORS[label]}"><span>{label}</span><strong>{bert_result.probabilities[label]:.2%}</strong></div>'
            for label in BERT_LABEL_ORDER
        )
        st.markdown(f'<div class="fs-score-cards">{score_cards}</div>', unsafe_allow_html=True)
        st.caption("Model class scores are softmax outputs and are not guaranteed to be calibrated probabilities.")
        _scope_note("This application analyses financial-news text. It does not predict future stock prices or investment returns.")

    with evidence_tab:
        _section_heading("SENTENCE EVIDENCE", "Sentence × class heatmap", "Every analyzed sentence and its real Full BERT class scores.")
        _render_sentence_heatmap(bert_result)
        _section_heading("JOURNEY", "Article sentiment journey", "Each block follows the article's sentence order. Select a block or evidence card to inspect it.")
        _render_sentiment_journey(bert_result)
        cards, selected = st.columns([1.08, .92], gap="large")
        with cards:
            _section_heading("EVIDENCE", "Strongest class evidence")
            _render_evidence_cards(bert_result)
        with selected:
            _section_heading("SELECTED", "Selected sentence")
            _render_selected_sentence(bert_result, "evidence")

    with semantic_tab:
        _section_heading("SEMANTIC MAP", "Semantic sentence map", "A deterministic PCA projection of final-layer Full BERT sentence representations.")
        _render_semantic_map(bert_result)
        _render_token_landscape(bert_result)
        semantic_left, semantic_right = st.columns([1, 1], gap="large")
        with semantic_left:
            _section_heading("SELECTED", "Selected evidence sentence")
            _render_selected_sentence(bert_result, "semantic")
        with semantic_right:
            _section_heading("CONTEXT", "Contextual token similarity")
            _render_contextual_network(bert_result, article_signature)

    with lexical_tab:
        _section_heading("SECONDARY ANALYSIS", "Lexical Cues — Secondary Analysis", "These rule-based cues supplement the Full BERT result. They do not determine the primary article sentiment.")
        _section_heading("SPECTRUM", "Rule-based Bearish-to-Bullish spectrum")
        _render_sentiment_spectrum(article_signal)
        _section_heading("LEXICAL EVIDENCE", "Detected contextual phrases")
        _render_evidence_cloud(article_text, article_signal)
        _section_heading("CONTRIBUTIONS", "Contextual phrase contributions", "Positive evidence extends right; negative evidence extends left.")
        _render_evidence_chart(article_text, article_signal)
        risk_left, risk_right = st.columns(2, gap="large")
        with risk_left:
            _section_heading("RISK LANGUAGE", "Heuristic article-risk indicator")
            _render_risk_gauge(article_signal)
            st.caption("This indicator reflects risk-related language in the article. It is not a prediction of financial loss.")
        with risk_right:
            _section_heading("RISK THEMES", "Detected risk-language themes")
            _render_risk_theme_chart(article_text, article_signal)

    with reader_tab:
        _render_article_reader(bert_result, article_signal, article_headline, article_body, article_word_count)
        _scope_note("Full BERT colours describe sentence classes. Inline lexical highlights show only configured phrase matches. Neither is investment advice.")


def _model_badge(text: str, kind: str = "current") -> str:
    """Return the unchanged HTML badge used by Model Results cards."""
    return f'<span class="ml-badge ml-{kind}">{_safe(text)}</span>'


def _format_optional_metric(value: float | None, percent: bool = False) -> str:
    """Format a stored metric without recalculating its experiment value."""
    if value is None:
        return "Not recorded"
    return f"{value * 100:.2f}%" if percent else f"{value:.4f}"


def _render_experiment_leaderboard(data: Any) -> None:
    """Render current and historical experiments in verified rank order."""
    import pandas as pd

    experiment_strategies = {
        "Full BERT historical": ("Full fine-tuning", "Performance champion"),
        "DistilBERT historical": ("Full fine-tuning", "Runtime-efficient alternative"),
        "BERT-LoRA historical": ("Parameter-efficient LoRA", "Parameter-efficient experiment"),
    }
    comparison_rows = []
    for rank, model_metrics in enumerate(leaderboard(data), start=1):
        training_strategy, experiment_status = experiment_strategies[model_metrics.name]
        comparison_rows.append({
            "Rank": rank, "Model": model_metrics.name.replace(" historical", ""),
            "Accuracy": f"{model_metrics.accuracy * 100:.2f}%", "Macro-F1": f"{model_metrics.macro_f1:.4f}",
            "Macro precision": _format_optional_metric(model_metrics.macro_precision),
            "Macro recall": _format_optional_metric(model_metrics.macro_recall),
            "Tuning strategy": training_strategy, "Experiment status": experiment_status,
        })
    st.dataframe(pd.DataFrame(comparison_rows), hide_index=True, width="stretch", height=178)


def _render_experiment_comparison(data: Any) -> None:
    """Compare stored accuracy and macro-F1 values across model runs."""
    import plotly.graph_objects as go

    experiment_runs = [data.current, *data.historical.values()]
    model_names = [model_result.name for model_result in experiment_runs]
    model_colors = ["#22d3ee", "#818cf8", "#38bdf8", "#a78bfa"]
    chart_figure = go.Figure()
    # Accuracy counts all correct predictions; macro-F1 weights each class equally.
    for metric_name, field, shade in [("Accuracy", "accuracy", 1.0), ("Macro-F1", "macro_f1", .65)]:
        chart_figure.add_trace(go.Bar(
            name=metric_name, y=model_names, x=[getattr(model_result, field) * 100 for model_result in experiment_runs], orientation="h",
            marker_color=model_colors, marker_opacity=shade,
            text=[f"{getattr(model_result, field) * 100:.2f}%" for model_result in experiment_runs], textposition="outside",
            customdata=[[model_result.source, "Current reproduced run" if model_result is data.current else "Verified historical experiment"] for model_result in experiment_runs],
            hovertemplate="<b>%{y}</b><br>" + metric_name + ": %{x:.4f}%<br>%{customdata[1]}<br>Source: %{customdata[0]}<extra></extra>",
        ))
    chart_figure.update_layout(
        **_plotly_layout(height=420, showlegend=True, barmode="group", margin={"l": 165, "r": 75, "t": 65, "b": 55}),
        xaxis={"title": "Held-out test score (%)", "range": [78, 94], "ticksuffix": "%", "gridcolor": "rgba(148,163,184,.12)"},
        yaxis={"title": "", "autorange": "reversed"}, legend={"orientation": "h", "y": 1.12},
    )
    st.plotly_chart(chart_figure, width="stretch", config=_plotly_config(), key="ml_performance_comparison")


def _render_training_loss(data: Any) -> None:
    """Plot training and validation losses recorded during the completed run."""
    import plotly.graph_objects as go

    training = [item for item in data.history if "loss" in item and "eval_loss" not in item]
    validation = [item for item in data.history if "eval_loss" in item]
    if not training and not validation:
        st.info("Training and validation loss were not recorded in the current trainer history.")
        return
    fig = go.Figure()
    if training:
        fig.add_trace(go.Scatter(x=[item["step"] for item in training], y=[item["loss"] for item in training], mode="lines+markers", name="Training loss", line={"color": "#22d3ee", "width": 2}, hovertemplate="Step %{x}<br>Training loss: %{y:.6f}<extra></extra>"))
    if validation:
        fig.add_trace(go.Scatter(x=[item["step"] for item in validation], y=[item["eval_loss"] for item in validation], mode="lines+markers", name="Validation loss", line={"color": "#fbbf24", "width": 3}, marker={"size": 10}, customdata=[[item.get("epoch")] for item in validation], hovertemplate="Step %{x}<br>Epoch %{customdata[0]:.0f}<br>Validation loss: %{y:.6f}<extra></extra>"))
    best_step = max(item.get("step", 0) for item in validation) if validation else None
    if best_step:
        fig.add_vline(x=best_step, line_dash="dot", line_color="#34d399", annotation_text=data.best_checkpoint, annotation_position="top left")
    fig.update_layout(**_plotly_layout(height=360, showlegend=True), xaxis={"title": "Training step"}, yaxis={"title": "Logged loss"}, legend={"orientation": "h", "y": 1.12})
    st.plotly_chart(fig, width="stretch", config=_plotly_config(), key="ml_loss_chart")
    st.caption("Training loss is logged every 25 steps. Validation loss is logged only at completed evaluation epochs; no points are interpolated.")


def _render_validation_dynamics(data: Any) -> None:
    """Plot stored validation accuracy and macro-F1 by training epoch."""
    import plotly.graph_objects as go

    evaluations = [item for item in data.history if "eval_accuracy" in item and "eval_macro_f1" in item]
    if not evaluations:
        st.info("Validation accuracy and macro-F1 were not recorded in the current trainer history.")
        return
    fig = go.Figure()
    for name, field, color in [("Validation accuracy", "eval_accuracy", "#22d3ee"), ("Validation macro-F1", "eval_macro_f1", "#34d399")]:
        fig.add_trace(go.Scatter(x=[item["epoch"] for item in evaluations], y=[item[field] * 100 for item in evaluations], mode="lines+markers", name=name, line={"color": color, "width": 3}, marker={"size": 10}, customdata=[[item["step"]] for item in evaluations], hovertemplate="Epoch %{x:.0f}<br>" + name + ": %{y:.4f}%<br>Step %{customdata[0]}<extra></extra>"))
    best_f1 = max(evaluations, key=lambda item: item["eval_macro_f1"])
    fig.add_annotation(x=best_f1["epoch"], y=best_f1["eval_macro_f1"] * 100, text=f'{data.best_checkpoint}<br>Best macro-F1 {best_f1["eval_macro_f1"]:.4f}', showarrow=True, arrowcolor="#34d399", bgcolor="rgba(7,18,36,.94)", bordercolor="#34d399")
    fig.update_layout(**_plotly_layout(height=360, showlegend=True), xaxis={"title": "Epoch", "dtick": 1}, yaxis={"title": "Validation score (%)", "range": [80, 95]}, legend={"orientation": "h", "y": 1.12})
    st.plotly_chart(fig, width="stretch", config=_plotly_config(), key="ml_validation_chart")


def _render_learning_rate(data: Any) -> None:
    """Plot the learning-rate values preserved in trainer history."""
    import plotly.graph_objects as go

    logged = [item for item in data.history if "learning_rate" in item]
    if not logged:
        st.info("Learning-rate progression was not preserved in the trainer history.")
        return
    fig = go.Figure(go.Scatter(x=[item["step"] for item in logged], y=[item["learning_rate"] for item in logged], mode="lines+markers", line={"color": "#a78bfa", "width": 2.5}, marker={"size": 6}, hovertemplate="Step %{x}<br>Learning rate: %{y:.8f}<extra></extra>"))
    fig.update_layout(**_plotly_layout(height=300), xaxis={"title": "Training step"}, yaxis={"title": "Logged learning rate", "tickformat": ".2e"})
    st.plotly_chart(fig, width="stretch", config=_plotly_config(), key="ml_learning_rate")


def _render_training_timeline() -> None:
    """Render the fixed sequence of completed experiment stages."""
    labels = ["Financial PhraseBank", "Deduplication", "Stratified split", "BERT tokenization", "GPU fine-tuning", "Validation", "Best checkpoint", "Held-out test"]
    st.markdown('<div class="ml-timeline">' + ''.join(f'<div><span>{index:02d}</span><strong>{_safe(label)}</strong></div>' for index, label in enumerate(labels, 1)) + '</div>', unsafe_allow_html=True)


def _render_training_configuration(data: Any) -> None:
    """Render verified model and training settings from stored artifacts."""
    import pandas as pd

    config = data.manifest.get("configuration", {})
    training_args_path = PROJECT_ROOT / "artifacts" / "models" / "bert_sentiment" / "final_model" / "training_args.bin"
    optimizer = "Not recorded"
    if training_args_path.exists():
        try:
            import torch
            arguments = torch.load(training_args_path, map_location="cpu", weights_only=False)
            optimizer = str(getattr(arguments, "optim", "Not recorded"))
        except Exception:
            optimizer = "Not recorded"
    training_seconds = data.manifest.get("timing", {}).get("training_seconds")
    fields = [
        ("Base checkpoint", config.get("model_id")), ("Model revision", config.get("model_revision")),
        ("Maximum sequence length", config.get("max_length")), ("Number of classes", 3),
        ("Batch size", config.get("train_batch_size")),
        ("Effective batch size", (config.get("train_batch_size") or 0) * (config.get("gradient_accumulation_steps") or 0) or None),
        ("Gradient accumulation", config.get("gradient_accumulation_steps")), ("Learning rate", config.get("learning_rate")),
        ("Weight decay", config.get("weight_decay")), ("Optimizer", optimizer),
        ("Epochs", config.get("number_of_epochs")), ("Best-model metric", "Macro-F1"),
        ("Seed", config.get("random_seed")), ("Device", data.manifest.get("device")),
        ("Training duration", f"{training_seconds:.2f} seconds" if isinstance(training_seconds, (int, float)) else None),
    ]
    st.dataframe(pd.DataFrame([{"Configuration": name, "Verified value": str(value) if value is not None else "Not recorded"} for name, value in fields]), hide_index=True, width="stretch", height=390)


def _render_resource_panel(data: Any) -> None:
    """Render measured training and inference resource information."""
    benchmark = data.benchmark
    timing = data.manifest.get("timing", {})
    cards = [
        ("GPU", benchmark.get("device_name", "Not recorded")),
        ("Training time", f'{timing.get("training_seconds"):.2f} seconds' if isinstance(timing.get("training_seconds"), (int, float)) else "Not recorded"),
        ("Peak CUDA allocation", f'{benchmark.get("cuda_peak_allocated_bytes") / 1_000_000:.1f} MB' if isinstance(benchmark.get("cuda_peak_allocated_bytes"), (int, float)) else "Not recorded"),
        ("Model artifact", f'{benchmark.get("model_artifact_size_bytes"):,} bytes' if isinstance(benchmark.get("model_artifact_size_bytes"), int) else "Not recorded"),
        ("Best checkpoint", data.best_checkpoint),
    ]
    st.markdown('<div class="ml-resource-grid">' + ''.join(f'<div class="fs-card"><span>{_safe(name)}</span><strong>{_safe(value)}</strong></div>' for name, value in cards) + '</div>', unsafe_allow_html=True)
    st.caption("Peak CUDA allocation is the measured inference-process allocation, not total GPU capacity.")


def _render_efficiency_frontier(data: Any) -> None:
    """Compare stored model quality with measured artifact size and latency."""
    import plotly.graph_objects as go

    benchmark = data.benchmark
    artifact_size = benchmark.get("model_artifact_size_bytes")
    if not isinstance(artifact_size, int):
        st.info("A verified model artifact size is not available for the measured runtime frontier.")
        return
    size_mib = artifact_size / (1024 ** 2)
    fig = go.Figure(go.Scatter(
        x=[size_mib], y=[data.current.macro_f1], mode="markers+text", text=["Full BERT current"], textposition="top center",
        marker={"size": 30, "color": "#22d3ee", "line": {"color": "#f8fafc", "width": 2}},
        customdata=[[data.current.accuracy, benchmark.get("warm_sentence_milliseconds"), artifact_size, "Full fine-tuning"]],
        hovertemplate="<b>Full BERT current</b><br>Accuracy: %{customdata[0]:.4%}<br>Macro-F1: %{y:.4f}<br>Warm GPU latency: %{customdata[1]:.3f} ms/sentence<br>Artifact size: %{customdata[2]:,} bytes<br>%{customdata[3]}<extra></extra>",
    ))
    fig.update_layout(**_plotly_layout(height=330), xaxis={"title": "Verified model artifact size (MiB)", "range": [max(0, size_mib - 70), size_mib + 70]}, yaxis={"title": "Held-out macro-F1", "range": [.78, .92]})
    st.plotly_chart(fig, width="stretch", config=_plotly_config(), key="ml_efficiency_frontier")
    st.caption("Only Full BERT has a preserved runtime benchmark. DistilBERT and BERT-LoRA are omitted from this measured frontier rather than assigned estimated latency or size.")


def _render_runtime_panel(data: Any) -> None:
    """Render the saved Full BERT runtime benchmark values."""
    benchmark = data.benchmark
    values = [
        ("Model load", f'{benchmark.get("model_load_seconds"):.2f} seconds' if isinstance(benchmark.get("model_load_seconds"), (int, float)) else "Not recorded"),
        ("Warm GPU inference", f'{benchmark.get("warm_sentence_milliseconds"):.2f} ms / sentence' if isinstance(benchmark.get("warm_sentence_milliseconds"), (int, float)) else "Not recorded"),
        ("Artifact size", f'{benchmark.get("model_artifact_size_bytes"):,} bytes' if isinstance(benchmark.get("model_artifact_size_bytes"), int) else "Not recorded"),
        ("Peak CUDA allocation", f'{benchmark.get("cuda_peak_allocated_bytes") / 1_000_000:.1f} MB' if isinstance(benchmark.get("cuda_peak_allocated_bytes"), (int, float)) else "Not recorded"),
        ("Device", benchmark.get("device_name", "Not recorded")),
        ("CPU latency", f'{benchmark["cpu_sentence_milliseconds"]:.2f} ms / sentence' if isinstance(benchmark.get("cpu_sentence_milliseconds"), (int, float)) else "Not recorded"),
    ]
    st.markdown('<div class="ml-runtime-grid">' + ''.join(f'<div class="fs-card"><span>{_safe(name)} {_model_badge("RUNTIME", "runtime")}</span><strong>{_safe(value)}</strong></div>' for name, value in values) + '</div>', unsafe_allow_html=True)


def _render_architecture_cards(data: Any) -> None:
    """Summarize the compared Transformer experiment architectures."""
    cards = [
        ("Full BERT", "Complete Transformer encoder", "Full fine-tuning", "Strongest verified balanced performance", "Selected local inference model"),
        ("DistilBERT", "Compressed Transformer architecture", "Lower operational footprint", "Small verified performance reduction", "Efficient deployment alternative"),
        ("BERT-LoRA", "Parameter-efficient fine-tuning", "Reduced training adaptation", "Lower held-out performance in this experiment", "Experimental alternative"),
    ]
    st.markdown('<div class="ml-architecture-grid">' + ''.join(f'<div class="fs-card"><div class="fs-eye">{_safe(model)}</div><strong>{_safe(a)}</strong><span>{_safe(b)}</span><span>{_safe(c)}</span><span>{_safe(d)}</span></div>' for model, a, b, c, d in cards) + '</div>', unsafe_allow_html=True)


def _render_confusion_lab(data: Any) -> None:
    """Render the current confusion matrix in fixed sentiment-class order."""
    import plotly.graph_objects as go

    matrix_view = st.segmented_control("Matrix view", ["Counts", "Normalized by actual class", "Normalized by predicted class"], default="Counts", key="ml_confusion_mode", width="stretch") or "Counts"
    confusion_matrix = data.current.confusion_matrix
    matrix_values = normalize_confusion(confusion_matrix, matrix_view)
    row_totals = [sum(matrix_row) for matrix_row in confusion_matrix]
    column_totals = [sum(confusion_matrix[row_index][column_index] for row_index in range(3)) for column_index in range(3)]
    # Rows are actual classes and columns are predictions in fixed BERT class order.
    hover_values = [[[
        confusion_matrix[row_index][column_index], 100 * confusion_matrix[row_index][column_index] / row_totals[row_index] if row_totals[row_index] else 0,
        100 * confusion_matrix[row_index][column_index] / column_totals[column_index] if column_totals[column_index] else 0,
    ] for column_index in range(3)] for row_index in range(3)]
    annotations = [[f"{value:.1f}%" if matrix_view != "Counts" else str(int(value)) for value in matrix_row] for matrix_row in matrix_values]
    chart_figure = go.Figure(go.Heatmap(
        z=matrix_values, x=list(BERT_LABEL_ORDER), y=list(BERT_LABEL_ORDER), customdata=hover_values,
        text=annotations, texttemplate="%{text}", textfont={"size": 16},
        colorscale=[[0, "#0b1628"], [.5, "#155e75"], [1, "#34d399"]],
        hovertemplate="True: %{y}<br>Predicted: %{x}<br>Count: %{customdata[0]}<br>Row percentage: %{customdata[1]:.2f}%<br>Column percentage: %{customdata[2]:.2f}%<extra></extra>",
        colorbar={"title": "%" if matrix_view != "Counts" else "Records", "thickness": 13},
    ))
    chart_figure.update_layout(**_plotly_layout(height=430, margin={"l": 78, "r": 35, "t": 35, "b": 68}), xaxis={"title": "Predicted label", "side": "bottom"}, yaxis={"title": "True label", "autorange": "reversed"})
    st.plotly_chart(chart_figure, width="stretch", config=_plotly_config(), key="ml_confusion_matrix")


def _render_per_class_metrics(data: Any) -> None:
    """Render stored precision, recall, and F1 for each sentiment class."""
    import plotly.graph_objects as go

    fig = go.Figure()
    for measure, color in [("precision", "#818cf8"), ("recall", "#22d3ee"), ("f1", "#34d399")]:
        fig.add_trace(go.Bar(name=measure.title(), y=list(BERT_LABEL_ORDER), x=[data.current.per_class[label][measure] * 100 for label in BERT_LABEL_ORDER], orientation="h", marker_color=color, customdata=[[data.current.per_class[label]["support"]] for label in BERT_LABEL_ORDER], hovertemplate="<b>%{y}</b><br>" + measure.title() + ": %{x:.4f}%<br>Support: %{customdata[0]:.0f}<extra></extra>"))
    fig.update_layout(**_plotly_layout(height=350, showlegend=True, barmode="group", margin={"l": 80, "r": 35, "t": 55, "b": 55}), xaxis={"title": "Held-out score (%)", "range": [75, 100]}, yaxis={"title": ""}, legend={"orientation": "h", "y": 1.13})
    st.plotly_chart(fig, width="stretch", config=_plotly_config(), key="ml_per_class")


def _render_error_flows(data: Any) -> None:
    """Visualize misclassification counts from the stored confusion matrix."""
    import plotly.graph_objects as go

    flows = sorted(error_flows(data.current.confusion_matrix), key=lambda item: (-item[2], item[0], item[1]))
    if not flows:
        st.info("The current confusion matrix contains no off-diagonal errors.")
        return
    labels = [f"True {source} → Predicted {target}" for source, target, _ in flows]
    counts = [count for _, _, count in flows]
    fig = go.Figure(go.Bar(x=counts, y=labels, orientation="h", marker_color="#fb7185", text=counts, textposition="outside", hovertemplate="%{y}<br>Error count: %{x}<extra></extra>"))
    fig.update_layout(**_plotly_layout(height=340, margin={"l": 215, "r": 45, "t": 30, "b": 50}), xaxis={"title": "Misclassified test records", "dtick": 2}, yaxis={"title": "", "autorange": "reversed"})
    st.plotly_chart(fig, width="stretch", config=_plotly_config(), key="ml_error_flows")


def _render_diagnostics(data: Any) -> None:
    """Render evaluation diagnostics without rerunning model training."""
    findings = diagnostic_findings(data.current)
    strongest = findings["strongest"]
    weakest_label, weakest_measure, weakest_value = findings["weakest"]
    source, target, count = findings["largest_flow"]
    cards = [
        ("Strongest precision", f'{strongest["precision"]} · {data.current.per_class[strongest["precision"]]["precision"]:.4f}'),
        ("Strongest recall", f'{strongest["recall"]} · {data.current.per_class[strongest["recall"]]["recall"]:.4f}'),
        ("Strongest F1", f'{strongest["f1"]} · {data.current.per_class[strongest["f1"]]["f1"]:.4f}'),
        ("Weakest class metric", f'{weakest_label} {weakest_measure} · {weakest_value:.4f}'),
        ("Largest error direction", f'{source} → {target} · {count} records'),
        ("Largest error share", f'{findings["largest_error_share"]:.1%} of {findings["total_errors"]} errors'),
    ]
    st.markdown('<div class="ml-diagnostic-grid">' + ''.join(f'<div class="fs-card"><span>{_safe(name)}</span><strong>{_safe(value)}</strong></div>' for name, value in cards) + '</div>', unsafe_allow_html=True)
    st.caption(f'Error pattern: {findings["error_pattern"]}. This statement is calculated from the current off-diagonal confusion-matrix counts.')


def _render_dataset_provenance(data: Any) -> None:
    """Render stored Financial PhraseBank counts and split provenance."""
    import plotly.graph_objects as go

    summary = data.manifest.get("dataset_summary", {})
    counts = {label: sum(int(split.get(label, 0)) for split in summary.values() if isinstance(split, dict)) for label in BERT_LABEL_ORDER}
    details, chart = st.columns([1, 1], gap="large")
    with details:
        st.markdown("""
        | Field | Verified value |
        |---|---|
        | Dataset | Financial PhraseBank |
        | Configuration | `sentences_75agree` |
        | Original records | 3,453 |
        | Deduplicated records | 3,448 |
        | Duplicates removed | 5 |
        | Classes | Bearish, Neutral, Bullish |
        | Split | Fixed stratified train, validation and test |
        """)
    with chart:
        if sum(counts.values()) == 3448:
            fig = go.Figure(go.Bar(x=list(counts), y=[counts[label] for label in BERT_LABEL_ORDER], marker_color=[_BERT_COLORS[label] for label in BERT_LABEL_ORDER], text=[f'{counts[label]:,}' for label in BERT_LABEL_ORDER], textposition="outside", hovertemplate="%{x}: %{y:,} records<extra></extra>"))
            fig.update_layout(**_plotly_layout(height=280, margin={"l": 50, "r": 25, "t": 25, "b": 45}), yaxis={"title": "Deduplicated records"}, xaxis={"title": ""})
            st.plotly_chart(fig, width="stretch", config=_plotly_config(), key="ml_dataset_distribution")
        else:
            st.info("Exact class-distribution counts were not available in the current manifest.")


def _render_metric_provenance(data: Any) -> None:
    """Show where current and historical experiment metrics were loaded."""
    st.markdown("""
    **Current reproduced run**
    Test accuracy, macro metrics, class metrics and confusion matrix: `bert_sentiment_current_run_metrics.json`
    Trainer loss, validation metrics, learning rate and best checkpoint: `bert_sentiment_current_run_history.json`
    Dataset, configuration, device and training timing: `bert_sentiment_current_run_manifest.json`

    **Verified historical experiments**
    Full BERT: `bert_sentiment_metrics.json`
    DistilBERT: `distilbert_sentiment_metrics.json`
    BERT-LoRA: `bert_lora_sentiment_metrics.json`

    **Runtime benchmark**
    Model load time, warm GPU latency, CUDA allocation, device and measured artifact size: `bert_sentiment_current_run_benchmark.json`

    Current values are never substituted for historical values, and historical values are never substituted for the reproduced run.
    """)


def _render_verified_model_results_page() -> None:
    """Render the AI Experiment Lab from verified stored results."""

    try:
        experiment_results = load_experiment_lab_data(PROJECT_ROOT)
    except ExperimentDataError as data_error:
        st.markdown('<div class="fs-head"><div class="fs-eye">MODEL EVALUATION · VERIFIED EXPERIMENTS</div><h1 class="fs-title">Full BERT Experiment Lab</h1></div>', unsafe_allow_html=True)
        st.error(f"Verified experiment data could not be loaded: {data_error}")
        return
    # These are stored experiment results; rendering this page never retrains a model.
    current_result = experiment_results.current
    historical_bert = experiment_results.historical["Full BERT historical"]
    training_seconds = experiment_results.manifest.get("timing", {}).get("training_seconds")
    st.markdown(
        '<div class="fs-head"><div class="fs-eye">MODEL EVALUATION · VERIFIED EXPERIMENTS</div>'
        '<h1 class="fs-title">Full BERT Experiment Lab</h1>'
        '<p class="fs-copy">Verified Transformer training, evaluation, efficiency and error analysis on Financial PhraseBank.</p></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="fs-card ml-champion"><div><span class="fs-badge">FULL BERT</span><h2>Deployed local inference model</h2>'
        f'<p>Current reproduced run {_model_badge("CURRENT RUN")} and verified historical experiment {_model_badge("HISTORICAL", "historical")} are reported separately.</p></div>'
        f'<div class="ml-champion-metrics"><div><span>Current test accuracy</span><strong>{current_result.accuracy * 100:.2f}%</strong></div>'
        f'<div><span>Current macro-F1</span><strong>{current_result.macro_f1:.4f}</strong></div>'
        f'<div><span>Historical best accuracy</span><strong>{historical_bert.accuracy * 100:.2f}%</strong></div>'
        f'<div><span>Historical best macro-F1</span><strong>{historical_bert.macro_f1:.4f}</strong></div>'
        f'<div><span>Best checkpoint</span><strong>{_safe(experiment_results.best_checkpoint)}</strong></div>'
        f'<div><span>Training time</span><strong>{training_seconds:.2f} seconds</strong></div></div></div>', unsafe_allow_html=True,
    )
    summary_items = ["3 Transformer experiments", "3,448 deduplicated sentences", "Full GPU fine-tuning", "3 sentiment classes", "Fixed held-out test evaluation"]
    st.markdown('<div class="ml-summary-strip">' + ''.join(f'<span>{_safe(summary_item)}</span>' for summary_item in summary_items) + '</div>', unsafe_allow_html=True)

    executive, training, efficiency, errors = st.tabs(["Executive Results", "Training Dynamics", "Benchmark & Efficiency", "Error Analysis"])
    with executive:
        _section_heading("LEADERBOARD", "Verified model leaderboard", "Ranked by verified historical macro-F1.")
        _render_experiment_leaderboard(experiment_results)
        _section_heading("COMPARISON", "Current and historical held-out performance")
        _render_experiment_comparison(experiment_results)
        st.caption("Accuracy measures overall correctness. Macro-F1 gives equal importance to all three sentiment classes.")
        _section_heading("INTERPRETATION", "Champion interpretation")
        st.markdown("""
        - Full BERT achieved the strongest verified historical macro-F1.
        - The reproduced Full BERT run remained close to the historical Full BERT benchmark.
        - DistilBERT offered the strongest performance among the lighter alternatives.
        - BERT-LoRA reduced fine-tuning requirements but produced lower held-out performance in this experiment.
        """)
    with training:
        loss_col, validation_col = st.columns(2, gap="large")
        with loss_col:
            _section_heading("LOSS", "Training and validation loss")
            _render_training_loss(experiment_results)
        with validation_col:
            _section_heading("VALIDATION", "Validation performance")
            _render_validation_dynamics(experiment_results)
        _section_heading("SCHEDULE", "Logged learning-rate progression")
        _render_learning_rate(experiment_results)
        _section_heading("PROCESS", "Training workflow")
        _render_training_timeline()
        config_col, resource_col = st.columns([1.15, .85], gap="large")
        with config_col:
            _section_heading("CONFIGURATION", "Verified training configuration")
            _render_training_configuration(experiment_results)
        with resource_col:
            _section_heading("RESOURCES", "Current-run resource evidence")
            _render_resource_panel(experiment_results)
    with efficiency:
        frontier, runtime = st.columns([1.08, .92], gap="large")
        with frontier:
            _section_heading("MEASURED FRONTIER", "Performance–efficiency frontier")
            _render_efficiency_frontier(experiment_results)
        with runtime:
            _section_heading("RUNTIME", "Current Full BERT runtime")
            _render_runtime_panel(experiment_results)
        _section_heading("ARCHITECTURES", "Experiment architecture comparison")
        _render_architecture_cards(experiment_results)
        _section_heading("DEPLOYMENT", "Deployment interpretation")
        _scope_note("Full BERT is suitable for the local presentation application. Its approximately 419 MiB artifact requires deliberate cloud distribution. DistilBERT remains a practical constrained-deployment alternative. No public-cloud runtime claim is made until hosting and runtime are validated.")
    with errors:
        matrix_col, class_col = st.columns(2, gap="large")
        with matrix_col:
            _section_heading("CONFUSION MATRIX", "Current reproduced-run errors")
            _render_confusion_lab(experiment_results)
        with class_col:
            _section_heading("PER CLASS", "Precision, recall and F1")
            _render_per_class_metrics(experiment_results)
        flow_col, findings_col = st.columns([1.05, .95], gap="large")
        with flow_col:
            _section_heading("ERROR FLOW", "Ranked misclassification directions")
            _render_error_flows(experiment_results)
        with findings_col:
            _section_heading("DIAGNOSTICS", "Calculated diagnostic findings")
            _render_diagnostics(experiment_results)

    with st.expander("Dataset preparation"):
        _render_dataset_provenance(experiment_results)
    with st.expander("Experiment configuration"):
        _render_training_configuration(experiment_results)
    with st.expander("Metric provenance"):
        _render_metric_provenance(experiment_results)

def _ar_flow(nodes: list[tuple[str, str, str]]) -> str:
    """Render one responsive technical connector lane."""

    return '<div class="ar-flow">' + '<i class="ar-connector"></i>'.join(
        f'<div class="ar-node"><small>{_safe(status)}</small><strong>{_safe(title)}</strong><span>{_safe(detail)}</span></div>'
        for title, detail, status in nodes
    ) + '</div>'





def _render_compact_architecture_mode(mode: str, data: Any) -> None:
    """Render one architecture blueprint without running training or inference."""
    current_result = data.current
    historical_result = data.historical["Full BERT historical"]
    if mode == "Runtime system":
        workflow_steps = [
            ("Article source", "URL, pasted text or sample", "Operational"),
            ("Clean extraction", "Readable content retained", "Validated"),
            ("Sentence segmentation", "Ordered sentence units", "Operational"),
            ("Batched Full BERT inference", "Cached GPU or CPU batches", "Operational"),
            ("Sentence class scores", "Bearish · Neutral · Bullish", "Verified"),
            ("Article aggregation", "Arithmetic mean of scores", "Transparent"),
            ("Evidence workspace", "Heatmap · map · tokens · cues", "Validated locally"),
        ]
        default_stage, widget_key = "Batched Full BERT inference", "ar_runtime_node"
    elif mode == "Training system":
        workflow_steps = [
            ("Financial PhraseBank", "3,453 source records", "Verified"),
            ("Deduplication", "3,448 records · 5 duplicates removed", "Verified"),
            ("Stratified splits", "Fixed train · validation · test", "Reproducible"),
            ("Tokenization", "BERT tokenizer · length 128", "Verified"),
            ("Transformer experiments", "Three model branches", "Complete"),
            ("Held-out evaluation", "Reports · matrix · macro-F1", "Verified"),
            ("Champion selection", "checkpoint-453", "Verified"),
            ("Champion model artifact", "Weights · tokenizer · config", "Local artifact ready"),
        ]
        default_stage, widget_key = "Champion model artifact", "ar_training_node"
    else:
        workflow_steps = [
            ("Current local inference", "Streamlit and cached Full BERT", "Operational locally"),
            ("Public interface", "Public product client", "Planned"),
            ("Authenticated API", "HTTPS controls and limits", "Planned"),
            ("Model service", "Containerised FastAPI runtime", "Planned"),
            ("Managed runtime", "Monitoring and scaling", "Planned"),
        ]
        default_stage, widget_key = "Current local inference", "ar_deployment_node"
    selected_stage = st.segmented_control("Inspect stage", [stage[0] for stage in workflow_steps], default=default_stage, key=widget_key, width="stretch") or default_stage

    if mode == "Runtime system":
        st.markdown('<section class="ar-blueprint ar-runtime-blueprint"><div class="ar-mode-head"><div><small>RUNTIME SYSTEM</small><strong>Article to inspectable evidence</strong></div><span>WORKING LOCALLY</span></div>' + _ar_flow(workflow_steps) + '<div class="ar-artifact-dock"><small>SAVED MODEL HANDOFF</small><strong>FULL BERT ARTIFACT</strong><span>google-bert/bert-base-uncased · 3 classes · 109.5M · maximum length 128 · 438,912,889 bytes</span><b>LOCAL ARTIFACT READY</b></div></section>', unsafe_allow_html=True)
    elif mode == "Training system":
        st.markdown('<section class="ar-blueprint ar-training-blueprint"><div class="ar-mode-head"><div><small>TRAINING SYSTEM</small><strong>Dataset to champion artifact</strong></div><span>VERIFIED LINEAGE</span></div>' + _ar_flow(workflow_steps[:4]) + '<div class="ar-branch"><i></i><div><span>FULL BERT</span><span>DISTILBERT</span><span>BERT-LoRA</span></div><i></i></div>' + _ar_flow(workflow_steps[5:]) + f'<div class="ar-training-proof"><div><small>CURRENT FULL BERT</small><strong>{current_result.accuracy * 100:.2f}% accuracy · {current_result.macro_f1:.4f} macro-F1</strong><span>checkpoint-453 · 80.93 seconds · Bearish 420 · Neutral 2,141 · Bullish 887</span></div><div><small>HISTORICAL BEST · SEPARATE RUN</small><strong>{historical_result.accuracy * 100:.2f}% accuracy · {historical_result.macro_f1:.4f} macro-F1</strong></div></div></section>', unsafe_allow_html=True)
    else:
        st.markdown('<section class="ar-blueprint ar-deployment-blueprint"><div class="ar-topology ar-current"><div class="ar-mode-head"><div><small>CURRENT WORKING TOPOLOGY</small><strong>Operational locally</strong></div><span>VALIDATED</span></div>' + _ar_flow([("Streamlit interface","Presentation-ready","Local"),("Local cached Full BERT","Saved artifact","Ready"),("GPU or CPU inference","Both paths validated","Operational"),("Evidence visualisations","Inspectable output","Ready")]) + '<p>Windows validated · NVIDIA GPU validated · CPU fallback validated · health endpoint operational · presentation-ready</p></div><div class="ar-topology ar-target"><div class="ar-mode-head"><div><small>TARGET PRODUCTION TOPOLOGY</small><strong>Architecture target — not yet publicly validated</strong></div><span>PLANNED</span></div>' + _ar_flow([("Public web interface","Product client","Planned"),("Authenticated HTTPS API","Security and limits","Planned"),("FastAPI model service","Containerised runtime","Planned"),("Versioned model storage","Artifact distribution","Planned"),("Managed CPU/GPU","Monitoring and scale","Planned")]) + '<p>Unresolved: model artifact distribution · hosted inference service · security and request controls · monitoring and scaling</p></div></section>', unsafe_allow_html=True)
        with st.expander("Detailed deployment requirements"):
            st.markdown("- Versioned artifact distribution\n- Authenticated HTTPS inference\n- Host-specific PyTorch validation\n- Request controls, health monitoring, scaling and cost validation")

    selected_stage_details = next(stage for stage in workflow_steps if stage[0] == selected_stage)
    inspector_text = {
        "Runtime system": ("Financial article or prior stage output.", "Runs the selected article-processing stage.", "Model or evidence output for the next stage."),
        "Training system": ("Verified dataset or prior experiment artifact.", "Runs the selected preparation, experiment or evaluation stage.", "Reproducible evidence or saved model artifact."),
        "Deployment topology": ("Validated local component or planned service input.", "Runs locally now or describes a production responsibility.", "Operational local output or planned service handoff."),
    }[mode]
    st.markdown(f'<div class="ar-inspector"><div><small>INPUT</small><strong>{inspector_text[0]}</strong></div><div><small>PROCESS</small><strong>{inspector_text[1]}</strong></div><div><small>OUTPUT</small><strong>{inspector_text[2]}</strong></div><div><small>STATUS</small><strong>{selected_stage_details[2]}</strong></div></div>', unsafe_allow_html=True)


def _render_sentiment_architecture_page() -> None:
    """Document the active system architecture without running model inference."""
    try:
        experiment_results = load_experiment_lab_data(PROJECT_ROOT)
    except ExperimentDataError as data_error:
        st.error(f"Verified architecture evidence could not be loaded: {data_error}")
        return
    current_result = experiment_results.current
    runtime_benchmark = experiment_results.benchmark
    historical_result = experiment_results.historical["Full BERT historical"]
    artifact_bytes = runtime_benchmark.get("model_artifact_size_bytes", 438_912_889)
    artifact_size_mib = artifact_bytes / (1024 ** 2)

    st.markdown('''<section class="ar-hero"><div class="ar-eyebrow">SYSTEM ARCHITECTURE · MODEL LINEAGE · MLOPS</div><h1>One model artifact. Two connected systems.</h1><p>A verified Full BERT model links the training laboratory with a reusable financial-news inference product.</p></section>''', unsafe_allow_html=True)
    system_statuses=[("FULL BERT","Verified","ok"),("LOCAL INFERENCE","Operational","ok"),("GPU RUNTIME","Operational","ok"),("CPU FALLBACK","Validated","ok"),("PUBLIC INTERFACE","Validated locally","local"),("CLOUD MODEL SERVICE","Not yet validated","planned"),("TEST SUITE","Passing","ok")]
    st.markdown('<section class="ar-command-bar">' + ''.join(f'<div><small>{status_name}</small><strong class="ar-{status_class}">{status_value}</strong></div>' for status_name,status_value,status_class in system_statuses) + '</section>', unsafe_allow_html=True)
    st.markdown(f'''<section class="ar-observatory"><div class="ar-observatory-head"><div><div class="ar-eyebrow">INTERACTIVE ARCHITECTURE BLUEPRINT</div><h2>Training lineage connected to article intelligence</h2></div><span>SELECT A MODE · INSPECT A STAGE</span></div><div class="ar-system-map"><div class="ar-system-side"><small>TRAINING LABORATORY</small><strong>Data → experiments → evaluation</strong><span>Financial PhraseBank · Full BERT · DistilBERT · BERT-LoRA</span></div><i class="ar-map-link"></i><div class="ar-central-artifact"><small>FULL BERT ARTIFACT</small><strong>google-bert/bert-base-uncased</strong><span>3 classes · 109.5M parameters · maximum length 128</span><span>{artifact_bytes:,} bytes</span><b>LOCAL ARTIFACT READY</b></div><i class="ar-map-link"></i><div class="ar-system-side"><small>INFERENCE PRODUCT</small><strong>Article → model → evidence</strong><span>Extraction · sentence scores · aggregation · workspace</span></div></div></section>''', unsafe_allow_html=True)
    architecture_view=st.segmented_control("Architecture view",["Runtime system","Training system","Deployment topology"],default="Runtime system",key="ar_view_mode",width="stretch") or "Runtime system"
    _render_compact_architecture_mode(architecture_view,experiment_results)

    cpu_latency=runtime_benchmark.get("cpu_sentence_milliseconds")
    st.markdown(f'''<section class="ar-evidence"><div class="ar-section-head"><div><div class="ar-eyebrow">ENGINEERING EVIDENCE</div><h2>Verified measurements and artifact lineage</h2></div><span>CURRENT RUN · NOT LIVE LOGS</span></div><div class="ar-instrument"><div class="ar-instrument-primary"><div><small>MODEL</small><strong>Full BERT</strong></div><div><small>STATUS</small><strong>Ready</strong></div><div><small>DEVICE</small><strong>{_safe(runtime_benchmark.get("device_name","Not recorded"))}</strong></div></div><div class="ar-gauges"><div><small>MODEL LOAD</small><strong>{runtime_benchmark.get("model_load_seconds"):.2f} s</strong></div><div><small>WARM GPU INFERENCE</small><strong>{runtime_benchmark.get("warm_sentence_milliseconds"):.2f} ms / sentence</strong></div><div><small>MODEL ARTIFACT</small><strong>{artifact_size_mib:.2f} MiB</strong></div><div><small>CURRENT MACRO-F1</small><strong>{current_result.macro_f1:.4f}</strong></div></div><div class="ar-technical-row"><span>Current accuracy <b>{current_result.accuracy*100:.2f}%</b></span><span>Best checkpoint <b>{experiment_results.best_checkpoint}</b></span><span>CUDA peak allocation <b>{runtime_benchmark.get("cuda_peak_allocated_bytes")/1_000_000:.1f} MB</b></span><span>CPU latency <b>{f"{cpu_latency:.2f} ms" if isinstance(cpu_latency,(int,float)) else "Not recorded"}</b></span></div></div>''', unsafe_allow_html=True)
    lineage=[("SOURCE","3,453 Financial PhraseBank sentences","ACQUISITION"),("PREPARATION","3,448 deduplicated records","JSONL"),("SPLITS","Fixed stratified train, validation and test","MANIFEST"),("TRAINING","Trainer history and checkpoints","CHECKPOINTS"),("EVALUATION","Classification report and confusion matrix","METRICS"),("MODEL","Final Full BERT weights and tokenizer","ARTIFACT"),("INFERENCE","Article and sentence evidence outputs","RUNTIME")]
    st.markdown('<div class="ar-lineage"><div class="ar-section-head"><div><div class="ar-eyebrow">MODEL AND DATA LINEAGE</div><h2>One continuous artifact trail</h2></div></div><div class="ar-lineage-track">'+'<i></i>'.join(f'<div><small>{a}</small><strong>{b}</strong><span>{c}</span></div>' for a,b,c in lineage)+'</div></div>',unsafe_allow_html=True)
    with st.expander("Artifact registry"):
        st.markdown("- Current-run manifest, metrics, benchmark and trainer history\n- checkpoint-453 trainer state\n- Final model weights, tokenizer and configuration")
    guarantees=[("01","Verified class mapping","0 Bearish · 1 Neutral · 2 Bullish"),("02","Fixed held-out evaluation","Current and historical metrics remain separated"),("03","Deterministic inference","Cached evaluation mode with batched sentence processing"),("04","Transparent aggregation","Article result is the arithmetic mean of sentence scores"),("05","Evidence boundaries","Sentence evidence is model-based; lexical cues remain secondary"),("06","Product safety","No price prediction, return forecast or investment recommendation")]
    st.markdown('<div class="ar-guarantees"><div class="ar-section-head"><div><div class="ar-eyebrow">ENGINEERING GUARANTEES</div><h2>Six constraints carried through the product</h2></div></div>'+''.join(f'<div><b>{a}</b><strong>{b}</strong><span>{c}</span></div>' for a,b,c in guarantees)+'</div>',unsafe_allow_html=True)
    with st.expander("Full integrity contract"):
        st.markdown("- Artifact validation before inference\n- Cached GPU execution with CPU fallback\n- Stale-state clearing on input changes\n- WordPieces are not importance scores\n- Token similarity is not causality\n- Visible URL extraction failures")
    with st.expander("Technology by system layer"):
        st.markdown("**Product:** Streamlit · HTML/CSS · Plotly — Active locally  \n**Inference:** PyTorch · Transformers — Active locally  \n**Models:** Full BERT · DistilBERT · BERT-LoRA — Experimentation  \n**Data:** Financial PhraseBank · scikit-learn — Experimentation  \n**Serving:** FastAPI · PyTorch runtime — Production target  \n**Validation:** Docker · pytest · AppTest · local Kubernetes — Locally validated  \n**Source control:** Git · GitHub — Active locally")
    does=["classify financial-news language","calculate sentence class scores","aggregate article sentiment","surface supporting evidence","compare verified Transformer experiments"]
    nots=["predict stock prices","forecast returns","provide investment advice","consume a live market-price feed","claim token similarity proves causality","claim public Full BERT hosting is complete"]
    st.markdown('<section class="ar-boundary"><div><small>THE SYSTEM DOES</small>'+''.join(f'<span>{x}</span>' for x in does)+'</div><i></i><div><small>THE SYSTEM DOES NOT</small>'+''.join(f'<span>{x}</span>' for x in nots)+'</div></section>',unsafe_allow_html=True)
    st.markdown('''<section class="ar-closing"><div><div class="ar-eyebrow">END-TO-END TRANSFORMER PRODUCT</div><h2>Built as an end-to-end Transformer product — not only a trained notebook</h2><p>Designed, trained and engineered by Ruturaj Mokashi across financial NLP, Transformer fine-tuning, evaluation, inference engineering, evidence visualisation and deployment-aware architecture.</p></div><aside>This system analyses financial-news language. It does not predict stock prices, investment returns or future market outcomes.</aside></section>''',unsafe_allow_html=True)
    a,b,c=st.columns(3,gap="small")
    with a: st.button("Run article analysis",key="ar_run_analysis",type="primary",width="stretch",on_click=_navigate_public_page,args=("Analyze Article",))
    with b: st.button("Inspect model experiments",key="ar_model_results",width="stretch",on_click=_navigate_public_page,args=("Model Results",))
    with c: st.button("Return to Overview",key="ar_overview",width="stretch",on_click=_navigate_public_page,args=("Overview",))

def render_public_streamlit_cloud_app(project_root: Path | str | None = None) -> None:
    """Configure and route the stable four-page public Streamlit application."""

    _apply_theme()
    _render_premium_sentiment_styles()
    selected_page = _render_sidebar()

    # Each stable route name maps to exactly one active page renderer.
    page_renderers = {
        "Overview": _render_sentiment_overview_page,
        "Analyze Article": _render_sentiment_analyze_page,
        "Model Results": _render_verified_model_results_page,
        "About / Architecture": _render_sentiment_architecture_page,
    }
    page_renderer = page_renderers.get(selected_page)
    if page_renderer is None:
        st.error("This page is not available.")
        return
    page_renderer()

if __name__ == "__main__":
    # Keep the same startup target for direct runs and Streamlit execution.
    render_public_streamlit_cloud_app()
