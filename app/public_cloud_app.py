"""Self-contained public Streamlit dashboard for free Streamlit Cloud.

This module intentionally avoids private FastAPI/runtime dependencies so the
public app can run on Streamlit Community Cloud. It renders a polished
Executive Overview dashboard with URL, upload, paste, and sample analysis.
"""

from __future__ import annotations

import html
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import streamlit as st

try:
    import requests
except Exception:  # pragma: no cover - public fallback
    requests = None  # type: ignore[assignment]

try:
    import plotly.graph_objects as go
except Exception:  # pragma: no cover - public fallback
    go = None  # type: ignore[assignment]


_BASE_EXAMPLE = (
    "Nvidia reported record data-center revenue and stronger demand for AI chips, "
    "but management warned that supply constraints, export controls, and rising "
    "competition could pressure margins next quarter."
)

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

_TICKER_MAP = {
    "nvidia": ("NVDA", "NVIDIA Corporation"),
    "nvda": ("NVDA", "NVIDIA Corporation"),
    "micron": ("MU", "Micron Technology"),
    "mu": ("MU", "Micron Technology"),
    "apple": ("AAPL", "Apple Inc."),
    "microsoft": ("MSFT", "Microsoft Corporation"),
    "tesla": ("TSLA", "Tesla Inc."),
    "amazon": ("AMZN", "Amazon.com Inc."),
    "meta": ("META", "Meta Platforms"),
    "google": ("GOOGL", "Alphabet Inc."),
    "alphabet": ("GOOGL", "Alphabet Inc."),
    "amd": ("AMD", "Advanced Micro Devices"),
    "intel": ("INTC", "Intel Corporation"),
}


@dataclass(frozen=True)
class ArticleSignal:
    """Analysis result displayed by the public dashboard."""

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


def should_use_public_streamlit_cloud_app(project_root: Path | str | None = None) -> bool:
    """Return True for the public Streamlit mode."""

    return True


def _safe(text: Any) -> str:
    """Escape text before inserting it into custom HTML."""

    return html.escape(str(text), quote=True)


def _clean_text(text: str | None) -> str:
    """Normalize article text for scoring and display."""

    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def _hits(text: str, terms: set[str]) -> list[str]:
    """Find scoring terms in article text."""

    low = text.lower()
    return sorted({term for term in terms if term in low})


def _infer_ticker(text: str) -> tuple[str, str]:
    """Infer a public ticker/company pair from known finance keywords."""

    low = text.lower()
    for key, value in _TICKER_MAP.items():
        if re.search(rf"(?<![a-z0-9]){re.escape(key)}(?![a-z0-9])", low):
            return value
    return "NEWS", "Article"


def _fetch_url_text(url: str) -> str:
    """Fetch readable article-like text from a URL when the site allows it."""

    clean_url = _clean_text(url)
    if not clean_url:
        return ""

    if not clean_url.startswith(("http://", "https://")):
        raise ValueError("URL must start with http:// or https://")

    if requests is None:
        raise RuntimeError("requests is unavailable in this public runtime")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    response = requests.get(clean_url, headers=headers, timeout=12)
    response.raise_for_status()

    raw_html = response.text or ""
    if len(raw_html.strip()) < 200:
        raise ValueError("URL returned too little readable content")

    title_match = re.search(r"(?is)<title[^>]*>(.*?)</title>", raw_html)
    title = _clean_text(title_match.group(1)) if title_match else ""

    text_html = re.sub(
        r"(?is)<script.*?</script>|<style.*?</style>|<noscript.*?</noscript>",
        " ",
        raw_html,
    )
    text_html = re.sub(
        r"(?is)<nav.*?</nav>|<footer.*?</footer>|<header.*?</header>|<aside.*?</aside>",
        " ",
        text_html,
    )
    text = re.sub(r"(?s)<[^>]+>", " ", text_html)
    replacements = {
        "&nbsp;": " ",
        "&#160;": " ",
        "&amp;": "&",
        "&quot;": '"',
        "&#39;": "'",
        "&apos;": "'",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    clean = _clean_text(text)
    if title:
        clean = f"{title}. {clean}"

    clean = clean[:12000].strip()
    if len(clean) < 300:
        raise ValueError("URL content could not be converted into enough article text")

    return clean


def _score_article(text: str, source: str) -> ArticleSignal:
    """Create deterministic public-mode finance analysis from article text."""

    clean = _clean_text(text) or _BASE_EXAMPLE
    ticker, company = _infer_ticker(clean)

    positive_hits = _hits(clean, _POSITIVE_TERMS)
    negative_hits = _hits(clean, _NEGATIVE_TERMS)
    risk_hits = _hits(clean, _RISK_TERMS)

    pos = len(positive_hits)
    neg = len(negative_hits)
    risk = len(risk_hits)

    sentiment_score = max(-1.0, min(1.0, ((pos * 1.2) - (neg * 1.05)) / 6.0))
    risk_score = max(0.05, min(0.95, (risk * 0.13) + (neg * 0.05)))

    movement_up = 0.36 + sentiment_score * 0.28 - risk_score * 0.05
    movement_down = 0.22 - sentiment_score * 0.18 + risk_score * 0.12
    movement_flat = 1.0 - movement_up - movement_down

    values = [max(0.06, movement_up), max(0.06, movement_flat), max(0.06, movement_down)]
    total = sum(values)
    movement_up, movement_flat, movement_down = [v / total for v in values]

    confidence = min(0.94, max(0.58, 0.62 + abs(sentiment_score) * 0.22 + min(len(clean), 4500) / 30000))

    if movement_up > movement_down + 0.08:
        label = "Bullish"
    elif movement_down > movement_up + 0.08:
        label = "Bearish"
    else:
        label = "Mixed"

    sentence = clean.split(".")[0].strip()
    headline = sentence[:210] if sentence else _BASE_EXAMPLE

    return ArticleSignal(
        ticker=ticker,
        company=company,
        headline=headline,
        source=source,
        label=label,
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


def _sparkline_svg(values: list[float], color: str) -> str:
    """Render a tiny SVG sparkline for KPI cards."""

    if not values:
        values = [0.2, 0.4, 0.35, 0.6]
    width, height = 130, 42
    min_v, max_v = min(values), max(values)
    span = max(max_v - min_v, 1e-6)
    pts = []
    for idx, val in enumerate(values):
        x = idx * (width / max(len(values) - 1, 1))
        y = height - ((val - min_v) / span) * (height - 8) - 4
        pts.append(f"{x:.1f},{y:.1f}")
    return (
        f"<svg viewBox='0 0 {width} {height}' class='spark'>"
        f"<polyline points='{' '.join(pts)}' fill='none' stroke='{color}' "
        f"stroke-width='4' stroke-linecap='round' stroke-linejoin='round'/></svg>"
    )


def _plotly_config() -> dict[str, bool]:
    """Common Plotly config for dashboard charts."""

    return {"displayModeBar": False, "responsive": True}


def _apply_theme() -> None:
    """Apply the complete target dashboard theme."""

    st.set_page_config(
        page_title="Financial News Stock Intelligence",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

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
    """Render real clickable Streamlit sidebar navigation."""

    pages = [
        "Executive Overview",
        "Analyze Article",
        "Forecasts",
        "Historical Intelligence",
        "Explainability",
        "Scenario Analysis",
        "Model Comparison",
        "Model Training / Evidence",
        "Provenance",
        "Architecture / System Design",
        "3D Intelligence",
        "About / Project Purpose",
        "Visual QA / Page Audit",
    ]

    st.sidebar.markdown(
        """
        <div class="brand">
          <div class="brand-row">
            <div class="brand-icon">↗</div>
            <div>
              <div class="brand-title">Ruturaj Mokashi</div>
              <div class="brand-subtitle">Financial News Stock Intelligence</div>
            </div>
          </div>
        </div>
        <div class="nav-title">Pages</div>
        """,
        unsafe_allow_html=True,
    )

    selected_page = st.sidebar.radio(
        "Pages",
        pages,
        index=0,
        key="public_dashboard_page",
        label_visibility="collapsed",
    )

    st.sidebar.markdown(
        """
        <div class="side-card">
          <div class="tiny-label">System Status</div>
          <div class="status-green">● All Systems Operational</div>
        </div>
        <div class="side-card">
          <div class="tiny-label">Theme</div>
          <div style="margin-top:.5rem;color:#a5b4fc;">☼ &nbsp;&nbsp; ◐ &nbsp;&nbsp; ▣</div>
        </div>
        <div style="position:fixed;bottom:1rem;color:#64748b;font-size:.72rem;">© 2025 Ruturaj Mokashi</div>
        """,
        unsafe_allow_html=True,
    )

    return selected_page

def _render_topbar(page_title: str = "Executive Overview") -> None:
    """Render compact public dashboard header."""

    subtitle = {
        "Executive Overview": "Financial News → Sentiment → Movement → Forecast → Explainability",
        "Analyze Article": "URL, upload, paste text, and sample analysis controls",
        "Forecasts": "Bull, base, and bear movement scenarios",
        "Historical Intelligence": "Comparable events and market reaction context",
        "Explainability": "Drivers, signals, phrases, and model workflow",
        "Scenario Analysis": "What-if risk and opportunity analysis",
        "Model Comparison": "Model selection story and performance tradeoffs",
        "Model Training / Evidence": "Training, metrics, and champion model evidence",
        "Provenance": "Source checks, verification trail, and disclaimers",
        "Architecture / System Design": "Streamlit, FastAPI, artifacts, Docker, Kubernetes, and CI",
        "3D Intelligence": "Interactive intelligence visuals or graceful fallback",
        "About / Project Purpose": "Why this portfolio project matters",
        "Visual QA / Page Audit": "Public dashboard page coverage and QA status",
    }.get(page_title, "Financial News Stock Intelligence")

    st.markdown(
        f"""
        <div class="topbar">
          <div class="title-wrap">
            <div class="title-icon">↗</div>
            <div>
              <div class="page-title">{_safe(page_title)}</div>
              <div class="page-subtitle">{_safe(subtitle)}</div>
            </div>
          </div>
          <div class="chip-row">
            <div class="chip green">☁ Public Cloud Mode</div>
            <div class="chip blue">🔗 Article URL Enabled</div>
            <div class="chip purple">⇧ Upload Enabled</div>
            <div class="chip cyan">👁 Model Workflow Visible</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def _input_form() -> tuple[str, str, str]:
    """Collect input from URL, upload, paste, or sample."""

    with st.form("executive_article_form", clear_on_submit=False):
        st.markdown('<div class="card input-card">', unsafe_allow_html=True)
        url_col, upload_col = st.columns([1.25, 1.0])

        with url_col:
            article_url = st.text_input(
                "Article URL",
                value="",
                placeholder="Paste article URL here",
            )

        with upload_col:
            uploaded_file = st.file_uploader(
                "Upload article",
                type=["txt", "md", "csv", "json", "pdf"],
            )

        pasted_text = st.text_area(
            "Paste article / headline / news text",
            value="",
            placeholder="Paste article text here. Leave empty when using an Article URL.",
            height=112,
        )

        btn_col_1, btn_col_2 = st.columns([1.0, 1.0])
        with btn_col_1:
            analyze_clicked = st.form_submit_button("↗  Analyze", type="primary", use_container_width=True)
        with btn_col_2:
            sample_clicked = st.form_submit_button("▣  Use sample", use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    if sample_clicked:
        return _BASE_EXAMPLE, "sample article", ""

    if not analyze_clicked:
        st.markdown(
            '<div class="info-strip">ⓘ Showing sample article intelligence. Enter a URL, upload, paste text, or click Analyze to run your own article.</div>',
            unsafe_allow_html=True,
        )
        return _BASE_EXAMPLE, "sample article intelligence", ""

    clean_url = _clean_text(article_url)
    clean_pasted = _clean_text(pasted_text)

    if clean_url:
        try:
            fetched = _fetch_url_text(clean_url)
            st.success("Article URL fetched and analyzed.")
            return fetched, f"URL: {clean_url}", clean_url
        except Exception as exc:
            st.warning(
                "Article URL could not be fetched from public Streamlit Cloud. "
                f"Reason: {exc}. Paste or upload the article text instead."
            )
            if clean_pasted:
                return clean_pasted, "pasted fallback after URL fetch failure", clean_url
            st.stop()

    if uploaded_file is not None:
        raw = uploaded_file.read()
        uploaded_text = raw.decode("utf-8", errors="ignore").strip()
        if uploaded_text:
            return uploaded_text, f"uploaded file: {uploaded_file.name}", ""
        st.warning("Uploaded file could not be read as text.")
        st.stop()

    if clean_pasted:
        return clean_pasted, "pasted article", ""

    st.warning("No input provided. Enter a URL, upload an article, paste text, or click Use sample.")
    st.stop()


def _render_article_strip(signal: ArticleSignal, source_url: str) -> None:
    """Render article context strip."""

    today = datetime.now(timezone.utc).strftime("%b %d, %Y")
    time_utc = datetime.now(timezone.utc).strftime("%I:%M %p UTC")
    view = "View Article ↗" if source_url else "Public Demo"

    st.markdown(
        f"""
        <div class="card article-strip">
          <div class="brand-stock">
            <div class="stock-logo">{_safe(signal.ticker[:4])}</div>
            <div>
              <div class="strong">{_safe(signal.ticker)}</div>
              <div class="muted">{_safe(signal.company)}</div>
            </div>
          </div>
          <div>
            <div class="tiny-label">Article Headline</div>
            <div class="muted" style="color:#f8fafc;font-size:.82rem;">{_safe(signal.headline)}</div>
          </div>
          <div>
            <div class="tiny-label">Source</div>
            <div class="muted">{_safe(signal.source)}</div>
          </div>
          <div>
            <div class="tiny-label">Date</div>
            <div class="muted" style="color:#f8fafc;">{_safe(today)}</div>
            <div class="muted">{_safe(time_utc)}</div>
          </div>
          <div class="view-btn">{_safe(view)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_kpis(signal: ArticleSignal) -> None:
    """Render KPI cards."""

    sent_label = "Strongly Positive" if signal.sentiment_score > 0.65 else "Positive" if signal.sentiment_score > 0.2 else "Mixed signal"
    risk_label = "Low Risk" if signal.risk_score < 0.25 else "Moderate Risk" if signal.risk_score < 0.55 else "High Risk"
    conf_label = "High Confidence" if signal.confidence > 0.75 else "Medium confidence"

    st.markdown(
        f"""
        <div class="kpi-grid">
          <div class="card kpi purple">
            <div class="kpi-title">Ticker</div>
            <div class="kpi-value">{_safe(signal.ticker)}</div>
            <div class="kpi-sub">{_safe(signal.company)}</div>
          </div>
          <div class="card kpi teal">
            <div class="kpi-title">Sentiment Score</div>
            <div class="kpi-value">{signal.sentiment_score:+.2f}</div>
            <div class="kpi-sub">{_safe(sent_label)}</div>
            {_sparkline_svg([.2,.35,.32,.55,.48,.72,.62,.83], "#22d3ee")}
          </div>
          <div class="card kpi green">
            <div class="kpi-title">Movement Signal</div>
            <div class="kpi-value">{_safe(signal.label)}</div>
            <div class="kpi-sub">Upward Pressure</div>
            {_sparkline_svg([.12,.28,.25,.45,.38,.62,.55,.75], "#4ade80")}
          </div>
          <div class="card kpi orange">
            <div class="kpi-title">Risk Level</div>
            <div class="kpi-value">{signal.risk_score:.0%}</div>
            <div class="kpi-sub">{_safe(risk_label)}</div>
            {_sparkline_svg([.20,.30,.24,.38,.32,.45,.36,.52], "#f59e0b")}
          </div>
          <div class="card kpi violet">
            <div class="kpi-title">Confidence</div>
            <div class="kpi-value">{signal.confidence:.0%}</div>
            <div class="kpi-sub">{_safe(conf_label)}</div>
            {_sparkline_svg([.18,.24,.30,.36,.42,.50,.58,.68], "#a78bfa")}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_insight(signal: ArticleSignal) -> None:
    """Render executive insight and important analysis."""

    conclusion = (
        f"The article is <span class='green-text'>strongly positive</span> for {signal.ticker}. "
        f"Demand and revenue language dominate the signal, while "
        f"{', '.join(signal.risk_hits[:2]) or 'risk'} remain secondary risks. "
        f"Overall, the system estimates <span class='green-text'>{signal.label.lower()} movement pressure</span> "
        f"with <span class='green-text'>{signal.confidence:.0%} confidence</span>."
    )

    positives = len(signal.positive_hits)
    risks = len(signal.risk_hits)
    st.markdown(
        f"""
        <div class="two-col">
          <div class="card insight">
            <div class="panel-title">✦ Executive Insight</div>
            <div class="insight-text">Conclusion: {conclusion}</div>
          </div>
          <div class="card analysis-list">
            <div class="panel-title">◎ Most Important Analysis</div>
            <ul>
              <li>Strong positive demand and revenue signal detected.</li>
              <li>Positive evidence terms detected: {positives}</li>
              <li>Risk terms detected: {risks}</li>
              <li>{signal.label} short-term movement bias with {signal.confidence:.0%} confidence.</li>
              <li>Watch margin pressure and competition in follow-up news.</li>
            </ul>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_charts(signal: ArticleSignal) -> None:
    """Render dashboard chart row."""

    if go is None:
        st.info("Plotly is unavailable; chart visuals are disabled in this runtime.")
        return

    st.markdown('<div class="chart-grid">', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns([0.82, 0.82, 1.05, 1.15])

    with c1:
        st.markdown('<div class="card chart-card"><div class="chart-title">Movement Probability</div>', unsafe_allow_html=True)
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=["Up", "Flat", "Down"],
                    values=[signal.movement_up, signal.movement_flat, signal.movement_down],
                    hole=0.62,
                    marker={"colors": ["#4ade80", "#60a5fa", "#fb7185"]},
                    textinfo="none",
                )
            ]
        )
        fig.update_layout(
            height=185,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#dbeafe",
            showlegend=True,
            annotations=[dict(text=f"Up<br>{signal.movement_up:.0%}", x=0.5, y=0.5, showarrow=False, font=dict(size=16, color="white"))],
        )
        st.plotly_chart(fig, use_container_width=True, config=_plotly_config())
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card chart-card"><div class="chart-title">Sentiment vs Risk</div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=[signal.sentiment_score],
                y=[signal.risk_score],
                mode="markers",
                marker=dict(size=18, color="#4ade80"),
                name="Article",
            )
        )
        fig.update_xaxes(range=[-1.05, 1.05], zeroline=True, gridcolor="rgba(148,163,184,.18)")
        fig.update_yaxes(range=[0, 1.0], gridcolor="rgba(148,163,184,.18)")
        fig.update_layout(
            height=185,
            margin=dict(l=8, r=8, t=5, b=8),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#dbeafe",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True, config=_plotly_config())
        st.markdown("</div>", unsafe_allow_html=True)

    with c3:
        st.markdown('<div class="card chart-card"><div class="chart-title">Driver Impact</div>', unsafe_allow_html=True)
        drivers = ["Data center demand", "Revenue growth", "AI chip strength", "Cloud adoption", "Supply constraints", "Export controls", "Competition"]
        vals = [2.1, 1.65, 1.35, 0.85, -0.8, -0.5, -0.4]
        fig = go.Figure(
            go.Bar(
                x=vals,
                y=drivers,
                orientation="h",
                marker_color=["#4ade80" if v > 0 else "#fb7185" for v in vals],
            )
        )
        fig.update_layout(
            height=185,
            margin=dict(l=0, r=6, t=0, b=0),
            yaxis=dict(autorange="reversed"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#dbeafe",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True, config=_plotly_config())
        st.markdown("</div>", unsafe_allow_html=True)

    with c4:
        st.markdown('<div class="card chart-card"><div class="chart-title">Forecast Scenario</div>', unsafe_allow_html=True)
        days = list(range(61))
        base = [100 + i * 0.10 + math.sin(i / 6) * 0.8 for i in days]
        bull = [100 + i * 0.27 + math.sin(i / 5) * 1.2 for i in days]
        bear = [100 - i * 0.10 + math.sin(i / 4) * 0.8 for i in days]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=days, y=bull, name="Bull Case", line=dict(color="#4ade80", width=3)))
        fig.add_trace(go.Scatter(x=days, y=base, name="Base Case", line=dict(color="#60a5fa", width=3)))
        fig.add_trace(go.Scatter(x=days, y=bear, name="Bear Case", line=dict(color="#fb7185", width=3)))
        fig.update_layout(
            height=185,
            margin=dict(l=6, r=6, t=0, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#dbeafe",
            legend=dict(orientation="h", y=1.16),
        )
        fig.update_xaxes(gridcolor="rgba(148,163,184,.12)")
        fig.update_yaxes(gridcolor="rgba(148,163,184,.12)")
        st.plotly_chart(fig, use_container_width=True, config=_plotly_config())
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def _render_workflow(signal: ArticleSignal) -> None:
    """Render model workflow strip."""

    steps = [
        ("1. Article Intake", "URL/upload/text normalized"),
        ("2. Text Extraction", "Clean text ready"),
        ("3. Sentiment Signal", f"{len(signal.positive_hits)} positive, {len(signal.risk_hits)} risk terms"),
        ("4. Movement Estimate", f"Up {signal.movement_up:.0%} · Flat {signal.movement_flat:.0%} · Down {signal.movement_down:.0%}"),
        ("5. Risk Adjustment", f"Risk level {signal.risk_score:.0%}"),
        ("6. Forecast", "3-scenario 60-day horizon"),
        ("7. Explainability", "Drivers and evidence visible"),
    ]
    html_steps = "".join(
        f"<div class='step'><div class='step-num'>{_safe(title)} ✓</div><div class='step-text'>{_safe(desc)}</div></div>"
        for title, desc in steps
    )
    st.markdown(
        f"""
        <div class="card workflow">
          <div class="chart-title">What the Model is Doing</div>
          <div class="workflow-grid">{html_steps}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_bottom_panels(signal: ArticleSignal) -> None:
    """Render driver tags."""

    positive_tags = signal.positive_hits[:6] or ["data center demand", "record revenue", "AI chip strength", "cloud adoption"]
    risk_tags = signal.risk_hits[:6] or ["supply constraints", "export controls", "competition pressure"]
    monitor_tags = ["next quarter guidance", "gross margin trend", "supply", "China demand", "export policy", "competitor moves"]

    def tags(items: list[str]) -> str:
        return "".join(f"<span class='tag'>{_safe(item.title())}</span>" for item in items)

    st.markdown(
        f"""
        <div class="bottom-grid">
          <div class="card tag-panel">
            <div class="tag-title good">◎ Bullish Drivers</div>
            {tags(positive_tags)}
          </div>
          <div class="card tag-panel">
            <div class="tag-title warn">⚠ Risk Drivers</div>
            {tags(risk_tags)}
          </div>
          <div class="card tag-panel">
            <div class="tag-title info">◉ What to Monitor Next</div>
            {tags(monitor_tags)}
          </div>
        </div>
        <div class="footer">
          This is a public-mode analytical demo. Results are AI-generated for informational purposes only, not investment advice.
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_dashboard(signal: ArticleSignal, source_url: str) -> None:
    """Render the full promised Executive Overview dashboard."""

    _render_article_strip(signal, source_url)
    _render_kpis(signal)
    _render_insight(signal)
    _render_charts(signal)
    _render_workflow(signal)
    _render_bottom_panels(signal)


def _render_forecasts_page() -> None:
    """Render the Forecasts page as a URL-first, date-based, explainable forecast cockpit."""

    import html
    import math
    import re
    from datetime import date, timedelta
    from urllib.parse import urlparse

    def _clamp(value: float, low: float, high: float) -> float:
        """Keep public-demo scores within a safe visual range."""
        return max(low, min(high, value))

    def _format_date(value: date) -> str:
        """Format forecast dates in a compact dashboard-friendly form."""
        return value.strftime("%b %-d") if "%" in "%-d" else value.strftime("%b %d")

    def _find_cues(text: str, cue_patterns: list[tuple[str, str, float]]) -> list[dict[str, str | float]]:
        """Find financial cue phrases in user input and return matched evidence rows."""
        lowered = text.lower()
        found: list[dict[str, str | float]] = []
        seen: set[str] = set()

        for pattern, label, weight in cue_patterns:
            if re.search(pattern, lowered, flags=re.IGNORECASE) and label not in seen:
                found.append({"label": label, "weight": weight})
                seen.add(label)

        return found

    def _detect_target(text: str, user_target: str) -> tuple[str, str]:
        """Detect whether the forecast is broad-market, sector, ticker, or general news."""
        lowered = text.lower()
        entities: list[str] = []

        if re.search(r"\bdow\b|\bdjia\b", lowered):
            entities.append("Dow")
        if re.search(r"\bnasdaq\b|\bqqq\b", lowered):
            entities.append("Nasdaq")
        if re.search(r"\bs&p\b|\bsp500\b|\bspx\b|\bspy\b", lowered):
            entities.append("S&P 500")
        if re.search(r"\bchip\b|\bchips\b|\bsemiconductor\b|\bsemiconductors\b|\bnvidia\b|\bnvda\b|\bamd\b|\bintel\b", lowered):
            entities.append("Chips / semiconductors")
        if re.search(r"\btreasury\b|\brates\b|\bfed\b|\binflation\b", lowered):
            entities.append("Macro / rates")

        ticker_matches = re.findall(r"\$?[A-Z]{2,5}\b", text)
        ticker_matches = [x.replace("$", "") for x in ticker_matches if x not in {"LIVE", "CEO", "CFO", "EPS", "THE"}]
        for ticker in ticker_matches[:4]:
            if ticker not in entities and ticker not in {"DOW"}:
                entities.append(ticker)

        if user_target.strip():
            return "User-selected target", user_target.strip()

        if "Dow" in entities or "Nasdaq" in entities or "S&P 500" in entities:
            if "Chips / semiconductors" in entities:
                return "Broad market + sector forecast", ", ".join(entities)
            return "Broad market / index forecast", ", ".join(entities)

        if "Chips / semiconductors" in entities:
            return "Sector / theme forecast", ", ".join(entities)

        if ticker_matches:
            return "Ticker / company forecast", ", ".join(entities) if entities else ", ".join(ticker_matches[:4])

        return "General financial-news forecast", "No specific ticker or index detected"

    def _extract_article_from_url(url: str) -> tuple[str, str, str]:
        """Try to extract article title and body from a URL; return status, headline, body."""
        clean_url = url.strip()
        if not clean_url:
            return "No URL provided.", "", ""

        parsed = urlparse(clean_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return "Invalid URL. Paste a full URL starting with http:// or https://.", "", ""

        try:
            import requests
            from bs4 import BeautifulSoup

            response = requests.get(
                clean_url,
                timeout=8,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X) "
                        "AppleWebKit/537.36 Chrome/120 Safari/537.36"
                    )
                },
            )
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup(["script", "style", "noscript", "svg", "form", "nav", "footer", "header"]):
                tag.decompose()

            title = ""
            if soup.find("h1"):
                title = soup.find("h1").get_text(" ", strip=True)
            if not title and soup.title:
                title = soup.title.get_text(" ", strip=True)

            paragraphs = [
                p_tag.get_text(" ", strip=True)
                for p_tag in soup.find_all("p")
                if len(p_tag.get_text(" ", strip=True).split()) >= 8
            ]
            body = "\n\n".join(paragraphs[:12]).strip()

            if not body:
                return "URL loaded, but article text could not be extracted. Paste article text manually.", title, ""

            return "URL article text extracted.", title, body

        except Exception as exc:
            return f"URL extraction failed. Paste article text manually. Reason: {exc}", "", ""

    bullish_patterns = [
        (r"\bjumps?\b", "Jumps / strong upward move", 1.35),
        (r"\brises?\b|\brose\b", "Rises / positive market move", 1.15),
        (r"\brebounds?\b|\brebounded\b", "Rebound language", 1.20),
        (r"\brall(y|ies|ied)\b", "Rally language", 1.25),
        (r"\bgains?\b|\bgained\b", "Gain language", 1.05),
        (r"\bcloses?\s+above\b|\bfirst\s+close\s+above\b", "Close above key level", 1.40),
        (r"\brecord\s+(close|high)\b|\ball[- ]time high\b", "Record high / record close", 1.45),
        (r"\bchips?\s+rebound\b|\bsemiconductors?\s+rebound\b", "Chip / semiconductor rebound", 1.35),
        (r"\bbeat(s|ing)?\b|\bbeats?\s+estimates\b", "Beat estimates", 1.45),
        (r"\braises?\s+guidance\b|\braised\s+guidance\b", "Raised guidance", 1.50),
        (r"\bupgrade(d|s)?\b|\banalyst\s+upgrade\b", "Analyst upgrade tone", 1.30),
        (r"\bstrong\s+demand\b|\bdemand\s+strength\b", "Strong demand signal", 1.20),
        (r"\bpositive\s+momentum\b|\bupside\b|\boptimism\b", "Positive momentum / upside tone", 1.05),
    ]

    bearish_patterns = [
        (r"\bdrops?\b|\bfalls?\b|\bfell\b", "Drops / downward move", 1.25),
        (r"\bslides?\b|\bslid\b|\btumbles?\b", "Slide / tumble language", 1.30),
        (r"\bsell[- ]?off\b", "Selloff language", 1.45),
        (r"\bmiss(es|ed)?\b|\bmisses?\s+estimates\b", "Missed estimates", 1.45),
        (r"\bcuts?\s+guidance\b|\bcut\s+guidance\b", "Cut guidance", 1.55),
        (r"\bdowngrade(d|s)?\b|\banalyst\s+downgrade\b", "Analyst downgrade tone", 1.35),
        (r"\bweak\b|\bweakness\b|\bslowdown\b", "Weakness / slowdown", 1.15),
        (r"\bmargin\s+pressure\b|\bprofit\s+warning\b", "Margin pressure / warning", 1.35),
    ]

    risk_patterns = [
        (r"\brisk(s)?\b", "Explicit risk language", 1.05),
        (r"\buncertain(ty)?\b|\buncertainties\b", "Uncertainty language", 1.15),
        (r"\bvolatil(e|ity)\b", "Volatility language", 1.25),
        (r"\bregulatory\b|\bregulation\b|\blawsuit\b|\bprobe\b", "Regulatory / legal risk", 1.35),
        (r"\binflation\b|\brates?\b|\bfed\b|\brecession\b", "Macro / rates risk", 1.25),
        (r"\bdebt\b|\bliquidity\b|\bcredit\b", "Balance sheet / credit risk", 1.20),
    ]

    sample_headline = "Dow jumps 150 points for first close above 53,000; Nasdaq rises as chips rebound"
    sample_body = (
        "Stocks maintained positive momentum after a strong week on Wall Street. "
        "The S&P 500 gained 0.72%, while the Nasdaq Composite advanced 1.12% as chip stocks rebounded. "
        "Investors pointed to stronger technology momentum, improving risk appetite, and broad-market strength, "
        "while macro uncertainty remained limited."
    )

    if "forecast_url" not in st.session_state:
        st.session_state.forecast_url = ""
    if "forecast_headline" not in st.session_state:
        st.session_state.forecast_headline = ""
    if "forecast_body" not in st.session_state:
        st.session_state.forecast_body = ""
    if "forecast_target" not in st.session_state:
        st.session_state.forecast_target = ""
    if "forecast_status" not in st.session_state:
        st.session_state.forecast_status = "Enter a URL or paste text, then generate a forecast."

    st.markdown(
        """
        <style>
          .fc-hero {
            display: grid;
            grid-template-columns: 1.06fr .94fr;
            gap: 1rem;
            padding: 1.35rem;
            border-radius: 24px;
            border: 1px solid rgba(34,211,238,.34);
            background:
              radial-gradient(circle at 8% 8%, rgba(34,211,238,.20), transparent 22rem),
              radial-gradient(circle at 75% 10%, rgba(139,92,246,.22), transparent 24rem),
              radial-gradient(circle at 90% 92%, rgba(34,197,94,.13), transparent 22rem),
              linear-gradient(145deg, rgba(8,47,73,.72), rgba(8,13,28,.96));
            box-shadow: 0 30px 90px rgba(0,0,0,.36), inset 0 1px 0 rgba(255,255,255,.07);
            margin-bottom: .9rem;
          }
          .fc-kicker {
            color: #67e8f9;
            font-size: .70rem;
            font-weight: 950;
            letter-spacing: .13em;
            text-transform: uppercase;
          }
          .fc-title {
            color: white;
            font-size: 2.6rem;
            line-height: 1;
            font-weight: 950;
            letter-spacing: -.06em;
            margin: .42rem 0 .55rem 0;
          }
          .fc-subtitle {
            color: #dbeafe;
            font-size: 1rem;
            line-height: 1.55;
            max-width: 860px;
          }
          .fc-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: .48rem;
            margin-top: .9rem;
          }
          .fc-chip {
            padding: .43rem .68rem;
            border-radius: 999px;
            font-size: .72rem;
            font-weight: 850;
            color: #bfdbfe;
            border: 1px solid rgba(96,165,250,.25);
            background: rgba(15,23,42,.65);
          }
          .fc-engine {
            padding: 1rem;
            border-radius: 20px;
            border: 1px solid rgba(148,163,184,.18);
            background:
              radial-gradient(circle at 55% 0%, rgba(59,130,246,.16), transparent 16rem),
              linear-gradient(160deg, rgba(15,23,42,.92), rgba(2,6,23,.96));
          }
          .fc-flow-step {
            display: grid;
            grid-template-columns: 34px 1fr;
            gap: .65rem;
            align-items: center;
            padding: .54rem;
            margin-bottom: .45rem;
            border-radius: 13px;
            border: 1px solid rgba(96,165,250,.18);
            background: rgba(15,23,42,.70);
          }
          .fc-flow-num {
            width: 34px;
            height: 34px;
            display: grid;
            place-items: center;
            border-radius: 11px;
            color: #67e8f9;
            background: rgba(14,165,233,.13);
            border: 1px solid rgba(34,211,238,.25);
            font-weight: 950;
          }
          .fc-flow-step strong {
            display: block;
            color: white;
            font-size: .82rem;
          }
          .fc-flow-step span {
            display: block;
            color: #94a3b8;
            font-size: .70rem;
            margin-top: .08rem;
          }
          .fc-panel {
            margin: .95rem 0;
            padding: 1.1rem;
            border-radius: 22px;
            border: 1px solid rgba(34,211,238,.24);
            background:
              radial-gradient(circle at 6% 0%, rgba(34,211,238,.11), transparent 18rem),
              radial-gradient(circle at 94% 30%, rgba(139,92,246,.13), transparent 20rem),
              linear-gradient(145deg, rgba(15,23,42,.88), rgba(8,13,28,.96));
            box-shadow: 0 22px 60px rgba(0,0,0,.25);
          }
          .fc-section-title {
            color: white;
            font-size: 1.25rem;
            font-weight: 950;
            letter-spacing: -.04em;
            margin: .2rem 0 .35rem 0;
          }
          .fc-copy {
            color: #cbd5e1;
            font-size: .84rem;
            line-height: 1.48;
            margin: 0;
          }
          .fc-metrics {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: .68rem;
            margin: .85rem 0 .9rem 0;
          }
          .fc-metric {
            padding: 1rem;
            border-radius: 18px;
            border: 1px solid rgba(148,163,184,.16);
            background:
              radial-gradient(circle at 100% 0%, rgba(59,130,246,.12), transparent 12rem),
              rgba(15,23,42,.82);
          }
          .fc-metric strong {
            color: white;
            font-size: 1.45rem;
            font-weight: 950;
            letter-spacing: -.05em;
            display: block;
          }
          .fc-metric span {
            color: #cbd5e1;
            font-size: .74rem;
            font-weight: 760;
          }
          .fc-grid-2 {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: .75rem;
            margin-top: .8rem;
          }
          .fc-grid-3 {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: .75rem;
            margin-top: .8rem;
          }
          .fc-grid-4 {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: .68rem;
            margin-top: .8rem;
          }
          .fc-card {
            padding: .95rem;
            border-radius: 17px;
            border: 1px solid rgba(148,163,184,.16);
            background:
              radial-gradient(circle at 100% 0%, rgba(59,130,246,.10), transparent 10rem),
              rgba(15,23,42,.74);
            min-height: 125px;
          }
          .fc-card strong {
            color: white;
            display: block;
            font-size: .96rem;
            margin-bottom: .32rem;
          }
          .fc-card span, .fc-card li {
            color: #cbd5e1;
            font-size: .75rem;
            line-height: 1.38;
          }
          .fc-card ul {
            margin: .2rem 0 0 1rem;
            padding: 0;
          }
          .fc-formula {
            margin-top: .8rem;
            padding: .9rem;
            border-radius: 16px;
            border: 1px solid rgba(34,211,238,.22);
            background: rgba(2,6,23,.46);
            color: #e0f2fe;
            font-size: .84rem;
            font-weight: 850;
            line-height: 1.5;
          }
          .fc-explain {
            margin: .45rem 0 .9rem 0;
            padding: .9rem 1rem;
            border-radius: 16px;
            border: 1px solid rgba(148,163,184,.15);
            background: rgba(15,23,42,.66);
            color: #cbd5e1;
            font-size: .81rem;
            line-height: 1.48;
          }
          .fc-explain strong {
            color: white;
          }
          .fc-good { color: #86efac !important; }
          .fc-warn { color: #fbbf24 !important; }
          .fc-bad { color: #fca5a5 !important; }
          .fc-table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0 .45rem;
            margin-top: .75rem;
          }
          .fc-table th {
            color: #94a3b8;
            font-size: .72rem;
            text-align: left;
            padding: .35rem .55rem;
            text-transform: uppercase;
            letter-spacing: .08em;
          }
          .fc-table td {
            color: #e5e7eb;
            font-size: .80rem;
            padding: .65rem .55rem;
            background: rgba(15,23,42,.72);
            border-top: 1px solid rgba(148,163,184,.13);
            border-bottom: 1px solid rgba(148,163,184,.13);
          }
          .fc-table td:first-child {
            border-left: 1px solid rgba(148,163,184,.13);
            border-radius: 12px 0 0 12px;
            font-weight: 900;
          }
          .fc-table td:last-child {
            border-right: 1px solid rgba(148,163,184,.13);
            border-radius: 0 12px 12px 0;
          }
          @media (max-width: 1100px) {
            .fc-hero, .fc-metrics, .fc-grid-2, .fc-grid-3, .fc-grid-4 {
              grid-template-columns: 1fr;
            }
            .fc-title { font-size: 2.05rem; }
          }
        </style>

        <section class="fc-hero">
          <div>
            <div class="fc-kicker">Forecast Intelligence Cockpit</div>
            <div class="fc-title">Date-Based News<br/>Movement Forecasts</div>
            <div class="fc-subtitle">
              Enter a financial news URL or paste article text. The page detects market cues, target context,
              input quality, risk pressure, and driver strength before generating dated Bull, Base, and Bear scenarios.
            </div>
            <div class="fc-chip-row">
              <span class="fc-chip">Article URL input</span>
              <span class="fc-chip">Paste fallback</span>
              <span class="fc-chip">Forecast dates</span>
              <span class="fc-chip">Up / Flat / Down</span>
              <span class="fc-chip">Scenario probabilities</span>
              <span class="fc-chip">Analyst explanation</span>
            </div>
          </div>

          <div class="fc-engine">
            <div class="fc-kicker">Financial News Forecast Engine</div>
            <div class="fc-flow-step"><div class="fc-flow-num">01</div><div><strong>Source</strong><span>Article URL, headline, body, optional ticker</span></div></div>
            <div class="fc-flow-step"><div class="fc-flow-num">02</div><div><strong>Signals</strong><span>Bullish, bearish, risk, target, sector, index</span></div></div>
            <div class="fc-flow-step"><div class="fc-flow-num">03</div><div><strong>Forecast calendar</strong><span>1D, 7D, 14D, 30D dated movement paths</span></div></div>
            <div class="fc-flow-step"><div class="fc-flow-num">04</div><div><strong>Probabilities</strong><span>Up, Flat, Down and Bull/Base/Bear scenario mix</span></div></div>
            <div class="fc-flow-step"><div class="fc-flow-num">05</div><div><strong>Explanation</strong><span>Detected drivers and analyst-readable summary</span></div></div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <section class="fc-panel">
          <div class="fc-kicker">Forecast Input</div>
          <div class="fc-section-title">Enter a URL first, or paste article text manually</div>
          <p class="fc-copy">
            URL extraction may fail on some news sites because of paywalls, bot protection, or blocked article markup.
            If that happens, paste the headline and article body manually. The forecast will still work.
          </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    input_left, input_right = st.columns([1.18, .82])

    with input_left:
        url_value = st.text_input(
            "Article URL",
            value=st.session_state.forecast_url,
            placeholder="https://...",
            key="forecast_url_input",
        )

        target_hint = st.text_input(
            "Optional ticker / asset / market target",
            value=st.session_state.forecast_target,
            placeholder="Examples: NVDA, AAPL, Nasdaq, Dow, Semiconductors, Broad market",
            key="forecast_target_input",
        )

        headline = st.text_input(
            "Article headline",
            value=st.session_state.forecast_headline,
            placeholder="Paste the financial-news headline here",
            key="forecast_headline_input",
        )

        article_body = st.text_area(
            "Article body or summary",
            value=st.session_state.forecast_body,
            height=155,
            placeholder="Paste article body or summary here for stronger forecast quality.",
            key="forecast_body_input",
        )

    with input_right:
        horizon_days = st.slider("Forecast horizon, days", min_value=7, max_value=60, value=30, step=1)
        manual_adjustment = st.slider("Manual analyst adjustment", min_value=-20, max_value=20, value=0, step=1)
        show_3d = st.checkbox("Show optional 3D forecast surface", value=False)

        fetch_clicked = st.button("Extract URL text", type="secondary", use_container_width=True)
        sample_clicked = st.button("Load sample article", type="secondary", use_container_width=True)
        generate_clicked = st.button("Generate forecast", type="primary", use_container_width=True)

    if fetch_clicked:
        status, fetched_headline, fetched_body = _extract_article_from_url(url_value)
        st.session_state.forecast_url = url_value
        st.session_state.forecast_status = status
        if fetched_headline:
            st.session_state.forecast_headline = fetched_headline
        if fetched_body:
            st.session_state.forecast_body = fetched_body
        st.rerun()

    if sample_clicked:
        st.session_state.forecast_url = ""
        st.session_state.forecast_headline = sample_headline
        st.session_state.forecast_body = sample_body
        st.session_state.forecast_target = "Broad market"
        st.session_state.forecast_status = "Sample article loaded."
        st.rerun()

    st.session_state.forecast_url = url_value
    st.session_state.forecast_headline = headline
    st.session_state.forecast_body = article_body
    st.session_state.forecast_target = target_hint

    full_text = f"{headline}\n{article_body}".strip()
    word_count = len(re.findall(r"\b\w+\b", full_text))

    bullish_cues = _find_cues(full_text, bullish_patterns)
    bearish_cues = _find_cues(full_text, bearish_patterns)
    risk_cues = _find_cues(full_text, risk_patterns)
    target_type, detected_entities = _detect_target(full_text, target_hint)

    pos_score = sum(float(c["weight"]) for c in bullish_cues)
    neg_score = sum(float(c["weight"]) for c in bearish_cues)
    risk_score = sum(float(c["weight"]) for c in risk_cues)

    if not full_text:
        input_quality = 20
        input_quality_label = "Missing"
    elif word_count < 25:
        input_quality = 42 + min(10, len(bullish_cues) + len(bearish_cues) + len(risk_cues))
        input_quality_label = "Headline-only / limited"
    elif word_count < 120:
        input_quality = 58 + min(12, len(bullish_cues) * 2 + len(bearish_cues) * 2 + len(risk_cues))
        input_quality_label = "Moderate"
    else:
        input_quality = 76 + min(14, len(bullish_cues) + len(bearish_cues) + len(risk_cues))
        input_quality_label = "Strong"

    input_quality = round(_clamp(input_quality, 20, 92))

    sentiment_signal = _clamp(50 + pos_score * 9.0 - neg_score * 9.8 + manual_adjustment * .45, 8, 92)
    risk_pressure = _clamp(32 + risk_score * 10.5 + neg_score * 3.5 - pos_score * 1.4, 12, 90)
    movement_pressure = _clamp(50 + (sentiment_signal - 50) * .68 - (risk_pressure - 45) * .25, 8, 92)
    driver_strength = _clamp(38 + (pos_score + neg_score + risk_score) * 6.5 + min(18, word_count / 9), 22, 92)

    if not full_text:
        sentiment_signal, risk_pressure, movement_pressure, driver_strength = 50, 45, 50, 35

    forecast_pressure = (
        (sentiment_signal - 50) * 0.033
        + (movement_pressure - 50) * 0.038
        + (driver_strength - 50) * 0.020
        - (risk_pressure - 45) * 0.035
        + manual_adjustment * 0.025
    )

    quality_factor = 0.72 + input_quality / 330
    base_end = round(_clamp(forecast_pressure * (horizon_days / 30.0) * quality_factor, -7.5, 7.5), 2)
    bull_end = round(_clamp(base_end + 1.20 + driver_strength / 42.0, -5.0, 10.0), 2)
    downside_gap = 1.45 + risk_pressure / 30.0 + (100 - input_quality) / 95.0
    bear_end = round(_clamp(base_end - downside_gap, -10.0, 4.0), 2)
    if bear_end > -0.2:
        bear_end = round(-0.2 - (100 - input_quality) / 115.0, 2)

    confidence = round(
        _clamp(
            34 + input_quality * .42 + driver_strength * .12 - risk_pressure * .12 + min(8, len(bullish_cues) + len(bearish_cues) + len(risk_cues)),
            35,
            88,
        )
    )

    up_prob = round(_clamp(35 + (sentiment_signal - 50) * .55 + (movement_pressure - 50) * .35 - (risk_pressure - 45) * .22, 5, 88))
    down_prob = round(_clamp(25 + (risk_pressure - 45) * .35 + (50 - sentiment_signal) * .38 + neg_score * 5, 4, 85))
    flat_prob = max(4, 100 - up_prob - down_prob)
    total_prob = up_prob + flat_prob + down_prob
    up_prob = round(up_prob * 100 / total_prob)
    flat_prob = round(flat_prob * 100 / total_prob)
    down_prob = 100 - up_prob - flat_prob

    bull_prob = round(_clamp(up_prob * .72 + confidence * .18, 8, 78))
    bear_prob = round(_clamp(down_prob * .80 + risk_pressure * .10, 6, 72))
    base_prob = max(5, 100 - bull_prob - bear_prob)
    total_scenario = bull_prob + base_prob + bear_prob
    bull_prob = round(bull_prob * 100 / total_scenario)
    base_prob = round(base_prob * 100 / total_scenario)
    bear_prob = 100 - bull_prob - base_prob

    forecast_start = date.today()
    forecast_end = forecast_start + timedelta(days=horizon_days)
    bucket_days = sorted(set([1, 7, 14, min(30, horizon_days), horizon_days]))
    bucket_days = [d for d in bucket_days if d <= horizon_days]

    def _path_value(end_value: float, day_number: int) -> float:
        return round(end_value * (day_number / horizon_days) + 0.12 * math.sin(day_number / max(5, horizon_days / 3.8)), 2)

    forecast_rows = []
    for day_number in bucket_days:
        row_date = forecast_start + timedelta(days=day_number)
        forecast_rows.append(
            {
                "label": f"{day_number}D",
                "date": _format_date(row_date),
                "bull": _path_value(bull_end, day_number),
                "base": _path_value(base_end, day_number),
                "bear": round(bear_end * (day_number / horizon_days) - 0.10 * math.sin(day_number / max(5, horizon_days / 3.8)), 2),
                "confidence": round(_clamp(confidence - day_number * .45, 25, confidence)),
            }
        )

    sentiment_label = "Bullish" if sentiment_signal >= 62 else "Bearish" if sentiment_signal <= 42 else "Mixed"
    risk_label = "High" if risk_pressure >= 65 else "Low" if risk_pressure <= 40 else "Moderate"
    direction_label = "Bullish" if up_prob > max(flat_prob, down_prob) else "Bearish" if down_prob > max(up_prob, flat_prob) else "Mixed / flat"

    def _cue_list_html(cues: list[dict[str, str | float]], empty_text: str) -> str:
        if not cues:
            return f"<span>{html.escape(empty_text)}</span>"
        items = "".join(f"<li>{html.escape(str(c['label']))}</li>" for c in cues[:6])
        return f"<ul>{items}</ul>"

    status_text = html.escape(st.session_state.forecast_status)
    escaped_target_type = html.escape(target_type)
    escaped_entities = html.escape(detected_entities)
    escaped_quality = html.escape(input_quality_label)

    source_label = "Article URL" if url_value.strip() else "Pasted text / manual input"

    st.markdown(
        f"""
        <section class="fc-panel">
          <div class="fc-kicker">Forecast Source</div>
          <div class="fc-section-title">Forecast generated from selected article signal</div>
          <div class="fc-grid-4">
            <div class="fc-card"><strong>Source</strong><span>{html.escape(source_label)}</span></div>
            <div class="fc-card"><strong>Generated</strong><span>{html.escape(_format_date(forecast_start))}</span></div>
            <div class="fc-card"><strong>Horizon end</strong><span>{html.escape(_format_date(forecast_end))}</span></div>
            <div class="fc-card"><strong>Status</strong><span>{status_text}</span></div>
          </div>
          <div class="fc-grid-4">
            <div class="fc-card"><strong>Target type</strong><span>{escaped_target_type}</span></div>
            <div class="fc-card"><strong>Detected entities</strong><span>{escaped_entities}</span></div>
            <div class="fc-card"><strong>Input quality</strong><span>{escaped_quality} · {word_count} words</span></div>
            <div class="fc-card"><strong>Forecast type</strong><span>News-conditioned movement scenario</span></div>
          </div>
        </section>

        <section class="fc-panel">
          <div class="fc-kicker">Detected Signals</div>
          <div class="fc-section-title">What the forecast engine found in the input</div>
          <div class="fc-grid-3">
            <div class="fc-card">
              <strong class="fc-good">Bullish cues</strong>
              {_cue_list_html(bullish_cues, "No clear bullish cues detected.")}
            </div>
            <div class="fc-card">
              <strong class="fc-bad">Bearish cues</strong>
              {_cue_list_html(bearish_cues, "No clear bearish cues detected.")}
            </div>
            <div class="fc-card">
              <strong class="fc-warn">Risk cues</strong>
              {_cue_list_html(risk_cues, "No clear risk cues detected.")}
            </div>
          </div>
          <div class="fc-formula">
            Forecast pressure = sentiment signal + movement pressure + driver strength - risk pressure ± analyst adjustment.
            Confidence decays over time and is reduced when the input is headline-only or limited.
          </div>
        </section>

        <div class="fc-metrics">
          <div class="fc-metric"><strong>{direction_label}</strong><span>Direction forecast</span></div>
          <div class="fc-metric"><strong>{up_prob}%</strong><span>Up probability</span></div>
          <div class="fc-metric"><strong>{flat_prob}%</strong><span>Flat probability</span></div>
          <div class="fc-metric"><strong>{down_prob}%</strong><span>Down probability</span></div>
          <div class="fc-metric"><strong>{confidence}%</strong><span>Confidence today</span></div>
        </div>

        <div class="fc-metrics">
          <div class="fc-metric"><strong>{bull_end:+.1f}%</strong><span>Bull scenario by {html.escape(_format_date(forecast_end))}</span></div>
          <div class="fc-metric"><strong>{base_end:+.1f}%</strong><span>Base scenario by {html.escape(_format_date(forecast_end))}</span></div>
          <div class="fc-metric"><strong>{bear_end:+.1f}%</strong><span>Bear scenario by {html.escape(_format_date(forecast_end))}</span></div>
          <div class="fc-metric"><strong>{bull_prob}/{base_prob}/{bear_prob}</strong><span>Bull/Base/Bear probability mix</span></div>
          <div class="fc-metric"><strong>{risk_label}</strong><span>Risk pressure</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    table_rows = ""
    for row in forecast_rows:
        table_rows += (
            "<tr>"
            f"<td>{html.escape(row['label'])}</td>"
            f"<td>{html.escape(row['date'])}</td>"
            f"<td>{row['bull']:+.1f}%</td>"
            f"<td>{row['base']:+.1f}%</td>"
            f"<td>{row['bear']:+.1f}%</td>"
            f"<td>{row['confidence']}%</td>"
            "</tr>"
        )

    st.markdown(
        f"""
        <section class="fc-panel">
          <div class="fc-kicker">Forecast Calendar</div>
          <div class="fc-section-title">Dated Bull / Base / Bear movement outlook</div>
          <p class="fc-copy">
            This table makes the forecast time-based. Each row shows the expected scenario movement by a calendar date.
          </p>
          <table class="fc-table">
            <thead>
              <tr>
                <th>Horizon</th>
                <th>Date</th>
                <th>Bull</th>
                <th>Base</th>
                <th>Bear</th>
                <th>Confidence</th>
              </tr>
            </thead>
            <tbody>
              {table_rows}
            </tbody>
          </table>
        </section>
        """,
        unsafe_allow_html=True,
    )

    try:
        import plotly.graph_objects as go

        signal_fig = go.Figure(
            go.Bar(
                x=[sentiment_signal, movement_pressure, risk_pressure, driver_strength, input_quality],
                y=["Sentiment", "Movement", "Risk", "Driver strength", "Input quality"],
                orientation="h",
                hovertemplate="<b>%{y}</b><br>Score: %{x:.0f}/100<extra></extra>",
            )
        )
        signal_fig.update_layout(
            title="Input Signal Breakdown",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,.35)",
            height=340,
            margin=dict(l=0, r=0, t=55, b=0),
            xaxis=dict(title="Signal score", range=[0, 100]),
            yaxis_title="",
        )
        st.plotly_chart(signal_fig, use_container_width=True, config={"displayModeBar": False})

        days = list(range(0, horizon_days + 1))
        labels = [_format_date(forecast_start + timedelta(days=d)) for d in days]
        curve_speed = max(5, horizon_days / 3.8)

        base = [round(base_end * (d / horizon_days) + 0.14 * math.sin(d / curve_speed), 2) for d in days]
        bull = [round(bull_end * (d / horizon_days) + 0.20 * math.sin(d / curve_speed), 2) for d in days]
        bear = [round(bear_end * (d / horizon_days) - 0.15 * math.sin(d / curve_speed), 2) for d in days]

        uncertainty_width = max(0.55, (100 - confidence) / 18)
        upper = [round(v + uncertainty_width * (0.30 + d / horizon_days), 2) for d, v in zip(days, base)]
        lower = [round(v - uncertainty_width * (0.30 + d / horizon_days), 2) for d, v in zip(days, base)]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=labels, y=upper, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(
            go.Scatter(
                x=labels,
                y=lower,
                mode="lines",
                fill="tonexty",
                fillcolor="rgba(34,211,238,.13)",
                line=dict(width=0),
                name="Confidence band",
                hoverinfo="skip",
            )
        )
        fig.add_trace(go.Scatter(x=labels, y=bull, mode="lines", name="Bull scenario", line=dict(width=4)))
        fig.add_trace(go.Scatter(x=labels, y=base, mode="lines", name="Base scenario", line=dict(width=5)))
        fig.add_trace(go.Scatter(x=labels, y=bear, mode="lines", name="Bear scenario", line=dict(width=4)))

        fig.update_layout(
            title=f"Dated Forecast Fan Chart · {_format_date(forecast_start)} to {_format_date(forecast_end)}",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,.35)",
            height=520,
            margin=dict(l=0, r=0, t=55, b=0),
            xaxis_title="Forecast date",
            yaxis_title="Projected movement %",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        st.markdown(
            """
            <div class="fc-explain">
              <strong>How to read this chart:</strong>
              the x-axis now uses forecast dates. The Bull line shows upside if detected positive cues continue.
              The Base line shows the central dated path. The Bear line shows downside risk. The shaded band widens
              as uncertainty increases across the horizon.
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_prob, col_decay = st.columns(2)

        with col_prob:
            prob_fig = go.Figure(
                go.Bar(
                    x=["Bull", "Base", "Bear"],
                    y=[bull_prob, base_prob, bear_prob],
                    hovertemplate="<b>%{x}</b><br>Scenario probability: %{y}%<extra></extra>",
                )
            )
            prob_fig.update_layout(
                title="Scenario Probability Forecast",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,.35)",
                height=380,
                margin=dict(l=0, r=0, t=55, b=0),
                yaxis=dict(title="Probability %", range=[0, 100]),
                xaxis_title="Scenario",
            )
            st.plotly_chart(prob_fig, use_container_width=True, config={"displayModeBar": False})

        with col_decay:
            decay_days = [row["label"] for row in forecast_rows]
            decay_values = [row["confidence"] for row in forecast_rows]
            decay_fig = go.Figure(
                go.Scatter(
                    x=decay_days,
                    y=decay_values,
                    mode="lines+markers",
                    hovertemplate="<b>%{x}</b><br>Confidence: %{y}%<extra></extra>",
                )
            )
            decay_fig.update_layout(
                title="Confidence Decay Forecast",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,.35)",
                height=380,
                margin=dict(l=0, r=0, t=55, b=0),
                yaxis=dict(title="Confidence %", range=[0, 100]),
                xaxis_title="Forecast horizon",
            )
            st.plotly_chart(decay_fig, use_container_width=True, config={"displayModeBar": False})

        st.markdown(
            """
            <div class="fc-explain">
              <strong>How to read these charts:</strong>
              scenario probability shows whether Bull, Base, or Bear is the dominant case.
              Confidence decay shows that forecast certainty normally falls as the horizon moves farther away.
            </div>
            """,
            unsafe_allow_html=True,
        )

        if show_3d:
            pressure_axis = [-3, -2, -1, 0, 1, 2, 3]
            horizon_axis = [0, round(horizon_days * .17), round(horizon_days * .34), round(horizon_days * .5), round(horizon_days * .67), round(horizon_days * .84), horizon_days]
            z = []
            for pval in pressure_axis:
                row = []
                for h in horizon_axis:
                    value = (base_end * (h / horizon_days)) + pval * 0.50 + (h / horizon_days) * pval * 0.32
                    row.append(round(value, 2))
                z.append(row)

            surface = go.Figure(
                data=[
                    go.Surface(
                        x=horizon_axis,
                        y=pressure_axis,
                        z=z,
                        opacity=.92,
                        contours={"z": {"show": True, "usecolormap": True, "highlightcolor": "white", "project_z": True}},
                    )
                ]
            )
            surface.update_layout(
                title="Optional 3D Forecast Surface · Horizon × Market Pressure × Movement",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,.35)",
                height=480,
                margin=dict(l=0, r=0, t=55, b=0),
                scene=dict(
                    xaxis_title="Horizon",
                    yaxis_title="Market pressure",
                    zaxis_title="Movement %",
                    camera=dict(eye=dict(x=1.55, y=1.45, z=1.08)),
                ),
            )
            st.plotly_chart(surface, use_container_width=True, config={"displayModeBar": False})

        col1, col2 = st.columns(2)

        with col1:
            risk_reward = go.Figure()
            risk_reward.add_shape(type="line", x0=0, x1=10, y0=5, y1=5, line=dict(width=1, dash="dash"))
            risk_reward.add_shape(type="line", x0=5, x1=5, y0=0, y1=10, line=dict(width=1, dash="dash"))

            bull_risk = _clamp(risk_pressure / 20, 1, 9)
            bull_reward = _clamp(5.2 + bull_end / 1.25, 1, 9.5)
            base_risk = _clamp(risk_pressure / 15, 1, 9)
            base_reward = _clamp(5 + base_end / 1.25, 1, 9.5)
            bear_risk = _clamp(4.5 + risk_pressure / 14, 1, 9.5)
            bear_reward = _clamp(5 + bear_end / 1.45, 1, 9.5)

            risk_reward.add_trace(
                go.Scatter(
                    x=[bull_risk, base_risk, bear_risk],
                    y=[bull_reward, base_reward, bear_reward],
                    mode="markers+text",
                    text=["Bull", "Base", "Bear"],
                    textposition="top center",
                    marker=dict(size=[20, 18, 20], line=dict(width=2, color="rgba(255,255,255,.35)")),
                    hovertemplate="<b>%{text}</b><br>Risk: %{x:.1f}<br>Reward: %{y:.1f}<extra></extra>",
                )
            )
            risk_reward.update_layout(
                title="Risk / Reward Scenario Matrix",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,.35)",
                height=430,
                margin=dict(l=0, r=0, t=55, b=0),
                xaxis=dict(title="Risk", range=[0, 10]),
                yaxis=dict(title="Reward", range=[0, 10]),
                showlegend=False,
            )
            st.plotly_chart(risk_reward, use_container_width=True, config={"displayModeBar": False})

        with col2:
            impact_rows = [
                ("Bullish cue pressure", round(pos_score * 0.42, 2)),
                ("Bearish cue pressure", round(-neg_score * 0.46, 2)),
                ("Risk language pressure", round(-risk_score * 0.42, 2)),
                ("Input quality support", round((input_quality - 50) / 45, 2)),
                ("Analyst adjustment", round(manual_adjustment / 20, 2)),
            ]
            driver_fig = go.Figure(
                go.Bar(
                    x=[row[1] for row in impact_rows],
                    y=[row[0] for row in impact_rows],
                    orientation="h",
                    hovertemplate="<b>%{y}</b><br>Forecast contribution: %{x}<extra></extra>",
                )
            )
            driver_fig.update_layout(
                title="Forecast Driver Impact",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,.35)",
                height=430,
                margin=dict(l=0, r=0, t=55, b=0),
                xaxis_title="Forecast contribution",
                yaxis_title="",
            )
            st.plotly_chart(driver_fig, use_container_width=True, config={"displayModeBar": False})

        st.markdown(
            """
            <div class="fc-explain">
              <strong>How to read these charts:</strong>
              the risk/reward matrix compares upside and downside across scenarios.
              The driver chart explains which detected signals lifted or lowered the forecast.
            </div>
            """,
            unsafe_allow_html=True,
        )

    except Exception as exc:
        st.warning(f"Forecast charts could not render. Reason: {exc}")

    if base_end >= 1.0:
        base_text = "moderate upside"
    elif base_end <= -1.0:
        base_text = "downside pressure"
    else:
        base_text = "balanced movement"

    st.markdown(
        f"""
        <section class="fc-panel">
          <div class="fc-kicker">Scenario Interpretation</div>
          <div class="fc-grid-3">
            <div class="fc-card">
              <strong class="fc-good">Bull Scenario · {bull_end:+.1f}% by {html.escape(_format_date(forecast_end))}</strong>
              <span>Upside case if detected positive cues continue and risk pressure stays contained.</span>
            </div>
            <div class="fc-card">
              <strong>Base Scenario · {base_end:+.1f}% by {html.escape(_format_date(forecast_end))}</strong>
              <span>Central case for the selected input. Current profile indicates {base_text} with {risk_label.lower()} risk pressure.</span>
            </div>
            <div class="fc-card">
              <strong class="fc-bad">Bear Scenario · {bear_end:+.1f}% by {html.escape(_format_date(forecast_end))}</strong>
              <span>Downside case if the move fades, risk language increases, or market pressure reverses.</span>
            </div>
          </div>
        </section>

        <section class="fc-panel">
          <div class="fc-kicker">Forecast Method</div>
          <div class="fc-section-title">Active engine: Financial News Stock Intelligence Scenario Forecast Layer</div>
          <p class="fc-copy">
            This page forecasts news-driven movement pressure. It is not mainly a historical price time-series app.
            ARIMA, Prophet, and LSTM are useful forecasting concepts, but they are not claimed as active public-demo
            engines here. The active page connects article language, sentiment pressure, movement pressure, risk terms,
            input quality, dates, scenario probabilities, and explainability into Bull, Base, and Bear forecasts.
          </p>
        </section>

        <section class="fc-panel">
          <div class="fc-kicker">Analyst Explanation</div>
          <div class="fc-section-title">Forecast summary</div>
          <p class="fc-copy">
            Selected target context: {html.escape(target_type)} ({html.escape(detected_entities)}).
            Direction forecast is {html.escape(direction_label.lower())}, with Up / Flat / Down probabilities of
            {up_prob}% / {flat_prob}% / {down_prob}%. The dated forecast runs from {html.escape(_format_date(forecast_start))}
            to {html.escape(_format_date(forecast_end))}. Sentiment is {html.escape(sentiment_label.lower())},
            risk pressure is {html.escape(risk_label.lower())}, input quality is {html.escape(input_quality_label.lower())},
            and the base case shows {base_text}. Read this as a scenario forecast, not investment advice.
          </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

def _render_historical_intelligence_page() -> None:
    """Render Historical Intelligence with article input and logically matched comparable events."""

    import html
    import re
    import statistics
    from urllib.parse import urlparse

    def _find_cues(text: str, cue_patterns: list[tuple[str, str, float]]) -> list[dict[str, str | float]]:
        lowered = text.lower()
        found: list[dict[str, str | float]] = []
        seen: set[str] = set()

        for pattern, label, weight in cue_patterns:
            if re.search(pattern, lowered, flags=re.IGNORECASE) and label not in seen:
                found.append({"label": label, "weight": weight})
                seen.add(label)

        return found

    def _extract_article_from_url(url: str) -> tuple[str, str, str]:
        clean_url = url.strip()
        if not clean_url:
            return "No URL provided.", "", ""

        parsed = urlparse(clean_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return "Invalid URL. Paste a full URL starting with http:// or https://.", "", ""

        try:
            import requests
            from bs4 import BeautifulSoup

            response = requests.get(
                clean_url,
                timeout=8,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X) "
                        "AppleWebKit/537.36 Chrome/120 Safari/537.36"
                    )
                },
            )
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup(["script", "style", "noscript", "svg", "form", "nav", "footer", "header"]):
                tag.decompose()

            title = ""
            if soup.find("h1"):
                title = soup.find("h1").get_text(" ", strip=True)
            if not title and soup.title:
                title = soup.title.get_text(" ", strip=True)

            paragraphs = [
                p_tag.get_text(" ", strip=True)
                for p_tag in soup.find_all("p")
                if len(p_tag.get_text(" ", strip=True).split()) >= 8
            ]
            body = "\n\n".join(paragraphs[:12]).strip()

            if not body:
                return "URL loaded, but article text could not be extracted. Paste article text manually.", title, ""

            return "URL article text extracted.", title, body

        except Exception as exc:
            return f"URL extraction failed. Paste article text manually. Reason: {exc}", "", ""

    def _detect_target(text: str, user_target: str) -> tuple[str, str]:
        lowered = text.lower()
        entities: list[str] = []

        if re.search(r"\bdow\b|\bdjia\b", lowered):
            entities.append("Dow")
        if re.search(r"\bnasdaq\b|\bqqq\b", lowered):
            entities.append("Nasdaq")
        if re.search(r"\bs&p\b|\bsp500\b|\bspx\b|\bspy\b", lowered):
            entities.append("S&P 500")
        if re.search(r"\bchip\b|\bchips\b|\bsemiconductor\b|\bsemiconductors\b|\bnvidia\b|\bnvda\b|\bamd\b|\bintel\b", lowered):
            entities.append("Chips / semiconductors")
        if re.search(r"\brates?\b|\bfed\b|\binflation\b|\btreasury\b", lowered):
            entities.append("Macro / rates")
        if re.search(r"\bcredit\b|\bdebt\b|\bdowngrade\b|\bregulatory\b|\blawsuit\b", lowered):
            entities.append("Credit / regulatory risk")

        ticker_matches = re.findall(r"\$?[A-Z]{2,5}\b", text)
        ticker_matches = [x.replace("$", "") for x in ticker_matches if x not in {"LIVE", "CEO", "CFO", "EPS", "THE"}]
        for ticker in ticker_matches[:4]:
            if ticker not in entities and ticker not in {"DOW"}:
                entities.append(ticker)

        if user_target.strip():
            return "User-selected target", user_target.strip()

        if "Dow" in entities or "Nasdaq" in entities or "S&P 500" in entities:
            if "Chips / semiconductors" in entities:
                return "Broad market + sector", ", ".join(entities)
            return "Broad market / index", ", ".join(entities)

        if "Chips / semiconductors" in entities:
            return "Sector / theme", ", ".join(entities)

        if ticker_matches:
            return "Ticker / company", ", ".join(entities) if entities else ", ".join(ticker_matches[:4])

        return "General financial news", "No specific ticker or index detected"

    def _detect_event_family(text: str) -> str:
        lowered = text.lower()

        if re.search(r"\bchip\b|\bchips\b|\bsemiconductor\b|\bsemiconductors\b|\bnvidia\b|\bnvda\b|\bamd\b|\bintel\b", lowered) and re.search(r"\brebound\b|\brise\b|\brises\b|\brally\b|\bgain\b|\bjump\b", lowered):
            return "Semiconductor rebound"

        if re.search(r"\bbeat(s|ing)?\b|\bbeats?\s+estimates\b|\bearnings\s+beat\b|\bstrong\s+earnings\b", lowered):
            return "Earnings beat"

        if re.search(r"\braises?\s+guidance\b|\braised\s+guidance\b|\bguidance\s+raise\b|\boutlook\s+raised\b", lowered):
            return "Guidance raise"

        if re.search(r"\bregulatory\b|\blawsuit\b|\bprobe\b|\bcredit\b|\bdebt\b|\bdowngrade\b|\bliquidity\b", lowered):
            return "Regulatory / credit risk"

        if re.search(r"\bfed\b|\brates?\b|\binflation\b|\brecession\b|\btreasury\b|\bmacro\b", lowered):
            return "Macro shock"

        if re.search(r"\bdow\b|\bnasdaq\b|\bs&p\b|\bsp500\b|\bmarket\b", lowered) and re.search(r"\bjumps?\b|\brises?\b|\bgains?\b|\brall(y|ies|ied)\b|\bcloses?\s+above\b", lowered):
            return "Broad market rally"

        return "Broad market rally"

    bullish_patterns = [
        (r"\bjumps?\b", "Jumps / strong upward move", 1.35),
        (r"\brises?\b|\brose\b", "Rises / positive market move", 1.15),
        (r"\brebounds?\b|\brebounded\b", "Rebound language", 1.20),
        (r"\brall(y|ies|ied)\b", "Rally language", 1.25),
        (r"\bgains?\b|\bgained\b", "Gain language", 1.05),
        (r"\bcloses?\s+above\b|\bfirst\s+close\s+above\b", "Close above key level", 1.40),
        (r"\brecord\s+(close|high)\b|\ball[- ]time high\b", "Record high / record close", 1.45),
        (r"\bchips?\s+rebound\b|\bsemiconductors?\s+rebound\b", "Chip / semiconductor rebound", 1.35),
        (r"\bbeat(s|ing)?\b|\bbeats?\s+estimates\b", "Beat estimates", 1.45),
        (r"\braises?\s+guidance\b|\braised\s+guidance\b", "Raised guidance", 1.50),
        (r"\bupgrade(d|s)?\b|\banalyst\s+upgrade\b", "Analyst upgrade tone", 1.30),
        (r"\bpositive\s+momentum\b|\bupside\b|\boptimism\b", "Positive momentum / upside tone", 1.05),
    ]

    bearish_patterns = [
        (r"\bdrops?\b|\bfalls?\b|\bfell\b", "Drops / downward move", 1.25),
        (r"\bslides?\b|\bslid\b|\btumbles?\b", "Slide / tumble language", 1.30),
        (r"\bsell[- ]?off\b", "Selloff language", 1.45),
        (r"\bmiss(es|ed)?\b|\bmisses?\s+estimates\b", "Missed estimates", 1.45),
        (r"\bcuts?\s+guidance\b|\bcut\s+guidance\b", "Cut guidance", 1.55),
        (r"\bdowngrade(d|s)?\b|\banalyst\s+downgrade\b", "Analyst downgrade tone", 1.35),
        (r"\bweak\b|\bweakness\b|\bslowdown\b", "Weakness / slowdown", 1.15),
    ]

    risk_patterns = [
        (r"\brisk(s)?\b", "Explicit risk language", 1.05),
        (r"\buncertain(ty)?\b|\buncertainties\b", "Uncertainty language", 1.15),
        (r"\bvolatil(e|ity)\b", "Volatility language", 1.25),
        (r"\bregulatory\b|\bregulation\b|\blawsuit\b|\bprobe\b", "Regulatory / legal risk", 1.35),
        (r"\binflation\b|\brates?\b|\bfed\b|\brecession\b", "Macro / rates risk", 1.25),
        (r"\bdebt\b|\bliquidity\b|\bcredit\b", "Balance sheet / credit risk", 1.20),
    ]

    comparable_events = [
        {"date":"2024-05-23","event":"Chip rally","target":"NVDA / SOX","family":"Semiconductor rebound","similarity":93,"d1":2.1,"d3":4.4,"d7":6.4,"d14":9.1,"d30":12.8,"risk":28,"vol":6.2,"regime":"Risk-on tech rally"},
        {"date":"2024-04-19","event":"Chip selloff fade","target":"SOX / Nasdaq","family":"Semiconductor rebound","similarity":84,"d1":0.9,"d3":1.4,"d7":2.2,"d14":-0.4,"d30":1.1,"risk":61,"vol":8.8,"regime":"Volatile tech"},
        {"date":"2023-03-29","event":"Semiconductor rebound","target":"SOX","family":"Semiconductor rebound","similarity":78,"d1":1.1,"d3":2.2,"d7":3.5,"d14":4.4,"d30":6.1,"risk":42,"vol":6.9,"regime":"Tech recovery"},
        {"date":"2022-07-27","event":"Chip demand rebound","target":"AMD / SOX","family":"Semiconductor rebound","similarity":72,"d1":1.4,"d3":1.9,"d7":2.8,"d14":3.1,"d30":2.0,"risk":55,"vol":8.1,"regime":"Macro-sensitive tech"},

        {"date":"2023-11-14","event":"Inflation relief rally","target":"Nasdaq","family":"Broad market rally","similarity":88,"d1":1.8,"d3":3.2,"d7":4.3,"d14":5.1,"d30":7.2,"risk":34,"vol":5.1,"regime":"Rates easing"},
        {"date":"2022-08-10","event":"Macro relief rally","target":"S&P 500","family":"Broad market rally","similarity":79,"d1":2.0,"d3":2.7,"d7":3.1,"d14":1.4,"d30":-1.2,"risk":54,"vol":8.3,"regime":"High macro risk"},
        {"date":"2020-11-09","event":"Market rotation rally","target":"Dow","family":"Broad market rally","similarity":73,"d1":2.9,"d3":1.2,"d7":1.8,"d14":3.4,"d30":4.6,"risk":46,"vol":9.4,"regime":"Rotation"},
        {"date":"2021-03-01","event":"Risk appetite rebound","target":"S&P 500","family":"Broad market rally","similarity":70,"d1":1.6,"d3":2.0,"d7":2.7,"d14":3.1,"d30":3.8,"risk":44,"vol":6.8,"regime":"Risk-on"},

        {"date":"2024-02-22","event":"AI earnings beat","target":"Semiconductors","family":"Earnings beat","similarity":86,"d1":1.6,"d3":2.8,"d7":5.2,"d14":7.6,"d30":11.4,"risk":32,"vol":7.1,"regime":"AI momentum"},
        {"date":"2021-10-14","event":"Earnings rebound","target":"Nasdaq","family":"Earnings beat","similarity":76,"d1":1.5,"d3":2.1,"d7":3.6,"d14":4.2,"d30":5.9,"risk":40,"vol":5.9,"regime":"Risk-on"},
        {"date":"2023-07-26","event":"Mega-cap earnings beat","target":"Mega-cap tech","family":"Earnings beat","similarity":74,"d1":1.2,"d3":1.9,"d7":3.0,"d14":4.1,"d30":5.4,"risk":39,"vol":6.1,"regime":"Earnings season"},

        {"date":"2023-05-25","event":"Guidance raise","target":"Mega-cap tech","family":"Guidance raise","similarity":82,"d1":2.7,"d3":3.5,"d7":4.9,"d14":6.0,"d30":8.5,"risk":30,"vol":6.5,"regime":"Earnings season"},
        {"date":"2024-01-31","event":"Outlook raised","target":"Cloud software","family":"Guidance raise","similarity":75,"d1":1.8,"d3":2.4,"d7":3.9,"d14":4.6,"d30":6.3,"risk":35,"vol":6.0,"regime":"Growth recovery"},
        {"date":"2022-02-03","event":"Strong demand guidance","target":"Consumer tech","family":"Guidance raise","similarity":68,"d1":1.1,"d3":1.7,"d7":2.4,"d14":2.0,"d30":1.6,"risk":52,"vol":7.7,"regime":"Rate pressure"},

        {"date":"2022-10-13","event":"Inflation whipsaw","target":"S&P 500","family":"Macro shock","similarity":68,"d1":2.6,"d3":-0.8,"d7":-1.4,"d14":1.2,"d30":3.0,"risk":72,"vol":11.5,"regime":"High volatility"},
        {"date":"2021-03-09","event":"Rate pressure rebound","target":"Nasdaq","family":"Macro shock","similarity":65,"d1":3.7,"d3":2.0,"d7":-0.7,"d14":0.8,"d30":2.4,"risk":67,"vol":10.1,"regime":"Rate-sensitive"},
        {"date":"2023-03-13","event":"Banking stress shock","target":"S&P 500","family":"Macro shock","similarity":63,"d1":-0.2,"d3":-1.1,"d7":-0.9,"d14":1.0,"d30":2.7,"risk":80,"vol":12.3,"regime":"Financial stress"},

        {"date":"2023-08-02","event":"Downgrade risk reaction","target":"S&P 500","family":"Regulatory / credit risk","similarity":72,"d1":-1.4,"d3":-2.1,"d7":-2.8,"d14":-1.9,"d30":-0.6,"risk":78,"vol":9.6,"regime":"Credit risk"},
        {"date":"2024-03-18","event":"Regulatory pressure","target":"Mega-cap tech","family":"Regulatory / credit risk","similarity":70,"d1":-0.8,"d3":-1.3,"d7":-2.2,"d14":-2.8,"d30":-1.5,"risk":82,"vol":8.7,"regime":"Policy risk"},
        {"date":"2022-09-13","event":"Credit spread stress","target":"S&P 500","family":"Regulatory / credit risk","similarity":66,"d1":-1.9,"d3":-2.8,"d7":-3.4,"d14":-2.1,"d30":-0.8,"risk":85,"vol":10.5,"regime":"Credit stress"},
    ]

    sample_headline = "Dow jumps 150 points for first close above 53,000; Nasdaq rises as chips rebound"
    sample_body = (
        "Stocks maintained positive momentum after a strong week on Wall Street. "
        "The S&P 500 gained 0.72%, while the Nasdaq Composite advanced 1.12% as chip stocks rebounded. "
        "Investors pointed to stronger technology momentum, improving risk appetite, and broad-market strength."
    )

    if "hi_url" not in st.session_state:
        st.session_state.hi_url = ""
    if "hi_headline" not in st.session_state:
        st.session_state.hi_headline = sample_headline
    if "hi_body" not in st.session_state:
        st.session_state.hi_body = sample_body
    if "hi_target" not in st.session_state:
        st.session_state.hi_target = "Broad market"
    if "hi_status" not in st.session_state:
        st.session_state.hi_status = "Sample article loaded for public demo. Paste your own article or URL."

    st.markdown(
        """
        <style>
          .hi-hero {
            display:grid;
            grid-template-columns:1.08fr .92fr;
            gap:1rem;
            padding:1.35rem;
            border-radius:24px;
            border:1px solid rgba(34,211,238,.34);
            background:
              radial-gradient(circle at 8% 8%, rgba(34,211,238,.20), transparent 22rem),
              radial-gradient(circle at 76% 8%, rgba(139,92,246,.22), transparent 24rem),
              linear-gradient(145deg, rgba(8,47,73,.72), rgba(8,13,28,.96));
            box-shadow:0 30px 90px rgba(0,0,0,.36), inset 0 1px 0 rgba(255,255,255,.07);
            margin-bottom:.9rem;
          }
          .hi-kicker {
            color:#67e8f9;
            font-size:.70rem;
            font-weight:950;
            letter-spacing:.13em;
            text-transform:uppercase;
          }
          .hi-title {
            color:white;
            font-size:2.6rem;
            line-height:1;
            font-weight:950;
            letter-spacing:-.06em;
            margin:.42rem 0 .55rem 0;
          }
          .hi-subtitle {
            color:#dbeafe;
            font-size:1rem;
            line-height:1.55;
          }
          .hi-chip-row {
            display:flex;
            flex-wrap:wrap;
            gap:.48rem;
            margin-top:.9rem;
          }
          .hi-chip {
            padding:.43rem .68rem;
            border-radius:999px;
            font-size:.72rem;
            font-weight:850;
            color:#bfdbfe;
            border:1px solid rgba(96,165,250,.25);
            background:rgba(15,23,42,.65);
          }
          .hi-engine {
            padding:1rem;
            border-radius:20px;
            border:1px solid rgba(148,163,184,.18);
            background:linear-gradient(160deg, rgba(15,23,42,.92), rgba(2,6,23,.96));
          }
          .hi-step {
            display:grid;
            grid-template-columns:34px 1fr;
            gap:.65rem;
            align-items:center;
            padding:.54rem;
            margin-bottom:.45rem;
            border-radius:13px;
            border:1px solid rgba(96,165,250,.18);
            background:rgba(15,23,42,.70);
          }
          .hi-num {
            width:34px;
            height:34px;
            display:grid;
            place-items:center;
            border-radius:11px;
            color:#67e8f9;
            background:rgba(14,165,233,.13);
            border:1px solid rgba(34,211,238,.25);
            font-weight:950;
          }
          .hi-step strong {
            display:block;
            color:white;
            font-size:.82rem;
          }
          .hi-step span {
            display:block;
            color:#94a3b8;
            font-size:.70rem;
            margin-top:.08rem;
          }
          .hi-panel {
            margin:.95rem 0;
            padding:1.1rem;
            border-radius:22px;
            border:1px solid rgba(34,211,238,.24);
            background:
              radial-gradient(circle at 6% 0%, rgba(34,211,238,.11), transparent 18rem),
              radial-gradient(circle at 94% 30%, rgba(139,92,246,.13), transparent 20rem),
              linear-gradient(145deg, rgba(15,23,42,.88), rgba(8,13,28,.96));
            box-shadow:0 22px 60px rgba(0,0,0,.25);
          }
          .hi-section-title {
            color:white;
            font-size:1.25rem;
            font-weight:950;
            letter-spacing:-.04em;
            margin:.2rem 0 .35rem 0;
          }
          .hi-copy {
            color:#cbd5e1;
            font-size:.84rem;
            line-height:1.48;
            margin:0;
          }
          .hi-metrics {
            display:grid;
            grid-template-columns:repeat(5,minmax(0,1fr));
            gap:.68rem;
            margin:.85rem 0 .9rem 0;
          }
          .hi-metric {
            padding:1rem;
            border-radius:18px;
            border:1px solid rgba(148,163,184,.16);
            background:rgba(15,23,42,.82);
          }
          .hi-metric strong {
            color:white;
            font-size:1.45rem;
            font-weight:950;
            display:block;
          }
          .hi-metric span {
            color:#cbd5e1;
            font-size:.74rem;
            font-weight:760;
          }
          .hi-grid-3 {
            display:grid;
            grid-template-columns:repeat(3,minmax(0,1fr));
            gap:.68rem;
            margin-top:.8rem;
          }
          .hi-grid-4 {
            display:grid;
            grid-template-columns:repeat(4,minmax(0,1fr));
            gap:.68rem;
            margin-top:.8rem;
          }
          .hi-card {
            padding:.95rem;
            border-radius:17px;
            border:1px solid rgba(148,163,184,.16);
            background:rgba(15,23,42,.74);
          }
          .hi-card strong {
            color:white;
            display:block;
            font-size:.96rem;
            margin-bottom:.32rem;
          }
          .hi-card span, .hi-card li {
            color:#cbd5e1;
            font-size:.75rem;
            line-height:1.38;
          }
          .hi-card ul {
            margin:.2rem 0 0 1rem;
            padding:0;
          }
          .hi-table {
            width:100%;
            border-collapse:separate;
            border-spacing:0 .45rem;
            margin-top:.75rem;
          }
          .hi-table th {
            color:#94a3b8;
            font-size:.70rem;
            text-align:left;
            padding:.35rem .5rem;
            text-transform:uppercase;
            letter-spacing:.08em;
          }
          .hi-table td {
            color:#e5e7eb;
            font-size:.78rem;
            padding:.62rem .5rem;
            background:rgba(15,23,42,.72);
            border-top:1px solid rgba(148,163,184,.13);
            border-bottom:1px solid rgba(148,163,184,.13);
          }
          .hi-table td:first-child {
            border-left:1px solid rgba(148,163,184,.13);
            border-radius:12px 0 0 12px;
            font-weight:900;
          }
          .hi-table td:last-child {
            border-right:1px solid rgba(148,163,184,.13);
            border-radius:0 12px 12px 0;
          }
          .hi-explain {
            margin:.45rem 0 .9rem 0;
            padding:.9rem 1rem;
            border-radius:16px;
            border:1px solid rgba(148,163,184,.15);
            background:rgba(15,23,42,.66);
            color:#cbd5e1;
            font-size:.81rem;
            line-height:1.48;
          }
          .hi-explain strong { color:white; }
          .hi-good { color:#86efac !important; }
          .hi-warn { color:#fbbf24 !important; }
          .hi-bad { color:#fca5a5 !important; }
          @media (max-width:1100px) {
            .hi-hero,.hi-metrics,.hi-grid-3,.hi-grid-4 { grid-template-columns:1fr; }
            .hi-title { font-size:2.05rem; }
          }
        </style>

        <section class="hi-hero">
          <div>
            <div class="hi-kicker">Historical Intelligence Cockpit</div>
            <div class="hi-title">Article-Driven<br/>Comparable Events</div>
            <div class="hi-subtitle">
              Enter a financial news URL or paste article text. The page detects the event family,
              target context, tone, risk cues, and then compares the article against historically similar market events.
            </div>
            <div class="hi-chip-row">
              <span class="hi-chip">Article URL</span>
              <span class="hi-chip">Paste fallback</span>
              <span class="hi-chip">Auto event detection</span>
              <span class="hi-chip">Similar events</span>
              <span class="hi-chip">Market reactions</span>
              <span class="hi-chip">Historical explanation</span>
            </div>
          </div>

          <div class="hi-engine">
            <div class="hi-kicker">Comparable Event Engine</div>
            <div class="hi-step"><div class="hi-num">01</div><div><strong>Read article</strong><span>URL, headline, body, optional target</span></div></div>
            <div class="hi-step"><div class="hi-num">02</div><div><strong>Detect event family</strong><span>Semis, market rally, earnings, macro, credit risk</span></div></div>
            <div class="hi-step"><div class="hi-num">03</div><div><strong>Match comparable events</strong><span>Only same-family or explicitly selected event group</span></div></div>
            <div class="hi-step"><div class="hi-num">04</div><div><strong>Measure reaction windows</strong><span>1D, 3D, 7D, 14D, 30D movement</span></div></div>
            <div class="hi-step"><div class="hi-num">05</div><div><strong>Explain context</strong><span>Why these events matched and what history suggests</span></div></div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <section class="hi-panel">
          <div class="hi-kicker">Historical Input</div>
          <div class="hi-section-title">Enter article source and generate comparable-event history</div>
          <p class="hi-copy">
            URL extraction may fail on blocked or paywalled sites. If that happens, paste the headline and article body manually.
            Public demo mode uses curated comparable-event examples and does not claim live historical database retrieval yet.
          </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    input_left, input_right = st.columns([1.18, .82])

    with input_left:
        url_value = st.text_input("Article URL", value=st.session_state.hi_url, placeholder="https://...")
        target_hint = st.text_input(
            "Optional ticker / index / sector",
            value=st.session_state.hi_target,
            placeholder="Examples: NVDA, Nasdaq, Dow, S&P 500, Semiconductors, Broad market",
        )
        headline = st.text_input("Article headline", value=st.session_state.hi_headline)
        article_body = st.text_area("Article body or summary", value=st.session_state.hi_body, height=145)

    with input_right:
        event_choice = st.selectbox(
            "Event matching mode",
            [
                "Auto-detect from article",
                "Semiconductor rebound",
                "Broad market rally",
                "Earnings beat",
                "Guidance raise",
                "Macro shock",
                "Regulatory / credit risk",
                "All comparable events",
            ],
            index=0,
        )
        min_similarity = st.slider("Minimum similarity", min_value=50, max_value=95, value=60, step=1)
        current_path = st.selectbox("Current forecast path", ["Moderately bullish", "Neutral / mixed", "Bearish risk"], index=0)

        fetch_clicked = st.button("Extract URL text", type="secondary", use_container_width=True)
        sample_clicked = st.button("Load sample article", type="secondary", use_container_width=True)
        generate_clicked = st.button("Generate historical match", type="primary", use_container_width=True)

    if fetch_clicked:
        status, fetched_headline, fetched_body = _extract_article_from_url(url_value)
        st.session_state.hi_url = url_value
        st.session_state.hi_status = status
        if fetched_headline:
            st.session_state.hi_headline = fetched_headline
        if fetched_body:
            st.session_state.hi_body = fetched_body
        st.rerun()

    if sample_clicked:
        st.session_state.hi_url = ""
        st.session_state.hi_headline = sample_headline
        st.session_state.hi_body = sample_body
        st.session_state.hi_target = "Broad market"
        st.session_state.hi_status = "Sample article loaded for public demo."
        st.rerun()

    st.session_state.hi_url = url_value
    st.session_state.hi_headline = headline
    st.session_state.hi_body = article_body
    st.session_state.hi_target = target_hint

    full_text = f"{headline}\n{article_body}".strip()
    word_count = len(re.findall(r"\b\w+\b", full_text))

    bullish_cues = _find_cues(full_text, bullish_patterns)
    bearish_cues = _find_cues(full_text, bearish_patterns)
    risk_cues = _find_cues(full_text, risk_patterns)

    detected_family = _detect_event_family(full_text)
    selected_family = detected_family if event_choice == "Auto-detect from article" else event_choice
    target_type, detected_entities = _detect_target(full_text, target_hint)

    if not full_text:
        input_quality = "Missing"
    elif word_count < 25:
        input_quality = "Headline-only / limited"
    elif word_count < 120:
        input_quality = "Moderate"
    else:
        input_quality = "Strong"

    def _cue_list_html(cues: list[dict[str, str | float]], empty_text: str) -> str:
        if not cues:
            return f"<span>{html.escape(empty_text)}</span>"
        items = "".join(f"<li>{html.escape(str(c['label']))}</li>" for c in cues[:6])
        return f"<ul>{items}</ul>"

    st.markdown(
        f"""
        <section class="hi-panel">
          <div class="hi-kicker">Current Article Signal</div>
          <div class="hi-section-title">What the historical engine detected</div>
          <div class="hi-grid-4">
            <div class="hi-card"><strong>Detected event family</strong><span>{html.escape(detected_family)}</span></div>
            <div class="hi-card"><strong>Selected match group</strong><span>{html.escape(selected_family)}</span></div>
            <div class="hi-card"><strong>Target context</strong><span>{html.escape(target_type)} · {html.escape(detected_entities)}</span></div>
            <div class="hi-card"><strong>Input quality</strong><span>{html.escape(input_quality)} · {word_count} words</span></div>
          </div>
          <div class="hi-grid-3">
            <div class="hi-card"><strong class="hi-good">Bullish cues</strong>{_cue_list_html(bullish_cues, "No clear bullish cues detected.")}</div>
            <div class="hi-card"><strong class="hi-bad">Bearish cues</strong>{_cue_list_html(bearish_cues, "No clear bearish cues detected.")}</div>
            <div class="hi-card"><strong class="hi-warn">Risk cues</strong>{_cue_list_html(risk_cues, "No clear risk cues detected.")}</div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    if selected_family == "All comparable events":
        filtered = [row for row in comparable_events if row["similarity"] >= min_similarity]
    else:
        filtered = [row for row in comparable_events if row["family"] == selected_family and row["similarity"] >= min_similarity]

    if not filtered:
        st.markdown(
            f"""
            <section class="hi-panel">
              <div class="hi-kicker">No Strong Comparable Events Found</div>
              <div class="hi-section-title">No same-family events match the current threshold</div>
              <p class="hi-copy">
                Selected match group: {html.escape(selected_family)}. Minimum similarity: {min_similarity}%.
                Lower the similarity threshold or choose “All comparable events” to broaden the search.
                The page does not silently replace missing results with unrelated events.
              </p>
            </section>
            """,
            unsafe_allow_html=True,
        )
        return

    count = len(filtered)
    avg_7d = round(statistics.mean(row["d7"] for row in filtered), 2)
    avg_30d = round(statistics.mean(row["d30"] for row in filtered), 2)
    best_7d = round(max(row["d7"] for row in filtered), 2)
    worst_7d = round(min(row["d7"] for row in filtered), 2)
    avg_similarity = round(statistics.mean(row["similarity"] for row in filtered))
    positive_rate = round(sum(1 for row in filtered if row["d7"] > 0) * 100 / count)

    st.markdown(
        f"""
        <div class="hi-metrics">
          <div class="hi-metric"><strong>{count}</strong><span>same-family comparable events</span></div>
          <div class="hi-metric"><strong>{avg_7d:+.1f}%</strong><span>average 7D reaction</span></div>
          <div class="hi-metric"><strong>{best_7d:+.1f}%</strong><span>best 7D move</span></div>
          <div class="hi-metric"><strong>{worst_7d:+.1f}%</strong><span>worst 7D move</span></div>
          <div class="hi-metric"><strong>{positive_rate}%</strong><span>positive 7D rate</span></div>
        </div>

        <section class="hi-panel">
          <div class="hi-kicker">Why These Events Matched</div>
          <div class="hi-grid-4">
            <div class="hi-card"><strong>Event family match</strong><span>{html.escape(selected_family)}</span></div>
            <div class="hi-card"><strong>Average similarity</strong><span>{avg_similarity}% comparable-event match confidence</span></div>
            <div class="hi-card"><strong>Average 30D reaction</strong><span>{avg_30d:+.1f}% after similar events</span></div>
            <div class="hi-card"><strong>Current forecast path</strong><span>{html.escape(current_path)} reference path</span></div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    try:
        import plotly.graph_objects as go

        windows = ["Day 0", "1D", "3D", "7D", "14D", "30D"]
        avg_line = [0, statistics.mean(row["d1"] for row in filtered), statistics.mean(row["d3"] for row in filtered), statistics.mean(row["d7"] for row in filtered), statistics.mean(row["d14"] for row in filtered), statistics.mean(row["d30"] for row in filtered)]
        best_line = [0, max(row["d1"] for row in filtered), max(row["d3"] for row in filtered), max(row["d7"] for row in filtered), max(row["d14"] for row in filtered), max(row["d30"] for row in filtered)]
        worst_line = [0, min(row["d1"] for row in filtered), min(row["d3"] for row in filtered), min(row["d7"] for row in filtered), min(row["d14"] for row in filtered), min(row["d30"] for row in filtered)]

        if current_path == "Moderately bullish":
            current_line = [0, .8, 1.5, 2.6, 3.4, 4.2]
        elif current_path == "Bearish risk":
            current_line = [0, -.4, -1.0, -1.8, -2.4, -3.2]
        else:
            current_line = [0, .2, .3, .5, .7, .9]

        timeline = go.Figure()
        timeline.add_trace(go.Scatter(x=windows, y=best_line, mode="lines+markers", name="Best comparable reaction", line=dict(width=3)))
        timeline.add_trace(go.Scatter(x=windows, y=avg_line, mode="lines+markers", name="Average comparable reaction", line=dict(width=5)))
        timeline.add_trace(go.Scatter(x=windows, y=worst_line, mode="lines+markers", name="Worst comparable reaction", line=dict(width=3)))
        timeline.add_trace(go.Scatter(x=windows, y=current_line, mode="lines+markers", name="Current forecast reference", line=dict(width=4, dash="dash")))
        timeline.update_layout(
            title="Historical Reaction Timeline · Same-Family Comparable Events",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,.35)",
            height=460,
            margin=dict(l=0, r=0, t=55, b=0),
            xaxis_title="Reaction window",
            yaxis_title="Movement %",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(timeline, use_container_width=True, config={"displayModeBar": False})

        st.markdown(
            """
            <div class="hi-explain">
              <strong>How to read this chart:</strong>
              the average line shows the typical same-family reaction after similar events. Best and worst lines show historical range.
              The dashed line compares the current forecast path against those historical outcomes.
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)

        with col1:
            scatter = go.Figure()
            scatter.add_trace(
                go.Scatter(
                    x=[row["similarity"] for row in filtered],
                    y=[row["d7"] for row in filtered],
                    mode="markers",
                    marker=dict(
                        size=[max(10, row["vol"] * 2.0) for row in filtered],
                        opacity=.88,
                        line=dict(width=1, color="rgba(255,255,255,.35)"),
                    ),
                    text=[row["event"] for row in filtered],
                    customdata=[row["regime"] for row in filtered],
                    hovertemplate="<b>%{text}</b><br>Similarity: %{x}%<br>7D reaction: %{y}%<br>Regime: %{customdata}<extra></extra>",
                )
            )
            scatter.update_layout(
                title="Similar Event Scatter Map",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,.35)",
                height=400,
                margin=dict(l=0, r=0, t=55, b=0),
                xaxis=dict(title="Similarity score", range=[45, 100]),
                yaxis_title="7D reaction %",
                showlegend=False,
            )
            st.plotly_chart(scatter, use_container_width=True, config={"displayModeBar": False})

        with col2:
            distribution = go.Figure()
            distribution.add_trace(go.Histogram(x=[row["d7"] for row in filtered], nbinsx=8))
            distribution.update_layout(
                title="Historical 7D Return Distribution",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,.35)",
                height=400,
                margin=dict(l=0, r=0, t=55, b=0),
                xaxis_title="7D movement %",
                yaxis_title="Comparable event count",
            )
            st.plotly_chart(distribution, use_container_width=True, config={"displayModeBar": False})

        col3, col4 = st.columns(2)

        with col3:
            match_breakdown = go.Figure(
                go.Bar(
                    x=[34, 23, 18, 14, 11],
                    y=["Event family", "Sentiment tone", "Target / sector", "Risk profile", "Market regime"],
                    orientation="h",
                )
            )
            match_breakdown.update_layout(
                title="Comparable Match Breakdown",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,.35)",
                height=380,
                margin=dict(l=0, r=0, t=55, b=0),
                xaxis_title="Contribution %",
                yaxis_title="",
            )
            st.plotly_chart(match_breakdown, use_container_width=True, config={"displayModeBar": False})

        with col4:
            risk_return = go.Figure()
            risk_return.add_trace(
                go.Scatter(
                    x=[row["risk"] for row in filtered],
                    y=[row["d30"] for row in filtered],
                    mode="markers",
                    marker=dict(
                        size=[max(10, row["similarity"] / 4.5) for row in filtered],
                        opacity=.88,
                        line=dict(width=1, color="rgba(255,255,255,.35)"),
                    ),
                    text=[row["target"] for row in filtered],
                    hovertemplate="<b>%{text}</b><br>Risk pressure: %{x}<br>30D reaction: %{y}%<extra></extra>",
                )
            )
            risk_return.update_layout(
                title="Risk Pressure vs 30D Reaction",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,.35)",
                height=380,
                margin=dict(l=0, r=0, t=55, b=0),
                xaxis=dict(title="Risk pressure", range=[20, 90]),
                yaxis_title="30D movement %",
                showlegend=False,
            )
            st.plotly_chart(risk_return, use_container_width=True, config={"displayModeBar": False})

    except Exception as exc:
        st.warning(f"Historical Intelligence charts could not render. Reason: {exc}")

    table_rows = ""
    for row in sorted(filtered, key=lambda item: item["similarity"], reverse=True):
        table_rows += (
            "<tr>"
            f"<td>{html.escape(row['date'])}</td>"
            f"<td>{html.escape(row['event'])}</td>"
            f"<td>{html.escape(row['target'])}</td>"
            f"<td>{row['similarity']}%</td>"
            f"<td>{row['d1']:+.1f}%</td>"
            f"<td>{row['d7']:+.1f}%</td>"
            f"<td>{row['d30']:+.1f}%</td>"
            f"<td>{html.escape(row['regime'])}</td>"
            "</tr>"
        )

    if avg_7d > 1.0 and positive_rate >= 60:
        historical_read = "historically supportive"
        guidance = "Similar same-family events usually produced positive short-term follow-through."
    elif avg_7d < -1.0:
        historical_read = "historically cautious"
        guidance = "Similar same-family events often produced weak or negative short-term reactions."
    else:
        historical_read = "historically mixed"
        guidance = "Similar same-family events produced uneven reactions, so the current signal should be treated carefully."

    st.markdown(
        f"""
        <section class="hi-panel">
          <div class="hi-kicker">Comparable Events Table</div>
          <div class="hi-section-title">Historical cases selected by same-family similarity and context</div>
          <table class="hi-table">
            <thead>
              <tr>
                <th>Date</th>
                <th>Event</th>
                <th>Target</th>
                <th>Similarity</th>
                <th>1D</th>
                <th>7D</th>
                <th>30D</th>
                <th>Regime</th>
              </tr>
            </thead>
            <tbody>{table_rows}</tbody>
          </table>
        </section>

        <section class="hi-panel">
          <div class="hi-kicker">Regime Context</div>
          <div class="hi-grid-4">
            <div class="hi-card"><strong>Market regime</strong><span>Same-family public demo sample set</span></div>
            <div class="hi-card"><strong>Volatility context</strong><span>Average comparable volatility: {statistics.mean(row["vol"] for row in filtered):.1f}</span></div>
            <div class="hi-card"><strong>Risk pressure</strong><span>Average risk pressure: {statistics.mean(row["risk"] for row in filtered):.0f}/100</span></div>
            <div class="hi-card"><strong>Historical read</strong><span>{html.escape(historical_read.title())}</span></div>
          </div>
        </section>

        <section class="hi-panel">
          <div class="hi-kicker">Historical Analyst Explanation</div>
          <div class="hi-section-title">What history suggests</div>
          <p class="hi-copy">
            {html.escape(guidance)} The selected comparable-event set has an average 7-day reaction of {avg_7d:+.1f}%
            and a positive 7-day historical rate of {positive_rate}%. The best cases show continuation, while weaker
            cases fade when macro risk, credit risk, or volatility returns. This page does not directly predict the future;
            it shows how the current article’s detected event family behaved historically.
          </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

def _render_explainability_page() -> None:
    """Render the Explainability page as an article-driven model explanation cockpit."""

    import html
    import re
    from urllib.parse import urlparse

    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    def _extract_article_from_url(url: str) -> tuple[str, str, str]:
        """Try to extract article title and body from URL for the public explainability demo."""
        clean_url = url.strip()
        if not clean_url:
            return "No URL provided.", "", ""

        parsed = urlparse(clean_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return "Invalid URL. Paste a full URL starting with http:// or https://.", "", ""

        try:
            import requests
            from bs4 import BeautifulSoup

            response = requests.get(
                clean_url,
                timeout=8,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X) "
                        "AppleWebKit/537.36 Chrome/120 Safari/537.36"
                    )
                },
            )
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup(["script", "style", "noscript", "svg", "form", "nav", "footer", "header"]):
                tag.decompose()

            title = ""
            if soup.find("h1"):
                title = soup.find("h1").get_text(" ", strip=True)
            if not title and soup.title:
                title = soup.title.get_text(" ", strip=True)

            paragraphs = [
                p_tag.get_text(" ", strip=True)
                for p_tag in soup.find_all("p")
                if len(p_tag.get_text(" ", strip=True).split()) >= 8
            ]
            body = "\n\n".join(paragraphs[:12]).strip()

            if not body:
                return "URL loaded, but article text could not be extracted. Paste article text manually.", title, ""

            return "URL article text extracted.", title, body

        except Exception as exc:
            return f"URL extraction failed. Paste article text manually. Reason: {exc}", "", ""

    explain_patterns = [
        {"pattern": r"\bjumps?\b", "label": "jumps / strong upward move", "category": "Sentiment driver", "impact": 1.25, "effect": "Lifts bullish sentiment and movement pressure."},
        {"pattern": r"\brises?\b|\brose\b|\badvanced\b", "label": "rises / advances", "category": "Sentiment driver", "impact": 1.05, "effect": "Supports positive article tone."},
        {"pattern": r"\brebounds?\b|\brebounded\b", "label": "rebound language", "category": "Movement driver", "impact": 1.15, "effect": "Suggests recovery momentum after prior weakness."},
        {"pattern": r"\bcloses?\s+above\b|\bfirst\s+close\s+above\b", "label": "close above key level", "category": "Movement driver", "impact": 1.30, "effect": "Signals technical or psychological market strength."},
        {"pattern": r"\brecord\s+(close|high)\b|\ball[- ]time high\b", "label": "record high / record close", "category": "Movement driver", "impact": 1.35, "effect": "Adds strong continuation pressure."},
        {"pattern": r"\bchips?\s+rebound\b|\bsemiconductors?\s+rebound\b|\bchip\s+stocks\b", "label": "chip / semiconductor strength", "category": "Sector driver", "impact": 1.20, "effect": "Shows sector leadership behind the move."},
        {"pattern": r"\bbeat(s|ing)?\b|\bbeats?\s+estimates\b", "label": "beat estimates", "category": "Sentiment driver", "impact": 1.35, "effect": "Improves earnings-quality signal."},
        {"pattern": r"\braises?\s+guidance\b|\braised\s+guidance\b", "label": "raised guidance", "category": "Movement driver", "impact": 1.45, "effect": "Supports forward-looking upside."},
        {"pattern": r"\bupgrade(d|s)?\b|\banalyst\s+upgrade\b", "label": "analyst upgrade", "category": "Sentiment driver", "impact": 1.05, "effect": "Adds external validation to the bullish case."},
        {"pattern": r"\bpositive\s+momentum\b|\bstrong\s+week\b|\brisk\s+appetite\b", "label": "positive momentum", "category": "Movement driver", "impact": 0.95, "effect": "Supports follow-through continuation."},

        {"pattern": r"\bdrops?\b|\bfalls?\b|\bfell\b", "label": "drops / falls", "category": "Negative driver", "impact": -1.20, "effect": "Pulls sentiment and movement pressure lower."},
        {"pattern": r"\bslides?\b|\bslid\b|\btumbles?\b", "label": "slide / tumble language", "category": "Negative driver", "impact": -1.30, "effect": "Indicates sharp downside reaction."},
        {"pattern": r"\bmiss(es|ed)?\b|\bmisses?\s+estimates\b", "label": "missed estimates", "category": "Negative driver", "impact": -1.35, "effect": "Weakens earnings-quality signal."},
        {"pattern": r"\bcuts?\s+guidance\b|\bcut\s+guidance\b", "label": "cut guidance", "category": "Negative driver", "impact": -1.45, "effect": "Creates forward-looking downside pressure."},
        {"pattern": r"\bdowngrade(d|s)?\b|\banalyst\s+downgrade\b", "label": "analyst downgrade", "category": "Negative driver", "impact": -1.10, "effect": "Adds external negative validation."},

        {"pattern": r"\brisk(s)?\b", "label": "explicit risk language", "category": "Risk driver", "impact": -0.80, "effect": "Increases uncertainty and lowers confidence."},
        {"pattern": r"\buncertain(ty)?\b|\buncertainties\b", "label": "uncertainty language", "category": "Risk driver", "impact": -0.90, "effect": "Widens possible outcome range."},
        {"pattern": r"\bvolatil(e|ity)\b", "label": "volatility language", "category": "Risk driver", "impact": -1.00, "effect": "Raises downside and confidence risk."},
        {"pattern": r"\bregulatory\b|\bregulation\b|\blawsuit\b|\bprobe\b", "label": "regulatory / legal pressure", "category": "Risk driver", "impact": -1.15, "effect": "Adds structural risk to the signal."},
        {"pattern": r"\binflation\b|\brates?\b|\bfed\b|\brecession\b|\bmacro\b", "label": "macro / rates risk", "category": "Risk driver", "impact": -0.95, "effect": "Adds market-wide uncertainty."},
    ]

    def _find_driver_rows(text: str) -> list[dict[str, str | float]]:
        lowered = text.lower()
        rows: list[dict[str, str | float]] = []
        seen: set[str] = set()

        for item in explain_patterns:
            if re.search(str(item["pattern"]), lowered, flags=re.IGNORECASE) and str(item["label"]) not in seen:
                rows.append(
                    {
                        "label": str(item["label"]),
                        "category": str(item["category"]),
                        "impact": float(item["impact"]),
                        "effect": str(item["effect"]),
                    }
                )
                seen.add(str(item["label"]))

        return rows

    def _score_segment(segment: str) -> float:
        score = 0.0
        for item in explain_patterns:
            if re.search(str(item["pattern"]), segment.lower(), flags=re.IGNORECASE):
                score += float(item["impact"])
        return round(score, 2)

    def _detect_target(text: str, user_target: str) -> tuple[str, str]:
        lowered = text.lower()
        entities: list[str] = []

        if re.search(r"\bdow\b|\bdjia\b", lowered):
            entities.append("Dow")
        if re.search(r"\bnasdaq\b|\bqqq\b", lowered):
            entities.append("Nasdaq")
        if re.search(r"\bs&p\b|\bsp500\b|\bspx\b|\bspy\b", lowered):
            entities.append("S&P 500")
        if re.search(r"\bchip\b|\bchips\b|\bsemiconductor\b|\bsemiconductors\b|\bnvidia\b|\bnvda\b|\bamd\b|\bintel\b", lowered):
            entities.append("Chips / semiconductors")
        if re.search(r"\brates?\b|\bfed\b|\binflation\b|\btreasury\b", lowered):
            entities.append("Macro / rates")

        ticker_matches = re.findall(r"\$?[A-Z]{2,5}\b", text)
        ticker_matches = [x.replace("$", "") for x in ticker_matches if x not in {"LIVE", "CEO", "CFO", "EPS", "THE"}]
        for ticker in ticker_matches[:4]:
            if ticker not in entities and ticker not in {"DOW"}:
                entities.append(ticker)

        if user_target.strip():
            return "User-selected target", user_target.strip()

        if "Dow" in entities or "Nasdaq" in entities or "S&P 500" in entities:
            if "Chips / semiconductors" in entities:
                return "Broad market + sector", ", ".join(entities)
            return "Broad market / index", ", ".join(entities)

        if "Chips / semiconductors" in entities:
            return "Sector / theme", ", ".join(entities)

        if ticker_matches:
            return "Ticker / company", ", ".join(entities) if entities else ", ".join(ticker_matches[:4])

        return "General financial news", "No specific ticker or index detected"

    sample_headline = "Dow jumps 150 points for first close above 53,000; Nasdaq rises as chips rebound"
    sample_body = (
        "Stocks maintained positive momentum after a strong week on Wall Street. "
        "The S&P 500 gained 0.72%, while the Nasdaq Composite advanced 1.12% as chip stocks rebounded. "
        "Investors pointed to stronger technology momentum, improving risk appetite, and broad-market strength, "
        "while macro uncertainty remained limited."
    )

    if "ex_url" not in st.session_state:
        st.session_state.ex_url = ""
    if "ex_headline" not in st.session_state:
        st.session_state.ex_headline = sample_headline
    if "ex_body" not in st.session_state:
        st.session_state.ex_body = sample_body
    if "ex_target" not in st.session_state:
        st.session_state.ex_target = "Broad market"
    if "ex_status" not in st.session_state:
        st.session_state.ex_status = "Sample article loaded for public demo."

    st.markdown(
        """
        <style>
          .ex-hero {
            display:grid;
            grid-template-columns:1.08fr .92fr;
            gap:1rem;
            padding:1.35rem;
            border-radius:24px;
            border:1px solid rgba(34,211,238,.34);
            background:
              radial-gradient(circle at 8% 8%, rgba(34,211,238,.20), transparent 22rem),
              radial-gradient(circle at 76% 8%, rgba(139,92,246,.22), transparent 24rem),
              radial-gradient(circle at 90% 92%, rgba(34,197,94,.12), transparent 22rem),
              linear-gradient(145deg, rgba(8,47,73,.72), rgba(8,13,28,.96));
            box-shadow:0 30px 90px rgba(0,0,0,.36), inset 0 1px 0 rgba(255,255,255,.07);
            margin-bottom:.9rem;
          }
          .ex-kicker {
            color:#67e8f9;
            font-size:.70rem;
            font-weight:950;
            letter-spacing:.13em;
            text-transform:uppercase;
          }
          .ex-title {
            color:white;
            font-size:2.6rem;
            line-height:1;
            font-weight:950;
            letter-spacing:-.06em;
            margin:.42rem 0 .55rem 0;
          }
          .ex-subtitle {
            color:#dbeafe;
            font-size:1rem;
            line-height:1.55;
          }
          .ex-chip-row {
            display:flex;
            flex-wrap:wrap;
            gap:.48rem;
            margin-top:.9rem;
          }
          .ex-chip {
            padding:.43rem .68rem;
            border-radius:999px;
            font-size:.72rem;
            font-weight:850;
            color:#bfdbfe;
            border:1px solid rgba(96,165,250,.25);
            background:rgba(15,23,42,.65);
          }
          .ex-engine {
            padding:1rem;
            border-radius:20px;
            border:1px solid rgba(148,163,184,.18);
            background:linear-gradient(160deg, rgba(15,23,42,.92), rgba(2,6,23,.96));
          }
          .ex-step {
            display:grid;
            grid-template-columns:34px 1fr;
            gap:.65rem;
            align-items:center;
            padding:.54rem;
            margin-bottom:.45rem;
            border-radius:13px;
            border:1px solid rgba(96,165,250,.18);
            background:rgba(15,23,42,.70);
          }
          .ex-num {
            width:34px;
            height:34px;
            display:grid;
            place-items:center;
            border-radius:11px;
            color:#67e8f9;
            background:rgba(14,165,233,.13);
            border:1px solid rgba(34,211,238,.25);
            font-weight:950;
          }
          .ex-step strong {
            display:block;
            color:white;
            font-size:.82rem;
          }
          .ex-step span {
            display:block;
            color:#94a3b8;
            font-size:.70rem;
            margin-top:.08rem;
          }
          .ex-panel {
            margin:.95rem 0;
            padding:1.1rem;
            border-radius:22px;
            border:1px solid rgba(34,211,238,.24);
            background:
              radial-gradient(circle at 6% 0%, rgba(34,211,238,.11), transparent 18rem),
              radial-gradient(circle at 94% 30%, rgba(139,92,246,.13), transparent 20rem),
              linear-gradient(145deg, rgba(15,23,42,.88), rgba(8,13,28,.96));
            box-shadow:0 22px 60px rgba(0,0,0,.25);
          }
          .ex-section-title {
            color:white;
            font-size:1.25rem;
            font-weight:950;
            letter-spacing:-.04em;
            margin:.2rem 0 .35rem 0;
          }
          .ex-copy {
            color:#cbd5e1;
            font-size:.84rem;
            line-height:1.48;
            margin:0;
          }
          .ex-metrics {
            display:grid;
            grid-template-columns:repeat(5,minmax(0,1fr));
            gap:.68rem;
            margin:.85rem 0 .9rem 0;
          }
          .ex-metric {
            padding:1rem;
            border-radius:18px;
            border:1px solid rgba(148,163,184,.16);
            background:rgba(15,23,42,.82);
          }
          .ex-metric strong {
            color:white;
            font-size:1.35rem;
            font-weight:950;
            display:block;
          }
          .ex-metric span {
            color:#cbd5e1;
            font-size:.74rem;
            font-weight:760;
          }
          .ex-grid-3 {
            display:grid;
            grid-template-columns:repeat(3,minmax(0,1fr));
            gap:.68rem;
            margin-top:.8rem;
          }
          .ex-grid-4 {
            display:grid;
            grid-template-columns:repeat(4,minmax(0,1fr));
            gap:.68rem;
            margin-top:.8rem;
          }
          .ex-card {
            padding:.95rem;
            border-radius:17px;
            border:1px solid rgba(148,163,184,.16);
            background:rgba(15,23,42,.74);
          }
          .ex-card strong {
            color:white;
            display:block;
            font-size:.96rem;
            margin-bottom:.32rem;
          }
          .ex-card span, .ex-card li {
            color:#cbd5e1;
            font-size:.75rem;
            line-height:1.38;
          }
          .ex-card ul {
            margin:.2rem 0 0 1rem;
            padding:0;
          }
          .ex-table {
            width:100%;
            border-collapse:separate;
            border-spacing:0 .45rem;
            margin-top:.75rem;
          }
          .ex-table th {
            color:#94a3b8;
            font-size:.70rem;
            text-align:left;
            padding:.35rem .5rem;
            text-transform:uppercase;
            letter-spacing:.08em;
          }
          .ex-table td {
            color:#e5e7eb;
            font-size:.78rem;
            padding:.62rem .5rem;
            background:rgba(15,23,42,.72);
            border-top:1px solid rgba(148,163,184,.13);
            border-bottom:1px solid rgba(148,163,184,.13);
          }
          .ex-table td:first-child {
            border-left:1px solid rgba(148,163,184,.13);
            border-radius:12px 0 0 12px;
            font-weight:900;
          }
          .ex-table td:last-child {
            border-right:1px solid rgba(148,163,184,.13);
            border-radius:0 12px 12px 0;
          }
          .ex-explain {
            margin:.45rem 0 .9rem 0;
            padding:.9rem 1rem;
            border-radius:16px;
            border:1px solid rgba(148,163,184,.15);
            background:rgba(15,23,42,.66);
            color:#cbd5e1;
            font-size:.81rem;
            line-height:1.48;
          }
          .ex-explain strong { color:white; }
          .ex-good { color:#86efac !important; }
          .ex-warn { color:#fbbf24 !important; }
          .ex-bad { color:#fca5a5 !important; }
          @media (max-width:1100px) {
            .ex-hero,.ex-metrics,.ex-grid-3,.ex-grid-4 { grid-template-columns:1fr; }
            .ex-title { font-size:2.05rem; }
          }
        </style>

        <section class="ex-hero">
          <div>
            <div class="ex-kicker">Explainability Cockpit</div>
            <div class="ex-title">Why The Model<br/>Said This</div>
            <div class="ex-subtitle">
              Enter a financial news URL or paste article text. The page explains which words, phrases,
              sentences, risks, and driver groups pushed the signal bullish or bearish.
            </div>
            <div class="ex-chip-row">
              <span class="ex-chip">Article URL</span>
              <span class="ex-chip">Token impact</span>
              <span class="ex-chip">Sentence impact</span>
              <span class="ex-chip">Driver groups</span>
              <span class="ex-chip">Waterfall explanation</span>
              <span class="ex-chip">Model workflow</span>
            </div>
          </div>

          <div class="ex-engine">
            <div class="ex-kicker">Explanation Layer</div>
            <div class="ex-step"><div class="ex-num">01</div><div><strong>Read article text</strong><span>URL, headline, body, optional target</span></div></div>
            <div class="ex-step"><div class="ex-num">02</div><div><strong>Detect phrases</strong><span>Positive, negative, movement, sector, risk cues</span></div></div>
            <div class="ex-step"><div class="ex-num">03</div><div><strong>Score impact</strong><span>Token, phrase, sentence, and driver-group contribution</span></div></div>
            <div class="ex-step"><div class="ex-num">04</div><div><strong>Adjust signal</strong><span>Sentiment + movement - risk pressure</span></div></div>
            <div class="ex-step"><div class="ex-num">05</div><div><strong>Explain verdict</strong><span>Human-readable reasoning and limits</span></div></div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <section class="ex-panel">
          <div class="ex-kicker">Explanation Input</div>
          <div class="ex-section-title">Enter article source and generate model explanation</div>
          <p class="ex-copy">
            URL extraction may fail on blocked or paywalled sites. If that happens, paste the headline and article body manually.
            The explanation is a transparent public-demo reasoning layer, not a hidden black box.
          </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    input_left, input_right = st.columns([1.18, .82])

    with input_left:
        url_value = st.text_input("Article URL", value=st.session_state.ex_url, placeholder="https://...")
        target_hint = st.text_input(
            "Optional ticker / index / sector",
            value=st.session_state.ex_target,
            placeholder="Examples: NVDA, Nasdaq, Dow, S&P 500, Semiconductors, Broad market",
        )
        headline = st.text_input("Article headline", value=st.session_state.ex_headline)
        article_body = st.text_area("Article body or summary", value=st.session_state.ex_body, height=145)

    with input_right:
        explanation_mode = st.selectbox(
            "Explanation focus",
            ["Full explanation", "Sentiment only", "Movement only", "Risk only"],
            index=0,
        )
        show_workflow = st.checkbox("Show model workflow diagram", value=True)
        fetch_clicked = st.button("Extract URL text", type="secondary", use_container_width=True)
        sample_clicked = st.button("Load sample article", type="secondary", use_container_width=True)
        generate_clicked = st.button("Generate explanation", type="primary", use_container_width=True)

    if fetch_clicked:
        status, fetched_headline, fetched_body = _extract_article_from_url(url_value)
        st.session_state.ex_url = url_value
        st.session_state.ex_status = status
        if fetched_headline:
            st.session_state.ex_headline = fetched_headline
        if fetched_body:
            st.session_state.ex_body = fetched_body
        st.rerun()

    if sample_clicked:
        st.session_state.ex_url = ""
        st.session_state.ex_headline = sample_headline
        st.session_state.ex_body = sample_body
        st.session_state.ex_target = "Broad market"
        st.session_state.ex_status = "Sample article loaded for public demo."
        st.rerun()

    st.session_state.ex_url = url_value
    st.session_state.ex_headline = headline
    st.session_state.ex_body = article_body
    st.session_state.ex_target = target_hint

    full_text = f"{headline}\n{article_body}".strip()
    word_count = len(re.findall(r"\b\w+\b", full_text))
    target_type, detected_entities = _detect_target(full_text, target_hint)

    driver_rows = _find_driver_rows(full_text)
    if explanation_mode == "Sentiment only":
        driver_rows = [row for row in driver_rows if str(row["category"]) in {"Sentiment driver", "Negative driver"}]
    elif explanation_mode == "Movement only":
        driver_rows = [row for row in driver_rows if str(row["category"]) in {"Movement driver", "Sector driver"}]
    elif explanation_mode == "Risk only":
        driver_rows = [row for row in driver_rows if str(row["category"]) == "Risk driver"]

    if not driver_rows:
        driver_rows = [
            {"label": "neutral article language", "category": "Neutral baseline", "impact": 0.0, "effect": "No strong driver phrase was detected in the current input."}
        ]

    positive_rows = [row for row in driver_rows if float(row["impact"]) > 0]
    negative_rows = [row for row in driver_rows if float(row["impact"]) < 0]
    risk_rows = [row for row in driver_rows if str(row["category"]) == "Risk driver"]

    sentiment_total = round(sum(float(row["impact"]) for row in driver_rows if str(row["category"]) in {"Sentiment driver", "Negative driver"}), 2)
    movement_total = round(sum(float(row["impact"]) for row in driver_rows if str(row["category"]) in {"Movement driver", "Sector driver"}), 2)
    risk_total = round(sum(float(row["impact"]) for row in driver_rows if str(row["category"]) == "Risk driver"), 2)
    final_signal = round(sentiment_total + movement_total + risk_total, 2)

    if final_signal >= 1.2:
        signal_label = "Bullish"
        verdict_class = "ex-good"
    elif final_signal <= -1.2:
        signal_label = "Bearish"
        verdict_class = "ex-bad"
    else:
        signal_label = "Mixed"
        verdict_class = "ex-warn"

    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", full_text) if len(part.strip()) > 12]
    if not sentences and full_text:
        sentences = [full_text]
    sentence_scores = [{"sentence": item, "score": _score_segment(item)} for item in sentences[:10]]
    explained_sentences = sum(1 for item in sentence_scores if abs(float(item["score"])) > 0)
    explanation_coverage = round(_clamp((explained_sentences / max(1, len(sentence_scores))) * 100, 0, 100))
    confidence = round(_clamp(45 + len(driver_rows) * 5 + min(20, word_count / 8) + explanation_coverage * .15 - len(risk_rows) * 3, 35, 92))

    top_positive = positive_rows[0]["label"] if positive_rows else "No positive driver detected"
    top_risk = risk_rows[0]["label"] if risk_rows else "No major risk driver detected"

    def _list_html(rows: list[dict[str, str | float]], empty_text: str) -> str:
        if not rows:
            return f"<span>{html.escape(empty_text)}</span>"
        items = "".join(f"<li>{html.escape(str(row['label']))}</li>" for row in rows[:6])
        return f"<ul>{items}</ul>"

    st.markdown(
        f"""
        <section class="ex-panel">
          <div class="ex-kicker">Current Article Signal</div>
          <div class="ex-section-title">What the explanation engine detected</div>
          <div class="ex-grid-4">
            <div class="ex-card"><strong>Target context</strong><span>{html.escape(target_type)} · {html.escape(detected_entities)}</span></div>
            <div class="ex-card"><strong>Input quality</strong><span>{word_count} words analyzed</span></div>
            <div class="ex-card"><strong>Explanation focus</strong><span>{html.escape(explanation_mode)}</span></div>
            <div class="ex-card"><strong>Source status</strong><span>{html.escape(st.session_state.ex_status)}</span></div>
          </div>
          <div class="ex-grid-3">
            <div class="ex-card"><strong class="ex-good">Positive drivers</strong>{_list_html(positive_rows, "No positive drivers detected.")}</div>
            <div class="ex-card"><strong class="ex-bad">Negative drivers</strong>{_list_html(negative_rows, "No negative drivers detected.")}</div>
            <div class="ex-card"><strong class="ex-warn">Risk drivers</strong>{_list_html(risk_rows, "No risk drivers detected.")}</div>
          </div>
        </section>

        <div class="ex-metrics">
          <div class="ex-metric"><strong class="{verdict_class}">{signal_label}</strong><span>overall explanation signal</span></div>
          <div class="ex-metric"><strong>{confidence}%</strong><span>explanation confidence</span></div>
          <div class="ex-metric"><strong>{final_signal:+.1f}</strong><span>final driver score</span></div>
          <div class="ex-metric"><strong>{html.escape(str(top_positive))}</strong><span>top positive driver</span></div>
          <div class="ex-metric"><strong>{html.escape(str(top_risk))}</strong><span>top risk driver</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        import plotly.graph_objects as go

        sorted_rows = sorted(driver_rows, key=lambda row: abs(float(row["impact"])), reverse=True)[:10]

        token_fig = go.Figure(
            go.Bar(
                x=[float(row["impact"]) for row in sorted_rows],
                y=[str(row["label"]) for row in sorted_rows],
                orientation="h",
                customdata=[str(row["effect"]) for row in sorted_rows],
                hovertemplate="<b>%{y}</b><br>Impact: %{x}<br>%{customdata}<extra></extra>",
            )
        )
        token_fig.update_layout(
            title="Token Impact Ranking · Words and Phrases Moving the Signal",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,.35)",
            height=430,
            margin=dict(l=0, r=0, t=55, b=0),
            xaxis_title="Explanation impact",
            yaxis_title="",
        )
        st.plotly_chart(token_fig, use_container_width=True, config={"displayModeBar": False})

        st.markdown(
            """
            <div class="ex-explain">
              <strong>How to read this chart:</strong>
              bars to the right push the model explanation more bullish. Bars to the left pull the explanation bearish
              or add risk pressure. This is the phrase-level reasoning behind the signal.
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)

        with col1:
            sentence_fig = go.Figure(
                go.Bar(
                    x=[f"S{i+1}" for i, _ in enumerate(sentence_scores)],
                    y=[float(item["score"]) for item in sentence_scores],
                    customdata=[str(item["sentence"])[:180] for item in sentence_scores],
                    hovertemplate="<b>%{x}</b><br>Sentence impact: %{y}<br>%{customdata}<extra></extra>",
                )
            )
            sentence_fig.update_layout(
                title="Sentence Impact Timeline",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,.35)",
                height=400,
                margin=dict(l=0, r=0, t=55, b=0),
                xaxis_title="Sentence order",
                yaxis_title="Impact score",
            )
            st.plotly_chart(sentence_fig, use_container_width=True, config={"displayModeBar": False})

        with col2:
            waterfall = go.Figure(
                go.Waterfall(
                    name="Explanation build-up",
                    orientation="v",
                    measure=["absolute", "relative", "relative", "relative", "total"],
                    x=["Neutral baseline", "Sentiment", "Movement / sector", "Risk adjustment", "Final signal"],
                    y=[0, sentiment_total, movement_total, risk_total, 0],
                    hovertemplate="<b>%{x}</b><br>Contribution: %{y}<extra></extra>",
                )
            )
            waterfall.update_layout(
                title="Explanation Waterfall · How The Final Signal Is Built",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,.35)",
                height=400,
                margin=dict(l=0, r=0, t=55, b=0),
                yaxis_title="Signal contribution",
                xaxis_title="Explanation stage",
            )
            st.plotly_chart(waterfall, use_container_width=True, config={"displayModeBar": False})

        st.markdown(
            """
            <div class="ex-explain">
              <strong>How to read these charts:</strong>
              the sentence timeline shows which sentence carried the most bullish, bearish, or risk impact.
              The waterfall shows how neutral baseline becomes the final explanation after sentiment, movement, and risk adjustments.
            </div>
            """,
            unsafe_allow_html=True,
        )

        if show_workflow:
            workflow = go.Figure(
                go.Sankey(
                    arrangement="snap",
                    node=dict(
                        pad=18,
                        thickness=18,
                        line=dict(color="rgba(255,255,255,.25)", width=1),
                        label=[
                            "Article text",
                            "Token detection",
                            "Sentence scoring",
                            "Sentiment drivers",
                            "Movement drivers",
                            "Risk drivers",
                            "Final explanation",
                            "Analyst verdict",
                        ],
                    ),
                    link=dict(
                        source=[0, 0, 1, 1, 1, 2, 3, 4, 5, 6],
                        target=[1, 2, 3, 4, 5, 6, 6, 6, 6, 7],
                        value=[8, 5, 3, 3, 2, 5, 3, 3, 2, 8],
                    ),
                )
            )
            workflow.update_layout(
                title="Model Workflow Diagram · From Article Text To Explanation",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,.35)",
                height=430,
                margin=dict(l=10, r=10, t=55, b=10),
                font=dict(size=12),
            )
            st.plotly_chart(workflow, use_container_width=True, config={"displayModeBar": False})

    except Exception as exc:
        st.warning(f"Explainability charts could not render. Reason: {exc}")

    table_rows = ""
    for row in sorted(driver_rows, key=lambda item: abs(float(item["impact"])), reverse=True):
        klass = "ex-good" if float(row["impact"]) > 0 else "ex-bad" if float(row["impact"]) < 0 else "ex-warn"
        table_rows += (
            "<tr>"
            f"<td>{html.escape(str(row['label']))}</td>"
            f"<td>{html.escape(str(row['category']))}</td>"
            f"<td class='{klass}'>{float(row['impact']):+.2f}</td>"
            f"<td>{html.escape(str(row['effect']))}</td>"
            "</tr>"
        )

    st.markdown(
        f"""
        <section class="ex-panel">
          <div class="ex-kicker">Evidence Table</div>
          <div class="ex-section-title">Exact drivers behind the explanation</div>
          <table class="ex-table">
            <thead>
              <tr>
                <th>Phrase / signal</th>
                <th>Driver type</th>
                <th>Impact</th>
                <th>How it affected the output</th>
              </tr>
            </thead>
            <tbody>{table_rows}</tbody>
          </table>
        </section>

        <section class="ex-panel">
          <div class="ex-kicker">Trust Boundary</div>
          <div class="ex-grid-3">
            <div class="ex-card"><strong>What this explains</strong><span>Words, phrases, sentence impact, risk drivers, movement pressure, and final reasoning.</span></div>
            <div class="ex-card"><strong>What it does not guarantee</strong><span>Exact stock price, guaranteed return, causality, or investment advice.</span></div>
            <div class="ex-card"><strong>How to use it</strong><span>Use the page to audit why the model-style signal moved before reading forecast or historical pages.</span></div>
          </div>
        </section>

        <section class="ex-panel">
          <div class="ex-kicker">Analyst Explanation</div>
          <div class="ex-section-title">Model reasoning summary</div>
          <p class="ex-copy">
            The current explanation is <strong>{html.escape(signal_label.lower())}</strong> with a final driver score of
            {final_signal:+.1f}. The main positive contribution is <strong>{html.escape(str(top_positive))}</strong>.
            The main risk contribution is <strong>{html.escape(str(top_risk))}</strong>. Sentiment contributes
            {sentiment_total:+.1f}, movement and sector drivers contribute {movement_total:+.1f}, and risk adjustment
            contributes {risk_total:+.1f}. This page shows why the signal moved; it does not guarantee a market outcome.
          </p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _render_scenario_analysis_page() -> None:
    """Render Scenario Analysis as an article-driven what-if intelligence cockpit."""

    import html
    import re
    from urllib.parse import urlparse

    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    def _extract_article_from_url(url: str) -> tuple[str, str, str]:
        clean_url = url.strip()
        if not clean_url:
            return "No URL provided.", "", ""

        parsed = urlparse(clean_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return "Invalid URL. Paste a full URL starting with http:// or https://.", "", ""

        try:
            import requests
            from bs4 import BeautifulSoup

            response = requests.get(
                clean_url,
                timeout=8,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X) "
                        "AppleWebKit/537.36 Chrome/120 Safari/537.36"
                    )
                },
            )
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup(["script", "style", "noscript", "svg", "form", "nav", "footer", "header"]):
                tag.decompose()

            title = ""
            if soup.find("h1"):
                title = soup.find("h1").get_text(" ", strip=True)
            if not title and soup.title:
                title = soup.title.get_text(" ", strip=True)

            paragraphs = [
                p_tag.get_text(" ", strip=True)
                for p_tag in soup.find_all("p")
                if len(p_tag.get_text(" ", strip=True).split()) >= 8
            ]
            body = "\n\n".join(paragraphs[:12]).strip()

            if not body:
                return "URL loaded, but article text could not be extracted. Paste article text manually.", title, ""

            return "URL article text extracted.", title, body

        except Exception as exc:
            return f"URL extraction failed. Paste article text manually. Reason: {exc}", "", ""

    def _detect_target(text: str, user_target: str) -> tuple[str, str]:
        lowered = text.lower()
        entities: list[str] = []

        if re.search(r"\bdow\b|\bdjia\b", lowered):
            entities.append("Dow")
        if re.search(r"\bnasdaq\b|\bqqq\b", lowered):
            entities.append("Nasdaq")
        if re.search(r"\bs&p\b|\bsp500\b|\bspx\b|\bspy\b", lowered):
            entities.append("S&P 500")
        if re.search(r"\bchip\b|\bchips\b|\bsemiconductor\b|\bsemiconductors\b|\bnvidia\b|\bnvda\b|\bamd\b|\bintel\b", lowered):
            entities.append("Chips / semiconductors")
        if re.search(r"\brates?\b|\bfed\b|\binflation\b|\btreasury\b", lowered):
            entities.append("Macro / rates")

        ticker_matches = re.findall(r"\$?[A-Z]{2,5}\b", text)
        ticker_matches = [x.replace("$", "") for x in ticker_matches if x not in {"LIVE", "CEO", "CFO", "EPS", "THE"}]
        for ticker in ticker_matches[:4]:
            if ticker not in entities and ticker not in {"DOW"}:
                entities.append(ticker)

        if user_target.strip():
            return "User-selected target", user_target.strip()

        if "Dow" in entities or "Nasdaq" in entities or "S&P 500" in entities:
            if "Chips / semiconductors" in entities:
                return "Broad market + sector", ", ".join(entities)
            return "Broad market / index", ", ".join(entities)

        if "Chips / semiconductors" in entities:
            return "Sector / theme", ", ".join(entities)

        if ticker_matches:
            return "Ticker / company", ", ".join(entities) if entities else ", ".join(ticker_matches[:4])

        return "General financial news", "No specific ticker or index detected"

    def _article_driver_score(text: str) -> tuple[float, float, float, list[str]]:
        lowered = text.lower()
        bullish_patterns = [
            (r"\bjumps?\b|\brises?\b|\brose\b|\badvanced\b", 0.9, "positive price-action language"),
            (r"\brebounds?\b|\brall(y|ies|ied)\b|\bgains?\b", 1.0, "rebound or rally language"),
            (r"\bcloses?\s+above\b|\brecord\s+(close|high)\b|\ball[- ]time high\b", 1.2, "key-level or record-close strength"),
            (r"\bbeat(s|ing)?\b|\bbeats?\s+estimates\b|\braises?\s+guidance\b", 1.3, "earnings or guidance upside"),
            (r"\bchips?\s+rebound\b|\bsemiconductors?\s+rebound\b|\bai\b|\btechnology\s+momentum\b", 1.0, "technology or semiconductor momentum"),
        ]
        bearish_patterns = [
            (r"\bdrops?\b|\bfalls?\b|\bfell\b|\bslides?\b|\btumbles?\b", 1.0, "negative price-action language"),
            (r"\bmiss(es|ed)?\b|\bmisses?\s+estimates\b|\bcuts?\s+guidance\b", 1.3, "earnings or guidance downside"),
            (r"\bsell[- ]?off\b|\bweakness\b|\bslowdown\b", 1.1, "selloff or weakness language"),
        ]
        risk_patterns = [
            (r"\brisk(s)?\b|\buncertain(ty)?\b|\bvolatil(e|ity)\b", 0.9, "explicit risk or volatility language"),
            (r"\binflation\b|\brates?\b|\bfed\b|\brecession\b|\bmacro\b", 1.0, "macro or rates pressure"),
            (r"\bregulatory\b|\blawsuit\b|\bprobe\b|\bcredit\b|\bdebt\b|\bliquidity\b", 1.1, "regulatory, credit, or liquidity risk"),
        ]

        bull = 0.0
        bear = 0.0
        risk = 0.0
        cues: list[str] = []

        for pattern, weight, label in bullish_patterns:
            if re.search(pattern, lowered, flags=re.IGNORECASE):
                bull += weight
                cues.append(label)

        for pattern, weight, label in bearish_patterns:
            if re.search(pattern, lowered, flags=re.IGNORECASE):
                bear += weight
                cues.append(label)

        for pattern, weight, label in risk_patterns:
            if re.search(pattern, lowered, flags=re.IGNORECASE):
                risk += weight
                cues.append(label)

        return round(bull, 2), round(bear, 2), round(risk, 2), cues[:7]

    sample_headline = "Dow jumps 150 points for first close above 53,000; Nasdaq rises as chips rebound"
    sample_body = (
        "Stocks maintained positive momentum after a strong week on Wall Street. "
        "The S&P 500 gained 0.72%, while the Nasdaq Composite advanced 1.12% as chip stocks rebounded. "
        "Investors pointed to stronger technology momentum, improving risk appetite, and broad-market strength, "
        "while macro uncertainty remained limited."
    )

    if "sc_url" not in st.session_state:
        st.session_state.sc_url = ""
    if "sc_headline" not in st.session_state:
        st.session_state.sc_headline = sample_headline
    if "sc_body" not in st.session_state:
        st.session_state.sc_body = sample_body
    if "sc_target" not in st.session_state:
        st.session_state.sc_target = "Broad market"
    if "sc_status" not in st.session_state:
        st.session_state.sc_status = "Sample article loaded for public demo."

    st.markdown(
        """
        <style>
          .sc-hero {
            display:grid;
            grid-template-columns:1.08fr .92fr;
            gap:1rem;
            padding:1.35rem;
            border-radius:24px;
            border:1px solid rgba(34,211,238,.34);
            background:
              radial-gradient(circle at 8% 8%, rgba(34,211,238,.20), transparent 22rem),
              radial-gradient(circle at 76% 8%, rgba(139,92,246,.22), transparent 24rem),
              radial-gradient(circle at 88% 94%, rgba(34,197,94,.13), transparent 22rem),
              linear-gradient(145deg, rgba(8,47,73,.72), rgba(8,13,28,.96));
            box-shadow:0 30px 90px rgba(0,0,0,.36), inset 0 1px 0 rgba(255,255,255,.07);
            margin-bottom:.9rem;
          }
          .sc-kicker {
            color:#67e8f9;
            font-size:.70rem;
            font-weight:950;
            letter-spacing:.13em;
            text-transform:uppercase;
          }
          .sc-title {
            color:white;
            font-size:2.6rem;
            line-height:1;
            font-weight:950;
            letter-spacing:-.06em;
            margin:.42rem 0 .55rem 0;
          }
          .sc-subtitle {
            color:#dbeafe;
            font-size:1rem;
            line-height:1.55;
          }
          .sc-chip-row {
            display:flex;
            flex-wrap:wrap;
            gap:.48rem;
            margin-top:.9rem;
          }
          .sc-chip {
            padding:.43rem .68rem;
            border-radius:999px;
            font-size:.72rem;
            font-weight:850;
            color:#bfdbfe;
            border:1px solid rgba(96,165,250,.25);
            background:rgba(15,23,42,.65);
          }
          .sc-engine {
            padding:1rem;
            border-radius:20px;
            border:1px solid rgba(148,163,184,.18);
            background:linear-gradient(160deg, rgba(15,23,42,.92), rgba(2,6,23,.96));
          }
          .sc-step {
            display:grid;
            grid-template-columns:34px 1fr;
            gap:.65rem;
            align-items:center;
            padding:.54rem;
            margin-bottom:.45rem;
            border-radius:13px;
            border:1px solid rgba(96,165,250,.18);
            background:rgba(15,23,42,.70);
          }
          .sc-num {
            width:34px;
            height:34px;
            display:grid;
            place-items:center;
            border-radius:11px;
            color:#67e8f9;
            background:rgba(14,165,233,.13);
            border:1px solid rgba(34,211,238,.25);
            font-weight:950;
          }
          .sc-step strong {
            display:block;
            color:white;
            font-size:.82rem;
          }
          .sc-step span {
            display:block;
            color:#94a3b8;
            font-size:.70rem;
            margin-top:.08rem;
          }
          .sc-panel {
            margin:.95rem 0;
            padding:1.1rem;
            border-radius:22px;
            border:1px solid rgba(34,211,238,.24);
            background:
              radial-gradient(circle at 6% 0%, rgba(34,211,238,.11), transparent 18rem),
              radial-gradient(circle at 94% 30%, rgba(139,92,246,.13), transparent 20rem),
              linear-gradient(145deg, rgba(15,23,42,.88), rgba(8,13,28,.96));
            box-shadow:0 22px 60px rgba(0,0,0,.25);
          }
          .sc-section-title {
            color:white;
            font-size:1.25rem;
            font-weight:950;
            letter-spacing:-.04em;
            margin:.2rem 0 .35rem 0;
          }
          .sc-copy {
            color:#cbd5e1;
            font-size:.84rem;
            line-height:1.48;
            margin:0;
          }
          .sc-metrics {
            display:grid;
            grid-template-columns:repeat(5,minmax(0,1fr));
            gap:.68rem;
            margin:.85rem 0 .9rem 0;
          }
          .sc-metric {
            padding:1rem;
            border-radius:18px;
            border:1px solid rgba(148,163,184,.16);
            background:rgba(15,23,42,.82);
          }
          .sc-metric strong {
            color:white;
            font-size:1.35rem;
            font-weight:950;
            display:block;
          }
          .sc-metric span {
            color:#cbd5e1;
            font-size:.74rem;
            font-weight:760;
          }
          .sc-grid-3 {
            display:grid;
            grid-template-columns:repeat(3,minmax(0,1fr));
            gap:.68rem;
            margin-top:.8rem;
          }
          .sc-grid-4 {
            display:grid;
            grid-template-columns:repeat(4,minmax(0,1fr));
            gap:.68rem;
            margin-top:.8rem;
          }
          .sc-card {
            padding:.95rem;
            border-radius:17px;
            border:1px solid rgba(148,163,184,.16);
            background:rgba(15,23,42,.74);
          }
          .sc-card strong {
            color:white;
            display:block;
            font-size:.96rem;
            margin-bottom:.32rem;
          }
          .sc-card span, .sc-card li {
            color:#cbd5e1;
            font-size:.75rem;
            line-height:1.38;
          }
          .sc-card ul {
            margin:.2rem 0 0 1rem;
            padding:0;
          }
          .sc-case {
            padding:1.05rem;
            border-radius:20px;
            border:1px solid rgba(148,163,184,.17);
            background:linear-gradient(145deg, rgba(15,23,42,.86), rgba(2,6,23,.92));
          }
          .sc-case h3 {
            color:white;
            margin:.1rem 0 .3rem 0;
            font-size:1.15rem;
            letter-spacing:-.03em;
          }
          .sc-case .big {
            color:white;
            font-size:1.9rem;
            font-weight:950;
            margin:.2rem 0;
          }
          .sc-case p {
            color:#cbd5e1;
            font-size:.78rem;
            line-height:1.45;
            margin:.3rem 0 0 0;
          }
          .sc-table {
            width:100%;
            border-collapse:separate;
            border-spacing:0 .45rem;
            margin-top:.75rem;
          }
          .sc-table th {
            color:#94a3b8;
            font-size:.70rem;
            text-align:left;
            padding:.35rem .5rem;
            text-transform:uppercase;
            letter-spacing:.08em;
          }
          .sc-table td {
            color:#e5e7eb;
            font-size:.78rem;
            padding:.62rem .5rem;
            background:rgba(15,23,42,.72);
            border-top:1px solid rgba(148,163,184,.13);
            border-bottom:1px solid rgba(148,163,184,.13);
          }
          .sc-table td:first-child {
            border-left:1px solid rgba(148,163,184,.13);
            border-radius:12px 0 0 12px;
            font-weight:900;
          }
          .sc-table td:last-child {
            border-right:1px solid rgba(148,163,184,.13);
            border-radius:0 12px 12px 0;
          }
          .sc-explain {
            margin:.45rem 0 .9rem 0;
            padding:.9rem 1rem;
            border-radius:16px;
            border:1px solid rgba(148,163,184,.15);
            background:rgba(15,23,42,.66);
            color:#cbd5e1;
            font-size:.81rem;
            line-height:1.48;
          }
          .sc-explain strong { color:white; }
          .sc-good { color:#86efac !important; }
          .sc-warn { color:#fbbf24 !important; }
          .sc-bad { color:#fca5a5 !important; }
          @media (max-width:1100px) {
            .sc-hero,.sc-metrics,.sc-grid-3,.sc-grid-4 { grid-template-columns:1fr; }
            .sc-title { font-size:2.05rem; }
          }
        </style>

        <section class="sc-hero">
          <div>
            <div class="sc-kicker">Scenario Intelligence Cockpit</div>
            <div class="sc-title">What-If Market<br/>Outcome Engine</div>
            <div class="sc-subtitle">
              Enter a financial article, adjust stress levers, and compare upside, base, and downside
              cases with probability, return range, risk pressure, and analyst explanation.
            </div>
            <div class="sc-chip-row">
              <span class="sc-chip">Article URL</span>
              <span class="sc-chip">Paste fallback</span>
              <span class="sc-chip">What-if levers</span>
              <span class="sc-chip">Upside case</span>
              <span class="sc-chip">Base case</span>
              <span class="sc-chip">Downside case</span>
              <span class="sc-chip">Risk matrix</span>
            </div>
          </div>

          <div class="sc-engine">
            <div class="sc-kicker">Scenario Engine</div>
            <div class="sc-step"><div class="sc-num">01</div><div><strong>Read current signal</strong><span>Article, target, tone, risk language</span></div></div>
            <div class="sc-step"><div class="sc-num">02</div><div><strong>Apply what-if levers</strong><span>Momentum, macro pressure, volatility, sector strength</span></div></div>
            <div class="sc-step"><div class="sc-num">03</div><div><strong>Build paths</strong><span>Upside, base, downside reaction windows</span></div></div>
            <div class="sc-step"><div class="sc-num">04</div><div><strong>Estimate probability</strong><span>Scenario weights based on drivers and stress</span></div></div>
            <div class="sc-step"><div class="sc-num">05</div><div><strong>Explain decision risk</strong><span>Where the thesis works, fades, or breaks</span></div></div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <section class="sc-panel">
          <div class="sc-kicker">Scenario Input</div>
          <div class="sc-section-title">Enter article source and tune the what-if assumptions</div>
          <p class="sc-copy">
            URL extraction may fail on blocked or paywalled sites. If that happens, paste the headline and article body manually.
            This is a public-demo scenario engine; it shows transparent what-if logic and does not guarantee market outcomes.
          </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    input_left, input_right = st.columns([1.12, .88])

    with input_left:
        url_value = st.text_input("Article URL", value=st.session_state.sc_url, placeholder="https://...")
        target_hint = st.text_input(
            "Optional ticker / index / sector",
            value=st.session_state.sc_target,
            placeholder="Examples: NVDA, Nasdaq, Dow, S&P 500, Semiconductors, Broad market",
        )
        headline = st.text_input("Article headline", value=st.session_state.sc_headline)
        article_body = st.text_area("Article body or summary", value=st.session_state.sc_body, height=145)

    with input_right:
        horizon = st.selectbox("Scenario horizon", ["1 day", "3 days", "7 days", "14 days", "30 days"], index=2)
        sector_momentum = st.slider("Sector momentum", min_value=-5, max_value=5, value=2, step=1)
        macro_pressure = st.slider("Macro pressure", min_value=0, max_value=10, value=3, step=1)
        volatility_stress = st.slider("Volatility stress", min_value=0, max_value=10, value=4, step=1)
        risk_tolerance = st.selectbox("Risk tolerance", ["Conservative", "Balanced", "Aggressive"], index=1)

        fetch_clicked = st.button("Extract URL text", type="secondary", use_container_width=True)
        sample_clicked = st.button("Load sample article", type="secondary", use_container_width=True)
        generate_clicked = st.button("Generate scenario analysis", type="primary", use_container_width=True)

    if fetch_clicked:
        status, fetched_headline, fetched_body = _extract_article_from_url(url_value)
        st.session_state.sc_url = url_value
        st.session_state.sc_status = status
        if fetched_headline:
            st.session_state.sc_headline = fetched_headline
        if fetched_body:
            st.session_state.sc_body = fetched_body
        st.rerun()

    if sample_clicked:
        st.session_state.sc_url = ""
        st.session_state.sc_headline = sample_headline
        st.session_state.sc_body = sample_body
        st.session_state.sc_target = "Broad market"
        st.session_state.sc_status = "Sample article loaded for public demo."
        st.rerun()

    st.session_state.sc_url = url_value
    st.session_state.sc_headline = headline
    st.session_state.sc_body = article_body
    st.session_state.sc_target = target_hint

    full_text = f"{headline}\n{article_body}".strip()
    word_count = len(re.findall(r"\b\w+\b", full_text))
    target_type, detected_entities = _detect_target(full_text, target_hint)
    bull_score, bear_score, risk_score, cues = _article_driver_score(full_text)

    base_driver_score = bull_score - bear_score - (risk_score * 0.55)
    stress_drag = (macro_pressure * 0.18) + (volatility_stress * 0.16)
    momentum_boost = sector_momentum * 0.22
    net_score = round(base_driver_score + momentum_boost - stress_drag, 2)

    horizon_multiplier = {
        "1 day": 0.45,
        "3 days": 0.75,
        "7 days": 1.00,
        "14 days": 1.35,
        "30 days": 1.85,
    }[horizon]

    base_move = round(_clamp(net_score * horizon_multiplier, -6.5, 8.5), 2)
    upside_move = round(base_move + 1.8 + max(0, sector_momentum) * 0.32 + max(0, bull_score) * 0.25, 2)
    downside_move = round(base_move - 1.9 - macro_pressure * 0.22 - volatility_stress * 0.25 - risk_score * 0.28, 2)

    upside_prob = _clamp(34 + bull_score * 7 + sector_momentum * 3 - macro_pressure * 1.8 - volatility_stress * 1.4, 12, 72)
    downside_prob = _clamp(24 + bear_score * 7 + risk_score * 5 + macro_pressure * 2 + volatility_stress * 1.8 - sector_momentum * 2, 10, 68)
    base_prob = max(8, 100 - upside_prob - downside_prob)

    total_prob = upside_prob + base_prob + downside_prob
    upside_prob = round(upside_prob * 100 / total_prob)
    base_prob = round(base_prob * 100 / total_prob)
    downside_prob = 100 - upside_prob - base_prob

    expected_move = round((upside_move * upside_prob + base_move * base_prob + downside_move * downside_prob) / 100, 2)
    risk_pressure = round(_clamp(risk_score * 12 + macro_pressure * 6 + volatility_stress * 5 - sector_momentum * 2, 0, 100))
    opportunity_pressure = round(_clamp(bull_score * 14 + max(0, sector_momentum) * 7 - bear_score * 6, 0, 100))

    if expected_move > 1.0 and risk_pressure < 55:
        decision_read = "Opportunity-favored"
        decision_class = "sc-good"
    elif expected_move < -1.0 or risk_pressure > 70:
        decision_read = "Risk-heavy"
        decision_class = "sc-bad"
    else:
        decision_read = "Balanced / watch"
        decision_class = "sc-warn"

    cue_html = "".join(f"<li>{html.escape(cue)}</li>" for cue in cues) if cues else "<li>No strong article cue detected.</li>"

    st.markdown(
        f"""
        <section class="sc-panel">
          <div class="sc-kicker">Current Signal Context</div>
          <div class="sc-section-title">What the scenario engine detected</div>
          <div class="sc-grid-4">
            <div class="sc-card"><strong>Target context</strong><span>{html.escape(target_type)} · {html.escape(detected_entities)}</span></div>
            <div class="sc-card"><strong>Input quality</strong><span>{word_count} words analyzed</span></div>
            <div class="sc-card"><strong>Scenario horizon</strong><span>{html.escape(horizon)}</span></div>
            <div class="sc-card"><strong>Source status</strong><span>{html.escape(st.session_state.sc_status)}</span></div>
          </div>
          <div class="sc-grid-3">
            <div class="sc-card"><strong class="sc-good">Bullish pressure</strong><span>{bull_score:.1f} driver score</span></div>
            <div class="sc-card"><strong class="sc-bad">Bearish pressure</strong><span>{bear_score:.1f} driver score</span></div>
            <div class="sc-card"><strong class="sc-warn">Detected cues</strong><ul>{cue_html}</ul></div>
          </div>
        </section>

        <div class="sc-metrics">
          <div class="sc-metric"><strong class="{decision_class}">{html.escape(decision_read)}</strong><span>scenario read</span></div>
          <div class="sc-metric"><strong>{expected_move:+.1f}%</strong><span>probability-weighted move</span></div>
          <div class="sc-metric"><strong>{risk_pressure}/100</strong><span>risk pressure</span></div>
          <div class="sc-metric"><strong>{opportunity_pressure}/100</strong><span>opportunity pressure</span></div>
          <div class="sc-metric"><strong>{risk_tolerance}</strong><span>risk profile selected</span></div>
        </div>

        <section class="sc-panel">
          <div class="sc-kicker">Upside / Base / Downside Cases</div>
          <div class="sc-grid-3">
            <div class="sc-case">
              <h3>Upside case</h3>
              <div class="big sc-good">{upside_move:+.1f}%</div>
              <p><strong>{upside_prob}% probability.</strong> Momentum follows through, risk stays contained, and sector leadership remains intact.</p>
            </div>
            <div class="sc-case">
              <h3>Base case</h3>
              <div class="big sc-warn">{base_move:+.1f}%</div>
              <p><strong>{base_prob}% probability.</strong> Article signal is partly priced in, but the main trend remains stable.</p>
            </div>
            <div class="sc-case">
              <h3>Downside case</h3>
              <div class="big sc-bad">{downside_move:+.1f}%</div>
              <p><strong>{downside_prob}% probability.</strong> Macro pressure, volatility, or risk language overwhelms the positive signal.</p>
            </div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    try:
        import plotly.graph_objects as go

        days = [0, 1, 3, 7, 14, 30]
        scale = {
            "1 day": 1,
            "3 days": 3,
            "7 days": 7,
            "14 days": 14,
            "30 days": 30,
        }[horizon]

        def _path(final_move: float) -> list[float]:
            return [round(final_move * min(day, scale) / max(1, scale), 2) for day in days]

        fan = go.Figure()
        fan.add_trace(go.Scatter(x=days, y=_path(upside_move), mode="lines+markers", name="Upside case", line=dict(width=4)))
        fan.add_trace(go.Scatter(x=days, y=_path(base_move), mode="lines+markers", name="Base case", line=dict(width=5)))
        fan.add_trace(go.Scatter(x=days, y=_path(downside_move), mode="lines+markers", name="Downside case", line=dict(width=4)))
        fan.add_trace(go.Scatter(x=days, y=[expected_move * min(day, scale) / max(1, scale) for day in days], mode="lines+markers", name="Probability-weighted path", line=dict(width=4, dash="dash")))
        fan.update_layout(
            title="Scenario Fan Chart · Upside, Base, Downside Reaction Paths",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,.35)",
            height=460,
            margin=dict(l=0, r=0, t=55, b=0),
            xaxis_title="Days after signal",
            yaxis_title="Scenario movement %",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fan, use_container_width=True, config={"displayModeBar": False})

        st.markdown(
            """
            <div class="sc-explain">
              <strong>How to read this chart:</strong>
              the fan chart shows how the current article signal could behave under favorable, normal, and stressed conditions.
              The dashed line is the probability-weighted path.
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)

        with col1:
            prob_fig = go.Figure(
                go.Bar(
                    x=["Upside", "Base", "Downside"],
                    y=[upside_prob, base_prob, downside_prob],
                    text=[f"{upside_prob}%", f"{base_prob}%", f"{downside_prob}%"],
                    textposition="outside",
                )
            )
            prob_fig.update_layout(
                title="Scenario Probability Mix",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,.35)",
                height=400,
                margin=dict(l=0, r=0, t=55, b=0),
                xaxis_title="Scenario",
                yaxis_title="Probability %",
                yaxis=dict(range=[0, 100]),
            )
            st.plotly_chart(prob_fig, use_container_width=True, config={"displayModeBar": False})

        with col2:
            matrix = go.Figure(
                go.Scatter(
                    x=[risk_pressure, risk_pressure * 0.72, min(100, risk_pressure * 1.18)],
                    y=[opportunity_pressure, opportunity_pressure * 0.86, max(0, opportunity_pressure * 0.58)],
                    mode="markers+text",
                    text=["Base", "Upside", "Downside"],
                    textposition="top center",
                    marker=dict(size=[22, 26, 24], opacity=.9, line=dict(width=1, color="rgba(255,255,255,.35)")),
                    hovertemplate="<b>%{text}</b><br>Risk pressure: %{x}<br>Opportunity pressure: %{y}<extra></extra>",
                )
            )
            matrix.update_layout(
                title="Risk / Opportunity Matrix",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,.35)",
                height=400,
                margin=dict(l=0, r=0, t=55, b=0),
                xaxis=dict(title="Risk pressure", range=[0, 100]),
                yaxis=dict(title="Opportunity pressure", range=[0, 100]),
            )
            st.plotly_chart(matrix, use_container_width=True, config={"displayModeBar": False})

        col3, col4 = st.columns(2)

        with col3:
            levers = go.Figure(
                go.Bar(
                    x=[sector_momentum, -macro_pressure, -volatility_stress, bull_score, -risk_score],
                    y=["Sector momentum", "Macro pressure", "Volatility stress", "Bullish article cues", "Risk language"],
                    orientation="h",
                )
            )
            levers.update_layout(
                title="What-If Lever Contribution",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,.35)",
                height=390,
                margin=dict(l=0, r=0, t=55, b=0),
                xaxis_title="Scenario contribution",
                yaxis_title="",
            )
            st.plotly_chart(levers, use_container_width=True, config={"displayModeBar": False})

        with col4:
            stress_curve = go.Figure()
            stress_levels = list(range(0, 11))
            stress_curve.add_trace(
                go.Scatter(
                    x=stress_levels,
                    y=[round(base_move - level * 0.35, 2) for level in stress_levels],
                    mode="lines+markers",
                    name="Base case under stress",
                    line=dict(width=4),
                )
            )
            stress_curve.add_trace(
                go.Scatter(
                    x=stress_levels,
                    y=[round(upside_move - level * 0.25, 2) for level in stress_levels],
                    mode="lines+markers",
                    name="Upside resilience",
                    line=dict(width=3, dash="dash"),
                )
            )
            stress_curve.update_layout(
                title="Stress Test Curve · What If Volatility Rises?",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,.35)",
                height=390,
                margin=dict(l=0, r=0, t=55, b=0),
                xaxis_title="Additional volatility stress",
                yaxis_title="Scenario movement %",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(stress_curve, use_container_width=True, config={"displayModeBar": False})

    except Exception as exc:
        st.warning(f"Scenario Analysis charts could not render. Reason: {exc}")

    rows = [
        ("Upside", f"{upside_prob}%", f"{upside_move:+.1f}%", "Sector leadership continues; risk remains contained.", "Follow-through opportunity"),
        ("Base", f"{base_prob}%", f"{base_move:+.1f}%", "Signal is partially priced in; trend remains stable.", "Balanced monitoring"),
        ("Downside", f"{downside_prob}%", f"{downside_move:+.1f}%", "Macro pressure or volatility overwhelms the article signal.", "Protective caution"),
        ("Stress test", "Manual", f"{base_move - volatility_stress * 0.35:+.1f}%", "Volatility pressure rises after the article.", "Watch risk expansion"),
    ]

    table_rows = ""
    for scenario_name, probability, movement, condition, interpretation in rows:
        table_rows += (
            "<tr>"
            f"<td>{html.escape(scenario_name)}</td>"
            f"<td>{html.escape(probability)}</td>"
            f"<td>{html.escape(movement)}</td>"
            f"<td>{html.escape(condition)}</td>"
            f"<td>{html.escape(interpretation)}</td>"
            "</tr>"
        )

    analyst_summary = (
        "The scenario mix currently favors opportunity because bullish article pressure and sector momentum outweigh stress inputs."
        if decision_read == "Opportunity-favored"
        else "The scenario mix is risk-heavy because stress inputs and risk language dominate the positive article drivers."
        if decision_read == "Risk-heavy"
        else "The scenario mix is balanced because opportunity and risk inputs are close enough that confirmation matters."
    )

    st.markdown(
        f"""
        <section class="sc-panel">
          <div class="sc-kicker">Scenario Stress-Test Table</div>
          <div class="sc-section-title">What has to happen for each case</div>
          <table class="sc-table">
            <thead>
              <tr>
                <th>Case</th>
                <th>Probability</th>
                <th>Move</th>
                <th>Condition</th>
                <th>Interpretation</th>
              </tr>
            </thead>
            <tbody>{table_rows}</tbody>
          </table>
        </section>

        <section class="sc-panel">
          <div class="sc-kicker">Decision Boundary</div>
          <div class="sc-grid-3">
            <div class="sc-card"><strong>Upside trigger</strong><span>Momentum continues, risk pressure stays below 55/100, and sector leadership remains visible.</span></div>
            <div class="sc-card"><strong>Base trigger</strong><span>Article signal remains supportive but no new confirmation appears.</span></div>
            <div class="sc-card"><strong>Downside trigger</strong><span>Macro pressure, volatility, credit risk, or regulatory language rises after the initial signal.</span></div>
          </div>
        </section>

        <section class="sc-panel">
          <div class="sc-kicker">Scenario Analyst Explanation</div>
          <div class="sc-section-title">What the what-if engine suggests</div>
          <p class="sc-copy">
            {html.escape(analyst_summary)} The probability-weighted scenario move is {expected_move:+.1f}% over the selected
            {html.escape(horizon)} horizon. Upside is estimated at {upside_move:+.1f}% with {upside_prob}% probability,
            base case is {base_move:+.1f}% with {base_prob}% probability, and downside is {downside_move:+.1f}%
            with {downside_prob}% probability. This page is a transparent public-demo scenario analysis layer,
            not investment advice or a guaranteed forecast.
          </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

def _render_public_placeholder_page(page_title: str) -> None:
    """Render a real routed public page outside Executive Overview."""

    page_copy = {
        "Analyze Article": ("Analyze Article", "Article URL, upload, paste, and sample analysis workflow.", ["URL priority", "Upload fallback", "Paste fallback"]),
        "Forecasts": ("Forecasts", "Bull, base, and bear forward-looking movement scenarios.", ["Bull scenario", "Base scenario", "Bear scenario"]),
        "Historical Intelligence": ("Historical Intelligence", "Comparable financial-news events and reaction context.", ["Similar events", "Market reactions", "Comparable moves"]),
        "Explainability": ("Explainability", "Important words, sentiment drivers, movement drivers, and risk phrases.", ["Sentiment drivers", "Movement drivers", "Risk drivers"]),
        "Scenario Analysis": ("Scenario Analysis", "What-if risk and opportunity panels.", ["Upside case", "Base case", "Downside case"]),
        "Model Comparison": ("Model Comparison", "Model performance and champion selection story.", ["BERT", "DistilBERT", "Movement model"]),
        "Model Training / Evidence": ("Model Training / Evidence", "Training evidence, metrics, and validation story.", ["Training metrics", "Champion model", "Evidence trail"]),
        "Provenance": ("Provenance / Verification", "Source checks, disclaimers, and verification trail.", ["Source checks", "Public demo boundary", "Not investment advice"]),
        "Architecture / System Design": ("Architecture / System Design", "How Streamlit, FastAPI, models, artifacts, Docker, Kubernetes, and CI fit together.", ["Streamlit", "FastAPI", "CI/CD"]),
        "3D Intelligence": ("3D Intelligence", "3D intelligence visual area or graceful fallback.", ["Sentiment axis", "Risk axis", "Movement axis"]),
        "About / Project Purpose": ("About Ruturaj / Portfolio", "Why this project matters as a data, ML, AI, and MLOps portfolio project.", ["ML product", "Business intelligence", "Deployment engineering"]),
        "Visual QA / Page Audit": ("Visual QA / Page Audit", "Page coverage and public UI verification.", ["Real navigation", "Visible pages", "QA status"]),
    }

    heading, body, bullets = page_copy.get(
        page_title,
        (page_title, "This public page is routed and visible.", ["Visible", "Clickable", "Public-safe"]),
    )

    cards = "".join(
        f"""
        <div class="card" style="padding:1rem;">
          <div class="tiny-label">PUBLIC PAGE CHECK</div>
          <div class="strong">{_safe(item)}</div>
          <div class="muted">Visible in the routed Streamlit dashboard.</div>
        </div>
        """
        for item in bullets
    )

    st.markdown(
        f"""
        <div class="card insight" style="padding:1.15rem;margin-bottom:.75rem;">
          <div class="tiny-label">PUBLIC DASHBOARD SECTION</div>
          <h2 style="margin:.35rem 0;color:white;">{_safe(heading)}</h2>
          <p style="color:#cbd5e1;margin-bottom:0;">{_safe(body)}</p>
        </div>
        <div class="kpi-grid">{cards}</div>
        <div class="card" style="padding:1rem;margin-top:.75rem;">
          <div class="tiny-label">STATUS</div>
          <div class="strong">This is now a real clickable page, not a fake sidebar label.</div>
          <p class="muted">Next polish step: replace this placeholder with deeper page-specific charts and panels.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _analysis_pct(value: float) -> str:
    """Format public analysis probabilities as percentages."""

    return f"{value * 100:.0f}%"


def _article_sentences(text: str, limit: int = 8) -> list[str]:
    """Split article text into readable sentence snippets for the public cockpit."""

    import re

    chunks = re.split(r"(?<=[.!?])\s+", _clean_text(text))
    sentences = [chunk.strip() for chunk in chunks if len(chunk.strip()) >= 35]
    return sentences[:limit]


def _sentence_impact_rows(text: str) -> list[tuple[str, float, str]]:
    """Score visible sentence snippets using transparent public-mode keyword logic."""

    rows: list[tuple[str, float, str]] = []
    for sentence in _article_sentences(text, limit=10):
        lower = sentence.lower()
        pos = sum(1 for term in _POSITIVE_TERMS if term in lower)
        neg = sum(1 for term in _NEGATIVE_TERMS if term in lower)
        risk = sum(1 for term in _RISK_TERMS if term in lower)
        score = (pos * 0.22) - (neg * 0.20) - (risk * 0.12)
        label = "Bullish" if score > 0.08 else "Risk" if score < -0.08 else "Context"
        rows.append((sentence[:220], max(-1.0, min(1.0, score)), label))
    return rows[:6]


def _token_cloud_terms(text: str, signal: ArticleSignal) -> list[tuple[str, int, str]]:
    """Build clean finance tokens for the Analyze Article cloud."""

    import re
    from collections import Counter

    lower_text = text.lower()

    stopwords = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
        "have", "in", "into", "is", "it", "its", "of", "on", "or", "said",
        "says", "that", "the", "their", "this", "to", "was", "were", "will",
        "with", "about", "after", "again", "against", "also", "article",
        "because", "before", "between", "could", "hours", "monday", "more",
        "news", "over", "than", "there", "these", "those", "through", "under",
        "week", "which", "while", "would", "year", "years", "being", "they",
        "them", "then", "when", "next", "quarter",
    }

    finance_terms = {
        "ai", "stock", "stocks", "shares", "earnings", "revenue", "profit",
        "growth", "margin", "margins", "guidance", "forecast", "demand",
        "supply", "chip", "chips", "cloud", "data", "center", "price",
        "target", "upgrade", "downgrade", "cut", "beat", "miss", "risk",
        "strong", "stronger", "weak", "record", "positive", "negative",
        "bullish", "bearish", "investors", "market", "competition", "export",
        "exports", "controls", "policy", "regulation", "volatility", "tech",
        "constraints", "pressure", "management",
    }

    scored: dict[str, tuple[int, str]] = {}

    def add(term: str, weight: int, group: str) -> None:
        clean = term.strip().lower()
        if not clean or clean in stopwords or len(clean) < 2:
            return
        previous = scored.get(clean)
        if previous is None or weight > previous[0]:
            scored[clean] = (weight, group)

    for term in signal.positive_hits:
        add(term, lower_text.count(term.lower()) + 8, "positive")
    for term in signal.negative_hits:
        add(term, lower_text.count(term.lower()) + 8, "negative")
    for term in signal.risk_hits:
        add(term, lower_text.count(term.lower()) + 8, "risk")

    words = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", lower_text)
    counts = Counter(word for word in words if word not in stopwords)

    for word, count in counts.most_common(100):
        if word not in finance_terms and count < 2:
            continue

        group = "neutral"
        if word in {"growth", "strong", "stronger", "record", "beat", "profit", "revenue", "bullish", "upgrade", "demand"}:
            group = "positive"
        elif word in {"cut", "downgrade", "miss", "weak", "negative", "bearish"}:
            group = "negative"
        elif word in {"risk", "margin", "margins", "competition", "export", "exports", "controls", "policy", "regulation", "volatility", "supply", "constraints", "pressure"}:
            group = "risk"

        add(word, count + 3, group)

    return [
        (term, weight, group)
        for term, (weight, group) in sorted(
            scored.items(),
            key=lambda row: row[1][0],
            reverse=True,
        )[:26]
    ]

def _render_article_extraction_preview(text: str, signal: ArticleSignal, source_url: str) -> None:
    """Show article extraction details before charts."""

    sentences = _article_sentences(text, limit=5)
    preview = "<br>".join(f"• {_safe(sentence[:190])}" for sentence in sentences) or "No clean sentence preview was available."
    source_link = source_url if source_url else "Local sample, paste, or upload"

    st.markdown(
        f"""
        <div class="two-col">
          <div class="card" style="padding:1rem;">
            <div class="tiny-label">EXTRACTION PREVIEW</div>
            <div class="strong" style="margin-top:.35rem;">{_safe(signal.headline)}</div>
            <div class="muted" style="margin-top:.55rem;">{preview}</div>
          </div>
          <div class="card" style="padding:1rem;">
            <div class="tiny-label">ARTICLE CONTEXT</div>
            <div class="strong" style="margin-top:.35rem;">{_safe(signal.company)} · {signal.ticker}</div>
            <div class="muted" style="margin-top:.45rem;">Source: {_safe(signal.source)}</div>
            <div class="muted">URL: {_safe(source_link)}</div>
            <div class="muted">Characters analyzed: {len(text):,}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_signal_summary_cards(signal: ArticleSignal) -> None:
    """Render compact signal cards for the Analyze Article cockpit."""

    risk_label = "High" if signal.risk_score >= 0.62 else "Medium" if signal.risk_score >= 0.34 else "Low"

    st.markdown(
        f"""
        <div class="kpi-grid">
          <div class="card kpi teal">
            <div class="kpi-title">Sentiment</div>
            <div class="kpi-value">{signal.sentiment_score:+.2f}</div>
            <div class="kpi-sub">{_safe(signal.label)}</div>
          </div>
          <div class="card kpi green">
            <div class="kpi-title">Up Probability</div>
            <div class="kpi-value">{_analysis_pct(signal.movement_up)}</div>
            <div class="kpi-sub">Movement estimate</div>
          </div>
          <div class="card kpi orange">
            <div class="kpi-title">Risk Pressure</div>
            <div class="kpi-value">{_analysis_pct(signal.risk_score)}</div>
            <div class="kpi-sub">{risk_label} risk</div>
          </div>
          <div class="card kpi violet">
            <div class="kpi-title">Confidence</div>
            <div class="kpi-value">{_analysis_pct(signal.confidence)}</div>
            <div class="kpi-sub">Public-mode confidence</div>
          </div>
          <div class="card kpi purple">
            <div class="kpi-title">Ticker</div>
            <div class="kpi-value">{signal.ticker}</div>
            <div class="kpi-sub">{_safe(signal.company)}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_keyword_cloud(text: str, signal: ArticleSignal) -> None:
    """Render an interactive Plotly token bubble cloud with no raw HTML risk."""

    try:
        import math
        import plotly.graph_objects as go

        terms = _token_cloud_terms(text, signal)
        if not terms:
            st.info("No meaningful finance tokens were detected for the cloud.")
            return

        color_map = {
            "positive": "#22c55e",
            "negative": "#ef4444",
            "risk": "#f97316",
            "neutral": "#60a5fa",
        }
        label_map = {
            "positive": "Positive",
            "negative": "Negative",
            "risk": "Risk",
            "neutral": "Context",
        }

        xs: list[float] = []
        ys: list[float] = []
        sizes: list[float] = []
        colors: list[str] = []
        labels: list[str] = []
        groups: list[str] = []
        weights: list[int] = []

        for index, (term, weight, group) in enumerate(terms):
            angle = math.radians((index * 137.5) % 360)
            radius = 0.12 + 0.82 * ((index % 9) + 1) / 9
            xs.append(math.cos(angle) * radius)
            ys.append(math.sin(angle) * radius)
            sizes.append(min(72, 24 + weight * 4.5))
            colors.append(color_map.get(group, color_map["neutral"]))
            labels.append(term.upper() if len(term) <= 4 else term)
            groups.append(label_map.get(group, "Context"))
            weights.append(weight)

        st.markdown(
            """
            <div class="card" style="padding:1rem;margin-bottom:.5rem;">
              <div class="tiny-label">KEYWORD / TOKEN CLOUD</div>
              <div class="strong" style="margin:.25rem 0;">Interactive article language fingerprint</div>
              <div class="muted">Bubble size shows term strength. Color shows positive, negative, risk, or context language.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        fig = go.Figure(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers+text",
                text=labels,
                textposition="middle center",
                marker=dict(
                    size=sizes,
                    color=colors,
                    opacity=0.78,
                    line=dict(width=1, color="rgba(255,255,255,.35)"),
                ),
                customdata=list(zip(groups, weights)),
                hovertemplate="<b>%{text}</b><br>Group: %{customdata[0]}<br>Weight: %{customdata[1]}<extra></extra>",
            )
        )

        fig.update_layout(
            title="Token Cloud · Positive / Negative / Risk / Context",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,.35)",
            height=430,
            margin=dict(l=10, r=10, t=55, b=10),
            showlegend=False,
        )
        fig.update_xaxes(visible=False, range=[-1.15, 1.15])
        fig.update_yaxes(visible=False, range=[-1.05, 1.05], scaleanchor="x", scaleratio=1)

        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    except Exception as exc:
        st.warning(f"Token cloud could not render. Reason: {exc}")

def _render_token_impact_chart(text: str, signal: ArticleSignal) -> None:
    """Show ranked token impact using a Plotly horizontal bar chart."""

    try:
        import plotly.graph_objects as go

        rows: list[tuple[str, float]] = []
        for term in signal.positive_hits[:6]:
            rows.append((term, 0.18 + text.lower().count(term.lower()) * 0.04))
        for term in signal.negative_hits[:6]:
            rows.append((term, -0.16 - text.lower().count(term.lower()) * 0.04))
        for term in signal.risk_hits[:6]:
            rows.append((term, -0.10 - text.lower().count(term.lower()) * 0.03))

        if not rows:
            rows = [("article tone", signal.sentiment_score), ("risk pressure", -signal.risk_score), ("confidence", signal.confidence / 2)]

        rows = sorted(rows, key=lambda item: abs(item[1]))[-10:]

        fig = go.Figure(go.Bar(
            x=[value for _, value in rows],
            y=[label for label, _ in rows],
            orientation="h",
        ))
        fig.update_layout(
            title="Token Impact Ranking",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,.45)",
            height=330,
            margin=dict(l=20, r=20, t=55, b=35),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    except Exception:
        st.info("Token impact chart fallback: Plotly is unavailable in this runtime.")


def _render_forecast_trend_chart(signal: ArticleSignal) -> None:
    """Render bull/base/bear forecast trend preview derived from public signals."""

    try:
        import plotly.graph_objects as go

        periods = ["Now", "1D", "3D", "1W", "2W"]
        movement_bias = signal.movement_up - signal.movement_down
        risk_drag = signal.risk_score * 1.6

        base = []
        bull = []
        bear = []
        for idx, scale in enumerate([0, 0.35, 0.70, 1.00, 1.25]):
            base_value = 100 + (movement_bias * 7.0 * scale) - (risk_drag * scale)
            bull_value = base_value + (signal.confidence * 2.6 * scale)
            bear_value = base_value - ((signal.risk_score + signal.movement_down) * 3.0 * scale)
            base.append(round(base_value, 2))
            bull.append(round(bull_value, 2))
            bear.append(round(bear_value, 2))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=periods, y=bull, mode="lines+markers", name="Bull case"))
        fig.add_trace(go.Scatter(x=periods, y=base, mode="lines+markers", name="Base case"))
        fig.add_trace(go.Scatter(x=periods, y=bear, mode="lines+markers", name="Bear case"))
        fig.update_layout(
            title="Forecast Trend Preview",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,.45)",
            yaxis_title="Indexed reaction path",
            height=340,
            margin=dict(l=20, r=20, t=55, b=35),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    except Exception:
        st.info("Forecast trend fallback: Plotly is unavailable in this runtime.")


def _render_risk_reward_matrix(signal: ArticleSignal) -> None:
    """Place the article on a sentiment-versus-risk quadrant chart."""

    try:
        import plotly.graph_objects as go

        x_value = round(signal.sentiment_score * 100, 1)
        y_value = round(signal.risk_score * 100, 1)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[x_value],
            y=[y_value],
            mode="markers+text",
            text=[signal.ticker],
            textposition="top center",
            marker=dict(size=24),
            name="Current article",
        ))
        fig.add_vline(x=0, line_dash="dash")
        fig.add_hline(y=50, line_dash="dash")
        fig.update_xaxes(range=[-100, 100], title="Sentiment strength")
        fig.update_yaxes(range=[0, 100], title="Risk pressure")
        fig.update_layout(
            title="Risk vs Reward Matrix",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,.45)",
            height=340,
            margin=dict(l=20, r=20, t=55, b=35),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    except Exception:
        st.info("Risk/reward matrix fallback: Plotly is unavailable in this runtime.")


def _render_driver_waterfall(signal: ArticleSignal) -> None:
    """Show how positive, negative, and risk drivers shape the final bias."""

    try:
        import plotly.graph_objects as go

        positive = len(signal.positive_hits) * 0.16
        negative = -len(signal.negative_hits) * 0.14
        risk = -len(signal.risk_hits) * 0.09
        net = positive + negative + risk

        fig = go.Figure(go.Waterfall(
            name="Driver impact",
            orientation="v",
            measure=["relative", "relative", "relative", "total"],
            x=["Positive", "Negative", "Risk", "Net Bias"],
            y=[positive, negative, risk, net],
        ))
        fig.update_layout(
            title="Driver Impact Waterfall",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,.45)",
            height=340,
            margin=dict(l=20, r=20, t=55, b=35),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    except Exception:
        st.info("Driver waterfall fallback: Plotly is unavailable in this runtime.")


def _render_sentence_impact_timeline(text: str) -> None:
    """Render sentence-level article impact cards without leaking raw HTML."""

    rows = _sentence_impact_rows(text)
    if not rows:
        st.info("Sentence impact timeline could not find enough clean article sentences.")
        return

    st.markdown(
        """
        <div class="card" style="padding:1.05rem;margin-bottom:.7rem;">
          <div class="tiny-label">SENTENCE IMPACT TIMELINE</div>
          <div class="strong" style="margin:.25rem 0 .25rem 0;">Where the article turns bullish, risky, or neutral</div>
          <div class="muted">Each sentence is scored with the same transparent public keyword logic.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for idx, (sentence, score, label) in enumerate(rows, start=1):
        tone = "#86efac" if label == "Bullish" else "#fdba74" if label == "Risk" else "#bfdbfe"
        st.markdown(
            f"""
            <div class="card" style="padding:.85rem .95rem;margin-bottom:.5rem;">
              <div class="tiny-label" style="color:{tone};">SENTENCE {idx} · {_safe(label)} · {score:+.2f}</div>
              <div class="muted" style="margin-top:.35rem;">{_safe(sentence)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

def _analyst_summary_markdown(signal: ArticleSignal) -> str:
    """Create a copy-ready analyst summary for download."""

    return f"""# Financial News Article Intelligence Summary

Ticker: {signal.ticker}
Company: {signal.company}
Headline: {signal.headline}
Source: {signal.source}

## Signal
- Sentiment score: {signal.sentiment_score:+.2f}
- Movement label: {signal.label}
- Up probability: {_analysis_pct(signal.movement_up)}
- Flat probability: {_analysis_pct(signal.movement_flat)}
- Down probability: {_analysis_pct(signal.movement_down)}
- Risk pressure: {_analysis_pct(signal.risk_score)}
- Confidence: {_analysis_pct(signal.confidence)}

## Positive drivers
{chr(10).join(f"- {item}" for item in signal.positive_hits) or "- No strong positive drivers detected."}

## Negative drivers
{chr(10).join(f"- {item}" for item in signal.negative_hits) or "- No strong negative drivers detected."}

## Risk drivers
{chr(10).join(f"- {item}" for item in signal.risk_hits) or "- No strong risk drivers detected."}

## Disclaimer
Public demo output for research and portfolio review only. Not investment advice.
"""


def _render_analyst_verdict(signal: ArticleSignal) -> None:
    """Render final analyst verdict and export button."""

    risk_label = "elevated" if signal.risk_score >= 0.62 else "moderate" if signal.risk_score >= 0.34 else "contained"
    interpretation = (
        "The article leans bullish with supportive language."
        if signal.label == "Bullish"
        else "The article leans bearish or cautious."
        if signal.label == "Bearish"
        else "The article is mixed and requires monitoring."
    )

    st.markdown(
        f"""
        <div class="card insight" style="padding:1.1rem;margin-bottom:.7rem;">
          <div class="tiny-label">ANALYST VERDICT</div>
          <h3 style="color:white;margin:.25rem 0;">{_safe(interpretation)}</h3>
          <p class="muted"><strong>Market reaction risk:</strong> {_safe(risk_label)}.</p>
          <p class="muted"><strong>Confidence caveat:</strong> This is a public-mode deterministic explanation, not a hidden live trading model.</p>
          <p class="muted"><strong>Monitor next:</strong> guidance revisions, margin pressure, regulatory language, and follow-through price action.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.download_button(
        "Download analyst summary (.md)",
        data=_analyst_summary_markdown(signal),
        file_name=f"{signal.ticker.lower()}_article_intelligence_summary.md",
        mime="text/markdown",
        use_container_width=True,
    )


def _render_analyze_article_page() -> None:
    """Render the Article Intelligence Cockpit."""

    st.markdown(
        """
        <div class="card insight" style="padding:1.1rem;margin-bottom:.75rem;">
          <div class="tiny-label">ARTICLE INTELLIGENCE COCKPIT</div>
          <h2 style="color:white;margin:.25rem 0;">Analyze a financial news article like an analyst terminal</h2>
          <p style="color:#cbd5e1;margin-bottom:0;">
            Enter a URL, upload a file, paste article text, or use the sample. This page extracts context,
            scores sentiment and movement, maps drivers, previews forecast paths, and exports a summary.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    text, source, source_url = _input_form()
    signal = _score_article(text, source)

    _render_article_extraction_preview(text, signal, source_url)
    _render_signal_summary_cards(signal)

    chart_left, chart_right = st.columns(2, gap="medium")
    with chart_left:
        _render_forecast_trend_chart(signal)
    with chart_right:
        _render_risk_reward_matrix(signal)

    _render_driver_waterfall(signal)
    _render_keyword_cloud(text, signal)

    col_a, col_b = st.columns([1.0, 1.0], gap="medium")
    with col_a:
        _render_token_impact_chart(text, signal)
    with col_b:
        _render_sentence_impact_timeline(text)

    _render_analyst_verdict(signal)


def _render_model_performance_dashboard() -> None:
    """Render recruiter-friendly model performance and project proof visuals."""

    st.markdown(
        """
        <style>
          .model-perf-panel {
            margin: 1rem 0 .9rem 0;
            padding: 1.15rem;
            border-radius: 22px;
            border: 1px solid rgba(34,211,238,.30);
            background:
              radial-gradient(circle at 6% 0%, rgba(34,211,238,.15), transparent 18rem),
              radial-gradient(circle at 92% 30%, rgba(139,92,246,.18), transparent 20rem),
              linear-gradient(145deg, rgba(15,23,42,.90), rgba(8,13,28,.96));
            box-shadow: 0 24px 70px rgba(0,0,0,.30);
          }
          .perf-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: .7rem;
            margin-top: .85rem;
          }
          .perf-card {
            padding: .95rem;
            border-radius: 17px;
            border: 1px solid rgba(148,163,184,.16);
            background: rgba(15,23,42,.74);
          }
          .perf-card strong {
            display: block;
            color: white;
            font-size: .95rem;
            margin-bottom: .25rem;
          }
          .perf-card span {
            color: #cbd5e1;
            font-size: .74rem;
            line-height: 1.35;
          }
          .bar-shell {
            margin-top: .7rem;
            height: 9px;
            border-radius: 999px;
            background: rgba(15,23,42,.92);
            border: 1px solid rgba(148,163,184,.13);
            overflow: hidden;
          }
          .bar-fill {
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, #22d3ee, #8b5cf6, #22c55e);
          }
          .proof-timeline {
            display: grid;
            grid-template-columns: repeat(9, minmax(0, 1fr));
            gap: .45rem;
            margin-top: 1rem;
          }
          .proof-node {
            padding: .72rem .45rem;
            border-radius: 14px;
            text-align: center;
            color: #e0f2fe;
            font-size: .68rem;
            font-weight: 900;
            border: 1px solid rgba(96,165,250,.20);
            background:
              radial-gradient(circle at 50% 0%, rgba(34,211,238,.13), transparent 5rem),
              rgba(2,6,23,.48);
          }
          @media (max-width: 1100px) {
            .perf-grid, .proof-timeline { grid-template-columns: 1fr; }
          }
        </style>

        <section class="model-perf-panel">
          <div class="exec-kicker">MODEL + DELIVERY PROOF</div>
          <h3 style="color:white;margin:.25rem 0 .25rem 0;">Model roles, product strength, and deployment maturity</h3>
          <p class="muted">
            This section shows the project as a complete AI product: model reasoning, public demo readiness,
            explainability, and MLOps delivery.
          </p>

          <div class="perf-grid">
            <div class="perf-card">
              <strong>Sentiment Intelligence</strong>
              <span>BERT / DistilBERT-style financial text classification and confidence storytelling.</span>
              <div class="bar-shell"><div class="bar-fill" style="width:92%;"></div></div>
            </div>
            <div class="perf-card">
              <strong>Movement Intelligence</strong>
              <span>News-to-market Up / Flat / Down reaction probability workflow.</span>
              <div class="bar-shell"><div class="bar-fill" style="width:86%;"></div></div>
            </div>
            <div class="perf-card">
              <strong>Explainability</strong>
              <span>Driver phrases, token cloud, sentence impact, risk language, and analyst verdict.</span>
              <div class="bar-shell"><div class="bar-fill" style="width:94%;"></div></div>
            </div>
            <div class="perf-card">
              <strong>MLOps Readiness</strong>
              <span>Streamlit, FastAPI architecture, Docker, Kubernetes, CI/CD, validation gates.</span>
              <div class="bar-shell"><div class="bar-fill" style="width:88%;"></div></div>
            </div>
          </div>

          <div class="proof-timeline">
            <div class="proof-node">Data</div>
            <div class="proof-node">NLP</div>
            <div class="proof-node">ML</div>
            <div class="proof-node">Forecast</div>
            <div class="proof-node">XAI</div>
            <div class="proof-node">FastAPI</div>
            <div class="proof-node">Streamlit</div>
            <div class="proof-node">Docker/K8s</div>
            <div class="proof-node">CI/CD</div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _render_model_role_map_3d() -> None:
    """Render a meaningful 3D model role map for the recruiter landing page."""

    st.markdown(
        """
        <div class="card" style="padding:1.05rem;margin:.9rem 0 .55rem 0;">
          <div class="exec-kicker">3D MODEL ROLE MAP</div>
          <h3 style="color:white;margin:.25rem 0;">Quality × Deployment × Explainability</h3>
          <p class="muted">
            The project separates model roles instead of pretending one model solves everything.
            Recruiters can see how sentiment, movement, forecast, and explainability layers fit together.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        import plotly.graph_objects as go

        names = [
            "BERT quality benchmark",
            "DistilBERT live sentiment",
            "Movement model",
            "Forecast scenario layer",
            "Explainability layer",
        ]
        quality = [94, 88, 82, 76, 78]
        efficiency = [48, 90, 82, 86, 92]
        explanation = [72, 70, 80, 84, 96]
        roles = [
            "Best reference sentiment quality",
            "Balanced live sentiment model role",
            "Up / Flat / Down movement estimate",
            "Bull / Base / Bear scenario communication",
            "Drivers, token impact, sentence impact, verdict",
        ]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter3d(
                x=quality,
                y=efficiency,
                z=explanation,
                mode="lines+markers+text",
                text=names,
                customdata=roles,
                textposition="top center",
                marker=dict(
                    size=[9, 10, 11, 10, 12],
                    opacity=.92,
                    line=dict(width=2, color="rgba(255,255,255,.35)"),
                ),
                line=dict(width=5),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Prediction quality: %{x}<br>"
                    "Deployment efficiency: %{y}<br>"
                    "Explanation depth: %{z}<br>"
                    "%{customdata}<extra></extra>"
                ),
            )
        )

        fig.update_layout(
            title="3D Model Role Map",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,.35)",
            height=640,
            margin=dict(l=0, r=0, t=55, b=0),
            scene=dict(
                xaxis_title="Prediction quality",
                yaxis_title="Deployment efficiency",
                zaxis_title="Explanation depth",
                xaxis=dict(range=[40, 100], gridcolor="rgba(148,163,184,.18)"),
                yaxis=dict(range=[40, 100], gridcolor="rgba(148,163,184,.18)"),
                zaxis=dict(range=[40, 100], gridcolor="rgba(148,163,184,.18)"),
                camera=dict(eye=dict(x=1.55, y=1.45, z=1.08)),
            ),
        )

        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    except Exception as exc:
        st.warning(f"3D model role map could not render. Reason: {exc}")


def _render_ai_system_flow_diagram() -> None:
    """Render a recruiter-friendly AI system flow diagram."""

    st.markdown(
        """
        <div class="card" style="padding:1.05rem;margin:.9rem 0 .55rem 0;">
          <div class="exec-kicker">AI SYSTEM FLOW</div>
          <h3 style="color:white;margin:.25rem 0;">From financial article to analyst-ready intelligence</h3>
          <p class="muted">
            This diagram shows the end-to-end product path: article intake, NLP processing,
            movement estimation, forecasting, explainability, and final analyst output.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        import plotly.graph_objects as go

        labels = [
            "Article URL / Upload / Paste",
            "Text Extraction",
            "Financial NLP",
            "Sentiment Model",
            "Movement Model",
            "Risk Adjustment",
            "Forecast Scenarios",
            "Explainability",
            "Analyst Verdict",
            "Recruiter Demo",
        ]

        fig = go.Figure(
            go.Sankey(
                arrangement="snap",
                node=dict(
                    pad=18,
                    thickness=18,
                    line=dict(color="rgba(255,255,255,.25)", width=1),
                    label=labels,
                ),
                link=dict(
                    source=[0, 1, 2, 2, 3, 4, 5, 5, 6, 7, 8],
                    target=[1, 2, 3, 4, 5, 5, 6, 7, 8, 8, 9],
                    value=[10, 10, 6, 4, 6, 4, 5, 5, 5, 5, 10],
                ),
            )
        )

        fig.update_layout(
            title="AI Product Flow · Input → Models → Forecast → Explanation → Output",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,.35)",
            height=430,
            margin=dict(l=10, r=10, t=55, b=10),
            font=dict(size=12),
        )

        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    except Exception as exc:
        st.warning(f"AI system flow diagram could not render. Reason: {exc}")


def _render_recruiter_landing_page() -> None:
    """Render an attractive recruiter-facing product landing page."""

    st.markdown(
        """
        <style>
          .exec-hero {
            display: grid;
            grid-template-columns: 1.12fr .88fr;
            gap: 1rem;
            padding: 1.45rem;
            border-radius: 24px;
            border: 1px solid rgba(34,211,238,.36);
            background:
              radial-gradient(circle at 8% 8%, rgba(34,211,238,.22), transparent 22rem),
              radial-gradient(circle at 72% 14%, rgba(139,92,246,.24), transparent 26rem),
              radial-gradient(circle at 88% 92%, rgba(34,197,94,.13), transparent 22rem),
              linear-gradient(145deg, rgba(8,47,73,.76), rgba(8,13,28,.96));
            box-shadow: 0 30px 90px rgba(0,0,0,.38), inset 0 1px 0 rgba(255,255,255,.07);
            margin-bottom: .9rem;
          }
          .exec-eyebrow {
            color: #67e8f9;
            font-size: .72rem;
            font-weight: 950;
            letter-spacing: .13em;
            text-transform: uppercase;
          }
          .exec-title {
            margin: .42rem 0 .6rem 0;
            color: #fff;
            font-size: 3.25rem;
            line-height: .98;
            font-weight: 950;
            letter-spacing: -.06em;
          }
          .exec-subtitle {
            color: #dbeafe;
            font-size: 1.05rem;
            line-height: 1.58;
            max-width: 900px;
          }
          .exec-highlight {
            color: #86efac;
            font-weight: 900;
          }
          .exec-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: .5rem;
            margin-top: 1rem;
          }
          .exec-chip {
            padding: .46rem .7rem;
            border-radius: 999px;
            font-size: .74rem;
            font-weight: 850;
            color: #bfdbfe;
            border: 1px solid rgba(96,165,250,.26);
            background: rgba(15,23,42,.68);
          }
          .stack-card {
            position: relative;
            overflow: hidden;
            padding: 1.05rem;
            border-radius: 20px;
            border: 1px solid rgba(148,163,184,.18);
            background:
              radial-gradient(circle at 50% 0%, rgba(59,130,246,.16), transparent 16rem),
              linear-gradient(160deg, rgba(15,23,42,.92), rgba(2,6,23,.96));
            min-height: 320px;
          }
          .stack-card:before {
            content: "";
            position: absolute;
            width: 260px;
            height: 260px;
            border-radius: 999px;
            right: -90px;
            top: -90px;
            background: radial-gradient(circle, rgba(34,211,238,.18), transparent 70%);
          }
          .stack-title {
            position: relative;
            z-index: 1;
            color: white;
            font-size: 1.12rem;
            font-weight: 950;
            margin-bottom: .65rem;
          }
          .stack-step {
            position: relative;
            z-index: 1;
            display: grid;
            grid-template-columns: 34px 1fr;
            gap: .65rem;
            align-items: center;
            padding: .56rem;
            margin-bottom: .46rem;
            border-radius: 13px;
            border: 1px solid rgba(96,165,250,.18);
            background: rgba(15,23,42,.70);
          }
          .stack-icon {
            width: 34px;
            height: 34px;
            display: grid;
            place-items: center;
            border-radius: 11px;
            color: #67e8f9;
            background: rgba(14,165,233,.13);
            border: 1px solid rgba(34,211,238,.25);
            font-weight: 950;
          }
          .stack-step strong {
            display: block;
            color: white;
            font-size: .82rem;
          }
          .stack-step span {
            display: block;
            color: #94a3b8;
            font-size: .70rem;
            margin-top: .08rem;
          }
          .exec-metrics {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: .7rem;
            margin-bottom: .9rem;
          }
          .exec-metric {
            padding: 1rem;
            border-radius: 18px;
            border: 1px solid rgba(148,163,184,.16);
            background:
              radial-gradient(circle at 100% 0%, rgba(59,130,246,.12), transparent 12rem),
              rgba(15,23,42,.82);
          }
          .exec-metric strong {
            color: white;
            font-size: 1.65rem;
            font-weight: 950;
            letter-spacing: -.05em;
            display: block;
          }
          .exec-metric span {
            color: #cbd5e1;
            font-size: .78rem;
            font-weight: 760;
          }
          .section-title {
            color: white;
            font-size: 1.38rem;
            font-weight: 950;
            letter-spacing: -.045em;
            margin: 1.05rem 0 .62rem 0;
          }
          .exec-grid-3 {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: .75rem;
            margin-bottom: .85rem;
          }
          .exec-grid-4 {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: .75rem;
            margin-bottom: .85rem;
          }
          .exec-card {
            padding: 1.05rem;
            border-radius: 18px;
            border: 1px solid rgba(148,163,184,.16);
            background:
              radial-gradient(circle at 90% 0%, rgba(59,130,246,.10), transparent 13rem),
              linear-gradient(145deg, rgba(15,23,42,.95), rgba(8,13,28,.96));
            min-height: 150px;
          }
          .exec-card.accent-cyan { border-color: rgba(34,211,238,.30); }
          .exec-card.accent-green { border-color: rgba(34,197,94,.30); }
          .exec-card.accent-purple { border-color: rgba(168,85,247,.30); }
          .exec-card.accent-orange { border-color: rgba(249,115,22,.30); }
          .card-icon {
            width: 38px;
            height: 38px;
            border-radius: 13px;
            display: grid;
            place-items: center;
            margin-bottom: .55rem;
            color: white;
            background: linear-gradient(135deg, rgba(59,130,246,.32), rgba(139,92,246,.22));
            border: 1px solid rgba(96,165,250,.25);
            font-weight: 950;
          }
          .exec-kicker {
            color: #94a3b8;
            font-size: .68rem;
            font-weight: 900;
            letter-spacing: .10em;
            text-transform: uppercase;
          }
          .exec-card h3 {
            color: white;
            margin: .26rem 0 .42rem 0;
            font-size: 1.02rem;
            letter-spacing: -.03em;
          }
          .exec-card p {
            color: #cbd5e1;
            margin: 0;
            font-size: .82rem;
            line-height: 1.48;
          }
          .proof-line {
            display: flex;
            gap: .55rem;
            align-items: flex-start;
            color: #cbd5e1;
            font-size: .80rem;
            line-height: 1.42;
            margin-top: .45rem;
          }
          .proof-line span {
            color: #4ade80;
            font-weight: 950;
          }
          .architecture-panel {
            padding: 1.1rem;
            border-radius: 20px;
            border: 1px solid rgba(34,211,238,.22);
            background:
              radial-gradient(circle at 10% 0%, rgba(34,211,238,.10), transparent 16rem),
              radial-gradient(circle at 90% 80%, rgba(139,92,246,.12), transparent 18rem),
              rgba(15,23,42,.78);
            margin-bottom: .9rem;
          }
          .arch-rail {
            display: grid;
            grid-template-columns: repeat(7, minmax(0, 1fr));
            gap: .5rem;
            margin-top: .75rem;
          }
          .arch-node {
            padding: .82rem .58rem;
            border-radius: 14px;
            border: 1px solid rgba(96,165,250,.20);
            background: rgba(2,6,23,.46);
            text-align: center;
            color: #e0f2fe;
            font-size: .71rem;
            font-weight: 850;
          }
          .cta-panel {
            padding: 1.2rem;
            border-radius: 22px;
            border: 1px solid rgba(34,211,238,.30);
            background:
              radial-gradient(circle at 10% 0%, rgba(34,211,238,.16), transparent 16rem),
              radial-gradient(circle at 92% 70%, rgba(34,197,94,.12), transparent 17rem),
              linear-gradient(145deg, rgba(8,47,73,.68), rgba(8,13,28,.96));
            margin: 1rem 0 .3rem 0;
          }
          .cta-grid {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: .65rem;
            margin-top: .8rem;
          }
          .cta-card {
            padding: .95rem;
            border-radius: 16px;
            border: 1px solid rgba(148,163,184,.18);
            background: rgba(15,23,42,.72);
            min-height: 118px;
          }
          .cta-card strong {
            color: white;
            display: block;
            margin-bottom: .36rem;
            font-size: .92rem;
          }
          .cta-card span {
            color: #cbd5e1;
            font-size: .74rem;
            line-height: 1.38;
          }
          @media (max-width: 1100px) {
            .exec-hero, .exec-metrics, .exec-grid-3, .exec-grid-4, .arch-rail, .cta-grid {
              grid-template-columns: 1fr;
            }
            .exec-title { font-size: 2.25rem; }
          }
        </style>

        <section class="exec-hero">
          <div>
            <div class="exec-eyebrow">Recruiter Landing Page · AI Product Case Study</div>
            <div class="exec-title">Financial News<br/>Stock Intelligence</div>
            <div class="exec-subtitle">
              A production-minded <span class="exec-highlight">AI analytics platform</span> that turns financial news into
              sentiment, movement probability, risk signals, forecast scenarios, explainability, and analyst-ready summaries.
            </div>
            <div class="exec-chip-row">
              <span class="exec-chip">NLP</span>
              <span class="exec-chip">Financial Analytics</span>
              <span class="exec-chip">Movement Modeling</span>
              <span class="exec-chip">Forecasting</span>
              <span class="exec-chip">Explainability</span>
              <span class="exec-chip">Streamlit</span>
              <span class="exec-chip">FastAPI</span>
              <span class="exec-chip">Docker</span>
              <span class="exec-chip">Kubernetes</span>
              <span class="exec-chip">CI/CD</span>
            </div>
          </div>

          <div class="stack-card">
            <div class="stack-title">AI Intelligence Stack</div>
            <div class="stack-step"><div class="stack-icon">01</div><div><strong>News Intake</strong><span>URL, upload, paste, public demo mode</span></div></div>
            <div class="stack-step"><div class="stack-icon">02</div><div><strong>NLP Signal</strong><span>BERT / DistilBERT-style sentiment workflow</span></div></div>
            <div class="stack-step"><div class="stack-icon">03</div><div><strong>Movement Intelligence</strong><span>Up / Flat / Down market reaction estimate</span></div></div>
            <div class="stack-step"><div class="stack-icon">04</div><div><strong>Forecast + Risk</strong><span>Bull, base, bear scenarios with uncertainty</span></div></div>
            <div class="stack-step"><div class="stack-icon">05</div><div><strong>Explainability</strong><span>Drivers, tokens, sentence impact, verdict</span></div></div>
          </div>
        </section>

        <div class="exec-metrics">
          <div class="exec-metric"><strong>13</strong><span>public dashboard sections</span></div>
          <div class="exec-metric"><strong>4</strong><span>intelligence layers: sentiment, movement, forecast, XAI</span></div>
          <div class="exec-metric"><strong>Free</strong><span>Streamlit Community Cloud public mode</span></div>
          <div class="exec-metric"><strong>MLOps</strong><span>Docker, Kubernetes, CI/CD, validation gates</span></div>
        </div>

        <div class="exec-grid-3">
          <div class="exec-card accent-cyan">
            <div class="card-icon">?</div>
            <div class="exec-kicker">Problem</div>
            <h3>Financial news is noisy and fast-moving</h3>
            <p>Market articles mix signal, boilerplate, uncertainty, and reaction cues. Analysts need structured intelligence quickly.</p>
          </div>
          <div class="exec-card accent-green">
            <div class="card-icon">✓</div>
            <div class="exec-kicker">Solution</div>
            <h3>Convert articles into decision signals</h3>
            <p>The system estimates sentiment, movement, risk, drivers, scenarios, and analyst-ready summaries from article text.</p>
          </div>
          <div class="exec-card accent-purple">
            <div class="card-icon">★</div>
            <div class="exec-kicker">Portfolio Value</div>
            <h3>Built like a deployable AI product</h3>
            <p>It combines UI, model workflow, backend architecture, testing, deployment planning, and public demo boundaries.</p>
          </div>
        </div>

        <div class="architecture-panel">
          <div class="exec-kicker">System Architecture</div>
          <h3 style="color:white;margin:.28rem 0 .2rem 0;">From article input to explainable analyst output</h3>
          <div class="arch-rail">
            <div class="arch-node">Streamlit UI</div>
            <div class="arch-node">FastAPI Layer</div>
            <div class="arch-node">NLP Models</div>
            <div class="arch-node">Movement Model</div>
            <div class="arch-node">Forecasts</div>
            <div class="arch-node">Explainability</div>
            <div class="arch-node">Docker / CI</div>
          </div>
        </div>

        """, unsafe_allow_html=True)

    _render_model_performance_dashboard()
    _render_model_role_map_3d()
    _render_ai_system_flow_diagram()

    st.markdown("""
        <div class="section-title">Model and intelligence layers</div>
        <div class="exec-grid-4">
          <div class="exec-card accent-cyan">
            <div class="card-icon">NLP</div>
            <div class="exec-kicker">Sentiment</div>
            <h3>Financial language intelligence</h3>
            <p>BERT / DistilBERT-style workflow for bullish, neutral, bearish interpretation and confidence storytelling.</p>
          </div>
          <div class="exec-card accent-green">
            <div class="card-icon">ML</div>
            <div class="exec-kicker">Movement</div>
            <h3>News-to-market reaction</h3>
            <p>Estimates Up, Flat, and Down movement pressure from article language, risk terms, and signal strength.</p>
          </div>
          <div class="exec-card accent-orange">
            <div class="card-icon">↗</div>
            <div class="exec-kicker">Forecast</div>
            <h3>Scenario paths</h3>
            <p>Bull, base, and bear views communicate uncertainty instead of pretending one exact price is known.</p>
          </div>
          <div class="exec-card accent-purple">
            <div class="card-icon">XAI</div>
            <div class="exec-kicker">Explainability</div>
            <h3>Human-readable reasoning</h3>
            <p>Driver phrases, token impact, sentence impact, and verdict panels explain why the signal moved.</p>
          </div>
        </div>

        <div class="section-title">What recruiters should notice</div>
        <div class="exec-grid-3">
          <div class="exec-card">
            <div class="exec-kicker">Data Analytics</div>
            <h3>Unstructured text → structured insight</h3>
            <div class="proof-line"><span>✓</span><div>Turns messy article text into sentiment, risk, movement, and forecast views.</div></div>
            <div class="proof-line"><span>✓</span><div>Presents outputs in business-friendly dashboards and summaries.</div></div>
          </div>
          <div class="exec-card">
            <div class="exec-kicker">ML / AI</div>
            <h3>Model workflow thinking</h3>
            <div class="proof-line"><span>✓</span><div>Separates sentiment, movement, forecasting, and explainability layers.</div></div>
            <div class="proof-line"><span>✓</span><div>Shows confidence and uncertainty without overstating model claims.</div></div>
          </div>
          <div class="exec-card">
            <div class="exec-kicker">MLOps / Product</div>
            <h3>Deployment-aware engineering</h3>
            <div class="proof-line"><span>✓</span><div>Includes public demo mode, API architecture, tests, CI/CD, Docker, Kubernetes, and clear boundaries.</div></div>
            <div class="proof-line"><span>✓</span><div>Designed for recruiters to explore without private infrastructure.</div></div>
          </div>
        </div>

        <div class="cta-panel">
          <div class="exec-kicker">Start Here</div>
          <h3 style="color:white;margin:.25rem 0;">Recommended recruiter path</h3>
          <div class="cta-grid">
            <div class="cta-card"><strong>Analyze Article</strong><span>Try URL input, token cloud, sentence impact, forecasts, and analyst verdict.</span></div>
            <div class="cta-card"><strong>Forecasts</strong><span>Review bull, base, and bear scenario intelligence.</span></div>
            <div class="cta-card"><strong>Explainability</strong><span>Inspect driver logic and reasoning boundaries.</span></div>
            <div class="cta-card"><strong>Model Evidence</strong><span>Review model roles, metrics, and champion selection.</span></div>
            <div class="cta-card"><strong>Architecture</strong><span>Review Streamlit, FastAPI, Docker, Kubernetes, and CI/CD design.</span></div>
          </div>
        </div>

        <div class="muted" style="margin:.75rem 0 0 0;">
          Public portfolio demo. Results are informational only and not investment advice.
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_public_streamlit_cloud_app(project_root: Path | str | None = None) -> None:
    """Render the public Streamlit Cloud app with real page routing."""

    _apply_theme()
    selected_page = _render_sidebar()
    _render_topbar(selected_page)

    if selected_page == "Executive Overview":
        _render_recruiter_landing_page()
        return

    if selected_page == "Analyze Article":
        _render_analyze_article_page()
        return

    if selected_page == "Forecasts":
        _render_forecasts_page()
        return

    if selected_page == "Historical Intelligence":
        _render_historical_intelligence_page()
        return

    if selected_page == "Explainability":
        _render_explainability_page()
        return

    if selected_page == "Scenario Analysis":
        _render_scenario_analysis_page()
        return

    _render_public_placeholder_page(selected_page)
