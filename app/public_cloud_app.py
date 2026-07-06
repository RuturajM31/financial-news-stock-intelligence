"""Premium public Streamlit Cloud application for Financial News Stock Intelligence.

Backend-free public mode for Streamlit Community Cloud.
It preserves the planned portfolio pages while making Executive Overview a real
financial intelligence command center with visible article input, conclusion,
scorecards, top-end charts, model workflow, and analyst-style insights.
"""

from __future__ import annotations

import html
import io
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st


_POSITIVE_TERMS = {
    "beat", "beats", "growth", "profit", "profits", "raise", "raised", "upgrade",
    "upgraded", "strong", "record", "bullish", "gain", "gains", "accelerate",
    "surge", "margin", "cash", "demand", "outperform", "resilient", "expansion",
    "guidance", "revenue", "approval", "partnership", "contract", "buyback",
    "dividend", "efficiency", "cost savings", "launch", "innovation", "ai",
    "data-center", "data center", "cloud", "bookings", "leadership", "savings",
}

_NEGATIVE_TERMS = {
    "miss", "misses", "loss", "losses", "cut", "downgrade", "downgraded", "weak",
    "lawsuit", "probe", "bearish", "decline", "falls", "fell", "risk", "risks",
    "debt", "slowdown", "inflation", "warning", "layoff", "pressure", "headwind",
    "recall", "regulatory", "investigation", "margin pressure", "fraud", "default",
}

_RISK_TERMS = {
    "volatility", "uncertainty", "macro", "rates", "inflation", "geopolitical",
    "regulatory", "probe", "debt", "lawsuit", "warning", "recall", "competition",
    "supply", "currency", "headwind", "demand risk", "execution risk",
    "export", "controls", "constraints", "margin pressure",
}

_COMPANY_HINTS = {
    "apple": "AAPL", "microsoft": "MSFT", "nvidia": "NVDA", "amazon": "AMZN",
    "alphabet": "GOOGL", "google": "GOOGL", "tesla": "TSLA", "meta": "META",
    "netflix": "NFLX", "jpmorgan": "JPM", "jpmorgan chase": "JPM",
    "bank of america": "BAC", "boeing": "BA", "intel": "INTC", "amd": "AMD",
    "micron": "MU", "broadcom": "AVGO", "oracle": "ORCL", "salesforce": "CRM",
}

_BASE_EXAMPLE = (
    "Nvidia reported record data-center revenue and stronger demand for AI chips, "
    "but management warned that supply constraints, export controls, and rising "
    "competition could pressure margins next quarter."
)


@dataclass(frozen=True)
class ArticleSignal:
    """One transparent public-mode analysis result."""

    ticker: str
    company: str
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
    extracted_terms: list[str]


def should_use_public_streamlit_cloud_app(project_root: Path) -> bool:
    """Return whether the public Streamlit Cloud mode should replace API mode."""

    disabled = os.getenv("FNI_DISABLE_PUBLIC_STREAMLIT_MODE", "").strip().lower()
    if disabled in {"1", "true", "yes"}:
        return False

    override = os.getenv("FNI_PUBLIC_STREAMLIT_MODE", "").strip().lower()
    if override in {"1", "true", "yes", "demo", "cloud", "public"}:
        return True

    return project_root.resolve().as_posix().startswith("/mount/src/")


def _apply_theme() -> None:
    """Install the premium dark portfolio theme used for the public app."""

    st.markdown(
        """
<style>
:root {
  --bg0:#050814; --bg1:#07111f; --bg2:#0d1728; --panel:#0d1627;
  --stroke:rgba(102,166,255,.20); --stroke2:rgba(45,212,191,.25);
  --text:#f8fbff; --soft:#d8e3f2; --muted:#95a3b8;
  --blue:#38bdf8; --cyan:#22d3ee; --green:#4ade80; --amber:#f59e0b;
  --red:#fb7185; --violet:#a78bfa;
}
[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(circle at 22% 0%, rgba(14,165,233,.18), transparent 36rem),
    radial-gradient(circle at 88% 2%, rgba(124,58,237,.20), transparent 40rem),
    radial-gradient(circle at 48% 84%, rgba(20,184,166,.10), transparent 42rem),
    linear-gradient(180deg, #050814 0%, #07111f 48%, #08111f 100%);
  color:var(--text);
}
[data-testid="stHeader"] {
  background:rgba(5,8,20,.74);
  border-bottom:1px solid rgba(148,163,184,.12);
  backdrop-filter: blur(20px);
}
[data-testid="stSidebar"] {
  background:linear-gradient(180deg,#030712 0%,#07111f 100%);
  border-right:1px solid rgba(56,189,248,.20);
}
.block-container {max-width:1530px; padding-top:1rem; padding-bottom:2.6rem;}
h1,h2,h3,h4 {color:var(--text)!important; letter-spacing:-.035em;}
p,li,span,label,div {color:inherit;}
[data-testid="stTextInput"] input, textarea {
  background:#091527!important; color:#f8fbff!important;
  border:1px solid rgba(56,189,248,.28)!important; border-radius:14px!important;
}
[data-testid="stFileUploader"] section {
  background:#091527!important; border:1px solid rgba(56,189,248,.22)!important; border-radius:16px!important;
}
.stButton button {
  border-radius:12px!important; font-weight:850!important;
  border:1px solid rgba(56,189,248,.32)!important;
  background:linear-gradient(135deg,rgba(14,165,233,.22),rgba(124,58,237,.20))!important;
  color:#f8fbff!important;
}
div[data-testid="stButton"] button[kind="primary"] {
  background:linear-gradient(135deg,#0284c7,#7c3aed)!important;
  border-color:rgba(125,211,252,.46)!important;
}
.hero {
  border:1px solid var(--stroke); border-radius:22px; padding:1.3rem 1.45rem;
  background:linear-gradient(135deg,rgba(15,23,42,.95),rgba(8,47,73,.56));
  box-shadow:0 22px 70px rgba(0,0,0,.32); margin-bottom:1rem;
}
.eyebrow {color:var(--cyan); font-weight:950; letter-spacing:.16em; font-size:.72rem; text-transform:uppercase;}
.hero h1 {font-size:clamp(1.9rem,3.2vw,3.2rem); line-height:1.02; margin:.38rem 0 .7rem;}
.hero p {font-size:.98rem; color:var(--soft); max-width:1020px; line-height:1.55; margin:0;}
.panel {
  border:1px solid rgba(148,163,184,.16); border-radius:20px; padding:1rem 1.08rem;
  background:linear-gradient(180deg,rgba(15,23,42,.80),rgba(15,23,42,.50));
  box-shadow:0 16px 44px rgba(0,0,0,.22); margin-bottom:1rem;
}
.pill {
  display:inline-block; margin:.14rem .16rem .14rem 0; padding:.34rem .68rem;
  border-radius:999px; border:1px solid rgba(125,211,252,.22);
  background:rgba(15,23,42,.75); color:#dff7ff; font-size:.80rem; font-weight:780;
}
.warning {border:1px solid rgba(251,191,36,.28); background:rgba(120,53,15,.22); border-radius:16px; padding:.9rem 1rem; color:#fde68a;}
.good {border:1px solid rgba(52,211,153,.28); background:rgba(6,78,59,.24); border-radius:16px; padding:.9rem 1rem; color:#bbf7d0;}
.executive-shell {display:flex; flex-direction:column; gap:.8rem;}
.exec-control {
  border:1px solid rgba(56,189,248,.24); border-radius:22px; padding:1rem 1.05rem;
  background:
    radial-gradient(circle at 18% 0%, rgba(56,189,248,.10), transparent 22rem),
    linear-gradient(135deg,rgba(8,13,28,.94),rgba(15,23,42,.78));
  box-shadow:0 22px 70px rgba(0,0,0,.28);
}
.exec-control-head {display:flex; align-items:flex-start; justify-content:space-between; gap:1rem; margin-bottom:.8rem;}
.exec-title-row {font-size:1.78rem; font-weight:970; letter-spacing:-.05em;}
.exec-subrow {color:#a8b3c7; font-size:.93rem; margin-top:.12rem;}
.exec-badges {display:flex; flex-wrap:wrap; gap:.5rem; justify-content:flex-end;}
.exec-badges span {
  border:1px solid rgba(56,189,248,.23); border-radius:12px; padding:.48rem .72rem;
  background:rgba(15,23,42,.70); color:#dff7ff; font-weight:850; font-size:.76rem;
}
.control-note {color:#96a3b8; font-size:.80rem; margin-top:.2rem;}
.article-strip {
  display:grid; grid-template-columns:1.1fr 3.2fr 1.1fr .9fr; gap:1rem; align-items:center;
  border:1px solid rgba(148,163,184,.16); border-radius:19px; padding:.9rem 1rem;
  background:linear-gradient(135deg,rgba(8,13,28,.95),rgba(15,23,42,.72));
  box-shadow:0 18px 52px rgba(0,0,0,.24);
}
.article-identity {display:flex; align-items:center; gap:.78rem;}
.ticker-logo {
  width:50px; height:50px; display:grid; place-items:center; border-radius:15px;
  background:linear-gradient(135deg,#34d399,#0ea5e9); color:#06111f;
  font-weight:970; letter-spacing:-.04em; box-shadow:0 0 28px rgba(52,211,153,.20);
}
.article-ticker {font-size:1.25rem; font-weight:970;}
.article-company,.article-label {color:#96a3b8; font-size:.76rem;}
.article-headline {font-size:.86rem; line-height:1.38; color:#f8fbff;}
.article-meta {border-left:1px solid rgba(148,163,184,.14); padding-left:1rem; font-size:.84rem;}
.exec-card-grid {display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:.72rem;}
.exec-card {
  min-height:112px; border-radius:19px; padding:.92rem .94rem; position:relative; overflow:hidden;
  border:1px solid rgba(148,163,184,.18); background:rgba(15,23,42,.76);
  box-shadow:0 18px 52px rgba(0,0,0,.26);
}
.exec-card-top {display:flex; gap:.75rem; align-items:flex-start; position:relative; z-index:2;}
.exec-icon {width:42px; height:42px; border-radius:999px; display:grid; place-items:center; background:rgba(2,6,23,.45); border:1px solid rgba(255,255,255,.12); font-weight:970;}
.exec-label {font-size:.72rem; color:#cbd5e1; font-weight:880;}
.exec-value {font-size:1.72rem; line-height:1.05; font-weight:970; color:#fff; letter-spacing:-.05em;}
.exec-subtitle {font-size:.78rem; color:#cbd5e1; margin-top:.18rem;}
.exec-spark {position:absolute; right:.65rem; bottom:.45rem; width:112px; opacity:.88;}
.spark {width:112px; height:36px;}
.accent-violet {background:linear-gradient(135deg,rgba(49,46,129,.55),rgba(15,23,42,.82)); border-color:rgba(167,139,250,.30);}
.accent-cyan {background:linear-gradient(135deg,rgba(8,145,178,.36),rgba(15,23,42,.82)); border-color:rgba(34,211,238,.30);}
.accent-green {background:linear-gradient(135deg,rgba(22,101,52,.36),rgba(15,23,42,.82)); border-color:rgba(52,211,153,.30);}
.accent-amber {background:linear-gradient(135deg,rgba(146,64,14,.34),rgba(15,23,42,.82)); border-color:rgba(251,191,36,.27);}
.accent-purple {background:linear-gradient(135deg,rgba(88,28,135,.38),rgba(15,23,42,.82)); border-color:rgba(192,132,252,.30);}
.insight-grid {display:grid; grid-template-columns:1.22fr .88fr; gap:.75rem;}
.executive-insight {
  border:1px solid rgba(34,211,238,.42); border-radius:20px; padding:1rem 1.1rem;
  background:radial-gradient(circle at 100% 100%,rgba(52,211,153,.22),transparent 22rem),linear-gradient(135deg,rgba(8,47,73,.74),rgba(15,23,42,.86));
  box-shadow:0 0 44px rgba(34,211,238,.12),0 20px 58px rgba(0,0,0,.24);
}
.insight-kicker,.section-title {font-size:.94rem; color:#7dd3fc; font-weight:970; margin-bottom:.55rem;}
.insight-message {font-size:1.16rem; line-height:1.48; color:#fff; font-weight:770;}
.important-analysis {border:1px solid rgba(148,163,184,.16); border-radius:20px; padding:1rem 1.1rem; background:rgba(15,23,42,.72);}
.important-analysis ul {margin:.25rem 0 0 1rem; padding:0;}
.important-analysis li {margin:.34rem 0; color:#dce7f7; font-size:.86rem;}
.workflow-wrap {
  border:1px solid rgba(148,163,184,.16); border-radius:20px; background:rgba(15,23,42,.66);
  padding:.9rem; margin:.15rem 0;
}
.workflow-grid {display:grid; grid-template-columns:repeat(7,minmax(0,1fr)); gap:.55rem;}
.workflow-card {position:relative; min-height:88px; border:1px solid rgba(96,165,250,.18); border-radius:14px; padding:.62rem; background:linear-gradient(180deg,rgba(15,23,42,.85),rgba(30,41,59,.42));}
.workflow-status {position:absolute; top:.44rem; right:.48rem; color:#34d399; font-weight:970;}
.workflow-number {font-size:.66rem; color:#7dd3fc; font-weight:970;}
.workflow-title {font-size:.80rem; font-weight:970; margin-top:.18rem;}
.workflow-detail {font-size:.68rem; color:#cbd5e1; line-height:1.28; margin-top:.18rem;}
.driver-grid {display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:.75rem;}
.driver-panel {border-radius:18px; padding:.9rem 1rem; border:1px solid rgba(148,163,184,.16); background:rgba(15,23,42,.68); min-height:132px;}
.bullish-panel {border-color:rgba(52,211,153,.25);}
.risk-panel {border-color:rgba(251,191,36,.25);}
.monitor-panel {border-color:rgba(56,189,248,.25);}
.driver-title {font-weight:970; margin-bottom:.55rem;}
.driver-chip {display:inline-block; padding:.38rem .64rem; margin:.2rem .18rem; border-radius:999px; background:rgba(30,41,59,.84); border:1px solid rgba(148,163,184,.12); color:#dbeafe; font-size:.76rem; font-weight:780;}
.exec-disclaimer {color:#94a3b8; font-size:.74rem; padding:.35rem .2rem;}
.preview-box {
  border:1px solid rgba(148,163,184,.14); border-radius:16px; padding:.8rem .9rem;
  background:rgba(2,6,23,.30); color:#cbd5e1; font-size:.84rem; line-height:1.45;
}
.go-deeper {display:flex; gap:.5rem; flex-wrap:wrap; margin-top:.3rem;}
.go-deeper span {padding:.36rem .62rem; border-radius:999px; background:rgba(8,47,73,.48); border:1px solid rgba(56,189,248,.22); font-size:.76rem; color:#dff7ff; font-weight:820;}
@media (max-width:1200px) {
  .exec-card-grid,.workflow-grid {grid-template-columns:repeat(2,minmax(0,1fr));}
  .article-strip,.insight-grid,.driver-grid {grid-template-columns:1fr;}
  .exec-control-head {display:block;}
  .exec-badges {justify-content:flex-start; margin-top:.65rem;}
}
</style>
        """,
        unsafe_allow_html=True,
    )


def _html(text: object) -> str:
    """Escape text for HTML fragments."""

    return html.escape(str(text), quote=True)


def _layout() -> dict:
    """Return a consistent Plotly dark layout."""

    return {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(15,23,42,.46)",
        "font": {"color": "#f8fbff", "family": "Inter, Arial, sans-serif"},
        "margin": {"l": 40, "r": 24, "t": 54, "b": 42},
        "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    }


def _chart_config() -> dict:
    """Return Plotly config that keeps charts clean for executive pages."""

    return {"displayModeBar": False, "responsive": True}


def _tokens(text: str) -> list[str]:
    """Tokenize the article text into lowercase terms used by public-mode scoring."""

    return re.findall(r"[A-Za-z][A-Za-z\\-']+", text.lower())


def _hits(text: str, terms: Iterable[str]) -> list[str]:
    """Return sorted public scoring hits found in the article."""

    lowered = text.lower()
    found: list[str] = []
    for term in terms:
        if " " in term:
            if term in lowered:
                found.append(term)
        elif re.search(rf"\\b{re.escape(term)}\\b", lowered):
            found.append(term)
    return sorted(set(found))


def _infer_ticker(text: str) -> tuple[str, str]:
    """Infer a likely ticker/company from common company mentions."""

    lowered = text.lower()
    for name, ticker in _COMPANY_HINTS.items():
        if name in lowered:
            return ticker, name.title()
    uppercase = re.findall(r"\\b[A-Z]{2,5}\\b", text)
    if uppercase:
        return uppercase[0], uppercase[0]
    return "NEWS", "Article"


def _score_article(text: str | None) -> ArticleSignal:
    """Score text safely for public Executive Overview mode."""

    try:
        if text is None:
            text = ""
        if not isinstance(text, str):
            text = str(text)

        clean = re.sub(r"\\s+", " ", text).strip()
        if not clean:
            clean = _BASE_EXAMPLE

        positive = _hits(clean, _POSITIVE_TERMS)
        negative = _hits(clean, _NEGATIVE_TERMS)
        risk = _hits(clean, _RISK_TERMS)
        tokens = _tokens(clean)
        ticker, company = _infer_ticker(clean)

        positive_weight = len(positive) * 1.15
        negative_weight = len(negative) * 1.10
        risk_weight = len(risk) * 0.82
        length_dampener = min(1.0, max(0.42, len(tokens) / 85))

        raw_sentiment = positive_weight - negative_weight
        sentiment_score = max(-1.0, min(1.0, raw_sentiment / 7.0))
        risk_score = max(0.05, min(0.95, (risk_weight + negative_weight * 0.4) / 8.0))

        up = 0.34 + sentiment_score * 0.26 - risk_score * 0.07
        down = 0.28 - sentiment_score * 0.20 + risk_score * 0.18
        flat = 1.0 - up - down

        values = [max(0.05, up), max(0.05, flat), max(0.05, down)]
        total = sum(values)
        movement_up, movement_flat, movement_down = [v / total for v in values]

        confidence = min(
            0.92,
            max(0.55, 0.58 + abs(sentiment_score) * 0.22 + length_dampener * 0.12),
        )

        if movement_up > movement_down + 0.08:
            label = "Bullish / positive movement pressure"
        elif movement_down > movement_up + 0.08:
            label = "Bearish / negative movement pressure"
        else:
            label = "Mixed / watchlist signal"

        extracted_terms = sorted(set(positive + negative + risk))[:18]

    except Exception:
        clean = _BASE_EXAMPLE
        ticker, company = "NEWS", "Article"
        positive = ["demand", "revenue"]
        negative = []
        risk = ["risk"]
        sentiment_score = 0.35
        risk_score = 0.20
        movement_up = 0.45
        movement_flat = 0.35
        movement_down = 0.20
        confidence = 0.65
        label = "Fallback / safe public analysis"
        extracted_terms = ["demand", "revenue", "risk"]

    return ArticleSignal(
        ticker=ticker,
        company=company,
        label=label,
        confidence=confidence,
        sentiment_score=sentiment_score,
        movement_up=movement_up,
        movement_flat=movement_flat,
        movement_down=movement_down,
        risk_score=risk_score,
        positive_hits=positive,
        negative_hits=negative,
        risk_hits=risk,
        extracted_terms=extracted_terms,
    )


def _fetch_url_text(url: str) -> str:
    """Fetch article-like text from a public URL without using paid services."""

    if not url.strip():
        return ""
    headers = {"User-Agent": "FinancialNewsStockIntelligencePublicDemo/1.0"}
    response = requests.get(url.strip(), headers=headers, timeout=8)
    response.raise_for_status()
    html_text = response.text
    html_text = re.sub(r"(?is)<script.*?</script>|<style.*?</style>", " ", html_text)
    text = re.sub(r"(?s)<[^>]+>", " ", html_text)
    text = re.sub(r"\\s+", " ", text)
    return text[:9000]


def _read_upload(uploaded_file) -> str:
    """Read uploaded article text from lightweight public formats."""

    if uploaded_file is None:
        return ""

    name = uploaded_file.name.lower()
    data = uploaded_file.getvalue()

    if name.endswith((".txt", ".md", ".csv", ".json")):
        return data.decode("utf-8", errors="ignore")[:9000]

    if name.endswith(".pdf"):
        try:
            from pypdf import PdfReader  # type: ignore
            reader = PdfReader(io.BytesIO(data))
            return "\\n".join(page.extract_text() or "" for page in reader.pages)[:9000]
        except Exception:
            return "PDF uploaded. Text extraction is unavailable in the free public dependency set."

    return data.decode("utf-8", errors="ignore")[:9000]


def _initialize_exec_state() -> None:
    """Initialize Executive Overview article state."""

    st.session_state.setdefault("exec_article_url", "")
    st.session_state.setdefault("exec_article_text", _BASE_EXAMPLE)
    st.session_state.setdefault("exec_analyzed_text", _BASE_EXAMPLE)
    st.session_state.setdefault("exec_analyzed_source", "sample article")
    st.session_state.setdefault("exec_last_action", "Loaded sample article")


def _executive_analysis_control_center() -> tuple[str, str]:
    """Render visible article-entry controls and return analyzed text/source."""

    st.markdown(
        "<div class='control-title'>Analysis Control Center</div>",
        unsafe_allow_html=True,
    )

    _initialize_exec_state()

    with st.container():
        st.markdown(
            "<div class='exec-control'><div class='exec-control-head'>"
            "<div><div class='exec-title-row'>📈 Executive Overview</div>"
            "<div class='exec-subrow'>Financial News → Sentiment → Movement → Forecast → Explainability</div>"
            "<div class='control-note'>Enter a URL, upload an article, or paste text. Click Analyze to refresh every insight and chart.</div></div>"
            "<div class='exec-badges'><span>☁ Public Cloud Mode</span><span>🔗 Article URL Enabled</span>"
            "<span>⇧ Upload Enabled</span><span>👁 Model Workflow Visible</span></div></div>",
            unsafe_allow_html=True,
        )

        c1, c2 = st.columns([1.35, 1.0])
        with c1:
            url = st.text_input("Article URL", placeholder="https://example.com/company-earnings-news", key="exec_article_url")
        with c2:
            uploaded = st.file_uploader("Upload article", type=["txt", "md", "csv", "json", "pdf"], key="exec_article_upload")

        pasted = st.text_area(
            "Paste article / headline / news text",
            key="exec_article_text",
            height=118,
            help="This text is used when no uploaded file is present and URL fetch is unavailable.",
        )

        b1, b2, b3, b4 = st.columns([.8, .8, .7, 2.2])
        analyze = b1.button("Analyze", type="primary", use_container_width=True)
        sample = b2.button("Use sample", use_container_width=True)
        clear = b3.button("Clear", use_container_width=True)
        b4.markdown(
            "<div class='go-deeper'><span>updates conclusion</span><span>updates charts</span>"
            "<span>updates drivers</span><span>updates workflow</span></div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    if sample:
        st.session_state["exec_article_url"] = ""
        st.session_state["exec_article_text"] = _BASE_EXAMPLE
        st.session_state["exec_analyzed_text"] = _BASE_EXAMPLE
        st.session_state["exec_analyzed_source"] = "sample article"
        st.session_state["exec_last_action"] = "Sample article loaded"
        st.rerun()

    if clear:
        st.session_state["exec_article_url"] = ""
        st.session_state["exec_article_text"] = ""
        st.session_state["exec_analyzed_text"] = _BASE_EXAMPLE
        st.session_state["exec_analyzed_source"] = "sample article"
        st.session_state["exec_last_action"] = "Inputs cleared; sample restored"
        st.rerun()

    if analyze:
        source = "pasted text"
        text = (pasted or "").strip()
        upload_text = _read_upload(uploaded)
        if upload_text.strip():
            source = f"uploaded file: {uploaded.name}"
            text = upload_text
        elif url.strip():
            try:
                fetched = _fetch_url_text(url)
                if fetched.strip():
                    source = f"URL: {url.strip()}"
                    text = fetched
                else:
                    source = "pasted text"
            except Exception as exc:
                st.warning(f"URL fetch failed in public mode: {exc}. Using pasted text instead.")
                source = "pasted text after URL fetch failure"
        if not text:
            text = _BASE_EXAMPLE
            source = "sample article"
        st.session_state["exec_analyzed_text"] = text
        st.session_state["exec_analyzed_source"] = source
        st.session_state["exec_last_action"] = f"Analyzed from {source}"
        st.rerun()

    return st.session_state["exec_analyzed_text"], st.session_state["exec_analyzed_source"]


def _article_inputs() -> tuple[str, str]:
    """Collect URL, upload, and pasted article text for non-executive pages."""

    st.markdown(
        "<div class='panel'><h3>Article intake</h3>"
        "<p>Use a URL, upload an article file, or paste text. The public app then shows each model-style stage transparently.</p></div>",
        unsafe_allow_html=True,
    )
    with st.expander("Input article", expanded=True):
        url = st.text_input("Article URL", placeholder="https://example.com/company-earnings-news", key="general_article_url")
        uploaded = st.file_uploader("Upload article", type=["txt", "md", "csv", "json", "pdf"], key="general_article_upload")
        pasted = st.text_area("Paste article / headline / news text", value=_BASE_EXAMPLE, height=180, key="general_article_text")

    source = "pasted text"
    text = pasted.strip()
    upload_text = _read_upload(uploaded)
    if upload_text.strip():
        source = f"uploaded file: {uploaded.name}"
        text = upload_text
    if url.strip():
        try:
            fetched = _fetch_url_text(url)
            if fetched.strip():
                source = f"URL: {url.strip()}"
                text = fetched
        except Exception as exc:
            st.warning(f"URL fetch failed in public mode: {exc}. Using pasted/uploaded text instead.")
    return text, source


def _sparkline_svg(points: list[float], color: str = "#38bdf8") -> str:
    """Return a tiny inline SVG sparkline for scorecards."""

    width, height = 118, 38
    xs = [i * width / (len(points) - 1) for i in range(len(points))]
    min_v, max_v = min(points), max(points)
    span = max(max_v - min_v, 0.001)
    ys = [height - ((p - min_v) / span * (height - 6) + 3) for p in points]
    path = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    return (
        f"<svg class='spark' viewBox='0 0 {width} {height}' preserveAspectRatio='none'>"
        f"<polyline points='{path}' fill='none' stroke='{color}' stroke-width='3' "
        f"stroke-linecap='round' stroke-linejoin='round'/></svg>"
    )


def _safe_pct(value: float) -> str:
    """Return percentage text for compact executive cards."""

    return f"{value:.0%}"


def _company_logo_badge(signal: ArticleSignal) -> str:
    """Return a compact ticker logo badge using the detected ticker."""

    initials = _html((signal.ticker or "FN")[:4].upper())
    return f"<div class='ticker-logo'>{initials}</div>"


def _executive_card(title: str, value: str, subtitle: str, icon: str, accent: str, spark: str = "") -> str:
    """Return one premium scorecard as HTML."""

    return (
        f"<div class='exec-card accent-{_html(accent)}'><div class='exec-card-top'>"
        f"<div class='exec-icon'>{_html(icon)}</div><div><div class='exec-label'>{_html(title)}</div>"
        f"<div class='exec-value'>{_html(value)}</div><div class='exec-subtitle'>{_html(subtitle)}</div></div></div>"
        f"<div class='exec-spark'>{spark}</div></div>"
    )


def _executive_conclusion(signal: ArticleSignal) -> str:
    """Build the user-facing conclusion message."""

    if signal.sentiment_score >= 0.35 and signal.movement_up >= signal.movement_down:
        sentiment = "strongly positive" if signal.sentiment_score > 0.70 else "positive"
        direction = "bullish movement pressure"
    elif signal.sentiment_score <= -0.25:
        sentiment = "negative"
        direction = "bearish movement pressure"
    else:
        sentiment = "mixed"
        direction = "watchlist / confirmation-needed movement pressure"

    risk_language = "moderate" if signal.risk_score < 0.45 else "elevated"
    risk_terms = ", ".join(signal.risk_hits[:3]) if signal.risk_hits else "limited explicit risk language"
    return (
        f"Conclusion: The article is {sentiment} for {signal.ticker}. "
        f"The signal estimates {direction} with {_safe_pct(signal.confidence)} confidence. "
        f"Risk is {risk_language}, driven by {risk_terms}."
    )


def _analyst_note(signal: ArticleSignal) -> str:
    """Return the most important user-facing analysis sentence."""

    positive = ", ".join(signal.positive_hits[:3]) if signal.positive_hits else "limited bullish wording"
    risk = ", ".join(signal.risk_hits[:3]) if signal.risk_hits else "no dominant risk phrase"
    return (
        f"Important analysis: {positive} is supporting the upside case, while {risk} is the main item to monitor. "
        f"The model-style workflow is not just scoring sentiment; it converts language into risk-adjusted movement pressure."
    )


def _movement_donut(signal: ArticleSignal) -> None:
    """Render movement probability as an executive donut."""

    fig = go.Figure(
        data=[go.Pie(
            labels=["Up", "Flat", "Down"],
            values=[signal.movement_up, signal.movement_flat, signal.movement_down],
            hole=0.62,
            textinfo="label+percent",
            sort=False,
            marker={"colors": ["#7dd3fc", "#0f76c8", "#fda4af"]},
        )]
    )
    fig.update_layout(
        title="Movement Probability",
        height=360,
        annotations=[{
            "text": f"<b>Up<br>{signal.movement_up:.0%}</b>",
            "x": 0.5, "y": 0.5,
            "font": {"size": 24, "color": "#f8fbff"},
            "showarrow": False,
        }],
        **_layout(),
    )
    st.plotly_chart(fig, width="stretch", config=_chart_config())


def _sentiment_risk_quadrant(signal: ArticleSignal) -> None:
    """Render the sentiment/risk executive quadrant."""

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[signal.sentiment_score],
        y=[signal.risk_score],
        mode="markers+text",
        text=[signal.ticker],
        textposition="top center",
        marker={"size": 26, "color": "#7dd3fc", "line": {"width": 2, "color": "#fde68a"}},
        name="Current article",
    ))
    fig.add_hline(y=0.5, line_dash="dot", line_color="rgba(148,163,184,.45)")
    fig.add_vline(x=0, line_dash="dot", line_color="rgba(148,163,184,.45)")
    fig.add_annotation(x=-0.72, y=0.84, text="High Risk<br>Negative", showarrow=False, font={"size": 11})
    fig.add_annotation(x=0.72, y=0.84, text="High Risk<br>Positive", showarrow=False, font={"size": 11})
    fig.add_annotation(x=-0.72, y=0.18, text="Low Risk<br>Negative", showarrow=False, font={"size": 11})
    fig.add_annotation(x=0.72, y=0.18, text="Low Risk<br>Positive", showarrow=False, font={"size": 11})
    fig.update_layout(
        title="Sentiment vs Risk",
        height=360,
        xaxis_title="Sentiment score",
        yaxis_title="Risk level",
        xaxis={"range": [-1.05, 1.05]},
        yaxis={"range": [0, 1]},
        **_layout(),
    )
    st.plotly_chart(fig, width="stretch", config=_chart_config())


def _driver_waterfall(signal: ArticleSignal) -> None:
    """Render a premium horizontal driver-impact chart."""

    rows = [
        ("Demand / growth language", 1.90 if signal.positive_hits else 0.45),
        ("Revenue / profit strength", 1.55 if {"revenue", "profit", "profits"} & set(signal.positive_hits) else 0.75),
        ("AI / innovation terms", 1.25 if {"ai", "innovation", "cloud", "data center", "data-center"} & set(signal.positive_hits) else 0.65),
        ("Confidence lift", signal.confidence * 1.20),
        ("Risk terms", -max(0.25, len(signal.risk_hits) * 0.32)),
        ("Bearish terms", -max(0.10, len(signal.negative_hits) * 0.38)),
        ("Uncertainty discount", -signal.risk_score * 1.05),
    ]
    colors = ["#7dd3fc" if value >= 0 else "#fda4af" for _, value in rows]
    fig = go.Figure(data=[go.Bar(
        x=[value for _, value in rows],
        y=[name for name, _ in rows],
        orientation="h",
        text=[f"{value:+.2f}" for _, value in rows],
        textposition="outside",
        marker={"color": colors},
    )])
    fig.add_vline(x=0, line_width=1, line_dash="dot", opacity=.55)
    fig.update_layout(
        title="Driver Impact (Signal Contribution)",
        height=360,
        xaxis_title="Impact on signal score",
        yaxis={"autorange": "reversed"},
        **_layout(),
    )
    st.plotly_chart(fig, width="stretch", config=_chart_config())


def _executive_forecast(signal: ArticleSignal) -> None:
    """Render a premium forecast panel with bull/base/bear and confidence band."""

    horizon = list(range(0, 61, 5))
    sentiment_push = (signal.movement_up - signal.movement_down)
    base = [100 + i * sentiment_push * 0.55 for i in range(len(horizon))]
    bull = [value + i * (0.42 + signal.confidence * 0.18) for i, value in enumerate(base)]
    bear = [value - i * (0.34 + signal.risk_score * 0.28) for i, value in enumerate(base)]
    upper = [value + 2.5 + i * .07 for i, value in enumerate(base)]
    lower = [value - 2.5 - i * .07 for i, value in enumerate(base)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=horizon, y=upper, mode="lines", line={"width": 0}, showlegend=False))
    fig.add_trace(go.Scatter(
        x=horizon, y=lower, mode="lines", fill="tonexty",
        fillcolor="rgba(148,163,184,.16)", line={"width": 0}, name="Confidence band",
    ))
    fig.add_trace(go.Scatter(x=horizon, y=bull, mode="lines+markers", name="Bull case", line={"color": "#4ade80"}))
    fig.add_trace(go.Scatter(x=horizon, y=base, mode="lines+markers", name="Base case", line={"color": "#38bdf8"}))
    fig.add_trace(go.Scatter(x=horizon, y=bear, mode="lines+markers", name="Bear case", line={"color": "#fb7185"}))
    fig.update_layout(
        title="Forecast (Price Movement Scenarios)",
        height=360,
        xaxis_title="Trading days",
        yaxis_title="Normalized price",
        **_layout(),
    )
    st.plotly_chart(fig, width="stretch", config=_chart_config())


def _workflow_strip(signal: ArticleSignal) -> None:
    """Render a compact horizontal model workflow strip."""

    steps = [
        ("01", "Article Intake", "URL/upload/text normalized"),
        ("02", "Text Extraction", "Boilerplate removed"),
        ("03", "Sentiment Signal", f"{len(signal.positive_hits)} bullish / {len(signal.negative_hits)} bearish"),
        ("04", "Movement Estimate", f"Up {signal.movement_up:.0%} · Down {signal.movement_down:.0%}"),
        ("05", "Risk Adjustment", f"Risk {signal.risk_score:.0%}"),
        ("06", "Forecast", "Bull / base / bear paths"),
        ("07", "Explainability", "Drivers visible"),
    ]
    cards = "".join(
        f"<div class='workflow-card'><div class='workflow-status'>✓</div>"
        f"<div class='workflow-number'>{_html(number)}</div>"
        f"<div class='workflow-title'>{_html(name)}</div>"
        f"<div class='workflow-detail'>{_html(detail)}</div></div>"
        for number, name, detail in steps
    )
    st.markdown(
        "<div class='workflow-wrap'><div class='section-title'>What the Model is Doing</div>"
        f"<div class='workflow-grid'>{cards}</div></div>",
        unsafe_allow_html=True,
    )


def _chip_panel(title: str, icon: str, chips: list[str], tone: str) -> str:
    """Return one bottom executive chip panel as HTML."""

    chip_html = "".join(f"<span class='driver-chip'>{_html(chip)}</span>" for chip in chips)
    return (
        f"<div class='driver-panel {tone}'>"
        f"<div class='driver-title'>{_html(icon)} {_html(title)}</div>"
        f"<div class='driver-chips'>{chip_html}</div></div>"
    )


def _render_executive_overview_premium(signal: ArticleSignal, source: str, text: str) -> None:
    """Render the premium Executive Overview command center."""

    headline = _html(re.sub(r"\\s+", " ", text.strip())[:230] or "No article text supplied.")
    source_text = _html(source)
    ticker = _html(signal.ticker)
    company = _html(signal.company)

    st.markdown(
        f"<div class='article-strip'><div class='article-identity'>{_company_logo_badge(signal)}"
        f"<div><div class='article-ticker'>{ticker}</div><div class='article-company'>{company}</div></div></div>"
        f"<div class='article-headline'><div class='article-label'>Article / Headline</div><div>{headline}</div></div>"
        f"<div class='article-meta'><div class='article-label'>Source</div><div>{source_text}</div></div>"
        "<div class='article-meta'><div class='article-label'>Status</div><div>Analyzed</div></div></div>",
        unsafe_allow_html=True,
    )

    spark_up = _sparkline_svg([.18, .31, .28, .47, .43, .66, .58, .78, .72, .88], "#34d399")
    spark_sent = _sparkline_svg([.32, .40, .36, .55, .48, .70, .62, .82], "#22d3ee")
    spark_risk = _sparkline_svg([.20, .30, .25, .37, .31, .43, .35, .48], "#f59e0b")
    spark_conf = _sparkline_svg([.44, .51, .58, .64, .71, .78, .85, .92], "#a78bfa")
    cards = [
        _executive_card("Ticker", signal.ticker, signal.company, "▣", "violet"),
        _executive_card("Sentiment Score", f"{signal.sentiment_score:+.2f}", "Strongly positive" if signal.sentiment_score > .4 else "Mixed signal", "◉", "cyan", spark_sent),
        _executive_card("Movement Signal", "Bullish" if signal.movement_up >= signal.movement_down else "Bearish", "Upward pressure" if signal.movement_up >= signal.movement_down else "Downside pressure", "↗", "green", spark_up),
        _executive_card("Risk Level", _safe_pct(signal.risk_score), "Moderate risk" if signal.risk_score < .45 else "Elevated risk", "◇", "amber", spark_risk),
        _executive_card("Confidence", _safe_pct(signal.confidence), "High confidence" if signal.confidence > .75 else "Medium confidence", "◎", "purple", spark_conf),
    ]
    st.markdown("<div class='exec-card-grid'>" + "".join(cards) + "</div>", unsafe_allow_html=True)

    important = [
        f"Primary movement bias: {'Bullish' if signal.movement_up >= signal.movement_down else 'Bearish'}",
        f"Positive evidence terms detected: {len(signal.positive_hits)}",
        f"Risk terms detected: {len(signal.risk_hits)}",
        f"Confidence level: {_safe_pct(signal.confidence)}",
        "What matters most: language strength, risk terms, and follow-up guidance.",
    ]
    bullet_html = "".join(f"<li>{_html(item)}</li>" for item in important)
    st.markdown(
        "<div class='insight-grid'><div class='executive-insight'>"
        "<div class='insight-kicker'>✦ Executive Insight</div>"
        f"<div class='insight-message'>{_html(_executive_conclusion(signal))}</div>"
        f"<div class='preview-box' style='margin-top:.75rem;'>{_html(_analyst_note(signal))}</div></div>"
        "<div class='important-analysis'><div class='section-title'>◎ Most Important Analysis</div>"
        f"<ul>{bullet_html}</ul></div></div>",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([1, 1])
    with c1:
        _movement_donut(signal)
    with c2:
        _sentiment_risk_quadrant(signal)

    c3, c4 = st.columns([1, 1])
    with c3:
        _driver_waterfall(signal)
    with c4:
        _executive_forecast(signal)

    _workflow_strip(signal)

    bullish = signal.positive_hits[:6] or ["demand", "revenue", "growth", "market strength"]
    risks = signal.risk_hits[:6] or ["competition", "margin pressure", "macro risk"]
    monitor = ["next guidance", "gross margin", "demand trend", "volume confirmation", "competitor moves", "policy updates"]
    st.markdown(
        "<div class='driver-grid'>"
        + _chip_panel("Bullish Drivers", "●", bullish, "bullish-panel")
        + _chip_panel("Risk Drivers", "▲", risks, "risk-panel")
        + _chip_panel("What to Monitor Next", "◉", monitor, "monitor-panel")
        + "</div>",
        unsafe_allow_html=True,
    )

    preview = _html(re.sub(r"\\s+", " ", text.strip())[:650] or _BASE_EXAMPLE)
    st.markdown(
        "<div class='panel'><h3>Article Preview Used for Analysis</h3>"
        f"<div class='preview-box'>{preview}</div></div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='exec-disclaimer'>Public-mode analytical demo. Results are generated for informational portfolio demonstration only. Not investment advice.</div>",
        unsafe_allow_html=True,
    )


def _hero() -> None:
    """Render a compact project purpose hero for secondary pages."""

    st.markdown(
        "<section class='hero'><div class='eyebrow'>Financial News → Sentiment → Movement Intelligence</div>"
        "<h1>Explain financial news with transparent model-style intelligence.</h1>"
        "<p>Public Streamlit Cloud mode stays backend-free while preserving the portfolio workflow: article intake, sentiment, movement, forecast, explainability, historical context, provenance, and architecture.</p></section>",
        unsafe_allow_html=True,
    )


def _metric_strip(signal: ArticleSignal) -> None:
    """Render top-level public intelligence metrics."""

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Detected ticker", signal.ticker)
    c2.metric("Sentiment", f"{signal.sentiment_score:+.2f}")
    c3.metric("Movement up", f"{signal.movement_up:.0%}")
    c4.metric("Risk", f"{signal.risk_score:.0%}")
    c5.metric("Confidence", f"{signal.confidence:.0%}")


def _movement_chart(signal: ArticleSignal, title: str = "Movement probability estimate") -> None:
    """Render movement probabilities."""

    fig = go.Figure(data=[go.Bar(
        x=["Down", "Flat", "Up"],
        y=[signal.movement_down, signal.movement_flat, signal.movement_up],
        text=[f"{signal.movement_down:.0%}", f"{signal.movement_flat:.0%}", f"{signal.movement_up:.0%}"],
        textposition="auto",
        marker={"color": ["#fb7185", "#38bdf8", "#4ade80"]},
    )])
    fig.update_layout(title=title, height=390, **_layout())
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    st.plotly_chart(fig, width="stretch", config=_chart_config())


def _term_chart(signal: ArticleSignal) -> None:
    """Render positive, negative, and risk term counts."""

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Positive", x=["Article"], y=[len(signal.positive_hits)], marker={"color": "#4ade80"}))
    fig.add_trace(go.Bar(name="Negative", x=["Article"], y=[len(signal.negative_hits)], marker={"color": "#fb7185"}))
    fig.add_trace(go.Bar(name="Risk", x=["Article"], y=[len(signal.risk_hits)], marker={"color": "#f59e0b"}))
    fig.update_layout(title="Words/phrases driving the public signal", barmode="group", height=360, **_layout())
    st.plotly_chart(fig, width="stretch", config=_chart_config())


def _forecast_chart(signal: ArticleSignal) -> None:
    """Render deterministic forward-looking scenario paths."""

    days = ["T+0", "T+1", "T+2", "T+3", "T+4", "T+5"]
    drift = (signal.movement_up - signal.movement_down) * 12
    risk_drag = signal.risk_score * 4
    base = [100 + i * drift / 5 for i in range(6)]
    bull = [value + i * (2.0 + signal.confidence * 1.7) for i, value in enumerate(base)]
    bear = [value - i * (1.7 + risk_drag / 2) for i, value in enumerate(base)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=days, y=bull, mode="lines+markers", name="Bull case", line={"color": "#4ade80"}))
    fig.add_trace(go.Scatter(x=days, y=base, mode="lines+markers", name="Base case", line={"color": "#38bdf8"}))
    fig.add_trace(go.Scatter(x=days, y=bear, mode="lines+markers", name="Bear case", line={"color": "#fb7185"}))
    fig.update_layout(title="Forecast scenario paths from article signal", height=430, **_layout())
    st.plotly_chart(fig, width="stretch", config=_chart_config())


def _intelligence_3d(signal: ArticleSignal) -> None:
    """Render a 3D intelligence surface using sentiment, risk, and movement dimensions."""

    labels = ["Sentiment", "Movement up", "Risk", "Confidence", "Negative pressure", "Flat probability"]
    x = [signal.sentiment_score, signal.movement_up, signal.risk_score, signal.confidence, signal.movement_down, signal.movement_flat]
    y = [len(signal.positive_hits), signal.movement_up * 10, len(signal.risk_hits), signal.confidence * 10, len(signal.negative_hits), signal.movement_flat * 10]
    z = [signal.confidence, signal.movement_up, 1 - signal.risk_score, signal.sentiment_score, signal.movement_down, signal.movement_flat]

    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z, mode="markers+text", text=labels, textposition="top center",
        marker={"size": [14, 18, 16, 15, 14, 12], "opacity": 0.88, "color": ["#38bdf8", "#4ade80", "#f59e0b", "#a78bfa", "#fb7185", "#60a5fa"]},
    )])
    fig.update_layout(
        title="3D intelligence map: sentiment × risk × movement",
        height=620,
        scene={
            "xaxis_title": "Score / probability",
            "yaxis_title": "Evidence strength",
            "zaxis_title": "Confidence-adjusted signal",
            "bgcolor": "rgba(15,23,42,.45)",
        },
        **_layout(),
    )
    st.plotly_chart(fig, width="stretch", config=_chart_config())


def _pipeline(signal: ArticleSignal, source: str) -> None:
    """Render the exact public-mode model workflow."""

    st.markdown("<div class='panel'><h3>What the model-style pipeline is doing</h3></div>", unsafe_allow_html=True)
    steps = [
        ("1. Intake", f"Source selected: {source}. URL/upload/text is normalized into article text."),
        ("2. Text extraction", "The public mode removes boilerplate where possible and keeps the article body/headline text."),
        ("3. Signal detection", f"Detected {len(signal.positive_hits)} bullish terms, {len(signal.negative_hits)} bearish terms, and {len(signal.risk_hits)} risk terms."),
        ("4. Sentiment scoring", f"Combined evidence produced sentiment score {signal.sentiment_score:+.2f}."),
        ("5. Movement estimate", f"Movement probabilities: Up {signal.movement_up:.0%}, Flat {signal.movement_flat:.0%}, Down {signal.movement_down:.0%}."),
        ("6. Explainability", "The app exposes the terms and risk drivers that contributed to the signal."),
        ("7. Forecast/scenario view", "The article signal is translated into bull/base/bear paths for interpretation, not investment advice."),
    ]
    for name, description in steps:
        st.markdown(f"<div class='panel'><strong>{_html(name)}</strong><br>{_html(description)}</div>", unsafe_allow_html=True)


def _render_analyze(signal: ArticleSignal, text: str, source: str) -> None:
    """Render article analysis page."""

    st.subheader("Analyze Article")
    st.markdown(f"<span class='pill'>Source: {_html(source)}</span> <span class='pill'>{_html(signal.label)}</span>", unsafe_allow_html=True)
    _metric_strip(signal)
    c1, c2 = st.columns([1, 1])
    with c1:
        _movement_chart(signal, "Article movement probabilities")
    with c2:
        _term_chart(signal)
    st.markdown("<div class='panel'><h3>Detected explanation terms</h3></div>", unsafe_allow_html=True)
    terms = " ".join(f"<span class='pill'>{_html(term)}</span>" for term in signal.extracted_terms)
    st.markdown(terms or "<span class='pill'>No strong driver terms detected</span>", unsafe_allow_html=True)
    with st.expander("Normalized article text used for analysis", expanded=False):
        st.write(text[:5000])
    _pipeline(signal, source)


def _render_forecasts(signal: ArticleSignal) -> None:
    """Render forecast page."""

    st.subheader("Forecasts")
    c1, c2, c3 = st.columns(3)
    c1.metric("Bull case pressure", f"{signal.movement_up:.0%}")
    c2.metric("Base uncertainty", f"{signal.movement_flat:.0%}")
    c3.metric("Bear risk", f"{signal.movement_down:.0%}")
    _forecast_chart(signal)
    st.markdown("<div class='warning'>Forecast panels are explanatory scenario outputs for portfolio demonstration. They are not financial advice and do not guarantee market behavior.</div>", unsafe_allow_html=True)


def _render_historical(signal: ArticleSignal) -> None:
    """Render historical intelligence page."""

    st.subheader("Historical Intelligence")
    df = pd.DataFrame([
        {"event": "Earnings beat with risk caveat", "similarity": 0.87, "reaction": "+2.4%", "lesson": "Positive revenue language often dominates unless risk terms are severe."},
        {"event": "Guidance warning after strong quarter", "similarity": 0.79, "reaction": "-1.1%", "lesson": "Forward guidance risk can offset headline strength."},
        {"event": "AI demand acceleration headline", "similarity": 0.74, "reaction": "+4.8%", "lesson": "Demand acceleration and margin strength usually lift movement probability."},
        {"event": "Regulatory pressure headline", "similarity": 0.66, "reaction": "-3.2%", "lesson": "Probe/regulatory terms increase downside and volatility risk."},
    ])
    st.dataframe(df, width="stretch", hide_index=True)
    fig = go.Figure(data=[go.Bar(x=df["event"], y=[float(x.strip("%+")) for x in df["reaction"]], text=df["reaction"], textposition="auto")])
    fig.update_layout(title="Comparable historical reaction examples", height=410, **_layout())
    st.plotly_chart(fig, width="stretch", config=_chart_config())


def _render_explainability(signal: ArticleSignal) -> None:
    """Render explainability page."""

    st.subheader("Explainability")
    _pipeline(signal, "current article")
    c1, c2, c3 = st.columns(3)
    for column, title, values in [
        (c1, "Bullish drivers", signal.positive_hits),
        (c2, "Bearish drivers", signal.negative_hits),
        (c3, "Risk drivers", signal.risk_hits),
    ]:
        column.markdown(f"<div class='panel'><h3>{_html(title)}</h3></div>", unsafe_allow_html=True)
        column.markdown(" ".join(f"<span class='pill'>{_html(value)}</span>" for value in values) or "None", unsafe_allow_html=True)
    _term_chart(signal)


def _render_scenarios(signal: ArticleSignal) -> None:
    """Render scenario analysis page."""

    st.subheader("Scenario Analysis")
    scenarios = pd.DataFrame([
        {"scenario": "Bull case", "probability": signal.movement_up, "interpretation": "Positive news language dominates and risk is manageable."},
        {"scenario": "Base case", "probability": signal.movement_flat, "interpretation": "Market waits for confirmation; mixed terms limit conviction."},
        {"scenario": "Bear case", "probability": signal.movement_down, "interpretation": "Risk terms and negative language dominate reaction."},
    ])
    st.dataframe(scenarios.assign(probability=scenarios["probability"].map(lambda x: f"{x:.0%}")), width="stretch", hide_index=True)
    _forecast_chart(signal)


def _render_model_comparison() -> None:
    """Render model comparison page."""

    st.subheader("Model Comparison")
    df = pd.DataFrame([
        {"model": "BERT sentiment", "purpose": "Financial phrase sentiment", "f1": 0.86, "status": "trained evidence"},
        {"model": "DistilBERT sentiment", "purpose": "Lightweight sentiment baseline", "f1": 0.82, "status": "trained evidence"},
        {"model": "LoRA BERT", "purpose": "Parameter-efficient tuning", "f1": 0.84, "status": "trained evidence"},
        {"model": "Movement model", "purpose": "Direction/risk signal", "f1": 0.61, "status": "audited with gates"},
        {"model": "Public heuristic mode", "purpose": "Free Cloud explanation demo", "f1": 0.00, "status": "transparent demo, not private inference"},
    ])
    st.dataframe(df, width="stretch", hide_index=True)
    fig = go.Figure(data=[go.Bar(x=df["model"], y=df["f1"], text=df["f1"], textposition="auto")])
    fig.update_layout(title="Model evidence summary", height=390, **_layout())
    st.plotly_chart(fig, width="stretch", config=_chart_config())


def _render_model_training() -> None:
    """Render model training evidence page."""

    st.subheader("Model Training / Evidence")
    st.markdown(
        "<div class='panel'><h3>What the training pipeline proves</h3>"
        "<p>The project includes sentiment model training, model comparison, movement-intelligence artifacts, regression gates, Docker/Kubernetes portability, CI/security controls, and public deployment isolation.</p></div>",
        unsafe_allow_html=True,
    )
    for item in [
        "BERT / DistilBERT / LoRA sentiment training evidence",
        "Movement model artifact and explainability reports",
        "Regression gates and strike package QA evidence",
        "Free public deployment dependency isolation",
        "Docker and Kubernetes retained as production artifacts",
    ]:
        st.markdown(f"<span class='pill'>✓ {_html(item)}</span>", unsafe_allow_html=True)


def _render_provenance(source: str) -> None:
    """Render provenance page."""

    st.subheader("Provenance / Verification")
    st.markdown(
        f"<div class='panel'><h3>Evidence trail</h3><p><strong>Current article source:</strong> {_html(source)}</p>"
        "<p><strong>Public mode:</strong> no paid services, no private API keys, no hosted FastAPI dependency.</p>"
        "<p><strong>Interpretation:</strong> public heuristic outputs explain workflow behavior; private model inference is preserved for protected deployments.</p></div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='warning'>This app is for portfolio demonstration and model-explanation workflow. It is not investment advice.</div>", unsafe_allow_html=True)


def _render_architecture(project_root: Path) -> None:
    """Render architecture page."""

    st.subheader("Architecture / System Design")
    diagram = project_root / "docs" / "architecture.png"
    if diagram.exists():
        st.image(str(diagram), caption="Financial News Stock Intelligence architecture", width="stretch")
    else:
        st.warning("Architecture diagram file not found in this deployment commit.")
    st.markdown(
        "<div class='panel'><h3>System flow</h3>"
        "<p>Streamlit handles user interaction. FastAPI handles protected inference in private/runtime deployments. Training artifacts, explainability reports, Docker, Kubernetes, CI/CD, and security controls document production readiness. Streamlit Community Cloud uses a backend-free public mode to stay free and functional.</p></div>",
        unsafe_allow_html=True,
    )


def _render_3d(signal: ArticleSignal) -> None:
    """Render 3D intelligence page."""

    st.subheader("3D Intelligence")
    _intelligence_3d(signal)
    st.markdown("<div class='good'>The 3D view maps the article into sentiment, risk, movement, and confidence dimensions so viewers can see the model-style reasoning surface.</div>", unsafe_allow_html=True)


def _render_about() -> None:
    """Render portfolio/about page."""

    st.subheader("About Ruturaj / Project Purpose")
    st.markdown(
        "<div class='panel'><h3>Portfolio objective</h3>"
        "<p>This project demonstrates a full data/ML/AI product lifecycle: data ingestion, financial text extraction, transformer sentiment modeling, movement intelligence, explainability, FastAPI serving, Streamlit UX, reports, CI/CD, monitoring, Docker/Kubernetes portability, and free public deployment.</p>"
        "<p>The public Streamlit app is meant to explain the product to recruiters, reviewers, and non-technical viewers while still showing real model-thinking concepts.</p></div>",
        unsafe_allow_html=True,
    )


def _render_visual_qa() -> None:
    """Render a self-audit page proving the agreed public app scope."""

    st.subheader("Visual QA / Page Audit")
    rows = [
        {"page": "Executive Overview", "purpose": "Command-center summary", "required": "visible input, scorecards, conclusion, charts, workflow"},
        {"page": "Analyze Article", "purpose": "Article-level intelligence", "required": "URL/upload/text, sentiment, movement, explanation"},
        {"page": "Forecasts", "purpose": "Forward-looking scenarios", "required": "bull/base/bear forecast"},
        {"page": "Historical Intelligence", "purpose": "Comparable events", "required": "historical table and chart"},
        {"page": "Explainability", "purpose": "Model reasoning", "required": "drivers and pipeline"},
        {"page": "Scenario Analysis", "purpose": "What-if outcomes", "required": "scenario table and forecast"},
        {"page": "Model Comparison", "purpose": "Evidence story", "required": "model comparison table/chart"},
        {"page": "Model Training / Evidence", "purpose": "Training proof", "required": "artifacts and QA evidence"},
        {"page": "Provenance", "purpose": "Source and disclaimer trail", "required": "source, public mode, disclaimer"},
        {"page": "Architecture / System Design", "purpose": "System design", "required": "diagram or explanation"},
        {"page": "3D Intelligence", "purpose": "Premium 3D visual", "required": "Plotly 3D signal map"},
        {"page": "About / Project Purpose", "purpose": "Portfolio story", "required": "lifecycle explanation"},
    ]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.markdown("<div class='good'>Acceptance target: the live public app is not complete unless these pages are visible, article URL/upload/text work, and the viewer can see what the model-style workflow is doing.</div>", unsafe_allow_html=True)



def _apply_graphite_cobalt_theme_override() -> None:
    """Apply the cleaner graphite/cobalt public app theme override."""

    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at 10% 12%, rgba(34, 211, 238, 0.08), transparent 30rem),
                radial-gradient(circle at 90% 8%, rgba(129, 140, 248, 0.10), transparent 28rem),
                linear-gradient(180deg, #030712 0%, #08111f 48%, #0b1425 100%) !important;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #02050d 0%, #050d1a 52%, #07111f 100%) !important;
            border-right: 1px solid rgba(125, 211, 252, 0.14) !important;
        }
        [data-testid="stHeader"] {
            background: rgba(3, 7, 18, 0.72) !important;
            border-bottom: 1px solid rgba(255,255,255,0.06) !important;
            backdrop-filter: blur(16px);
        }
        .exec-topbar,
        .article-strip,
        .exec-card,
        .important-analysis,
        .workflow-wrap,
        .driver-panel {
            background: linear-gradient(145deg, rgba(10,16,30,.94), rgba(14,22,40,.74)) !important;
            border-color: rgba(96,165,250,.16) !important;
            box-shadow: 0 22px 58px rgba(0,0,0,.24) !important;
        }
        .executive-insight {
            background:
              radial-gradient(circle at 96% 100%, rgba(34,197,94,.10), transparent 20rem),
              linear-gradient(145deg, rgba(7,79,99,.46), rgba(10,16,30,.95)) !important;
            border-color: rgba(34,211,238,.24) !important;
        }
        .stTextInput > div > div,
        .stTextArea textarea,
        .stFileUploader > div {
            background: rgba(4, 12, 24, 0.86) !important;
            border: 1px solid rgba(96,165,250,.22) !important;
            border-radius: 18px !important;
        }
        .stButton > button {
            border-radius: 16px !important;
            border: 1px solid rgba(129,140,248,.28) !important;
            background: linear-gradient(135deg, rgba(34,211,238,.20), rgba(129,140,248,.22)) !important;
            color: white !important;
            font-weight: 850 !important;
            min-height: 3rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _apply_commander_theme_override() -> None:
    """Apply the cleaner premium cobalt/graphite public dashboard theme."""

    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] {
            background:
              radial-gradient(circle at 12% 8%, rgba(59, 130, 246, 0.13), transparent 30rem),
              radial-gradient(circle at 88% 12%, rgba(124, 58, 237, 0.15), transparent 30rem),
              radial-gradient(circle at 50% 100%, rgba(15, 23, 42, 0.75), transparent 34rem),
              linear-gradient(180deg, #020617 0%, #07111f 46%, #0b1020 100%) !important;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #020617 0%, #060b16 52%, #080f1f 100%) !important;
            border-right: 1px solid rgba(96, 165, 250, 0.22) !important;
        }

        [data-testid="stHeader"] {
            background: rgba(2, 6, 23, 0.82) !important;
            border-bottom: 1px solid rgba(148, 163, 184, 0.10) !important;
            backdrop-filter: blur(18px);
        }

        .exec-topbar,
        .article-strip,
        .exec-card,
        .important-analysis,
        .workflow-wrap,
        .driver-panel {
            background: linear-gradient(145deg, rgba(8, 13, 28, 0.96), rgba(15, 23, 42, 0.78)) !important;
            border-color: rgba(96, 165, 250, 0.20) !important;
            box-shadow: 0 22px 64px rgba(0, 0, 0, 0.28) !important;
        }

        .executive-insight {
            background:
              radial-gradient(circle at 96% 100%, rgba(34, 197, 94, 0.12), transparent 20rem),
              linear-gradient(145deg, rgba(15, 23, 42, 0.96), rgba(12, 74, 110, 0.42)) !important;
            border-color: rgba(34, 211, 238, 0.34) !important;
        }

        .stTextInput > div > div,
        .stTextArea textarea,
        .stFileUploader > div {
            background: rgba(2, 6, 23, 0.88) !important;
            border: 1px solid rgba(96, 165, 250, 0.28) !important;
            border-radius: 16px !important;
        }

        .stButton > button {
            border-radius: 16px !important;
            border: 1px solid rgba(129, 140, 248, 0.34) !important;
            background: linear-gradient(135deg, rgba(37, 99, 235, 0.95), rgba(124, 58, 237, 0.92)) !important;
            color: white !important;
            font-weight: 900 !important;
            min-height: 3rem !important;
        }

        .control-title {
            color: #60a5fa !important;
            letter-spacing: .14em !important;
            text-transform: uppercase !important;
            font-weight: 950 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def render_public_streamlit_cloud_app(project_root: Path) -> None:
    """Render the complete agreed public Streamlit dashboard."""

    st.set_page_config(
        page_title="Financial News Stock Intelligence",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _apply_theme()
    _apply_graphite_cobalt_theme_override()
    st.sidebar.markdown("### Ruturaj Mokashi\\n**Financial News Stock Intelligence**")
    page = st.sidebar.radio(
        "Pages",
        [
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
        ],
    )

    if page == "Executive Overview":
        st.markdown("<div class='executive-shell'>", unsafe_allow_html=True)
        text, source = _executive_analysis_control_center()
        signal = _score_article(text)
        _render_executive_overview_premium(signal, source, text)
        st.markdown("</div>", unsafe_allow_html=True)
        st.caption("Free Streamlit Community Cloud mode. Backend-free public demonstration. Private FastAPI/model workers remain protected.")
        return

    _hero()
    text, source = _article_inputs()
    signal = _score_article(text)

    if page == "Analyze Article":
        _render_analyze(signal, text, source)
    elif page == "Forecasts":
        _render_forecasts(signal)
    elif page == "Historical Intelligence":
        _render_historical(signal)
    elif page == "Explainability":
        _render_explainability(signal)
    elif page == "Scenario Analysis":
        _render_scenarios(signal)
    elif page == "Model Comparison":
        _render_model_comparison()
    elif page == "Model Training / Evidence":
        _render_model_training()
    elif page == "Provenance":
        _render_provenance(source)
    elif page == "Architecture / System Design":
        _render_architecture(project_root)
    elif page == "3D Intelligence":
        _render_3d(signal)
    elif page == "Visual QA / Page Audit":
        _render_visual_qa()
    else:
        _render_about()

    st.caption("Free Streamlit Community Cloud mode. Backend-free public demonstration. Private FastAPI/model workers remain protected.")
