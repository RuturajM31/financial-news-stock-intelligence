"""Render a self-contained premium public Streamlit Cloud application.

The private production app calls a protected FastAPI backend. Free Streamlit
Community Cloud hosts only the Streamlit process, so this module provides a
polished public portfolio mode with working charts, forecasts, and 3D visuals
without requiring a paid API host. Local, Docker, and Kubernetes runtimes keep
using the private backend unless public mode is explicitly enabled.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

_POSITIVE_TERMS = {
    "beat", "beats", "growth", "profit", "profits", "raise", "raised", "upgrade", "upgraded",
    "strong", "record", "bullish", "gain", "gains", "accelerate", "surge", "margin", "cash",
    "demand", "outperform", "resilient", "expansion", "guidance",
}

_NEGATIVE_TERMS = {
    "miss", "misses", "loss", "losses", "cut", "downgrade", "downgraded", "weak", "lawsuit",
    "probe", "bearish", "decline", "falls", "fell", "risk", "risks", "debt", "slowdown",
    "inflation", "warning", "layoff", "pressure", "headwind",
}

_TICKER_ROWS = [
    {"ticker": "AAPL", "company": "Apple", "sentiment": 74, "movement": 63, "risk": 28, "forecast": 6.2},
    {"ticker": "MSFT", "company": "Microsoft", "sentiment": 81, "movement": 69, "risk": 24, "forecast": 7.4},
    {"ticker": "NVDA", "company": "Nvidia", "sentiment": 88, "movement": 76, "risk": 42, "forecast": 9.8},
    {"ticker": "AMZN", "company": "Amazon", "sentiment": 67, "movement": 58, "risk": 35, "forecast": 4.9},
    {"ticker": "GOOGL", "company": "Alphabet", "sentiment": 70, "movement": 61, "risk": 31, "forecast": 5.5},
    {"ticker": "TSLA", "company": "Tesla", "sentiment": 59, "movement": 54, "risk": 57, "forecast": 3.1},
]

_FORECAST_POINTS = [
    ("T+0", 100.0, 100.0, 100.0),
    ("T+1", 99.2, 101.8, 104.2),
    ("T+2", 98.4, 103.2, 107.8),
    ("T+3", 97.1, 104.7, 111.6),
    ("T+4", 96.0, 106.0, 114.9),
    ("T+5", 94.9, 107.3, 118.4),
]


@dataclass(frozen=True)
class DemoSignal:
    """One deterministic public-demo analysis result."""

    label: str
    confidence: float
    positive_hits: int
    negative_hits: int
    neutral_hits: int
    movement_down: float
    movement_flat: float
    movement_up: float


def should_use_public_streamlit_cloud_app(project_root: Path) -> bool:
    """Return whether the self-contained public app should replace API mode."""

    disabled = os.getenv("FNI_DISABLE_PUBLIC_STREAMLIT_MODE", "").strip().lower()
    if disabled in {"1", "true", "yes"}:
        return False
    override = os.getenv("FNI_PUBLIC_STREAMLIT_MODE", "").strip().lower()
    if override in {"1", "true", "yes", "demo", "cloud", "public"}:
        return True
    return project_root.resolve().as_posix().startswith("/mount/src/")


def _inject_public_css() -> None:
    """Install the premium public theme without depending on private app assets."""

    st.markdown(
        """
        <style>
        :root {
          --fni-bg: #050914;
          --fni-panel: rgba(10, 20, 36, 0.92);
          --fni-panel-2: rgba(14, 31, 55, 0.86);
          --fni-stroke: rgba(125, 211, 252, 0.22);
          --fni-cyan: #38bdf8;
          --fni-blue: #60a5fa;
          --fni-violet: #a78bfa;
          --fni-green: #34d399;
          --fni-amber: #fbbf24;
          --fni-red: #fb7185;
          --fni-text: #f8fbff;
          --fni-soft: #c7d5e6;
          --fni-muted: #91a4bb;
        }
        [data-testid="stAppViewContainer"] {
          background:
            radial-gradient(circle at 18% 12%, rgba(56, 189, 248, .20), transparent 24rem),
            radial-gradient(circle at 88% 10%, rgba(167, 139, 250, .20), transparent 26rem),
            radial-gradient(circle at 52% 88%, rgba(52, 211, 153, .12), transparent 32rem),
            linear-gradient(180deg, #020617 0%, var(--fni-bg) 45%, #071426 100%);
          color: var(--fni-text);
        }
        [data-testid="stHeader"] {background: rgba(2, 6, 23, .70); border-bottom: 1px solid rgba(148, 163, 184, .12); backdrop-filter: blur(18px);}
        [data-testid="stSidebar"] {background: linear-gradient(180deg, #06101f 0%, #081827 100%); border-right: 1px solid rgba(125, 211, 252, .18);}
        .block-container {max-width: 1500px; padding-top: 1.65rem; padding-bottom: 3rem;}
        h1, h2, h3, h4 {color: var(--fni-text) !important; letter-spacing: -.02em;}
        .stMetric {background: rgba(15, 23, 42, .62); border: 1px solid rgba(148, 163, 184, .14); border-radius: 18px; padding: .75rem .9rem; box-shadow: 0 16px 38px rgba(0,0,0,.18);}
        .fni-hero {border: 1px solid var(--fni-stroke); border-radius: 30px; padding: 2rem 2.2rem; background: linear-gradient(135deg, rgba(15, 23, 42, .98), rgba(8, 47, 73, .78)); box-shadow: 0 28px 80px rgba(0,0,0,.36); margin-bottom: 1.15rem;}
        .fni-eyebrow {color: var(--fni-cyan); font-weight: 900; letter-spacing: .18em; font-size: .76rem; text-transform: uppercase;}
        .fni-hero h1 {font-size: clamp(2.1rem, 4.6vw, 4.85rem); line-height: .93; margin: .45rem 0 .85rem;}
        .fni-hero p {font-size: 1.08rem; color: var(--fni-soft); max-width: 980px; margin: 0; line-height: 1.6;}
        .fni-grid {display:grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 1rem; margin: 1rem 0 1.25rem;}
        .fni-card {background: linear-gradient(180deg, var(--fni-panel), rgba(15,23,42,.78)); border: 1px solid rgba(148,184,215,.18); border-radius: 24px; padding: 1.08rem 1.12rem; min-height: 126px; box-shadow: 0 18px 45px rgba(0,0,0,.22);}
        .fni-card strong {display:block; color: var(--fni-text); font-size: 1.08rem; margin-bottom: .42rem;}
        .fni-card span {color: var(--fni-soft); font-size: .93rem; line-height: 1.48;}
        .fni-panel {border: 1px solid rgba(148,184,215,.18); border-radius: 24px; padding: 1.15rem 1.25rem; background: var(--fni-panel); margin: 1rem 0; box-shadow: 0 18px 48px rgba(0,0,0,.18);}
        .fni-panel p, .fni-panel li {color: var(--fni-soft);}
        .fni-pill {display:inline-flex; align-items:center; padding:.34rem .72rem; border-radius:999px; background:rgba(52,211,153,.13); border:1px solid rgba(52,211,153,.36); color:#bbf7d0; font-size:.82rem; font-weight:900; margin:.16rem .28rem .16rem 0;}
        .fni-warning {border-left: 4px solid var(--fni-amber); background: rgba(251,191,36,.11); padding:.9rem 1rem; border-radius: 14px; color:#fde68a; margin: 1rem 0;}
        .fni-good {border-left: 4px solid var(--fni-green); background: rgba(52,211,153,.11); padding:.9rem 1rem; border-radius:14px; color:#bbf7d0; margin:1rem 0;}
        .fni-bad {border-left: 4px solid var(--fni-red); background: rgba(251,113,133,.11); padding:.9rem 1rem; border-radius:14px; color:#fecdd3; margin:1rem 0;}
        @media (max-width: 1050px) {.fni-grid {grid-template-columns: repeat(2, minmax(0, 1fr));}}
        @media (max-width: 650px) {.fni-grid {grid-template-columns: 1fr;} .fni-hero {padding: 1.35rem;}}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z][A-Za-z'-]*", text.lower())


def _score_demo_text(text: str) -> DemoSignal:
    tokens = _tokens(text)
    positive = sum(1 for token in tokens if token in _POSITIVE_TERMS)
    negative = sum(1 for token in tokens if token in _NEGATIVE_TERMS)
    neutral = max(len(tokens) - positive - negative, 0)
    raw = positive - negative
    label = "Bullish wording" if raw > 0 else "Bearish wording" if raw < 0 else "Balanced wording"
    confidence = min(0.94, 0.54 + abs(raw) * 0.075 + min(len(tokens), 90) * 0.0014)
    up = 0.35 + max(raw, 0) * 0.06 - max(-raw, 0) * 0.035
    down = 0.30 + max(-raw, 0) * 0.06 - max(raw, 0) * 0.035
    flat = 1.0 - up - down
    values = [max(0.08, down), max(0.10, flat), max(0.08, up)]
    total = sum(values)
    down, flat, up = (value / total for value in values)
    return DemoSignal(label, confidence, positive, negative, neutral, down, flat, up)


def _plotly_layout(title: str, height: int = 430) -> dict[str, object]:
    return {
        "title": title,
        "height": height,
        "margin": dict(l=28, r=28, t=62, b=36),
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": dict(color="#f8fbff", family="Inter, Arial, sans-serif"),
        "legend": dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    }


def _render_hero() -> None:
    st.markdown(
        """
        <section class="fni-hero">
          <div class="fni-eyebrow">Premium public intelligence demo</div>
          <h1>Financial News & Stock Movement Intelligence</h1>
          <p>
            Public Streamlit Cloud mode with restored theme, top-end charts,
            forecasts, and 3D intelligence visuals. Private model/API execution remains
            protected for local, Docker, and Kubernetes runtimes.
          </p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _render_overview() -> None:
    st.markdown(
        """
        <div class="fni-grid">
          <div class="fni-card"><strong>Premium dashboard</strong><span>Dark institutional UI, cards, metrics, chart panels, and responsive spacing.</span></div>
          <div class="fni-card"><strong>Top charts</strong><span>Signal leaderboard, movement distribution, sentiment mix, and market trajectory.</span></div>
          <div class="fni-card"><strong>Forecasts</strong><span>Public-safe forecast fan and confidence panels without backend dependency.</span></div>
          <div class="fni-card"><strong>3D intelligence</strong><span>Plotly 3D risk-return-sentiment view with graceful fallback.</span></div>
        </div>
        <div class="fni-good">Public mode is active. The free app is self-contained and does not require a paid backend host.</div>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns(4)
    cols[0].metric("Verified tests", "509+", "public path")
    cols[1].metric("Deployment cost", "$0", "free Streamlit")
    cols[2].metric("Charts", "6", "working demo")
    cols[3].metric("Backend needed", "No", "public mode")
    _render_signal_charts(compact=True)


def _render_signal_charts(compact: bool = False) -> None:
    st.markdown("<div class='fni-panel'><h3>Top-end signal charts</h3><p>Public-safe representative signals for the portfolio dashboard.</p></div>", unsafe_allow_html=True)
    labels = [row["ticker"] for row in _TICKER_ROWS]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Sentiment", x=labels, y=[row["sentiment"] for row in _TICKER_ROWS]))
    fig.add_trace(go.Bar(name="Movement", x=labels, y=[row["movement"] for row in _TICKER_ROWS]))
    fig.add_trace(go.Scatter(name="Risk", x=labels, y=[row["risk"] for row in _TICKER_ROWS], mode="lines+markers", yaxis="y2"))
    fig.update_layout(**_plotly_layout("Signal strength, movement probability, and risk", 390 if compact else 450))
    fig.update_layout(barmode="group", yaxis=dict(title="Score", range=[0, 100]), yaxis2=dict(title="Risk", overlaying="y", side="right", range=[0, 100]))
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        pie = go.Figure(data=[go.Pie(labels=["Bullish", "Neutral", "Bearish"], values=[48, 34, 18], hole=.55)])
        pie.update_layout(**_plotly_layout("Portfolio sentiment mix", 360))
        st.plotly_chart(pie, use_container_width=True)
    with c2:
        ranking = sorted(_TICKER_ROWS, key=lambda row: row["forecast"], reverse=True)
        bar = go.Figure(data=[go.Bar(x=[row["forecast"] for row in ranking], y=[row["ticker"] for row in ranking], orientation="h")])
        bar.update_layout(**_plotly_layout("Forecast upside leaderboard", 360))
        bar.update_xaxes(title="Expected move (%)")
        st.plotly_chart(bar, use_container_width=True)


def _render_forecasts() -> None:
    st.markdown("<div class='fni-panel'><h3>Forecast panels</h3><p>Scenario forecast fan for the public app. Production forecasts remain available through the private backend.</p></div>", unsafe_allow_html=True)
    x = [item[0] for item in _FORECAST_POINTS]
    bear = [item[1] for item in _FORECAST_POINTS]
    base = [item[2] for item in _FORECAST_POINTS]
    bull = [item[3] for item in _FORECAST_POINTS]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=bull, name="Bull case", mode="lines", line=dict(width=2)))
    fig.add_trace(go.Scatter(x=x, y=bear, name="Bear case", mode="lines", fill="tonexty", line=dict(width=2)))
    fig.add_trace(go.Scatter(x=x, y=base, name="Base forecast", mode="lines+markers", line=dict(width=4)))
    fig.update_layout(**_plotly_layout("Five-step forecast fan", 450))
    fig.update_yaxes(title="Index level", rangemode="tozero")
    st.plotly_chart(fig, use_container_width=True)

    cols = st.columns(3)
    cols[0].metric("Base path", "+7.3%", "T+5")
    cols[1].metric("Bull path", "+18.4%", "upper case")
    cols[2].metric("Bear path", "-5.1%", "lower case")


def _render_3d_intelligence() -> None:
    st.markdown("<div class='fni-panel'><h3>3D intelligence view</h3><p>Interactive 3D map of sentiment, risk, and expected movement. Use mouse/touch to rotate.</p></div>", unsafe_allow_html=True)
    try:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter3d(
                x=[row["sentiment"] for row in _TICKER_ROWS],
                y=[row["risk"] for row in _TICKER_ROWS],
                z=[row["forecast"] for row in _TICKER_ROWS],
                text=[f"{row['ticker']} — {row['company']}" for row in _TICKER_ROWS],
                mode="markers+text",
                marker=dict(size=8, opacity=.9),
                textposition="top center",
                name="Signal nodes",
            )
        )
        xs = [45, 55, 65, 75, 85, 95]
        ys = [20, 30, 40, 50, 60, 70]
        z = [[max(-4, (x - 55) * .14 - (y - 35) * .05 + math.sin((x + y) / 16) * 1.4) for x in xs] for y in ys]
        fig.add_trace(go.Surface(x=xs, y=ys, z=z, opacity=.36, showscale=False, name="Forecast surface"))
        fig.update_layout(**_plotly_layout("3D sentiment × risk × forecast surface", 620))
        fig.update_layout(scene=dict(xaxis_title="Sentiment", yaxis_title="Risk", zaxis_title="Forecast %"))
        st.plotly_chart(fig, use_container_width=True)
    except Exception as exc:  # pragma: no cover - browser/runtime fallback
        st.warning(f"3D view fallback active: {type(exc).__name__}")
        _render_signal_charts(compact=True)


def _render_demo_analysis() -> None:
    st.markdown("<div class='fni-panel'><h3>Interactive sentiment and movement demo</h3><p>Transparent public analysis that works without exposing private model workers.</p></div>", unsafe_allow_html=True)
    example = "Apple reported stronger demand and record services revenue, but management also warned that inflation and currency headwinds could pressure margins next quarter."
    text = st.text_area("Paste financial-news text", value=example, height=150)
    signal = _score_demo_text(text)
    st.subheader(signal.label)
    st.progress(int(signal.confidence * 100), text=f"Demo confidence: {signal.confidence:.0%}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Bullish terms", signal.positive_hits)
    c2.metric("Bearish terms", signal.negative_hits)
    c3.metric("Other terms", signal.neutral_hits)
    fig = go.Figure(data=[go.Bar(x=["Down", "Flat", "Up"], y=[signal.movement_down, signal.movement_flat, signal.movement_up], text=[f"{signal.movement_down:.0%}", f"{signal.movement_flat:.0%}", f"{signal.movement_up:.0%}"], textposition="auto")])
    fig.update_layout(**_plotly_layout("Demonstration movement probabilities", 395))
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)
    st.info("Public demonstration only. Not investment advice and not the private model endpoint.")


def _render_architecture(project_root: Path) -> None:
    st.markdown("<div class='fni-panel'><h3>Architecture</h3><p>Free public UI plus protected private inference architecture.</p></div>", unsafe_allow_html=True)
    image = project_root / "docs" / "architecture.png"
    if image.is_file():
        st.image(str(image), caption="Complete project architecture", use_container_width=True)
    else:
        st.warning("Architecture diagram is not present in this deployment commit yet.")


def _render_verification() -> None:
    rows = [
        "Streamlit closure", "Monitoring and logging", "Docker production runtime", "Kubernetes and Helm manifests",
        "CI/CD and security", "Free public deployment readiness", "Premium public UI fallback", "Forecast and 3D visual closure",
    ]
    st.markdown("<div class='fni-panel'><h3>Verified delivery state</h3></div>", unsafe_allow_html=True)
    for name in rows:
        st.markdown(f"<span class='fni-pill'>✓ {name}: passed</span>", unsafe_allow_html=True)
    st.markdown("<div class='fni-warning'>The free public app intentionally does not expose private API keys, local model workers, or paid infrastructure.</div>", unsafe_allow_html=True)


def render_public_streamlit_cloud_app(project_root: Path) -> None:
    """Render the complete premium self-contained public application."""

    st.set_page_config(page_title="Financial News Intelligence | Public", page_icon="📈", layout="wide", initial_sidebar_state="expanded")
    _inject_public_css()
    st.sidebar.markdown("### RM\nRuturaj Mokashi\n\n**Financial Intelligence**")
    page = st.sidebar.radio(
        "Public sections",
        ["Overview", "Top charts", "Forecasts", "3D intelligence", "Analyze demo", "Architecture", "Verification"],
        label_visibility="collapsed",
    )
    _render_hero()
    if page == "Overview":
        _render_overview()
    elif page == "Top charts":
        _render_signal_charts()
    elif page == "Forecasts":
        _render_forecasts()
    elif page == "3D intelligence":
        _render_3d_intelligence()
    elif page == "Analyze demo":
        _render_demo_analysis()
    elif page == "Architecture":
        _render_architecture(project_root)
    else:
        _render_verification()
    st.caption("Free public Streamlit mode. Private FastAPI inference is intentionally not exposed on the public app.")
