"""Render shared page layout, premium cards, and plain-language explanations.

The functions in this module accept the imported Streamlit module as an input.
This keeps visual code separate from the entry point and allows focused tests to
check HTML and content without starting a web server.
"""

from __future__ import annotations

from html import escape
from typing import Any

from app.branding import (
    PortfolioBrand,
    build_brand_header_html,
    build_footer_html,
)
from app.configuration import AppSettings
from app.navigation import NavigationItem


_MAX_STYLESHEET_BYTES = 200_000


def _read_approved_stylesheet(settings: AppSettings) -> str:
    """Read the local stylesheet after size and content safety checks.

    The stylesheet is application source, not user input. The size limit catches
    an accidental replacement with a large or unrelated file. Rejecting a
    closing ``style`` tag prevents the file from breaking out of its wrapper.
    """

    stylesheet_size = settings.css_path.stat().st_size
    if stylesheet_size > _MAX_STYLESHEET_BYTES:
        raise ValueError(
            "The premium stylesheet is larger than the approved 200 KB limit."
        )

    css_text = settings.css_path.read_text(encoding="utf-8")
    if not css_text.strip():
        raise ValueError("The premium stylesheet must not be empty.")
    if "</style" in css_text.lower():
        raise ValueError("The premium stylesheet contains an unsafe closing tag.")
    return css_text


def apply_premium_theme(st: Any, settings: AppSettings) -> None:
    """Load the approved local CSS into the current Streamlit page."""

    css_text = _read_approved_stylesheet(settings)
    st.markdown(f"<style>{css_text}</style>", unsafe_allow_html=True)


def render_page_header(
    st: Any,
    brand: PortfolioBrand,
    selected_page: NavigationItem,
) -> None:
    """Render the product hero and a small selected-page label."""

    st.markdown(build_brand_header_html(brand), unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="rm-current-page">
          <span>Current section</span>
          <strong>{escape(selected_page.label)}</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_foundation_overview(st: Any, brand: PortfolioBrand) -> None:
    """Show the Package 1 landing experience and honest delivery status.

    The values here describe verified project direction, not live model output.
    Live health, sentiment, and movement values will be connected in Package 2
    and later packages.
    """

    st.markdown(
        """
        <section class="rm-section-heading">
          <p class="rm-eyebrow">PORTFOLIO PRODUCT FOUNDATION</p>
          <h2>A clear path from financial news to market intelligence</h2>
          <p>
            This first package establishes the visual system, navigation, page
            structure, and personal branding. Live analysis is deliberately not
            shown until the FastAPI connection passes its own checks.
          </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    columns = st.columns(4, gap="medium")
    cards = (
        (
            "01",
            "Understand the news",
            "Measure positive, neutral, and negative meaning in plain words.",
            "Package 3",
        ),
        (
            "02",
            "Forecast the session",
            "Show Down, Flat, and Up chances without hiding uncertainty.",
            "Package 4",
        ),
        (
            "03",
            "Explain the result",
            "Show which factors mattered and what conclusion follows.",
            "Package 5",
        ),
        (
            "04",
            "Verify the evidence",
            "Connect every result to its source, model, checks, and limits.",
            "Package 6",
        ),
    )
    for column, (number, title, body, package_label) in zip(columns, cards):
        with column:
            st.markdown(
                f"""
                <article class="rm-feature-card">
                  <div class="rm-feature-number">{number}</div>
                  <h3>{title}</h3>
                  <p>{body}</p>
                  <span class="rm-package-pill">{package_label}</span>
                </article>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('<div class="rm-section-gap"></div>', unsafe_allow_html=True)
    left_column, right_column = st.columns([1.45, 1], gap="large")

    with left_column:
        st.markdown(
            """
            <section class="rm-panel rm-panel-primary">
              <div class="rm-panel-kicker">PRODUCT EXPERIENCE</div>
              <h3>Advanced visuals with simple conclusions</h3>
              <p>
                Every important chart will explain what it shows, why it
                matters, and the practical conclusion. Carefully chosen 3D
                charts will always include a simpler 2D view.
              </p>
              <div class="rm-explanation-grid">
                <div>
                  <span>WHAT IT SHOWS</span>
                  <strong>The evidence visible in the chart</strong>
                </div>
                <div>
                  <span>WHY IT MATTERS</span>
                  <strong>The business or model meaning</strong>
                </div>
                <div>
                  <span>CONCLUSION</span>
                  <strong>The clear point the user should take away</strong>
                </div>
              </div>
            </section>
            """,
            unsafe_allow_html=True,
        )

    with right_column:
        st.markdown(
            f"""
            <section class="rm-panel rm-owner-panel">
              <div class="rm-owner-monogram" aria-hidden="true">RM</div>
              <div class="rm-panel-kicker">PROJECT OWNER</div>
              <h3>{escape(brand.owner_name)}</h3>
              <p>
                Building an end-to-end AI portfolio product across data,
                models, APIs, interface design, testing, containerization,
                logging, monitoring, and deployment.
              </p>
              <div class="rm-verified-line">
                <span aria-hidden="true">✓</span>
                Claims are updated only after their verification gates pass.
              </div>
            </section>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="rm-section-gap"></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <section class="rm-panel rm-roadmap-panel">
          <div class="rm-panel-kicker">STREAMLIT DELIVERY MAP</div>
          <h3>Eight controlled packages</h3>
          <div class="rm-roadmap">
            <div class="rm-roadmap-item is-current"><b>1</b><span>Foundation</span></div>
            <div class="rm-roadmap-line"></div>
            <div class="rm-roadmap-item"><b>2</b><span>API connection</span></div>
            <div class="rm-roadmap-line"></div>
            <div class="rm-roadmap-item"><b>3</b><span>Overview & Analyze</span></div>
            <div class="rm-roadmap-line"></div>
            <div class="rm-roadmap-item"><b>4</b><span>Forecasts & History</span></div>
            <div class="rm-roadmap-line"></div>
            <div class="rm-roadmap-item"><b>5</b><span>Why & Models</span></div>
            <div class="rm-roadmap-line"></div>
            <div class="rm-roadmap-item"><b>6</b><span>Scenario & Evidence</span></div>
            <div class="rm-roadmap-line"></div>
            <div class="rm-roadmap-item"><b>7</b><span>Tests & Security</span></div>
            <div class="rm-roadmap-line"></div>
            <div class="rm-roadmap-item"><b>8</b><span>Final verification</span></div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_planned_page(st: Any, selected_page: NavigationItem) -> None:
    """Show an honest, useful state for a page delivered in a later package."""

    st.markdown(
        f"""
        <section class="rm-planned-page">
          <div class="rm-planned-icon" aria-hidden="true">{escape(selected_page.icon)}</div>
          <p class="rm-eyebrow">PLANNED AND AUDITED</p>
          <h2>{escape(selected_page.label)}</h2>
          <p>{escape(selected_page.summary)}</p>
          <div class="rm-planned-callout">
            <strong>Delivery:</strong> Streamlit Strike Package
            {selected_page.package_number}
          </div>
          <p class="rm-muted-copy">
            This page is visible in the navigation so the complete product map
            is clear. It will not show sample or invented results before its
            real FastAPI connection and tests pass.
          </p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_portfolio_footer(st: Any, brand: PortfolioBrand) -> None:
    """Render the shared project-owner footer."""

    st.markdown(build_footer_html(brand), unsafe_allow_html=True)
