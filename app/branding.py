"""Provide safe, reusable portfolio branding for Ruturaj Mokashi.

All text is kept in one place so future pages and downloadable reports use the
same product name, author credit, and plain-language description.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape


@dataclass(frozen=True)
class PortfolioBrand:
    """Store approved public branding text.

    The values are public portfolio information. This object must never contain
    an email address, token, local path, or private provider detail.
    """

    owner_name: str
    product_name: str
    short_name: str
    tagline: str
    portfolio_statement: str


def get_portfolio_brand() -> PortfolioBrand:
    """Return the approved project identity used throughout the interface."""

    return PortfolioBrand(
        owner_name="Ruturaj Mokashi",
        product_name="Financial News & Stock Movement Intelligence",
        short_name="Financial Intelligence",
        tagline=(
            "Clear sentiment, market forecasts, explanations, and historical "
            "evidence from one verified workflow."
        ),
        portfolio_statement=(
            "Designed as an end-to-end AI and data product with testing, "
            "security, monitoring, containerization, and deployment in scope."
        ),
    )


def build_brand_header_html(brand: PortfolioBrand) -> str:
    """Build escaped HTML for the main product header.

    Args:
        brand: Approved public text to display.

    Returns:
        A complete HTML fragment. Every value is escaped before insertion so a
        future text change cannot inject active browser content.
    """

    return f"""
    <section class="rm-hero" aria-labelledby="rm-product-title">
      <div class="rm-hero-glow rm-hero-glow-one" aria-hidden="true"></div>
      <div class="rm-hero-glow rm-hero-glow-two" aria-hidden="true"></div>
      <div class="rm-hero-content">
        <p class="rm-eyebrow">AI FINANCIAL INTELLIGENCE</p>
        <h1 id="rm-product-title">{escape(brand.product_name)}</h1>
        <p class="rm-hero-tagline">{escape(brand.tagline)}</p>
        <div class="rm-author-line">
          <span class="rm-author-dot" aria-hidden="true"></span>
          Designed and built by <strong>{escape(brand.owner_name)}</strong>
        </div>
      </div>
      <div class="rm-hero-orbit" aria-hidden="true">
        <span class="rm-orbit-ring rm-ring-one"></span>
        <span class="rm-orbit-ring rm-ring-two"></span>
        <span class="rm-orbit-core">RM</span>
      </div>
    </section>
    """.strip()


def build_footer_html(brand: PortfolioBrand) -> str:
    """Build escaped footer HTML with clear ownership and product scope."""

    return f"""
    <footer class="rm-footer">
      <div>
        <strong>{escape(brand.owner_name)}</strong>
        <span> • Designed and built as a portfolio-quality AI product</span>
      </div>
      <div class="rm-footer-note">{escape(brand.portfolio_statement)}</div>
    </footer>
    """.strip()
