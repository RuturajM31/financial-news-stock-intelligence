"""Define the audited Streamlit page map through Package 7.

Available pages are marked honestly. Future sections remain visible so reviewers
can understand the full product plan without mistaking planned work for finished
functionality.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.branding import PortfolioBrand


@dataclass(frozen=True)
class NavigationItem:
    """Describe one user-facing page and its delivery status."""

    key: str
    label: str
    icon: str
    summary: str
    package_number: int
    is_available: bool

    @property
    def display_label(self) -> str:
        """Return an accessible label with a package note for future pages."""

        if self.is_available:
            return f"{self.icon}  {self.label}"
        return f"{self.icon}  {self.label}  ·  Package {self.package_number}"


_NAVIGATION_ITEMS = (
    NavigationItem(
        key="executive_overview",
        label="Executive Overview",
        icon="◈",
        summary="See live service status, model roles, and recent session results.",
        package_number=3,
        is_available=True,
    ),
    NavigationItem(
        key="analyze",
        label="Analyze News",
        icon="⌁",
        summary="Add text, a file, or a public link and receive clear results.",
        package_number=3,
        is_available=True,
    ),
    NavigationItem(
        key="forecasts",
        label="Forecasts",
        icon="↗",
        summary="Review Down, Flat, and Up chances with a clear conclusion.",
        package_number=4,
        is_available=True,
    ),
    NavigationItem(
        key="historical_intelligence",
        label="Historical Intelligence",
        icon="◷",
        summary="Compare the current event with strictly earlier market events.",
        package_number=4,
        is_available=True,
    ),
    NavigationItem(
        key="explainability",
        label="Why This Result",
        icon="◎",
        summary="See which words and factors pushed the result in each direction.",
        package_number=5,
        is_available=True,
    ),
    NavigationItem(
        key="model_training",
        label="Model Training & Results",
        icon="▦",
        summary="Review training data, progress, mistakes, and verified results.",
        package_number=6,
        is_available=True,
    ),
    NavigationItem(
        key="model_comparison",
        label="Model Comparison",
        icon="◇",
        summary="Compare BERT, DistilBERT, and BERT LoRA using verified evidence.",
        package_number=6,
        is_available=True,
    ),
    NavigationItem(
        key="scenario_analysis",
        label="Scenario Analysis",
        icon="△",
        summary="Explore possible downside, central, and upside examples.",
        package_number=7,
        is_available=True,
    ),
    NavigationItem(
        key="provenance",
        label="Evidence & Verification",
        icon="✓",
        summary="See the source, model, checks, and limits behind each result.",
        package_number=7,
        is_available=True,
    ),
    NavigationItem(
        key="about_ruturaj",
        label="About Ruturaj Mokashi",
        icon="RM",
        summary="Review the project story, skills, and engineering approach.",
        package_number=7,
        is_available=True,
    ),
)


def get_navigation_items() -> tuple[NavigationItem, ...]:
    """Return the complete page map in display order."""

    return _NAVIGATION_ITEMS


def get_navigation_item(key: str) -> NavigationItem:
    """Return one page definition or fail clearly for an unknown key."""

    for item in _NAVIGATION_ITEMS:
        if item.key == key:
            return item
    raise ValueError(f"Unknown Streamlit page key: {key!r}.")


def render_sidebar_navigation(st: Any, brand: PortfolioBrand) -> str:
    """Render branded sidebar navigation and return the selected page key."""

    st.sidebar.markdown(
        f"""
        <div class="rm-sidebar-brand">
          <div class="rm-sidebar-mark" aria-hidden="true">RM</div>
          <div>
            <div class="rm-sidebar-owner">{brand.owner_name}</div>
            <div class="rm-sidebar-product">{brand.short_name}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        '<p class="rm-sidebar-label">PRODUCT SECTIONS</p>',
        unsafe_allow_html=True,
    )

    items = get_navigation_items()
    labels = [item.display_label for item in items]
    selected_label = st.sidebar.radio(
        "Product sections",
        labels,
        index=0,
        label_visibility="collapsed",
        key="rm_primary_navigation",
    )
    label_to_key = {item.display_label: item.key for item in items}

    st.sidebar.markdown(
        """
        <div class="rm-sidebar-status">
          <span class="rm-status-dot" aria-hidden="true"></span>
          <div>
            <strong>Packages 1–7 ready</strong>
            <small>Core analysis, scenarios, provenance, reports, and portfolio pages passed.</small>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return label_to_key[selected_label]
