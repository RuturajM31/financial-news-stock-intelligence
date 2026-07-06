"""Create plain-language explanations below every important chart."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from typing import Any


@dataclass(frozen=True)
class ChartExplanation:
    """Store the three required parts of a useful chart explanation."""

    what_it_shows: str
    why_it_matters: str
    conclusion: str
    uncertainty: str | None = None

    def validate(self) -> None:
        """Reject empty explanations before a chart reaches the interface."""

        required = {
            "what_it_shows": self.what_it_shows,
            "why_it_matters": self.why_it_matters,
            "conclusion": self.conclusion,
        }
        empty = [name for name, value in required.items() if not value.strip()]
        if empty:
            raise ValueError(
                "Chart explanation fields must not be empty: " + ", ".join(empty)
            )


def build_chart_explanation_html(explanation: ChartExplanation) -> str:
    """Return escaped HTML for the required chart interpretation panel."""

    explanation.validate()
    uncertainty_html = ""
    if explanation.uncertainty and explanation.uncertainty.strip():
        uncertainty_html = f"""
        <div>
          <strong>Uncertainty</strong>
          <p>{escape(explanation.uncertainty.strip())}</p>
        </div>
        """
    return f"""
    <section class="rm-panel rm-chart-explanation">
      <div>
        <strong>What this chart shows</strong>
        <p>{escape(explanation.what_it_shows.strip())}</p>
      </div>
      <div>
        <strong>Why it matters</strong>
        <p>{escape(explanation.why_it_matters.strip())}</p>
      </div>
      <div>
        <strong>Conclusion</strong>
        <p>{escape(explanation.conclusion.strip())}</p>
      </div>
      {uncertainty_html}
    </section>
    """.strip()


def render_chart_explanation(
    st: Any,
    explanation: ChartExplanation,
) -> None:
    """Render one explanation directly below its related chart."""

    st.markdown(
        build_chart_explanation_html(explanation),
        unsafe_allow_html=True,
    )
