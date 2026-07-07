"""Build reusable result cards with escaped text and consistent meaning."""

from __future__ import annotations

from html import escape
from typing import Any


_ALLOWED_TONES = frozenset({"positive", "neutral", "negative", "verified", "info"})


def build_result_card_html(
    title: str,
    value: str,
    detail: str,
    *,
    tone: str = "info",
    eyebrow: str | None = None,
) -> str:
    """Return safe HTML for one clear result or system-status card.

    The value is always accompanied by text. Color supports the meaning but is
    never the only way a user can understand the card.
    """

    if tone not in _ALLOWED_TONES:
        raise ValueError(f"Unsupported result-card tone: {tone!r}.")
    if not title.strip() or not value.strip() or not detail.strip():
        raise ValueError("Result-card title, value, and detail must not be empty.")

    eyebrow_html = ""
    if eyebrow and eyebrow.strip():
        eyebrow_html = (
            f'<p class="rm-eyebrow">{escape(eyebrow.strip())}</p>'
        )
    return f"""
    <section class="rm-panel rm-result-card rm-tone-{escape(tone)}">
      {eyebrow_html}
      <h3>{escape(title.strip())}</h3>
      <div class="rm-result-value">{escape(value.strip())}</div>
      <p>{escape(detail.strip())}</p>
    </section>
    """.strip()


def render_result_card(
    st: Any,
    title: str,
    value: str,
    detail: str,
    *,
    tone: str = "info",
    eyebrow: str | None = None,
) -> None:
    """Render one escaped result card in the current Streamlit container."""

    st.markdown(
        build_result_card_html(
            title,
            value,
            detail,
            tone=tone,
            eyebrow=eyebrow,
        ),
        unsafe_allow_html=True,
    )
