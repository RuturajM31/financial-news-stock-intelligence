"""Show verified phrase evidence without inventing model explanations.

The sentiment endpoint does not return important phrases. This component can
highlight phrases only when a verified backend response supplies them. Until
then, it shows an honest empty state instead of selecting words with an
unapproved rule.
"""

from __future__ import annotations

import re
from html import escape
from typing import Any, Iterable, Sequence


MAXIMUM_PREVIEW_CHARACTERS = 2_000
MAXIMUM_PHRASES = 20


def normalize_verified_phrases(phrases: Iterable[str] | None) -> tuple[str, ...]:
    """Clean, deduplicate, and limit phrase evidence from a verified response."""

    if phrases is None:
        return ()
    normalized: list[str] = []
    seen: set[str] = set()
    for value in phrases:
        if not isinstance(value, str):
            raise ValueError("Verified phrases must contain text values only.")
        phrase = value.strip()
        if not phrase:
            continue
        key = phrase.casefold()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(phrase)
        if len(normalized) >= MAXIMUM_PHRASES:
            break
    return tuple(normalized)


def build_highlighted_text_html(
    text: str,
    phrases: Sequence[str],
) -> str:
    """Escape a bounded text preview and mark exact verified phrase matches.

    The function never discovers phrases. It only marks phrases supplied by a
    verified backend response. Long source text is shortened before rendering
    so the page remains readable and does not become a hidden storage surface.
    """

    if not isinstance(text, str):
        raise ValueError("The phrase preview must be text.")
    preview = text.strip()
    if not preview:
        raise ValueError("The phrase preview must not be empty.")
    preview = preview[:MAXIMUM_PREVIEW_CHARACTERS]
    safe_phrases = normalize_verified_phrases(phrases)
    if not safe_phrases:
        return f'<div class="rm-panel"><p>{escape(preview)}</p></div>'

    # Longer phrases are matched first so a short phrase does not split a more
    # useful longer phrase. The source and phrases are escaped after matching.
    ordered = sorted(safe_phrases, key=len, reverse=True)
    pattern = re.compile(
        "(" + "|".join(re.escape(value) for value in ordered) + ")",
        flags=re.IGNORECASE,
    )
    parts: list[str] = []
    cursor = 0
    for match in pattern.finditer(preview):
        parts.append(escape(preview[cursor : match.start()]))
        parts.append(
            '<mark title="Verified important phrase">'
            f"{escape(match.group(0))}</mark>"
        )
        cursor = match.end()
    parts.append(escape(preview[cursor:]))
    return '<div class="rm-panel rm-text-preview"><p>' + "".join(parts) + "</p></div>"


def render_phrase_evidence(
    st: Any,
    *,
    phrases: Sequence[str] | None,
    text_preview: str | None = None,
) -> None:
    """Render verified phrases or explain clearly why none are available."""

    safe_phrases = normalize_verified_phrases(phrases)
    st.markdown("### Important phrases")
    if not safe_phrases:
        st.info(
            "The current sentiment endpoint does not return important phrase "
            "evidence. Package 4 will show phrases supplied by the verified "
            "Historical Intelligence endpoint. No words are guessed here."
        )
        if text_preview and text_preview.strip():
            st.caption(
                "Submitted text preview. No words are highlighted because the "
                "backend did not return phrase evidence."
            )
            st.markdown(
                build_highlighted_text_html(text_preview, ()),
                unsafe_allow_html=True,
            )
        return

    st.caption(
        "Only phrases returned by the verified backend are highlighted."
    )
    if text_preview and text_preview.strip():
        st.markdown(
            build_highlighted_text_html(text_preview, safe_phrases),
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            " ".join(f"`{phrase}`" for phrase in safe_phrases)
        )
