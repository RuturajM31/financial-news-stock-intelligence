"""Validate and render strictly earlier historical event matches."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from html import escape
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit


ALLOWED_SENTIMENTS = ("Bearish", "Neutral", "Bullish")
ALLOWED_MOVEMENTS = ("Down", "Flat", "Up")


def _require_text(mapping: Mapping[str, Any], key: str, location: str) -> str:
    """Read one required text field from a historical match."""

    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location}.{key} must be non-empty text.")
    return value.strip()




def _require_iso_date(
    mapping: Mapping[str, Any],
    key: str,
    location: str,
) -> str:
    """Read one ISO calendar date used by earlier-only checks."""

    value = _require_text(mapping, key, location)
    try:
        date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{location}.{key} must be an ISO calendar date.") from error
    return value


def _require_number(mapping: Mapping[str, Any], key: str, location: str) -> float:
    """Read one finite numeric field while rejecting booleans."""

    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location}.{key} must be a number.")
    result = float(value)
    if result != result or result in {float("inf"), float("-inf")}:
        raise ValueError(f"{location}.{key} must be finite.")
    return result


def _safe_source_label(url: str) -> str:
    """Return only the public host, never a full query or local path."""

    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Historical source URL must be a public HTTP or HTTPS URL.")
    return parsed.hostname.lower()


@dataclass(frozen=True)
class HistoricalMatchView:
    """Store one checked earlier event for cards, charts, and tables."""

    article_id: str
    ticker: str
    target_session_date: str
    source_host: str
    sentiment_label: str
    movement_label: str
    reaction_return_percent: float
    similarity_score: float


def parse_historical_matches(value: Any) -> tuple[HistoricalMatchView, ...]:
    """Validate the complete match list and return immutable view values."""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("historical response.matches must be a list.")
    matches: list[HistoricalMatchView] = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, Mapping):
            raise ValueError(f"historical match {index} must be a JSON object.")
        location = f"historical match {index}"
        sentiment = _require_text(item, "sentiment_label", location)
        movement = _require_text(item, "movement_label", location)
        if sentiment not in ALLOWED_SENTIMENTS:
            raise ValueError(f"{location}.sentiment_label is unsupported.")
        if movement not in ALLOWED_MOVEMENTS:
            raise ValueError(f"{location}.movement_label is unsupported.")
        similarity = _require_number(item, "similarity_score", location)
        if similarity < 0.0 or similarity > 1.0:
            raise ValueError(f"{location}.similarity_score must be between 0 and 1.")
        source_url = _require_text(item, "source_url", location)
        matches.append(
            HistoricalMatchView(
                article_id=_require_text(item, "article_id", location),
                ticker=_require_text(item, "ticker", location),
                target_session_date=_require_iso_date(
                    item,
                    "target_session_date",
                    location,
                ),
                source_host=_safe_source_label(source_url),
                sentiment_label=sentiment,
                movement_label=movement,
                reaction_return_percent=_require_number(
                    item,
                    "reaction_return_percent",
                    location,
                ),
                similarity_score=similarity,
            )
        )
    return tuple(matches)


def render_historical_event_cards(
    st: Any,
    matches: Sequence[HistoricalMatchView],
) -> None:
    """Render compact evidence cards without exposing restricted source content."""

    if not matches:
        st.info(
            "No earlier events met the selected similarity level. Lower the "
            "minimum similarity carefully or use a different verified event."
        )
        return

    columns = st.columns(2)
    for index, match in enumerate(matches):
        movement_class = match.movement_label.lower()
        return_sign = "+" if match.reaction_return_percent > 0 else ""
        with columns[index % 2]:
            st.markdown(
                f"""
                <article class="rm-history-card rm-history-{movement_class}">
                  <div class="rm-history-card-head">
                    <span>{escape(match.target_session_date)}</span>
                    <strong>{match.similarity_score * 100:.1f}% similar</strong>
                  </div>
                  <h3>{escape(match.ticker)} · {escape(match.movement_label)}</h3>
                  <div class="rm-history-return">
                    {return_sign}{match.reaction_return_percent:.2f}%
                  </div>
                  <div class="rm-history-meta">
                    <span>News tone · {escape(match.sentiment_label)}</span>
                    <span>Source · {escape(match.source_host)}</span>
                  </div>
                </article>
                """,
                unsafe_allow_html=True,
            )
