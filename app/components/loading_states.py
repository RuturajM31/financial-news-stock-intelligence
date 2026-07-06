"""Provide consistent plain-language loading messages for Streamlit pages."""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LoadingMessage:
    """Describe one visible step without technical jargon."""

    key: str
    title: str
    detail: str


_LOADING_MESSAGES = {
    "check_services": LoadingMessage(
        key="check_services",
        title="Checking the live services...",
        detail="Confirming that FastAPI and its verified workers are ready.",
    ),
    "check_input": LoadingMessage(
        key="check_input",
        title="Checking the input...",
        detail="Making sure the submitted content is safe and complete.",
    ),
    "extract_text": LoadingMessage(
        key="extract_text",
        title="Extracting the text...",
        detail="Reading the useful article content from the selected source.",
    ),
    "sentiment": LoadingMessage(
        key="sentiment",
        title="Running sentiment analysis...",
        detail="Measuring positive, neutral, and negative meaning.",
    ),
    "forecast": LoadingMessage(
        key="forecast",
        title="Generating the market forecast...",
        detail="Calculating Down, Flat, and Up chances for the mapped session.",
    ),
    "history": LoadingMessage(
        key="history",
        title="Finding similar earlier events...",
        detail="Searching only events that happened before the current case.",
    ),
    "explanation": LoadingMessage(
        key="explanation",
        title="Preparing the explanation...",
        detail="Turning model drivers into clear, practical language.",
    ),
    "report": LoadingMessage(
        key="report",
        title="Building the report...",
        detail="Preparing licence-safe results and supporting evidence.",
    ),
}


def get_loading_message(key: str) -> LoadingMessage:
    """Return one approved loading message or fail for an unknown step."""

    try:
        return _LOADING_MESSAGES[key]
    except KeyError as error:
        raise ValueError(f"Unknown loading step: {key!r}.") from error


def loading_context(st: Any, key: str) -> AbstractContextManager[Any]:
    """Create a Streamlit spinner using the approved short message."""

    message = get_loading_message(key)
    return st.spinner(message.title)
