"""Collect safe Analyze-page input without storing raw content in session state.

Purpose
-------
The Analyze page accepts pasted text, one supported file, one public URL, or a
built-in example. This module validates the visible input before the FastAPI
client sends it to the verified backend.

Security boundary
-----------------
Validation here improves the user experience but does not replace FastAPI's
server-side checks. Raw text and file bytes are returned only to the current
page rerun and must never be copied into Streamlit session history.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


MAXIMUM_TEXT_CHARACTERS = 20_000
MAXIMUM_URL_CHARACTERS = 2_048
ALLOWED_UPLOAD_SUFFIXES = frozenset({".txt", ".pdf", ".docx", ".csv"})
EXAMPLE_NEWS_TEXT = (
    "Example Corporation reported stronger revenue and improved operating "
    "profit for the quarter. Management also warned that demand may slow in "
    "the next reporting period because of higher costs."
)


@dataclass(frozen=True)
class AnalysisSubmission:
    """Store one validated request for the current page rerun only.

    Attributes:
        source_kind: One of ``text``, ``file``, ``url``, or ``example``.
        display_name: Safe short label shown beside the result.
        text: Pasted or example text when the request uses text input.
        url: Public URL when the request uses URL input.
        filename: Base filename for an approved upload.
        file_bytes: Uploaded bytes for the current request only.
        csv_text_column: Optional CSV column name expected by FastAPI.

    The object must not be saved in session state because it can contain raw
    article text or uploaded file bytes.
    """

    source_kind: str
    display_name: str
    text: str | None = None
    url: str | None = None
    filename: str | None = None
    file_bytes: bytes | None = None
    csv_text_column: str | None = None


def normalize_text_input(value: str, *, source_name: str) -> str:
    """Trim one text value and enforce the FastAPI character boundary."""

    if not isinstance(value, str):
        raise ValueError(f"{source_name} must be text.")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{source_name} must not be empty.")
    if len(normalized) > MAXIMUM_TEXT_CHARACTERS:
        raise ValueError(
            f"{source_name} must contain no more than "
            f"{MAXIMUM_TEXT_CHARACTERS:,} characters."
        )
    return normalized


def validate_public_url(value: str) -> str:
    """Validate the visible URL shape before FastAPI performs network checks.

    This check accepts only HTTP or HTTPS, rejects embedded credentials, and
    requires a host name. FastAPI still blocks private networks, unsafe
    redirects, oversized responses, and unsupported content.
    """

    normalized = normalize_text_input(value, source_name="The public URL")
    if len(normalized) > MAXIMUM_URL_CHARACTERS:
        raise ValueError(
            f"The public URL must contain no more than {MAXIMUM_URL_CHARACTERS:,} "
            "characters."
        )
    parsed = urlsplit(normalized)
    if parsed.scheme.lower() not in {"http", "https"}:
        raise ValueError("The public URL must start with http:// or https://.")
    if not parsed.hostname:
        raise ValueError("The public URL must include a website name.")
    if parsed.username or parsed.password:
        raise ValueError("The public URL must not contain a username or password.")
    return normalized


def validate_upload(
    filename: str,
    content: bytes,
    *,
    maximum_bytes: int,
    csv_text_column: str | None = None,
) -> tuple[str, bytes, str | None]:
    """Validate one upload and return its safe base name and current bytes."""

    if not isinstance(filename, str) or not filename.strip():
        raise ValueError("The uploaded file must have a name.")
    safe_name = Path(filename).name
    if safe_name != filename:
        raise ValueError("The uploaded filename must not contain a folder path.")
    suffix = Path(safe_name).suffix.lower()
    if suffix not in ALLOWED_UPLOAD_SUFFIXES:
        raise ValueError("Only TXT, PDF, DOCX, and CSV files are supported.")
    if not isinstance(content, bytes) or not content:
        raise ValueError("The uploaded file must not be empty.")
    if len(content) > maximum_bytes:
        maximum_megabytes = maximum_bytes / (1024 * 1024)
        raise ValueError(
            f"The uploaded file is larger than the approved "
            f"{maximum_megabytes:.0f} MB limit."
        )

    normalized_column = None
    if suffix == ".csv" and csv_text_column:
        normalized_column = normalize_text_input(
            csv_text_column,
            source_name="The CSV text column",
        )
        if len(normalized_column) > 128:
            raise ValueError("The CSV text column name is too long.")
    return safe_name, content, normalized_column


def _render_text_tab(st: Any) -> AnalysisSubmission | None:
    """Render the pasted-text form and return a request only after submission."""

    with st.form("rm_text_analysis_form", clear_on_submit=False):
        text = st.text_area(
            "Financial news text",
            height=240,
            max_chars=MAXIMUM_TEXT_CHARACTERS,
            placeholder=(
                "Paste one financial news article, announcement, or short report."
            ),
            help=(
                "The text is sent to FastAPI for sentiment analysis. It is not "
                "saved in Streamlit session history."
            ),
        )
        st.caption(f"Maximum length: {MAXIMUM_TEXT_CHARACTERS:,} characters.")
        submitted = st.form_submit_button(
            "Analyze pasted text",
            type="primary",
            use_container_width=True,
        )
    if not submitted:
        return None
    normalized = normalize_text_input(text, source_name="The pasted text")
    return AnalysisSubmission(
        source_kind="text",
        display_name="Pasted text",
        text=normalized,
    )


def _render_file_tab(st: Any, maximum_upload_bytes: int) -> AnalysisSubmission | None:
    """Render the file form and return validated bytes only after submission."""

    with st.form("rm_file_analysis_form", clear_on_submit=False):
        upload = st.file_uploader(
            "Choose a financial news file",
            type=["txt", "pdf", "docx", "csv"],
            accept_multiple_files=False,
            help="Supported types: TXT, PDF, DOCX, and CSV.",
        )
        csv_column = st.text_input(
            "CSV text column",
            placeholder="Optional, for example: article_text",
            help=(
                "Leave this empty unless the CSV text is stored in a named column."
            ),
        )
        submitted = st.form_submit_button(
            "Analyze file",
            type="primary",
            use_container_width=True,
        )
    if not submitted:
        return None
    if upload is None:
        raise ValueError("Choose a supported file before starting the analysis.")
    content = upload.getvalue()
    safe_name, safe_content, safe_column = validate_upload(
        upload.name,
        content,
        maximum_bytes=maximum_upload_bytes,
        csv_text_column=csv_column or None,
    )
    return AnalysisSubmission(
        source_kind="file",
        display_name=safe_name,
        filename=safe_name,
        file_bytes=safe_content,
        csv_text_column=safe_column,
    )


def _render_url_tab(st: Any) -> AnalysisSubmission | None:
    """Render the public-URL form and return a validated address."""

    with st.form("rm_url_analysis_form", clear_on_submit=False):
        url = st.text_input(
            "Public article URL",
            placeholder="https://example.com/financial-news-article",
            help=(
                "FastAPI checks the address, redirects, response size, and network "
                "location before extracting text."
            ),
        )
        submitted = st.form_submit_button(
            "Analyze public URL",
            type="primary",
            use_container_width=True,
        )
    if not submitted:
        return None
    normalized = validate_public_url(url)
    hostname = urlsplit(normalized).hostname or "Public website"
    return AnalysisSubmission(
        source_kind="url",
        display_name=hostname,
        url=normalized,
    )


def _render_example_tab(st: Any) -> AnalysisSubmission | None:
    """Render a clearly labelled demonstration input for portfolio visitors."""

    st.info(
        "This is demonstration text created for the interface. It is not a real "
        "company announcement and must not be treated as market evidence."
    )
    st.code(EXAMPLE_NEWS_TEXT, language=None)
    if not st.button(
        "Analyze the demonstration text",
        key="rm_analyze_example",
        type="primary",
        use_container_width=True,
    ):
        return None
    return AnalysisSubmission(
        source_kind="example",
        display_name="Built-in demonstration",
        text=EXAMPLE_NEWS_TEXT,
    )


def render_analysis_inputs(
    st: Any,
    *,
    maximum_upload_bytes: int,
) -> AnalysisSubmission | None:
    """Render four input choices and return at most one validated submission.

    Validation errors intentionally propagate to the page controller. The page
    converts them into the standard four-part error panel so the user sees what
    failed, where it failed, why, and the exact safe next step.
    """

    text_tab, file_tab, url_tab, example_tab = st.tabs(
        ["Text", "File", "URL", "Example"]
    )
    with text_tab:
        text_submission = _render_text_tab(st)
    with file_tab:
        file_submission = _render_file_tab(st, maximum_upload_bytes)
    with url_tab:
        url_submission = _render_url_tab(st)
    with example_tab:
        example_submission = _render_example_tab(st)

    submissions = [
        item
        for item in (
            text_submission,
            file_submission,
            url_submission,
            example_submission,
        )
        if item is not None
    ]
    if len(submissions) > 1:
        raise ValueError("Only one analysis request may be submitted at a time.")
    return submissions[0] if submissions else None
