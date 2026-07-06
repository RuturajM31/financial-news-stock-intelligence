"""Render the live sentiment-analysis workflow over the verified FastAPI API.

The page accepts one safe input, calls the matching FastAPI route, validates the
response, stores only a small result summary, and explains the conclusion in
plain language. Model inference never runs inside Streamlit.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from app.components.analysis_inputs import (
    AnalysisSubmission,
    render_analysis_inputs,
)
from app.components.loading_states import loading_context
from app.components.phrase_highlights import render_phrase_evidence
from app.components.sentiment_results import (
    SentimentView,
    parse_sentiment_response,
    render_sentiment_result,
)
from app.components.source_verification import (
    SourceEvidence,
    build_source_evidence,
    render_source_evidence,
)
from app.components.status_badges import render_problem
from app.services.api_client import FinancialNewsApiClient, StreamlitApiError
from app.services.api_contracts import ApiProblemDetails
from app.services.session_state import add_analysis_summary


CURRENT_ANALYSIS_KEY = "rm_current_sentiment_analysis"


def _run_sentiment_request(
    client: FinancialNewsApiClient,
    submission: AnalysisSubmission,
) -> Mapping[str, Any]:
    """Send one validated request to its exact FastAPI sentiment route."""

    if submission.source_kind in {"text", "example"}:
        if submission.text is None:
            raise ValueError("The text request is missing its validated text.")
        return client.sentiment_text(submission.text)
    if submission.source_kind == "url":
        if submission.url is None:
            raise ValueError("The URL request is missing its validated address.")
        return client.sentiment_url(submission.url)
    if submission.source_kind == "file":
        if submission.filename is None or submission.file_bytes is None:
            raise ValueError("The file request is missing its validated upload.")
        return client.sentiment_file(
            submission.filename,
            submission.file_bytes,
            submission.csv_text_column,
        )
    raise ValueError(f"Unsupported analysis source: {submission.source_kind!r}.")


def _build_source_for_submission(submission: AnalysisSubmission) -> SourceEvidence:
    """Describe only the source-processing evidence proved by a passed request."""

    return build_source_evidence(
        source_kind=submission.source_kind,
        display_name=submission.display_name,
        public_url=submission.url,
    )


def _store_current_analysis(
    state: Any,
    *,
    views: Sequence[SentimentView],
    source: SourceEvidence,
) -> None:
    """Store only checked result values and safe source metadata for this session."""

    state[CURRENT_ANALYSIS_KEY] = {
        "views": [view.as_dict() for view in views],
        "source": source.as_dict(),
    }


def _read_current_analysis(
    state: Any,
) -> tuple[tuple[SentimentView, ...], SourceEvidence] | None:
    """Rebuild the current safe result or reject changed session data."""

    saved = state.get(CURRENT_ANALYSIS_KEY)
    if saved is None:
        return None
    if not isinstance(saved, Mapping):
        raise ValueError("The saved sentiment analysis has an invalid shape.")
    views_raw = saved.get("views")
    source_raw = saved.get("source")
    if not isinstance(views_raw, list) or not views_raw:
        raise ValueError("The saved sentiment result list is missing or empty.")
    if not isinstance(source_raw, dict):
        raise ValueError("The saved source evidence is missing.")
    views = tuple(
        SentimentView.from_dict(value)
        for value in views_raw
        if isinstance(value, Mapping)
    )
    if len(views) != len(views_raw):
        raise ValueError("A saved sentiment result has an invalid shape.")
    return views, SourceEvidence.from_dict(source_raw)


def _clear_current_analysis(state: Any) -> None:
    """Remove only the current safe result and leave session history unchanged."""

    state.pop(CURRENT_ANALYSIS_KEY, None)


def _render_input_error(st: Any, reason: str) -> None:
    """Convert local validation failures into the standard four-part message."""

    render_problem(
        st,
        ApiProblemDetails(
            error_code="streamlit_analysis_input_invalid",
            what_failed="The analysis did not start.",
            where_failed="Analyze News input",
            why_failed=reason,
            safe_next_step=(
                "Correct the input described above and submit the request again."
            ),
        ),
    )


def _render_response_error(st: Any, reason: str) -> None:
    """Explain a response-contract failure without displaying raw API content."""

    render_problem(
        st,
        ApiProblemDetails(
            error_code="streamlit_sentiment_response_invalid",
            what_failed="The sentiment result could not be displayed.",
            where_failed="Streamlit sentiment response validation",
            why_failed=reason,
            safe_next_step=(
                "Restore the verified FastAPI sentiment response and retry once."
            ),
        ),
    )


def _record_history(
    state: Any,
    *,
    views: Sequence[SentimentView],
    source: SourceEvidence,
) -> None:
    """Add one small summary without raw content or source addresses."""

    leading = views[0]
    add_analysis_summary(
        state,
        {
            "label": leading.label,
            "confidence_percent": leading.confidence * 100,
            "source_kind": source.source_kind,
            "result_count": len(views),
            "deployment_model": leading.deployment_model,
        },
    )


def _render_saved_result(
    st: Any,
    *,
    views: Sequence[SentimentView],
    source: SourceEvidence,
    text_preview: str | None,
) -> None:
    """Render source evidence, selected sentiment, phrases, and next step."""

    st.markdown(
        """
        <section class="rm-section-heading rm-section-gap">
          <p class="rm-eyebrow">ANALYSIS RESULT</p>
          <h2>What the submitted wording means</h2>
          <p>
            The values below came from the verified DistilBERT service through
            FastAPI. They describe the wording, not a guaranteed price move.
          </p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    render_source_evidence(st, source)
    render_sentiment_result(st, views)
    render_phrase_evidence(
        st,
        phrases=None,
        text_preview=text_preview,
    )
    st.info(
        "Next step: Package 4 will connect this result to the verified Down, "
        "Flat, and Up movement forecast and earlier-event evidence."
    )


def render_analyze_page(st: Any, client: FinancialNewsApiClient) -> None:
    """Render the complete Package 3 Analyze page and current safe result."""

    st.markdown(
        """
        <section class="rm-section-heading">
          <p class="rm-eyebrow">ANALYZE FINANCIAL NEWS</p>
          <h2>Add one source and receive a clear sentiment result</h2>
          <p>
            Choose pasted text, a supported file, a public URL, or the labelled
            example. FastAPI performs the extraction and DistilBERT analysis.
          </p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "The result is for educational and research use. It is not investment advice."
    )

    text_preview: str | None = None
    try:
        submission = render_analysis_inputs(
            st,
            maximum_upload_bytes=client.settings.maximum_upload_bytes,
        )
    except ValueError as error:
        _render_input_error(st, str(error))
        submission = None

    if submission is not None:
        # Remove any older result before a new request starts. If the new request
        # fails, the page must not leave a previous result under the new error.
        _clear_current_analysis(st.session_state)
        try:
            with loading_context(st, "sentiment"):
                payload = _run_sentiment_request(client, submission)
            views = parse_sentiment_response(payload)
            source = _build_source_for_submission(submission)
            _store_current_analysis(st.session_state, views=views, source=source)
            _record_history(st.session_state, views=views, source=source)
            if submission.source_kind in {"text", "example"}:
                text_preview = submission.text
        except StreamlitApiError as error:
            render_problem(st, error.problem)
        except ValueError as error:
            _render_response_error(st, str(error))

    try:
        current = _read_current_analysis(st.session_state)
    except ValueError as error:
        _clear_current_analysis(st.session_state)
        _render_response_error(st, str(error))
        current = None

    if current is None:
        st.info(
            "Choose one input option above. The sentiment result will appear here "
            "after FastAPI returns a verified response."
        )
        return

    views, source = current
    clear_result = st.button(
        "Clear current result",
        key="rm_clear_current_sentiment_result",
        type="secondary",
    )
    if clear_result:
        _clear_current_analysis(st.session_state)
        st.rerun()

    _render_saved_result(
        st,
        views=views,
        source=source,
        text_preview=text_preview,
    )
