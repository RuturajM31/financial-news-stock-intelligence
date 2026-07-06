"""Verify current-session storage stays small, temporary, and licence-safe."""

from __future__ import annotations

import pytest

from tests.streamlit_test_support import import_project_module, source_text


def test_session_initialization_creates_only_approved_keys() -> None:
    """Create the connection check and analysis history without extra state."""

    module = import_project_module("app.services.session_state")
    state: dict[str, object] = {}
    module.initialize_session_state(state)
    assert set(state) == {module.API_CHECK_KEY, module.ANALYSIS_HISTORY_KEY}


def test_analysis_history_rejects_raw_text() -> None:
    """Prevent complete article text from entering session history."""

    module = import_project_module("app.services.session_state")
    state: dict[str, object] = {}
    with pytest.raises(ValueError, match="raw_text"):
        module.add_analysis_summary(state, {"raw_text": "private article"})


def test_analysis_history_rejects_file_bytes_and_secrets() -> None:
    """Prevent uploaded data and authentication values from being retained."""

    module = import_project_module("app.services.session_state")
    for key in ("file_bytes", "api_key", "token", "secret"):
        state: dict[str, object] = {}
        with pytest.raises(ValueError, match=key):
            module.add_analysis_summary(state, {key: "blocked"})


def test_analysis_history_keeps_only_the_latest_twenty_records() -> None:
    """Bound temporary browser state so it cannot grow without limit."""

    module = import_project_module("app.services.session_state")
    state: dict[str, object] = {}
    for index in range(25):
        module.add_analysis_summary(state, {"request": index})
    history = state[module.ANALYSIS_HISTORY_KEY]
    assert isinstance(history, list)
    assert len(history) == module.MAXIMUM_SESSION_HISTORY
    assert history[0]["request"] == 5


def test_clear_history_does_not_remove_other_session_values() -> None:
    """Remove only analysis summaries and keep unrelated widget state."""

    module = import_project_module("app.services.session_state")
    state: dict[str, object] = {"other_widget": "keep"}
    module.add_analysis_summary(state, {"ticker": "AAPL"})
    module.clear_analysis_history(state)
    assert state["other_widget"] == "keep"
    assert state[module.ANALYSIS_HISTORY_KEY] == []


def test_analyze_page_stores_a_summary_instead_of_submitted_text() -> None:
    """Require the analysis page to pass only small result fields to history."""

    source = source_text("app/pages/analyze.py")
    assert "add_analysis_summary" in source
    assert '"raw_text"' not in source
    assert '"file_bytes"' not in source


def test_forecast_page_does_not_store_the_submitted_article() -> None:
    """Keep forecast history free from the original financial-news text."""

    source = source_text("app/pages/forecasts.py")
    assert "_store_safe_forecast" in source
    assert '"text": submission.text' not in source
