"""Render movement explainability, word influence, and honest model limits."""

from __future__ import annotations

from typing import Any

from app.components.attention_explorer import render_attention_explorer_status
from app.components.charts_3d import render_driver_landscape
from app.components.driver_charts import render_driver_charts
from app.components.explainability_inputs import render_explainability_inputs
from app.components.status_badges import render_problem
from app.components.movement_results import (
    render_movement_hero,
    render_probability_visual,
)
from app.components.word_influence import render_word_influence_result
from app.services.api_client import FinancialNewsApiClient, StreamlitApiError
from app.services.explanation_service import (
    MAXIMUM_INFLUENCE_WORDS,
    compute_word_influence,
    parse_explainability_response,
)


def _render_word_influence_tool(
    st: Any,
    api_client: FinancialNewsApiClient,
) -> None:
    """Render a separate short-text action to protect rate and clarity limits."""

    st.markdown("## Word influence")
    st.write(
        "Use a short sentence to see which words changed the live DistilBERT "
        "sentiment result. This is not raw attention."
    )
    with st.form("rm_word_influence_form", clear_on_submit=False):
        short_text = st.text_area(
            "Short sentence",
            height=110,
            max_chars=500,
            placeholder="Profits rose after strong customer demand",
            help=(
                f"Use 3 to {MAXIMUM_INFLUENCE_WORDS} words. The check sends one "
                "baseline request and one request for each word."
            ),
        )
        submitted = st.form_submit_button(
            "Measure word influence",
            use_container_width=True,
        )
    if not submitted:
        return

    try:
        with st.status("Measuring the effect of each word...", expanded=True):
            st.write("Running the original sentence.")
            st.write("Removing one word at a time.")
            st.write("Turning the measured changes into percentages.")
            result = compute_word_influence(api_client, short_text)
        render_word_influence_result(st, result)
    except StreamlitApiError as error:
        render_problem(st, error.problem)
    except ValueError as error:
        st.error(str(error))


def render_explainability_page(
    st: Any,
    api_client: FinancialNewsApiClient,
) -> None:
    """Render the complete Package 5 explanation experience."""

    st.markdown(
        """
        <section class="rm-panel">
          <p class="rm-panel-kicker">MODEL EXPLANATION</p>
          <h2>Why did the forecast choose this result?</h2>
          <p>
            Compare factors that usually matter with factors that affected this
            forecast. Every chart includes a plain conclusion and a clear limit.
          </p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    submission = render_explainability_inputs(st)
    if submission is None:
        st.info(
            "Add the news text, ticker, and publication time, then choose "
            "Explain this forecast."
        )
        _render_word_influence_tool(st, api_client)
        return

    try:
        with st.status("Building the forecast explanation...", expanded=True):
            st.write("Checking the movement result.")
            st.write("Reading general model drivers.")
            st.write("Reading the factors for this result.")
            payload = api_client.explainability(
                submission.text,
                submission.ticker,
                submission.published_at,
                submission.top_n,
            )
            view = parse_explainability_response(payload)
    except StreamlitApiError as error:
        render_problem(st, error.problem)
        _render_word_influence_tool(st, api_client)
        return
    except ValueError as error:
        st.error(
            "The explanation response could not be displayed because its "
            f"verified fields were incomplete: {error}"
        )
        _render_word_influence_tool(st, api_client)
        return

    render_movement_hero(st, view.prediction)
    render_probability_visual(st, view.prediction)
    st.markdown("## Driver evidence")
    render_driver_charts(st, view.global_drivers, view.local_drivers)
    render_driver_landscape(st, view.global_drivers, view.local_drivers)
    st.markdown("## Limits")
    st.warning(view.limitation)
    st.caption(f"Reference data used by the API: {view.reference_scope}")
    render_attention_explorer_status(st, payload)
    _render_word_influence_tool(st, api_client)
