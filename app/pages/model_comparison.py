"""Render the verified BERT, DistilBERT, and BERT LoRA comparison page."""

from __future__ import annotations

from typing import Any

from app.components.model_comparison_charts import (
    build_3d_tradeoff_figure,
    build_efficiency_figure,
    build_parameter_figure,
    build_quality_figure,
    comparison_rows,
)
from app.components.sentiment_results import parse_sentiment_response, render_sentiment_result
from app.components.training_visuals import render_chart_explanation
from app.components.status_badges import render_problem
from app.services.api_client import FinancialNewsApiClient, StreamlitApiError
from app.services.model_evidence import load_benchmark_evidence

# FastAPI currently exposes DistilBERT only.
# The page will not pretend to run them live.
# Streamlit will not load those models directly.


def render_model_comparison_page(st: Any, api_client: FinancialNewsApiClient) -> None:
    """Render verified benchmark comparisons and one honest live-model check."""

    evidence = load_benchmark_evidence()
    st.markdown("## Model Comparison")
    st.write(
        "BERT, DistilBERT, and BERT LoRA were tested on the same 518 "
        "financial sentences. The page separates best quality from the best "
        "live-app choice."
    )

    top = st.columns(3)
    top[0].metric("Best prediction quality", evidence.model(evidence.quality_champion).display_name)
    top[1].metric("Live application choice", evidence.model(evidence.deployment_champion).display_name)
    top[2].metric("Models tested", f"{len(evidence.models)}")

    st.dataframe(comparison_rows(evidence), use_container_width=True, hide_index=True)
    st.plotly_chart(build_quality_figure(evidence), use_container_width=True, config={"displaylogo": False})
    render_chart_explanation(
        st,
        what="The bars compare accuracy, macro F1, and weighted F1 on exactly the same test sentences.",
        why="Accuracy alone can hide weak performance on smaller classes, so the F1 scores are also required.",
        conclusion=(
            "BERT had the strongest quality. DistilBERT stayed close while using "
            "less memory and responding faster."
        ),
    )

    first, second = st.columns(2)
    first.plotly_chart(build_efficiency_figure(evidence), use_container_width=True, config={"displaylogo": False})
    second.plotly_chart(build_parameter_figure(evidence), use_container_width=True, config={"displaylogo": False})
    render_chart_explanation(
        st,
        what=(
            "The left chart compares response time with quality. The right chart "
            "shows how many parameters changed during training."
        ),
        why="A live product needs both good predictions and acceptable response cost.",
        conclusion=(
            "DistilBERT is the deployment champion. BERT LoRA changed far fewer "
            "parameters, but it did not give the best quality or response time "
            "in this run."
        ),
    )

    st.markdown("### Interactive 3D trade-off")
    st.plotly_chart(
        build_3d_tradeoff_figure(evidence),
        use_container_width=True,
        config={"displaylogo": False, "scrollZoom": True},
    )
    render_chart_explanation(
        st,
        what="Move or rotate the chart to compare response time, memory use, and weighted F1 together.",
        why="No single number decides the live model. The best choice must balance all three measures.",
        conclusion=(
            "DistilBERT sits in the strongest live-product position: lower "
            "response time, lower memory, and quality close to BERT."
        ),
        limitation="Use the table above for exact values because 3D perspective can make distances look different.",
    )
    with st.expander("Open the accessible 2D fallback"):
        st.dataframe(comparison_rows(evidence), use_container_width=True, hide_index=True)

    st.markdown("### Why each model was or was not selected")
    for model in evidence.models:
        with st.container(border=True):
            st.markdown(f"#### {model.display_name} — {model.role}")
            st.write(model.selection_reason)
            st.caption(model.limitation)

    st.markdown("### Try the live deployment model")
    st.write(
        "The live API currently exposes DistilBERT only. BERT and BERT LoRA "
        "remain verified benchmark models, so the app will not pretend to run "
        "them live."
    )
    text = st.text_area(
        "Financial sentence",
        max_chars=4000,
        placeholder=(
            "Example: The company raised its full-year profit forecast after "
            "stronger demand."
        ),
        key="rm_model_comparison_text",
    )
    if st.button("Run the live DistilBERT check", type="primary", key="rm_model_comparison_run"):
        if not text.strip():
            st.warning("Add a financial sentence before running the live check.")
        else:
            try:
                response = api_client.sentiment_text(text.strip())
                render_sentiment_result(st, parse_sentiment_response(response))
            except StreamlitApiError as error:
                render_problem(st, error.problem)
    st.info(
        "A true same-text three-model live comparison requires separate "
        "protected BERT and BERT LoRA FastAPI routes. Streamlit will not load "
        "those models directly."
    )
