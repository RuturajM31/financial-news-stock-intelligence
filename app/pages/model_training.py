"""Render the portfolio Model Training & Results page."""

from __future__ import annotations

from typing import Any

from app.components.training_visuals import (
    build_confusion_figure,
    build_split_figure,
    build_test_balance_figure,
    build_training_history_figure,
    build_validation_score_figure,
    class_result_rows,
    confusion_table,
    error_summary_rows,
    render_chart_explanation,
)
from app.services.model_evidence import load_benchmark_evidence


def _format_seconds(value: float | None) -> str:
    """Return a plain duration or an honest unavailable label."""

    if value is None:
        return "Not in verified evidence"
    minutes = value / 60.0
    return f"{minutes:.1f} minutes"


def render_model_training_page(st: Any) -> None:
    """Render verified training data, learning progress, results, and gaps."""

    evidence = load_benchmark_evidence()
    st.markdown("## Model Training & Results")
    st.write(
        "This page shows what was trained, how it improved, where it made "
        "mistakes, and what evidence is still missing."
    )

    summary = st.columns(4)
    summary[0].metric("Training sentences", f"{evidence.split_rows['training']:,}")
    summary[1].metric("Validation sentences", f"{evidence.split_rows['validation']:,}")
    summary[2].metric("Test sentences", f"{evidence.split_rows['test']:,}")
    summary[3].metric("Models compared", f"{len(evidence.models)}")

    left, right = st.columns(2)
    left.plotly_chart(build_split_figure(evidence), use_container_width=True, config={"displaylogo": False})
    right.plotly_chart(build_test_balance_figure(evidence), use_container_width=True, config={"displaylogo": False})
    render_chart_explanation(
        st,
        what=(
            "The first chart shows how many sentences were used for training, "
            "checking, and final testing. The second shows the labels in the "
            "untouched test data."
        ),
        why="A fair test must be separate from training, and class balance affects how easy accuracy is to interpret.",
        conclusion=(
            "All models were compared on the same 518 test sentences. Neutral "
            "sentences were the largest group, so macro F1 is important because "
            "it gives every class equal weight."
        ),
    )

    st.markdown("### Learning progress")
    selected_name = st.selectbox(
        "Choose a model",
        [model.display_name for model in evidence.models],
        key="rm_training_model",
    )
    model = next(item for item in evidence.models if item.display_name == selected_name)
    history_chart = build_training_history_figure(model)
    score_chart = build_validation_score_figure(model)
    if history_chart is None or score_chart is None:
        st.info(
            "The verified comparison summary did not contain this model's "
            "round-by-round training history. The app will not invent a curve."
        )
    else:
        first, second = st.columns(2)
        first.plotly_chart(history_chart, use_container_width=True, config={"displaylogo": False})
        second.plotly_chart(score_chart, use_container_width=True, config={"displaylogo": False})
        best = max(model.training_history or (), key=lambda row: row.validation_macro_f1)
        render_chart_explanation(
            st,
            what="Prediction error fell during training while validation scores improved across the three rounds.",
            why="Validation results show whether learning also works on sentences the model did not train on.",
            conclusion=(
                f"{model.display_name} reached its strongest recorded validation "
                f"macro F1 in round {best.epoch}. The saved test result must still "
                "be used for the final comparison."
            ),
            limitation="The displayed training loss is the average of the verified log points within each round.",
        )

    st.markdown("### Final test result")
    metrics = st.columns(4)
    metrics[0].metric("Accuracy", f"{100.0 * model.accuracy:.2f}%")
    metrics[1].metric("Macro F1", f"{100.0 * model.macro_f1:.2f}%")
    metrics[2].metric("Weighted F1", f"{100.0 * model.weighted_f1:.2f}%")
    metrics[3].metric("Training time", _format_seconds(model.training_seconds))
    st.caption(model.selection_reason)

    confusion = build_confusion_figure(model)
    if confusion is None:
        st.info("The verified summary did not contain this model's confusion matrix or class-by-class scores.")
    else:
        st.plotly_chart(confusion, use_container_width=True, config={"displaylogo": False})
        with st.expander("Open the accessible count table"):
            st.dataframe(confusion_table(model), use_container_width=True, hide_index=True)
        total_predictions = sum(sum(row) for row in model.confusion_matrix)
        correct_predictions = sum(
            model.confusion_matrix[index][index] for index in range(3)
        )
        mistake_count = total_predictions - correct_predictions
        render_chart_explanation(
            st,
            what=(
                "Numbers on the main diagonal are correct predictions. Other "
                "cells show which labels were confused."
            ),
            why=(
                "Two models can have similar accuracy but make different kinds "
                "of mistakes."
            ),
            conclusion=(
                f"{model.display_name} made {mistake_count} mistakes across 518 "
                "test sentences."
            ),
        )
        st.markdown("#### Result by class")
        st.dataframe(class_result_rows(model), use_container_width=True, hide_index=True)
        st.markdown("#### Largest error groups")
        st.dataframe(error_summary_rows(model), use_container_width=True, hide_index=True)
        st.markdown("#### Correct and incorrect sentence examples")
        st.info(
            "The benchmark summary did not include the original sentence text. "
            "This page will not invent correct or incorrect examples."
        )

    st.markdown("### Confidence reliability")
    st.info(
        "A confidence-reliability chart needs the saved probability for every "
        "test prediction. Those values were not included in the verified "
        "benchmark summary, so no reliability curve is shown."
    )

    st.markdown("### Reproducibility and limits")
    details = {
        "Benchmark": evidence.benchmark_name,
        "Completed": evidence.completed_date,
        "Model revision": evidence.model_revision,
        "Test rows": evidence.split_rows["test"],
        "Post-training transformer tests": evidence.verification.get("post_training_transformer_tests"),
        "Post-training project tests": evidence.verification.get("post_training_project_tests"),
        "Deployment changed during benchmark": evidence.verification.get("application_deployment_changed"),
    }
    st.dataframe([details], use_container_width=True, hide_index=True)
    with st.expander("Open evidence that was not included in the benchmark summary"):
        for title, message in evidence.evidence_gaps.items():
            readable_title = title.replace("_", " ").title()
            st.markdown(f"**{readable_title}**")
            st.write(message)
