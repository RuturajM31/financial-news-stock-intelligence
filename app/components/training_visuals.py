"""Build premium training-result visuals with plain-language fallbacks.

All functions accept already validated evidence. Plotly specifications are built
without model imports. Tables remain available when a chart is difficult to read
or when the source evidence does not contain the required values.
"""

from __future__ import annotations

from typing import Any, Sequence

import plotly.graph_objects as go

from app.services.model_evidence import BenchmarkEvidenceView, ModelEvidenceView


def build_split_figure(evidence: BenchmarkEvidenceView) -> go.Figure:
    """Build a compact chart of training, validation, and test row counts."""

    labels = ["Training", "Validation", "Test"]
    values = [evidence.split_rows[label.lower()] for label in labels]
    figure = go.Figure(go.Bar(x=labels, y=values, text=values, textposition="outside"))
    figure.update_layout(title="How the benchmark data was divided", yaxis_title="Number of sentences", height=380)
    return figure


def build_test_balance_figure(evidence: BenchmarkEvidenceView) -> go.Figure:
    """Build a test-class balance chart from verified row counts."""

    labels = list(evidence.test_class_counts)
    values = [evidence.test_class_counts[label] for label in labels]
    figure = go.Figure(go.Bar(x=labels, y=values, text=values, textposition="outside"))
    figure.update_layout(title="Test data balance", yaxis_title="Number of sentences", height=380)
    return figure


def build_training_history_figure(model: ModelEvidenceView) -> go.Figure | None:
    """Build training and validation loss lines when history exists."""

    if not model.training_history:
        return None
    epochs = [row.epoch for row in model.training_history]
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=epochs,
            y=[row.average_training_loss for row in model.training_history],
            mode="lines+markers",
            name="Average training loss",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=epochs,
            y=[row.validation_loss for row in model.training_history],
            mode="lines+markers",
            name="Validation loss",
        )
    )
    figure.update_layout(
        title=f"{model.display_name} learning progress",
        xaxis_title="Training round",
        yaxis_title="Prediction error",
        height=430,
    )
    return figure


def build_validation_score_figure(model: ModelEvidenceView) -> go.Figure | None:
    """Build validation accuracy and macro F1 lines when history exists."""

    if not model.training_history:
        return None
    epochs = [row.epoch for row in model.training_history]
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=epochs,
            y=[100.0 * row.validation_accuracy for row in model.training_history],
            mode="lines+markers",
            name="Validation accuracy",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=epochs,
            y=[100.0 * row.validation_macro_f1 for row in model.training_history],
            mode="lines+markers",
            name="Validation macro F1",
        )
    )
    figure.update_layout(
        title=f"{model.display_name} validation scores",
        xaxis_title="Training round",
        yaxis_title="Score (%)",
        height=430,
    )
    return figure


def build_confusion_figure(model: ModelEvidenceView) -> go.Figure | None:
    """Build a labelled confusion matrix when verified counts exist."""

    if model.confusion_matrix is None:
        return None
    labels = ["Bearish", "Neutral", "Bullish"]
    figure = go.Figure(
        data=go.Heatmap(
            z=model.confusion_matrix,
            x=labels,
            y=labels,
            text=model.confusion_matrix,
            texttemplate="%{text}",
            hovertemplate=(
                "Actual: %{y}<br>Predicted: %{x}<br>"
                "Sentences: %{z}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        title=f"{model.display_name} — correct and incorrect predictions",
        xaxis_title="Model prediction",
        yaxis_title="Correct label",
        height=500,
    )
    return figure


def confusion_table(model: ModelEvidenceView) -> list[dict[str, Any]]:
    """Return the accessible table behind one confusion matrix."""

    if model.confusion_matrix is None:
        return []
    labels = ("Bearish", "Neutral", "Bullish")
    rows: list[dict[str, Any]] = []
    for row_index, actual in enumerate(labels):
        row: dict[str, Any] = {"Correct label": actual}
        for column_index, predicted in enumerate(labels):
            row[f"Predicted {predicted}"] = model.confusion_matrix[row_index][column_index]
        rows.append(row)
    return rows


def class_result_rows(model: ModelEvidenceView) -> list[dict[str, Any]]:
    """Return plain class-level result rows or an empty list."""

    if model.per_class is None:
        return []
    return [
        {
            "Class": row.label,
            "Precision": round(100.0 * row.precision, 2),
            "Recall": round(100.0 * row.recall, 2),
            "F1": round(100.0 * row.f1, 2),
            "Test sentences": row.support,
        }
        for row in model.per_class
    ]


def error_summary_rows(model: ModelEvidenceView) -> list[dict[str, Any]]:
    """Summarize the largest verified confusion counts without inventing text examples."""

    if model.confusion_matrix is None:
        return []
    labels = ("Bearish", "Neutral", "Bullish")
    rows: list[dict[str, Any]] = []
    for actual_index, actual in enumerate(labels):
        for predicted_index, predicted in enumerate(labels):
            if actual_index == predicted_index:
                continue
            rows.append(
                {
                    "Correct label": actual,
                    "Predicted label": predicted,
                    "Mistakes": model.confusion_matrix[actual_index][predicted_index],
                }
            )
    return sorted(rows, key=lambda row: int(row["Mistakes"]), reverse=True)


def render_chart_explanation(st: Any, *, what: str, why: str, conclusion: str, limitation: str | None = None) -> None:
    """Render the standard simple explanation under a chart."""

    st.markdown(f"**What this shows:** {what}")
    st.markdown(f"**Why it matters:** {why}")
    st.markdown(f"**Conclusion:** {conclusion}")
    if limitation:
        st.caption(f"Limit: {limitation}")
