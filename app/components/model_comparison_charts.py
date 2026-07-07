"""Build verified 2D and 3D model-comparison visuals.

The 3D chart explains the quality, speed, and memory trade-off. A complete 2D
table is always provided because 3D perspective can make exact comparison hard.
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go

from app.services.model_evidence import BenchmarkEvidenceView


def comparison_rows(evidence: BenchmarkEvidenceView) -> list[dict[str, Any]]:
    """Return the exact benchmark leaderboard used by all comparison views."""

    return [
        {
            "Model": model.display_name,
            "Role": model.role,
            "Accuracy (%)": round(100.0 * model.accuracy, 3),
            "Macro F1 (%)": round(100.0 * model.macro_f1, 3),
            "Weighted F1 (%)": round(100.0 * model.weighted_f1, 3),
            "Response time (ms)": round(model.latency_ms_per_record, 3),
            "Memory (MiB)": round(model.memory_mib, 3),
            "Trainable parameters": model.trainable_parameters,
            "Artifact size": (
                "Not in verified evidence"
                if model.artifact_size_bytes is None
                else model.artifact_size_bytes
            ),
        }
        for model in evidence.models
    ]


def build_quality_figure(evidence: BenchmarkEvidenceView) -> go.Figure:
    """Build grouped bars for the three main quality scores."""

    names = [model.display_name for model in evidence.models]
    figure = go.Figure()
    for field, label in (("accuracy", "Accuracy"), ("macro_f1", "Macro F1"), ("weighted_f1", "Weighted F1")):
        figure.add_trace(go.Bar(name=label, x=names, y=[100.0 * getattr(model, field) for model in evidence.models]))
    figure.update_layout(
        title="Prediction quality on the same test data",
        yaxis_title="Score (%)",
        barmode="group",
        height=460,
    )
    return figure


def build_efficiency_figure(evidence: BenchmarkEvidenceView) -> go.Figure:
    """Build a 2D quality-versus-speed chart with memory in hover text."""

    figure = go.Figure(go.Scatter(
        x=[model.latency_ms_per_record for model in evidence.models],
        y=[100.0 * model.weighted_f1 for model in evidence.models],
        mode="markers+text",
        text=[model.display_name for model in evidence.models],
        textposition="top center",
        marker={"size": [max(14, min(42, model.memory_mib / 12.0)) for model in evidence.models]},
        customdata=[[model.memory_mib, model.trainable_parameters] for model in evidence.models],
        hovertemplate=(
            "%{text}<br>Response time: %{x:.2f} ms<br>"
            "Weighted F1: %{y:.2f}%<br>"
            "Memory: %{customdata[0]:.1f} MiB<br>"
            "Trainable parameters: %{customdata[1]:,}<extra></extra>"
        ),
    ))
    figure.update_layout(
        title="Quality compared with response time",
        xaxis_title="Response time per sentence (ms)",
        yaxis_title="Weighted F1 (%)",
        height=460,
    )
    return figure


def build_3d_tradeoff_figure(evidence: BenchmarkEvidenceView) -> go.Figure:
    """Build the approved 3D quality, speed, and memory trade-off view."""

    figure = go.Figure(go.Scatter3d(
        x=[model.latency_ms_per_record for model in evidence.models],
        y=[model.memory_mib for model in evidence.models],
        z=[100.0 * model.weighted_f1 for model in evidence.models],
        mode="markers+text",
        text=[model.display_name for model in evidence.models],
        marker={"size": [14, 12, 10]},
        customdata=[[model.role, model.trainable_parameters] for model in evidence.models],
        hovertemplate=(
            "%{text}<br>%{customdata[0]}<br>"
            "Response time: %{x:.2f} ms<br>"
            "Memory: %{y:.1f} MiB<br>"
            "Weighted F1: %{z:.2f}%<br>"
            "Trainable parameters: %{customdata[1]:,}<extra></extra>"
        ),
    ))
    figure.update_layout(
        title="3D model trade-off",
        scene={
            "xaxis": {"title": "Response time (ms)"},
            "yaxis": {"title": "Memory (MiB)"},
            "zaxis": {"title": "Weighted F1 (%)"},
            "camera": {"eye": {"x": 1.45, "y": 1.45, "z": 1.15}},
        },
        height=650,
    )
    return figure


def build_parameter_figure(evidence: BenchmarkEvidenceView) -> go.Figure:
    """Build a logarithmic parameter chart so LoRA remains visible."""

    figure = go.Figure(
        go.Bar(
            x=[model.display_name for model in evidence.models],
            y=[model.trainable_parameters for model in evidence.models],
            text=[f"{model.trainable_parameters:,}" for model in evidence.models],
            textposition="outside",
        )
    )
    figure.update_layout(
        title="Parameters changed during training",
        yaxis_title="Trainable parameters (log scale)",
        yaxis_type="log",
        height=450,
    )
    return figure
