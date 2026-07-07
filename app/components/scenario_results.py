"""Validate and render research-only scenario results from FastAPI.

The module repeats the small mathematical and date checks required by the user
interface. It never changes the backend outcome, estimates missing values, or
turns historical ranges into a promise of future performance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from html import escape
from typing import Any, Mapping, Sequence

import plotly.graph_objects as go

from app.components.movement_results import MovementView, parse_movement_response
from app.components.chart_explanations import (
    ChartExplanation,
    render_chart_explanation,
)

FRIENDLY_SCENARIO_NAMES = {
    "low": "Downside",
    "downside": "Downside",
    "base": "Central",
    "central": "Central",
    "high": "Upside",
    "upside": "Upside",
}
SCENARIO_ORDER = ("Downside", "Central", "Upside")


def _finite_number(value: Any, location: str) -> float:
    """Return one finite number while rejecting booleans and text."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location} must be a number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{location} must be finite.")
    return result


def _required_text(mapping: Mapping[str, Any], key: str, location: str) -> str:
    """Read one non-empty text field from a checked mapping."""

    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location}.{key} must be non-empty text.")
    return value.strip()


def _text_list(value: Any, location: str) -> tuple[str, ...]:
    """Validate a bounded list of plain-text fallback messages."""

    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{location} must be a list of text values.")
    if len(value) > 20:
        raise ValueError(f"{location} contains too many values.")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{location} contains invalid text.")
        result.append(item.strip())
    return tuple(result)


@dataclass(frozen=True)
class ScenarioOutcomeView:
    """Store one checked downside, central, or upside result."""

    name: str
    historical_return_percent: float
    shares_purchased: float
    cash_balance: float
    estimated_tax: float
    net_final_value: float
    gain_loss: float
    gain_loss_percent: float

    def as_row(self, currency: str) -> dict[str, Any]:
        """Return a flat, licence-safe row for tables and downloads."""

        return {
            "Scenario": self.name,
            "Historical return (%)": round(self.historical_return_percent, 4),
            "Shares purchased": round(self.shares_purchased, 8),
            f"Cash balance ({currency})": round(self.cash_balance, 2),
            f"Estimated tax ({currency})": round(self.estimated_tax, 2),
            f"Final value ({currency})": round(self.net_final_value, 2),
            f"Gain or loss ({currency})": round(self.gain_loss, 2),
            "Gain or loss (%)": round(self.gain_loss_percent, 4),
        }


@dataclass(frozen=True)
class ScenarioResultView:
    """Store the verified scenario response and user-controlled assumptions."""

    prediction: MovementView
    evidence_count: int
    evidence_end_date: str
    class_median_fallbacks: tuple[str, ...]
    outcomes: tuple[ScenarioOutcomeView, ...]
    method: str
    disclaimer: str
    currency: str
    investment_amount: float

    def rows(self) -> list[dict[str, Any]]:
        """Return all outcomes in stable downside-to-upside order."""

        return [outcome.as_row(self.currency) for outcome in self.outcomes]

    def as_public_dict(self) -> dict[str, Any]:
        """Return report-safe evidence without article text or provider rows."""

        return {
            "status": "PASSED",
            "ticker": self.prediction.ticker,
            "target_session_date": self.prediction.target_session_date,
            "predicted_direction": self.prediction.direction,
            "prediction_confidence_percent": round(
                self.prediction.confidence * 100,
                4,
            ),
            "movement_model": self.prediction.champion_model,
            "investment_amount": self.investment_amount,
            "currency": self.currency,
            "evidence_count": self.evidence_count,
            "evidence_end_date": self.evidence_end_date,
            "class_median_fallbacks": list(self.class_median_fallbacks),
            "method": self.method,
            "outcomes": self.rows(),
            "disclaimer": self.disclaimer,
        }


def _parse_outcome(
    value: Any,
    *,
    location: str,
    investment_amount: float,
) -> ScenarioOutcomeView:
    """Validate one outcome and verify its gain calculations."""

    if not isinstance(value, Mapping):
        raise ValueError(f"{location} must be a JSON object.")
    raw_name = _required_text(value, "scenario", location).lower()
    if raw_name not in FRIENDLY_SCENARIO_NAMES:
        raise ValueError(f"{location}.scenario is unsupported.")

    shares = _finite_number(value.get("shares_purchased"), f"{location}.shares_purchased")
    cash = _finite_number(value.get("cash_balance"), f"{location}.cash_balance")
    tax = _finite_number(value.get("estimated_tax"), f"{location}.estimated_tax")
    final_value = _finite_number(value.get("net_final_value"), f"{location}.net_final_value")
    gain_loss = _finite_number(value.get("gain_loss"), f"{location}.gain_loss")
    gain_percent = _finite_number(
        value.get("gain_loss_percent"),
        f"{location}.gain_loss_percent",
    )
    for label, number in (
        ("shares_purchased", shares),
        ("cash_balance", cash),
        ("estimated_tax", tax),
        ("net_final_value", final_value),
    ):
        if number < 0.0:
            raise ValueError(f"{location}.{label} cannot be negative.")

    expected_gain = final_value - investment_amount
    if abs(expected_gain - gain_loss) > 0.05:
        raise ValueError(f"{location}.gain_loss does not match final value.")
    expected_percent = (gain_loss / investment_amount) * 100.0
    if abs(expected_percent - gain_percent) > 0.05:
        raise ValueError(f"{location}.gain_loss_percent is inconsistent.")

    return ScenarioOutcomeView(
        name=FRIENDLY_SCENARIO_NAMES[raw_name],
        historical_return_percent=_finite_number(
            value.get("historical_return_percent"),
            f"{location}.historical_return_percent",
        ),
        shares_purchased=shares,
        cash_balance=cash,
        estimated_tax=tax,
        net_final_value=final_value,
        gain_loss=gain_loss,
        gain_loss_percent=gain_percent,
    )


def parse_scenario_response(
    payload: Mapping[str, Any],
    *,
    currency: str,
    investment_amount: float,
) -> ScenarioResultView:
    """Validate the exact response fields used by the scenario page."""

    if not isinstance(payload, Mapping):
        raise ValueError("Scenario response must be a JSON object.")
    if payload.get("status") != "PASSED":
        raise ValueError("Scenario response.status must be PASSED.")
    prediction_raw = payload.get("prediction")
    if not isinstance(prediction_raw, Mapping):
        raise ValueError("Scenario response.prediction must be a JSON object.")
    prediction = parse_movement_response(prediction_raw)

    evidence_count = payload.get("evidence_count")
    if isinstance(evidence_count, bool) or not isinstance(evidence_count, int):
        raise ValueError("Scenario response.evidence_count must be a whole number.")
    if evidence_count < 1:
        raise ValueError("Scenario response.evidence_count must be at least one.")

    evidence_end_text = _required_text(
        payload,
        "evidence_end_date",
        "scenario response",
    )
    try:
        evidence_end = date.fromisoformat(evidence_end_text)
        target_date = date.fromisoformat(prediction.target_session_date)
    except ValueError as error:
        raise ValueError("Scenario dates must use ISO calendar format.") from error
    if evidence_end >= target_date:
        raise ValueError(
            "Scenario evidence must end before the target market session."
        )

    raw_outcomes = payload.get("outcomes")
    if not isinstance(raw_outcomes, Sequence) or isinstance(
        raw_outcomes,
        (str, bytes),
    ):
        raise ValueError("Scenario response.outcomes must be a list.")
    if len(raw_outcomes) != 3:
        raise ValueError("Scenario response must contain exactly three outcomes.")
    parsed = tuple(
        _parse_outcome(
            item,
            location=f"scenario response.outcomes[{index}]",
            investment_amount=investment_amount,
        )
        for index, item in enumerate(raw_outcomes)
    )
    names = [item.name for item in parsed]
    if sorted(names) != sorted(SCENARIO_ORDER):
        raise ValueError("Scenario outcomes must include Downside, Central, and Upside.")
    ordered = tuple(sorted(parsed, key=lambda item: SCENARIO_ORDER.index(item.name)))

    normalized_currency = str(currency).strip().upper()
    if len(normalized_currency) != 3 or not normalized_currency.isalpha():
        raise ValueError("Scenario currency must be a three-letter code.")
    if investment_amount <= 0.0 or not math.isfinite(investment_amount):
        raise ValueError("Scenario investment amount must be positive and finite.")

    return ScenarioResultView(
        prediction=prediction,
        evidence_count=evidence_count,
        evidence_end_date=evidence_end.isoformat(),
        class_median_fallbacks=_text_list(
            payload.get("class_median_fallbacks", []),
            "scenario response.class_median_fallbacks",
        ),
        outcomes=ordered,
        method=_required_text(payload, "method", "scenario response"),
        disclaimer=_required_text(payload, "disclaimer", "scenario response"),
        currency=normalized_currency,
        investment_amount=float(investment_amount),
    )


def _money(value: float, currency: str) -> str:
    """Return one readable amount without guessing a currency symbol."""

    return f"{value:,.2f} {currency}"


def build_scenario_bar_figure(view: ScenarioResultView) -> go.Figure:
    """Build a precise 2D comparison of final values and gains or losses."""

    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=[item.name for item in view.outcomes],
            y=[item.net_final_value for item in view.outcomes],
            text=[_money(item.net_final_value, view.currency) for item in view.outcomes],
            textposition="outside",
            name="Final value",
            hovertemplate=(
                "<b>%{x}</b><br>Final value: %{y:,.2f} "
                + view.currency
                + "<extra></extra>"
            ),
        )
    )
    figure.add_hline(
        y=view.investment_amount,
        line_dash="dash",
        annotation_text="Starting amount",
    )
    figure.update_layout(
        title="Possible final value from historical scenarios",
        xaxis_title="Scenario",
        yaxis_title=f"Final value ({view.currency})",
        template="plotly_dark",
        margin=dict(l=20, r=20, t=70, b=20),
        showlegend=False,
    )
    return figure


def build_scenario_3d_figure(view: ScenarioResultView) -> go.Figure:
    """Build a rotatable view using only the three verified outcome values."""

    figure = go.Figure(
        data=[
            go.Scatter3d(
                x=[item.historical_return_percent for item in view.outcomes],
                y=[item.shares_purchased for item in view.outcomes],
                z=[item.net_final_value for item in view.outcomes],
                mode="lines+markers+text",
                text=[item.name for item in view.outcomes],
                textposition="top center",
                marker=dict(
                    size=8,
                    color=[item.gain_loss_percent for item in view.outcomes],
                    colorscale="RdYlGn",
                    colorbar=dict(title="Gain/loss %"),
                ),
                hovertemplate=(
                    "<b>%{text}</b><br>Historical return: %{x:.2f}%"
                    "<br>Shares: %{y:.6f}<br>Final value: %{z:,.2f} "
                    + view.currency
                    + "<extra></extra>"
                ),
            )
        ]
    )
    figure.update_layout(
        title="Interactive scenario relationship",
        scene=dict(
            xaxis_title="Historical return (%)",
            yaxis_title="Shares purchased",
            zaxis_title=f"Final value ({view.currency})",
        ),
        template="plotly_dark",
        margin=dict(l=0, r=0, t=55, b=0),
    )
    return figure


def build_scenario_explanation(view: ScenarioResultView) -> ChartExplanation:
    """Return the plain-language message below the scenario charts."""

    downside, central, upside = view.outcomes
    return ChartExplanation(
        what_it_shows=(
            "The charts compare three outcomes built from earlier market "
            "reactions. They show the same verified values in 2D and 3D."
        ),
        why_it_matters=(
            "A range is more honest than one exact return because earlier "
            "events did not all produce the same result."
        ),
        conclusion=(
            f"The central example ends at {_money(central.net_final_value, view.currency)}. "
            f"The displayed range runs from {_money(downside.net_final_value, view.currency)} "
            f"to {_money(upside.net_final_value, view.currency)}."
        ),
        uncertainty=(
            "These are historical research examples, not promised future values. "
            "Unexpected news, market conditions, costs, and taxes can change the result."
        ),
    )


def render_scenario_result(st: Any, view: ScenarioResultView) -> None:
    """Render outcome cards, 2D and 3D charts, tables, and limitations."""

    st.markdown(
        f"""
        <section class="rm-scenario-hero">
          <div>
            <p class="rm-panel-kicker">RESEARCH SCENARIO</p>
            <h2>{escape(view.prediction.ticker)} · {escape(view.prediction.direction)}</h2>
            <p>{view.prediction.confidence * 100:.1f}% model support for the mapped session</p>
          </div>
          <div class="rm-scenario-evidence">
            <span>Earlier events used</span><strong>{view.evidence_count}</strong>
            <span>Evidence ends</span><strong>{escape(view.evidence_end_date)}</strong>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    columns = st.columns(3, gap="medium")
    tone_by_name = {"Downside": "down", "Central": "flat", "Upside": "up"}
    for column, outcome in zip(columns, view.outcomes):
        with column:
            tone = tone_by_name[outcome.name]
            st.markdown(
                f"""
                <article class="rm-scenario-card rm-scenario-{tone}">
                  <span>{escape(outcome.name)}</span>
                  <h3>{escape(_money(outcome.net_final_value, view.currency))}</h3>
                  <strong>{outcome.gain_loss_percent:+.2f}%</strong>
                  <p>Gain or loss: {escape(_money(outcome.gain_loss, view.currency))}</p>
                </article>
                """,
                unsafe_allow_html=True,
            )

    st.plotly_chart(
        build_scenario_bar_figure(view),
        use_container_width=True,
        config={"displaylogo": False},
    )
    st.markdown("### Interactive 3D view")
    st.plotly_chart(
        build_scenario_3d_figure(view),
        use_container_width=True,
        config={"displaylogo": False, "scrollZoom": True},
    )
    render_chart_explanation(st, build_scenario_explanation(view))
    with st.expander("Open the accessible 2D fallback and exact values"):
        st.dataframe(view.rows(), use_container_width=True, hide_index=True)

    st.markdown("### How the result was built")
    st.write(view.method)
    if view.class_median_fallbacks:
        st.warning(
            "Some outcome classes needed a wider historical fallback: "
            + "; ".join(view.class_median_fallbacks)
        )
    st.caption(view.disclaimer)
