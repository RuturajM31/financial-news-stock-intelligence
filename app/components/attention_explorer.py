"""Explain the attention boundary honestly and prepare for future verified data."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def has_verified_attention(payload: Mapping[str, Any]) -> bool:
    """Return true only for a complete optional attention contract."""

    attention = payload.get("attention")
    if not isinstance(attention, Mapping):
        return False
    tokens = attention.get("tokens")
    values = attention.get("values")
    method = attention.get("method")
    if not isinstance(tokens, Sequence) or isinstance(tokens, (str, bytes)):
        return False
    if not tokens or not all(isinstance(token, str) and token for token in tokens):
        return False
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return False
    if len(values) != len(tokens):
        return False
    if not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
        return False
    return isinstance(method, str) and bool(method.strip())


def render_attention_explorer_status(st: Any, payload: Mapping[str, Any]) -> None:
    """Render verified attention only when FastAPI supplies its full contract."""

    st.markdown("### Attention explorer")
    if not has_verified_attention(payload):
        st.info(
            "The verified FastAPI service does not currently expose attention "
            "values. The app will not invent them or label them as word importance."
        )
        st.markdown(
            """
            **How to read this boundary**

            - Attention shows where the model looked.
            - It does not always show the true reason for the prediction.
            - The verified word-influence check above measures a real change in
              the live result and is therefore the main explanation shown here.
            """
        )
        return

    attention = payload["attention"]
    rows = [
        {"Token": token, "Attention value": float(value)}
        for token, value in zip(
            attention["tokens"],
            attention["values"],
            strict=True,
        )
    ]
    st.warning(
        "Attention is an advanced diagnostic. It is not a percentage of true "
        "importance and must not be read as cause and effect."
    )
    st.dataframe(rows, width="stretch")
    st.caption(f"Verified method supplied by FastAPI: {attention['method']}")
