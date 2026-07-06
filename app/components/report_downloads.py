"""Render report downloads created from checked public result objects."""

from __future__ import annotations

from typing import Any

from app.branding import PortfolioBrand
from app.components.provenance_panels import ProvenanceView
from app.components.scenario_results import ScenarioResultView
from app.services.report_builder import (
    build_provenance_download,
    build_scenario_downloads,
)


def render_scenario_downloads(
    st: Any,
    view: ScenarioResultView,
    brand: PortfolioBrand,
) -> None:
    """Render PDF, JSON, and CSV buttons without including article text."""

    st.markdown("### Download this research result")
    st.caption(
        "The files contain derived scenario values, model references, and limits. "
        "They do not contain the submitted article text or private provider data."
    )
    columns = st.columns(3)
    for column, artifact in zip(
        columns,
        build_scenario_downloads(view, owner=brand.owner_name),
    ):
        with column:
            st.download_button(
                artifact.label,
                data=artifact.data,
                file_name=artifact.file_name,
                mime=artifact.mime_type,
                use_container_width=True,
                on_click="ignore",
            )


def render_provenance_download(st: Any, view: ProvenanceView) -> None:
    """Render the sanitized provenance JSON download."""

    artifact = build_provenance_download(view)
    st.download_button(
        artifact.label,
        data=artifact.data,
        file_name=artifact.file_name,
        mime=artifact.mime_type,
        use_container_width=True,
        on_click="ignore",
    )
