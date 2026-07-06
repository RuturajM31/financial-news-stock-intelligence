"""Render the recruiter-focused portfolio story for Ruturaj Mokashi."""

from __future__ import annotations

from typing import Any


def render_about_ruturaj_page(st: Any) -> None:
    """Present verified background, project ownership, skills, and next phases."""

    st.markdown(
        """
        <section class="rm-about-hero">
          <div class="rm-about-monogram" aria-hidden="true">RM</div>
          <div>
            <p class="rm-panel-kicker">PROJECT OWNER</p>
            <h2>Ruturaj Mokashi</h2>
            <p>
              AI and data professional building practical end-to-end systems
              that connect business questions, reliable data, machine learning,
              software engineering, and clear user experience.
            </p>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Professional background")
    st.write(
        "Ruturaj has experience in business intelligence, data analysis, and data "
        "engineering. At Ippen Digital, he worked on reporting, dashboards, data "
        "quality, stakeholder training, traffic, revenue, retention, search, and "
        "editorial analytics. His work included Python, SQL, Apache Druid, Apache "
        "Superset, the Imply Stack, and Google Data Studio."
    )

    education = st.columns(3)
    education[0].metric("BCA", "University of Pune · 2009")
    education[1].metric("Executive MBA", "Symbiosis · 2017")
    education[2].metric("MSc", "Big Data Analytics · IÉSEG · 2020")

    st.markdown("### What Ruturaj built in this project")
    capabilities = (
        (
            "Financial intelligence",
            "News sentiment, market-session mapping, stock-movement forecasting, "
            "historical evidence, and research scenarios.",
        ),
        (
            "Model engineering",
            "BERT, DistilBERT, and BERT LoRA training, comparison, error review, "
            "word influence, and explainability.",
        ),
        (
            "Application engineering",
            "A verified FastAPI backend, isolated model workers, a premium Streamlit "
            "interface, safe reports, testing, and rollback protection.",
        ),
        (
            "Product thinking",
            "Simple language, clear conclusions, accessible 2D fallbacks, purposeful "
            "3D visuals, source verification, and honest limits.",
        ),
    )
    for title, body in capabilities:
        with st.container(border=True):
            st.markdown(f"#### {title}")
            st.write(body)

    st.markdown("### Engineering approach")
    st.markdown(
        """
        - Use verified evidence instead of assumptions.
        - Prevent later market information from leaking into earlier predictions.
        - Keep model workers separate for stability.
        - Fail safely when required data is missing or changed.
        - Explain every major chart in plain words.
        - Protect private data and provider restrictions.
        - Test real workflows, not only individual functions.
        - Keep rollback evidence for every controlled package.
        """
    )

    st.markdown("### Delivery status")
    completed, next_steps = st.columns(2)
    with completed:
        st.success(
            "Completed and verified: market data foundation, sentiment models, "
            "movement intelligence, explainability, FastAPI, and Streamlit "
            "Packages 1–7."
        )
    with next_steps:
        st.info(
            "Next phases: logging and monitoring, Docker containerization, "
            "Kubernetes and Helm, CI/CD and security closure, public deployment, "
            "and final end-to-end documentation."
        )

    st.markdown("### Recruiter summary")
    st.markdown(
        """
        <section class="rm-recruiter-summary">
          <strong>Portfolio position</strong>
          <p>
            Ruturaj Mokashi designed and built a complete financial intelligence
            product across data, machine learning, APIs, interface design,
            testing, security boundaries, and controlled delivery. The project
            demonstrates the ability to turn a technical model into a clear,
            testable, and useful product for real users.
          </p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "Containerization, monitoring, and public deployment are shown as next "
        "phases until their own verification gates pass."
    )
