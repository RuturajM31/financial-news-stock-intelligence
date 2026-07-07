"""Present honest source-processing evidence for one completed analysis.

The component explains what FastAPI verified during text extraction. It never
labels a publisher as trustworthy unless a future verified source registry says
so. Accepting and extracting a public URL proves safe processing, not editorial
accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from typing import Any
from urllib.parse import urlsplit

from app.components.status_badges import build_status_badge_html


_ALLOWED_SOURCE_KINDS = frozenset({"text", "file", "url", "example"})


@dataclass(frozen=True)
class SourceEvidence:
    """Store the safe source facts that may be shown or kept in session."""

    source_kind: str
    display_name: str
    verification_label: str
    explanation: str
    limitation: str

    def as_dict(self) -> dict[str, str]:
        """Return a session-safe value without raw text, bytes, or full URLs."""

        return {
            "source_kind": self.source_kind,
            "display_name": self.display_name,
            "verification_label": self.verification_label,
            "explanation": self.explanation,
            "limitation": self.limitation,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SourceEvidence":
        """Rebuild evidence from a checked current-session mapping."""

        required = (
            "source_kind",
            "display_name",
            "verification_label",
            "explanation",
            "limitation",
        )
        missing = [key for key in required if not isinstance(value.get(key), str)]
        if missing:
            raise ValueError(
                "Saved source evidence is missing text fields: " + ", ".join(missing)
            )
        evidence = cls(**{key: value[key].strip() for key in required})
        evidence.validate()
        return evidence

    def validate(self) -> None:
        """Reject unsupported source types or empty user-facing text."""

        if self.source_kind not in _ALLOWED_SOURCE_KINDS:
            raise ValueError(f"Unsupported source kind: {self.source_kind!r}.")
        fields = {
            "display_name": self.display_name,
            "verification_label": self.verification_label,
            "explanation": self.explanation,
            "limitation": self.limitation,
        }
        empty = [name for name, value in fields.items() if not value.strip()]
        if empty:
            raise ValueError(
                "Source evidence fields must not be empty: " + ", ".join(empty)
            )


def build_source_evidence(
    *,
    source_kind: str,
    display_name: str,
    public_url: str | None = None,
) -> SourceEvidence:
    """Build evidence that matches what the successful API request proved."""

    safe_name = display_name.strip()
    if not safe_name:
        raise ValueError("The source display name must not be empty.")

    if source_kind == "text":
        evidence = SourceEvidence(
            source_kind="text",
            display_name="Pasted text",
            verification_label="Input accepted",
            explanation=(
                "FastAPI accepted the submitted text and returned a complete "
                "sentiment response."
            ),
            limitation=(
                "This confirms successful processing. It does not verify who "
                "originally wrote the text or whether every statement is true."
            ),
        )
    elif source_kind == "example":
        evidence = SourceEvidence(
            source_kind="example",
            display_name="Built-in demonstration",
            verification_label="Demonstration only",
            explanation=(
                "The interface used the clearly labelled example text included "
                "with this portfolio application."
            ),
            limitation=(
                "The example is not a real company announcement and must not be "
                "used as market evidence."
            ),
        )
    elif source_kind == "file":
        evidence = SourceEvidence(
            source_kind="file",
            display_name=safe_name,
            verification_label="File processed",
            explanation=(
                "FastAPI accepted the supported file, extracted its text, and "
                "returned one or more sentiment results."
            ),
            limitation=(
                "Successful extraction does not prove that the file came from an "
                "official or accurate publisher."
            ),
        )
    elif source_kind == "url":
        if not public_url:
            raise ValueError("A public URL is required for URL source evidence.")
        hostname = urlsplit(public_url).hostname
        if not hostname:
            raise ValueError("The public URL does not contain a website name.")
        evidence = SourceEvidence(
            source_kind="url",
            display_name=hostname,
            verification_label="Public URL processed",
            explanation=(
                "FastAPI accepted the public address, applied its network and "
                "size checks, extracted text, and returned a sentiment result."
            ),
            limitation=(
                "This confirms safe extraction from the displayed website. It "
                "does not guarantee that the publisher or article is accurate."
            ),
        )
    else:
        raise ValueError(f"Unsupported source kind: {source_kind!r}.")

    evidence.validate()
    return evidence


def render_source_evidence(st: Any, evidence: SourceEvidence) -> None:
    """Render source evidence and its limitation in a clear summary panel."""

    evidence.validate()
    badge_tone = "warning" if evidence.source_kind == "example" else "passed"
    badge = build_status_badge_html(evidence.verification_label, badge_tone)
    st.markdown(
        f"""
        <section class="rm-panel rm-section-gap">
          {badge}
          <h3>Source and processing</h3>
          <p><strong>Source:</strong> {escape(evidence.display_name)}</p>
          <p><strong>What was checked:</strong> {escape(evidence.explanation)}</p>
          <p><strong>Limit:</strong> {escape(evidence.limitation)}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
