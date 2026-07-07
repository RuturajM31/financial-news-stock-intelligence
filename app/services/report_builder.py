"""Build small, licence-safe portfolio report downloads without extra libraries.

The PDF writer intentionally supports plain text only. Charts remain available
in the application, while the report records exact values, evidence boundaries,
and the required disclaimer. No article text, token, local path, or restricted
provider row is included.
"""

from __future__ import annotations

import csv
import io
import json
import textwrap
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

from app.components.provenance_panels import ProvenanceView
from app.components.scenario_results import ScenarioResultView


@dataclass(frozen=True)
class DownloadArtifact:
    """Store one downloadable file with a safe name and media type."""

    label: str
    file_name: str
    mime_type: str
    data: bytes


def _safe_pdf_text(value: Any) -> str:
    """Return printable text supported by the built-in Helvetica font."""

    text = str(value).replace("\r", " ").replace("\n", " ").strip()
    return text.encode("latin-1", errors="replace").decode("latin-1")


def _pdf_escape(value: str) -> str:
    """Escape the PDF string delimiters used by one text command."""

    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _wrapped_lines(sections: Sequence[tuple[str, Sequence[str]]]) -> list[str]:
    """Flatten titled sections into readable lines with stable wrapping."""

    lines: list[str] = []
    for title, values in sections:
        if lines:
            lines.append("")
        lines.append(title.upper())
        for value in values:
            safe = _safe_pdf_text(value)
            wrapped = textwrap.wrap(
                safe,
                width=88,
                break_long_words=False,
                break_on_hyphens=False,
            )
            lines.extend(wrapped or [""])
    return lines


def build_plain_pdf(
    *,
    title: str,
    owner: str,
    sections: Sequence[tuple[str, Sequence[str]]],
) -> bytes:
    """Create a valid multi-page PDF using only the Python standard library."""

    report_lines = [
        _safe_pdf_text(title),
        f"Designed and built by {_safe_pdf_text(owner)}",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        *_wrapped_lines(sections),
    ]
    lines_per_page = 48
    pages = [
        report_lines[index:index + lines_per_page]
        for index in range(0, len(report_lines), lines_per_page)
    ] or [[_safe_pdf_text(title)]]

    objects: dict[int, bytes] = {}
    objects[1] = b"<< /Type /Catalog /Pages 2 0 R >>"
    objects[3] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
    page_object_numbers: list[int] = []
    next_object = 4
    for page_index, page_lines in enumerate(pages, start=1):
        page_number = next_object
        content_number = next_object + 1
        next_object += 2
        page_object_numbers.append(page_number)

        commands = ["BT", "/F1 10 Tf", "50 795 Td", "14 TL"]
        for line_index, line in enumerate(page_lines):
            if line_index == 0 and page_index == 1:
                commands.extend(["/F1 18 Tf", f"({_pdf_escape(line)}) Tj", "0 -24 Td", "/F1 10 Tf"])
            else:
                commands.extend([f"({_pdf_escape(line)}) Tj", "T*"])
        commands.append("ET")
        stream = "\n".join(commands).encode("latin-1")
        objects[content_number] = (
            f"<< /Length {len(stream)} >>\nstream\n".encode("ascii")
            + stream
            + b"\nendstream"
        )
        objects[page_number] = (
            "<< /Type /Page /Parent 2 0 R "
            "/MediaBox [0 0 595 842] "
            "/Resources << /Font << /F1 3 0 R >> >> "
            f"/Contents {content_number} 0 R >>"
        ).encode("ascii")

    kids = " ".join(f"{number} 0 R" for number in page_object_numbers)
    objects[2] = f"<< /Type /Pages /Count {len(pages)} /Kids [{kids}] >>".encode("ascii")

    output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = {0: 0}
    for number in range(1, max(objects) + 1):
        offsets[number] = len(output)
        output.extend(f"{number} 0 obj\n".encode("ascii"))
        output.extend(objects[number])
        output.extend(b"\nendobj\n")

    xref_offset = len(output)
    output.extend(f"xref\n0 {max(objects) + 1}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for number in range(1, max(objects) + 1):
        output.extend(f"{offsets[number]:010d} 00000 n \n".encode("ascii"))
    output.extend(
        (
            f"trailer\n<< /Size {max(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    return bytes(output)


def build_json_bytes(value: Mapping[str, Any]) -> bytes:
    """Serialize one public mapping with stable ordering and readable spacing."""

    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def build_csv_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    """Serialize flat scenario rows as UTF-8 CSV with a spreadsheet marker."""

    materialized = list(rows)
    if not materialized:
        raise ValueError("CSV download requires at least one row.")
    fieldnames = list(materialized[0].keys())
    if any(list(row.keys()) != fieldnames for row in materialized):
        raise ValueError("CSV rows must have the same ordered fields.")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(materialized)
    return b"\xef\xbb\xbf" + buffer.getvalue().encode("utf-8")


def build_scenario_downloads(
    view: ScenarioResultView,
    *,
    owner: str,
) -> tuple[DownloadArtifact, ...]:
    """Build branded PDF, JSON, and CSV files from one checked scenario."""

    public = view.as_public_dict()
    scenario_lines = [
        f"Ticker: {view.prediction.ticker}",
        f"Target session: {view.prediction.target_session_date}",
        f"Predicted direction: {view.prediction.direction}",
        f"Model support: {view.prediction.confidence * 100:.2f}%",
        f"Investment amount: {view.investment_amount:,.2f} {view.currency}",
        f"Earlier events used: {view.evidence_count}",
        f"Evidence end date: {view.evidence_end_date}",
    ]
    outcome_lines = [
        (
            f"{item.name}: final value {item.net_final_value:,.2f} {view.currency}; "
            f"gain or loss {item.gain_loss:+,.2f} {view.currency} "
            f"({item.gain_loss_percent:+.2f}%)."
        )
        for item in view.outcomes
    ]
    pdf = build_plain_pdf(
        title="Financial News Scenario Report",
        owner=owner,
        sections=(
            ("Executive summary", scenario_lines),
            ("Historical scenarios", outcome_lines),
            ("Method", [view.method]),
            ("Limitations", [view.disclaimer]),
        ),
    )
    stem = f"scenario_{view.prediction.ticker.lower()}_{view.prediction.target_session_date}"
    return (
        DownloadArtifact("Download PDF report", f"{stem}.pdf", "application/pdf", pdf),
        DownloadArtifact("Download JSON evidence", f"{stem}.json", "application/json", build_json_bytes(public)),
        DownloadArtifact("Download scenario CSV", f"{stem}.csv", "text/csv", build_csv_bytes(view.rows())),
    )


def build_provenance_download(view: ProvenanceView) -> DownloadArtifact:
    """Build one safe JSON provenance file after display sanitization."""

    return DownloadArtifact(
        label="Download verified provenance JSON",
        file_name="financial_intelligence_provenance.json",
        mime_type="application/json",
        data=build_json_bytes({"status": "PASSED", "provenance": view.data}),
    )
