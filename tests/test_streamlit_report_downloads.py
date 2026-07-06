"""Verify PDF, JSON, and CSV downloads are valid and licence-safe."""

from __future__ import annotations

import json

import pytest

from tests.streamlit_test_support import import_project_module


def test_plain_pdf_has_a_valid_header_and_end_marker() -> None:
    """Require the built-in report writer to create a complete PDF document."""

    module = import_project_module("app.services.report_builder")
    data = module.build_plain_pdf(
        title="Portfolio Report",
        owner="Ruturaj Mokashi",
        sections=(("Summary", ["Verified result"]),),
    )
    assert data.startswith(b"%PDF-1.4")
    assert data.rstrip().endswith(b"%%EOF")
    assert b"Ruturaj Mokashi" in data


def test_pdf_escapes_parentheses_and_backslashes() -> None:
    """Keep report text from breaking PDF string commands."""

    module = import_project_module("app.services.report_builder")
    data = module.build_plain_pdf(
        title="Result (checked)",
        owner="Ruturaj \\ Mokashi",
        sections=(("Summary", ["Value (safe) \\ verified"]),),
    )
    assert b"\\(checked\\)" in data
    assert b"\\\\" in data


def test_json_download_is_stable_and_utf8() -> None:
    """Require readable sorted JSON with a final line break."""

    module = import_project_module("app.services.report_builder")
    data = module.build_json_bytes({"z": 1, "a": "verified"})
    decoded = data.decode("utf-8")
    assert decoded.endswith("\n")
    assert decoded.index('"a"') < decoded.index('"z"')
    assert json.loads(decoded) == {"a": "verified", "z": 1}


def test_csv_download_has_utf8_spreadsheet_marker() -> None:
    """Require a UTF-8 byte-order marker for common spreadsheet tools."""

    module = import_project_module("app.services.report_builder")
    data = module.build_csv_bytes(({"Case": "Central", "Value": 100.0},))
    assert data.startswith(b"\xef\xbb\xbf")
    assert b"Case,Value" in data


def test_csv_rejects_empty_rows() -> None:
    """Do not create a misleading empty report."""

    module = import_project_module("app.services.report_builder")
    with pytest.raises(ValueError, match="at least one row"):
        module.build_csv_bytes(())


def test_csv_rejects_inconsistent_columns() -> None:
    """Keep every output row aligned to the same visible columns."""

    module = import_project_module("app.services.report_builder")
    with pytest.raises(ValueError, match="same ordered fields"):
        module.build_csv_bytes(({"A": 1}, {"B": 2}))


def test_provenance_download_uses_only_sanitized_view_data() -> None:
    """Build evidence JSON from the already checked public object."""

    report_module = import_project_module("app.services.report_builder")
    provenance_module = import_project_module("app.components.provenance_panels")
    view = provenance_module.ProvenanceView(data={"model": "DistilBERT"})
    artifact = report_module.build_provenance_download(view)
    assert artifact.mime_type == "application/json"
    decoded = json.loads(artifact.data)
    assert decoded["provenance"] == {"model": "DistilBERT"}


def test_download_file_names_do_not_contain_local_paths() -> None:
    """Keep public download names independent of the developer machine."""

    module = import_project_module("app.services.report_builder")
    artifact = module.DownloadArtifact("Download", "safe_report.json", "application/json", b"{}")
    assert "/" not in artifact.file_name
    assert "\\" not in artifact.file_name
