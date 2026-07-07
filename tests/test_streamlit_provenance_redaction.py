"""Verify provenance removes secrets, paths, and unsafe nested values."""

from __future__ import annotations

import math

import pytest

from tests.streamlit_test_support import import_project_module


def test_sensitive_keys_are_removed_at_every_nested_level() -> None:
    """Drop token-like fields without discarding safe neighbouring evidence."""

    module = import_project_module("app.components.provenance_panels")
    value = {
        "model": "DistilBERT",
        "api_key": "blocked",
        "nested": {"token": "blocked", "version": "1.0"},
    }
    assert module.sanitize_public_value(value) == {
        "model": "DistilBERT",
        "nested": {"version": "1.0"},
    }


def test_local_user_paths_are_rejected() -> None:
    """Prevent developer-machine locations from reaching the interface."""

    module = import_project_module("app.components.provenance_panels")
    with pytest.raises(ValueError, match="local user path"):
        module.sanitize_public_value("/" + "Users" + "/example/private/file.json")


def test_authentication_text_is_rejected() -> None:
    """Reject bearer and X-API-Key values even under an innocent field name."""

    module = import_project_module("app.components.provenance_panels")
    with pytest.raises(ValueError, match="private authentication"):
        module.sanitize_public_value("Bearer abcdefghijklmnop")
    with pytest.raises(ValueError, match="private authentication"):
        module.sanitize_public_value("X-API-Key: abcdefghijklmnop")


def test_non_finite_numbers_are_rejected() -> None:
    """Do not serialize NaN or infinity into evidence downloads."""

    module = import_project_module("app.components.provenance_panels")
    for value in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError, match="non-finite"):
            module.sanitize_public_value(value)


def test_excessive_nesting_is_rejected() -> None:
    """Bound recursive provenance so display work stays predictable."""

    module = import_project_module("app.components.provenance_panels")
    value: object = "end"
    for index in range(module.MAXIMUM_DEPTH + 2):
        value = {f"level_{index}": value}
    with pytest.raises(ValueError, match="nested too deeply"):
        module.sanitize_public_value(value)


def test_provenance_parser_requires_passed_status() -> None:
    """Fail closed when FastAPI does not mark the evidence response as passed."""

    module = import_project_module("app.components.provenance_panels")
    with pytest.raises(ValueError, match="must be PASSED"):
        module.parse_provenance_response({"status": "FAILED", "provenance": {"a": 1}})


def test_provenance_rows_flatten_nested_evidence_stably() -> None:
    """Provide an accessible text table for nested public evidence."""

    module = import_project_module("app.components.provenance_panels")
    view = module.ProvenanceView(data={"model": {"name": "DistilBERT"}, "checks": ["hash"]})
    rows = view.rows()
    fields = {row["Evidence field"] for row in rows}
    assert "model → name" in fields
    assert "checks → item 1" in fields


def test_empty_public_provenance_is_rejected_after_redaction() -> None:
    """Do not present an empty evidence page when every field was sensitive."""

    module = import_project_module("app.components.provenance_panels")
    with pytest.raises(ValueError, match="No public provenance"):
        module.parse_provenance_response(
            {"status": "PASSED", "provenance": {"token": "blocked"}}
        )
