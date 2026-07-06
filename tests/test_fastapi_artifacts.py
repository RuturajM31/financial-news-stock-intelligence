"""Tests for artifact checksum and permission verification."""

import hashlib
from pathlib import Path

import pytest

from financial_news_intelligence.api.artifacts import _verify_inventory
from financial_news_intelligence.api.errors import ApiProblem


def test_artifact_inventory_accepts_matching_owner_only_file(tmp_path: Path) -> None:
    """Prepare a matching artifact, run verification, and check resolved path."""

    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"verified")
    artifact.chmod(0o600)
    entries = [
        {
            "path": "artifact.bin",
            "sha256": hashlib.sha256(b"verified").hexdigest(),
            "size_bytes": len(b"verified"),
        }
    ]

    result = _verify_inventory(tmp_path, entries, "test inventory")

    assert result["artifact.bin"] == artifact.resolve()


def test_artifact_inventory_rejects_changed_checksum(tmp_path: Path) -> None:
    """Prepare changed bytes, run verification, and check checksum failure."""

    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"changed")
    artifact.chmod(0o600)
    entries = [
        {
            "path": "artifact.bin",
            "sha256": hashlib.sha256(b"verified").hexdigest(),
            "size_bytes": len(b"changed"),
        }
    ]

    with pytest.raises(ApiProblem) as captured:
        _verify_inventory(tmp_path, entries, "test inventory")

    assert captured.value.error_code == "artifact_checksum_changed"
