"""Verify data sources and create reproducible provenance records."""

import hashlib
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from financial_news_intelligence.paths import PROJECT_ROOT
from financial_news_intelligence.schemas.provenance import (
    DataPurpose,
    SourceAssessment,
    SourceProvenance,
    VerificationStatus,
)


# ============================================================
# 1. DEFAULT REGISTRY
# ============================================================

DEFAULT_SOURCE_REGISTRY = (
    PROJECT_ROOT / "configs" / "verified_sources.yaml"
)


# ============================================================
# 2. DOMAIN-SPECIFIC ERRORS
# ============================================================

class UnverifiedSourceError(ValueError):
    """Raised when a URL is absent from the approved registry."""


class SourceUsageError(ValueError):
    """Raised when data is not approved for a requested purpose."""


# ============================================================
# 3. LOAD AND VALIDATE THE REGISTRY
# ============================================================

def load_verified_source_registry(
    registry_path: Path = DEFAULT_SOURCE_REGISTRY,
) -> dict[str, dict[str, Any]]:
    """
    Load approved source definitions.

    The configuration uses JSON-compatible YAML so it can be parsed
    without introducing a new third-party dependency.
    """

    if not registry_path.exists():
        raise FileNotFoundError(
            f"Verified-source registry not found: {registry_path}"
        )

    with registry_path.open("r", encoding="utf-8") as registry_file:
        registry_document = json.load(registry_file)

    source_rows = registry_document.get("sources")

    if not isinstance(source_rows, list) or not source_rows:
        raise ValueError(
            "Verified-source registry must contain a non-empty sources list."
        )

    required_fields = {
        "id",
        "name",
        "domains",
        "verification_status",
        "allowed_purposes",
        "requires_cross_check",
    }

    registry: dict[str, dict[str, Any]] = {}

    for source_row in source_rows:
        missing_fields = required_fields.difference(source_row)

        if missing_fields:
            raise ValueError(
                "Registry source is missing fields: "
                + ", ".join(sorted(missing_fields))
            )

        source_id = str(source_row["id"]).strip()

        if not source_id:
            raise ValueError("Registry source ID cannot be empty.")

        if source_id in registry:
            raise ValueError(
                f"Duplicate registry source ID: {source_id}"
            )

        # Validate controlled enum values while loading the configuration.
        verification_status = VerificationStatus(
            source_row["verification_status"]
        )

        allowed_purposes = tuple(
            DataPurpose(purpose)
            for purpose in source_row["allowed_purposes"]
        )

        normalized_domains = [
            str(domain).lower().strip().lstrip(".")
            for domain in source_row["domains"]
            if str(domain).strip()
        ]

        registry[source_id] = {
            **source_row,
            "id": source_id,
            "domains": normalized_domains,
            "verification_status": verification_status,
            "allowed_purposes": allowed_purposes,
        }

    return registry


# ============================================================
# 4. MATCH A URL TO AN APPROVED SOURCE
# ============================================================

def assess_source_url(
    source_url: str,
    registry_path: Path = DEFAULT_SOURCE_REGISTRY,
) -> SourceAssessment:
    """
    Match a source URL against approved registry domains.

    Unknown domains are rejected instead of silently entering the
    training or investment-scenario pipelines.
    """

    parsed_url = urlparse(source_url)
    host = (parsed_url.hostname or "").lower()

    if parsed_url.scheme not in {"http", "https"} or not host:
        raise ValueError(
            "source_url must be a valid HTTP or HTTPS URL."
        )

    registry = load_verified_source_registry(registry_path)

    for source_id, source_definition in registry.items():
        for approved_domain in source_definition["domains"]:
            domain_matches = (
                host == approved_domain
                or host.endswith(f".{approved_domain}")
            )

            if domain_matches:
                return SourceAssessment(
                    source_id=source_id,
                    source_name=source_definition["name"],
                    source_url=source_url,
                    host=host,
                    verification_status=source_definition[
                        "verification_status"
                    ],
                    allowed_purposes=source_definition[
                        "allowed_purposes"
                    ],
                    requires_cross_check=source_definition[
                        "requires_cross_check"
                    ],
                )

    raise UnverifiedSourceError(
        f"Source domain is not approved: {host}"
    )


# ============================================================
# 5. REPRODUCIBLE CHECKSUM
# ============================================================

def compute_sha256(raw_payload: Any) -> str:
    """
    Create a reproducible SHA-256 checksum.

    Dictionaries are serialized with sorted keys so logically identical
    payloads receive the same checksum.
    """

    if isinstance(raw_payload, bytes):
        payload_bytes = raw_payload

    elif isinstance(raw_payload, str):
        payload_bytes = raw_payload.encode("utf-8")

    else:
        canonical_payload = json.dumps(
            raw_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        )
        payload_bytes = canonical_payload.encode("utf-8")

    return hashlib.sha256(payload_bytes).hexdigest()


# ============================================================
# 6. BUILD A COMPLETE PROVENANCE RECORD
# ============================================================

def build_provenance(
    *,
    source_url: str,
    raw_payload: Any,
    retrieved_at: datetime,
    as_of_date: date,
    raw_record_count: int,
    cross_checked: bool = False,
    cross_check_source_url: str | None = None,
    registry_path: Path = DEFAULT_SOURCE_REGISTRY,
) -> SourceProvenance:
    """
    Verify a source and attach traceability metadata to its data.

    Output is used as the gate before training, reporting, or scenario
    calculations.
    """

    assessment = assess_source_url(
        source_url,
        registry_path,
    )

    return SourceProvenance(
        source_id=assessment.source_id,
        source_name=assessment.source_name,
        source_url=source_url,
        retrieved_at=retrieved_at,
        as_of_date=as_of_date,
        verification_status=assessment.verification_status,
        allowed_purposes=assessment.allowed_purposes,
        requires_cross_check=assessment.requires_cross_check,
        checksum_sha256=compute_sha256(raw_payload),
        raw_record_count=raw_record_count,
        cross_checked=cross_checked,
        cross_check_source_url=cross_check_source_url,
    )


# ============================================================
# 7. BLOCK UNSAFE DATA USE
# ============================================================

def assert_usage_allowed(
    provenance: SourceProvenance,
    purpose: DataPurpose,
) -> None:
    """
    Stop unapproved data from reaching a protected pipeline.

    Secondary sources requiring confirmation remain blocked until their
    data has been cross-checked against another traceable source.
    """

    normalized_purpose = DataPurpose(purpose)

    if provenance.verification_status not in {
        VerificationStatus.VERIFIED_PRIMARY,
        VerificationStatus.VERIFIED_SECONDARY,
    }:
        raise SourceUsageError(
            "Unverified or blocked data cannot be used."
        )

    if normalized_purpose not in provenance.allowed_purposes:
        raise SourceUsageError(
            f"Source is not approved for: {normalized_purpose.value}"
        )

    if (
        provenance.requires_cross_check
        and not provenance.cross_checked
    ):
        raise SourceUsageError(
            "This secondary source requires a documented cross-check."
        )
