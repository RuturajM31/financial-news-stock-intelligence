"""Schemas for source verification and data provenance."""

import re
from datetime import date, datetime
from enum import Enum
from urllib.parse import urlparse

from pydantic import Field, field_validator, model_validator

from financial_news_intelligence.schemas.common import ProjectSchema


# ============================================================
# 1. CONTROLLED VALUES
# ============================================================

class VerificationStatus(str, Enum):
    """Trust level assigned to a data source."""

    VERIFIED_PRIMARY = "verified_primary"
    VERIFIED_SECONDARY = "verified_secondary"
    UNVERIFIED = "unverified"
    BLOCKED = "blocked"


class DataPurpose(str, Enum):
    """Ways in which verified data may be used."""

    TRAINING = "training"
    INVESTMENT_SCENARIOS = "investment_scenarios"
    REPORTING = "reporting"


# ============================================================
# 2. SOURCE ASSESSMENT
# ============================================================

class SourceAssessment(ProjectSchema):
    """Result returned after checking a URL against the registry."""

    source_id: str = Field(min_length=1)
    source_name: str = Field(min_length=1)
    source_url: str
    host: str = Field(min_length=1)
    verification_status: VerificationStatus
    allowed_purposes: tuple[DataPurpose, ...]
    requires_cross_check: bool = False


# ============================================================
# 3. PER-DATASET PROVENANCE
# ============================================================

class SourceProvenance(ProjectSchema):
    """Traceability metadata attached to downloaded or collected data."""

    source_id: str = Field(min_length=1)
    source_name: str = Field(min_length=1)
    source_url: str

    retrieved_at: datetime
    as_of_date: date

    verification_status: VerificationStatus
    allowed_purposes: tuple[DataPurpose, ...]
    requires_cross_check: bool = False

    checksum_sha256: str
    raw_record_count: int = Field(ge=1)

    cross_checked: bool = False
    cross_check_source_url: str | None = None

    @field_validator("source_url", "cross_check_source_url")
    @classmethod
    def validate_web_url(cls, value: str | None) -> str | None:
        """Accept only traceable HTTP or HTTPS source URLs."""

        if value is None:
            return None

        parsed_url = urlparse(str(value))

        if (
            parsed_url.scheme not in {"http", "https"}
            or not parsed_url.hostname
        ):
            raise ValueError("Source URL must be a valid HTTP or HTTPS URL.")

        return str(value)

    @field_validator("retrieved_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        """Retrieval timestamps must include an explicit timezone."""

        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("retrieved_at must be timezone-aware.")

        return value

    @field_validator("checksum_sha256")
    @classmethod
    def validate_checksum(cls, value: str) -> str:
        """A SHA-256 checksum must contain exactly 64 hexadecimal characters."""

        normalized_value = value.lower()

        if not re.fullmatch(r"[0-9a-f]{64}", normalized_value):
            raise ValueError("checksum_sha256 must be a valid SHA-256 value.")

        return normalized_value

    @model_validator(mode="after")
    def validate_provenance_relationships(self) -> "SourceProvenance":
        """Check temporal order and cross-check metadata."""

        if self.as_of_date > self.retrieved_at.date():
            raise ValueError(
                "as_of_date cannot be later than retrieved_at."
            )

        if self.cross_checked and not self.cross_check_source_url:
            raise ValueError(
                "cross_check_source_url is required when cross_checked is true."
            )

        return self
