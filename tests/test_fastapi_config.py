"""Tests for explicit FastAPI configuration validation."""

from pathlib import Path

import pytest

from financial_news_intelligence.api.config import ApiSettings


def test_settings_require_api_key_when_authentication_is_enabled(
    tmp_path: Path,
) -> None:
    """Prepare missing key, run validation, and check fail-closed behavior."""

    with pytest.raises(ValueError, match="FNI_API_KEY is required"):
        ApiSettings(project_root=tmp_path, require_api_key=True, api_key=None)


def test_settings_accept_test_mode_without_api_key(tmp_path: Path) -> None:
    """Prepare test settings, run validation, and check normalized values."""

    settings = ApiSettings(
        project_root=tmp_path,
        environment="TEST",
        require_api_key=False,
    )

    assert settings.environment == "test"
    assert settings.project_root == tmp_path.resolve()


def test_production_configuration_cannot_disable_api_authentication(
    tmp_path: Path,
) -> None:
    """Prepare unsafe production settings, validate them, and check rejection."""

    with pytest.raises(ValueError, match="cannot be disabled"):
        ApiSettings(
            project_root=tmp_path,
            environment="production",
            require_api_key=False,
        )


def test_text_limit_cannot_exceed_request_schema_boundary(tmp_path: Path) -> None:
    """Prepare an oversized text limit, validate it, and check rejection."""

    with pytest.raises(ValueError, match="hard request-schema limit"):
        ApiSettings(
            project_root=tmp_path,
            environment="test",
            require_api_key=False,
            max_text_characters=20_001,
        )



def test_readiness_probe_is_lightweight_by_default(tmp_path: Path) -> None:
    """Prepare default settings, inspect readiness mode, and check no deep probe."""

    settings = ApiSettings(
        project_root=tmp_path,
        environment="test",
        require_api_key=False,
    )

    assert settings.deep_readiness_probe is False
