"""Static production contracts for Docker Package 11.7."""

from __future__ import annotations

from pathlib import Path
import importlib.util


ROOT = Path(__file__).resolve().parents[1]


def text(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_fastapi_image_uses_pinned_supported_python_and_cpu_torch() -> None:
    content = text("docker/Dockerfile.fastapi")
    assert "FROM python:3.10.20-slim-bookworm" in content
    assert "https://download.pytorch.org/whl/cpu" in content
    assert "torch==2.2.2" in content


def test_images_run_as_non_root_users() -> None:
    assert "USER fni" in text("docker/Dockerfile.fastapi")
    assert "USER fni" in text("docker/Dockerfile.streamlit")


def test_images_define_healthchecks() -> None:
    assert "HEALTHCHECK" in text("docker/Dockerfile.fastapi")
    assert "HEALTHCHECK" in text("docker/Dockerfile.streamlit")


def test_compose_binds_only_to_loopback_with_long_port_syntax() -> None:
    content = text("docker/compose.yaml")
    assert 'published: "${FNI_FASTAPI_PORT:-18000}"' in content
    assert 'published: "${FNI_STREAMLIT_PORT:-18501}"' in content
    assert content.count('host_ip: "127.0.0.1"') == 2
    assert "target: 8000" in content
    assert "target: 8501" in content
    assert "0.0.0.0" not in content


def test_compose_enforces_container_hardening() -> None:
    content = text("docker/compose.yaml")
    assert content.count("read_only: true") == 2
    assert content.count("no-new-privileges:true") == 2
    assert content.count("cap_drop:") == 2
    assert content.count("- ALL") == 2
    assert content.count("pids_limit:") == 2


def test_compose_uses_project_scoped_bridge_network() -> None:
    content = text("docker/compose.yaml")
    assert "driver: bridge" in content
    assert "internal: true" not in content
    assert "http://fastapi:8000" in content


def test_compose_requires_api_authentication() -> None:
    content = text("docker/compose.yaml")
    assert "FNI_API_ENVIRONMENT: production" in content
    assert "FNI_REQUIRE_API_KEY: \"true\"" in content
    assert "FNI_API_KEY: ${FNI_API_KEY:?FNI_API_KEY is required}" in content


def test_dockerignore_excludes_private_and_secret_material() -> None:
    content = text(".dockerignore")
    for marker in (
        "data/private",
        ".env.*",
        "**/secrets.toml",
        ".strike_backups",
        ".venv-*",
        "**/*.key",
    ):
        assert marker in content


def test_artifact_registry_supports_safe_container_model_override() -> None:
    content = text("src/financial_news_intelligence/api/artifacts.py")
    assert 'os.getenv("FNI_SENTIMENT_MODEL_DIRECTORY")' in content
    assert "_safe_path(" in content
    assert "configured_model_directory.strip()" in content


def test_entrypoints_use_exec_and_validate_api_key() -> None:
    for relative in (
        "docker/entrypoint-fastapi.sh",
        "docker/entrypoint-streamlit.sh",
    ):
        content = text(relative)
        assert "set -eu" in content
        assert "${#FNI_API_KEY}" in content
        assert "exec " in content


def test_no_secret_value_is_committed_in_example_environment() -> None:
    content = text("docker/docker.env.example")
    assert "replace-with-at-least-24-random-characters" in content
    assert "ghp_" not in content
    assert "github_pat_" not in content


def test_smoke_test_covers_health_readiness_metrics_and_ui() -> None:
    content = text("docker/smoke_test.py")
    for endpoint in ("/health", "/ready", "/metrics", "/_stcore/health"):
        assert endpoint in content


def test_smoke_test_retries_transient_host_publication_failures() -> None:
    content = text("docker/smoke_test.py")
    assert "wait_for_endpoint" in content
    assert "startup_timeout_seconds" in content
    assert "urllib.error.URLError" in content
    assert "time.sleep" in content



def test_metrics_smoke_test_uses_runtime_api_key_without_cli_exposure() -> None:
    content = text("docker/smoke_test.py")
    assert 'os.environ.get("FNI_API_KEY", "")' in content
    assert 'authenticated_headers = {"X-API-Key": api_key}' in content
    assert 'headers=authenticated_headers' in content
    assert '"--api-key"' not in content



def test_rendered_port_validation_supports_compose_2_22_json_shapes() -> None:
    verifier_path = ROOT / "scripts/verify_docker_production.py"
    spec = importlib.util.spec_from_file_location("verify_docker_production", verifier_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.rendered_port_matches(
        {"target": 8000, "published": "18000", "host_ip": "127.0.0.1"},
        8000,
        18000,
    )
    assert module.rendered_port_matches(
        {"target": 8000, "published": "18000"},
        8000,
        18000,
    )
    assert module.rendered_port_matches("127.0.0.1:18000:8000/tcp", 8000, 18000)
    assert not module.rendered_port_matches("0.0.0.0:18000:8000/tcp", 8000, 18000)


def test_rendered_port_validation_uses_effective_runtime_ports() -> None:
    verifier_path = ROOT / "scripts/verify_docker_production.py"
    spec = importlib.util.spec_from_file_location("verify_docker_production_dynamic", verifier_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    environment = {
        "FNI_FASTAPI_PORT": "56724",
        "FNI_STREAMLIT_PORT": "56725",
    }
    assert module.environment_port(environment, "FNI_FASTAPI_PORT", 18000) == 56724
    fastapi_port = module.environment_port(environment, "FNI_FASTAPI_PORT", 18000)
    streamlit_port = module.environment_port(environment, "FNI_STREAMLIT_PORT", 18501)
    assert fastapi_port == 56724
    assert streamlit_port == 56725
    assert module.rendered_port_matches(
        {"target": 8000, "published": "56724", "host_ip": "127.0.0.1"},
        8000,
        fastapi_port,
    )
    assert module.rendered_port_matches(
        {"target": 8501, "published": "56725", "host_ip": "127.0.0.1"},
        8501,
        streamlit_port,
    )
    assert module.environment_port({}, "FNI_FASTAPI_PORT", 18000) == 18000
    verifier_source = verifier_path.read_text(encoding="utf-8")
    assert 'environment_port(environment, "FNI_FASTAPI_PORT", 18000)' in verifier_source
    assert 'environment_port(environment, "FNI_STREAMLIT_PORT", 18501)' in verifier_source


def test_effective_runtime_port_validation_rejects_invalid_values() -> None:
    verifier_path = ROOT / "scripts/verify_docker_production.py"
    spec = importlib.util.spec_from_file_location("verify_docker_production_invalid", verifier_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for value in ("not-a-port", "0", "65536"):
        try:
            module.environment_port({"FNI_FASTAPI_PORT": value}, "FNI_FASTAPI_PORT", 18000)
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"Invalid port value was accepted: {value}")
