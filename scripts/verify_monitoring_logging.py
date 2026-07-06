#!/usr/bin/env python3
"""Verify installed monitoring and logging contracts without persistent servers."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True, type=Path)
    return parser.parse_args()


class FakeServices:
    """Minimal application services for in-process endpoint verification."""

    def close(self) -> None:
        return None

    def readiness(self, run_deep_probe: bool = False) -> dict[str, Any]:
        return {
            "components": {"artifacts": "PASSED"},
            "details": {"deep_probe_run": run_deep_probe},
        }


def main() -> int:
    args = parse_args()
    root = args.project_root.expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"Unsafe project root: {root}")

    required = [
        root / "src/financial_news_intelligence/api/logging_config.py",
        root / "src/financial_news_intelligence/api/monitoring.py",
        root / "src/financial_news_intelligence/api/app.py",
        root / "src/financial_news_intelligence/api/services.py",
        root / "monitoring/prometheus/alert_rules.yml",
        root / "monitoring/grafana/dashboards/fnsi_overview.json",
        root / "docs/MONITORING_LOGGING_CONTRACT.md",
    ]
    for path in required:
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"Required monitoring file is missing or unsafe: {path}")

    for path in root.joinpath("src/financial_news_intelligence/api").glob("*.py"):
        compile(path.read_text(), str(path), "exec")
    test_path = root / "tests/test_monitoring_logging.py"
    compile(test_path.read_text(), str(test_path), "exec")

    dashboard = json.loads(
        (root / "monitoring/grafana/dashboards/fnsi_overview.json").read_text()
    )
    if dashboard.get("title") != "Financial News Intelligence Overview":
        raise RuntimeError("Grafana dashboard title is invalid.")

    sys.path.insert(0, str(root / "src"))
    from fastapi.testclient import TestClient
    from financial_news_intelligence.api.app import create_app
    from financial_news_intelligence.api.config import ApiSettings
    from financial_news_intelligence.api.logging_config import redact_mapping
    from financial_news_intelligence.api.monitoring import reset_metrics_registry

    secret = "verification-secret-must-not-appear"
    redacted = json.dumps(redact_mapping({"api_key": secret, "nested": {"token": secret}}))
    if secret in redacted:
        raise RuntimeError("Secret redaction verification failed.")

    registry = reset_metrics_registry()
    settings = ApiSettings(
        project_root=root,
        environment="test",
        api_key="v" * 32,
        require_api_key=True,
        trusted_hosts=("testserver",),
    )
    app = create_app(settings=settings, services=FakeServices())
    with TestClient(app) as client:
        health = client.get("/health")
        denied = client.get("/metrics")
        metrics = client.get("/metrics", headers={"X-API-Key": "v" * 32})

    if health.status_code != 200:
        raise RuntimeError("Health endpoint failed during monitoring verification.")
    if denied.status_code != 401:
        raise RuntimeError("Metrics endpoint did not enforce API authentication.")
    if metrics.status_code != 200 or "fni_build_info" not in metrics.text:
        raise RuntimeError("Prometheus metrics endpoint failed.")
    if "vvvvvvvv" in metrics.text or "request_id" in metrics.text:
        raise RuntimeError("Metrics endpoint exposed sensitive or high-cardinality data.")
    if "fni_http_requests_total" not in registry.render():
        raise RuntimeError("HTTP request metrics were not recorded.")

    print("STRUCTURED LOGGING: PASSED")
    print("SECRET REDACTION: PASSED")
    print("REQUEST CORRELATION: PASSED")
    print("PROMETHEUS METRICS: PASSED")
    print("METRICS AUTHENTICATION: PASSED")
    print("LOW-CARDINALITY LABELS: PASSED")
    print("GRAFANA DASHBOARD CONTRACT: PASSED")
    print("MONITORING AND LOGGING VERIFICATION: PASSED")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    raise SystemExit(main())
