"""Static contracts for Kubernetes and Helm Production Package 12.3."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

import yaml

ROOT = Path(__file__).resolve().parents[1]
CHART = ROOT / "helm/financial-news-intelligence"


def text(relative: str) -> str:
    """Read one payload file from either package or installed-project layout."""
    return (ROOT / relative).read_text(encoding="utf-8")


def test_chart_metadata_is_v2_and_versioned() -> None:
    chart = yaml.safe_load((CHART / "Chart.yaml").read_text(encoding="utf-8"))
    assert chart["apiVersion"] == "v2"
    assert chart["version"] == "12.3.0"
    assert chart["appVersion"] == "11.7"
    assert chart["kubeVersion"].startswith(">=1.28")


def test_values_schema_is_closed_and_clusterip_only() -> None:
    schema = json.loads((CHART / "values.schema.json").read_text(encoding="utf-8"))
    assert schema["additionalProperties"] is False
    assert "apiKey" in schema["required"]
    for workload in ("fastapi", "streamlit"):
        service_type = schema["properties"][workload]["properties"]["service"]["properties"]["type"]
        assert service_type == {"enum": ["ClusterIP"]}


def test_chart_never_templates_a_secret() -> None:
    assert not (CHART / "templates/secret.yaml").exists()
    combined = "\n".join(path.read_text(encoding="utf-8") for path in CHART.rglob("*") if path.is_file())
    assert "kind: Secret" not in combined


def test_api_key_uses_existing_secret_only() -> None:
    combined = text("helm/financial-news-intelligence/templates/fastapi-deployment.yaml") + text("helm/financial-news-intelligence/templates/streamlit-deployment.yaml")
    assert combined.count("secretKeyRef:") == 2
    assert combined.count("name: FNI_API_KEY") == 2
    assert "apiKey.existingSecret is required" in text("helm/financial-news-intelligence/templates/_helpers.tpl")


def test_pods_use_restricted_security_defaults() -> None:
    values = yaml.safe_load((CHART / "values.yaml").read_text(encoding="utf-8"))
    pod = values["podSecurityContext"]
    container = values["containerSecurityContext"]
    assert pod["runAsNonRoot"] is True
    assert pod["runAsUser"] == 10001
    assert pod["runAsGroup"] == 10001
    assert pod["fsGroup"] == 10001
    assert pod["seccompProfile"]["type"] == "RuntimeDefault"
    assert container["allowPrivilegeEscalation"] is False
    assert container["readOnlyRootFilesystem"] is True
    assert container["capabilities"]["drop"] == ["ALL"]


def test_service_account_token_is_disabled() -> None:
    values = yaml.safe_load((CHART / "values.yaml").read_text(encoding="utf-8"))
    assert values["automountServiceAccountToken"] is False
    assert "automountServiceAccountToken:" in text("helm/financial-news-intelligence/templates/serviceaccount.yaml")
    assert text("helm/financial-news-intelligence/templates/fastapi-deployment.yaml").count("automountServiceAccountToken:") == 1


def test_only_bounded_memory_emptydir_is_writable() -> None:
    combined = text("helm/financial-news-intelligence/templates/fastapi-deployment.yaml") + text("helm/financial-news-intelligence/templates/streamlit-deployment.yaml")
    assert combined.count("emptyDir:") == 2
    assert combined.count("medium: Memory") == 2
    assert combined.count("sizeLimit:") == 2
    assert "hostPath:" not in combined


def test_fastapi_probes_use_health_and_ready() -> None:
    content = text("helm/financial-news-intelligence/templates/fastapi-deployment.yaml")
    assert content.count("path: /health") == 2
    assert "path: /ready" in content
    for probe in ("startupProbe:", "readinessProbe:", "livenessProbe:"):
        assert probe in content


def test_streamlit_probes_use_streamlit_health() -> None:
    content = text("helm/financial-news-intelligence/templates/streamlit-deployment.yaml")
    assert content.count("path: /_stcore/health") == 3


def test_hpa_managed_deployments_omit_fixed_replicas() -> None:
    fastapi = text("helm/financial-news-intelligence/templates/fastapi-deployment.yaml")
    streamlit = text("helm/financial-news-intelligence/templates/streamlit-deployment.yaml")
    assert "if not .Values.autoscaling.fastapi.enabled" in fastapi
    assert "if not .Values.autoscaling.streamlit.enabled" in streamlit


def test_services_are_clusterip_only() -> None:
    values = yaml.safe_load((CHART / "values.yaml").read_text(encoding="utf-8"))
    assert values["fastapi"]["service"]["type"] == "ClusterIP"
    assert values["streamlit"]["service"]["type"] == "ClusterIP"
    combined = text("helm/financial-news-intelligence/templates/fastapi-service.yaml") + text("helm/financial-news-intelligence/templates/streamlit-service.yaml")
    assert "NodePort" not in combined and "LoadBalancer" not in combined


def test_ingress_is_disabled_by_default_and_routes_streamlit_only() -> None:
    values = yaml.safe_load((CHART / "values.yaml").read_text(encoding="utf-8"))
    assert values["ingress"]["enabled"] is False
    ingress = text("helm/financial-news-intelligence/templates/ingress.yaml")
    assert "streamlitFullname" in ingress
    assert "fastapiFullname" not in ingress


def test_network_policies_are_enabled_and_cross_namespace_ingress_is_opt_in() -> None:
    values = yaml.safe_load((CHART / "values.yaml").read_text(encoding="utf-8"))
    assert values["networkPolicy"]["enabled"] is True
    assert values["networkPolicy"]["streamlitIngressFromAllNamespaces"] is False
    policy = text("helm/financial-news-intelligence/templates/networkpolicy.yaml")
    assert policy.count("kind: NetworkPolicy") == 2
    assert "app.kubernetes.io/component: fastapi" in policy
    assert "app.kubernetes.io/component: streamlit" in policy
    assert "port: 53" in policy and "port: 8000" in policy and "port: 8501" in policy
    for private_range in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "169.254.0.0/16"):
        assert private_range in policy


def test_resource_requests_and_limits_are_defined() -> None:
    values = yaml.safe_load((CHART / "values.yaml").read_text(encoding="utf-8"))
    for workload in ("fastapi", "streamlit"):
        assert values[workload]["resources"]["requests"]
        assert values[workload]["resources"]["limits"]


def test_autoscaling_uses_stable_api_and_is_opt_in() -> None:
    values = yaml.safe_load((CHART / "values.yaml").read_text(encoding="utf-8"))
    assert values["autoscaling"]["fastapi"]["enabled"] is False
    assert values["autoscaling"]["streamlit"]["enabled"] is False
    assert text("helm/financial-news-intelligence/templates/hpa.yaml").count("apiVersion: autoscaling/v2") == 2


def test_pdb_uses_stable_api_and_is_opt_in() -> None:
    values = yaml.safe_load((CHART / "values.yaml").read_text(encoding="utf-8"))
    assert values["podDisruptionBudget"]["fastapi"]["enabled"] is False
    assert text("helm/financial-news-intelligence/templates/pdb.yaml").count("apiVersion: policy/v1") == 2


def test_secret_helper_keeps_api_key_out_of_arguments_and_output() -> None:
    content = text("kubernetes/create-api-secret.sh")
    assert "${#FNI_API_KEY}" in content
    assert "--from-file=api-key=/dev/stdin" in content
    assert "--from-literal" not in content
    assert "echo \"$FNI_API_KEY\"" not in content
    assert "printf '%s' \"$FNI_API_KEY\"" in content


def test_production_example_enables_only_streamlit_ingress() -> None:
    values = yaml.safe_load((ROOT / "kubernetes/production-values.example.yaml").read_text(encoding="utf-8"))
    assert values["ingress"]["enabled"] is True
    assert values["networkPolicy"]["streamlitIngressFromAllNamespaces"] is True


def test_documented_boundaries_defer_cluster_and_public_changes() -> None:
    content = text("docs/KUBERNETES_HELM_PRODUCTION_CONTRACT.md")
    assert "does not contact or mutate a" in content
    assert "Kubernetes API server" in content
    assert "public ingress remain deferred" in content


def test_chart_contains_no_plaintext_credential_patterns() -> None:
    combined = "\n".join(path.read_text(errors="ignore") for path in CHART.rglob("*") if path.is_file())
    for pattern in ("ghp_", "github_pat_", "sk-", "BEGIN PRIVATE KEY"):
        assert pattern not in combined


def test_package_manifest_pins_catalina_compatible_kubectl() -> None:
    manifest = json.loads(text("KUBERNETES_HELM_PACKAGE_MANIFEST.json"))
    assert manifest["version"] == "12.3"
    assert manifest["chart_version"] == "12.3.0"
    assert manifest["validation_kubectl"] == "1.30.2"
    assert manifest["validation_kubectl_go"] == "1.22.4"
    assert manifest["validation_macos_minimum"] == "10.15"


def load_verifier_module():
    """Load the installed offline verifier without requiring package layout."""
    verifier_path = ROOT / "scripts/verify_kubernetes_helm.py"
    spec = importlib.util.spec_from_file_location("fnsi_verify_kubernetes_helm", verifier_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_offline_verifier_allowlists_stable_kubernetes_apis() -> None:
    verifier = load_verifier_module()
    assert verifier.SUPPORTED_API_KINDS == {
        ("v1", "ServiceAccount"),
        ("v1", "Service"),
        ("apps/v1", "Deployment"),
        ("networking.k8s.io/v1", "NetworkPolicy"),
        ("networking.k8s.io/v1", "Ingress"),
        ("autoscaling/v2", "HorizontalPodAutoscaler"),
        ("policy/v1", "PodDisruptionBudget"),
    }


def test_offline_verifier_rejects_unknown_or_malformed_objects() -> None:
    verifier = load_verifier_module()
    with pytest.raises(RuntimeError, match="Unsupported Kubernetes API object"):
        verifier.verify_kubernetes_structure({"apiVersion": "extensions/v1beta1", "kind": "Ingress", "metadata": {"name": "bad"}, "spec": {}})
    with pytest.raises(RuntimeError, match="Service ports are missing"):
        verifier.verify_kubernetes_structure({
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {"name": "bad", "labels": {}},
            "spec": {"type": "ClusterIP", "selector": {}, "ports": []},
        })


def test_package_manifest_declares_offline_rendered_validation() -> None:
    manifest = json.loads(text("KUBERNETES_HELM_PACKAGE_MANIFEST.json"))
    assert manifest["validation_mode"] == "helm-strict-render-plus-offline-structural-security-verification"
    assert manifest["kubectl_usage"] == "generator-client-dry-run-capability-probe-only"
    assert manifest["kubectl_manifest_file_dry_run"] is False
