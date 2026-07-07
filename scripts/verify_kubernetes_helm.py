#!/usr/bin/env python3
"""Verify Helm source and rendered Kubernetes manifests fail closed.

The source-only mode checks the chart inventory and security defaults. When
rendered files are supplied, the verifier also inspects every Kubernetes object
that Helm produced. It never connects to a Kubernetes API server.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

REQUIRED_CHART_FILES = {
    "Chart.yaml",
    "values.yaml",
    "values.schema.json",
    "templates/_helpers.tpl",
    "templates/serviceaccount.yaml",
    "templates/fastapi-deployment.yaml",
    "templates/fastapi-service.yaml",
    "templates/streamlit-deployment.yaml",
    "templates/streamlit-service.yaml",
    "templates/networkpolicy.yaml",
    "templates/ingress.yaml",
    "templates/hpa.yaml",
    "templates/pdb.yaml",
    "templates/NOTES.txt",
}
SUPPORTED_API_KINDS = {
    ("v1", "ServiceAccount"),
    ("v1", "Service"),
    ("apps/v1", "Deployment"),
    ("networking.k8s.io/v1", "NetworkPolicy"),
    ("networking.k8s.io/v1", "Ingress"),
    ("autoscaling/v2", "HorizontalPodAutoscaler"),
    ("policy/v1", "PodDisruptionBudget"),
}


def require_mapping(value: Any, message: str) -> dict[str, Any]:
    """Return one mapping or fail with a precise structural error."""
    require(isinstance(value, dict), message)
    return value


def verify_kubernetes_structure(document: dict[str, Any]) -> None:
    """Validate the stable API identity and minimum structure of one object.

    This deliberately covers every kind the chart is allowed to emit. It is
    independent of cluster discovery and complements Helm linting plus the
    deeper workload/security checks below.
    """
    api_version = str(document.get("apiVersion", ""))
    kind = str(document.get("kind", ""))
    require((api_version, kind) in SUPPORTED_API_KINDS, f"Unsupported Kubernetes API object: {api_version}/{kind}.")

    metadata = require_mapping(document.get("metadata"), f"metadata must be an object for {kind}.")
    name = metadata.get("name")
    require(isinstance(name, str) and bool(name.strip()), f"metadata.name is required for {kind}.")
    labels = metadata.get("labels", {})
    require(isinstance(labels, dict), f"metadata.labels must be an object for {kind}/{name}.")

    if kind == "ServiceAccount":
        require(document.get("automountServiceAccountToken") is False, f"ServiceAccount token automount must be false for {name}.")
        return

    spec = require_mapping(document.get("spec"), f"spec must be an object for {kind}/{name}.")
    if kind == "Deployment":
        selector = require_mapping(spec.get("selector"), f"Deployment selector is missing for {name}.")
        match_labels = require_mapping(selector.get("matchLabels"), f"Deployment selector labels are missing for {name}.")
        template = require_mapping(spec.get("template"), f"Deployment template is missing for {name}.")
        template_metadata = require_mapping(template.get("metadata"), f"Pod template metadata is missing for {name}.")
        template_labels = require_mapping(template_metadata.get("labels"), f"Pod template labels are missing for {name}.")
        require(all(template_labels.get(key) == value for key, value in match_labels.items()), f"Deployment selector does not match pod labels for {name}.")
        require_mapping(template.get("spec"), f"Pod template spec is missing for {name}.")
    elif kind == "Service":
        require(spec.get("type") == "ClusterIP", f"Service {name} must be ClusterIP.")
        require_mapping(spec.get("selector"), f"Service selector is missing for {name}.")
        ports = spec.get("ports")
        require(isinstance(ports, list) and bool(ports), f"Service ports are missing for {name}.")
        for port in ports:
            port_mapping = require_mapping(port, f"Service port must be an object for {name}.")
            require(isinstance(port_mapping.get("port"), int), f"Service port must be an integer for {name}.")
            require("targetPort" in port_mapping, f"Service targetPort is missing for {name}.")
    elif kind == "NetworkPolicy":
        require_mapping(spec.get("podSelector"), f"NetworkPolicy podSelector is missing for {name}.")
        policy_types = spec.get("policyTypes")
        require(isinstance(policy_types, list) and set(policy_types) == {"Ingress", "Egress"}, f"NetworkPolicy policyTypes are invalid for {name}.")
        require(isinstance(spec.get("ingress"), list), f"NetworkPolicy ingress must be a list for {name}.")
        require(isinstance(spec.get("egress"), list), f"NetworkPolicy egress must be a list for {name}.")
    elif kind == "Ingress":
        rules = spec.get("rules")
        require(isinstance(rules, list) and bool(rules), f"Ingress rules are missing for {name}.")
        for rule in rules:
            http = require_mapping(require_mapping(rule, f"Ingress rule must be an object for {name}.").get("http"), f"Ingress HTTP rule is missing for {name}.")
            paths = http.get("paths")
            require(isinstance(paths, list) and bool(paths), f"Ingress paths are missing for {name}.")
            for path_rule in paths:
                backend = require_mapping(require_mapping(path_rule, f"Ingress path must be an object for {name}.").get("backend"), f"Ingress backend is missing for {name}.")
                service = require_mapping(backend.get("service"), f"Ingress service backend is missing for {name}.")
                require(bool(service.get("name")), f"Ingress service name is missing for {name}.")
                require_mapping(service.get("port"), f"Ingress service port is missing for {name}.")
    elif kind == "HorizontalPodAutoscaler":
        target = require_mapping(spec.get("scaleTargetRef"), f"HPA target is missing for {name}.")
        require(target.get("apiVersion") == "apps/v1" and target.get("kind") == "Deployment" and bool(target.get("name")), f"HPA target is invalid for {name}.")
        minimum = spec.get("minReplicas")
        maximum = spec.get("maxReplicas")
        require(isinstance(minimum, int) and isinstance(maximum, int) and 1 <= minimum <= maximum, f"HPA replica bounds are invalid for {name}.")
        require(isinstance(spec.get("metrics"), list) and bool(spec.get("metrics")), f"HPA metrics are missing for {name}.")
    elif kind == "PodDisruptionBudget":
        require_mapping(spec.get("selector"), f"PDB selector is missing for {name}.")
        availability = [key for key in ("minAvailable", "maxUnavailable") if key in spec]
        require(len(availability) == 1, f"PDB must define exactly one availability threshold for {name}.")



def require(condition: bool, message: str) -> None:
    """Raise one consistent fail-closed error when a contract is violated."""
    if not condition:
        raise RuntimeError(message)


def load_documents(path: Path) -> list[dict[str, Any]]:
    """Load all non-empty YAML documents and reject scalar/list documents."""
    documents: list[dict[str, Any]] = []
    for item in yaml.safe_load_all(path.read_text(encoding="utf-8")):
        if item is None:
            continue
        require(isinstance(item, dict), f"Non-object YAML document in {path}.")
        documents.append(item)
    return documents


def component(document: dict[str, Any]) -> str:
    """Return the application component label from one rendered object."""
    labels = document.get("metadata", {}).get("labels", {})
    return str(labels.get("app.kubernetes.io/component", ""))


def verify_deployment(document: dict[str, Any], *, production: bool) -> None:
    """Verify Pod Security Standards, probes, resources, and secret handling."""
    name = str(document.get("metadata", {}).get("name", "<unnamed>"))
    workload = component(document)
    require(workload in {"fastapi", "streamlit"}, f"Unknown Deployment component: {name}.")

    deployment_spec = document.get("spec", {})
    if production:
        require("replicas" not in deployment_spec, f"HPA-managed Deployment {name} must omit replicas.")
    else:
        require(deployment_spec.get("replicas") == 1, f"Default Deployment {name} must use one replica.")

    pod_spec = deployment_spec.get("template", {}).get("spec", {})
    require(pod_spec.get("automountServiceAccountToken") is False, f"Token automount is enabled in {name}.")
    require(bool(pod_spec.get("serviceAccountName")), f"ServiceAccount is missing in {name}.")
    require(not pod_spec.get("hostNetwork", False), f"hostNetwork is forbidden in {name}.")
    require(not pod_spec.get("hostPID", False), f"hostPID is forbidden in {name}.")
    require(not pod_spec.get("hostIPC", False), f"hostIPC is forbidden in {name}.")

    pod_security = pod_spec.get("securityContext", {})
    require(pod_security.get("runAsNonRoot") is True, f"Pod runAsNonRoot is missing in {name}.")
    require(pod_security.get("runAsUser") == 10001, f"Pod UID must be 10001 in {name}.")
    require(pod_security.get("runAsGroup") == 10001, f"Pod GID must be 10001 in {name}.")
    require(pod_security.get("fsGroup") == 10001, f"Pod fsGroup must be 10001 in {name}.")
    require(
        pod_security.get("seccompProfile", {}).get("type") == "RuntimeDefault",
        f"RuntimeDefault seccomp is missing in {name}.",
    )

    containers = pod_spec.get("containers", [])
    require(len(containers) == 1, f"Deployment {name} must have one application container.")
    container = containers[0]
    security = container.get("securityContext", {})
    require(security.get("allowPrivilegeEscalation") is False, f"Privilege escalation is not disabled in {name}.")
    require(security.get("readOnlyRootFilesystem") is True, f"Read-only root filesystem is missing in {name}.")
    require(security.get("runAsNonRoot") is True, f"Container runAsNonRoot is missing in {name}.")
    require(security.get("runAsUser") == 10001, f"Container UID must be 10001 in {name}.")
    require(security.get("runAsGroup") == 10001, f"Container GID must be 10001 in {name}.")
    require(security.get("capabilities", {}).get("drop") == ["ALL"], f"All capabilities are not dropped in {name}.")
    require(not security.get("privileged", False), f"Privileged mode is forbidden in {name}.")
    require(not container.get("ports", [{}])[0].get("hostPort"), f"hostPort is forbidden in {name}.")
    require("envFrom" not in container, f"Broad envFrom secret loading is forbidden in {name}.")

    for probe in ("startupProbe", "readinessProbe", "livenessProbe"):
        require(probe in container, f"{probe} is missing in {name}.")
    require(bool(container.get("resources", {}).get("requests")), f"Resource requests are missing in {name}.")
    require(bool(container.get("resources", {}).get("limits")), f"Resource limits are missing in {name}.")

    env = {entry.get("name"): entry for entry in container.get("env", [])}
    require("FNI_API_KEY" in env, f"FNI_API_KEY is missing in {name}.")
    secret_ref = env["FNI_API_KEY"].get("valueFrom", {}).get("secretKeyRef", {})
    require(bool(secret_ref.get("name")) and bool(secret_ref.get("key")), f"API key SecretKeyRef is invalid in {name}.")
    require("value" not in env["FNI_API_KEY"], f"Plaintext API key is forbidden in {name}.")

    volumes = pod_spec.get("volumes", [])
    require(all("hostPath" not in volume for volume in volumes), f"hostPath is forbidden in {name}.")
    tmp_volumes = [volume for volume in volumes if volume.get("name") == "tmp"]
    require(len(tmp_volumes) == 1, f"One /tmp volume is required in {name}.")
    empty_dir = tmp_volumes[0].get("emptyDir", {})
    require(empty_dir.get("medium") == "Memory", f"/tmp must be memory-backed in {name}.")
    require(bool(empty_dir.get("sizeLimit")), f"/tmp sizeLimit is missing in {name}.")
    mounts = container.get("volumeMounts", [])
    require(any(m.get("name") == "tmp" and m.get("mountPath") == "/tmp" for m in mounts), f"/tmp mount is missing in {name}.")


def verify_rendered(path: Path, *, production: bool) -> dict[str, int]:
    """Inspect resource counts, exposure, autoscaling, and workload security."""
    documents = load_documents(path)
    require(documents, f"No resources rendered in {path}.")

    kinds: dict[str, int] = {}
    identities: set[tuple[str, str]] = set()
    for document in documents:
        verify_kubernetes_structure(document)
        kind = str(document.get("kind", ""))
        name = str(document.get("metadata", {}).get("name", ""))
        identity = (kind, name)
        require(identity not in identities, f"Duplicate rendered Kubernetes object: {kind}/{name}.")
        identities.add(identity)
        kinds[kind] = kinds.get(kind, 0) + 1
        require(kind != "Secret", "The chart must never render a Secret.")

    require(kinds.get("Deployment") == 2, "Expected two Deployments.")
    require(kinds.get("Service") == 2, "Expected two Services.")
    require(kinds.get("ServiceAccount") == 1, "Expected one ServiceAccount.")
    require(kinds.get("NetworkPolicy") == 2, "Expected two NetworkPolicies.")

    for deployment in [item for item in documents if item.get("kind") == "Deployment"]:
        verify_deployment(deployment, production=production)

    services = [item for item in documents if item.get("kind") == "Service"]
    require(all(item.get("spec", {}).get("type") == "ClusterIP" for item in services), "Only ClusterIP Services are allowed.")
    require(
        all("nodePort" not in port for item in services for port in item.get("spec", {}).get("ports", [])),
        "NodePort is forbidden.",
    )

    policies = [item for item in documents if item.get("kind") == "NetworkPolicy"]
    require(all(set(item.get("spec", {}).get("policyTypes", [])) == {"Ingress", "Egress"} for item in policies), "Every NetworkPolicy must govern ingress and egress.")
    require({component(item) for item in policies} == {"fastapi", "streamlit"}, "NetworkPolicies must cover both components.")

    ingresses = [item for item in documents if item.get("kind") == "Ingress"]
    if production:
        require(len(ingresses) == 1, "Production profile must render one Ingress.")
        backend_names: list[str] = []
        for rule in ingresses[0].get("spec", {}).get("rules", []):
            for path_rule in rule.get("http", {}).get("paths", []):
                backend_names.append(path_rule.get("backend", {}).get("service", {}).get("name", ""))
        require(backend_names and all(name.endswith("-streamlit") for name in backend_names), "Ingress must route only to Streamlit.")
        require(kinds.get("HorizontalPodAutoscaler") == 2, "Production profile must render two HPAs.")
        require(kinds.get("PodDisruptionBudget") == 2, "Production profile must render two PDBs.")
    else:
        require(not ingresses, "Default profile must not render Ingress.")
        require(kinds.get("HorizontalPodAutoscaler", 0) == 0, "Default profile must not render HPAs.")
        require(kinds.get("PodDisruptionBudget", 0) == 0, "Default profile must not render PDBs.")
    return kinds


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--rendered-default", type=Path)
    parser.add_argument("--rendered-production", type=Path)
    args = parser.parse_args()

    root = args.project_root.resolve()
    chart = root / "helm/financial-news-intelligence"
    actual = {path.relative_to(chart).as_posix() for path in chart.rglob("*") if path.is_file()}
    require(actual == REQUIRED_CHART_FILES, f"Helm chart inventory mismatch: {sorted(actual ^ REQUIRED_CHART_FILES)}")

    chart_metadata = yaml.safe_load((chart / "Chart.yaml").read_text(encoding="utf-8"))
    values = yaml.safe_load((chart / "values.yaml").read_text(encoding="utf-8"))
    schema = json.loads((chart / "values.schema.json").read_text(encoding="utf-8"))
    require(chart_metadata.get("apiVersion") == "v2", "Chart apiVersion must be v2.")
    require(chart_metadata.get("version") == "12.3.0", "Chart version must be 12.3.0.")
    require(values.get("automountServiceAccountToken") is False, "Token automount default must be false.")
    require(values.get("networkPolicy", {}).get("enabled") is True, "NetworkPolicy must be enabled by default.")
    require(values.get("networkPolicy", {}).get("streamlitIngressFromAllNamespaces") is False, "Cross-namespace Streamlit ingress must be opt-in.")
    require(values.get("ingress", {}).get("enabled") is False, "Ingress must be disabled by default.")
    require(schema.get("additionalProperties") is False, "Top-level values schema must reject unknown keys.")
    for workload in ("fastapi", "streamlit"):
        service_type = schema["properties"][workload]["properties"]["service"]["properties"]["type"]
        require(service_type.get("enum") == ["ClusterIP"], f"{workload} service schema must enforce ClusterIP.")

    all_text = "\n".join(path.read_text(encoding="utf-8") for path in chart.rglob("*") if path.is_file())
    for forbidden in ("kind: Secret", "hostPath:", "privileged: true", "type: NodePort", "type: LoadBalancer", "hostPort:"):
        require(forbidden not in all_text, f"Forbidden chart source marker: {forbidden}")
    for required in ("RuntimeDefault", "readOnlyRootFilesystem", "allowPrivilegeEscalation", "capabilities:", "drop:", "FNI_API_KEY", "secretKeyRef", "/ready", "/_stcore/health", "medium: Memory"):
        require(required in all_text, f"Required chart source marker is missing: {required}")

    print("HELM CHART INVENTORY: PASSED")
    print("HELM VALUES SCHEMA: PASSED")
    print("KUBERNETES SOURCE SECURITY CONTRACT: PASSED")
    if args.rendered_default:
        default_kinds = verify_rendered(args.rendered_default.resolve(), production=False)
        print(f"DEFAULT RENDERED MANIFEST: PASSED ({sum(default_kinds.values())} resources)")
    if args.rendered_production:
        production_kinds = verify_rendered(args.rendered_production.resolve(), production=True)
        print(f"PRODUCTION RENDERED MANIFEST: PASSED ({sum(production_kinds.values())} resources)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
