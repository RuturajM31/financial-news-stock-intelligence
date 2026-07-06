# Kubernetes and Helm Production Contract

## Scope

Package 12.3 adds a Helm v3 application chart for the Package 11.7 FastAPI and
Streamlit images. It validates the chart locally without changing a cluster,
registry, DNS record, public endpoint, or Git repository.

## Security contract

- Pods and containers run as UID/GID 10001 with `RuntimeDefault` seccomp.
- Containers drop all Linux capabilities, prohibit privilege escalation, and
  use read-only root filesystems.
- Service-account token mounting is disabled.
- Only bounded, memory-backed `/tmp` `emptyDir` volumes are writable.
- The API key comes only from a pre-existing Kubernetes Secret.
- The chart never renders a Secret, plaintext API key, `hostPath`, `hostPort`,
  privileged container, host namespace, NodePort, or LoadBalancer.
- FastAPI is ClusterIP-only; optional Ingress traffic targets Streamlit only.
- NetworkPolicies restrict FastAPI ingress to Streamlit and Streamlit egress to
  DNS and FastAPI. Cross-namespace Streamlit ingress is disabled by default and
  must be explicitly enabled for an external ingress controller.
- FastAPI internet egress is limited to TCP 80/443 when explicitly enabled;
  private and link-local IPv4 ranges are excluded.

## Reliability contract

- FastAPI uses startup `/health`, readiness `/ready`, and liveness `/health`.
- Streamlit uses `/_stcore/health` for startup, readiness, and liveness.
- Resource requests and limits are mandatory defaults.
- HPA-managed Deployments omit fixed replica counts.
- RollingUpdate, optional HPAs, optional PodDisruptionBudgets, and configurable
  topology-spread controls are included.

## Validation boundary

The strike package performs strict Helm linting for default and production
profiles and two independent template renders. kubectl is executed only for a
generator-based local client dry-run capability probe using a loopback-deny
kubeconfig. Arbitrary rendered files are not passed to `kubectl create -f`
because that code path requires API discovery even with client dry-run and
validation disabled.

Every rendered object is instead checked by a fail-closed offline verifier that
enforces the exact supported stable API/kind pairs, required Kubernetes object
structure, selector relationships, workload security controls, resource counts,
service exposure, ingress routing, autoscaling, disruption budgets, and secret
handling. The complete isolated project regression runs afterward.

The strike does not contact or mutate a Kubernetes API server. Real cluster
rollout and public ingress remain deferred until the public-deployment stage.

## Monitoring limitation

`/metrics` remains authenticated with `X-API-Key`. Cluster Prometheus wiring is
operator-specific because the scraper must inject that header from a Secret.
The chart intentionally does not expose unauthenticated metrics or install a
Prometheus Operator custom resource.
