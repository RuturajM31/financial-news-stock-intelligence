# Monitoring and Centralized Logging Contract

## Scope

This package closes the local application monitoring and logging stage before
Docker. It adds structured JSON logs, request correlation, recursive secret
redaction, authenticated Prometheus metrics, service-operation instrumentation,
alert-rule templates, a Grafana dashboard, focused tests, and installation
verification.

## Logging contract

- One JSON object is emitted per line.
- Request bodies, uploaded bytes, article text, API keys, provider tokens, and
  private dataset rows are never logged.
- The request ID is propagated through a context variable and appears in API and
  service logs.
- Only allow-listed operational fields are emitted.
- Exception type may be emitted; exception values and traceback text are not
  serialized by the JSON formatter.

## Metrics contract

`GET /metrics` returns Prometheus text format and uses the existing API-key
authentication dependency. Metrics contain only fixed route templates, methods,
status codes, operation names, durations, lifecycle readiness, uptime, and
in-flight counts.

The registry is process-local by design because the current verified runtime
uses one FastAPI worker. The Docker and Kubernetes stages must scrape each
replica independently and aggregate in Prometheus.

## Alert templates

The package includes alerts for API unavailability, elevated 5xx rate, p95
latency, repeated service-operation failures, and sustained request backlog.
Deployment-specific thresholds can be changed only with documented evidence.

## Limitations

This package does not start Prometheus, Grafana, FastAPI, or Streamlit as a
persistent service. Container wiring, service discovery, durable retention, and
notification channels belong to the Docker and Kubernetes/Helm stages.
