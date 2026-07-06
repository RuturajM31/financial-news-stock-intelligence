# Kubernetes and Helm Runtime

This directory contains operator guidance and a secret-creation helper. The
Helm chart is under `helm/financial-news-intelligence`.

## Safe deployment sequence

1. Publish or load the Package 11.7 FastAPI and Streamlit images into the target
   cluster's image runtime.
2. Export a 24+ character API key in the shell. Do not write it to a file.
3. Run `kubernetes/create-api-secret.sh <namespace> <secret-name>`.
4. Copy `kubernetes/production-values.example.yaml` outside Git and replace the
   image tags and hostname.
5. Confirm that `streamlitIngressFromAllNamespaces` matches the ingress
   controller topology. Restrict it further when the controller is in the same
   namespace.
6. Run `helm upgrade --install` with `--create-namespace`, the approved values
   file, and an explicit namespace.
7. Confirm both Deployments, both Services, probes, NetworkPolicies, and the
   Streamlit endpoint before enabling public DNS.

The FastAPI Service is ClusterIP-only. The optional Ingress routes only to
Streamlit. The chart never creates or stores an API-key Secret.
