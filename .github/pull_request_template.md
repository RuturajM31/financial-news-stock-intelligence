## Change summary

-

## Validation checklist

- [ ] Focused pytest suite passed locally or in CI.
- [ ] `python scripts/verify_ci_cd_security.py --project-root .` passed.
- [ ] Docker production contract remains unchanged unless the Docker package is being updated.
- [ ] Kubernetes/Helm production contract remains unchanged unless the Kubernetes/Helm package is being updated.
- [ ] Public deployment was not changed unless this pull request is the public deployment package.

## Security checklist

- [ ] No secrets, tokens, credentials, or private data were committed.
- [ ] New workflows use least-privilege permissions.
- [ ] New deployment or registry mutation steps are explicitly documented and gated.
