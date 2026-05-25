# Operations

This folder holds everything that helps run, build, inspect, or reproduce the system without being part of the validator, trainer, or shared core packages.

- `docker/` - Dockerfiles, Docker patches, and reference environment rollout functions.
- `compose/` - Docker Compose stacks for validator dependencies and observability.
- `observability/` - Grafana, Loki, Prometheus, Tempo, Vector, and Nginx config.
- `validator_ops/` - validator/auditor startup, auto-update, Grafana setup, and local evaluation tools.
- `trainer_ops/` - trainer host service files and deployment helpers.
- `auditing/` - auditor workflow scripts.
- `examples/` - local task runners for text, image, and environment training jobs.
- `tools/config/` - interactive validator, auditor, and trainer `.env` generation.
- `tools/tournament/` - tournament status, completion, and task recovery tools.
- `tools/evaluation/` - manual environment/GRPO evaluation probes and reward-function helpers.
- `tools/datasets/` - synthetic SFT dataset generation.
- `tools/observability/` - trainer log-shipping smoke tests.
- `tools/simulations/` - exploratory scoring and tournament simulations.
- `manual/` - legacy manual smoke-test entrypoints, including validator evaluation scripts.
