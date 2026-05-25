# Manual Operations

This folder holds manual smoke tests, one-off recovery scripts, and local maintenance helpers that should not be mistaken for validator or trainer runtime code.

- `validator/` contains manual validator evaluation, scoring, weight-debugging, and task inspection entrypoints.
- `environment_tournament_flow_probe.py` walks through local environment tournament creation and training assignment checks.
- `dataset_whitelist_smoke.sh` checks miner-requested dataset whitelist handling.
- `move_docker_to_ephemeral.sh` is a destructive host maintenance helper that relocates Docker storage and deletes existing Docker data.

Files in this tree intentionally avoid `test_*.py` names so plain pytest runs only collect the real tests under `tests/`.
