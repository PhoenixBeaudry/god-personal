# Validator Shared

This package contains code used across validator subsystems:

- `config.py` and `dependencies.py` load runtime settings and FastAPI dependencies.
- `connections.py` verifies PostgreSQL and Redis connectivity at startup.
- `constants.py` stores validator-specific constants and endpoint names.
- `models.py` and `transfer_models.py` define validator-side task, evaluation, and transfer models.
- `weight_setting.py` calculates subnet weights from task, tournament, and burn data.
- `refresh_nodes.py` keeps validator node metadata current.

Cross-service contracts that are also used by the trainer live in top-level `core/`.
Shared network/status response models live in `core.models.network_models`.
