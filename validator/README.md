# Validator

`validator/` contains the API, background workers, scoring, and tournament orchestration for the subnet.

- `asgi.py` starts the public FastAPI application.
- `endpoints/` defines the stable HTTP API surface.
- `db/` owns database access, constants, and migrations. Keep schema and SQL changes backward-compatible.
- `lifecycle/` runs the long-lived validator worker loop that moves tasks between statuses.
- `resources/` contains static validator runtime resources such as eval prompts and local Axolotl config.
- `tasks/` prepares task data and converts tasks into trainer requests.
- `evaluation/` evaluates submitted models and computes task scores. See `evaluation/README.md` for the local module map.
- `tournament/` manages text, image, and environment tournaments. See `tournament/README.md` for the local module map.
- `infrastructure/` contains adapters for external services, object storage, substrate, cache cleanup, and retry behavior.
- `shared/` holds validator-specific config, dependencies, constants, models, and weight-setting code used across validator subsystems.

Cross-service contracts shared with the trainer live in top-level `core/`.
