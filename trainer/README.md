# Trainer

The trainer receives validator requests, clones a submitted training repository, runs Dockerized training/model-prep jobs on assigned GPUs, stores job state, and uploads successful outputs.

- `asgi.py` starts the FastAPI app and startup cleanup.
- `endpoints.py` defines the trainer API surface used by the validator.
- `runtime.py` starts trainer/model-prep/upload containers and handles training flow; `docker_runtime.py` owns shared Docker image, network, volume, and cleanup helpers.
- `job_state.py` persists training and model-prep job status/logs in `trainer/task_history.json`.
- `host.py` handles host-level concerns: repository cloning, GPU discovery, GPU conflict checks, W&B env paths, and container error extraction.
- `model_prep/` contains model augmentation and baseline-stat computation.
- `containers/` contains Docker entrypoints for model download, dataset cache, cache cleanup, and Hugging Face upload sidecars. Task-specific text/image training entrypoints live in external submitted repositories.
- `cleanup.py`, `telemetry.py`, `training_paths.py`, and `model_artifacts.py` hold trainer runtime helpers shared by the API process and containers.
