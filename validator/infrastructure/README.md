# Validator Infrastructure

This folder is reserved for adapters that talk to infrastructure outside the validator domain model.

- `content_service.py` - signed Fiber/content-service and Nineteen.ai HTTP calls.
- `llm.py` - Nineteen.ai chat payload helpers.
- `minio_client.py` and `storage.py` - MinIO upload and presigned URL helpers.
- `substrate.py` - direct substrate queries.
- `comfy_gateway.py` - ComfyUI image generation gateway checks.
- `cache.py` - local model and dataset cache cleanup used by validator workers.
- `retries.py` - shared retry decorators for infrastructure calls.

Task preparation, task detail shaping, dataset handling, and GRPO reward-function helpers live in `validator/tasks/`.
Database normalization helpers live in `validator/db/sql/`; startup connection checks live in `validator/shared/`.
Logging lives in `core/logging.py`.
