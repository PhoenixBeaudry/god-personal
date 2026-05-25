"""Stable HTTP path contracts shared by validator, trainer, and miners."""

# External trainer and tournament participant endpoint paths.
START_TRAINING_ENDPOINT = "/start_training/"
START_TRAINING_IMAGE_ENDPOINT = "/start_training_image/"
START_TRAINING_GRPO_ENDPOINT = "/start_training_grpo/"
TRAINING_REPO_ENDPOINT = "/training_repo"

# Trainer API endpoints called by the validator.
PROXY_TRAINING_IMAGE_ENDPOINT = "/v1/trainer/start_training"
MODEL_PREP_ENDPOINT = "/v1/trainer/model_prep"
MODEL_PREP_STATUS_ENDPOINT = "/v1/trainer/model_prep/{task_id}"
GET_GPU_AVAILABILITY_ENDPOINT = "/v1/trainer/get_gpu_availability"
TASK_DETAILS_ENDPOINT = "/v1/trainer/{task_id}"
GET_RECENT_TASKS_ENDPOINT = "/v1/trainer/get_recent_tasks"
