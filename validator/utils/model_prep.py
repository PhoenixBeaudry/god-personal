"""
Dispatch model prep (augmentation + baseline stats) to a trainer with GPU.
Called during task prep, before miners are assigned.
"""

import httpx

from core.models.payload_models import ModelPrepRequest
from core.models.payload_models import ModelPrepResponse
from core.models.tournament_models import GpuRequirement
from core.models.model_prep_models import AugmentationConfig
from core.models.model_prep_models import BaselineStats
from validator.core.config import Config
from validator.core.constants import MODEL_PREP_ENDPOINT
from validator.tournament.orchestrator import _check_suitable_gpus
from validator.tournament.utils import get_tournament_gpu_requirement
from validator.utils.logging import get_logger


logger = get_logger(__name__)

MODEL_PREP_TIMEOUT_SECONDS = 600


async def dispatch_model_prep(
    model_id: str,
    training_data_url: str,
    augmentation_config: AugmentationConfig | None,
    model_params_count: int,
    task_type,
    config: Config,
    reward_functions=None,
    environment_name: str | None = None,
    env_config: dict | None = None,
) -> ModelPrepResponse | None:
    """Dispatch model prep to a trainer with GPU and wait for results.

    Returns ModelPrepResponse with augmented_model_id and baseline_stats,
    or None if no trainer is available.
    """
    # Model prep is inference-only — 1 GPU is enough for any model up to ~70B
    suitable = await _check_suitable_gpus(config, GpuRequirement.H100_1X)

    if suitable is None:
        logger.warning(f"No suitable GPUs for model prep of {model_id}, skipping")
        return None

    trainer_ip, gpu_ids = suitable

    if ":" not in trainer_ip:
        trainer_ip_with_port = f"{trainer_ip}:8001"
    else:
        trainer_ip_with_port = trainer_ip

    task_type_str = task_type.value if hasattr(task_type, "value") else str(task_type)
    request = ModelPrepRequest(
        model_id=model_id,
        training_data_url=training_data_url,
        task_type=task_type_str,
        augmentation_config=augmentation_config,
        gpu_ids=gpu_ids,
        reward_functions=reward_functions,
        environment_name=environment_name,
        task_id_min=env_config.get("task_id_range", [0, 0])[0] if env_config else None,
        task_id_max=env_config.get("task_id_range", [0, 0])[1] if env_config else None,
        env_payload_extra=env_config.get("eval_payload_extra") if env_config else None,
    )

    url = f"http://{trainer_ip_with_port}{MODEL_PREP_ENDPOINT}"
    logger.info(f"Dispatching model prep to {url}")

    try:
        async with httpx.AsyncClient(timeout=MODEL_PREP_TIMEOUT_SECONDS) as client:
            response = await client.post(url, json=request.model_dump())
            response.raise_for_status()
            result = ModelPrepResponse.model_validate(response.json())
            logger.info(
                f"Model prep complete: augmented_model_id={result.augmented_model_id}, "
                f"baseline_stats={result.baseline_stats}"
            )
            return result
    except Exception as e:
        logger.error(f"Model prep dispatch failed: {e}")
        return None
