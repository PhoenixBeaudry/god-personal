import logging

import httpx
from tenacity import before_sleep_log
from tenacity import retry
from tenacity import stop_after_attempt
from tenacity import wait_exponential

import validator.tournament.constants as cst
from core.logging import get_logger
from core.models.payload_models import ModelPrepJob
from core.models.payload_models import TrainerProxyRequest
from core.models.payload_models import TrainerTaskLog
from core.models.utility_models import GPUInfo
from core.service_paths import GET_GPU_AVAILABILITY_ENDPOINT
from core.service_paths import MODEL_PREP_STATUS_ENDPOINT
from core.service_paths import PROXY_TRAINING_IMAGE_ENDPOINT
from core.service_paths import TASK_DETAILS_ENDPOINT


logger = get_logger(__name__)
MODEL_PREP_STATUS_TIMEOUT = 5.0  # seconds; fast checks keep orchestration cycles moving.

simple_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=4, max=10),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING),
)


def trainer_url(trainer_ip: str, path: str) -> str:
    trainer_ip_with_port = f"{trainer_ip}:8001" if ":" not in trainer_ip else trainer_ip
    return f"http://{trainer_ip_with_port}{path}"


@simple_retry
async def fetch_trainer_gpus(trainer_ip: str) -> list[GPUInfo]:
    """
    Fetch GPU availability information from a trainer.

    Args:
        trainer_ip: IP address of the trainer to contact

    Returns:
        List of GPUInfo objects from the trainer
    """
    url = trainer_url(trainer_ip, GET_GPU_AVAILABILITY_ENDPOINT)
    logger.info(f"Fetching GPU availability from trainer at {url}")

    async with httpx.AsyncClient(timeout=cst.TRAINER_HTTP_TIMEOUT) as client:
        response = await client.get(url)
        response.raise_for_status()

    gpu_data = response.json()
    gpu_infos = [GPUInfo.model_validate(gpu_info) for gpu_info in gpu_data]

    logger.info(f"Retrieved {len(gpu_infos)} GPUs from trainer {trainer_ip}")
    return gpu_infos


@simple_retry
async def start_training_task(trainer_ip: str, training_request: TrainerProxyRequest) -> bool:
    """
    Ask trainer to start training.


    Args:
        trainer_ip: IP address of the trainer
        training_request: The training request to send


    Returns:
        bool: True if training started successfully, False otherwise
    """
    try:
        # Validate the request by converting to dict and back
        validated_request = TrainerProxyRequest.model_validate(training_request.model_dump())
        logger.info("Schema validation passed for training request")
    except Exception as e:
        logger.error(f"Schema validation failed for training request: {str(e)}")
        logger.error(f"Request payload: {training_request.model_dump()}")
        return False

    url = trainer_url(trainer_ip, PROXY_TRAINING_IMAGE_ENDPOINT)
    logger.info(f"Requesting training from trainer at {url} with payload: {validated_request.model_dump()}")

    async with httpx.AsyncClient(timeout=cst.TRAINER_HTTP_TIMEOUT) as client:
        response = await client.post(url, json=validated_request.model_dump())
        response.raise_for_status()

    response_data = response.json()

    # Check for no retry flag
    if response_data.get("no_retry", False):
        logger.warning(
            f"Error cloning github repository for task {training_request.training_data.task_id} "
            f"with hotkey {training_request.hotkey}"
        )
        return cst.NO_RETRY_RESULT

    return response_data["message"] == cst.EXPECTED_TRAINING_START_MESSAGE


@simple_retry
async def get_training_task_details(trainer_ip: str, task_id: str, hotkey: str) -> TrainerTaskLog:
    """
    Get the details of a training task from a trainer.

    Args:
        trainer_ip: IP address of the trainer
        task_id: The task ID to get details for
        hotkey: The hotkey of the miner

    Returns:
        TrainerTaskLog: The task log from the trainer
    """
    url = trainer_url(trainer_ip, TASK_DETAILS_ENDPOINT.format(task_id=task_id))
    logger.debug(f"Getting task details from trainer at {url} for task {task_id}")

    async with httpx.AsyncClient(timeout=cst.TRAINER_HTTP_TIMEOUT) as client:
        response = await client.get(url, params={"hotkey": hotkey})
        response.raise_for_status()

    return TrainerTaskLog.model_validate(response.json())


async def get_model_prep_job(trainer_ip: str, task_id: str) -> ModelPrepJob | None:
    url = trainer_url(trainer_ip, MODEL_PREP_STATUS_ENDPOINT.format(task_id=task_id))
    async with httpx.AsyncClient(timeout=MODEL_PREP_STATUS_TIMEOUT) as client:
        response = await client.get(url)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return ModelPrepJob.model_validate(response.json())
