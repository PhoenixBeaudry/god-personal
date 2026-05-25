import asyncio
import uuid

import docker
from docker.errors import APIError
from docker.errors import BuildError

from core.logging import stream_image_build_logs
from trainer import constants as cst
from trainer.telemetry import logger


# logger = get_logger(__name__)

def ensure_internal_network(name: str = cst.INTERNAL_BRIDGE_NAME):
    client = docker.from_env()
    try:
        client.networks.get(name)
    except docker.errors.NotFound:
        client.networks.create(name, driver="bridge", internal=True)


def calculate_container_resources(gpu_ids: list[int]) -> tuple[str, int]:
    """Calculate memory limit and CPU limit based on GPU count.

    Returns:
        tuple: (memory_limit_str, cpu_limit_nanocpus)
    """
    num_gpus = len(gpu_ids)
    memory_limit = f"{num_gpus * cst.MEMORY_PER_GPU_GB}g"
    cpu_limit_nanocpus = num_gpus * cst.CPUS_PER_GPU * 1_000_000_000

    logger.info(f"Allocating resources for {num_gpus} GPUs: {memory_limit} memory, {num_gpus * cst.CPUS_PER_GPU} CPUs")
    return memory_limit, cpu_limit_nanocpus


def build_docker_image(
    dockerfile_path: str,
    log_labels: dict[str, str] | None = None,
    context_path: str = ".",
    is_image_task: bool = False,
    tag: str = None,
    no_cache: bool = True,
) -> tuple[str, str | None]:
    client: docker.DockerClient = docker.from_env()

    if tag is None:
        tag = f"standalone-image-trainer:{uuid.uuid4()}" if is_image_task else f"standalone-text-trainer:{uuid.uuid4()}"

    logger.info(f"Building Docker image '{tag}'...", extra=log_labels)

    try:
        build_output = client.api.build(
            path=context_path,
            dockerfile=dockerfile_path,
            tag=tag,
            nocache=no_cache,
            decode=True,
        )
        stream_image_build_logs(build_output, logger=logger, log_context=log_labels)

        logger.info("Docker image built successfully.", extra=log_labels)
        return tag, None
    except (BuildError, APIError) as e:
        logger.error(f"Docker build failed: {str(e)}", extra=log_labels)
        return None, str(e)


def delete_image_and_cleanup(tag: str):
    client = docker.from_env()
    try:
        client.images.remove(image=tag, force=True)
        logger.info(f"Deleted Docker image with tag: {tag}")
    except docker.errors.ImageNotFound:
        logger.error(f"No Docker image found with tag: {tag}")
    except Exception as e:
        logger.error(f"Failed to delete image '{tag}': {e}")

    try:
        client.images.prune(filters={"dangling": True})
        client.api.prune_builds()
        logger.info("Cleaned up dangling images and build cache.")
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")


def _create_volumes_sync():
    client: docker.DockerClient = docker.from_env()
    volume_names = cst.VOLUME_NAMES
    for volume_name in volume_names:
        try:
            client.volumes.get(volume_name)
        except docker.errors.NotFound:
            client.volumes.create(name=volume_name)
            logger.info(f"Volume '{volume_name}' created.")


async def create_volumes_if_dont_exist():
    await asyncio.to_thread(_create_volumes_sync)
