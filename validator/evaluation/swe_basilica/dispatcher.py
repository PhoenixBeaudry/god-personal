from __future__ import annotations

import asyncio
import logging
import os
import random
import time
import uuid
from dataclasses import dataclass
from typing import Any

import basilica
import requests

from core import constants as cst
from validator.core import constants as vcst
from validator.evaluation.swe_basilica.source import create_sglang_source
from validator.evaluation.swe_basilica.source import create_worker_source
from validator.evaluation.swe_basilica.task_cache import SweTaskCache
from validator.evaluation.utils import deployment_is_healthy
from validator.evaluation.utils import log_basilica_logs_block
from validator.evaluation.utils import wait_for_basilica_health


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SweDispatcherConfig:
    model_repo: str
    original_model: str
    base_seed: int
    temperature: float
    num_tasks: int
    task_id_min: int
    task_id_max: int
    task_timeout: int
    max_concurrency: int
    payload_extra: dict[str, Any]


def build_eval_tasks(base_seed: int, task_id_min: int, task_id_max: int, num_tasks: int) -> list[tuple[int, int]]:
    rng = random.Random(base_seed)
    population = range(task_id_min, task_id_max)
    if num_tasks <= len(population):
        task_ids = rng.sample(population, num_tasks)
    else:
        task_ids = [rng.randrange(task_id_min, task_id_max) for _ in range(num_tasks)]
    return [(task_id, task_id) for task_id in task_ids]


def _passthrough_env(*names: str) -> dict[str, str]:
    return {name: os.getenv(name, "") for name in names if os.getenv(name)}


def _deploy_kwargs(
    *,
    name: str,
    image: str,
    source: str,
    port: int,
    env: dict[str, str],
    cpu: str,
    memory: str,
    storage: bool | str = False,
    gpu_count: int | None = None,
    gpu_models: list[str] | None = None,
    min_gpu_memory_gb: int | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "name": name,
        "image": image,
        "source": source,
        "port": port,
        "cpu": cpu,
        "memory": memory,
        "storage": storage,
        "ttl_seconds": vcst.EVAL_BASILICA_TTL_SECONDS,
        "timeout": vcst.EVAL_BASILICA_TIMEOUT,
        "env": env,
    }
    if gpu_count and gpu_count > 0:
        kwargs["gpu_count"] = gpu_count
        if gpu_models:
            kwargs["gpu_models"] = gpu_models
        if min_gpu_memory_gb:
            kwargs["min_gpu_memory_gb"] = min_gpu_memory_gb
    return kwargs


async def _deploy(**kwargs):
    client = basilica.BasilicaClient(api_key=os.getenv("BASILICA_API_TOKEN") or os.getenv("BASILICA_API_KEY"))
    deployment = await asyncio.to_thread(client.deploy, **kwargs)
    return client, deployment


async def _delete_deployment(deployment, name: str) -> None:
    try:
        await asyncio.to_thread(deployment.delete)
        logger.info("Deleted Basilica deployment %s", name)
    except Exception as exc:
        logger.warning("Failed to delete Basilica deployment %s: %s", name, exc)


async def _wait_for_health(url: str, timeout: int, path: str = "/health") -> None:
    await asyncio.to_thread(wait_for_basilica_health, url, timeout, path)


def _post_json(url: str, payload: dict[str, Any], timeout: int = 60) -> dict[str, Any]:
    response = requests.post(url, json=payload, timeout=timeout)
    raw_text = response.text
    if response.status_code >= 300:
        raise RuntimeError(f"HTTP {response.status_code}: {raw_text[:1000]}")
    return response.json()


def _get_json(url: str, timeout: int = 30) -> dict[str, Any]:
    response = requests.get(url, timeout=timeout)
    raw_text = response.text
    if response.status_code >= 300:
        raise RuntimeError(f"HTTP {response.status_code}: {raw_text[:1000]}")
    return response.json()


async def _poll_worker_result(
    *,
    deployment,
    deployment_name: str,
    repo: str,
    timeout: int,
    poll_interval: int,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        await asyncio.to_thread(log_basilica_logs_block, logger, repo, deployment_name, deployment)
        payload = await asyncio.to_thread(_get_json, f"{deployment.url}/result")
        status = payload.get("status")
        if status == "completed":
            result = payload.get("result")
            if not isinstance(result, dict):
                raise RuntimeError(f"Worker completed with invalid result: {result!r}")
            return result
        if status == "failed":
            raise RuntimeError(f"Worker failed: {payload.get('error')}")
        logger.info("[%s] worker %s status=%s; polling again in %ss", repo, deployment_name, status, poll_interval)
        await asyncio.sleep(poll_interval)
    raise TimeoutError(f"Timed out waiting for SWE worker {deployment_name}")


async def _start_sglang(config: SweDispatcherConfig):
    deployment_name = f"swe-sglang-{uuid.uuid4().hex[:12]}"
    image = os.getenv("SWE_SGLANG_IMAGE") or cst.VALIDATOR_DOCKER_IMAGE_SWE
    env = {
        "MODELS": config.model_repo,
        "ORIGINAL_MODEL": config.original_model,
        "EVAL_SEED": str(config.base_seed),
        "SGLANG_PORT": "30000",
        "TRANSFORMERS_ALLOW_TORCH_LOAD": "true",
        **vcst.HF_CONTAINER_ENV,
        **_passthrough_env("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"),
    }
    logger.info("Deploying SWE SGLang server %s image=%s", deployment_name, image)
    client, deployment = await _deploy(
        **_deploy_kwargs(
            name=deployment_name,
            image=image,
            source=create_sglang_source(),
            port=30000,
            env=env,
            cpu=os.getenv("SWE_SGLANG_CPU", vcst.EVAL_BASILICA_CPU),
            memory=os.getenv("SWE_SGLANG_MEMORY", vcst.EVAL_BASILICA_MEMORY),
            storage=os.getenv("SWE_SGLANG_STORAGE", False),
            gpu_count=max(1, int(os.getenv("SWE_SGLANG_GPU_COUNT", "1"))),
            gpu_models=vcst.BASILICA_GPU_MODELS,
            min_gpu_memory_gb=vcst.BASILICA_SGLANG_MIN_GPU_MEMORY_GB,
        )
    )
    resolved_name = getattr(deployment, "name", None) or deployment_name
    await _wait_for_health(deployment.url, int(os.getenv("SWE_SGLANG_HEALTH_TIMEOUT", "1800")), "/v1/models")
    logger.info("SWE SGLang server healthy: %s -> %s", resolved_name, deployment.url)
    return client, deployment, resolved_name


def _task_image(task: dict[str, Any]) -> str:
    image = task.get("dockerhub_tag") or task.get("image") or task.get("docker_image")
    if not image:
        raise ValueError("SWE task is missing dockerhub_tag/image/docker_image")
    return str(image)


async def _run_worker(
    *,
    task: dict[str, Any],
    mode: str,
    payload: dict[str, Any],
    repo: str,
    timeout: int,
) -> dict[str, Any]:
    image = _task_image(task)
    deployment_name = f"swe-{mode}-{uuid.uuid4().hex[:12]}"
    worker_env = {
        "PYTHONUNBUFFERED": "1",
        "PORT": "8000",
        **_passthrough_env("CHUTES_API_KEY", "OPENAI_API_KEY"),
    }
    worker_gpu_count = int(os.getenv("SWE_WORKER_GPU_COUNT", "0"))
    logger.info("[%s] deploying %s worker %s image=%s", repo, mode, deployment_name, image)
    client, deployment = await _deploy(
        **_deploy_kwargs(
            name=deployment_name,
            image=image,
            source=create_worker_source(),
            port=8000,
            env=worker_env,
            cpu=os.getenv("SWE_WORKER_CPU", "4"),
            memory=os.getenv("SWE_WORKER_MEMORY", "8Gi"),
            storage=os.getenv("SWE_WORKER_STORAGE", False),
            gpu_count=worker_gpu_count,
            gpu_models=vcst.BASILICA_GPU_MODELS if worker_gpu_count else None,
            min_gpu_memory_gb=vcst.BASILICA_SGLANG_MIN_GPU_MEMORY_GB if worker_gpu_count else None,
        )
    )
    resolved_name = getattr(deployment, "name", None) or deployment_name
    try:
        if not deployment_is_healthy(deployment):
            await _wait_for_health(deployment.url, int(os.getenv("SWE_WORKER_HEALTH_TIMEOUT", "300")), "/health")
        await asyncio.to_thread(_post_json, f"{deployment.url}/run", payload)
        return await _poll_worker_result(
            deployment=deployment,
            deployment_name=resolved_name,
            repo=repo,
            timeout=timeout,
            poll_interval=int(os.getenv("SWE_WORKER_POLL_INTERVAL", "15")),
        )
    finally:
        await asyncio.to_thread(log_basilica_logs_block, logger, repo, resolved_name, deployment)
        await _delete_deployment(deployment, resolved_name)
        _ = client


async def _evaluate_single_task(
    *,
    task_id: int,
    seed: int,
    task: dict[str, Any],
    config: SweDispatcherConfig,
    sglang_base_url: str,
    task_idx: int,
    total_tasks: int,
) -> dict[str, Any]:
    task["_task_id"] = task_id
    task["task_id"] = task.get("task_id", task_id)
    repo = f"{config.model_repo}:task-{task_id}"
    model_payload = {
        "model": config.model_repo,
        "base_url": f"{sglang_base_url.rstrip('/')}/v1",
        "api_key": "test",
        "temperature": config.temperature,
    }
    agent_payload = {
        "agent": config.payload_extra.get("agent", "builtin"),
        "max_iterations": int(config.payload_extra.get("max_iterations", 100)),
        "timeout": config.task_timeout,
        "command_timeout": int(os.getenv("SWE_AGENT_COMMAND_TIMEOUT", "300")),
    }

    logger.info("SWE task %s/%s start task_id=%s seed=%s", task_idx + 1, total_tasks, task_id, seed)
    solve_payload = {
        "mode": "solve",
        "run_id": f"solve-{task_id}-{uuid.uuid4().hex[:8]}",
        "task": task,
        "model": model_payload,
        "agent": agent_payload,
        "seed": seed,
    }
    solve_result = await _run_worker(
        task=task,
        mode="solve",
        payload=solve_payload,
        repo=repo,
        timeout=config.task_timeout + 600,
    )
    patch = ((solve_result.get("extra") or {}).get("fix_patch") or "").strip()
    if patch:
        patch += "\n"
    else:
        logger.warning("SWE task_id=%s generated no patch; score=0.0", task_id)
        return {"task_id": task_id, "score": 0.0, "time": solve_result.get("time_taken", 0.0)}

    verify_payload = {
        "mode": "verify",
        "run_id": f"verify-{task_id}-{uuid.uuid4().hex[:8]}",
        "task": task,
        "patch": patch,
        "agent": agent_payload,
        "seed": seed,
    }
    verify_result = await _run_worker(
        task=task,
        mode="verify",
        payload=verify_payload,
        repo=repo,
        timeout=config.task_timeout + 600,
    )
    score = float(verify_result.get("score") or 0.0)
    latency = float(solve_result.get("time_taken") or 0.0) + float(verify_result.get("time_taken") or 0.0)
    logger.info("SWE task %s/%s done task_id=%s score=%.6f latency_s=%.3f", task_idx + 1, total_tasks, task_id, score, latency)
    return {"task_id": task_id, "score": score, "time": latency}


async def run_swe_dispatcher(config: SweDispatcherConfig) -> float:
    task_cache = SweTaskCache()
    eval_tasks = build_eval_tasks(config.base_seed, config.task_id_min, config.task_id_max, config.num_tasks)
    sglang_deployment = None
    sglang_name = None
    try:
        _client, sglang_deployment, sglang_name = await _start_sglang(config)
        semaphore = asyncio.Semaphore(config.max_concurrency)
        total = len(eval_tasks)

        async def _run_one(idx: int, seed: int, task_id: int) -> dict[str, Any]:
            async with semaphore:
                task = await asyncio.to_thread(task_cache.load, task_id)
                return await _evaluate_single_task(
                    task_id=task_id,
                    seed=seed,
                    task=task,
                    config=config,
                    sglang_base_url=sglang_deployment.url,
                    task_idx=idx,
                    total_tasks=total,
                )

        results = await asyncio.gather(
            *[asyncio.create_task(_run_one(idx, seed, task_id)) for idx, (seed, task_id) in enumerate(eval_tasks)],
            return_exceptions=True,
        )
        completed = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning("SWE task failed: %s", result, exc_info=True)
                continue
            completed.append(result)
        if not completed:
            logger.warning("SWE dispatcher completed no successful task attempts; returning 0.0")
            return 0.0
        avg = sum(float(item["score"]) for item in completed) / len(completed)
        logger.info("SWE dispatcher finished %s/%s tasks avg_score=%.6f", len(completed), total, avg)
        return avg
    finally:
        if sglang_deployment is not None:
            await asyncio.to_thread(
                log_basilica_logs_block,
                logger,
                config.model_repo,
                sglang_name or "swe-sglang",
                sglang_deployment,
            )
            await _delete_deployment(sglang_deployment, sglang_name or "swe-sglang")
