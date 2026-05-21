import asyncio
import glob
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from subprocess import Popen
from urllib.parse import urlparse

import aiohttp

from core import constants as cst
from core.models.utility_models import EnvironmentDatasetType
from validator.core import constants as vcst
from validator.evaluation.eval_environment import _build_sglang_command
from validator.evaluation.eval_environment import _configure_logging
from validator.evaluation.eval_environment import _download_lora_with_retry
from validator.evaluation.eval_environment import _download_model_with_retry
from validator.evaluation.eval_environment import _merge_base_and_lora
from validator.evaluation.eval_environment import _start_process
from validator.evaluation.eval_environment import _stop_process
from validator.evaluation.eval_environment import _stream_logs
from validator.evaluation.eval_environment import _wait_for_health
from validator.evaluation.utils import check_for_lora
from validator.evaluation.utils import check_lora_has_added_tokens


logger = logging.getLogger(__name__)


def _environment_value(env_name: object) -> str | None:
    return getattr(env_name, "value", env_name)


def _parse_environment_name() -> cst.EnvironmentName:
    dataset_type_raw = os.getenv("DATASET_TYPE", "{}")
    env_name = os.getenv("ENVIRONMENT_NAME")

    if not env_name:
        try:
            dataset_type = EnvironmentDatasetType.model_validate_json(dataset_type_raw)
            env_name = _environment_value(dataset_type.environment_name)
        except Exception:
            env_name = None

    if env_name != cst.EnvironmentName.SWE.value:
        raise ValueError(f"eval_swe invoked with environment_name={env_name!r}; expected 'swe'")
    return cst.EnvironmentName.SWE


def _port_from_url(url: str, default: int = 8001) -> int:
    parsed = urlparse(url)
    return parsed.port or default


def _candidate_swe_server_commands(env_base_url: str) -> list[str]:
    port = _port_from_url(env_base_url, 8001)
    explicit = (os.getenv("SWE_ENV_SERVER_CMD") or os.getenv("ENV_SERVER_CMD") or "").strip()
    if explicit:
        return [explicit]

    uvicorn_base = f"python -m uvicorn {{module}} --host 0.0.0.0 --port {port} --workers 1 --loop asyncio"
    return [
        uvicorn_base.format(module="_affinetes.server:app"),
        uvicorn_base.format(module="server:app"),
        uvicorn_base.format(module="app:app"),
        uvicorn_base.format(module="main:app"),
        uvicorn_base.format(module="swe_infinite.server:app"),
    ]


async def _start_swe_env_server(env_base_url: str) -> tuple[Popen | None, asyncio.Task | None]:
    health_timeout = int(os.getenv("ENV_SERVER_HEALTH_TIMEOUT", "600"))
    explicit = bool((os.getenv("SWE_ENV_SERVER_CMD") or os.getenv("ENV_SERVER_CMD") or "").strip())

    for idx, command in enumerate(_candidate_swe_server_commands(env_base_url), start=1):
        logger.info("eval_setup starting SWE env-server command %s: %s", idx, command)
        proc = _start_process(command, "swe-env-server")
        log_task = asyncio.create_task(_stream_logs(proc, "swe-env-server"))
        try:
            await _wait_for_health(
                env_base_url,
                os.getenv("ENV_SERVER_HEALTH_PATH", "/health"),
                health_timeout if explicit else min(health_timeout, 60),
                service_name="swe-env-server",
            )
            return proc, log_task
        except Exception as exc:
            _stop_process(proc, "swe-env-server")
            log_task.cancel()
            if explicit:
                raise
            logger.warning("SWE env-server candidate failed: %s", exc)

    raise RuntimeError("Unable to start SWE env-server; set SWE_ENV_SERVER_CMD for this image")


async def _prepare_sglang_for_model(model_repo: str, original_model: str, base_seed: int) -> tuple[str, str, str]:
    t_det = time.time()
    is_lora = await asyncio.to_thread(check_for_lora, model_repo, False)
    should_merge_lora = False
    if is_lora:
        should_merge_lora = await asyncio.to_thread(check_lora_has_added_tokens, model_repo, False)
    logger.info(
        "eval_setup LoRA detection in %.2fs: is_lora=%s merge_lora_to_base=%s",
        time.time() - t_det,
        is_lora,
        should_merge_lora,
    )

    sglang_command = os.getenv("SGLANG_START_CMD")
    if sglang_command:
        logger.info("eval_setup SGLang: using SGLANG_START_CMD from environment")
        return model_repo, model_repo, sglang_command

    if is_lora and not should_merge_lora:
        logger.info("eval_setup model path: LoRA + SGLang native (base=%s lora_repo=%s)", original_model, model_repo)
        model_path_for_sglang = await asyncio.to_thread(_download_model_with_retry, original_model)
        lora_dir = "/lora/trained_lora"
        await asyncio.to_thread(_download_lora_with_retry, model_repo, lora_dir)
        for model_file in glob.glob(os.path.join(lora_dir, "model-*.safetensors")):
            try:
                os.remove(model_file)
                logger.info("Removed incompatible LoRA file: %s", os.path.basename(model_file))
            except Exception as exc:
                logger.warning("Failed to remove %s: %s", model_file, exc)
        index_file = os.path.join(lora_dir, "model.safetensors.index.json")
        if os.path.exists(index_file):
            try:
                os.remove(index_file)
            except Exception as exc:
                logger.warning("Failed to remove index file: %s", exc)
        inference_model_name = f"{original_model}:trained_lora"
        sglang_command = (
            _build_sglang_command(model_path_for_sglang, base_seed)
            + " --enable-lora --lora-paths trained_lora=/lora/trained_lora --lora-backend triton"
        )
        return inference_model_name, model_path_for_sglang, sglang_command

    if is_lora and should_merge_lora:
        logger.info("eval_setup model path: merge LoRA into base then SGLang (base=%s lora=%s)", original_model, model_repo)
        base_path = await asyncio.to_thread(_download_model_with_retry, original_model)
        lora_temp_dir = "/tmp/lora/trained_lora"
        await asyncio.to_thread(_download_lora_with_retry, model_repo, lora_temp_dir)
        model_path_for_sglang = await asyncio.to_thread(_merge_base_and_lora, base_path, lora_temp_dir)
        return model_repo, model_path_for_sglang, _build_sglang_command(model_path_for_sglang, base_seed)

    logger.info("eval_setup model path: single HF repo (full weights) repo=%s", model_repo)
    model_path_for_sglang = await asyncio.to_thread(_download_model_with_retry, model_repo)
    return model_repo, model_path_for_sglang, _build_sglang_command(model_path_for_sglang, base_seed)


def _build_eval_tasks(base_seed: int, task_id_min: int, task_id_max: int, num_tasks: int) -> list[tuple[int, int]]:
    rng = random.Random(base_seed)
    population = range(task_id_min, task_id_max)
    if num_tasks <= len(population):
        task_ids = rng.sample(population, num_tasks)
    else:
        task_ids = [rng.randrange(task_id_min, task_id_max) for _ in range(num_tasks)]
    return [(task_id, task_id) for task_id in task_ids]


async def _run_swe_evaluation(
    *,
    sglang_url: str,
    env_url: str,
    eval_tasks: list[tuple[int, int]],
    inference_model_name: str,
    temperature: float,
    payload_extra: dict,
    task_timeout: int,
    max_concurrency: int,
) -> float:
    total_tasks = len(eval_tasks)
    all_results: list[dict] = []
    semaphore = asyncio.Semaphore(max_concurrency)

    async def post_json(session: aiohttp.ClientSession, path: str, payload: dict) -> dict:
        timeout = aiohttp.ClientTimeout(total=task_timeout + 120)
        async with session.post(
            f"{env_url}{path}",
            json=payload,
            timeout=timeout,
            headers={"Connection": "close"},
        ) as response:
            raw_text = await response.text()
            if response.status != 200:
                detail = f": {raw_text[:500]}" if raw_text else ""
                raise RuntimeError(f"HTTP {response.status}{detail}")
            return json.loads(raw_text)

    async def call_evaluate(session: aiohttp.ClientSession, payload: dict) -> dict:
        preferred_path = os.getenv("SWE_ENV_EVALUATE_PATH", "/call")
        preferred_payload = (
            {"method": "evaluate", "args": [], "kwargs": payload}
            if preferred_path == "/call"
            else payload
        )
        try:
            response_data = await post_json(session, preferred_path, preferred_payload)
        except RuntimeError as exc:
            if preferred_path == "/evaluate" or ("HTTP 404" not in str(exc) and "HTTP 405" not in str(exc)):
                raise
            response_data = await post_json(session, "/evaluate", payload)

        if not isinstance(response_data, dict):
            raise RuntimeError(f"Unexpected SWE response type: {type(response_data).__name__}")

        status = response_data.get("status")
        if status == "failed":
            detail = response_data.get("error") or response_data.get("result") or response_data
            raise RuntimeError(f"SWE env evaluate failed: {detail}")
        if status == "success" and "result" in response_data:
            result = response_data["result"]
            if not isinstance(result, dict):
                raise RuntimeError(f"Unexpected SWE result type: {type(result).__name__}")
            return result
        return response_data

    async def evaluate_single_task(session: aiohttp.ClientSession, seed: int, task_id: int, task_idx: int) -> dict:
        payload = {
            "task_id": task_id,
            "model": inference_model_name,
            "base_url": f"{sglang_url}/v1",
            "api_key": "test",
            "timeout": task_timeout,
            "temperature": temperature,
            "seed": seed,
            **payload_extra,
        }
        start_ts = time.time()
        try:
            logger.info("eval_progress %s/%s start task_id=%s seed=%s", task_idx + 1, total_tasks, task_id, seed)
            result = await call_evaluate(session, payload)
            score = float(result.get("score", 0.0))
            latency = result.get("time_taken", time.time() - start_ts)
            logger.info(
                "eval_progress %s/%s done task_id=%s score=%.6f latency_s=%.3f",
                task_idx + 1,
                total_tasks,
                task_id,
                score,
                latency,
            )
            return {"task_id": task_id, "score": score, "time": latency}
        except Exception as exc:
            logger.warning(
                "eval_progress %s/%s failed task_id=%s: %s; score=0.0",
                task_idx + 1,
                total_tasks,
                task_id,
                exc,
                exc_info=True,
            )
            return {"task_id": task_id, "score": 0.0, "time": 0.0}

    async def evaluate_with_semaphore(session: aiohttp.ClientSession, seed: int, task_id: int, task_idx: int) -> dict:
        async with semaphore:
            return await evaluate_single_task(session, seed, task_id, task_idx)

    session_timeout = aiohttp.ClientTimeout(total=vcst.ENV_EVAL_SESSION_TIMEOUT)
    async with aiohttp.ClientSession(timeout=session_timeout) as session:
        logger.info("eval_progress batch: %s SWE tasks (concurrency=%s)", total_tasks, max_concurrency)
        tasks = [
            asyncio.create_task(evaluate_with_semaphore(session, seed, task_id, idx))
            for idx, (seed, task_id) in enumerate(eval_tasks)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                logger.warning("eval_progress task raised exception: %s", result, exc_info=True)
            elif isinstance(result, dict):
                all_results.append(result)

    if not all_results:
        logger.warning("eval_progress batch: no completed SWE task results; returning 0.0")
        return 0.0
    avg = sum(r["score"] for r in all_results) / len(all_results)
    logger.info("eval_progress batch: finished %s/%s tasks, avg_score=%.6f", len(all_results), total_tasks, avg)
    return avg


async def _run() -> None:
    env_proc = None
    sglang_proc = None
    env_log_task = None
    sglang_log_task = None

    try:
        logger.info("eval_swe: start pid=%s EVAL_LOG_LEVEL=%s", os.getpid(), os.getenv("EVAL_LOG_LEVEL", "INFO"))

        models_raw = os.getenv("MODELS", "")
        model_repo = models_raw.split(",")[0].strip()
        if not model_repo:
            raise ValueError("MODELS is required and must contain a single repo")

        original_model = os.getenv("ORIGINAL_MODEL", model_repo)
        base_seed = int(os.getenv("EVAL_SEED", str(vcst.ENV_EVAL_DEFAULT_SEED)))
        temperature = float(os.getenv("ENV_EVAL_TEMPERATURE", str(vcst.ENV_EVAL_TEMPERATURE)))

        env_name = _parse_environment_name()
        env_config = cst.ENVIRONMENT_CONFIGS[env_name]
        task_id_min = env_config.task_id_min
        task_id_max = env_config.task_id_max
        num_tasks = int(os.getenv("ENV_EVAL_NUM_SEEDS", str(env_config.num_seeds)))
        payload_extra = dict(env_config.eval_payload_extra)
        task_timeout = int(os.getenv("SWE_EVAL_TASK_TIMEOUT", str(env_config.task_timeout or vcst.ENV_EVAL_TASK_TIMEOUT)))
        max_concurrency = int(
            os.getenv(
                "SWE_ENV_EVAL_MAX_CONCURRENT_REQUESTS",
                str(env_config.max_concurrent_requests or vcst.ENV_EVAL_MAX_CONCURRENT_REQUESTS),
            )
        )
        eval_tasks = _build_eval_tasks(base_seed, task_id_min, task_id_max, num_tasks)

        logger.info(
            "eval_setup config: env=%s num_tasks=%s task_id_range=(%s,%s) model_repo=%s original_model=%s "
            "eval_seed=%s temperature=%s task_timeout=%s",
            env_name,
            num_tasks,
            task_id_min,
            task_id_max,
            model_repo,
            original_model,
            base_seed,
            temperature,
            task_timeout,
        )

        inference_model_name, model_path_for_sglang, sglang_command = await _prepare_sglang_for_model(
            model_repo,
            original_model,
            base_seed,
        )

        sglang_health_timeout = int(os.getenv("SGLANG_HEALTH_TIMEOUT", "1800"))
        env_base_url = os.getenv("ENV_SERVER_BASE_URL", "http://127.0.0.1:8001")
        sglang_base_url = os.getenv("SGLANG_BASE_URL", "http://127.0.0.1:30000")

        logger.info(
            "eval_setup launching SGLang: model_path_for_sglang=%s inference_model_name=%s",
            model_path_for_sglang,
            inference_model_name,
        )
        logger.info("eval_setup SGLang command: %s", sglang_command)

        min_ws = vcst.SGLANG_FLASHINFER_WORKSPACE_MIN_BYTES
        try:
            cur_ws = int(os.environ.get("SGLANG_FLASHINFER_WORKSPACE_SIZE", "0") or "0")
        except ValueError:
            cur_ws = 0
        if cur_ws < min_ws:
            os.environ["SGLANG_FLASHINFER_WORKSPACE_SIZE"] = str(min_ws)

        sglang_proc = _start_process(sglang_command, "sglang")
        sglang_log_task = asyncio.create_task(_stream_logs(sglang_proc, "sglang"))
        await _wait_for_health(
            sglang_base_url,
            os.getenv("SGLANG_HEALTH_PATH", "/v1/models"),
            sglang_health_timeout,
            service_name="SGLang",
        )

        env_proc, env_log_task = await _start_swe_env_server(env_base_url)

        avg_score = await _run_swe_evaluation(
            sglang_url=sglang_base_url,
            env_url=env_base_url,
            eval_tasks=eval_tasks,
            inference_model_name=inference_model_name,
            temperature=temperature,
            payload_extra=payload_extra,
            task_timeout=task_timeout,
            max_concurrency=max_concurrency,
        )

        output = {model_repo: {"is_finetune": True, "eval_loss": avg_score}}
        result_path = Path(cst.CONTAINER_EVAL_RESULTS_PATH)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(output), encoding="utf-8")
        logger.info("eval_swe: wrote %s tasks=%s avg=%.6f", result_path, len(eval_tasks), avg_score)
    finally:
        _stop_process(env_proc, "swe-env-server")
        _stop_process(sglang_proc, "sglang")
        if env_log_task:
            env_log_task.cancel()
        if sglang_log_task:
            sglang_log_task.cancel()


def main() -> int:
    _configure_logging()
    try:
        asyncio.run(_run())
        return 0
    except Exception as exc:
        logger.exception("eval_swe failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
