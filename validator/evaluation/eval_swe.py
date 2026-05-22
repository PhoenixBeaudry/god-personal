from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from core import constants as cst
from core.models.utility_models import EnvironmentDatasetType
from validator.core import constants as vcst
from validator.evaluation.swe_basilica.dispatcher import SweDispatcherConfig
from validator.evaluation.swe_basilica.dispatcher import run_swe_dispatcher
from validator.evaluation.utils import configure_eval_logging


logger = logging.getLogger(__name__)


def _environment_value(env_name: object) -> str | None:
    return getattr(env_name, "value", env_name)


def _parse_environment_name() -> cst.EnvironmentName:
    dataset_type_raw = os.getenv("DATASET_TYPE", "{}")
    env_name = os.getenv("ENVIRONMENT_NAME")

    if not env_name:
        try:
            dataset_type = EnvironmentDatasetType.model_validate_json(dataset_type_raw)
            environment_names = dataset_type.environment_names or []
            env_name = _environment_value(environment_names[0]) if environment_names else None
        except Exception:
            env_name = None

    if env_name != cst.EnvironmentName.SWE.value:
        raise ValueError(f"eval_swe invoked with environment_name={env_name!r}; expected 'swe'")
    return cst.EnvironmentName.SWE


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw not in (None, "") else default


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw not in (None, "") else default


def _build_config(model_repo: str, original_model: str) -> SweDispatcherConfig:
    env_name = _parse_environment_name()
    env_config = cst.ENVIRONMENT_CONFIGS[env_name]
    payload_extra: dict[str, Any] = dict(env_config.eval_payload_extra or {})
    base_seed = _int_env("EVAL_SEED", vcst.ENV_EVAL_DEFAULT_SEED)

    return SweDispatcherConfig(
        model_repo=model_repo,
        original_model=original_model,
        base_seed=base_seed,
        temperature=_float_env("ENV_EVAL_TEMPERATURE", vcst.ENV_EVAL_TEMPERATURE),
        num_tasks=_int_env("ENV_EVAL_NUM_SEEDS", env_config.num_seeds),
        task_id_min=_int_env("SWE_TASK_ID_MIN", env_config.task_id_min),
        task_id_max=_int_env("SWE_TASK_ID_MAX", env_config.task_id_max),
        task_timeout=_int_env("SWE_EVAL_TASK_TIMEOUT", env_config.task_timeout or vcst.ENV_EVAL_TASK_TIMEOUT),
        max_concurrency=_int_env(
            "SWE_ENV_EVAL_MAX_CONCURRENT_REQUESTS",
            env_config.max_concurrent_requests or vcst.ENV_EVAL_MAX_CONCURRENT_REQUESTS,
        ),
        payload_extra=payload_extra,
    )


async def _run() -> None:
    models_raw = os.getenv("MODELS", "")
    model_repo = models_raw.split(",")[0].strip()
    if not model_repo:
        raise ValueError("MODELS is required and must contain a single repo")

    original_model = os.getenv("ORIGINAL_MODEL", model_repo)
    config = _build_config(model_repo, original_model)
    logger.info(
        "eval_swe dispatcher start model_repo=%s original_model=%s tasks=%s task_range=(%s,%s) seed=%s concurrency=%s",
        config.model_repo,
        config.original_model,
        config.num_tasks,
        config.task_id_min,
        config.task_id_max,
        config.base_seed,
        config.max_concurrency,
    )

    avg_score = await run_swe_dispatcher(config)
    output = {model_repo: {"is_finetune": True, "eval_loss": avg_score}}
    result_path = Path(cst.CONTAINER_EVAL_RESULTS_PATH)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(output), encoding="utf-8")
    logger.info("eval_swe dispatcher wrote %s avg=%.6f", result_path, avg_score)


def main() -> int:
    configure_eval_logging()
    try:
        asyncio.run(_run())
        return 0
    except Exception as exc:
        logger.exception("eval_swe failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())

