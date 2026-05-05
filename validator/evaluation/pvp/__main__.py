"""PvP evaluation container entry point.

Loads config, starts two SGLang instances (one per GPU),
runs all matchups, writes results JSON.

Usage: python -m validator.evaluation.pvp
"""

import asyncio
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

from core.constants import EnvironmentName
from core.models.pvp_models import (
    ChatCompletionConfig,
    PvPEnvironmentResult,
    PvPEvalConfig,
    PvPEvalMetadata,
    PvPEvalResults,
    PvPModelSpec,
)
from validator.core import constants as vcst
from validator.evaluation.eval_environment import (
    _stop_process,
    _wait_for_health,
)
from validator.evaluation.pvp.game_runner import run_matchup

logger = logging.getLogger(__name__)


def main() -> int:
    _configure_logging()
    try:
        config = _load_config()
        results = _run(config)
        _write_results(results)
        return 0
    except Exception as exc:
        logger.exception("PvP evaluation failed: %s", exc)
        return 1


def _configure_logging() -> None:
    level = os.getenv(vcst.PVP_LOG_LEVEL_ENV_VAR, "INFO").upper()
    logging.basicConfig(level=level, format=vcst.PVP_LOG_FORMAT, stream=sys.stderr)


def _load_config() -> PvPEvalConfig:
    """Load config from env var or mounted file."""
    config_raw = os.getenv(vcst.PVP_CONFIG_ENV_VAR)
    if config_raw:
        return PvPEvalConfig.model_validate_json(config_raw)

    config_path = Path(vcst.PVP_CONFIG_PATH)
    if config_path.exists():
        return PvPEvalConfig.model_validate_json(config_path.read_text())

    raise ValueError(
        f"No config found. Set {vcst.PVP_CONFIG_ENV_VAR} env var or mount {vcst.PVP_CONFIG_PATH}"
    )


def _resolve_spec(spec: PvPModelSpec, default_gpu: int, default_port: int) -> tuple[int, int]:
    """Apply defaults to GPU and port if not explicitly set."""
    gpu = spec.gpu_id if spec.gpu_id is not None else default_gpu
    port = spec.port if spec.port is not None else default_port
    return gpu, port


def _build_chat_config(port: int, eval_config: PvPEvalConfig, model_repo: str) -> ChatCompletionConfig:
    """Construct ChatCompletionConfig from resolved port and eval settings."""
    return ChatCompletionConfig(
        model=model_repo,
        base_url=f"http://{vcst.PVP_SGLANG_HOST}:{port}{vcst.PVP_SGLANG_API_PATH}",
        temperature=eval_config.temperature,
        seed=eval_config.seed,
    )


def _run(config: PvPEvalConfig) -> PvPEvalResults:
    """Start servers, run all matchups, return results."""
    start_time = time.time()

    gpu_a, port_a = _resolve_spec(config.model_a, default_gpu=0, default_port=vcst.PVP_SGLANG_PORT_A)
    gpu_b, port_b = _resolve_spec(config.model_b, default_gpu=1, default_port=vcst.PVP_SGLANG_PORT_B)

    sglang_a: subprocess.Popen | None = None
    sglang_b: subprocess.Popen | None = None

    try:
        sglang_a = _start_sglang(config.model_a.repo, gpu_a, port_a)
        sglang_b = _start_sglang(config.model_b.repo, gpu_b, port_b)
        asyncio.run(_wait_for_servers(port_a, port_b))

        config_a = _build_chat_config(port_a, config, config.model_a.repo)
        config_b = _build_chat_config(port_b, config, config.model_b.repo)

        env_results: dict[EnvironmentName, PvPEnvironmentResult] = {}
        for env_name, matchup_config in config.matchups.items():
            logger.info("Starting matchup: %s (%d seeds)", env_name.value, matchup_config.num_games)
            env_results[env_name] = run_matchup(
                env_name=env_name,
                matchup_config=matchup_config,
                config_a=config_a,
                config_b=config_b,
                base_seed=config.seed,
            )

        return PvPEvalResults(
            model_a=config.model_a.repo,
            model_b=config.model_b.repo,
            results=env_results,
            metadata=PvPEvalMetadata(
                seed=config.seed,
                temperature=config.temperature,
                wall_time_seconds=time.time() - start_time,
            ),
        )
    finally:
        _stop_process(sglang_a, "sglang-a")
        _stop_process(sglang_b, "sglang-b")


def _build_sglang_command(model_path: str, port: int, seed: int) -> str:
    """Build SGLang launch command with explicit port."""
    tensor_parallel = os.getenv("SGLANG_TENSOR_PARALLEL_SIZE", "1")
    dtype = os.getenv("SGLANG_DTYPE", "float16")
    extra = (os.getenv("SGLANG_ENV_EVAL_EXTRA_CLI") or vcst.SGLANG_ENV_EVAL_EXTRA_CLI).strip()

    cmd = (
        "python3 -m sglang.launch_server "
        f"--model-path {model_path} "
        f"--host 0.0.0.0 --port {port} "
        f"--tensor-parallel-size {tensor_parallel} "
        f"--dtype {dtype} "
        f"--enable-deterministic-inference --random-seed {seed}"
    )
    return f"{cmd} {extra}" if extra else cmd


def _start_sglang(model_path: str, gpu_id: int, port: int) -> subprocess.Popen:
    """Start an SGLang server on the specified GPU and port."""
    cmd = _build_sglang_command(model_path, port=port, seed=42)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
    logger.info("Starting SGLang on GPU %d port %d", gpu_id, port)
    return subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        preexec_fn=os.setsid,
        env=env,
    )


async def _wait_for_servers(port_a: int, port_b: int) -> None:
    """Wait for both SGLang instances to become healthy."""
    await asyncio.gather(
        _wait_for_health(
            f"http://{vcst.PVP_SGLANG_HOST}:{port_a}",
            vcst.PVP_SGLANG_HEALTH_PATH,
            vcst.PVP_SGLANG_HEALTH_TIMEOUT,
            service_name="sglang-a",
        ),
        _wait_for_health(
            f"http://{vcst.PVP_SGLANG_HOST}:{port_b}",
            vcst.PVP_SGLANG_HEALTH_PATH,
            vcst.PVP_SGLANG_HEALTH_TIMEOUT,
            service_name="sglang-b",
        ),
    )


def _write_results(results: PvPEvalResults) -> None:
    results_path = Path(vcst.PVP_RESULTS_PATH)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(results.model_dump_json(indent=2))
    logger.info("Results written to %s", results_path)


if __name__ == "__main__":
    sys.exit(main())
