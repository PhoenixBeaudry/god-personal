"""PvP evaluation container entry point.

Loads config, starts two SGLang instances (one per GPU),
runs all matchups, writes results JSON.

Usage: python -m validator.evaluation.pvp
"""

import asyncio
import glob
import logging
import os
import subprocess
import sys
import threading
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
    _download_lora_with_retry,
    _download_model_with_retry,
    _merge_base_and_lora,
    _stop_process,
    _wait_for_health,
)
from validator.evaluation.utils import check_for_lora, check_lora_has_added_tokens
from validator.evaluation.pvp.chat import create_client
from validator.evaluation.pvp.game_runner import Player, run_matchup

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


def _build_chat_config(port: int, eval_config: PvPEvalConfig, model_name: str) -> ChatCompletionConfig:
    """Construct ChatCompletionConfig from resolved port and eval settings."""
    return ChatCompletionConfig(
        model=model_name,
        base_url=f"http://{vcst.PVP_SGLANG_HOST}:{port}{vcst.PVP_SGLANG_API_PATH}",
        temperature=eval_config.temperature,
        seed=eval_config.seed,
    )


def _run(config: PvPEvalConfig) -> PvPEvalResults:
    """Prepare models, start servers, run all matchups, return results."""
    start_time = time.time()

    gpu_a, port_a = _resolve_spec(config.model_a, default_gpu=0, default_port=vcst.PVP_SGLANG_PORT_A)
    gpu_b, port_b = _resolve_spec(config.model_b, default_gpu=1, default_port=vcst.PVP_SGLANG_PORT_B)

    # Prepare models (download, detect LoRA, merge if needed)
    model_path_a, model_name_a, sglang_extra_a = _prepare_model(config.model_a, "a")
    model_path_b, model_name_b, sglang_extra_b = _prepare_model(config.model_b, "b")

    sglang_a: subprocess.Popen | None = None
    sglang_b: subprocess.Popen | None = None
    player_a: Player | None = None
    player_b: Player | None = None

    try:
        sglang_a = _start_sglang(model_path_a, gpu_a, port_a, config.seed, sglang_extra_a)
        sglang_b = _start_sglang(model_path_b, gpu_b, port_b, config.seed + 1, sglang_extra_b)
        asyncio.run(_wait_for_servers(port_a, port_b))

        config_a = _build_chat_config(port_a, config, model_name_a)
        config_b = _build_chat_config(port_b, config, model_name_b)

        player_a = Player(client=create_client(config_a), config=config_a)
        player_b = Player(client=create_client(config_b), config=config_b)

        env_results: dict[EnvironmentName, PvPEnvironmentResult] = {}
        for env_name, matchup_config in config.matchups.items():
            logger.info("Starting matchup: %s (%d seeds)", env_name.value, matchup_config.num_games)
            env_results[env_name] = run_matchup(
                env_name=env_name,
                matchup_config=matchup_config,
                player_a=player_a,
                player_b=player_b,
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
        if player_a:
            player_a.client.close()
        if player_b:
            player_b.client.close()
        _stop_process(sglang_a, "sglang-a")
        _stop_process(sglang_b, "sglang-b")


def _prepare_model(spec: PvPModelSpec, label: str) -> tuple[str, str, str]:
    """Download model and handle LoRA detection/merging.

    Returns:
        (model_path_for_sglang, inference_model_name, extra_sglang_args)
    """
    is_lora = check_for_lora(spec.repo, local_files_only=False)
    should_merge = is_lora and check_lora_has_added_tokens(spec.repo, local_files_only=False)

    logger.info("Model %s: repo=%s is_lora=%s merge=%s", label, spec.repo, is_lora, should_merge)

    if is_lora and not should_merge:
        # SGLang native LoRA: load base model + adapter separately
        model_path = _download_model_with_retry(spec.original_model)
        lora_dir = f"/lora/{label}_trained_lora"
        _download_lora_with_retry(spec.repo, lora_dir)
        _clean_lora_dir(lora_dir)
        inference_name = f"{spec.original_model}:{label}_trained_lora"
        extra_args = f"--enable-lora --lora-paths {label}_trained_lora={lora_dir} --lora-backend triton"
        return model_path, inference_name, extra_args

    if is_lora and should_merge:
        # LoRA with added tokens: merge into base, serve merged
        base_path = _download_model_with_retry(spec.original_model)
        lora_temp = f"/tmp/lora/{label}_trained_lora"
        _download_lora_with_retry(spec.repo, lora_temp)
        merged_path = _merge_base_and_lora(base_path, lora_temp, output_dir=f"/tmp/merged_{label}")
        return merged_path, spec.repo, ""

    # Full weights: download and serve directly
    model_path = _download_model_with_retry(spec.repo)
    return model_path, spec.repo, ""


def _clean_lora_dir(lora_dir: str) -> None:
    """Remove incompatible full-model files from a LoRA download."""
    for model_file in glob.glob(os.path.join(lora_dir, "model-*.safetensors")):
        try:
            os.remove(model_file)
            logger.info("Removed incompatible LoRA file: %s", os.path.basename(model_file))
        except OSError as exc:
            logger.warning("Failed to remove %s: %s", model_file, exc)

    index_file = os.path.join(lora_dir, "model.safetensors.index.json")
    if os.path.exists(index_file):
        try:
            os.remove(index_file)
        except OSError as exc:
            logger.warning("Failed to remove index file: %s", exc)


def _build_sglang_command(model_path: str, port: int, seed: int, extra_args: str) -> str:
    """Build SGLang launch command with explicit port and optional LoRA flags."""
    tensor_parallel = os.getenv("SGLANG_TENSOR_PARALLEL_SIZE", "1")
    dtype = os.getenv("SGLANG_DTYPE", "float16")
    cli_extra = (os.getenv("SGLANG_ENV_EVAL_EXTRA_CLI") or vcst.SGLANG_ENV_EVAL_EXTRA_CLI).strip()

    cmd = (
        "python3 -m sglang.launch_server "
        f"--model-path {model_path} "
        f"--host 0.0.0.0 --port {port} "
        f"--tensor-parallel-size {tensor_parallel} "
        f"--dtype {dtype} "
        f"--enable-deterministic-inference --random-seed {seed}"
    )
    if cli_extra:
        cmd = f"{cmd} {cli_extra}"
    if extra_args:
        cmd = f"{cmd} {extra_args}"
    return cmd


def _start_sglang(model_path: str, gpu_id: int, port: int, seed: int, extra_args: str) -> subprocess.Popen:
    """Start an SGLang server on the specified GPU and port."""
    cmd = _build_sglang_command(model_path, port, seed, extra_args)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
    logger.info("Starting SGLang on GPU %d port %d", gpu_id, port)
    proc = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        process_group=0,
        env=env,
    )
    _drain_stdout(proc, f"sglang-gpu{gpu_id}")
    return proc


def _drain_stdout(proc: subprocess.Popen, name: str) -> None:
    """Drain subprocess stdout in a background thread to prevent pipe buffer deadlock."""

    def _reader() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            logger.debug("[%s] %s", name, line.rstrip())
        proc.stdout.close()

    thread = threading.Thread(target=_reader, name=f"drain-{name}", daemon=True)
    thread.start()


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
