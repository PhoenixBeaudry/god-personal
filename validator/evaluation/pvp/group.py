"""Group round-robin PvP evaluation.

Loads the shared base model once per GPU with all LoRA adapters,
then plays all C(N,2) pairings without server restarts.
"""

import asyncio
import functools
import itertools
import logging
import os
import subprocess
import time

import openai

from core.constants import EnvironmentName
from core.models.pvp_models import (
    ChatCompletionConfig,
    PvPEnvironmentResult,
    PvPEvalConfig,
    PvPEvalMetadata,
    PvPGroupModelSpec,
    PvPGroupResults,
    PvPPairResult,
)
from validator.core import constants as vcst
from validator.evaluation.eval_environment import _wait_for_health
from validator.evaluation.utils import check_for_lora, stop_process
from validator.evaluation.pvp.chat import chat_completion, create_client
from validator.evaluation.pvp.game_runner import Player, run_matchup
from validator.evaluation.pvp.server import _drain_stdout

logger = logging.getLogger(__name__)


def run_group_evaluation(config: PvPEvalConfig) -> PvPGroupResults:
    """Run a full round-robin group evaluation."""
    if not config.models or not config.base_model:
        raise ValueError("Group mode requires 'models' and 'base_model' fields")

    start_time = time.time()
    models = config.models
    base_model = config.base_model

    lora_names = _detect_lora_names(models)
    multi_lora_args = _build_multi_lora_args(lora_names)

    sglang_a: subprocess.Popen | None = None
    sglang_b: subprocess.Popen | None = None
    client_a: openai.OpenAI | None = None
    client_b: openai.OpenAI | None = None

    try:
        sglang_a = _start_multi_lora_sglang(
            base_model, multi_lora_args, config.gpu_ids[0], config.ports[0], config.seed,
        )
        sglang_b = _start_multi_lora_sglang(
            base_model, multi_lora_args, config.gpu_ids[1], config.ports[1], config.seed + 1,
        )
        asyncio.run(_wait_for_both(config.ports[0], config.ports[1]))

        base_config_a = _base_chat_config(config, config.ports[0])
        base_config_b = _base_chat_config(config, config.ports[1])
        client_a = create_client(base_config_a)
        client_b = create_client(base_config_b)

        pairs = list(itertools.combinations(models, 2))
        pair_results: list[PvPPairResult] = []

        for idx, (spec_a, spec_b) in enumerate(pairs):
            logger.info("Pair %d/%d: %s vs %s", idx + 1, len(pairs), spec_a.hotkey, spec_b.hotkey)

            player_a = _make_player(client_a, base_config_a, base_model, lora_names[spec_a.repo])
            player_b = _make_player(client_b, base_config_b, base_model, lora_names[spec_b.repo])

            env_results: dict[EnvironmentName, PvPEnvironmentResult] = {}
            for env_name, matchup_config in config.matchups.items():
                env_results[env_name] = run_matchup(
                    env_name=env_name,
                    matchup_config=matchup_config,
                    player_a=player_a,
                    player_b=player_b,
                    base_seed=config.seed,
                )

            pair_results.append(PvPPairResult(
                hotkey_a=spec_a.hotkey,
                hotkey_b=spec_b.hotkey,
                results=env_results,
            ))

        return PvPGroupResults(
            base_model=base_model,
            hotkeys=[m.hotkey for m in models],
            pair_results=pair_results,
            metadata=PvPEvalMetadata(
                seed=config.seed,
                temperature=config.temperature,
                wall_time_seconds=time.time() - start_time,
            ),
        )
    finally:
        if client_a:
            client_a.close()
        if client_b:
            client_b.close()
        stop_process(sglang_a, "sglang-a")
        stop_process(sglang_b, "sglang-b")


def _detect_lora_names(models: list[PvPGroupModelSpec]) -> dict[str, str]:
    """Detect LoRA vs full weights and assign adapter names.

    Returns {repo: lora_name}. Full-weight models get empty string.
    """
    names: dict[str, str] = {}
    for i, spec in enumerate(models):
        is_lora = check_for_lora(spec.repo, local_files_only=False)
        names[spec.repo] = f"lora_{i}" if is_lora else ""
        logger.info("Model %s: is_lora=%s, adapter_name=%s", spec.hotkey, is_lora, names[spec.repo] or "(full weights)")
    return names


def _build_multi_lora_args(lora_names: dict[str, str]) -> str:
    """Build SGLang --enable-lora --lora-paths args for all detected adapters."""
    entries = [f"{name}={repo}" for repo, name in lora_names.items() if name]
    if not entries:
        return ""
    return f"--enable-lora --lora-paths {' '.join(entries)} --lora-backend triton"


def _start_multi_lora_sglang(
    base_model: str,
    extra_args: str,
    gpu_id: int,
    port: int,
    seed: int,
) -> subprocess.Popen:
    """Start SGLang with base model + all LoRA adapters."""
    tensor_parallel = os.getenv("SGLANG_TENSOR_PARALLEL_SIZE", "1")
    dtype = os.getenv("SGLANG_DTYPE", "float16")
    cli_extra = (os.getenv("SGLANG_ENV_EVAL_EXTRA_CLI") or vcst.SGLANG_ENV_EVAL_EXTRA_CLI).strip()

    cmd = (
        "python3 -m sglang.launch_server "
        f"--model-path {base_model} "
        f"--host 0.0.0.0 --port {port} "
        f"--tensor-parallel-size {tensor_parallel} "
        f"--dtype {dtype} "
        f"--enable-deterministic-inference --random-seed {seed}"
    )
    if cli_extra:
        cmd = f"{cmd} {cli_extra}"
    if extra_args:
        cmd = f"{cmd} {extra_args}"

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
    logger.info("Starting multi-LoRA SGLang on GPU %d port %d", gpu_id, port)
    proc = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, preexec_fn=os.setsid, env=env,
    )
    _drain_stdout(proc, f"sglang-gpu{gpu_id}")
    return proc


def _base_chat_config(config: PvPEvalConfig, port: int) -> ChatCompletionConfig:
    """Build a ChatCompletionConfig template (inference_model swapped per pair)."""
    return ChatCompletionConfig(
        inference_model=config.base_model or "",
        base_url=f"http://{vcst.PVP_SGLANG_HOST}:{port}{vcst.PVP_SGLANG_API_PATH}",
        temperature=config.temperature,
        seed=config.seed,
    )


def _make_player(
    client: openai.OpenAI,
    base_config: ChatCompletionConfig,
    base_model: str,
    lora_name: str,
) -> Player:
    """Build a Player targeting a specific adapter (or base model if no LoRA)."""
    inference_name = f"{base_model}:{lora_name}" if lora_name else base_model
    config = base_config.model_copy(update={"inference_model": inference_name})
    bound_chat = functools.partial(chat_completion, client)
    return Player(client=client, config=config, chat_fn=bound_chat)


async def _wait_for_both(port_a: int, port_b: int) -> None:
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
