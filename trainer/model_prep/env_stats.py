"""
Environment task stats: deploy model via SGLang, play episodes against env server.
Self-contained — no validator imports. SGLang helpers inlined from eval_environment.py.
"""

import asyncio
import logging
import os
import random
import signal
import statistics
import subprocess
import time

import aiohttp

from core.models.model_prep_models import EnvBaselineStats, EnvStats
from trainer.model_prep.stats import _compute_weight_stats

logger = logging.getLogger(__name__)

# Default SGLang CLI flags (inlined from validator.core.constants)
SGLANG_EXTRA_CLI_DEFAULT = (
    "--attention-backend triton --prefill-attention-backend triton "
    "--decode-attention-backend triton --sampling-backend pytorch"
)
SGLANG_HEALTH_TIMEOUT = 600
ENV_EVAL_TEMPERATURE = 0.0
ENV_EVAL_TASK_TIMEOUT = 150


# --- SGLang process management (from eval_environment.py) ---

def build_sglang_command(model_path: str, seed: int) -> str:
    tensor_parallel = os.getenv("SGLANG_TENSOR_PARALLEL_SIZE", "1")
    dtype = os.getenv("SGLANG_DTYPE", "float16")
    port = os.getenv("SGLANG_PORT", "30000")
    base = (
        "python3 -m sglang.launch_server "
        f"--model-path {model_path} "
        f"--host 0.0.0.0 --port {port} "
        f"--tensor-parallel-size {tensor_parallel} "
        f"--dtype {dtype} "
        f"--enable-deterministic-inference --random-seed {seed}"
    )
    extra = (os.getenv("SGLANG_ENV_EVAL_EXTRA_CLI") or SGLANG_EXTRA_CLI_DEFAULT).strip()
    return f"{base} {extra}" if extra else base


def start_process(command: str, name: str) -> subprocess.Popen:
    logger.info("Starting %s: %s", name, command)
    return subprocess.Popen(
        command, shell=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, preexec_fn=os.setsid,
    )


def stop_process(proc: subprocess.Popen | None, name: str) -> None:
    if proc is None:
        return
    try:
        if proc.poll() is None:
            logger.info("Stopping %s (pid=%s)", name, proc.pid)
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                proc.wait(timeout=10)
    except Exception as exc:
        logger.warning("Failed to stop %s cleanly: %s", name, exc)


async def wait_for_health(
    url: str, path: str, timeout_seconds: int, *, service_name: str = "service",
) -> None:
    deadline = time.time() + timeout_seconds
    started = time.time()
    async with aiohttp.ClientSession() as session:
        while time.time() < deadline:
            try:
                async with session.get(f"{url}{path}", timeout=aiohttp.ClientTimeout(total=8)) as resp:
                    if resp.status == 200:
                        logger.info("%s healthy after %.1fs", service_name, time.time() - started)
                        return
            except Exception:
                pass
            await asyncio.sleep(2)
    raise TimeoutError(f"{service_name} at {url}{path} not healthy within {timeout_seconds}s")


def _build_env_stats(environment_name: str, scores: list[float]) -> EnvStats:
    if scores:
        return EnvStats(
            environment_name=environment_name,
            num_episodes=len(scores),
            episode_scores=scores,
            mean_score=statistics.mean(scores),
            std_score=statistics.stdev(scores) if len(scores) > 1 else 0.0,
            min_score=min(scores),
            max_score=max(scores),
            median_score=statistics.median(scores),
        )
    return EnvStats(
        environment_name=environment_name,
        num_episodes=0,
        episode_scores=[],
    )


# --- Episode playback ---

async def compute_env_stats(
    model_path: str,
    model,
    environment_name: str,
    env_server_url: str,
    num_episodes: int = 50,
    task_id_min: int = 0,
    task_id_max: int = 99999999,
    env_payload_extra: dict | None = None,
) -> EnvBaselineStats:
    """Compute env stats: deploy model via SGLang, play episodes, collect scores."""

    print("Computing weight stats...", flush=True)
    weight_stats = _compute_weight_stats(model)

    sglang_cmd = build_sglang_command(model_path, seed=42)
    sglang_proc = start_process(sglang_cmd, "sglang")
    sglang_port = int(os.getenv("SGLANG_PORT", "30000"))
    sglang_url = f"http://localhost:{sglang_port}"

    try:
        await wait_for_health(sglang_url, "/v1/models", SGLANG_HEALTH_TIMEOUT, service_name="sglang")

        seed_rng = random.Random(42)
        scores = []

        print(f"Playing {num_episodes} episodes against {environment_name}...", flush=True)

        async with aiohttp.ClientSession() as session:
            for i in range(num_episodes):
                seed = seed_rng.randint(1, 1_000_000)
                task_rng = random.Random(seed)
                task_id = task_rng.randint(task_id_min + 1, task_id_max)

                payload = {
                    "model": os.path.basename(model_path),
                    "base_url": f"{sglang_url}/v1",
                    "task_id": task_id,
                    "temperature": ENV_EVAL_TEMPERATURE,
                    "seed": seed,
                }
                if env_payload_extra:
                    payload.update(env_payload_extra)

                try:
                    timeout = aiohttp.ClientTimeout(total=ENV_EVAL_TASK_TIMEOUT)
                    async with session.post(
                        f"{env_server_url}/evaluate", json=payload, timeout=timeout,
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            result = data.get("result", data)
                            score = float(result.get("score", 0.0))
                        else:
                            score = 0.0
                except Exception as e:
                    print(f"Episode {i+1}: error {e}", flush=True)
                    score = 0.0

                scores.append(score)
                print(f"Episode {i+1}/{num_episodes}: score={score:.3f}", flush=True)

        print(f"Done: {len(scores)} episodes, mean={statistics.mean(scores):.3f}", flush=True)

        return EnvBaselineStats(
            weights=weight_stats,
            env_stats=_build_env_stats(environment_name, scores),
        )

    except TimeoutError:
        print("SGLang failed to start within timeout", flush=True)
        return EnvBaselineStats(
            weights=weight_stats,
            env_stats=_build_env_stats(environment_name, []),
        )

    finally:
        stop_process(sglang_proc, "sglang")
