"""
Environment task stats: deploy model via SGLang, play episodes against env server.
Reuses SGLang infrastructure from validator/evaluation/eval_environment.py.
"""

import os
import random

import aiohttp

from core.models.model_prep_models import EnvBaselineStats, EnvStats
from trainer.model_prep.stats import _compute_weight_stats
from validator.core import constants as vcst
from validator.evaluation.eval_environment import (
    _build_sglang_command,
    _start_process,
    _stop_process,
    _wait_for_health,
)


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

    sglang_cmd = _build_sglang_command(model_path, seed=42)
    sglang_proc = _start_process(sglang_cmd, "sglang")
    sglang_port = int(os.getenv("SGLANG_PORT", "30000"))
    sglang_url = f"http://localhost:{sglang_port}"

    try:
        await _wait_for_health(
            sglang_url, "/v1/models",
            timeout_seconds=vcst.LOCAL_ENV_SGLANG_HEALTH_TIMEOUT,
            service_name="sglang",
        )

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
                    "temperature": vcst.ENV_EVAL_TEMPERATURE,
                    "seed": seed,
                }
                if env_payload_extra:
                    payload.update(env_payload_extra)

                try:
                    timeout = aiohttp.ClientTimeout(total=vcst.ENV_EVAL_TASK_TIMEOUT)
                    async with session.post(
                        f"{env_server_url}/evaluate",
                        json=payload,
                        timeout=timeout,
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

        mean = sum(scores) / max(len(scores), 1)
        print(f"Done: {len(scores)} episodes, mean={mean:.3f}", flush=True)

        return EnvBaselineStats(
            weights=weight_stats,
            env_stats=EnvStats(
                environment_name=environment_name,
                num_episodes=len(scores),
                episode_scores=scores,
            ),
        )

    except TimeoutError:
        print("SGLang failed to start within timeout", flush=True)
        return EnvBaselineStats(
            weights=weight_stats,
            env_stats=EnvStats(
                environment_name=environment_name,
                num_episodes=0,
                episode_scores=[],
            ),
        )

    finally:
        _stop_process(sglang_proc, "sglang")
