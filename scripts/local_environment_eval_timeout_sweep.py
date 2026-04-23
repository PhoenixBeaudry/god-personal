#!/usr/bin/env python3
"""
Sweeps ENV_EVAL_TASK_TIMEOUT values for the local environment evaluation flow
and records the wall-clock time and score for each value.

Edit the config constants below, then run:
    python -m scripts.local_environment_eval_timeout_sweep
"""

import asyncio
import time

from core.models.utility_models import EnvironmentDatasetType
from validator.evaluation.local_evaluation import run_evaluation_local_environment


# --- Model Configuration ---
BASE_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
LORA_MODEL_NAME = "gradients-io-tournaments/tournament-tourn_768016249c31033a_20260420-7afb25f0-fc5c-4f37-951c-51915a25a676-5FRdgPRd"  # e.g. "your-org/your-lora-repo"

# --- Evaluation Configuration ---
GAME_TO_EVAL = "gin_rummy"
GPU_ID = 0
NUM_ENV_SERVERS = 1
TIMEOUTS_TO_TEST = [45, 50, 60, 90, 130, 150]
BASE_SEED = 4578


async def run_evaluation(base_seed: int, timeout: int) -> float:
    dataset_type = EnvironmentDatasetType(environment_name=GAME_TO_EVAL)
    model_to_eval = LORA_MODEL_NAME or BASE_MODEL_NAME

    print(f"🚀 Running local environment evaluation for: {model_to_eval}")
    print(f"🎮 Environment: {GAME_TO_EVAL}")
    print(f"🎯 GPU ID: {GPU_ID}")
    print(f"🌱 Eval seed: {base_seed}")
    print(f"🖧  Env servers: {NUM_ENV_SERVERS}")
    print(f"⏱  Task timeout: {timeout}s")

    results = await run_evaluation_local_environment(
        models=[model_to_eval],
        original_model=BASE_MODEL_NAME,
        dataset_type=dataset_type,
        gpu_id=GPU_ID,
        eval_seed=base_seed,
        num_env_servers=NUM_ENV_SERVERS,
        task_timeout=timeout,
    )

    result_obj = results.results.get(model_to_eval)
    if isinstance(result_obj, Exception):
        raise RuntimeError(f"Evaluation failed: {result_obj}")

    print(f"✅ Score: {result_obj.eval_loss}")
    return result_obj.eval_loss


async def main() -> None:
    scores: list[float] = []
    times: list[float] = []
    for i in TIMEOUTS_TO_TEST:
        start = time.perf_counter()
        score = await run_evaluation(base_seed=BASE_SEED, timeout=i)
        end = time.perf_counter()
        elapsed = end - start
        times.append(elapsed)
        scores.append(score)
        print(f"Timeout: {i}")
        print(score)
        print(elapsed)
    print(scores)
    print(times)


if __name__ == "__main__":
    asyncio.run(main())
