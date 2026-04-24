#!/usr/bin/env python3
"""
Sweeps ENV_EVAL_TASK_TIMEOUT values for the local environment evaluation flow
and records the wall-clock time and score for each value.

Edit the config constants below, then run:
    python -m scripts.local_environment_eval_timeout_sweep
"""

import asyncio
import logging
import re
import sys
import threading
import time
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.models.utility_models import EnvironmentDatasetType  # noqa: E402
import validator.evaluation.local_evaluation as _local_eval  # noqa: E402
from validator.evaluation.local_evaluation import run_evaluation_local_environment  # noqa: E402


class _ConsoleProgressHandler(logging.Handler):
    """Prints per-episode progress (`[n/total] task <id> score=<s>`) to stdout.

    Reads the messages the env eval already emits on its environment logger:
    the `Starting N evaluations ...` banner (to pick up the total) and
    `Task ID X: Done (Score: Y)` per completion. Records are not mutated, so
    the VectorHandler still sees the original payload.
    """

    _done_re = re.compile(r"^Task ID (\d+): Done \(Score: (.+)\)$")
    _starting_re = re.compile(r"^Starting (\d+) evaluations sharded round-robin")
    _summary_re = re.compile(r"^Summary:")

    def __init__(self) -> None:
        super().__init__()
        self._completed = 0
        self._total = 0
        self._lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
            start_match = self._starting_re.match(msg)
            if start_match:
                with self._lock:
                    self._total = int(start_match.group(1))
                    self._completed = 0
                print(msg, flush=True)
                return
            done_match = self._done_re.match(msg)
            if done_match:
                task_id = done_match.group(1)
                score = done_match.group(2)
                with self._lock:
                    self._completed += 1
                    n = self._completed
                    total = self._total or "?"
                print(f"[{n}/{total}] task {task_id} score={score}", flush=True)
                return
            if self._summary_re.match(msg):
                print(msg, flush=True)
        except Exception:
            self.handleError(record)


def _install_console_progress_logging() -> None:
    handler = _ConsoleProgressHandler()
    original = _local_eval.get_environment_logger

    def patched(*args, **kwargs):
        lg = original(*args, **kwargs)
        if not any(isinstance(h, _ConsoleProgressHandler) for h in lg.handlers):
            lg.addHandler(handler)
        return lg

    _local_eval.get_environment_logger = patched


_install_console_progress_logging()


# --- Model Configuration ---
BASE_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
LORA_MODEL_NAME = "gradients-io-tournaments/tournament-tourn_768016249c31033a_20260420-7afb25f0-fc5c-4f37-951c-51915a25a676-5FRdgPRd"  # e.g. "your-org/your-lora-repo"

# --- Evaluation Configuration ---
GAME_TO_EVAL = "gin_rummy"
GPU_ID = 0
NUM_ENV_SERVERS = 6
TIMEOUTS_TO_TEST = [150]
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
