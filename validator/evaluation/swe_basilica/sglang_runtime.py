from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys

from validator.core import constants as vcst
from validator.evaluation.eval_environment import _start_process
from validator.evaluation.eval_environment import _stream_logs
from validator.evaluation.eval_environment import _wait_for_health
from validator.evaluation.swe_basilica.model_server import prepare_sglang_for_model
from validator.evaluation.utils import configure_eval_logging
from validator.evaluation.utils import stop_process


logger = logging.getLogger(__name__)


async def _run() -> None:
    models_raw = os.getenv("MODELS", "")
    model_repo = models_raw.split(",")[0].strip()
    if not model_repo:
        raise ValueError("MODELS is required for SWE SGLang deployment")

    original_model = os.getenv("ORIGINAL_MODEL", model_repo)
    base_seed = int(os.getenv("EVAL_SEED", str(vcst.ENV_EVAL_DEFAULT_SEED)))
    port = int(os.getenv("SGLANG_PORT", "30000"))
    health_timeout = int(os.getenv("SGLANG_HEALTH_TIMEOUT", "1800"))
    base_url = f"http://127.0.0.1:{port}"

    min_ws = vcst.SGLANG_FLASHINFER_WORKSPACE_MIN_BYTES
    try:
        cur_ws = int(os.environ.get("SGLANG_FLASHINFER_WORKSPACE_SIZE", "0") or "0")
    except ValueError:
        cur_ws = 0
    if cur_ws < min_ws:
        os.environ["SGLANG_FLASHINFER_WORKSPACE_SIZE"] = str(min_ws)

    inference_name, model_path, command = await prepare_sglang_for_model(model_repo, original_model, base_seed)
    if f"--port {port}" not in command:
        command = command.replace("--port 30000", f"--port {port}")

    logger.info("SWE SGLang server starting: inference_name=%s model_path=%s", inference_name, model_path)
    logger.info("SWE SGLang command: %s", command)

    proc = None
    log_task = None
    stop_event = asyncio.Event()

    def _stop(*_args) -> None:
        stop_event.set()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    try:
        proc = _start_process(command, "sglang")
        log_task = asyncio.create_task(_stream_logs(proc, "sglang"))
        await _wait_for_health(base_url, "/v1/models", health_timeout, service_name="SGLang")
        logger.info("SWE SGLang healthy at %s", base_url)
        while proc.poll() is None and not stop_event.is_set():
            await asyncio.sleep(5)
        if proc.poll() is not None:
            raise RuntimeError(f"SGLang exited with code {proc.returncode}")
    finally:
        stop_process(proc, "sglang")
        if log_task:
            log_task.cancel()


def main() -> int:
    configure_eval_logging()
    try:
        asyncio.run(_run())
        return 0
    except Exception as exc:
        logger.exception("SWE SGLang runtime failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
