import logging
import os
import signal
import subprocess
import sys

from huggingface_hub import HfApi

from core.logging import get_logger


logger = get_logger(__name__)
hf_api = HfApi()

EVAL_RESULT_STATUS_PATH = "/result"
_BASILICA_LOG_LINE_OFFSETS: dict[str, int] = {}

def configure_eval_logging() -> None:
    """Configure root logger for eval containers (stderr, replaces existing handlers)."""
    level_name = os.getenv("EVAL_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    fmt = "%(asctime)s %(levelname)s %(name)s - %(message)s"
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(fmt))
    root = logging.getLogger()
    root.setLevel(level)
    for existing in root.handlers[:]:
        root.removeHandler(existing)
        try:
            existing.close()
        except Exception:
            pass
    root.addHandler(handler)


def stop_process(proc: subprocess.Popen | None, name: str) -> None:
    """Gracefully stop a subprocess, escalating to SIGKILL if needed."""
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


def _log_eval_step(eval_logger: logging.Logger, step: str, **fields) -> None:
    field_text = " ".join(f"{key}={value}" for key, value in fields.items() if value is not None)
    eval_logger.info(f"eval_step={step} {field_text}".rstrip())
