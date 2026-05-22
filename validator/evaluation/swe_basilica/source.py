from __future__ import annotations

from pathlib import Path


_MODULE_DIR = Path(__file__).resolve().parent


def create_worker_source() -> str:
    """Return the self-contained task-worker source injected into task images."""

    return (_MODULE_DIR / "worker_runtime.py").read_text(encoding="utf-8")


def create_sglang_source() -> str:
    """Return a tiny source entrypoint for the dedicated SGLang deployment."""

    return "\n".join(
        [
            "from validator.evaluation.swe_basilica.sglang_runtime import main",
            "",
            'if __name__ == "__main__":',
            "    main()",
            "",
        ]
    )

