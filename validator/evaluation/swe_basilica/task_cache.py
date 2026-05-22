from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.error import URLError
from urllib.request import Request
from urllib.request import urlopen


DEFAULT_R2_BASE_URL = "https://pub-7882418a56434a479bf9a7febd660b36.r2.dev"
DEFAULT_R2_PREFIX = "bugs"
DEFAULT_CACHE_DIR = "/tmp/swe-task-cache"


def format_task_filename(task_id: int | str) -> str:
    try:
        return f"task_{int(task_id):011d}.json"
    except (TypeError, ValueError):
        return f"{task_id}.json"


class SweTaskCache:
    """Read SWE task metadata directly from the R2 JSON cache."""

    def __init__(
        self,
        *,
        cache_dir: str | None = None,
        base_url: str | None = None,
        prefix: str | None = None,
        url_template: str | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir or os.getenv("SWE_TASK_CACHE_DIR") or DEFAULT_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = (
            base_url
            or os.getenv("SWE_R2_BASE_URL")
            or os.getenv("R2_BASE_URL")
            or os.getenv("R2_PUBLIC_URL")
            or DEFAULT_R2_BASE_URL
        ).rstrip("/")
        self.prefix = (
            prefix
            if prefix is not None
            else (
                os.getenv("SWE_R2_PREFIX")
                or os.getenv("R2_PREFIX")
                or os.getenv("R2_PUBLIC_PREFIX")
                or DEFAULT_R2_PREFIX
            )
        ).strip("/")
        self.url_template = url_template or os.getenv("SWE_TASK_URL_TEMPLATE")

    def _local_path(self, task_id: int | str) -> Path:
        return self.cache_dir / format_task_filename(task_id)

    def _url(self, task_id: int | str) -> str:
        filename = format_task_filename(task_id)
        if self.url_template:
            return self.url_template.format(task_id=task_id, filename=filename)
        if self.prefix:
            return f"{self.base_url}/{self.prefix}/{filename}"
        return f"{self.base_url}/{filename}"

    def load(self, task_id: int | str) -> dict[str, Any]:
        local_path = self._local_path(task_id)
        if local_path.exists():
            return json.loads(local_path.read_text(encoding="utf-8"))

        url = self._url(task_id)
        request = Request(url, headers={"Accept": "application/json", "User-Agent": "gradients-swe-eval/1.0"})
        try:
            with urlopen(request, timeout=int(os.getenv("SWE_TASK_FETCH_TIMEOUT", "30"))) as response:
                payload = json.loads(response.read())
        except HTTPError as exc:
            raise FileNotFoundError(f"SWE task {task_id!r} not found at {url}: HTTP {exc.code}") from exc
        except (URLError, TimeoutError) as exc:
            raise RuntimeError(f"Failed to fetch SWE task {task_id!r} from {url}: {exc}") from exc

        local_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

