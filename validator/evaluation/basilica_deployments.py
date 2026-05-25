import asyncio
import json
import logging
import re
import time

import basilica
import requests
from huggingface_hub import HfApi

from core.logging import get_logger


logger = get_logger(__name__)
hf_api = HfApi()

EVAL_RESULT_STATUS_PATH = "/result"
_BASILICA_LOG_LINE_OFFSETS: dict[str, int] = {}

def clean_basilica_log_line(raw_line: str) -> str:
    line = raw_line.strip()
    if not line:
        return ""
    line = re.sub(r"^data:\s*", "", line).rstrip(", ")
    for _ in range(2):
        try:
            parsed = json.loads(line)
        except Exception:
            break

        if isinstance(parsed, dict):
            extracted = parsed.get("message") or parsed.get("log") or parsed.get("data")
            if isinstance(extracted, str) and extracted.strip():
                line = extracted.strip()
                continue
            line = str(parsed)
            break

        if isinstance(parsed, str):
            line = parsed.strip()
            continue

        line = str(parsed)
        break
    if "\\u001b" in line or "\\x1b" in line:
        try:
            line = bytes(line, "utf-8").decode("unicode_escape")
        except Exception:
            pass

    line = re.sub(r"\x1B\[[0-?]*[ -/]*[@-~]", "", line)
    line = re.sub(r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?\]\s*", "", line)
    line = re.sub(r"\s+", " ", line).strip()
    return line


def log_basilica_logs_block(eval_logger: logging.Logger, repo: str, deployment_name: str, deployment) -> None:
    try:
        raw_logs = deployment.logs()
    except Exception as e:
        eval_logger.warning(f"[BASILICA_LOG_FETCH_FAILED] repo={repo} deployment={deployment_name} error={e}")
        return

    if not raw_logs:
        eval_logger.info(f"[BASILICA_LOGS] repo={repo} deployment={deployment_name} lines=0 message=\"no logs returned\"")
        return

    if isinstance(raw_logs, bytes):
        raw_logs = raw_logs.decode("utf-8", errors="replace")

    lines = []
    for raw_line in str(raw_logs).splitlines():
        cleaned = clean_basilica_log_line(raw_line)
        if cleaned:
            lines.append(cleaned)

    if not lines:
        eval_logger.info(
            f"[BASILICA_LOGS] repo={repo} deployment={deployment_name} lines=0 "
            "message=\"log payload present but no parsable lines\""
        )
        return

    previous_count = _BASILICA_LOG_LINE_OFFSETS.get(deployment_name, 0)
    if previous_count > len(lines):
        previous_count = 0
    new_lines = lines[previous_count:]
    _BASILICA_LOG_LINE_OFFSETS[deployment_name] = len(lines)

    if not new_lines:
        eval_logger.info(
            f"[BASILICA_LOGS] repo={repo} deployment={deployment_name} new_lines=0 total_lines={len(lines)}"
        )
        return

    eval_logger.info(
        f"[BASILICA_LOGS] repo={repo} deployment={deployment_name} "
        f"new_lines={len(new_lines)} total_lines={len(lines)}"
    )
    for line_number, line in enumerate(new_lines, start=previous_count + 1):
        eval_logger.info(f"[BASILICA_LOG] repo={repo} deployment={deployment_name} line={line_number} | {line}")


def deployment_is_healthy(deployment, health_path: str = "/health", timeout: int = 8) -> bool:
    try:
        response = requests.get(f"{deployment.url}{health_path}", timeout=timeout)
        return response.status_code == 200
    except Exception:
        return False


async def delete_deployment_if_exists(deployment_name: str) -> None:
    try:
        client = basilica.BasilicaClient()
        deployments = await asyncio.to_thread(client.list)
        for dep in deployments:
            if getattr(dep, "name", None) == deployment_name:
                await asyncio.to_thread(dep.delete)
                return
    except Exception:
        return


async def cleanup_basilica_deployments_by_name(deployment_names: set[str]) -> None:
    """Cleanup specific Basilica deployments by name."""
    if not deployment_names:
        return
    try:
        client = basilica.BasilicaClient()
        deployments = await asyncio.to_thread(client.list)
    except Exception as e:
        logger.warning(f"Failed to list deployments for final cleanup: {e}")
        return

    by_name = {getattr(dep, "name", None): dep for dep in deployments}
    cleaned = 0
    for name in deployment_names:
        dep = by_name.get(name)
        if dep is None:
            continue
        try:
            await asyncio.to_thread(dep.delete)
            cleaned += 1
        except Exception as e:
            logger.warning(f"Failed final cleanup for deployment {name}: {e}")

    if cleaned:
        logger.info(f"Final cleanup removed {cleaned} lingering deployments for this evaluation batch")


def create_basilica_eval_runner_source(command: list[str], result_path: str) -> str:
    """Create a generic eval runner source with health and result endpoints.

    The runner executes a single eval command, then serves the parsed
    `evaluation_results.json` payload on `/result`.
    """
    command_json = json.dumps(command)
    result_path_json = json.dumps(result_path)
    return f"""from collections import deque
import json
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

COMMAND = {command_json}
RESULT_PATH = {result_path_json}
RESULT_STATUS_PATH = "{EVAL_RESULT_STATUS_PATH}"

_state = {{
    "status": "running",
    "result": None,
    "error": None,
}}

class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{{"status":"ok"}}')
            return
        if self.path == RESULT_STATUS_PATH:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(_state).encode("utf-8"))
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, format, *args):
        return

def _run_eval():
    try:
        tail = deque(maxlen=80)
        proc = subprocess.Popen(
            COMMAND,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            tail.append(line.rstrip("\\n"))
        returncode = proc.wait()
        if returncode != 0:
            tail_text = "\\n".join(tail)
            raise RuntimeError(f"Eval command failed with exit code {{returncode}}. Output tail:\\n{{tail_text}}")
        with open(RESULT_PATH, "r", encoding="utf-8") as f:
            _state["result"] = json.load(f)
        _state["status"] = "completed"
    except Exception as e:
        if _state["status"] != "completed":
            _state["status"] = "failed"
            _state["error"] = str(e)

def main():
    server = HTTPServer(("0.0.0.0", 8000), _Handler)
    worker = threading.Thread(target=_run_eval, daemon=True)
    worker.start()
    server.serve_forever()

if __name__ == "__main__":
    main()
"""


def wait_for_basilica_health(url: str, timeout: int = 3600, path: str = "/v1/models") -> bool:
    """Wait for Basilica service to be healthy."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{url}{path}", timeout=5)
            if response.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(5)

    error_msg = f"Service at {url} did not become healthy within {timeout} seconds"
    raise TimeoutError(error_msg)
