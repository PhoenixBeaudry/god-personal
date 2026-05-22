from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request
from urllib.request import urlopen


APP_DIR = Path(os.getenv("SWE_WORKDIR", "/app"))
RESULT_PATH = "/result"
SUBMIT_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"

_STATE: dict[str, Any] = {
    "status": "idle",
    "run_id": None,
    "result": None,
    "error": None,
    "started_at": None,
    "finished_at": None,
}
_LOCK = threading.Lock()


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    data = json.dumps(payload, default=str).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def _read_json_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length") or "0")
    raw = handler.rfile.read(length) if length else b"{}"
    return json.loads(raw.decode("utf-8"))


def _run_shell(command: str, *, timeout: int, cwd: Path = APP_DIR) -> dict[str, Any]:
    started = time.time()
    try:
        proc = subprocess.run(
            ["bash", "-lc", command],
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            errors="replace",
        )
        return {
            "returncode": proc.returncode,
            "output": proc.stdout,
            "time": time.time() - started,
            "timeout": False,
        }
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or exc.stderr or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        return {
            "returncode": -1,
            "output": output,
            "time": time.time() - started,
            "timeout": True,
        }


def _truncate(text: str, limit: int = 12000) -> str:
    if len(text) <= limit:
        return text
    return text[: limit // 2] + "\n...[truncated]...\n" + text[-limit // 2 :]


def _safe_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except Exception:
            return [value]
    return [str(value)]


def _chat_completion(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
) -> tuple[str, dict[str, Any]]:
    url = base_url.rstrip("/") + "/chat/completions"
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
    ).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=int(os.getenv("SWE_WORKER_MODEL_TIMEOUT", "600"))) as response:
            payload = json.loads(response.read())
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"model_api_error: HTTP {exc.code}: {detail}") from exc

    choice = (payload.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = message.get("content") or choice.get("text") or ""
    return str(content), payload.get("usage") or {}


def _extract_command(text: str) -> str | None:
    if SUBMIT_MARKER in text:
        return None
    for pattern in (
        r"<cmd>\s*(.*?)\s*</cmd>",
        r"```(?:bash|sh|shell)?\s*\n(.*?)\n```",
    ):
        match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
        if match:
            command = match.group(1).strip()
            return command or None
    return None


def _extract_patch() -> str:
    _run_shell("git config --global --add safe.directory /app || true", timeout=20)
    _run_shell("git add -A >/dev/null 2>&1 || true", timeout=60)
    result = _run_shell("git diff --cached -- . ':!*.pyc' ':!__pycache__/*'", timeout=60)
    patch = result["output"].lstrip()
    return patch.rstrip("\n") + "\n" if patch else ""


def _prepare_repo() -> None:
    _run_shell("git config --global --add safe.directory /app || true", timeout=20)
    _run_shell("git config user.email swe-worker@example.invalid || true", timeout=20)
    _run_shell("git config user.name swe-worker || true", timeout=20)


def _run_agent(payload: dict[str, Any]) -> dict[str, Any]:
    task = payload["task"]
    model_cfg = payload["model"]
    agent_cfg = payload.get("agent") or {}
    problem = task.get("problem_statement") or task.get("statement") or ""
    model = model_cfg["model"]
    base_url = model_cfg["base_url"]
    api_key = model_cfg.get("api_key") or "test"
    temperature = float(model_cfg.get("temperature", 0.0))
    max_iterations = int(agent_cfg.get("max_iterations") or 100)
    command_timeout = int(agent_cfg.get("command_timeout") or 300)
    max_tokens = int(agent_cfg.get("max_tokens") or 4096)
    conversation: list[dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You are a coding agent working in /app. Inspect and edit the repository to solve the task. "
                "When you need to run shell commands, respond with exactly one command wrapped in <cmd>...</cmd>. "
                f"When the fix is complete, respond with {SUBMIT_MARKER}."
            ),
        },
        {
            "role": "user",
            "content": (
                "Solve this software engineering task. Make the smallest correct code change.\n\n"
                f"{problem}\n\nStart by inspecting the repository."
            ),
        },
    ]
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    transcript: list[dict[str, Any]] = []

    _prepare_repo()
    for iteration in range(max_iterations):
        content, call_usage = _chat_completion(
            base_url=base_url,
            api_key=api_key,
            model=model,
            messages=conversation,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        for key in usage:
            usage[key] += int(call_usage.get(key) or 0)
        conversation.append({"role": "assistant", "content": content})
        command = _extract_command(content)
        transcript.append({"iteration": iteration + 1, "assistant": content, "command": command})
        if command is None:
            break

        command_result = _run_shell(command, timeout=command_timeout)
        output = _truncate(command_result["output"])
        observation = (
            f"returncode={command_result['returncode']} timeout={command_result['timeout']}\n"
            f"{output}"
        )
        transcript[-1]["observation"] = observation
        conversation.append({"role": "user", "content": observation})

    patch = _extract_patch()
    return {
        "patch": patch,
        "conversation": transcript,
        "usage": usage,
        "model_calls": len([item for item in transcript if "assistant" in item]),
    }


def _write_patch(path: Path, content: str) -> None:
    path.write_text(content or "", encoding="utf-8")


def _apply_patch(path: Path, label: str) -> tuple[bool, str]:
    if not path.exists() or not path.read_text(encoding="utf-8", errors="replace").strip():
        return True, ""
    result = subprocess.run(
        ["git", "apply", "--recount", "--whitespace=fix", str(path)],
        cwd=str(APP_DIR),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        errors="replace",
    )
    if result.returncode != 0:
        return False, f"{label} apply failed: {result.stdout[:2000]}"
    return True, result.stdout


def _parse_test_output(output: str, required: set[str]) -> tuple[set[str], set[str]]:
    passed: set[str] = set()
    failed: set[str] = set()
    for line in output.splitlines():
        clean = line.strip()
        if not clean:
            continue
        pytest_match = re.search(r"([A-Za-z0-9_./:-]+::[A-Za-z0-9_./:\[\]-]+)\s+(PASSED|FAILED|ERROR)", clean)
        if pytest_match:
            name, status = pytest_match.groups()
            if status == "PASSED":
                passed.add(name)
            else:
                failed.add(name)
        for test_name in required:
            if test_name in clean:
                upper = clean.upper()
                if any(token in upper for token in ("PASSED", " OK", "SUCCESS")):
                    passed.add(test_name)
                if any(token in upper for token in ("FAILED", "ERROR", "FAIL ")):
                    failed.add(test_name)
    return passed, failed


def _verify_patch(payload: dict[str, Any]) -> dict[str, Any]:
    task = payload["task"]
    patch = payload.get("patch") or ""
    timeout = int((payload.get("agent") or {}).get("timeout") or payload.get("timeout") or 1800)
    test_command = task.get("test_command") or "pytest -v --tb=no"
    fail_to_pass = set(_safe_list(task.get("fail_to_pass")))
    pass_to_pass = set(_safe_list(task.get("pass_to_pass")))
    required = fail_to_pass | pass_to_pass

    _prepare_repo()
    tmp_dir = Path("/tmp/swe-worker-patches")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    test_patch_path = tmp_dir / "test_patch.diff"
    augmented_patch_path = tmp_dir / "augmented_test_patch.diff"
    fix_patch_path = tmp_dir / "fix_patch.diff"
    _write_patch(test_patch_path, task.get("test_patch") or "")
    _write_patch(augmented_patch_path, task.get("augmented_test_patch") or "")
    _write_patch(fix_patch_path, patch)

    for path, label in (
        (test_patch_path, "test_patch"),
        (augmented_patch_path, "augmented_test_patch"),
        (fix_patch_path, "fix_patch"),
    ):
        ok, detail = _apply_patch(path, label)
        if not ok:
            return {
                "score": 0.0,
                "success": False,
                "test_stats": {"error": detail},
            }

    _run_shell("git add -A >/dev/null 2>&1 || true", timeout=60)
    result = _run_shell(test_command, timeout=timeout)
    output = result["output"]
    passed, failed = _parse_test_output(output, required)

    if required:
        if result["returncode"] == 0 and not failed:
            passed = required
        success = required <= passed
    else:
        success = result["returncode"] == 0

    missing = sorted(required - passed) if required else []
    return {
        "score": 1.0 if success else 0.0,
        "success": success,
        "test_stats": {
            "returncode": result["returncode"],
            "timeout": result["timeout"],
            "passed": sorted(passed),
            "failed": sorted(failed),
            "missing": missing,
            "required_count": len(required),
            "output_tail": _truncate(output, 6000),
        },
    }


def _execute(payload: dict[str, Any]) -> dict[str, Any]:
    started = time.time()
    mode = payload.get("mode") or "solve"
    task = payload.get("task") or {}
    if not isinstance(task, dict):
        raise ValueError("payload.task must be an object")

    if mode == "solve":
        solve = _run_agent(payload)
        result = {
            "task_name": "swe",
            "score": 0.0,
            "success": bool(solve["patch"].strip()),
            "time_taken": time.time() - started,
            "extra": {
                "mode": "solve",
                "task_id": task.get("_task_id") or task.get("task_id") or task.get("instance_id"),
                "instance_id": task.get("instance_id") or task.get("task_id"),
                "repo": task.get("repo", ""),
                "repo_language": task.get("repo_language", ""),
                "fix_patch": solve["patch"],
                "conversation": solve["conversation"],
                "usage": solve["usage"],
                "model_calls": solve["model_calls"],
            },
        }
    elif mode == "verify":
        verify = _verify_patch(payload)
        result = {
            "task_name": "swe",
            "score": verify["score"],
            "success": verify["success"],
            "time_taken": time.time() - started,
            "extra": {
                "mode": "verify",
                "task_id": task.get("_task_id") or task.get("task_id") or task.get("instance_id"),
                "instance_id": task.get("instance_id") or task.get("task_id"),
                "repo": task.get("repo", ""),
                "repo_language": task.get("repo_language", ""),
                "test_stats": verify["test_stats"],
            },
        }
    else:
        raise ValueError(f"Unsupported SWE worker mode: {mode}")

    return {"status": "completed", "result": result}


def _run_background(payload: dict[str, Any], run_id: str) -> None:
    try:
        result = _execute(payload)
        with _LOCK:
            _STATE["status"] = "completed"
            _STATE["result"] = result["result"]
            _STATE["error"] = None
            _STATE["finished_at"] = time.time()
    except Exception as exc:
        with _LOCK:
            _STATE["status"] = "failed"
            _STATE["error"] = {
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
            _STATE["finished_at"] = time.time()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/health":
            _json_response(self, 200, {"status": "ok", "worker": "gradients-swe-task-worker"})
            return
        if self.path == RESULT_PATH:
            with _LOCK:
                payload = dict(_STATE)
            _json_response(self, 200, payload)
            return
        _json_response(self, 404, {"error": "not_found"})

    def do_POST(self) -> None:
        if self.path != "/run":
            _json_response(self, 404, {"error": "not_found"})
            return
        try:
            payload = _read_json_body(self)
        except Exception as exc:
            _json_response(self, 400, {"error": f"invalid_json: {exc}"})
            return

        with _LOCK:
            if _STATE["status"] == "running":
                _json_response(self, 409, {"error": "worker_busy", "run_id": _STATE["run_id"]})
                return
            run_id = str(payload.get("run_id") or uuid.uuid4())
            _STATE.update(
                {
                    "status": "running",
                    "run_id": run_id,
                    "result": None,
                    "error": None,
                    "started_at": time.time(),
                    "finished_at": None,
                }
            )

        thread = threading.Thread(target=_run_background, args=(payload, run_id), daemon=True)
        thread.start()
        _json_response(self, 202, {"status": "running", "run_id": run_id})

    def log_message(self, _format: str, *args: Any) -> None:
        return


def main() -> None:
    port = int(os.getenv("PORT", "8000"))
    server = ThreadingHTTPServer(("0.0.0.0", port), _Handler)
    print(f"[swe-worker] serving on 0.0.0.0:{port}; app={APP_DIR}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()

