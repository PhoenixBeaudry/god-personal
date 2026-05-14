#!/usr/bin/env python3
"""FastAPI wrapper around Intercode-Bash.

    POST /reset    body: ResetRequest    -> ResetResponse
    POST /step     body: StepRequest     -> StepResponse
    GET  /health                         -> {"status": "ok"}
    GET  /metadata                       -> EnvironmentMetadata
    GET  /state                          -> State
    GET  /schema                         -> {action, observation, state} schemas

A single BashEnv container is held by the server (Intercode hard-codes its
container names, so only one episode runs at a time). /reset (a) maps the
global task_id 1..200 to a filesystem variant fs_1..fs_4, (b) ensures the
matching docker image is built and tagged as `intercode-nl2bash`, (c) tears
down and rebuilds the BashEnv if the fs changed, (d) resets to the local
task index and returns the natural-language query as the initial obs.

The action shape is `{"command": str}`:
  - `"submit"` terminates the episode and triggers Intercode's reward
    function (gold-replay diff + stdout TF-IDF).
  - anything else is treated as a raw bash command and forwarded to
    `env.step(...)`. Common ReAct wrappers (`execute[cmd]`, `Action N:
    execute[cmd]`) are unwrapped automatically.

Run:
    pip install intercode-bench openai fastapi uvicorn pydantic
    docker daemon running locally
    python scripts/intercode_server.py --host 0.0.0.0 --port 8000

Test:
    curl -s -X POST localhost:8000/reset -H 'content-type: application/json' \\
        -d '{"task_id": 1}' | jq
    curl -s -X POST localhost:8000/step  -H 'content-type: application/json' \\
        -d '{"action": {"command": "ls /testbed"}}' | jq
    curl -s -X POST localhost:8000/step  -H 'content-type: application/json' \\
        -d '{"action": {"command": "submit"}}' | jq
"""

# TODO Action from llm should be just raw completion?

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import threading
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ENV_NAME = "intercode-nl2bash"
ENV_VERSION = "0.1.0"
ENV_DESCRIPTION = (
    "InterCode NL2Bash benchmark (200 tasks across 4 filesystem variants) "
    "exposed as an OpenEnv-compatible HTTP environment."
)

OBS_TRUNCATE_CHARS = 350  # matches intercode/experiments/eval_react.py

INTERCODE_GIT_URL = "https://github.com/princeton-nlp/intercode.git"
INTERCODE_DATA_RAW = (
    "https://raw.githubusercontent.com/princeton-nlp/intercode/master/data/nl2bash"
)

CACHE_DIR = Path.home() / ".cache" / "intercode_server"
REPO_CACHE = CACHE_DIR / "intercode"
DATA_CACHE = CACHE_DIR / "data"

# BashEnv hard-codes `intercode-nl2bash` in its IMAGE_TO_SETTINGS lookup, so
# whatever fs-specific image we build must be tagged with that exact name
# before BashEnv() is constructed. We keep per-fs tags around for caching and
# re-tag the active one as `intercode-nl2bash` on each fs switch.
CANONICAL_IMAGE = "intercode-nl2bash"
CACHE_TAG_FOR_FS = {1: "intercode-nl2bash-fs1",
                    2: "intercode-nl2bash-fs2",
                    3: "intercode-nl2bash-fs3",
                    4: "intercode-nl2bash-fs1"}  # fs_4 is filesystem-agnostic
BUILD_FS_VERSION = {1: 1, 2: 2, 3: 3, 4: 1}

# Verbatim from princeton-nlp/intercode experiments/utils/utils.py.
_REACT_ACTION_RE = re.compile(r"execute\[(.*)\]", re.DOTALL)


def bash_parser_react(action: str) -> tuple[str, bool]:
    if action == "submit":
        return action, True
    matches = _REACT_ACTION_RE.findall(action)
    if matches:
        return matches[0], True
    return action, False


##############################################################################
# Data + image setup
##############################################################################

def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    print(f"📥 Downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)


def fetch_all_datasets() -> dict[int, Path]:
    paths: dict[int, Path] = {}
    for fs in (1, 2, 3, 4):
        local = DATA_CACHE / f"nl2bash_fs_{fs}.json"
        _download(f"{INTERCODE_DATA_RAW}/nl2bash_fs_{fs}.json", local)
        paths[fs] = local
    return paths


def compute_fs_ranges(data_paths: dict[int, Path]) -> list[tuple[int, int, int]]:
    """List of (fs_version, start_global_id, end_global_id) -- inclusive, 1-indexed."""
    ranges: list[tuple[int, int, int]] = []
    cursor = 1
    for fs in (1, 2, 3, 4):
        n = len(json.loads(data_paths[fs].read_text()))
        ranges.append((fs, cursor, cursor + n - 1))
        cursor += n
    return ranges


def map_task_id(global_id: int, ranges) -> tuple[int, int]:
    """1-indexed global id -> (fs_version, 0-indexed local id within fs)."""
    for fs, start, end in ranges:
        if start <= global_id <= end:
            return fs, global_id - start
    total = ranges[-1][2]
    raise ValueError(f"task id {global_id} out of range; valid range is 1..{total}")


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd, check=True)


def _ensure_intercode_repo() -> Path:
    dockerfile = REPO_CACHE / "docker" / "nl2bash.Dockerfile"
    if REPO_CACHE.exists() and dockerfile.exists():
        return REPO_CACHE
    REPO_CACHE.parent.mkdir(parents=True, exist_ok=True)
    if REPO_CACHE.exists():
        shutil.rmtree(REPO_CACHE)
    _run(["git", "clone", "--depth", "1", INTERCODE_GIT_URL, str(REPO_CACHE)])
    return REPO_CACHE


def _image_exists(image: str) -> bool:
    return subprocess.run(
        ["docker", "image", "inspect", image],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0


def ensure_image(fs_version: int) -> str:
    """Build (if missing) the per-fs image and tag it as the canonical
    `intercode-nl2bash` name that BashEnv expects. Returns CANONICAL_IMAGE."""
    cache_tag = CACHE_TAG_FOR_FS[fs_version]
    if not _image_exists(cache_tag):
        repo = _ensure_intercode_repo()
        src_dockerfile = repo / "docker" / "nl2bash.Dockerfile"
        if not src_dockerfile.exists():
            raise RuntimeError(f"missing Dockerfile at {src_dockerfile}")

        build_fs = BUILD_FS_VERSION[fs_version]
        patched = src_dockerfile.read_text().replace(
            "ENV file_system_version=1",
            f"ENV file_system_version={build_fs}",
        )
        # Keep the patched file alongside the original so the upstream
        # `COPY ../docker/bash_scripts/$script /` relative path resolves the
        # same way.
        patched_path = src_dockerfile.with_name(f"nl2bash.fs{build_fs}.Dockerfile")
        patched_path.write_text(patched)

        print(f"🐳 Building docker image {cache_tag} (fs_version={build_fs})…")
        _run([
            "docker", "build",
            "-t", cache_tag,
            "-f", str(patched_path),
            str(repo),
        ])

    print(f"🏷️  Tagging {cache_tag} as {CANONICAL_IMAGE}")
    _run(["docker", "tag", cache_tag, CANONICAL_IMAGE])
    return CANONICAL_IMAGE


# Unwrap "[Action N:] execute[<cmd>]" if the caller passes a ReAct-formatted
# response straight through.
_EXECUTE_RE = re.compile(r"execute\[(.*)\]", re.DOTALL)


##############################################################################
# Pydantic models (mirror OpenEnv's env_server/types.py)
##############################################################################

class ResetRequest(BaseModel):
    """OpenEnv ResetRequest, extended with the intercode-specific `task_id`."""
    task_id: int = Field(..., description="Global 1-indexed task id (1..200)")
    seed: Optional[int] = None
    episode_id: Optional[str] = None


class Action(BaseModel):
    command: str = Field(
        ...,
        description=(
            "Bash command to execute, or 'submit' to terminate the episode "
            "and trigger the gold-comparison reward. ReAct wrappers like "
            "'execute[<cmd>]' or 'Action N: execute[<cmd>]' are accepted."
        ),
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class StepRequest(BaseModel):
    action: Action


class Observation(BaseModel):
    done: bool = False
    reward: Optional[float] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResetResponse(BaseModel):
    observation: dict[str, Any]
    reward: Optional[float] = 0.0
    done: bool = False


class StepResponse(BaseModel):
    observation: dict[str, Any]
    reward: Optional[float] = None
    done: bool = False


class State(BaseModel):
    episode_id: Optional[str] = None
    step_count: int = 0
    task_id: Optional[int] = None
    fs_version: Optional[int] = None
    local_task_id: Optional[int] = None
    done: bool = False
    last_reward: Optional[float] = None


class EnvironmentMetadata(BaseModel):
    name: str
    description: str
    version: str
    author: Optional[str] = None


##############################################################################
# Session — single BashEnv guarded by a lock
##############################################################################

class IntercodeSession:
    """Holds the active BashEnv. BashEnv binds a fixed docker container name,
    so we only keep one at a time and rebuild it when the fs version changes."""

    def __init__(self):
        self._lock = threading.Lock()
        self._env = None  # intercode.envs.BashEnv
        self._data_paths: dict[int, Path] = {}
        self._ranges: list[tuple[int, int, int]] = []
        self._loaded_fs: Optional[int] = None
        self._loaded_data_path: Optional[Path] = None

        self.episode_id: Optional[str] = None
        self.step_count: int = 0
        self.task_id: Optional[int] = None
        self.fs_version: Optional[int] = None
        self.local_task_id: Optional[int] = None
        self.done: bool = False
        self.last_reward: Optional[float] = None
        self.last_query: Optional[str] = None

    def warmup(self) -> None:
        """Download data files (small) so /reset can resolve task ids without
        racing on first call."""
        self._data_paths = fetch_all_datasets()
        self._ranges = compute_fs_ranges(self._data_paths)
        print(f"📚 Loaded NL2Bash data: ranges={self._ranges}")

    def total_tasks(self) -> int:
        return self._ranges[-1][2] if self._ranges else 0

    def _ensure_env_for_fs(self, fs_version: int) -> None:
        """(Re)create BashEnv if the loaded fs differs from `fs_version`."""
        from intercode.envs import BashEnv  # type: ignore

        if self._env is not None and self._loaded_fs == fs_version:
            return

        if self._env is not None:
            try:
                self._env.close()
            except Exception as exc:  # noqa: BLE001
                print(f"⚠️  BashEnv.close() raised: {exc}")
            self._env = None

        image = ensure_image(fs_version)  # builds + tags as intercode-nl2bash
        data_path = self._data_paths[fs_version]
        print(f"🚀 Starting BashEnv(image={image}, data={data_path.name})")
        self._env = BashEnv(image, data_path=str(data_path), verbose=False)
        self._loaded_fs = fs_version
        self._loaded_data_path = data_path

    def reset(self, task_id: int, episode_id: Optional[str]) -> dict[str, Any]:
        with self._lock:
            if not self._ranges:
                self.warmup()
            total = self.total_tasks()
            if not (1 <= task_id <= total):
                raise HTTPException(
                    status_code=422,
                    detail=f"task_id {task_id} out of range (valid 1..{total})",
                )
            try:
                fs_version, local_id = map_task_id(task_id, self._ranges)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc))
            self._ensure_env_for_fs(fs_version)

            assert self._env is not None
            self._env.reset(local_id)
            query = self._env.query

            self.episode_id = episode_id or uuid.uuid4().hex
            self.step_count = 0
            self.task_id = task_id
            self.fs_version = fs_version
            self.local_task_id = local_id
            self.done = False
            self.last_reward = None
            self.last_query = query

            return {
                "query": query,
                "task_id": task_id,
                "fs_version": fs_version,
                "local_task_id": local_id,
                "episode_id": self.episode_id,
                "instructions": (
                    "Reply with a bash command via {action: {command: ...}}. "
                    "Send command='submit' to terminate the episode and "
                    "receive the gold-comparison reward."
                ),
            }

    def step(self, command: str) -> tuple[dict[str, Any], float, bool]:
        with self._lock:
            if self._env is None or self.task_id is None:
                raise HTTPException(
                    status_code=409, detail="call /reset before /step",
                )
            if self.done:
                raise HTTPException(
                    status_code=409,
                    detail="episode already terminated; call /reset to start a new one",
                )

            raw = command
            action_parsed, is_code = bash_parser_react(command)
            # InterCode's parser only matches `execute[...]` or `submit`. Be a
            # bit more forgiving: if the caller passed a plain bash command
            # without the wrapper, treat it as the command itself.
            if not is_code and action_parsed.strip() and action_parsed != "submit":
                m = _EXECUTE_RE.search(action_parsed)
                if m:
                    action_parsed = m.group(1)
                    is_code = True
                else:
                    action_parsed = action_parsed.strip()
                    is_code = True

            if not is_code or not action_parsed:
                obs_payload = {
                    "output": (
                        "Error: empty action. Send {action:{command:'<bash>'}} "
                        "or {action:{command:'submit'}}."
                    ),
                    "action_executed": False,
                    "command": raw,
                    "command_parsed": action_parsed,
                }
                self.step_count += 1
                return obs_payload, 0.0, False

            output, reward, done, info = self._env.step(action_parsed)
            if isinstance(output, str) and len(output) > OBS_TRUNCATE_CHARS:
                output = output[:OBS_TRUNCATE_CHARS]
            elif isinstance(output, list) and len(output) > 25:
                output = output[:25]

            action_executed = (
                bool(info.get("action_executed", False))
                if isinstance(info, dict) else False
            )

            reward_f = float(reward) if reward is not None else 0.0
            self.step_count += 1
            self.done = bool(done)
            self.last_reward = reward_f

            obs_payload = {
                "output": output if isinstance(output, str) else str(output),
                "action_executed": action_executed,
                "command": raw,
                "command_parsed": action_parsed,
            }
            return obs_payload, reward_f, self.done

    def state(self) -> State:
        with self._lock:
            return State(
                episode_id=self.episode_id,
                step_count=self.step_count,
                task_id=self.task_id,
                fs_version=self.fs_version,
                local_task_id=self.local_task_id,
                done=self.done,
                last_reward=self.last_reward,
            )

    def close(self) -> None:
        with self._lock:
            if self._env is not None:
                try:
                    self._env.close()
                except Exception:
                    pass
                self._env = None


##############################################################################
# FastAPI app
##############################################################################

def create_app(warm: bool = True) -> FastAPI:
    app = FastAPI(
        title=ENV_NAME,
        description=ENV_DESCRIPTION,
        version=ENV_VERSION,
    )
    session = IntercodeSession()
    if warm:
        session.warmup()
    app.state.session = session

    @app.on_event("shutdown")
    def _shutdown() -> None:
        session.close()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metadata", response_model=EnvironmentMetadata)
    def metadata() -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name=ENV_NAME,
            description=ENV_DESCRIPTION,
            version=ENV_VERSION,
            author="princeton-nlp/intercode (wrapped)",
        )

    @app.get("/state", response_model=State)
    def get_state() -> State:
        return session.state()

    @app.get("/schema")
    def schema() -> dict[str, Any]:
        return {
            "action": Action.model_json_schema(),
            "observation": {
                "type": "object",
                "properties": {
                    "output": {"type": "string"},
                    "action_executed": {"type": "boolean"},
                    "command": {"type": "string"},
                    "command_parsed": {"type": "string"},
                    "query": {"type": "string"},
                    "task_id": {"type": "integer"},
                    "fs_version": {"type": "integer"},
                    "local_task_id": {"type": "integer"},
                    "episode_id": {"type": "string"},
                },
            },
            "state": State.model_json_schema(),
            "reset_request": ResetRequest.model_json_schema(),
        }

    @app.post("/reset", response_model=ResetResponse)
    def reset(req: ResetRequest) -> ResetResponse:
        obs = session.reset(task_id=req.task_id, episode_id=req.episode_id)
        return ResetResponse(observation=obs, reward=0.0, done=False)

    @app.post("/step", response_model=StepResponse)
    def step(req: StepRequest) -> StepResponse:
        obs, reward, done = session.step(req.action.command)
        return StepResponse(observation=obs, reward=reward, done=done)

    return app


##############################################################################
# Entry point
##############################################################################

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="OpenEnv-compatible HTTP server for InterCode NL2Bash.",
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument(
        "--no-warm", action="store_true",
        help="Skip warm-up data download/preflight on boot.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    import uvicorn  # type: ignore
    app = create_app(warm=not args.no_warm)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
