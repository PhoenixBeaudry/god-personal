#!/usr/bin/env python3
"""Run a single Intercode-Bash NL2Bash task against the Chutes API.

Intercode (https://github.com/princeton-nlp/intercode) ships NL2Bash with
four filesystem variants. Global task IDs are 1-indexed and packed left-to-
right across the four variants. With current upstream counts:
    fs_1: global ids   1..60  (60 tasks, file_system_version=1)
    fs_2: global ids  61..113 (53 tasks, file_system_version=2)
    fs_3: global ids 114..173 (60 tasks, file_system_version=3)
    fs_4: global ids 174..200 (27 tasks, filesystem-agnostic)
Ranges are recomputed at runtime from the downloaded data files, so this
mapping stays correct if upstream sizes change.

What the script does:
  1. Downloads nl2bash_fs_{1..4}.json into a cache dir (skipped if present).
  2. Builds an `intercode-nl2bash-fs{N}` docker image on demand by cloning
     the intercode repo once and patching the `file_system_version` ENV in
     its Dockerfile so each filesystem variant gets its own image. fs_4 is
     filesystem-agnostic and reuses the fs_1 image.
  3. Spins up `intercode.envs.BashEnv` on that image, resets to the local
     task index, and runs a ReAct loop (same prompts as the InterCode
     paper) using Chutes' OpenAI-compatible `/v1/chat/completions` endpoint.
  4. Prints reward + token/turn stats and (optionally) writes a JSON dump.

Prereqs:
    pip install intercode-bench openai
    docker daemon running locally (used by intercode.envs.BashEnv)
    git available on $PATH (used to clone the intercode repo once)
    export CHUTES_API_TOKEN=...

Usage:
    python scripts/intercode_eval.py 5
    python scripts/intercode_eval.py 120 --max-turns 15 --output run.json
    python scripts/intercode_eval.py 5 --model Qwen/Qwen2.5-Coder-32B-Instruct
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# --- Chutes / model defaults (override via CLI) ---------------------------
DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-32B-Instruct"
CHUTES_BASE_URL = "https://llm.chutes.ai/v1/"
CHUTES_API_KEY_ENV = "CHUTES_API_TOKEN"

DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TURNS = 10
DEFAULT_MAX_TOKENS_PER_CALL = 512
OBS_TRUNCATE_CHARS = 350  # matches intercode/experiments/eval_react.py

INTERCODE_GIT_URL = "https://github.com/princeton-nlp/intercode.git"
INTERCODE_DATA_RAW = (
    "https://raw.githubusercontent.com/princeton-nlp/intercode/master/data/nl2bash"
)

CACHE_DIR = Path.home() / ".cache" / "intercode_eval"
REPO_CACHE = CACHE_DIR / "intercode"
DATA_CACHE = CACHE_DIR / "data"

# BashEnv hard-codes `intercode-nl2bash` in its IMAGE_TO_SETTINGS lookup, so
# whatever fs-specific image we build must be tagged with that exact name
# before we hand it to BashEnv. We keep per-fs tags around for caching and
# re-tag the active one as `intercode-nl2bash` per run.
CANONICAL_IMAGE = "intercode-nl2bash"
CACHE_TAG_FOR_FS = {1: "intercode-nl2bash-fs1",
                    2: "intercode-nl2bash-fs2",
                    3: "intercode-nl2bash-fs3",
                    4: "intercode-nl2bash-fs1"}  # fs_4 is filesystem-agnostic

# Filesystem version baked into the docker image -- fs_4 uses fs_1's image,
# so we build fs_version=1 for it (the dataset's commands don't depend on the
# filesystem layout).
BUILD_FS_VERSION = {1: 1, 2: 2, 3: 3, 4: 1}


##############################################################################
# ReAct prompt -- verbatim from princeton-nlp/intercode experiments/utils/
# prompts.py so token/turn counts stay comparable to the paper.
##############################################################################

REACT_INIT_MSG = """Interact with a Bourne Shell system using Bash queries to answer a question.
Use interleaving Thought, Action, Observation steps.
Thought can reason about the current situation, and Action can be two types:
(1) execute, which executes Bash code on the Bourne Shell system
(2) submit, which indicates that the previous observation is the answer
Here are some examples.
"""

DEMO_BASH_REACT = """Question: Convert the first 16 characters in "/testbed/textfile7.txt" to a single hexadecimal value
Thought 1: I can use the `xxd` command to get a hexdump of the /textbed/textfile7.txt file
Action 1: execute[xxd -p -l 16 /testbed/textfile7.txt]
Observation 1: /bin/bash: line 1: xxd: command not found
Thought 2: Seems like xxd is not available. I can try to use `od` (octal dump) command to get a hexadecimal dump.
Action 2: execute[od -A n -t x1 -N 16 /testbed/textfile7.txt]
Observation 2:  54 65 78 74 20 66 69 6c 65 20 37 20 77 69 74 68
Thought 3: This command is showing the hexadecimal representation of the first 16 bytes, but it is not a single value.
I should try stitching together the values into one continuous string.
Action 3: execute[od -A n -t x1 -N 16 /testbed/textfile7.txt | awk '{$1=$1;print}' | tr -d ' ']
Observation 3: 546578742066696c6520372077697468
Thought 4: I've configured
Action 4: submit
Question: Calculate a list of duplicate md5 sum hashes for all the ".java" files in the /testbed directory
Thought 1: Find all files with the `.java` extension in `testbed/`, then calculate the md5hash, sort them,
filter out duplicates, and only pick the first line of the hash.
Action 1: execute[find /testbed -name "*.java" -type f -exec md5sum {} + | sort | uniq -d | cut -d' ' -f1]
Observation 1:
Thought 2: Seems like this did not produce any results. I may have to use another command instead of cut
to pick out the hashes from the output string
Action 2: execute[find /testbed -name "*.java" -type f -exec md5sum {} + | sort | uniq -Dw 32]
Observation 2: f32a3a97638afeb2ee2a15cfe335ab72  /testbed/Hello.java
f32a3a97638afeb2ee2a15cfe335ab72  /testbed/Hello1.java
f32a3a97638afeb2ee2a15cfe335ab72  /testbed/dir1/AnotherHello.java
Thought 3: This shows the output hashes and they have the same values, indicating that these files are duplicates.
However, the file names are also shown, which are not needed.
Action 3: execute[find /testbed -name "*.java" -type f -exec md5sum {} + | sort | uniq -D -w 32 | cut -c 35-]
Observation 3: /testbed/Hello.java
/testbed/Hello1.java
/testbed/dir1/AnotherHello.java
Thought 4: This shows the file names exclusively, and no longer shows the hashes. It seems that the cut
command argument may not be the best choice for selecting file names.
Action 4: execute[find /testbed -name "*.java" -type f -exec md5sum {} + | sort | uniq -D -w 32 | awk '{print $2}']
Observation 4: /testbed/Hello.java
/testbed/Hello1.java
/testbed/dir1/AnotherHello.java
Thought 5: I use the awk command instead, but instead of printing out the hashes, it still prints out the file
names. I should select a different part of the output string instead of `$2`
Action 5: execute[find /testbed -name "*.java" -type f -exec md5sum {} + | sort | uniq -D -w 32 | awk '{print $1}']
Observation 5: f32a3a97638afeb2ee2a15cfe335ab72
f32a3a97638afeb2ee2a15cfe335ab72
f32a3a97638afeb2ee2a15cfe335ab72
Thought 6: This prints out identical hashes, and based on previous observations, I know that these are hashes of
duplicates `.java` files from the `testbed/` directory. This should be correct. I will submit.
Action 6: submit
Question: print disk usage in human readable format of files or folders in /workspace
Thought 1: The `du` command is useful for printing out disk usage of a specific directory. I can use this to
display this information for the `workspace` directory
Action 1: execute[du /workspace]
Observation 1: 48\t/workspace/dir1
8\t/workspace/dir2/mysql
24\t/workspace/dir2
100\t/workspace
Thought 2: The default `du` command gives storage in a non-human readble font. I can use the -h option
of the du command to print storage size with bytes.
Action 2: execute[du -h /workspace]
Observation 2: 48K\t/workspace/dir1
8.0K\t/workspace/dir2/mysql
24K\t/workspace/dir2
100K\t/workspace
Thought 3: This gives me storage information for every folder under the workspace directory, but
I only need the storage for just the `workspace/` directory. The `-s` option should help with this.
Action 3: execute[du -sh /workspace]
Observation 3: 100K\t/workspace
Thought 4: This shows data usage in human readable format for the `workspace` directory. I am finished.
Action 4: submit
Question: Count all the lines of all php files in the /testbed directory recursively
Thought 1: I should find the paths to all php files in the testbed directory, then apply the word
count command to each path.
Action 1: execute[find /testbed -name "*.php" | xargs wc -l]
Observation 1:  1 /testbed/dir1/info.php
 1 /testbed/hello.php
 2 total
Thought 2: This shows me too much information, I only need the total number of lines. I should add up
the lines together and output a single number.
Action 2: execute[find /testbed -name "*.php" -exec wc -l {} + | awk '{total += $1} END{print total}']
Observation 2: 4
Thought 3: This total is wrong, it doesn't match the previous observation, where total is 2. I only
need to apply the word count command.
Action 3: execute[find /testbed -name "*.php" -type f -exec cat {} + | wc -l]
Observation 3: 2
Thought 4: The value is 2, which matches the initial observation that the total lines of php files in the
testbed directory is 2. I can submit.
Action 4: submit
Question: Create a hello.txt file in the /testbed directory and add the text "Hello world" to it.
Thought 1: I can first create a `hello.txt` file in the `testbed/` directory
Action 1: touch testbed/hello.txt
Observation 1:
Thought 2: I should check that the file was created successfully.
Action 2: execute[ls testbed/]
Observation 2: dir1/
dir2/
dir3/
hello.txt
files.txt
Thought 3: I can now add the "Hello world" text to the hello.txt file
Action 3: execute[echo Hello world > hello.txt]
Observation 3:
Thought 4: I should check that the text was written successfully to the hello.txt file.
Action 4: execute[cat testbed/hello.txt]
Observation 4: Hello world
Thought 5: The hello.txt file has been created successfully in the testbed/ directory, and it contains
the Hello World text. I can submit.
Action 5: submit
"""

# Verbatim from princeton-nlp/intercode experiments/utils/utils.py
_REACT_ACTION_RE = re.compile(r"execute\[(.*)\]", re.DOTALL)


def bash_parser_react(action: str):
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
    """Return {fs_version: local data file path} for fs_1..fs_4."""
    paths: dict[int, Path] = {}
    for fs in (1, 2, 3, 4):
        local = DATA_CACHE / f"nl2bash_fs_{fs}.json"
        _download(f"{INTERCODE_DATA_RAW}/nl2bash_fs_{fs}.json", local)
        paths[fs] = local
    return paths


def compute_fs_ranges(data_paths: dict[int, Path]) -> list[tuple[int, int, int]]:
    """Return list of (fs_version, start_global_id, end_global_id) — inclusive,
    1-indexed."""
    ranges: list[tuple[int, int, int]] = []
    cursor = 1
    for fs in (1, 2, 3, 4):
        n = len(json.loads(data_paths[fs].read_text()))
        ranges.append((fs, cursor, cursor + n - 1))
        cursor += n
    return ranges


def map_task_id(global_id: int, ranges) -> tuple[int, int]:
    """Map 1-indexed global id -> (fs_version, 0-indexed local id within fs)."""
    for fs, start, end in ranges:
        if start <= global_id <= end:
            return fs, global_id - start
    total = ranges[-1][2]
    raise SystemExit(f"task id {global_id} out of range; valid range is 1..{total}")


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd, check=True, **kwargs)


def ensure_intercode_repo() -> Path:
    if REPO_CACHE.exists() and (REPO_CACHE / "docker" / "nl2bash.Dockerfile").exists():
        return REPO_CACHE
    REPO_CACHE.parent.mkdir(parents=True, exist_ok=True)
    if REPO_CACHE.exists():
        shutil.rmtree(REPO_CACHE)
    _run(["git", "clone", "--depth", "1", INTERCODE_GIT_URL, str(REPO_CACHE)])
    return REPO_CACHE


def image_exists(image: str) -> bool:
    res = subprocess.run(
        ["docker", "image", "inspect", image],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return res.returncode == 0


def ensure_image(fs_version: int) -> str:
    """Build (if missing) the per-fs image and tag it as the canonical
    `intercode-nl2bash` name that BashEnv expects. Returns CANONICAL_IMAGE."""
    cache_tag = CACHE_TAG_FOR_FS[fs_version]
    if not image_exists(cache_tag):
        repo = ensure_intercode_repo()
        src_dockerfile = repo / "docker" / "nl2bash.Dockerfile"
        if not src_dockerfile.exists():
            raise SystemExit(f"missing Dockerfile at {src_dockerfile}")

        build_fs = BUILD_FS_VERSION[fs_version]
        patched = src_dockerfile.read_text().replace(
            "ENV file_system_version=1",
            f"ENV file_system_version={build_fs}",
        )
        # Write patched Dockerfile alongside the original so the
        # `COPY ../docker/bash_scripts/$script /` relative path resolves the
        # same way as upstream.
        patched_path = src_dockerfile.with_name(f"nl2bash.fs{build_fs}.Dockerfile")
        patched_path.write_text(patched)

        print(f"🐳 Building docker image {cache_tag} (fs_version={build_fs})…")
        _run([
            "docker", "build",
            "-t", cache_tag,
            "-f", str(patched_path),
            str(repo),
        ])

    # BashEnv only recognises the canonical name. Re-tag the cached per-fs
    # image as `intercode-nl2bash` for this run.
    print(f"🏷️  Tagging {cache_tag} as {CANONICAL_IMAGE}")
    _run(["docker", "tag", cache_tag, CANONICAL_IMAGE])
    return CANONICAL_IMAGE


##############################################################################
# ReAct loop against Chutes
##############################################################################

def _chutes_chat(client, prompt: str, stop_seqs, max_tokens: int):
    resp = client.chat.completions.create(
        model=client._intercode_model,  # stashed below
        messages=[{"role": "user", "content": prompt}],
        temperature=client._intercode_temperature,
        stop=stop_seqs,
        max_tokens=max_tokens,
    )
    text = resp.choices[0].message.content or ""
    usage = getattr(resp, "usage", None)
    pt = int(getattr(usage, "prompt_tokens", 0) or 0)
    ct = int(getattr(usage, "completion_tokens", 0) or 0)
    return text, pt, ct


def run_react_episode(env, query: str, client, max_turns: int, max_tokens_per_call: int):
    prompt = REACT_INIT_MSG + DEMO_BASH_REACT + f"Question: {query}\n"
    total_pt = 0
    total_ct = 0
    num_calls = 0
    per_call_tokens: list[int] = []
    turn_history = {
        "thoughts": [], "actions": [], "actions_parsed": [],
        "observations": [], "rewards": [], "valid_action": [],
    }
    reward = 0.0
    done = False
    turn = 0
    valid_action = False

    for turn in range(1, max_turns + 1):
        thought_action, pt, ct = _chutes_chat(
            client,
            prompt + f"Thought {turn}:",
            stop_seqs=[f"\nObservation {turn}:"],
            max_tokens=max_tokens_per_call,
        )
        num_calls += 1
        total_pt += pt
        total_ct += ct
        per_call_tokens.append(pt + ct)

        text = thought_action.strip()
        try:
            thought, action = text.split(f"\nAction {turn}: ", 1)
        except ValueError:
            thought = text.split("\n")[0]
            action_text, pt, ct = _chutes_chat(
                client,
                prompt + f"Thought {turn}: {thought}\nAction {turn}:",
                stop_seqs=["\n"],
                max_tokens=256,
            )
            num_calls += 1
            total_pt += pt
            total_ct += ct
            per_call_tokens.append(pt + ct)
            action = action_text.strip()

        action_parsed, is_code = bash_parser_react(action)
        if not is_code:
            observation = (
                "Error executing query: Your last `execute` action did not "
                "contain bash code"
            )
            reward = 0.0
            valid_action = False
            done = False
        else:
            observation, reward, done, info = env.step(action_parsed)
            valid_action = bool(info.get("action_executed", False)) if isinstance(info, dict) else False

        if isinstance(observation, str) and len(observation) > OBS_TRUNCATE_CHARS:
            observation = observation[:OBS_TRUNCATE_CHARS]
        elif isinstance(observation, list) and len(observation) > 25:
            observation = observation[:25]

        step_str = (
            f"Thought {turn}: {thought}\n"
            f"Action {turn}: {action}\n"
            f"Observation {turn}: {observation}\n"
        )
        prompt += step_str

        turn_history["thoughts"].append(thought)
        turn_history["actions"].append(action)
        turn_history["actions_parsed"].append(action_parsed)
        turn_history["observations"].append(str(observation))
        turn_history["rewards"].append(reward)
        turn_history["valid_action"].append(valid_action)

        print(f"  turn {turn}: action={action_parsed!r}  reward={reward}")
        if done:
            break

    if not done:
        observation, reward, done, info = env.step("submit")
        turn_history["thoughts"].append("EXCEEDED MAX TURNS: submit")
        turn_history["actions"].append("submit")
        turn_history["actions_parsed"].append("submit")
        turn_history["observations"].append(str(observation))
        turn_history["rewards"].append(reward)
        turn_history["valid_action"].append(valid_action)

    return {
        "reward": float(reward) if reward is not None else 0.0,
        "num_calls": num_calls,
        "input_tokens": total_pt,
        "output_tokens": total_ct,
        "total_tokens": total_pt + total_ct,
        "per_call_tokens": per_call_tokens,
        "num_turns": turn,
        "turn_history": turn_history,
    }


##############################################################################
# Entry point
##############################################################################

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run one or more Intercode-Bash NL2Bash tasks via the Chutes API.",
    )
    p.add_argument(
        "task_id", type=int, nargs="?",
        help="Global 1-indexed task id (1..200). Omit when using --num-episodes.",
    )
    p.add_argument(
        "--num-episodes", type=int, default=None,
        help="Sample this many random task ids (without replacement) and "
             "evaluate them sequentially. Overrides positional task_id.",
    )
    p.add_argument(
        "--seed", type=int, default=None,
        help="Optional RNG seed for --num-episodes sampling (for reproducibility).",
    )
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    p.add_argument(
        "--max-tokens-per-call", type=int, default=DEFAULT_MAX_TOKENS_PER_CALL,
    )
    p.add_argument(
        "--output", type=Path, default=None,
        help="Optional path to dump per-task results + summary as JSON.",
    )
    args = p.parse_args()
    if args.task_id is None and args.num_episodes is None:
        p.error("provide a task_id or --num-episodes N")
    return args


def run_task(task_id: int, args, client, data_paths, ranges) -> dict:
    """Run a single global task_id and return its result dict (incl. turn history)."""
    from intercode.envs import BashEnv  # type: ignore

    fs_version, local_id = map_task_id(task_id, ranges)
    data_path = data_paths[fs_version]
    image = ensure_image(fs_version)

    print("=" * 72)
    print(f"task_id (global)   : {task_id}")
    print(f"filesystem variant : fs_{fs_version}  (local task index {local_id})")
    print(f"docker image       : {image}")
    print(f"data file          : {data_path}")
    print(f"model              : {args.model}")
    print("=" * 72)

    print(f"🚀 Starting BashEnv(image={image})…")
    env = BashEnv(image, data_path=str(data_path), verbose=False)
    start = time.perf_counter()
    error = None
    episode: dict = {}
    query: str | None = None
    try:
        env.reset(local_id)
        query = env.query
        print(f"Query: {query}")
        episode = run_react_episode(
            env, query, client,
            max_turns=args.max_turns,
            max_tokens_per_call=args.max_tokens_per_call,
        )
    except Exception as exc:  # noqa: BLE001 — report any failure to caller
        error = f"{type(exc).__name__}: {exc}"
        print(f"❌ {error}")
    finally:
        try:
            env.close()
        except Exception:
            pass

    elapsed = time.perf_counter() - start
    reward = float(episode.get("reward") or 0.0)
    success = reward >= 1.0

    result = {
        "task_id": task_id,
        "fs_version": fs_version,
        "local_task_id": local_id,
        "image": image,
        "model": args.model,
        "query": query,
        "reward": reward,
        "success": success,
        "time_taken": elapsed,
        "num_model_calls": episode.get("num_calls", 0),
        "input_tokens": episode.get("input_tokens", 0),
        "output_tokens": episode.get("output_tokens", 0),
        "total_tokens": episode.get("total_tokens", 0),
        "per_call_tokens": episode.get("per_call_tokens", []),
        "num_turns": episode.get("num_turns", 0),
        "error": error,
        "turn_history": episode.get("turn_history"),
    }

    print("=" * 72)
    print(f"Result: {'✅ SUCCESS' if success else '❌ FAILURE'}  reward={reward}")
    print(
        f"  turns={result['num_turns']}  calls={result['num_model_calls']}  "
        f"tokens={result['total_tokens']}  time={elapsed:.1f}s"
    )
    if error:
        print(f"  error: {error}")
    print("=" * 72)

    return result


def main() -> int:
    args = parse_args()

    api_key = os.getenv(CHUTES_API_KEY_ENV)
    if not api_key:
        print(f"error: set ${CHUTES_API_KEY_ENV} before running", file=sys.stderr)
        return 2

    from openai import OpenAI  # type: ignore

    data_paths = fetch_all_datasets()
    ranges = compute_fs_ranges(data_paths)
    total = ranges[-1][2]

    # Decide which task ids to run.
    if args.num_episodes is not None:
        if args.num_episodes < 1:
            print("error: --num-episodes must be >= 1", file=sys.stderr)
            return 2
        if args.num_episodes > total:
            print(
                f"error: --num-episodes {args.num_episodes} exceeds total tasks {total}",
                file=sys.stderr,
            )
            return 2
        rng = random.Random(args.seed)
        task_ids = rng.sample(range(1, total + 1), args.num_episodes)
        print(f"🎲 Sampled {args.num_episodes} task ids (seed={args.seed}): {task_ids}")
    else:
        if not (1 <= args.task_id <= total):
            print(
                f"error: task_id {args.task_id} out of range (valid 1..{total})",
                file=sys.stderr,
            )
            return 2
        task_ids = [args.task_id]

    client = OpenAI(api_key=api_key, base_url=CHUTES_BASE_URL)
    # Stash per-call options on the client so _chutes_chat doesn't need them
    # threaded through every call.
    client._intercode_model = args.model
    client._intercode_temperature = args.temperature

    results: list[dict] = []
    for i, tid in enumerate(task_ids, 1):
        print(f"\n────── Episode {i}/{len(task_ids)}  (task_id={tid}) ──────")
        results.append(run_task(tid, args, client, data_paths, ranges))

    # Aggregate.
    rewards = [r["reward"] for r in results]
    successes = sum(1 for r in results if r["success"])
    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
    pass_rate = successes / len(results) if results else 0.0
    total_tokens = sum(r.get("total_tokens") or 0 for r in results)
    total_time = sum(r.get("time_taken") or 0.0 for r in results)

    summary = {
        "num_episodes": len(results),
        "avg_reward": avg_reward,
        "pass_rate": pass_rate,
        "num_passed": successes,
        "total_tokens": total_tokens,
        "total_time": total_time,
        "model": args.model,
        "seed": args.seed,
        "task_ids": task_ids,
    }

    print("\n" + "=" * 72)
    print("Evaluation summary")
    print("=" * 72)
    print(f"  episodes        : {summary['num_episodes']}")
    print(f"  pass rate       : {pass_rate:.2%}  ({successes}/{len(results)})")
    print(f"  average reward  : {avg_reward:.4f}")
    print(f"  total tokens    : {total_tokens}")
    print(f"  total time      : {total_time:.1f}s")
    for r in results:
        mark = "✅" if r["success"] else "❌"
        print(
            f"   {mark} task {r['task_id']:>3} fs_{r['fs_version']}  "
            f"reward={r['reward']:.2f}  turns={r['num_turns']}  "
            f"tokens={r['total_tokens']}"
        )
    print("=" * 72)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(
            {"summary": summary, "results": results},
            default=str, indent=2,
        ))
        print(f"📁 Wrote {args.output}")

    if len(results) == 1:
        return 0 if results[0]["success"] else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
