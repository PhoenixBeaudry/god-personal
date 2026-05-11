"""Collect Intercode-Bash token/cost statistics by running ReAct episodes against the Chutes API.

Parallel to scripts/swe_env_sample_finder.py. Two structural differences:
  - There is no HTTP env server. `intercode-bench` is a Python package that spawns
    a Docker container under the hood (`BashEnv`), so the agent loop runs in-process
    here and calls Chutes' OpenAI-compatible endpoint directly.
  - Runs sequentially: BashEnv hard-codes container names as
    `{image_name}_ic_ctr` / `{image_name}_ic_ctr_eval`, so two envs on the same
    image would collide.

Prereqs:
    pip install intercode-bench openai docker
    # Build the default nl2bash image (file system 1, baked in at build time):
    git clone https://github.com/princeton-nlp/intercode.git /tmp/intercode
    docker build -t intercode-nl2bash -f /tmp/intercode/docker/nl2bash.Dockerfile /tmp/intercode

NOTE on per-fs evaluation: the stock `intercode-nl2bash` image only contains the
`fs_1` filesystem (see docker/nl2bash.Dockerfile — `file_system_version=1`). For
faithful rewards on fs_2/3, build per-fs images (edit the Dockerfile to take
`file_system_version` as a build arg, then build `intercode-nl2bash-fs{N}`) and
point DATASETS at them below. fs_4 is filesystem-agnostic.
"""

import json
import os
import re
import statistics
import time
import urllib.request
from pathlib import Path

from intercode.envs import BashEnv
from openai import OpenAI

# --- Chutes API Configuration ---
MODEL_NAME = "Qwen/Qwen2.5-Coder-32B-Instruct"
MODEL_BASE_URL = "https://llm.chutes.ai/v1/"
MODEL_API_KEY_ENV = "CHUTES_API_TOKEN"

# --- Evaluation Configuration ---
TEMPERATURE = 0.7
MAX_TURNS = 10                # per-task ReAct turn budget
OBS_TRUNCATE_CHARS = 350      # match intercode/experiments/eval_react.py
MAX_TOKENS_PER_CALL = 512
NUM_EVALS_PER_DATASET = None  # None = run every task in each dataset

# Each tuple: (logical name, dataset URL, BashEnv image_name)
INTERCODE_DATA_RAW = "https://raw.githubusercontent.com/princeton-nlp/intercode/master/data/nl2bash"
DATASETS = [
    ("fs_1", f"{INTERCODE_DATA_RAW}/nl2bash_fs_1.json", "intercode-nl2bash"),
    ("fs_2", f"{INTERCODE_DATA_RAW}/nl2bash_fs_2.json", "intercode-nl2bash"),
    ("fs_3", f"{INTERCODE_DATA_RAW}/nl2bash_fs_3.json", "intercode-nl2bash"),
    ("fs_4", f"{INTERCODE_DATA_RAW}/nl2bash_fs_4.json", "intercode-nl2bash"),
]

OUTPUT_ROOT = Path(__file__).resolve().parent / "intercode_bash_chutes_stats"
DATA_DIR = OUTPUT_ROOT / "data"

##############################################################################################

# Verbatim from princeton-nlp/intercode experiments/utils/prompts.py (DEMO_BASH_REACT
# + TemplateReAct.get_init_msg). Kept identical so token/turn numbers stay
# apples-to-apples with the InterCode paper.
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

# Verbatim from princeton-nlp/intercode experiments/utils/utils.py (bash_parser_react)
_REACT_ACTION_RE = re.compile(r"execute\[(.*)\]", re.DOTALL)


def bash_parser_react(action):
    if action == "submit":
        return action, True
    matches = _REACT_ACTION_RE.findall(action)
    if matches:
        return matches[0], True
    return action, False


def _safe_int(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def _safe_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _percentile(values, q):
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _stat(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return {
        "n": len(vals),
        "mean": statistics.mean(vals),
        "median": statistics.median(vals),
        "min": min(vals),
        "max": max(vals),
        "p90": _percentile(vals, 0.9),
        "p95": _percentile(vals, 0.95),
        "stdev": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "sum": sum(vals),
    }


def compute_summary(per_task):
    passed = [r for r in per_task if r.get("success")]
    failed = [r for r in per_task if not r.get("success")]

    def slice_stats(rows):
        return {
            "count": len(rows),
            "reward": _stat([r.get("reward") for r in rows]),
            "num_model_calls": _stat([r.get("num_model_calls") for r in rows]),
            "tokens_per_call_avg": _stat([
                (r["total_tokens"] / r["num_model_calls"])
                if r.get("num_model_calls") else None
                for r in rows
            ]),
            "total_tokens": _stat([r.get("total_tokens") for r in rows]),
            "time_taken": _stat([r.get("time_taken") for r in rows]),
        }

    return {
        "all": slice_stats(per_task),
        "passed": slice_stats(passed),
        "failed": slice_stats(failed),
        "pass_rate": len(passed) / len(per_task) if per_task else 0.0,
    }


def print_summary(summary):
    print("\n" + "=" * 72)
    print("Intercode-Bash Token Statistics Summary")
    print("=" * 72)
    print(
        f"Pass rate (reward >= 1.0): {summary['pass_rate']:.2%} "
        f"({summary['passed']['count']}/{summary['all']['count']})"
    )
    keys = ("reward", "tokens_per_call_avg", "num_model_calls", "time_taken")
    for slice_name in ("all", "passed", "failed"):
        s = summary[slice_name]
        if s["count"] == 0:
            continue
        print(f"\n[{slice_name.upper()}] n={s['count']}")
        for k in keys:
            v = s.get(k)
            if not v:
                continue
            print(
                f"  {k:22s}  mean={v['mean']:>10.2f}  median={v['median']:>10.2f}"
                f"  p90={v['p90']:>10.2f}  max={v['max']:>10.2f}"
            )
    print("=" * 72)


def _load_state(state_path):
    if state_path.exists():
        try:
            data = json.loads(state_path.read_text())
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {"tested_task_ids": [], "per_task": []}


def _save_state(state_path, state):
    tmp = state_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, default=str, indent=2))
    tmp.replace(state_path)


def ensure_data_files():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    resolved = []
    for name, url, image in DATASETS:
        local = DATA_DIR / Path(url).name
        if not local.exists():
            print(f"📥 Downloading {url} -> {local}")
            urllib.request.urlretrieve(url, local)
        resolved.append((name, local, image))
    return resolved


def _chutes_chat(client, prompt, stop_seqs, max_tokens):
    """Single Chutes call. Returns (text, prompt_tokens, completion_tokens)."""
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE,
        stop=stop_seqs,
        max_tokens=max_tokens,
    )
    text = resp.choices[0].message.content or ""
    usage = getattr(resp, "usage", None)
    pt = _safe_int(getattr(usage, "prompt_tokens", None)) or 0
    ct = _safe_int(getattr(usage, "completion_tokens", None)) or 0
    return text, pt, ct


def run_react_episode(env, query, client):
    """Run a ReAct loop for one task. Returns a dict of episode stats + trajectory."""
    prompt = REACT_INIT_MSG + DEMO_BASH_REACT + f"Question: {query}\n"
    total_pt = 0
    total_ct = 0
    num_calls = 0
    per_call_tokens = []
    turn_history = {
        "thoughts": [], "actions": [], "actions_parsed": [],
        "observations": [], "rewards": [], "valid_action": [],
    }
    reward = 0.0
    done = False
    turn = 0
    valid_action = False

    for turn in range(1, MAX_TURNS + 1):
        thought_action, pt, ct = _chutes_chat(
            client,
            prompt + f"Thought {turn}:",
            stop_seqs=[f"\nObservation {turn}:"],
            max_tokens=MAX_TOKENS_PER_CALL,
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
            observation = "Error executing query: Your last `execute` action did not contain bash code"
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

        step_str = f"Thought {turn}: {thought}\nAction {turn}: {action}\nObservation {turn}: {observation}\n"
        prompt += step_str

        turn_history["thoughts"].append(thought)
        turn_history["actions"].append(action)
        turn_history["actions_parsed"].append(action_parsed)
        turn_history["observations"].append(str(observation))
        turn_history["rewards"].append(reward)
        turn_history["valid_action"].append(valid_action)

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


def run_dataset(fs_name, data_path, image_name, client):
    output_dir = OUTPUT_ROOT / fs_name
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks_dir = output_dir / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "state.json"
    state = _load_state(state_path)
    per_task = list(state.get("per_task", []))
    tested_set = set(state.get("tested_task_ids", []))

    print(f"📁 [{fs_name}] output dir: {output_dir}")
    print(f"📦 [{fs_name}] loaded state: {len(tested_set)} task(s) already tested")
    print(f"🚀 [{fs_name}] starting BashEnv(image={image_name}, data_path={data_path.name})...")
    env = BashEnv(image_name, data_path=str(data_path), verbose=False)

    try:
        num_tasks = len(env.data_loader)
        eval_list = [i for i in range(num_tasks) if i not in tested_set]
        if NUM_EVALS_PER_DATASET is not None:
            eval_list = eval_list[:NUM_EVALS_PER_DATASET]
        if not eval_list:
            print(f"✅ [{fs_name}] all {num_tasks} task(s) already tested.")
            return per_task
        print(
            f"▶️  [{fs_name}] resuming from task {eval_list[0]} — "
            f"running {len(eval_list)}/{num_tasks} task(s)"
        )

        for completed_idx, task_id in enumerate(eval_list, 1):
            print(f"Testing Task ID: {fs_name}/{task_id}")
            start = time.perf_counter()
            error = None
            episode = {}
            query = None
            try:
                env.reset(task_id)
                query = env.query
                episode = run_react_episode(env, query, client)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            elapsed = time.perf_counter() - start

            reward = _safe_float(episode.get("reward")) or 0.0
            success = reward >= 1.0
            calls = episode.get("num_calls", 0) or 0
            total_tokens = episode.get("total_tokens", 0) or 0

            row = {
                "fs": fs_name,
                "task_id": task_id,
                "query": query,
                "reward": reward,
                "success": success,
                "time_taken": elapsed,
                "num_model_calls": calls,
                "input_tokens": episode.get("input_tokens", 0),
                "output_tokens": episode.get("output_tokens", 0),
                "total_tokens": total_tokens,
                "per_call_tokens": episode.get("per_call_tokens", []),
                "num_turns": episode.get("num_turns", 0),
                "error": error,
            }
            per_task.append(row)
            tested_set.add(task_id)

            (tasks_dir / f"task_{task_id}.json").write_text(json.dumps({
                "row": row,
                "turn_history": episode.get("turn_history"),
            }, indent=2, default=str))
            _save_state(state_path, {
                "tested_task_ids": sorted(tested_set),
                "per_task": per_task,
            })

            label = "Success" if success else "Failure"
            print(f"Task ID: {fs_name}/{task_id} completed, Result: {label}")
            if error:
                print(f"[{completed_idx}/{len(eval_list)}] {fs_name}/{task_id}: ERROR ({error})")
            else:
                print(
                    f"[{completed_idx}/{len(eval_list)}] {fs_name}/{task_id}: "
                    f"reward={reward:.2f} success={success} "
                    f"tokens={total_tokens} calls={calls} "
                    f"time={elapsed:.1f}s"
                )
    finally:
        print(f"🧹 [{fs_name}] closing BashEnv...")
        try:
            env.close()
        except Exception:
            pass

    return per_task


def run_evaluation():
    api_key = os.getenv(MODEL_API_KEY_ENV)
    if not api_key:
        raise RuntimeError(f"Set ${MODEL_API_KEY_ENV} before running.")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output root: {OUTPUT_ROOT}")
    paths = ensure_data_files()
    for name, path, image in paths:
        print(f"   {name}: data={path}, image={image}")

    client = OpenAI(api_key=api_key, base_url=MODEL_BASE_URL)

    all_per_task = []
    for fs_name, data_path, image_name in paths:
        print("\n" + "=" * 72)
        print(f"Phase: {fs_name}  ({data_path.name})  image={image_name}")
        print("=" * 72)
        per_task = run_dataset(fs_name, data_path, image_name, client)
        all_per_task.extend(per_task)

    summary = compute_summary(all_per_task)
    print_summary(summary)

    successful_tasks = []
    for r in all_per_task:
        if not r.get("success"):
            continue
        calls = r.get("num_model_calls") or 0
        total = r.get("total_tokens") or 0
        successful_tasks.append({
            "fs": r["fs"],
            "task_id": r["task_id"],
            "query": r.get("query"),
            "tokens_per_call_avg": (total / calls) if calls else None,
            "num_model_calls": calls,
            "time_taken": r.get("time_taken"),
        })
    successful_tasks.sort(key=lambda r: (r["fs"], r["task_id"]))

    (OUTPUT_ROOT / "per_task.json").write_text(json.dumps(all_per_task, default=str, indent=2))
    (OUTPUT_ROOT / "summary.json").write_text(json.dumps(summary, default=str, indent=2))
    (OUTPUT_ROOT / "successful_tasks.json").write_text(json.dumps({
        "tasks": successful_tasks,
        "count": len(successful_tasks),
    }, default=str, indent=2))
    (OUTPUT_ROOT / "config.json").write_text(json.dumps({
        "model": MODEL_NAME,
        "base_url": MODEL_BASE_URL,
        "temperature": TEMPERATURE,
        "max_turns": MAX_TURNS,
        "obs_truncate_chars": OBS_TRUNCATE_CHARS,
        "max_tokens_per_call": MAX_TOKENS_PER_CALL,
        "num_evals_per_dataset": NUM_EVALS_PER_DATASET,
        "datasets": [{"name": n, "data": str(p), "image": img} for n, p, img in paths],
    }, default=str, indent=2))

    print("\n" + "=" * 72)
    print(f"Successful tasks ({len(successful_tasks)}/{len(all_per_task)})")
    print("=" * 72)
    for r in successful_tasks:
        tpc = r["tokens_per_call_avg"]
        tpc_str = f"{tpc:>10.2f}" if tpc is not None else "       n/a"
        tt = r["time_taken"]
        tt_str = f"{tt:>8.2f}s" if tt is not None else "     n/a"
        print(
            f"  {r['fs']} task {r['task_id']:>4}  "
            f"tokens_per_call_avg={tpc_str}  "
            f"num_model_calls={r['num_model_calls']:>4}  "
            f"time_taken={tt_str}"
        )

    print(f"\n📁 Wrote results to {OUTPUT_ROOT}")
    return all_per_task


if __name__ == "__main__":
    print("Running Chutes Intercode-Bash token-stats eval (sequential, resumable)")
    start = time.perf_counter()
    run_evaluation()
    print(f"Took {time.perf_counter() - start:.1f}s")
