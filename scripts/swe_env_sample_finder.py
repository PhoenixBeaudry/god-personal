"""Collect SWE-infinite token/cost statistics by running episodes against the Chutes API.

Mirrors scripts/swe_env_eval.py but skips the local SGLang server — the env server
is pointed at Chutes' OpenAI-compatible endpoint instead. Per-task results
(including the full conversation) are dumped to disk; aggregate stats are split
by pass vs fail so we can compare token/turn profiles between the two.

Assumes the in-container agent (codex or miniswe) talks to its model via an
OpenAI-compatible client.
"""

import json
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import docker
import requests

# --- Chutes API Configuration ---
MODEL_NAME = "Qwen/Qwen2.5-Coder-32B-Instruct"
MODEL_BASE_URL = "https://llm.chutes.ai/v1/"
MODEL_API_KEY_ENV = "CHUTES_API_TOKEN"

# --- Evaluation Configuration ---
NUM_EVALS = 100
TEMPERATURE = 0.7
AGENT_TYPE = "miniswe"          # "miniswe", "codex", or "" for auto-select
MAX_ITERATIONS = 100          # miniswe only
EVAL_TIMEOUT = 1800           # per-task timeout (seconds)
TASK_ID_MIN = 0
TASK_ID_MAX = 7349

ENV_EVAL_MAX_CONCURRENT_REQUESTS = 1
NUM_ENV_SERVERS = 1

OUTPUT_ROOT = Path(__file__).resolve().parent / "swe_chutes_stats"

##############################################################################################

client = docker.from_env()

ENV_IMAGE = "phoenixbeaudry/swe-infinite:v1"
NETWORK_NAME = "agent_eval_net"


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


def extract_usage(extra):
    """Pull token / cost / per-call data out of result['extra']. Best-effort —
    affinetes' exact shape isn't strictly typed here, so we probe a few fields."""
    if not isinstance(extra, dict):
        return {}

    total_tokens = _safe_int(extra.get("total_tokens")) or 0
    cost = _safe_float(extra.get("model_cost")) or 0.0
    usage = extra.get("usage") if isinstance(extra.get("usage"), dict) else {}

    input_tokens = (
        _safe_int(usage.get("input_tokens"))
        or _safe_int(usage.get("prompt_tokens"))
        or 0
    )
    output_tokens = (
        _safe_int(usage.get("output_tokens"))
        or _safe_int(usage.get("completion_tokens"))
        or 0
    )

    # model_calls might be a count or a list of per-call records
    model_calls = extra.get("model_calls")
    per_call_tokens = []
    num_calls = 0
    if isinstance(model_calls, list):
        num_calls = len(model_calls)
        for call in model_calls:
            if not isinstance(call, dict):
                continue
            u = call.get("usage") if isinstance(call.get("usage"), dict) else {}
            tk = (
                _safe_int(u.get("total_tokens"))
                or _safe_int(call.get("total_tokens"))
                or (
                    (_safe_int(u.get("input_tokens")) or _safe_int(u.get("prompt_tokens")) or 0)
                    + (_safe_int(u.get("output_tokens")) or _safe_int(u.get("completion_tokens")) or 0)
                )
            )
            if tk:
                per_call_tokens.append(tk)
    elif model_calls is not None:
        num_calls = _safe_int(model_calls) or 0

    return {
        "total_tokens": total_tokens,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "num_model_calls": num_calls,
        "per_call_tokens": per_call_tokens,
        "model_cost": cost,
    }


def conversation_stats(conversation):
    if not isinstance(conversation, list):
        return {"num_messages": 0, "num_assistant_turns": 0, "approx_content_chars": 0}
    n_assistant = 0
    chars = 0
    for m in conversation:
        if not isinstance(m, dict):
            continue
        if m.get("role") == "assistant":
            n_assistant += 1
        content = m.get("content", "")
        if isinstance(content, str):
            chars += len(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    chars += len(str(part.get("text", "") or part.get("content", "")))
    return {
        "num_messages": len(conversation),
        "num_assistant_turns": n_assistant,
        "approx_content_chars": chars,
    }


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
            "num_model_calls": _stat([r.get("num_model_calls") for r in rows]),
            "tokens_per_call_avg": _stat([
                (r["total_tokens"] / r["num_model_calls"])
                if r.get("num_model_calls") else None
                for r in rows
            ]),
            "time_taken": _stat([r.get("time_taken") for r in rows]),
        }

    return {
        "all": slice_stats(per_task),
        "passed": slice_stats(passed),
        "failed": slice_stats(failed),
        "pass_rate": len(passed) / len(per_task) if per_task else 0.0,
        "total_cost": sum((r.get("model_cost") or 0.0) for r in per_task),
        "cost_per_success": (
            sum((r.get("model_cost") or 0.0) for r in per_task) / len(passed)
            if passed else None
        ),
    }


def print_summary(summary):
    print("\n" + "=" * 72)
    print("Token Statistics Summary")
    print("=" * 72)
    print(
        f"Pass rate: {summary['pass_rate']:.2%} "
        f"({summary['passed']['count']}/{summary['all']['count']})"
    )
    print(f"Total cost: ${summary['total_cost']:.4f}")
    if summary["cost_per_success"] is not None:
        print(f"Cost per success: ${summary['cost_per_success']:.4f}")

    keys = (
        "tokens_per_call_avg",
        "num_model_calls",
        "time_taken",
    )
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


def run_evaluation():
    api_key = os.getenv(MODEL_API_KEY_ENV)
    if not api_key:
        raise RuntimeError(f"Set ${MODEL_API_KEY_ENV} before running.")

    output_dir = OUTPUT_ROOT
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks_dir = output_dir / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "state.json"
    state = _load_state(state_path)
    per_task = list(state.get("per_task", []))
    tested_set = set(state.get("tested_task_ids", []))
    print(f"📁 Output dir: {output_dir}")
    print(f"📦 Loaded state: {len(tested_set)} task(s) already tested")

    next_id = (max(tested_set) + 1) if tested_set else TASK_ID_MIN
    eval_list = []
    tid = next_id
    while len(eval_list) < NUM_EVALS and tid <= TASK_ID_MAX:
        if tid not in tested_set:
            eval_list.append(tid)
        tid += 1
    if not eval_list:
        print(
            f"✅ All tasks in [{TASK_ID_MIN}, {TASK_ID_MAX}] have been tested. "
            "Nothing to do this run."
        )
        return per_task
    print(
        f"▶️  Resuming from task ID {eval_list[0]} — "
        f"testing {len(eval_list)} task(s): "
        f"{eval_list[0]}..{eval_list[-1]}"
    )

    containers = {}

    try:
        if not client.networks.list(names=[NETWORK_NAME]):
            client.networks.create(NETWORK_NAME, driver="bridge")

        print(f"🚀 Starting {NUM_ENV_SERVERS} SWE-INFINITE env server(s)...")
        agent_base_port = 8001
        agent_ports = []
        for i in range(NUM_ENV_SERVERS):
            host_port = agent_base_port + i
            agent = client.containers.run(
                ENV_IMAGE,
                name=f"swe-infinite-server-{i}",
                detach=True,
                network=NETWORK_NAME,
                ports={"8000/tcp": host_port},
                volumes={
                    "/var/run/docker.sock": {"bind": "/var/run/docker.sock", "mode": "rw"},
                },
                environment={
                    "DOCKER_HUB_USERNAME": os.getenv("DOCKER_HUB_USERNAME", ""),
                    "DOCKER_HUB_TOKEN": os.getenv("DOCKER_HUB_TOKEN", ""),
                    "CHUTES_API_KEY": os.getenv("CHUTES_API_KEY", ""),
                    "R2_BASE_URL": os.getenv("R2_BASE_URL", ""),
                    "R2_PREFIX": os.getenv("R2_PREFIX", ""),
                },
            )
            containers[f"agent-{i}"] = agent
            agent_ports.append(host_port)

        print("⏳ Waiting for env servers...")
        for port in agent_ports:
            deadline = time.time() + 180
            while time.time() < deadline:
                try:
                    r = requests.get(f"http://localhost:{port}/", timeout=2)
                    if r.status_code < 500:
                        break
                except Exception:
                    pass
                time.sleep(2)
            else:
                raise RuntimeError(f"Env server on port {port} did not come up")
        print("✅ Env servers ready.\n")

        def evaluate_task(task_id, host_port):
            payload = {
                "task_id": task_id,
                "model": MODEL_NAME,
                "base_url": MODEL_BASE_URL,
                "api_key": api_key,
                "timeout": EVAL_TIMEOUT,
                "agent": AGENT_TYPE,
                "max_iterations": MAX_ITERATIONS,
            }
            if TEMPERATURE is not None:
                payload["temperature"] = TEMPERATURE
            print(f"Testing Task ID: {task_id}")
            try:
                response = requests.post(
                    f"http://localhost:{host_port}/evaluate",
                    json=payload,
                    timeout=EVAL_TIMEOUT + 120,
                )
                result = response.json()
                inner = result.get("result") if isinstance(result, dict) else None
                data = inner if isinstance(inner, dict) else (result if isinstance(result, dict) else {})
                return task_id, data, None
            except Exception as e:
                return task_id, {}, str(e)

        total_concurrency = NUM_ENV_SERVERS * ENV_EVAL_MAX_CONCURRENT_REQUESTS
        print(
            f"Running {len(eval_list)} evals across {NUM_ENV_SERVERS} server(s) "
            f"× {ENV_EVAL_MAX_CONCURRENT_REQUESTS} req each (concurrency={total_concurrency})..."
        )

        executors = [
            ThreadPoolExecutor(max_workers=ENV_EVAL_MAX_CONCURRENT_REQUESTS)
            for _ in range(NUM_ENV_SERVERS)
        ]
        completed = 0
        try:
            futures = {}
            for idx, task_id in enumerate(eval_list):
                server_idx = idx % NUM_ENV_SERVERS
                fut = executors[server_idx].submit(evaluate_task, task_id, agent_ports[server_idx])
                futures[fut] = task_id

            for future in as_completed(futures):
                task_id, data, error = future.result()
                completed += 1
                score = _safe_float(data.get("score")) or 0.0
                success = bool(data.get("success", score > 0.0))
                extra = data.get("extra") if isinstance(data.get("extra"), dict) else {}
                usage = extract_usage(extra)
                conv = conversation_stats(extra.get("conversation"))

                row = {
                    "task_id": task_id,
                    "score": score,
                    "success": success,
                    "time_taken": _safe_float(data.get("time_taken")),
                    "instance_id": extra.get("instance_id"),
                    "repo": extra.get("repo"),
                    "repo_language": extra.get("repo_language"),
                    "agent_type": extra.get("agent_type"),
                    "error": error,
                    **usage,
                    **conv,
                }
                per_task.append(row)
                tested_set.add(task_id)

                # Persist the full env-server payload for offline analysis.
                (tasks_dir / f"task_{task_id}.json").write_text(
                    json.dumps(data, default=str, indent=2)
                )
                _save_state(state_path, {
                    "tested_task_ids": sorted(tested_set),
                    "per_task": per_task,
                })

                result_label = "Success" if success else "Failure"
                print(f"Task ID: {task_id} completed, Result: {result_label}")
                if error:
                    print(f"[{completed}/{len(eval_list)}] task {task_id}: FAILED ({error})")
                else:
                    print(
                        f"[{completed}/{len(eval_list)}] task {task_id}: "
                        f"score={score} success={success} "
                        f"tokens={usage.get('total_tokens')} "
                        f"calls={usage.get('num_model_calls')} "
                        f"cost=${usage.get('model_cost', 0.0):.4f}"
                    )
        finally:
            for ex in executors:
                ex.shutdown(wait=True)

        summary = compute_summary(per_task)
        print_summary(summary)

        successful_tasks = []
        for r in per_task:
            if not r.get("success"):
                continue
            calls = r.get("num_model_calls") or 0
            total = r.get("total_tokens") or 0
            tokens_per_call_avg = (total / calls) if calls else None
            successful_tasks.append({
                "task_id": r["task_id"],
                "tokens_per_call_avg": tokens_per_call_avg,
                "num_model_calls": calls,
                "time_taken": r.get("time_taken"),
            })
        successful_tasks.sort(key=lambda r: r["task_id"])
        successful_task_ids = [r["task_id"] for r in successful_tasks]

        (output_dir / "per_task.json").write_text(json.dumps(per_task, default=str, indent=2))
        (output_dir / "summary.json").write_text(json.dumps(summary, default=str, indent=2))
        (output_dir / "successful_tasks.json").write_text(
            json.dumps(
                {"task_ids": successful_task_ids, "tasks": successful_tasks},
                default=str,
                indent=2,
            )
        )
        (output_dir / "config.json").write_text(json.dumps({
            "model": MODEL_NAME,
            "base_url": MODEL_BASE_URL,
            "agent": AGENT_TYPE,
            "num_evals": NUM_EVALS,
            "temperature": TEMPERATURE,
            "max_iterations": MAX_ITERATIONS,
            "eval_timeout": EVAL_TIMEOUT,
            "task_id_min": TASK_ID_MIN,
            "task_id_max": TASK_ID_MAX,
            "tasks_this_run": eval_list,
            "total_tested": len(tested_set),
        }, default=str, indent=2))

        print("\n" + "=" * 72)
        print(f"Successful tasks ({len(successful_tasks)}/{len(per_task)})")
        print("=" * 72)
        for r in successful_tasks:
            tpc = r["tokens_per_call_avg"]
            tpc_str = f"{tpc:>10.2f}" if tpc is not None else "       n/a"
            tt = r["time_taken"]
            tt_str = f"{tt:>8.2f}s" if tt is not None else "     n/a"
            print(
                f"  task {r['task_id']:>5}  "
                f"tokens_per_call_avg={tpc_str}  "
                f"num_model_calls={r['num_model_calls']:>4}  "
                f"time_taken={tt_str}"
            )
        print("\nSUCCESSFUL_TASK_IDS = " + repr(successful_task_ids))

        print(f"\n📁 Wrote results to {output_dir}")

    finally:
        print("🧹 Cleaning up containers...")
        for c in containers.values():
            try:
                c.remove(force=True)
            except Exception:
                pass

    return per_task


if __name__ == "__main__":
    print("Running Chutes SWE token-stats eval (sequential, resumable)")
    start = time.perf_counter()
    run_evaluation()
    print(f"Took {time.perf_counter() - start:.1f}s")
