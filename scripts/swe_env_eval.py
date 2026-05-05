import os
import docker
import time
import requests
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import snapshot_download

# --- Model Configuration ---
BASE_MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"
BASE_MODEL_REVISION = None
LORA_MODEL_NAME = None # Put the name of your repo containing the LORA here
LORA_MODEL_REVISION = None
LOCAL_LORA_PATH = None

# --- Evaluation Configuration ---
NUM_EVALS = 100
TEMPERATURE = 0.0
AGENT_TYPE = "miniswe"              # "miniswe", "codex", or "" for auto-select
MAX_ITERATIONS = 100         # miniswe only
EVAL_TIMEOUT = 1800          # per-task timeout (seconds)
TASK_ID_MIN = 0
TASK_ID_MAX = 7349         # adjust to match the # of tasks available in the cache

ENV_EVAL_MAX_CONCURRENT_REQUESTS = 1
NUM_ENV_SERVERS = 1

##############################################################################################


client = docker.from_env()

SGLANG_IMAGE = "lmsysorg/sglang:latest"
ENV_IMAGE = "phoenixbeaudry/swe-infinite:v1"
NETWORK_NAME = "agent_eval_net"
SGLANG_PORT = 30000
HF_CACHE_DIR = "/mnt/hf_cache"

def run_evaluation(base_seed):
    RANDOM_SEED = base_seed
    containers = {}
    avg_score = 0.0
    win_count = 0

    try:
        # 1. Infrastructure Setup
        networks = client.networks.list(names=[NETWORK_NAME])
        if not networks: client.networks.create(NETWORK_NAME, driver="bridge")

        lora_dir = None
        use_lora = bool(LOCAL_LORA_PATH or LORA_MODEL_NAME)

        if use_lora:
            if LOCAL_LORA_PATH:
                print(f"🚀 Starting SGLang: {BASE_MODEL_NAME} w/ local LoRA at {LOCAL_LORA_PATH}")
                lora_dir = LOCAL_LORA_PATH
            else:
                print(f"🚀 Starting SGLang: {BASE_MODEL_NAME} w/ HF LoRA {LORA_MODEL_NAME}")
                safe_lora_name = LORA_MODEL_NAME.replace("/", "_")
                lora_dir = f"/tmp/sglang_lora/{safe_lora_name}"
                print(f"⬇️  Downloading LoRA to {lora_dir}...")
                snapshot_download(
                    repo_id=LORA_MODEL_NAME,
                    revision=LORA_MODEL_REVISION,
                    local_dir=lora_dir,
                    local_dir_use_symlinks=False,
                )

            sglang_command = (
                f"python3 -m sglang.launch_server --model-path {BASE_MODEL_NAME} "
                "--enable-lora --lora-paths trained_lora=/lora/trained_lora "
                "--lora-backend triton --max-lora-rank 64 "
                "--host 0.0.0.0 --port 30000 --dp-size 2 --dtype bfloat16 "
                "--context-length 131072 "
                "--json-model-override-args '{\"rope_scaling\":{\"rope_type\":\"yarn\",\"type\":\"yarn\",\"factor\":4.0,\"original_max_position_embeddings\":32768}}' "
                f"--random-seed {RANDOM_SEED}"
            )
        else:
            print(f"🚀 Starting SGLang: {BASE_MODEL_NAME}")
            sglang_command = (
                f"python3 -m sglang.launch_server --model-path {BASE_MODEL_NAME} "
                f"{'--revision ' + BASE_MODEL_REVISION if BASE_MODEL_REVISION else ''} "
                "--host 0.0.0.0 --port 30000 --dp-size 2 --dtype bfloat16 "
                "--context-length 131072 "
                "--json-model-override-args '{\"rope_scaling\":{\"rope_type\":\"yarn\",\"type\":\"yarn\",\"factor\":4.0,\"original_max_position_embeddings\":32768}}' "
                f"--random-seed {RANDOM_SEED}"
            )

        sglang = client.containers.run(
            SGLANG_IMAGE,
            command=sglang_command,
            name="sglang-server",
            detach=True,
            network=NETWORK_NAME,
            ports={f"{SGLANG_PORT}/tcp": SGLANG_PORT},
            device_requests=[docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])],
            environment={
                "HF_HOME": "/hf",
                "TRANSFORMERS_CACHE": "/hf",
                "HUGGINGFACE_HUB_CACHE": "/hf",
                "HF_HUB_ENABLE_HF_TRANSFER": "1",
                "PYTHONHASHSEED": str(RANDOM_SEED),
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "NVIDIA_TF32_OVERRIDE": "0",
                "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
            },
            volumes={
                HF_CACHE_DIR: {"bind": "/hf", "mode": "rw"},
                **({lora_dir: {"bind": "/lora/trained_lora", "mode": "ro"}} if lora_dir else {}),
            },
            ipc_mode="host",
        )
        containers['sglang'] = sglang

        print(f"🚀 Starting {NUM_ENV_SERVERS} SWE-INFINITE Env Server(s)...")
        agent_base_port = 8001
        agent_ports = []
        for i in range(NUM_ENV_SERVERS):
            host_port = agent_base_port + i
            agent = client.containers.run(
                ENV_IMAGE,
                name=f"swe-infinite-server-{i}",
                detach=True,
                network=NETWORK_NAME,
                ports={'8000/tcp': host_port},
                # SWE-INFINITE spawns docker containers for verification — needs the host socket.
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
            containers[f'agent-{i}'] = agent
            agent_ports.append(host_port)

        # 2. Wait for Readiness
        print("⏳ Waiting for SGLang health check...")
        sglang_start_deadline = time.time() + 300  # 5 min timeout (adjust if needed)
        last_err = None

        while time.time() < sglang_start_deadline:
            # If container exited/crashed, dump logs immediately
            try:
                sglang.reload()  # refresh container state
                if sglang.status in ("exited", "dead"):
                    print(f"❌ SGLang container exited early (status={sglang.status}). Dumping logs...\n")
                    try:
                        logs = sglang.logs(stdout=True, stderr=True, tail=500)
                        if isinstance(logs, (bytes, bytearray)):
                            logs = logs.decode("utf-8", errors="replace")
                        print("===== SGLANG STARTUP LOGS (tail=500) =====")
                        print(logs)
                        print("===== END SGLANG LOGS =====")
                    except Exception as log_e:
                        print(f"⚠️ Failed to read SGLang logs: {log_e}")
                    raise RuntimeError("SGLang container exited during startup")
            except Exception as state_e:
                # If reload itself fails, keep trying unless timeout hits
                last_err = state_e

            try:
                r = requests.get(f"http://localhost:{SGLANG_PORT}/v1/models", timeout=2)
                if r.status_code == 200:
                    print("✅ SGLang Ready.\n")
                    break
                last_err = RuntimeError(f"Health check returned status {r.status_code}")
            except Exception as e:
                last_err = e

            time.sleep(3)
        else:
            print("❌ Timed out waiting for SGLang to become ready. Dumping logs...\n")
            try:
                sglang.reload()
                print(f"SGLang container status: {sglang.status}")
            except Exception as state_e:
                print(f"⚠️ Could not reload SGLang container state: {state_e}")

            try:
                logs = sglang.logs(stdout=True, stderr=True, tail=500)
                if isinstance(logs, (bytes, bytearray)):
                    logs = logs.decode("utf-8", errors="replace")
                print("===== SGLANG STARTUP LOGS (tail=500) =====")
                print(logs)
                print("===== END SGLANG LOGS =====")
            except Exception as log_e:
                print(f"⚠️ Failed to read SGLang logs: {log_e}")

            raise RuntimeError(f"SGLang failed health check before timeout. Last error: {last_err}")

        # 3. Evaluation Loop
        random.seed(RANDOM_SEED)
        eval_list = random.sample(range(TASK_ID_MIN, TASK_ID_MAX), NUM_EVALS)
        total_score = 0.0

        if use_lora:
            inference_model_name = f"{BASE_MODEL_NAME}:trained_lora"
        else:
            inference_model_name = BASE_MODEL_NAME

        def evaluate_task(task_id, host_port):
            payload = {
                "task_id": task_id,
                "model": inference_model_name,
                "base_url": f"http://sglang-server:{SGLANG_PORT}/v1",
                "api_key": "test",
                "timeout": EVAL_TIMEOUT,
                "temperature": TEMPERATURE,
                "seed": task_id,
                "agent": AGENT_TYPE,
                "max_iterations": MAX_ITERATIONS,
            }
            try:
                response = requests.post(
                    f"http://localhost:{host_port}/evaluate",
                    json=payload,
                    timeout=EVAL_TIMEOUT + 120,
                )
                result = response.json()
                result_payload = result.get("result") if isinstance(result, dict) else None
                if isinstance(result_payload, dict):
                    data = result_payload
                else:
                    data = result if isinstance(result, dict) else {}
                return task_id, data.get('score', 0.0), None
            except Exception as e:
                return task_id, 0.0, str(e)

        total_concurrency = NUM_ENV_SERVERS * ENV_EVAL_MAX_CONCURRENT_REQUESTS
        print(
            f"Running {NUM_EVALS} evaluations across {NUM_ENV_SERVERS} env server(s) "
            f"x {ENV_EVAL_MAX_CONCURRENT_REQUESTS} req each (total concurrency={total_concurrency})..."
        )
        completed = 0
        executors = [
            ThreadPoolExecutor(max_workers=ENV_EVAL_MAX_CONCURRENT_REQUESTS)
            for _ in range(NUM_ENV_SERVERS)
        ]
        try:
            futures = {}
            for idx, task_id in enumerate(eval_list):
                server_idx = idx % NUM_ENV_SERVERS
                host_port = agent_ports[server_idx]
                fut = executors[server_idx].submit(evaluate_task, task_id, host_port)
                futures[fut] = task_id
            for future in as_completed(futures):
                task_id, score, error = future.result()
                completed += 1
                total_score += score
                if score == 1.0:
                    win_count += 1
                if error:
                    print(f"[{completed}/{NUM_EVALS}] Task {task_id}: FAILED ({error})")
                else:
                    print(f"[{completed}/{NUM_EVALS}] Task {task_id}: {score}")
        finally:
            for ex in executors:
                ex.shutdown(wait=True)

        # 4. Final Score
        avg_score = total_score / NUM_EVALS if NUM_EVALS > 0 else 0

        avg_score = round(avg_score, 4)

        print(f"\n✅ Evaluation complete.")
        print(f"Score: {total_score}/{NUM_EVALS} ({avg_score:.4f})")
        print(f"Win Count: {win_count}/{NUM_EVALS} ({win_count/NUM_EVALS})")

    finally:
        print("🧹 Cleaning up containers...")
        for c in containers.values():
            try: c.remove(force=True)
            except: pass

    return avg_score


if __name__ == "__main__":
    scores = []
    times = []
    for i in range(1):
        base_seed = random.randint(0, 100000)
        print(f"Running Eval {i+1}/10 with base seed: {base_seed}")
        start = time.perf_counter()
        score = run_evaluation(base_seed=base_seed)
        end = time.perf_counter()
        elapsed = end-start
        times.append(elapsed)
        scores.append(score)
    print(scores)
    print(f"Average time for Eval: {sum(times)/len(times)}")