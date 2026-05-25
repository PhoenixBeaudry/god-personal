# G.O.D Operator and Tournament Guide

This is the canonical Markdown documentation for the G.O.D repository. It replaces the old split between validator setup, trainer setup, tournament overview, environment tasks, scoring, auditing, and miner submission docs.

Coding agents working on the repository should read `AGENTS.md` for repo-specific maintenance rules and validation expectations.

## What This Repo Runs

The production surface is intentionally small:

- `validator/` runs task intake, tournament orchestration, evaluation, scoring, and weight setting. Validator-wide config, constants, and models live in `validator/shared/`; external service adapters live in `validator/infrastructure/`.
- `trainer/` runs training jobs, model prep jobs, GPU assignment state, logs, and model upload helpers. Docker training and sidecar entrypoints live in `trainer/containers/`.
- `core/` contains shared API models, enums, dataset contracts, training templates, and compatibility constants.
- `ops/` contains Docker assets, compose stacks, observability config, auditor scripts, examples, and manual tools.

The supported tournament families are text, image, and environment. The old local `miner/` runtime has been removed; tournament miners participate by running any service that exposes the external training repository endpoint described below.

## Tournament Families

Current tournament facts are centralized in `validator/tournament/specs.py` and backed by the legacy constants for backward compatibility. Shared task-type groups and helpers live in `core/models/utility_models.py`.

| Family | Task types | Schedule | Minimum participants | Fee |
| --- | --- | --- | --- | --- |
| Text | `InstructTextTask`, `ChatTask`, `DpoTask`, `GrpoTask` | Thursday 14:00 UTC | 8 | 0.2 TAO |
| Image | `ImageTask` | Thursday 15:00 UTC | 8 | 0.15 TAO |
| Environment | `EnvTask` | Monday 14:00 UTC | 5 | 0.2 TAO |

Tournaments are created only when no active or pending tournament of the same family exists and the scheduled window has arrived.

## Validator Flow

The tournament worker in `validator/tournament/runner.py` runs five long-lived loops:

1. `transfer_monitoring_cycle()` monitors TAO transfers and coldkey balances.
2. `process_pending_tournaments()` asks eligible miners for a training repository, validates the repository, charges the participation fee, and activates the tournament.
3. `process_pending_rounds()` creates synthetic tasks, assigns hotkeys to those tasks, and moves the round to active once every task has assignments.
4. `process_active_tournaments()` waits for task completion, computes winners, advances rounds, and records the tournament winner.
5. `process_tournament_scheduling()` creates tournaments when a family reaches its scheduled window.

Text and image tournaments use group rounds when the participant count is large, then knockout rounds. Environment tournaments use group rounds and a final boss comparison.

## Trainer Flow

The trainer receives validator requests, launches training containers, tracks task history, and reports recent jobs. It also handles model prep jobs used for augmentation and baseline statistics.

Important trainer endpoints are defined in `validator/shared/constants.py`:

- `/v1/trainer/start_training`
- `/v1/trainer/model_prep`
- `/v1/trainer/model_prep/{task_id}`
- `/v1/trainer/get_gpu_availability`
- `/v1/trainer/{task_id}`
- `/v1/trainer/get_recent_tasks`

Task history is persisted atomically by `trainer/job_state.py`, so trainer restarts can recover recent job state.

## Validator Setup

Prerequisites:

- Docker and Docker Compose.
- A Bittensor wallet and hotkey.
- A Hugging Face account and token.
- An S3-compatible bucket.
- Open outbound HTTPS access to `api.gradients.io`.

Basic setup:

```bash
git clone https://github.com/rayonlabs/G.O.D.git
cd G.O.D
task bootstrap
task config
task install
echo "MODEL_HASH_SALT=$(openssl rand -hex 32)" >> .vali.env
task validator
```

For development:

```bash
pip install -e '.[dev]'
pre-commit install
task validator_dev
```

Most users should consider running an auditor instead of a validator. Auditing is cheaper and does not require the same GPU infrastructure.

## Development

For local development without the full production bootstrap:

```bash
git clone https://github.com/rayonlabs/G.O.D.git
cd G.O.D
pip install -e '.[dev]'
pre-commit install
task setup
task validator
```

The local miner runtime has been removed. To test tournament repository discovery locally, run any small HTTP service that exposes the external `GET /training_repo/{task_type}` contract documented below.

Host-level helper scripts live under `ops/`. Trainer systemd helpers are in `ops/trainer_ops/`; manual SQL and destructive local maintenance helpers are in `ops/manual/`; purpose-specific operator tools are grouped under `ops/tools/config/`, `ops/tools/tournament/`, `ops/tools/evaluation/`, `ops/tools/datasets/`, `ops/tools/observability/`, and `ops/tools/simulations/`.

Tests mirror the source layout under `tests/`: shared contract tests in `tests/core/`, trainer tests in `tests/trainer/`, and validator tests split by DB, tasks, lifecycle, evaluation, and tournament behavior under `tests/validator/`.

## Auditor Setup

Auditors verify that the validator reports scores and weights fairly. The auditor workflow checks recent task details, score explanations, synthetic jobs, and on-chain weight outputs through the public auditing API.

Basic setup:

```bash
git clone https://github.com/rayonlabs/G.O.D.git
cd G.O.D
task bootstrap
task auditor-config
task install
task auditor
```

Use `task auditor-autoupdates` when you want the auditor auto-update wrapper. Auditing endpoints are exposed under `/auditing/*`; they are read-oriented product endpoints and not miner submission endpoints.

## Storage

Validators need an S3-compatible bucket for datasets, evaluation artifacts, and reports. Backblaze B2 works well, but any S3-compatible provider is acceptable.

Minimum setup:

1. Create a private bucket.
2. Create an application key with read/write access to that bucket.
3. Copy the provider's S3 endpoint, access key, secret key, and bucket name.
4. Put those values in `.vali.env` through `task config` or by editing the generated file.

For Backblaze B2, the S3 endpoint format is usually `s3.<region>.backblazeb2.com`.

## Trainer Setup

Create the trainer environment file and start the trainer service:

```bash
task trainer-config
task trainer
```

Trainer containers need enough GPU capacity for the tournament workload they will receive. Text tasks may require multiple H100s depending on model size and task type. Image tasks use image trainer containers. Environment tasks currently require larger H100 allocations because they run GRPO-style training plus environment interaction.

Optional log shipping:

```bash
task deploy-trainer-logs
```

## Observability

Validator logs are instrumented with OpenTelemetry and context tags such as `task_id`, `miner_hotkey`, GPU IDs, and Docker container names. The local observability stack lives in `ops/observability/` and is launched by the validator startup scripts.

Common commands:

| Command | Run on | Purpose |
| --- | --- | --- |
| `task deploy-observability-server` | Validator | Deploy Grafana, Loki, and Prometheus for trainer logs. |
| `task stop-observability-server` | Validator | Stop the remote observability stack. |
| `task deploy-trainer-logs` | Trainer | Start Vector log shipping from trainer containers. |
| `task stop-trainer-logs` | Trainer | Stop trainer log shipping. |
| `task logs-observability` | Validator | Tail observability stack logs. |
| `task logs-trainer-shipper` | Trainer | Tail Vector log shipper logs. |
| `task test-trainer-logs` | Trainer | Emit a test container log stream. |

Trainer log shipping expects `VALIDATOR_IP` in `.trainer.env`. Optional validator-side overrides include `OBSERVABILITY_DOMAIN`, `GRAFANA_TRAINING_PASSWORD`, `LOKI_PASSWORD`, and `GRAFANA_ANONYMOUS_ENABLED`.

Vector collects trainer containers named like `image-trainer-*`, `text-trainer-*`, `downloader-*`, and `hf-upload-*`. Grafana exposes validator and trainer dashboards; local validator logs are available at the Grafana port configured by the compose stack.

## Compute

Miners do not need production compute to participate in tournaments. They submit open-source training repositories; validators execute that code on trainer nodes. A local GPU is useful for testing but not required by the participation protocol.

Validator trainer nodes allocate GPUs by task type and model size:

| Workload | Allocation |
| --- | --- |
| Text models up to 4B parameters | 1x H100 |
| Text models from 4B to 12B | 2x H100 |
| Text models from 12B to 40B | 4x H100 |
| Text models above 40B | 8x H100 |
| Image tasks | 1x A100 |

DPO and GRPO-style tasks apply larger effective requirements because they are heavier than ordinary text fine-tuning. Containers are provisioned with GPU-scaled CPU and memory, and training containers should not depend on public internet access during training.

## Miner Participation Contract

Miners do not need to run this repository as a miner service. A miner only needs a subnet hotkey that posts an IP and serves the training repository endpoint.

The validator calls:

```http
GET /training_repo/{task_type}
```

`task_type` is one of:

- `text`
- `image`
- `environment`

The response must match `TrainingRepoResponse`:

```json
{
  "github_repo": "https://github.com/YOUR_ORG/YOUR_TRAINING_REPO",
  "commit_hash": "YOUR_COMMIT_SHA_OR_BRANCH",
  "github_token": null,
  "requested_datasets": null
}
```

`github_token` is optional and should only be a fine-grained, read-only token for private repositories. `requested_datasets` is optional and must contain Hugging Face dataset repo IDs from the whitelist in `core/datasets/whitelisted_sft_datasets.json`.

## Miner-Requested Datasets

Miners can request additional public Hugging Face datasets by returning `requested_datasets` in `TrainingRepoResponse`.

Rules:

- Every requested dataset must appear in `core/datasets/whitelisted_sft_datasets.json`.
- The validator downloads approved datasets and mounts them under `MINER_DATASETS_DIR`.
- `MINER_DATASETS` contains the downloaded directory names as a comma-separated list.
- Datasets outside the whitelist are ignored or rejected, and using unapproved bundled datasets can disqualify a submission.

## Miner Eligibility

To be accepted into a tournament, a miner must:

- Be registered on subnet 56 on mainnet, or subnet 241 on testnet.
- Post a reachable IP and port to the metagraph.
- Return a valid `TrainingRepoResponse` for the tournament family.
- Have enough coldkey balance for the tournament fee.
- Provide a repository at the returned commit for the full duration of the tournament.
- Include verbatim `LICENSE` or `LICENSE.md` and `NOTICE` files matching this repository.
- Avoid obfuscated files, compiled-only code, hidden model weights, or bundled private datasets.

Tournament fees can be checked at:

```bash
curl https://api.gradients.io/tournament/fees
curl https://api.gradients.io/tournament/balance/<coldkey>
```

## Training Repository Layout

Your training repository must include the Dockerfiles expected by the trainer:

```text
your-training-repo/
  dockerfiles/
    standalone-text-trainer.dockerfile
    standalone-image-trainer.dockerfile
    standalone-image-toolkit-trainer.dockerfile
```

The trainer only looks at these exact submitted-repository paths. Repositories that put trainer Dockerfiles under `ops/docker/` or any other folder are rejected by the trainer.

Text, chat, DPO, GRPO, and environment tasks use `dockerfiles/standalone-text-trainer.dockerfile`. SDXL and Flux image tasks use `dockerfiles/standalone-image-trainer.dockerfile`. Qwen Image and Z-Image tasks use `dockerfiles/standalone-image-toolkit-trainer.dockerfile`.

Submission checklist:

- Return the repository URL and exact commit from `GET /training_repo/{task_type}`.
- Keep the commit available until the tournament is fully complete.
- Include `LICENSE` or `LICENSE.md` and `NOTICE` files matching this repository.
- Include the expected trainer Dockerfiles under `dockerfiles/`.
- Upload the trained model, adapter, or LoRA to the Hugging Face repo name supplied as `--expected-repo-name`.
- Do not require interactive input, secrets outside the documented environment variables, or network access for hidden data or weights during training.
- Do not include obfuscated code, compiled-only training logic, bundled private datasets, or pre-trained final weights.

Recommended starting points:

- Text: `axolotlai/axolotl:main-py3.11-cu124-2.5.1`
- Image: `diagonalge/kohya_latest:latest`

Previous winners are published at [github.com/gradients-opensource](https://github.com/gradients-opensource).

## Training CLI Arguments

Text and environment training containers receive:

```bash
--task-id
--model
--dataset
--dataset-type
--task-type
--expected-repo-name
--hours-to-complete
```

Image training containers receive:

```bash
--task-id
--model
--dataset-zip
--model-type
--expected-repo-name
--hours-to-complete
```

The expected output is a Hugging Face model or adapter uploaded under the exact `expected-repo-name` assigned by the validator.

## Training Environment Variables

Training containers may receive:

| Variable | Meaning |
| --- | --- |
| `BASELINE_STATS_PATH` | Optional model prep statistics JSON. Safe to ignore if your code does not use it. |
| `MINER_DATASETS_DIR` | Parent directory containing whitelisted miner-requested datasets. |
| `MINER_DATASETS` | Comma-separated downloaded dataset directory names. |
| `ENVIRONMENT_SERVER_URLS` | Comma-separated environment server URLs for environment tasks. |

Use offline WandB logging if you log metrics, and avoid network-dependent training behavior unless it is explicitly part of the supplied task data.

## GRPO Reward Functions

GRPO reward functions that execute user-provided code must use the `restricted_execution` helper. Reward functions that execute arbitrary code without that wrapper are rejected.

```python
def restricted_execution(code: str, input_data: str) -> tuple[str, str]:
    """Return (printed_output, error_message)."""
```

`restricted_execution` captures `print()` output, returns errors as strings, and blocks file access, network calls, imports, system commands, `eval`, `exec`, `globals`, and `locals`. Common safe built-ins such as `sum`, `min`, `max`, `abs`, `round`, `sorted`, `len`, `str`, `int`, `float`, `list`, `dict`, `range`, `enumerate`, `zip`, `map`, and `filter` are available.

Typical reward-function shape:

```python
def my_reward_function(completions, extra_data=None, **kwargs):
    scores = []
    for response in completions:
        output, error = restricted_execution(response, input_data="")
        scores.append(1.0 if not error and output.strip() == extra_data["expected_output"] else 0.0)
    return scores
```

## Environment Tournaments

Environment tournaments use `EnvTask` and train against live environment servers. Your training code should read `ENVIRONMENT_SERVER_URLS`, connect to the provided servers, and implement rollout logic compatible with TRL GRPO custom rollouts.

Server addresses are passed as a comma-separated environment variable:

```python
import os

raw_urls = os.environ.get("ENVIRONMENT_SERVER_URLS", "")
server_list = [url.strip() for url in raw_urls.split(",") if url.strip()]
```

Miners must implement a rollout function associated with the task's `dataset_type.environment_name`. A rollout function should:

1. Generate completions through `generate_rollout_completions`.
2. Send those completions to the provided environment servers.
3. Return prompt tokens, completion tokens, logprobs, and reward signals in the format expected by the trainer.

A simple reward bridge can read rewards produced by the rollout function:

```python
def rollout_reward_func(completions, **kwargs):
    rewards = kwargs.get("env_rewards") if kwargs else None
    return [float(r) for r in rewards] if rewards is not None else [0.0] * len(completions)
```

Environment evaluation uses PvP evaluation across OpenSpiel environments. Models play head-to-head with position swaps for fairness. The main metric is tournament points: 3 for an environment win, 1 for a draw, and 0 for a loss.

Rules:

- Do not bundle your own datasets in the image.
- Do not bundle pretrained models in the image.
- SFT is allowed only with whitelisted datasets requested through `requested_datasets`.
- Use the provided task data and environment interaction as the source of competitive signal.

Current environment configuration lives in `ENVIRONMENT_CONFIGS` in `core/constants.py`. Environment training repositories may organize rollout code however they like, as long as their text trainer Dockerfile builds an image whose entrypoint accepts the documented CLI arguments and reads `ENVIRONMENT_SERVER_URLS`.

Useful optimization areas include training on the full episode rather than only the first prompt/completion, adding intermediate reward signals when valid, and improving VLLM placement for multi-GPU GRPO training.

## Image Training Notes

Image tournaments train diffusion LoRAs. Style tasks often tolerate lower learning rates, more repeats, and larger batches. Person, object, and concept tasks tend to overfit faster, so fewer repeats, fewer epochs, and a higher learning rate may work better.

The reference image trainer uses Kohya. The default optimizer is `AdamW8Bit`; alternatives such as `Prodigy` are allowed if they improve final evaluation quality and do not violate the tournament rules.

## Scoring Summary

Text and image tournaments award round points to task winners and use a boss round to decide whether the defending champion is dethroned. Environment tournaments use PvP-style environment outcomes and a final multi-task boss comparison.

Subnet emissions are tournament-based:

1. Tournament champions receive family-specific winner weights.
2. Tournament participants receive a small active-participation weight.
3. Undistributed weight goes to the burn address.

Current constants live in `validator/shared/constants.py`. Tournament family metadata lives in `validator/tournament/specs.py`.

| Weight | Meaning |
| --- | --- |
| `TOURNAMENT_TEXT_WEIGHT` | Text base allocation. |
| `TOURNAMENT_IMAGE_WEIGHT` | Image base allocation. |
| `TOURNAMENT_ENVIRONMENT_WEIGHT` | Environment base allocation. |
| `MAX_TEXT_TOURNAMENT_WEIGHT` | Text allocation cap. |
| `MAX_IMAGE_TOURNAMENT_WEIGHT` | Image allocation cap. |
| `MAX_ENVIRONMENT_TOURNAMENT_WEIGHT` | Environment allocation cap. |
| `TOURNAMENT_PARTICIPATION_WEIGHT` | Per-active-participant reward. |

When a champion beats the runner-up by more than `EMISSION_MULTIPLIER_THRESHOLD`, the excess performance can boost that family allocation. Long-reigning champions receive time-based decay, and each family is capped independently. Within a tournament, ranked participants are distributed by exponential decay using `TOURNAMENT_SIMPLE_DECAY_BASE`.

Tournament scores feed into emission weights through:

- `validator/evaluation/tournament_scoring.py`
- `validator/shared/weight_setting.py`
- `validator/tournament/specs.py`

The public result page for a completed tournament is:

```text
https://gradients.io/app/research/tournament/{TOURNAMENT_ID}
```

## Local Evaluation

Build current validator images before local evaluation:

```bash
docker build -f ops/docker/validator.dockerfile -t weightswandering/tuning_vali:latest .
docker build -f ops/docker/validator-diffusion.dockerfile -t diagonalge/tuning_validator_diffusion:latest .
```

Run:

```bash
python ops/validator_ops/run_evaluation.py --help
python ops/validator_ops/run_evaluation.py --task_id <task_id>
python ops/validator_ops/run_evaluation.py --task_id <task_id> --models <model_repo>
```

For local training task examples, see `ops/examples/`.

## Static References

Long-form project PDFs and static images remain in `docs/` for reference:

- `Gradient_White_Paper.pdf`
- `Gradients_Annual_Report_2025.pdf`
- `threshold_needed.png`
