# G.O.D Developer and Operator Guide

This guide is for repository maintainers, validator operators, trainer operators, auditors, and developers. Miner-facing requirements live in [miner.md](miner.md).

Coding agents working on the repository should read `AGENTS.md` for repo-specific maintenance rules and validation expectations.

## What This Repo Runs

The production surface is intentionally small:

- `validator/` runs task intake, tournament orchestration, evaluation, scoring, and weight setting. Validator-wide config, constants, and models live in `validator/shared/`; external service adapters live in `validator/infrastructure/`.
- `trainer/` runs submitted training repositories, model prep jobs, GPU assignment state, logs, and model upload helpers. Docker sidecar entrypoints live in `trainer/containers/`; task-specific training entrypoints live in external submitted repositories.
- `miner/` runs the reference miner HTTP endpoint for serving training repository metadata to validators.
- `core/` contains shared API models, enums, dataset contracts, training templates, and compatibility constants.
- `ops/` contains Docker assets, compose stacks, observability config, auditor scripts, examples, and manual tools.

The supported tournament families are text, image, and environment. Tournament miners participate through the training repository contract in [miner.md](miner.md); the bundled `miner/` service is the reference implementation of that endpoint.

## Tournament Families

Current tournament facts are backed by `validator/shared/constants.py`, `validator/tournament/constants.py`, and task-type helpers in `core/models/utility_models.py`.

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

Important trainer endpoint paths are defined in `core/service_paths.py`:

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

To test tournament repository discovery locally, run the bundled miner endpoint with `task miner` or any other HTTP service that exposes `GET /training_repo/{task_type}` as documented in [miner.md](miner.md).

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

Miners do not need production compute to participate in tournaments. They submit open-source training repositories; validators execute that code on trainer nodes. A local GPU is useful for testing but not required by the participation protocol. See [miner.md](miner.md) for the miner contract.

Validator trainer nodes allocate GPUs by task type and model size:

| Workload | Allocation |
| --- | --- |
| Text models up to 4B parameters | 1x H100 |
| Text models from 4B to 12B | 2x H100 |
| Text models from 12B to 40B | 4x H100 |
| Text models above 40B | 8x H100 |
| Image tasks | 1x A100 |

DPO and GRPO-style tasks apply larger effective requirements because they are heavier than ordinary text fine-tuning. Containers are provisioned with GPU-scaled CPU and memory, and training containers should not depend on public internet access during training.

## Scoring Summary

Text and image tournaments award round points to task winners and use a boss round to decide whether the defending champion is dethroned. Environment tournaments use PvP-style environment outcomes and a final multi-task boss comparison.

Subnet emissions are tournament-based:

1. Tournament champions receive family-specific winner weights.
2. Tournament participants receive a small active-participation weight.
3. Undistributed weight goes to the burn address.

Current constants live in `validator/shared/constants.py` and `validator/tournament/constants.py`.

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
- `validator/tournament/performance_calculator.py`

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

The old local miner training task runners have been removed. Submitted training repositories are external to this repo and must follow the contract in [miner.md](miner.md).

## Static References

Long-form project PDFs and static images remain in `docs/` for reference:

- `Gradient_White_Paper.pdf`
- `Gradients_Annual_Report_2025.pdf`
- `threshold_needed.png`
