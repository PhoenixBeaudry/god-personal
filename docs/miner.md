# G.O.D Miner Guide

This guide contains the miner-facing contract for Gradients on Demand tournaments. Miners participate by serving a training repository endpoint and providing a training repository that validators can build and run. This repository includes a small reference miner service under `miner/`.

Developer, validator, trainer, and auditor operations live in [developer.md](developer.md).

## Tournament Families

| Family | Task types | Schedule | Minimum participants | Fee |
| --- | --- | --- | --- | --- |
| Text | `InstructTextTask`, `ChatTask`, `DpoTask`, `GrpoTask` | Thursday 14:00 UTC | 8 | 0.2 TAO |
| Image | `ImageTask` | Thursday 15:00 UTC | 8 | 0.15 TAO |
| Environment | `EnvTask` | Monday 14:00 UTC | 5 | 0.2 TAO |

Tournaments are created only when no active or pending tournament of the same family exists and the scheduled window has arrived.

## What Miners Run

Miners need:

- A subnet hotkey registered on subnet 56 on mainnet, or subnet 241 on testnet.
- A reachable IP and port posted to the metagraph.
- A service that responds to `GET /training_repo/{task_type}`.
- A public or private GitHub training repository at the commit returned by that endpoint.

Miners do not need production GPU compute for the tournament run. Validators execute submitted training repositories on validator trainer nodes. A local GPU is useful for developing and testing your training repository, but it is not required by the participation protocol.

To use the bundled reference endpoint:

```bash
task miner-config
task install
task miner
```

The generated `.miner.env` uses shared values for all tournament families:

```env
MINER_TRAINING_REPO=https://github.com/YOUR_ORG/YOUR_TRAINING_REPO
MINER_TRAINING_COMMIT=YOUR_COMMIT_SHA
MINER_GITHUB_TOKEN=
MINER_REQUESTED_DATASETS=
```

Family-specific values are also supported, such as `MINER_TEXT_TRAINING_REPO`,
`MINER_IMAGE_TRAINING_REPO`, and `MINER_ENVIRONMENT_TRAINING_REPO`.

## Participation Endpoint

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

## Eligibility

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

Task-to-Dockerfile mapping:

| Tournament family | Dockerfile |
| --- | --- |
| Text, chat, DPO, GRPO, environment | `dockerfiles/standalone-text-trainer.dockerfile` |
| SDXL and Flux image tasks | `dockerfiles/standalone-image-trainer.dockerfile` |
| Qwen Image and Z-Image tasks | `dockerfiles/standalone-image-toolkit-trainer.dockerfile` |

Recommended starting points:

- Text: `axolotlai/axolotl:main-py3.11-cu124-2.5.1`
- Image: `diagonalge/kohya_latest:latest`

Previous winners are published at [github.com/gradients-opensource](https://github.com/gradients-opensource).

## Submission Checklist

- Return the repository URL and exact commit from `GET /training_repo/{task_type}`.
- Keep the commit available until the tournament is fully complete.
- Include `LICENSE` or `LICENSE.md` and `NOTICE` files matching this repository.
- Include the expected trainer Dockerfiles under `dockerfiles/`.
- Write the trained model, adapter, or LoRA into `/app/checkpoints/<task-id>/<expected-repo-name>` inside the training container. The trainer uploads that directory to Hugging Face after the training run is considered successful.
- Do not require interactive input, secrets outside the documented environment variables, or network access for hidden data or weights during training.
- Do not include obfuscated code, compiled-only training logic, bundled private datasets, or pre-trained final weights.

## Training CLI Arguments

Text and environment training containers receive:

```bash
--task-id
--model
--dataset
--dataset-type
--task-type
--file-format
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

The expected output is a model, adapter, or LoRA directory at `/app/checkpoints/<task-id>/<expected-repo-name>` inside the training container. After the trainer considers the training run successful, it starts its upload sidecar, creates or updates the Hugging Face repo named by `expected-repo-name`, and uploads that directory. Image task artifacts are uploaded under the `checkpoints/` subfolder of the Hugging Face repo; text and environment artifacts are uploaded at the repo root.

## Training Environment Variables

Training containers may receive:

| Variable | Meaning |
| --- | --- |
| `BASELINE_STATS` | Optional model prep statistics JSON. Safe to ignore if your code does not use it. |
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

The public result page for a completed tournament is:

```text
https://gradients.io/app/research/tournament/{TOURNAMENT_ID}
```
