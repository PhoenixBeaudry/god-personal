# Miner

This package provides the reference miner HTTP service for the tournament
training-repository contract.

The service exposes:

- `GET /training_repo/{task_type}`

`task_type` is `text`, `image`, or `environment`. The endpoint returns a
`TrainingRepoResponse` with the GitHub repository, commit, optional token, and
optional whitelisted datasets the validator should use for tournament training.

## Configuration

The miner reads `.miner.env` by default. Shared values apply to every tournament
family:

```env
MINER_TRAINING_REPO=https://github.com/YOUR_ORG/YOUR_TRAINING_REPO
MINER_TRAINING_COMMIT=YOUR_COMMIT_SHA
MINER_GITHUB_TOKEN=
MINER_REQUESTED_DATASETS=
```

Family-specific values override shared values:

```env
MINER_TEXT_TRAINING_REPO=https://github.com/YOUR_ORG/YOUR_TEXT_REPO
MINER_IMAGE_TRAINING_REPO=https://github.com/YOUR_ORG/YOUR_IMAGE_REPO
MINER_ENVIRONMENT_TRAINING_REPO=https://github.com/YOUR_ORG/YOUR_ENV_REPO
```

## Run

```bash
uvicorn miner.asgi:app --host 0.0.0.0 --port 7999 --env-file .miner.env
```
