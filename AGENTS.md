# Agent Guide

This file is for coding agents working inside this repository. Use it with the root `README.md` and the canonical user/operator docs in `docs/guide.md`.

## Current Shape

G.O.D is the validator, trainer, tournament, and auditing system for Gradients on Demand. The active product surface is intentionally narrow:

- `core/` contains shared contracts used by both validator and trainer.
- `validator/` contains the public API, DB access, task lifecycle, evaluation, tournament orchestration, infrastructure adapters, and validator-only shared models/config.
- `trainer/` contains the trainer API, GPU/job state, model prep, Docker runtime helpers, and container entrypoints.
- `ops/` contains Dockerfiles, compose stacks, observability config, operator scripts, local examples, and one-off tools.
- `tests/` mirrors the source tree.
- `docs/guide.md` is the canonical human-facing guide. Subtree `README.md` files are local maps, not replacement docs.

The repo supports three tournament families: text, image, and environment. The old in-repo miner runtime has been removed. Miners participate through the external `GET /training_repo/{task_type}` contract documented in `docs/guide.md`.

## Compatibility Boundary

Preserve backwards compatibility for:

- Public API routes and payloads.
- Database schema history and SQL behavior.
- Cross-service payload contracts in `core/models/` and validator/trainer request models.

Internal package layout, filenames, Docker paths, Taskfile commands, and helper modules may change when it makes the system simpler. Do not add compatibility shims for moved internal modules unless an API, DB, or deployed image still depends on them.

## Working Rules

- Keep organization descriptive. A folder should tell a new engineer what subsystem they are in.
- Prefer small, named modules over `utils.py`, `common.py`, or broad compatibility buckets.
- Keep validator-only code out of `core/`; `core/` must stay dependency-light and import neither `validator` nor `trainer`.
- Keep infrastructure adapters in `validator/infrastructure/`; keep task preparation in `validator/tasks/`; keep tournament bracket/result/orchestration code in `validator/tournament/`.
- Do not recreate a local `miner/` package or miner runtime.
- Keep docs centralized: update `docs/guide.md` for user/operator/miner behavior, and update local `README.md` files only when folder maps change.
- When moving files, update imports directly to the new module. Remove dead shims once repo searches prove they have no callers.
- Avoid unrelated refactors. This repo is large; move one responsibility boundary at a time and validate it.

## Validation

Use focused checks while iterating:

```bash
UV_CACHE_DIR=/private/tmp/uv-cache UV_PYTHON_INSTALL_DIR=/private/tmp/uv-python uv run --extra dev ruff check <paths>
UV_CACHE_DIR=/private/tmp/uv-cache UV_PYTHON_INSTALL_DIR=/private/tmp/uv-python uv run --extra dev python -m py_compile <paths>
UV_CACHE_DIR=/private/tmp/uv-cache UV_PYTHON_INSTALL_DIR=/private/tmp/uv-python uv run --extra dev pytest -q -o addopts='' <tests>
```

Useful focused suites:

```bash
tests/validator/tasks/test_dataset_mapping.py
tests/validator/evaluation/test_scoring_task_type_behavior.py
tests/validator/evaluation/test_tournament_scoring_pipeline.py
tests/validator/tournament/
tests/trainer/
```

If `uv` creates local artifacts during checks, clean them before handing off:

```bash
rm -rf .venv uv.lock .pytest_cache .ruff_cache
find core validator trainer ops tests -type d -name __pycache__ -prune -exec rm -rf {} +
```

## Documentation Checklist

Before calling a cleanup pass finished, verify:

- Root `README.md` still gives the right repo map and start-here path.
- `docs/guide.md` still documents validator setup, trainer setup, tournaments, scoring, local evaluation, and miner participation.
- Local READMEs under `core/`, `validator/`, `trainer/`, `ops/`, and `tests/` match the folders on disk.
- Searches for deleted paths such as `miner/`, `validator/core`, `validator/cycle`, `validator/utils`, `core/config`, `core/dataset`, `core/utils`, `trainer/utils`, `scripts/`, and `dockerfiles/` do not appear in active docs or imports unless they are historical notes.
- `pyproject.toml` source distribution metadata includes root handoff files such as `AGENTS.md`, `LICENSE.md`, `NOTICE`, and `README.md`.
