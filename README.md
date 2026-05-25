# G.O.D Subnet

G.O.D, Gradients on Demand, is the validator and trainer system behind Gradients.io tournaments on Bittensor subnet 56. The repo runs three tournament families:

- Text tournaments for instruct, chat, DPO, and GRPO training.
- Image tournaments for diffusion LoRA training.
- Environment tournaments for GRPO-style training against live game environments.

The current production surface is the validator, trainer, tournament orchestration, scoring, auditing, and local evaluation tooling. Miners participate through an external repository endpoint contract; the repo does not need to be used as a miner runtime.

## Start Here

The single primary guide is [docs/guide.md](docs/guide.md). It covers:

- Validator and trainer setup.
- Tournament lifecycle and scoring.
- Miner participation requirements.
- Required training repository layout, CLI arguments, environment variables, and outputs.
- Local evaluation commands.

Coding agents should also read [AGENTS.md](AGENTS.md) before making structural changes.

## Common Commands

```bash
task config          # create .vali.env for a validator
task trainer-config  # create .trainer.env for a trainer
task validator       # run validator services
task trainer         # run a trainer service
```

Run a local evaluation:

```bash
python ops/validator_ops/run_evaluation.py --help
python ops/validator_ops/run_evaluation.py --task_id <task_id>
python ops/validator_ops/run_evaluation.py --task_id <task_id> --models <model_repo>
```

## Repository Map

- `core/` - shared models, constants, logging, dataset helpers, training templates, and compatibility contracts.
- `validator/` - validator API, tournament orchestration, task creation, evaluation, scoring, infrastructure adapters, shared validator models, and weight setting.
- `trainer/` - trainer API, GPU job tracking, model prep, runtime helpers, and Docker sidecar entrypoints.
- `ops/` - operational assets: Dockerfiles, compose stacks, observability config, auditor scripts, manual probes, and one-off tools.
- `tests/` - source-aligned tests for core, trainer, validator DB, tasks, lifecycle, evaluation, and tournaments.
- `docs/guide.md` - the canonical docs.

## Public Resources

- Product: [gradients.io](https://gradients.io)
- API docs: [api.gradients.io/docs](https://api.gradients.io/docs)
- Tournament results: `https://gradients.io/app/research/tournament/{TOURNAMENT_ID}`
- Winner repositories: [github.com/gradients-opensource](https://github.com/gradients-opensource)
