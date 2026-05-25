# Core

`core/` is the cross-service contract package used by both the validator and trainer.

- `models/` defines shared payloads, task types, network/status models, tournament models, scoring models, and model-prep models.
- `datasets/` contains shared dataset adapters, diffusion dataset prep, examples, and the tournament dataset whitelist.
- `downloads.py` and `git.py` contain small shared IO helpers used across validator, trainer, and ops.
- `logging.py` provides shared structured logging, contextual tags, and container log streaming used by validator, trainer, ops, and tests.
- `training_config.py` builds Axolotl-compatible dataset/config entries from shared dataset models.
- `training_templates/` contains static Axolotl, diffusion, and PvP prompt templates.
- `constants.py` contains shared constants that remain part of the compatibility surface.

Validator-only models/config live in `validator/shared/`; trainer runtime concerns live in `trainer/`.
`core/` should stay dependency-light: validator and trainer may import it, but it should not import either service package.
