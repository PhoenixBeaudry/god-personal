# Validator Evaluation

`validator/evaluation/` turns a prepared task and a submitted model into score rows that the lifecycle and tournament systems can consume.

- `docker_evaluation.py` launches containerized evaluation jobs.
- `local_evaluation.py` runs the same evaluation path without the validator queue.
- `eval_instruct_text.py`, `eval_dpo.py`, `eval_grpo.py`, `eval_diffusion.py`, and `eval_environment.py` contain task-family-specific evaluators.
- `single_eval_*.py` modules are single-model/container entrypoints used by evaluator jobs.
- `basilica.py` and `basilica_deployments.py` manage Basilica-backed environment evaluation deployments.
- `pvp/` contains environment PvP agents, game serving, group evaluation, and scoring.
- `ranking.py`, `scoring.py`, and `tournament_scoring.py` convert raw metrics into validator and tournament scores.
- `result_processing.py` normalizes evaluator payloads into shared response models.
- `model_loading.py` and `model_checks.py` load models/tokenizers and validate model identity or LoRA metadata.
- `container_results.py`, `evaluation_config.py`, `evaluation_logging.py`, `kl_divergence.py`, `dataset_configs.py`, `image_io.py`, and `runtime.py` are small support modules for result-file IO, Axolotl eval config, logging, KL scoring, dataset config discovery, image payload handling, and process control.
