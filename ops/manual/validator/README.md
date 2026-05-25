# Validator Manual Probes

These scripts are for one-off local inspection and smoke checks. They are not pytest tests and should not run in CI.

- `check_scoring.py` inspects tournament scoring for one hotkey.
- `debug_weight_calculation.py` prints detailed tournament weight inputs and outputs.
- `instruct_eval_container.py` runs a manual instruct-text evaluation container.
- `run_text_evaluation_probe.py` and `run_image_evaluation_probe.py` run local Docker evaluation probes.
- `simple_eval_grpo.sh` runs a manual GRPO evaluation from a provided dataset URL or local JSON file.
- `tournament_burn_probe.py` demonstrates burn behavior with mock tournaments.
- `prepare_task_probe.py`, `burn_mock_cycle.py`, and `task_inspection.sql` support local validator debugging.

