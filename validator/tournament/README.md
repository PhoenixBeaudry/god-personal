# Validator Tournaments

Tournament code is organized around the three supported tournament families: text, image, and environment.

- `runner.py` is the long-running tournament worker entrypoint.
- `specs.py` defines the supported tournament families and compatibility helpers.
- `weighting.py` centralizes tournament weight scaling.
- `tournament_manager.py` coordinates tournament lifecycle state.
- `task_creator.py` creates tournament tasks for text, image, and environment rounds.
- `orchestrator.py` schedules trainer work for tournament submissions.
- `trainer_client.py` contains the HTTP client helpers used to talk to trainer services.
- `round_results.py` determines group, knockout, and boss-round winners.
- `environment_results.py` handles environment-specific group advancement and boss comparisons.
- `task_results.py`, `thresholds.py`, and `champions.py` load score rows, apply progressive thresholds, and resolve compatibility winner fields.
- `participants.py` validates participant repositories, GitHub tokens, and previous-winner contestants.
- `resources.py` maps task/model size to trainer GPU requirements.
- `reports.py`, `notifications.py`, and `brackets.py` handle completion reports, Discord messages, and readable bracket logs.
- `repo_diff_report.py` compares a tournament winner against the previous boss and uploads the report.
- `performance_calculator.py`, `performance_utils.py`, and `benchmark_utils.py` compute and normalize results.
- `dstack_orchestrator.py`, `repo_uploader.py`, and `transfer_monitoring.py` are infrastructure adapters.

Public tournament API routes live in `validator/endpoints/`.
