# SQL Helpers

SQL helpers are grouped around the tables or workflows they serve.

- `tasks.py`, `submissions_and_scoring.py`, `grpo.py`, and `benchmark_tasks.py` serve task intake, evaluation, and benchmark data.
- `tournaments.py` and `tournament_performance.py` serve tournament lifecycle, API details, and legacy performance fields.
- `auditing.py`, `nodes.py`, and `transfers.py` serve validator operations outside task creation.
- `normalization.py` contains small row-normalization helpers shared by SQL modules.
