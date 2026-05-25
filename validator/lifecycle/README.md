# Validator Lifecycle

This package contains the long-running worker loop for ordinary validator tasks.

- `runner.py` is the executable entrypoint used by validator deployments.
- `tasks.py` advances tasks through preparation, assignment, model prep, training, evaluation, and completion states.

Tournament-specific orchestration lives in `validator/tournament/`; task data preparation helpers live in `validator/tasks/`.
