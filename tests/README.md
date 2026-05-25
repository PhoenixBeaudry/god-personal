# Tests

Tests mirror the source tree so a failing path points at the owning subsystem.

- `core/` covers shared contracts and helpers.
- `trainer/` covers trainer routing, config paths, and model-prep behavior.
- `validator/db/` covers database SQL helpers and audit serialization.
- `validator/tasks/` covers task preparation, dataset mapping, augmentation, and task details.
- `validator/lifecycle/` covers validator task-processing lifecycle helpers.
- `validator/evaluation/` covers scoring, PvP evaluation, and tournament score conversion.
- `validator/tournament/` covers tournament specs, orchestration, analytics, performance, and weight behavior.
- `ops/` covers operational config and Dockerfile regressions that are not exercised by Python imports.
- `e2e/` contains local end-to-end PvP runners.

Manual probes and one-off scripts live under `ops/manual/` instead of this tree.

Run focused tests by path during development. Coverage is opt-in so the default pytest command stays fast:

```bash
pytest tests/trainer/test_trainer_task_routing.py
pytest --cov=core --cov=trainer --cov=validator --cov-report=term-missing --cov-report=xml
```
