from types import SimpleNamespace

import pytest

from core.models.utility_models import TaskType
from validator.lifecycle import tasks
from validator.shared import constants as cst


def _task(task_type: TaskType | str, params: int = 0, model_id: str = "base-model"):
    return SimpleNamespace(task_type=task_type, model_params_count=params, model_id=model_id)


def test_grpo_uses_batched_evaluation_without_pvp_lookup(monkeypatch):
    def fail_if_called(_task):
        raise AssertionError("GRPO should not need environment PvP lookup")

    monkeypatch.setattr(tasks, "should_use_pvp", fail_if_called)

    assert tasks._uses_batched_evaluation(_task(TaskType.GRPOTASK))


def test_environment_pvp_tasks_use_batched_evaluation(monkeypatch):
    monkeypatch.setattr(tasks, "should_use_pvp", lambda task: task.task_type == TaskType.ENVIRONMENTTASK)

    assert tasks._uses_batched_evaluation(_task(TaskType.ENVIRONMENTTASK))
    assert not tasks._uses_batched_evaluation(_task(TaskType.INSTRUCTTEXTTASK))


def test_only_grpo_recovery_resets_all_evaluation_rows():
    assert tasks._resets_all_evaluation_rows(_task(TaskType.GRPOTASK))
    assert tasks._resets_all_evaluation_rows(_task(TaskType.GRPOTASK.value))
    assert not tasks._resets_all_evaluation_rows(_task(TaskType.ENVIRONMENTTASK))
    assert not tasks._resets_all_evaluation_rows(_task(TaskType.INSTRUCTTEXTTASK))


@pytest.mark.parametrize(
    ("task_type", "base_params", "expected_gpus"),
    [
        (TaskType.INSTRUCTTEXTTASK, cst.MODEL_SIZE_REQUIRING_2_GPUS - 1, 1),
        (TaskType.DPOTASK, cst.MODEL_SIZE_REQUIRING_2_GPUS // 2, 2),
        (TaskType.DPOTASK.value, cst.MODEL_SIZE_REQUIRING_2_GPUS // 2, 2),
        (TaskType.GRPOTASK, cst.MODEL_SIZE_REQUIRING_3_GPUS // 3 + 1, 3),
        (TaskType.ENVIRONMENTTASK, cst.MODEL_SIZE_REQUIRING_3_GPUS // 3 + 1, 3),
        (TaskType.IMAGETASK, cst.MODEL_SIZE_REQUIRING_4_GPUS, 4),
    ],
)
def test_compute_required_gpus_preserves_task_type_multipliers(monkeypatch, task_type, base_params, expected_gpus):
    def fail_if_called(_model):
        raise AssertionError("cached model_params_count should be used")

    monkeypatch.setattr(tasks, "get_model_num_params", fail_if_called)

    assert tasks.compute_required_gpus(_task(task_type, params=base_params)) == expected_gpus


def test_compute_required_gpus_fetches_missing_model_params(monkeypatch):
    calls = []

    def get_model_num_params(model_id):
        calls.append(model_id)
        return cst.MODEL_SIZE_REQUIRING_4_GPUS

    monkeypatch.setattr(tasks, "get_model_num_params", get_model_num_params)

    assert tasks.compute_required_gpus(_task(TaskType.INSTRUCTTEXTTASK, model_id="missing-count")) == 4
    assert calls == ["missing-count"]


def test_compute_required_gpus_defaults_to_one_when_param_count_is_unknown(monkeypatch):
    monkeypatch.setattr(tasks, "get_model_num_params", lambda _model_id: 0)

    assert tasks.compute_required_gpus(_task(TaskType.INSTRUCTTEXTTASK)) == 1
