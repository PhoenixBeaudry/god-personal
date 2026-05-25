import pytest

from core.models.utility_models import TaskType
from validator.tournament.performance_calculator import _best_score
from validator.tournament.performance_calculator import _relative_performance_difference


@pytest.mark.parametrize(
    ("task_type", "expected"),
    [
        (TaskType.INSTRUCTTEXTTASK, 1.0),
        (TaskType.DPOTASK, 1.0),
        (TaskType.GRPOTASK, 3.0),
        (TaskType.ENVIRONMENTTASK, 3.0),
    ],
)
def test_best_score_respects_task_direction(task_type: TaskType, expected: float):
    assert _best_score([1.0, 2.0, 3.0], task_type) == expected


def test_relative_performance_difference_for_lower_is_better_tasks():
    diff = _relative_performance_difference(
        tournament_winner_score=1.25,
        benchmark_score=1.0,
        task_type=TaskType.INSTRUCTTEXTTASK,
    )

    assert diff == pytest.approx(0.25)


def test_relative_performance_difference_for_higher_is_better_tasks():
    diff = _relative_performance_difference(
        tournament_winner_score=0.75,
        benchmark_score=1.0,
        task_type=TaskType.GRPOTASK,
    )

    assert diff == pytest.approx(0.25)


def test_relative_performance_difference_handles_zero_benchmark():
    assert _relative_performance_difference(1.0, 0.0, TaskType.GRPOTASK) == 0.0
