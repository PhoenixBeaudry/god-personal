import pytest

from core.models.utility_models import HIGHER_IS_BETTER_TASK_TYPES
from core.models.utility_models import IMAGE_TASK_TYPES
from core.models.utility_models import TEXT_TOURNAMENT_TASK_TYPES
from core.models.utility_models import TEXT_TRAINER_TASK_TYPES
from core.models.utility_models import TRAINER_TASK_TYPES
from core.models.utility_models import ChatTemplateDatasetType
from core.models.utility_models import DpoDatasetType
from core.models.utility_models import EnvironmentDatasetType
from core.models.utility_models import GrpoDatasetType
from core.models.utility_models import InstructTextDatasetType
from core.models.utility_models import TaskType
from core.models.utility_models import is_environment_task
from core.models.utility_models import is_image_task
from core.models.utility_models import normalize_task_type
from core.models.utility_models import scores_higher_is_better
from core.models.utility_models import task_type_for_dataset_type
from core.models.utility_models import uses_text_trainer


def test_task_type_groups_are_explicit_and_stable():
    assert TEXT_TOURNAMENT_TASK_TYPES == (
        TaskType.INSTRUCTTEXTTASK,
        TaskType.CHATTASK,
        TaskType.DPOTASK,
        TaskType.GRPOTASK,
    )
    assert TEXT_TRAINER_TASK_TYPES == TEXT_TOURNAMENT_TASK_TYPES + (TaskType.ENVIRONMENTTASK,)
    assert IMAGE_TASK_TYPES == (TaskType.IMAGETASK,)
    assert TRAINER_TASK_TYPES == IMAGE_TASK_TYPES + TEXT_TRAINER_TASK_TYPES
    assert HIGHER_IS_BETTER_TASK_TYPES == (TaskType.GRPOTASK, TaskType.ENVIRONMENTTASK)


def test_task_type_helpers_accept_enum_and_string_values():
    assert normalize_task_type("ImageTask") == TaskType.IMAGETASK
    assert uses_text_trainer(TaskType.INSTRUCTTEXTTASK)
    assert uses_text_trainer("EnvTask")
    assert is_image_task(TaskType.IMAGETASK)
    assert is_image_task("ImageTask")
    assert is_environment_task(TaskType.ENVIRONMENTTASK)
    assert scores_higher_is_better(TaskType.GRPOTASK)
    assert scores_higher_is_better("EnvTask")


def test_task_type_helpers_reject_unknown_values():
    with pytest.raises(ValueError):
        normalize_task_type("AudioTask")


@pytest.mark.parametrize(
    ("dataset_type", "expected_task_type"),
    [
        (InstructTextDatasetType(), TaskType.INSTRUCTTEXTTASK),
        (ChatTemplateDatasetType(), TaskType.CHATTASK),
        (DpoDatasetType(), TaskType.DPOTASK),
        (GrpoDatasetType(), TaskType.GRPOTASK),
        (EnvironmentDatasetType(), TaskType.ENVIRONMENTTASK),
    ],
)
def test_task_type_for_dataset_type(dataset_type, expected_task_type):
    assert task_type_for_dataset_type(dataset_type) == expected_task_type
