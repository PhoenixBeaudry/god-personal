from datetime import datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest

from core.constants import EnvironmentName
from core.models.payload_models import ChatTaskDetails
from core.models.payload_models import DpoTaskDetails
from core.models.payload_models import EnvironmentTaskDetails
from core.models.payload_models import GrpoTaskDetails
from core.models.payload_models import ImageTaskDetails
from core.models.payload_models import InstructTextTaskDetails
from core.models.utility_models import ImageModelType
from core.models.utility_models import ImageTextPair
from core.models.utility_models import RewardFunction
from core.models.utility_models import TaskStatus
from core.models.utility_models import TaskType
from validator.tasks.details import convert_task_to_task_details
from validator.tasks.details import hide_sensitive_data_till_finished


def _task(task_type: TaskType, **overrides):
    task = SimpleNamespace(
        task_id=uuid4(),
        account_id=uuid4(),
        status=TaskStatus.PENDING,
        model_id="base/model",
        ds="dataset/repo",
        created_at=datetime(2025, 1, 1),
        started_at=None,
        termination_at=None,
        hours_to_complete=1.0,
        trained_model_repository=None,
        task_type=task_type,
        result_model_name="result",
        field_input="input",
        field_system="system",
        field_instruction="instruction",
        field_output="output",
        format=None,
        no_input_format=None,
        system_format=None,
        chat_template="chatml",
        chat_column="messages",
        chat_role_field="role",
        chat_content_field="content",
        chat_user_reference="user",
        chat_assistant_reference="assistant",
        image_text_pairs=[ImageTextPair(image_url="image", text_url="text")],
        model_type=ImageModelType.SDXL,
        field_prompt="prompt",
        field_chosen="chosen",
        field_rejected="rejected",
        prompt_format="{prompt}",
        chosen_format="{chosen}",
        rejected_format="{rejected}",
        reward_functions=[RewardFunction(reward_func="def reward(): pass", reward_weight=1.0)],
        environment_names=[EnvironmentName.LIARS_DICE],
        eval_seed=123,
        test_data="secret-test",
        training_data="secret-train",
    )
    for key, value in overrides.items():
        setattr(task, key, value)
    return task


@pytest.mark.parametrize(
    ("task_type", "expected_model"),
    [
        (TaskType.INSTRUCTTEXTTASK, InstructTextTaskDetails),
        (TaskType.CHATTASK, ChatTaskDetails),
        (TaskType.IMAGETASK, ImageTaskDetails),
        (TaskType.DPOTASK, DpoTaskDetails),
        (TaskType.GRPOTASK, GrpoTaskDetails),
        (TaskType.ENVIRONMENTTASK, EnvironmentTaskDetails),
    ],
)
def test_convert_task_to_task_details_preserves_response_model(task_type: TaskType, expected_model):
    details = convert_task_to_task_details(_task(task_type))

    assert isinstance(details, expected_model)
    assert details.task_type == task_type
    assert details.base_model_repository == "base/model"
    assert details.ds_repo == "dataset/repo"
    assert details.result_model_name == "result"


def test_hide_sensitive_data_hides_image_and_shared_data_for_in_flight_tasks():
    task = _task(TaskType.IMAGETASK)

    hidden = hide_sensitive_data_till_finished(task)

    assert hidden.image_text_pairs == [ImageTextPair(image_url="hidden", text_url="hidden")]
    assert hidden.test_data is None
    assert hidden.training_data is None
    assert hidden.ds == "Hidden"


def test_hide_sensitive_data_hides_environment_eval_seed_for_in_flight_tasks():
    task = _task(TaskType.ENVIRONMENTTASK)

    hidden = hide_sensitive_data_till_finished(task)

    assert hidden.eval_seed is None
    assert hidden.test_data is None
    assert hidden.training_data is None
    assert hidden.ds == "Hidden"


def test_hide_sensitive_data_leaves_completed_tasks_unchanged():
    task = _task(TaskType.IMAGETASK, status=TaskStatus.SUCCESS)

    visible = hide_sensitive_data_till_finished(task)

    assert visible.image_text_pairs == [ImageTextPair(image_url="image", text_url="text")]
    assert visible.test_data == "secret-test"
    assert visible.training_data == "secret-train"
    assert visible.ds == "dataset/repo"
