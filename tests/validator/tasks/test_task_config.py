from datetime import datetime
from uuid import uuid4

import pytest

from core.models.utility_models import TaskStatus
from core.service_paths import START_TRAINING_ENDPOINT
from core.service_paths import START_TRAINING_GRPO_ENDPOINT
from core.service_paths import START_TRAINING_IMAGE_ENDPOINT
from validator.shared.models import ChatRawTask
from validator.shared.models import DpoRawTask
from validator.shared.models import EnvRawTask
from validator.shared.models import GrpoRawTask
from validator.shared.models import ImageRawTask
from validator.shared.models import InstructTextRawTask
from validator.shared.models import RewardFunction
from validator.tasks.config import get_task_config


def _base_task_data() -> dict:
    return {
        "is_organic": True,
        "status": TaskStatus.PENDING.value,
        "model_id": "base-model",
        "ds": "dataset",
        "account_id": uuid4(),
        "hours_to_complete": 1.0,
        "created_at": datetime(2026, 1, 1),
    }


@pytest.mark.parametrize(
    ("task", "expected_endpoint"),
    [
        (InstructTextRawTask(**_base_task_data(), field_instruction="instruction"), START_TRAINING_ENDPOINT),
        (ChatRawTask(**_base_task_data()), START_TRAINING_ENDPOINT),
        (
            DpoRawTask(
                **_base_task_data(),
                field_prompt="prompt",
                field_chosen="chosen",
                field_rejected="rejected",
            ),
            START_TRAINING_ENDPOINT,
        ),
        (
            GrpoRawTask(
                **_base_task_data(),
                field_prompt="prompt",
                reward_functions=[RewardFunction(reward_func="def reward_func(): pass", reward_weight=1.0)],
            ),
            START_TRAINING_GRPO_ENDPOINT,
        ),
        (EnvRawTask(**_base_task_data()), START_TRAINING_GRPO_ENDPOINT),
        (ImageRawTask(**_base_task_data()), START_TRAINING_IMAGE_ENDPOINT),
    ],
)
def test_task_config_routes_raw_tasks_to_trainer_endpoint(task, expected_endpoint):
    assert get_task_config(task).start_training_endpoint == expected_endpoint
