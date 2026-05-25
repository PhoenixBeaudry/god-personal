from datetime import datetime
from datetime import timezone
from uuid import uuid4

from core.models.utility_models import ImageTextPair
from core.models.utility_models import RewardFunction
from core.models.utility_models import TaskStatus
from core.models.utility_models import TaskType
from validator.db import constants as cst
from validator.db.sql import auditing
from validator.shared.models import GrpoTask
from validator.shared.models import ImageTaskWithHotkeyDetails
from validator.shared.models import InstructTextTask


def _base_task_data(task_type: TaskType | str) -> dict:
    return {
        "is_organic": True,
        "task_id": uuid4(),
        "status": TaskStatus.READY.value,
        "model_id": "base-model",
        "ds": "dataset",
        "account_id": uuid4(),
        "hours_to_complete": 1.0,
        "created_at": datetime.now(timezone.utc),
        "task_type": task_type,
    }


def test_parse_image_text_pairs_accepts_json_strings_dicts_and_ignores_invalid_items():
    pairs = auditing._parse_image_text_pairs(
        [
            {"image_url": "s3://image-a", "text_url": "s3://text-a"},
            '{"image_url": "s3://image-b", "text_url": "s3://text-b"}',
            "{not-json",
        ]
    )

    assert pairs == [
        ImageTextPair(image_url="s3://image-a", text_url="s3://text-a"),
        ImageTextPair(image_url="s3://image-b", text_url="s3://text-b"),
    ]
    assert auditing._parse_image_text_pairs("{not-json") == []


def test_parse_reward_functions_accepts_json_strings_and_dicts():
    reward_functions = auditing._parse_reward_functions(
        [
            {
                "reward_func": "def reward_a(): return 1",
                "func_hash": "hash-a",
                "is_generic": True,
                "reward_weight": 0.25,
            },
            '{"reward_func": "def reward_b(): return 2", "func_hash": "hash-b", "is_generic": false, "reward_weight": 0.75}',
        ]
    )

    assert reward_functions == [
        RewardFunction(reward_func="def reward_a(): return 1", func_hash="hash-a", is_generic=True, reward_weight=0.25),
        RewardFunction(reward_func="def reward_b(): return 2", func_hash="hash-b", is_generic=False, reward_weight=0.75),
    ]


def test_build_task_for_audit_filters_join_aliases_and_accepts_string_types():
    task = auditing._build_task_for_audit(
        TaskType.INSTRUCTTEXTTASK.value,
        {
            **_base_task_data(TaskType.INSTRUCTTEXTTASK.value),
            "field_instruction": "instruction",
            "field_output": "output",
            "joined_alias_that_should_not_leak": "ignored",
        },
    )

    assert isinstance(task, InstructTextTask)
    assert task.field_instruction == "instruction"
    assert not hasattr(task, "joined_alias_that_should_not_leak")


def test_build_task_with_hotkey_details_uses_matching_wrapper_model():
    task = auditing._build_task_with_hotkey_details(
        TaskType.IMAGETASK.value,
        {
            **_base_task_data(TaskType.IMAGETASK.value),
            "status": TaskStatus.SUCCESS.value,
            "model_type": "sdxl",
            "image_text_pairs": [ImageTextPair(image_url="s3://image", text_url="s3://text")],
        },
        [],
    )

    assert isinstance(task, ImageTaskWithHotkeyDetails)
    assert task.hotkey_details == []
    assert task.image_text_pairs == [ImageTextPair(image_url="s3://image", text_url="s3://text")]


def test_group_task_ids_by_type_accepts_string_and_enum_task_types():
    text_id = str(uuid4())
    image_id = str(uuid4())
    grpo_id = str(uuid4())

    grouped = auditing._group_task_ids_by_type(
        {
            text_id: {cst.TASK_TYPE: TaskType.INSTRUCTTEXTTASK.value},
            image_id: {cst.TASK_TYPE: TaskType.IMAGETASK},
            grpo_id: {cst.TASK_TYPE: TaskType.GRPOTASK.value},
        }
    )

    assert grouped[TaskType.INSTRUCTTEXTTASK] == [text_id]
    assert grouped[TaskType.IMAGETASK] == [image_id]
    assert grouped[TaskType.GRPOTASK] == [grpo_id]


def test_higher_is_better_sql_values_follow_shared_task_group():
    assert "'GrpoTask'" in auditing._HIGHER_IS_BETTER_SQL_VALUES
    assert "'EnvTask'" in auditing._HIGHER_IS_BETTER_SQL_VALUES
    assert "'DpoTask'" not in auditing._HIGHER_IS_BETTER_SQL_VALUES


def test_build_grpo_task_for_audit_keeps_reward_functions():
    reward_functions = [RewardFunction(reward_func="def reward(): return 1", reward_weight=1.0)]

    task = auditing._build_task_for_audit(
        TaskType.GRPOTASK,
        {
            **_base_task_data(TaskType.GRPOTASK),
            "field_prompt": "prompt",
            "reward_functions": reward_functions,
        },
    )

    assert isinstance(task, GrpoTask)
    assert task.reward_functions == reward_functions
