from datetime import datetime
from datetime import timezone
from uuid import uuid4

import pytest

from core.models.utility_models import ImageModelType
from core.models.utility_models import ImageTextPair
from core.models.utility_models import RewardFunction
from core.models.utility_models import TaskStatus
from core.models.utility_models import TaskType
from validator.db.sql import tasks as tasks_sql
from validator.shared.models import GrpoRawTask
from validator.shared.models import ImageTask
from validator.shared.models import InstructTextRawTask


class _FakeConnection:
    def __init__(self, fields):
        self.fields = fields
        self.executions = []

    async def fetch(self, _query, _table_name):
        return [{"column_name": field} for field in self.fields]

    async def execute(self, query, *args):
        self.executions.append((query, args))


def _base_task_data(task_id, task_type: TaskType | str) -> dict:
    return {
        "is_organic": True,
        "task_id": task_id,
        "status": TaskStatus.READY.value,
        "model_id": "base-model",
        "ds": "dataset",
        "account_id": uuid4(),
        "hours_to_complete": 1.5,
        "created_at": datetime.now(timezone.utc),
        "task_type": task_type,
    }


@pytest.mark.asyncio
async def test_build_task_from_data_creates_raw_text_task_from_string_type():
    task_id = uuid4()
    task = await tasks_sql._build_task_from_data(
        TaskType.INSTRUCTTEXTTASK.value,
        {
            **_base_task_data(task_id, TaskType.INSTRUCTTEXTTASK.value),
            "field_instruction": "instruction",
            "field_output": "output",
        },
        task_id,
        object(),
    )

    assert isinstance(task, InstructTextRawTask)
    assert task.task_type == TaskType.INSTRUCTTEXTTASK
    assert task.field_instruction == "instruction"


@pytest.mark.asyncio
async def test_build_task_from_data_loads_image_pairs_for_public_task(monkeypatch):
    task_id = uuid4()
    psql_db = object()
    connection = object()
    calls = []

    async def get_image_text_pairs(received_task_id, received_psql_db, received_connection=None):
        calls.append((received_task_id, received_psql_db, received_connection))
        return [ImageTextPair(image_url="s3://image", text_url="s3://text")]

    monkeypatch.setattr(tasks_sql, "get_image_text_pairs", get_image_text_pairs)

    task = await tasks_sql._build_task_from_data(
        TaskType.IMAGETASK,
        {
            **_base_task_data(task_id, TaskType.IMAGETASK),
            "model_type": ImageModelType.SDXL.value,
            "trained_model_repository": "winner/repo",
        },
        task_id,
        psql_db,
        connection,
        public=True,
    )

    assert isinstance(task, ImageTask)
    assert task.trained_model_repository == "winner/repo"
    assert task.image_text_pairs == [ImageTextPair(image_url="s3://image", text_url="s3://text")]
    assert calls == [(task_id, psql_db, connection)]


@pytest.mark.asyncio
async def test_build_task_from_data_loads_grpo_reward_functions(monkeypatch):
    task_id = uuid4()
    psql_db = object()
    connection = object()
    reward_functions = [RewardFunction(reward_func="def reward(): return 1", reward_weight=1.0)]
    calls = []

    async def get_reward_functions(received_task_id, received_psql_db, received_connection=None):
        calls.append((received_task_id, received_psql_db, received_connection))
        return reward_functions

    monkeypatch.setattr(tasks_sql, "get_reward_functions", get_reward_functions)

    task = await tasks_sql._build_task_from_data(
        TaskType.GRPOTASK.value,
        {
            **_base_task_data(task_id, TaskType.GRPOTASK.value),
            "field_prompt": "prompt",
            "file_format": "hf",
            "extra_column": "metadata",
        },
        task_id,
        psql_db,
        connection,
    )

    assert isinstance(task, GrpoRawTask)
    assert task.reward_functions == reward_functions
    assert task.extra_column == "metadata"
    assert calls == [(task_id, psql_db, connection)]


@pytest.mark.asyncio
async def test_get_specific_task_updates_filters_to_table_fields_and_excludes_task_id():
    connection = _FakeConnection({"task_id", "field_prompt", "field_system"})

    updates = await tasks_sql._get_specific_task_updates(
        "dpo_tasks",
        {"task_id": "ignored", "field_prompt": "prompt", "model_id": "base"},
        connection,
    )

    assert updates == {"field_prompt": "prompt"}


@pytest.mark.asyncio
async def test_update_task_specific_fields_builds_one_parameterized_update():
    task_id = uuid4()
    connection = _FakeConnection({"task_id", "field_prompt", "field_system"})

    await tasks_sql._update_task_specific_fields(
        connection,
        task_id,
        "dpo_tasks",
        {"field_prompt": "prompt", "field_system": "system", "model_id": "base"},
    )

    assert len(connection.executions) == 1
    query, args = connection.executions[0]
    assert "UPDATE dpo_tasks" in query
    assert "field_prompt = $2" in query
    assert "field_system = $3" in query
    assert args == (task_id, "prompt", "system")
