import sys
from types import SimpleNamespace

import pytest

from core.models.utility_models import TaskType
from validator.shared.constants import STANDARD_DPO_CHOSEN_COLUMN
from validator.shared.constants import STANDARD_DPO_PROMPT_COLUMN
from validator.shared.constants import STANDARD_DPO_REJECTED_COLUMN
from validator.shared.constants import STANDARD_GRPO_PROMPT_COLUMN
from validator.shared.constants import STANDARD_INSTRUCT_COLUMN
from validator.shared.constants import STANDARD_OUTPUT_COLUMN
from validator.tasks import dataset_mapping


@pytest.mark.asyncio
async def test_get_dataset_column_mapping_uses_task_type_builders(monkeypatch):
    async def call_content_service_fast(_url, _keypair):
        return {
            "field_instruction": "question",
            "field_output": "answer",
            "field_input": None,
            "field_system": "system",
            "field_prompt": "prompt_text",
            "field_chosen": "accepted",
            "field_rejected": "rejected",
        }

    monkeypatch.setitem(
        sys.modules,
        "validator.infrastructure.content_service",
        SimpleNamespace(call_content_service_fast=call_content_service_fast),
    )

    assert await dataset_mapping.get_dataset_column_mapping("dataset-a", TaskType.CHATTASK.value, object()) == {
        "instruction": "question",
        "output": "answer",
        "system": "system",
    }
    assert await dataset_mapping.get_dataset_column_mapping("dataset-a", TaskType.DPOTASK, object()) == {
        "prompt": "prompt_text",
        "chosen": "accepted",
        "rejected": "rejected",
        "system": "system",
    }
    assert await dataset_mapping.get_dataset_column_mapping("dataset-a", TaskType.GRPOTASK, object()) == {
        "prompt": "prompt_text"
    }
    assert await dataset_mapping.get_dataset_column_mapping("dataset-a", TaskType.ENVIRONMENTTASK, object()) == {
        "prompt": "prompt"
    }


def test_standardize_samples_dispatches_by_string_task_type():
    task = SimpleNamespace(
        task_type=TaskType.DPOTASK.value,
        field_prompt="prompt",
        field_chosen="chosen",
        field_rejected="rejected",
    )

    standardized = dataset_mapping.standardize_samples(
        [{"prompt": {"nested": True}, "chosen": "yes", "rejected": None}],
        task,
    )

    assert standardized == [
        {
            STANDARD_DPO_PROMPT_COLUMN: '{"nested": true}',
            STANDARD_DPO_CHOSEN_COLUMN: "yes",
            STANDARD_DPO_REJECTED_COLUMN: "",
        }
    ]


def test_create_temp_task_from_mapping_supports_chat_text_columns():
    temp_task = dataset_mapping.create_temp_task_from_mapping(
        {"instruction": "question", "output": "answer"},
        TaskType.CHATTASK.value,
    )

    assert temp_task.task_type == TaskType.CHATTASK
    assert temp_task.field_instruction == "question"
    assert temp_task.field_output == "answer"
    assert dataset_mapping.standardize_samples([{"question": "What?", "answer": "This."}], temp_task) == [
        {
            STANDARD_INSTRUCT_COLUMN: "What?",
            STANDARD_OUTPUT_COLUMN: "This.",
        }
    ]


def test_create_temp_task_from_mapping_supports_grpo_columns():
    temp_task = dataset_mapping.create_temp_task_from_mapping({"prompt": "question"}, TaskType.GRPOTASK)

    assert temp_task.task_type == TaskType.GRPOTASK
    assert temp_task.field_prompt == "question"
    assert dataset_mapping.standardize_samples([{"question": "Solve it"}], temp_task) == [
        {STANDARD_GRPO_PROMPT_COLUMN: "Solve it"}
    ]


def test_create_temp_task_from_mapping_returns_none_for_unsupported_task_type():
    assert dataset_mapping.create_temp_task_from_mapping({}, TaskType.IMAGETASK) is None
