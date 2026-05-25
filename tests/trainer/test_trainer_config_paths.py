import os

import pytest

import core.constants as core_cst
from core.models.utility_models import ChatTemplateDatasetType
from core.models.utility_models import DpoDatasetType
from core.models.utility_models import EnvironmentDatasetType
from core.models.utility_models import FileFormat
from core.models.utility_models import GrpoDatasetType
from core.models.utility_models import InstructTextDatasetType
from core.training_config import create_dataset_entry
from trainer import constants as train_cst
from trainer.training_paths import get_axolotl_base_config_path


@pytest.mark.parametrize(
    ("dataset_type", "expected_config_name"),
    [
        (InstructTextDatasetType(), "base.yml"),
        (ChatTemplateDatasetType(), "base.yml"),
        (DpoDatasetType(), "base.yml"),
        (GrpoDatasetType(), "base_grpo.yml"),
        (EnvironmentDatasetType(), "base_environment.yml"),
    ],
)
def test_axolotl_base_config_path_uses_dataset_task_type(dataset_type, expected_config_name):
    expected = os.path.join(train_cst.AXOLOTL_DIRECTORIES["root"], expected_config_name)

    assert get_axolotl_base_config_path(dataset_type) == expected


def test_create_dataset_entry_for_instruct_completion_dataset():
    entry = create_dataset_entry(
        "dataset.json",
        InstructTextDatasetType(field_instruction="prompt"),
        FileFormat.S3,
    )

    assert entry["type"] == "completion"
    assert entry["field"] == "prompt"
    assert entry["ds_type"] == FileFormat.S3.value
    assert entry["data_files"] == ["dataset.json"]


@pytest.mark.parametrize(
    ("dataset_type", "expected_fields"),
    [
        (DpoDatasetType(), {"type": core_cst.DPO_DEFAULT_DATASET_TYPE, "split": "train"}),
        (GrpoDatasetType(), {"split": "train"}),
        (EnvironmentDatasetType(), {"split": "train"}),
    ],
)
def test_create_dataset_entry_for_specialized_text_trainers(dataset_type, expected_fields):
    entry = create_dataset_entry("dataset.json", dataset_type, FileFormat.S3)

    for key, value in expected_fields.items():
        assert entry[key] == value


def test_create_dataset_entry_for_chat_template_dataset():
    entry = create_dataset_entry(
        "dataset.json",
        ChatTemplateDatasetType(chat_column="messages", chat_role_field="role", chat_content_field="content"),
        FileFormat.HF,
    )

    assert entry["type"] == "chat_template"
    assert entry["field_messages"] == "messages"
    assert entry["message_field_role"] == "role"
    assert entry["message_field_content"] == "content"
    assert "ds_type" not in entry
