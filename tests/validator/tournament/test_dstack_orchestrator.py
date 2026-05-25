import json
from types import SimpleNamespace

import pytest

from core.models.tournament_models import GpuRequirement
from core.models.utility_models import Backend
from core.models.utility_models import TaskType
from validator.tournament import constants as t_cst
from validator.tournament import dstack_orchestrator


class _DatasetType:
    def model_dump(self):
        return {"kind": "instruct", "field_instruction": "instruction"}


def _task(task_type: TaskType | str, **overrides):
    values = {
        "task_id": "task-123",
        "task_type": task_type,
        "model_params_count": 1,
        "model_id": "base-model",
        "augmented_model_id": None,
        "hours_to_complete": 2.5,
        "training_data": "s3://bucket/train-data",
        "model_type": "sdxl",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_dstack_runtime_uses_image_profile_for_image_tasks(monkeypatch):
    monkeypatch.setenv("DSTACK_IMAGE_TASK_DOCKER_IMAGE", "image-trainer:test")

    runtime = dstack_orchestrator._get_dstack_task_runtime(TaskType.IMAGETASK.value, GpuRequirement.H100_4X)

    assert runtime.gpu_name == "H100"
    assert runtime.gpu_count == 4
    assert runtime.docker_image == "image-trainer:test"
    assert runtime.regions == t_cst.DSTACK_IMAGE_REGIONS


def test_dstack_runtime_uses_text_profile_for_text_and_environment_tasks(monkeypatch):
    monkeypatch.setenv("DSTACK_TEXT_TASK_DOCKER_IMAGE", "text-trainer:test")

    runtime = dstack_orchestrator._get_dstack_task_runtime(TaskType.ENVIRONMENTTASK, GpuRequirement.H100_4X)

    assert runtime.gpu_name == "H200"
    assert runtime.gpu_count == 3
    assert runtime.docker_image == "text-trainer:test"
    assert runtime.regions == t_cst.DSTACK_TEXT_REGIONS


def test_runpod_organic_training_filter_accepts_enum_and_string_backends():
    enum_backend = SimpleNamespace(priority=1, task=SimpleNamespace(backend=Backend.RUNPOD))
    string_backend = SimpleNamespace(priority=1, task=SimpleNamespace(backend=Backend.RUNPOD.value))
    wrong_priority = SimpleNamespace(priority=2, task=SimpleNamespace(backend=Backend.RUNPOD))
    missing_backend = SimpleNamespace(priority=1, task=SimpleNamespace(backend=None))

    assert dstack_orchestrator._is_runpod_organic_training_task(enum_backend)
    assert dstack_orchestrator._is_runpod_organic_training_task(string_backend)
    assert not dstack_orchestrator._is_runpod_organic_training_task(wrong_priority)
    assert not dstack_orchestrator._is_runpod_organic_training_task(missing_backend)


@pytest.mark.asyncio
async def test_create_dstack_request_builds_text_training_payload(monkeypatch):
    async def get_expected_repo_name(_task_id, _hotkey, _psql_db):
        return "expected-repo"

    monkeypatch.setattr(dstack_orchestrator.task_sql, "get_expected_repo_name", get_expected_repo_name)
    monkeypatch.setattr(dstack_orchestrator, "get_anonymous_model_dir", lambda model_id: f"anon/{model_id}")
    monkeypatch.setattr(dstack_orchestrator, "_get_dataset_type", lambda _task: _DatasetType())
    monkeypatch.setattr(
        dstack_orchestrator,
        "get_tournament_gpu_requirement",
        lambda _task_type, _params, _model_id: GpuRequirement.H100_2X,
    )
    monkeypatch.setenv("DSTACK_TEXT_TASK_DOCKER_IMAGE", "text-trainer:test")
    monkeypatch.setenv("HUGGINGFACE_USERNAME", "trainer-user")

    request = await dstack_orchestrator._create_dstack_request(
        _task(TaskType.INSTRUCTTEXTTASK, augmented_model_id="augmented-model"),
        "run-name",
        SimpleNamespace(psql_db=object()),
    )

    config = request["plan"]["run_spec"]["configuration"]
    env = config["env"]

    assert config["image"] == "text-trainer:test"
    assert config["resources"]["gpu"]["name"] == ["H200"]
    assert config["resources"]["gpu"]["count"] == {"min": 2, "max": 2}
    assert env["MODEL"] == "anon/augmented-model"
    assert env["TASK_TYPE"] == TaskType.INSTRUCTTEXTTASK.value
    assert env["EXPECTED_REPO_NAME"] == "expected-repo"
    assert env["DATASET"] == "s3://bucket/train-data"
    assert env["FILE_FORMAT"] == "s3"
    assert json.loads(env["DATASET_TYPE"]) == {"kind": "instruct", "field_instruction": "instruction"}
    assert env["HUGGINGFACE_USERNAME"] == "trainer-user"


@pytest.mark.asyncio
async def test_create_dstack_request_builds_image_training_payload(monkeypatch):
    async def get_expected_repo_name(_task_id, _hotkey, _psql_db):
        return None

    monkeypatch.setattr(dstack_orchestrator.task_sql, "get_expected_repo_name", get_expected_repo_name)
    monkeypatch.setattr(dstack_orchestrator, "get_anonymous_model_dir", lambda model_id: f"anon/{model_id}")
    monkeypatch.setattr(
        dstack_orchestrator,
        "get_tournament_gpu_requirement",
        lambda _task_type, _params, _model_id: GpuRequirement.H100_8X,
    )
    monkeypatch.setenv("DSTACK_IMAGE_TASK_DOCKER_IMAGE", "image-trainer:test")

    request = await dstack_orchestrator._create_dstack_request(
        _task(TaskType.IMAGETASK.value),
        "run-name",
        SimpleNamespace(psql_db=object()),
    )

    config = request["plan"]["run_spec"]["configuration"]
    env = config["env"]

    assert config["image"] == "image-trainer:test"
    assert config["resources"]["gpu"]["name"] == ["H100"]
    assert config["resources"]["gpu"]["count"] == {"min": 8, "max": 8}
    assert env["TASK_TYPE"] == TaskType.IMAGETASK.value
    assert env["EXPECTED_REPO_NAME"] == "organic_task-123"
    assert env["DATASET_ZIP"] == "s3://bucket/train-data"
    assert env["MODEL_TYPE"] == "sdxl"
    assert "DATASET" not in env
    assert "FILE_FORMAT" not in env
