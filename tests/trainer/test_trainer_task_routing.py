import pytest

from core.models.payload_models import TrainerProxyRequest
from core.models.payload_models import TrainRequestImage
from core.models.payload_models import TrainRequestText
from core.models.utility_models import TRAINER_TASK_TYPES
from core.models.utility_models import ChatTemplateDatasetType
from core.models.utility_models import DpoDatasetType
from core.models.utility_models import EnvironmentDatasetType
from core.models.utility_models import FileFormat
from core.models.utility_models import GrpoDatasetType
from core.models.utility_models import ImageModelType
from core.models.utility_models import InstructTextDatasetType
from core.models.utility_models import TaskType
from trainer import constants as cst
from trainer.containers.downloader import task_type_choices
from trainer.runtime import get_dockerfile_path
from trainer.runtime import get_task_type


def _proxy(training_data):
    return TrainerProxyRequest(
        training_data=training_data,
        github_repo="https://github.com/example/submission",
        gpu_ids=[0],
        hotkey="hotkey",
    )


def _text_request(dataset_type):
    return TrainRequestText(
        model="base-model",
        task_id="task-id",
        hours_to_complete=1,
        expected_repo_name="expected/repo",
        dataset="s3://dataset",
        dataset_type=dataset_type,
        file_format=FileFormat.S3,
    )


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
def test_get_task_type_routes_text_dataset_shapes(dataset_type, expected_task_type):
    assert get_task_type(_proxy(_text_request(dataset_type))) == expected_task_type


def test_get_task_type_routes_image_request():
    request = TrainRequestImage(
        model="base-image-model",
        task_id="task-id",
        hours_to_complete=1,
        expected_repo_name="expected/repo",
        dataset_zip="s3://dataset.zip",
        model_type=ImageModelType.SDXL,
    )

    assert get_task_type(_proxy(request)) == TaskType.IMAGETASK


@pytest.mark.parametrize(
    ("model_type", "expected_dockerfile"),
    [
        (ImageModelType.SDXL, cst.DEFAULT_IMAGE_DOCKERFILE_PATH),
        (ImageModelType.FLUX, cst.DEFAULT_IMAGE_DOCKERFILE_PATH),
        (ImageModelType.Z_IMAGE, cst.DEFAULT_IMAGE_TOOLKIT_DOCKERFILE_PATH),
        (ImageModelType.QWEN_IMAGE, cst.DEFAULT_IMAGE_TOOLKIT_DOCKERFILE_PATH),
    ],
)
def test_get_dockerfile_path_routes_image_model_types(model_type, expected_dockerfile):
    training_data = TrainRequestImage(
        model="base-image-model",
        task_id="task-id",
        hours_to_complete=1,
        dataset_zip="s3://dataset.zip",
        model_type=model_type,
    )

    assert get_dockerfile_path(TaskType.IMAGETASK, training_data, "/repo") == f"/repo/{expected_dockerfile}"


def test_get_dockerfile_path_routes_text_trainers_to_text_dockerfile():
    training_data = _text_request(InstructTextDatasetType())

    assert get_dockerfile_path(TaskType.ENVIRONMENTTASK, training_data, "/repo") == f"/repo/{cst.DEFAULT_TEXT_DOCKERFILE_PATH}"


def test_downloader_choices_follow_supported_trainer_task_types():
    assert task_type_choices() == [task_type.value for task_type in TRAINER_TASK_TYPES]
