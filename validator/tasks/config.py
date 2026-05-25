from dataclasses import dataclass
from typing import Callable

from core.models.payload_models import TrainerProxyRequest
from core.models.utility_models import TaskType
from core.service_paths import START_TRAINING_ENDPOINT
from core.service_paths import START_TRAINING_GRPO_ENDPOINT
from core.service_paths import START_TRAINING_IMAGE_ENDPOINT
from validator.shared.models import AnyTypeRawTask
from validator.shared.models import ChatRawTask
from validator.shared.models import DpoRawTask
from validator.shared.models import EnvRawTask
from validator.shared.models import GrpoRawTask
from validator.shared.models import ImageRawTask
from validator.shared.models import InstructTextRawTask
from validator.tasks.requests import get_fake_text_dataset_size
from validator.tasks.requests import get_total_image_dataset_size
from validator.tasks.requests import prepare_image_task_request
from validator.tasks.requests import prepare_text_task_request
from validator.tasks.requests import run_image_task_prep
from validator.tasks.requests import run_text_task_prep


DataSizeFunction = Callable[..., int]
TaskPrepFunction = Callable[..., object]
TaskRequestPrepareFunction = Callable[..., TrainerProxyRequest]


@dataclass(frozen=True)
class TaskConfig:
    task_type: TaskType
    data_size_function: DataSizeFunction
    task_prep_function: TaskPrepFunction
    task_request_prepare_function: TaskRequestPrepareFunction
    start_training_endpoint: str


TEXT_TASK_CONFIG = TaskConfig(
    task_type=TaskType.INSTRUCTTEXTTASK,
    data_size_function=get_fake_text_dataset_size,
    task_prep_function=run_text_task_prep,
    task_request_prepare_function=prepare_text_task_request,
    start_training_endpoint=START_TRAINING_ENDPOINT,
)

IMAGE_TASK_CONFIG = TaskConfig(
    task_type=TaskType.IMAGETASK,
    data_size_function=get_total_image_dataset_size,
    task_prep_function=run_image_task_prep,
    task_request_prepare_function=prepare_image_task_request,
    start_training_endpoint=START_TRAINING_IMAGE_ENDPOINT,
)

DPO_TASK_CONFIG = TaskConfig(
    task_type=TaskType.DPOTASK,
    data_size_function=get_fake_text_dataset_size,
    task_prep_function=run_text_task_prep,
    task_request_prepare_function=prepare_text_task_request,
    start_training_endpoint=START_TRAINING_ENDPOINT,
)

GRPO_TASK_CONFIG = TaskConfig(
    task_type=TaskType.GRPOTASK,
    data_size_function=get_fake_text_dataset_size,
    task_prep_function=run_text_task_prep,
    task_request_prepare_function=prepare_text_task_request,
    start_training_endpoint=START_TRAINING_GRPO_ENDPOINT,
)

ENV_TASK_CONFIG = TaskConfig(
    task_type=TaskType.ENVIRONMENTTASK,
    data_size_function=get_fake_text_dataset_size,
    task_prep_function=run_text_task_prep,
    task_request_prepare_function=prepare_text_task_request,
    start_training_endpoint=START_TRAINING_GRPO_ENDPOINT,
)

CHAT_TASK_CONFIG = TaskConfig(
    task_type=TaskType.CHATTASK,
    data_size_function=get_fake_text_dataset_size,
    task_prep_function=run_text_task_prep,
    task_request_prepare_function=prepare_text_task_request,
    start_training_endpoint=START_TRAINING_ENDPOINT,
)

TASK_CONFIG_BY_RAW_TASK_TYPE: tuple[tuple[type[AnyTypeRawTask], TaskConfig], ...] = (
    (InstructTextRawTask, TEXT_TASK_CONFIG),
    (ImageRawTask, IMAGE_TASK_CONFIG),
    (DpoRawTask, DPO_TASK_CONFIG),
    (GrpoRawTask, GRPO_TASK_CONFIG),
    (ChatRawTask, CHAT_TASK_CONFIG),
    (EnvRawTask, ENV_TASK_CONFIG),
)


def get_task_config(task: AnyTypeRawTask) -> TaskConfig:
    for task_model, config in TASK_CONFIG_BY_RAW_TASK_TYPE:
        if isinstance(task, task_model):
            return config
    raise ValueError(f"Unsupported task type: {type(task).__name__}")
