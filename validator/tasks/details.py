from core.models.payload_models import AnyTypeTaskDetails
from core.models.payload_models import ChatTaskDetails
from core.models.payload_models import DpoTaskDetails
from core.models.payload_models import EnvironmentTaskDetails
from core.models.payload_models import GrpoTaskDetails
from core.models.payload_models import ImageTaskDetails
from core.models.payload_models import InstructTextTaskDetails
from core.models.utility_models import TaskStatus
from core.models.utility_models import TaskType
from core.models.utility_models import is_environment_task
from core.models.utility_models import is_image_task
from validator.shared.models import AnyTypeTask
from validator.shared.models import ImageTextPair


TASK_DETAIL_MODEL_BY_TYPE: dict[TaskType, type[AnyTypeTaskDetails]] = {
    TaskType.INSTRUCTTEXTTASK: InstructTextTaskDetails,
    TaskType.CHATTASK: ChatTaskDetails,
    TaskType.IMAGETASK: ImageTaskDetails,
    TaskType.DPOTASK: DpoTaskDetails,
    TaskType.GRPOTASK: GrpoTaskDetails,
    TaskType.ENVIRONMENTTASK: EnvironmentTaskDetails,
}

TASK_DETAIL_EXTRA_FIELDS: dict[TaskType, tuple[str, ...]] = {
    TaskType.INSTRUCTTEXTTASK: (
        "field_input",
        "field_system",
        "field_instruction",
        "field_output",
        "format",
        "no_input_format",
        "system_format",
    ),
    TaskType.CHATTASK: (
        "chat_template",
        "chat_column",
        "chat_role_field",
        "chat_content_field",
        "chat_user_reference",
        "chat_assistant_reference",
    ),
    TaskType.IMAGETASK: ("image_text_pairs", "model_type"),
    TaskType.DPOTASK: (
        "field_prompt",
        "field_system",
        "field_chosen",
        "field_rejected",
        "prompt_format",
        "chosen_format",
        "rejected_format",
    ),
    TaskType.GRPOTASK: ("field_prompt", "reward_functions"),
    TaskType.ENVIRONMENTTASK: ("environment_names",),
}


def convert_task_to_task_details(task: AnyTypeTask) -> AnyTypeTaskDetails:
    detail_model = TASK_DETAIL_MODEL_BY_TYPE[task.task_type]
    fields = {
        "id": task.task_id,
        "account_id": task.account_id,
        "status": task.status,
        "base_model_repository": task.model_id,
        "ds_repo": task.ds,
        "created_at": task.created_at,
        "started_at": task.started_at,
        "finished_at": task.termination_at,
        "hours_to_complete": task.hours_to_complete,
        "trained_model_repository": task.trained_model_repository,
        "task_type": task.task_type,
        "result_model_name": task.result_model_name,
    }
    fields.update({field: getattr(task, field) for field in TASK_DETAIL_EXTRA_FIELDS[task.task_type]})
    return detail_model(**fields)


def is_task_in_flight(task: AnyTypeTask) -> bool:
    return task.status not in [
        TaskStatus.SUCCESS,
        TaskStatus.FAILURE,
        TaskStatus.FAILURE_FINDING_NODES,
        TaskStatus.PREP_TASK_FAILURE,
    ]


def hide_sensitive_data_till_finished(task: AnyTypeTask) -> AnyTypeTask:
    if is_task_in_flight(task):
        if is_image_task(task.task_type):
            task.image_text_pairs = [ImageTextPair(image_url="hidden", text_url="hidden")]
        if is_environment_task(task.task_type):
            task.eval_seed = None
        task.test_data = None

        task.training_data = None
        task.ds = "Hidden"
    return task
