from collections.abc import Awaitable
from collections.abc import Callable
from uuid import UUID

from asyncpg.connection import Connection

import validator.db.constants as cst
from core.models.utility_models import TaskType
from core.models.utility_models import normalize_task_type
from validator.db.database import PSQLDB
from validator.shared.models import AnyTypeRawTask
from validator.shared.models import AnyTypeTask
from validator.shared.models import ChatRawTask
from validator.shared.models import ChatTask
from validator.shared.models import DpoRawTask
from validator.shared.models import DpoTask
from validator.shared.models import EnvRawTask
from validator.shared.models import EnvTask
from validator.shared.models import GrpoRawTask
from validator.shared.models import GrpoTask
from validator.shared.models import ImageRawTask
from validator.shared.models import ImageTask
from validator.shared.models import InstructTextRawTask
from validator.shared.models import InstructTextTask
from validator.shared.models import RewardFunction


ImageTextPairsLoader = Callable[[UUID, PSQLDB, Connection | None], Awaitable[list]]
RewardFunctionsLoader = Callable[[UUID, PSQLDB, Connection | None], Awaitable[list[RewardFunction]]]

RAW_TASK_MODEL_BY_TYPE = {
    TaskType.INSTRUCTTEXTTASK: InstructTextRawTask,
    TaskType.CHATTASK: ChatRawTask,
    TaskType.IMAGETASK: ImageRawTask,
    TaskType.DPOTASK: DpoRawTask,
    TaskType.GRPOTASK: GrpoRawTask,
    TaskType.ENVIRONMENTTASK: EnvRawTask,
}
TASK_MODEL_BY_TYPE = {
    TaskType.INSTRUCTTEXTTASK: InstructTextTask,
    TaskType.CHATTASK: ChatTask,
    TaskType.IMAGETASK: ImageTask,
    TaskType.DPOTASK: DpoTask,
    TaskType.GRPOTASK: GrpoTask,
    TaskType.ENVIRONMENTTASK: EnvTask,
}
TASK_SPECIFIC_UPDATE_TABLE_BY_TYPE = {
    TaskType.INSTRUCTTEXTTASK: cst.INSTRUCT_TEXT_TASKS_TABLE,
    TaskType.CHATTASK: cst.CHAT_TASKS_TABLE,
    TaskType.DPOTASK: cst.DPO_TASKS_TABLE,
    TaskType.GRPOTASK: cst.GRPO_TASKS_TABLE,
    TaskType.ENVIRONMENTTASK: cst.ENV_TASKS_TABLE,
}


async def build_task_from_data(
    task_type: TaskType | str,
    task_data: dict,
    task_id: UUID,
    psql_db: PSQLDB,
    connection: Connection | None = None,
    *,
    public: bool = False,
    image_text_pairs_loader: ImageTextPairsLoader | None = None,
    reward_functions_loader: RewardFunctionsLoader | None = None,
) -> AnyTypeTask | AnyTypeRawTask | None:
    normalized_task_type = normalize_task_type(task_type)
    model_by_type = TASK_MODEL_BY_TYPE if public else RAW_TASK_MODEL_BY_TYPE
    task_model = model_by_type.get(normalized_task_type)
    if task_model is None:
        return None

    full_task_data = dict(task_data)
    if normalized_task_type == TaskType.IMAGETASK:
        if image_text_pairs_loader is None:
            raise ValueError("image_text_pairs_loader is required to build image tasks")
        image_text_pairs = await image_text_pairs_loader(task_id, psql_db, connection)
        return task_model(**full_task_data, image_text_pairs=image_text_pairs)
    if normalized_task_type == TaskType.GRPOTASK:
        if reward_functions_loader is None:
            raise ValueError("reward_functions_loader is required to build GRPO tasks")
        reward_functions = await reward_functions_loader(task_id, psql_db, connection)
        return task_model(**full_task_data, reward_functions=reward_functions)
    return task_model(**full_task_data)


async def get_table_fields(table_name: str, connection: Connection) -> set[str]:
    """Get all column names for a given table."""
    query = """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = $1
    """
    rows = await connection.fetch(query, table_name)
    return {row["column_name"] for row in rows}


async def get_specific_task_updates(table_name: str, updates: dict, connection: Connection) -> dict:
    table_fields = await get_table_fields(table_name, connection)
    specific_fields = [field for field in table_fields if field != cst.TASK_ID]
    return {key: value for key, value in updates.items() if key in specific_fields}


async def update_task_specific_fields(
    connection: Connection,
    task_id: UUID,
    table_name: str,
    updates: dict,
) -> None:
    specific_updates = await get_specific_task_updates(table_name, updates, connection)
    if not specific_updates:
        return

    specific_clause = ", ".join([f"{column} = ${i + 2}" for i, column in enumerate(specific_updates.keys())])
    specific_values = list(specific_updates.values())
    query = f"""
        UPDATE {table_name}
        SET {specific_clause}
        WHERE {cst.TASK_ID} = $1
    """
    await connection.execute(query, task_id, *specific_values)
