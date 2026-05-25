from asyncpg.connection import Connection

import validator.db.constants as cst
from validator.shared.models import AnyTypeRawTask
from validator.shared.models import ChatRawTask
from validator.shared.models import DpoRawTask
from validator.shared.models import EnvRawTask
from validator.shared.models import GrpoRawTask
from validator.shared.models import ImageRawTask
from validator.shared.models import InstructTextRawTask


async def insert_base_task(connection: Connection, task: AnyTypeRawTask) -> dict:
    """Insert the base task record and return it."""
    query_tasks = f"""
        INSERT INTO {cst.TASKS_TABLE}
        ({cst.ACCOUNT_ID},
        {cst.MODEL_ID},
        {cst.DS},
        {cst.STATUS},
        {cst.IS_ORGANIC},
        {cst.HOURS_TO_COMPLETE},
        {cst.TEST_DATA},
        {cst.TRAINING_DATA},
        {cst.CREATED_AT},
        {cst.TASK_TYPE},
        {cst.BACKEND},
        {cst.RESULT_MODEL_NAME},
        {cst.TRAINING_REPO_BACKUP},
        {cst.STARTED_AT},
        {cst.TERMINATION_AT},
        {cst.YARN_FACTOR},
        {cst.AUGMENTATION_CONFIG},
        {cst.AUGMENTED_MODEL_ID},
        {cst.BASELINE_STATS},
        {cst.TRAINING_START_POINT})
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20)
        RETURNING *
    """
    return await connection.fetchrow(
        query_tasks,
        task.account_id,
        task.model_id,
        task.ds,
        task.status,
        task.is_organic,
        task.hours_to_complete,
        task.test_data,
        task.training_data,
        task.created_at,
        task.task_type.value,
        task.backend.value if task.backend else None,
        task.result_model_name,
        task.training_repo_backup,
        task.started_at,
        task.termination_at,
        task.yarn_factor,
        task.augmentation_config.model_dump() if task.augmentation_config else None,
        task.augmented_model_id,
        task.baseline_stats.model_dump() if task.baseline_stats else None,
        task.training_start_point.value,
    )


async def insert_task_specific_data(connection: Connection, task: AnyTypeRawTask, task_record: dict) -> None:
    """Insert task type specific data based on the task type."""
    if isinstance(task, InstructTextRawTask):
        await _insert_instruct_text_task(connection, task, task_record)
    elif isinstance(task, ImageRawTask):
        await _insert_image_task(connection, task, task_record)
    elif isinstance(task, DpoRawTask):
        await _insert_dpo_task(connection, task, task_record)
    elif isinstance(task, GrpoRawTask):
        await _insert_grpo_task(connection, task, task_record)
    elif isinstance(task, EnvRawTask):
        await _insert_env_task(connection, task, task_record)
    elif isinstance(task, ChatRawTask):
        await _insert_chat_task(connection, task, task_record)


async def _insert_instruct_text_task(connection: Connection, task: InstructTextRawTask, task_record: dict) -> None:
    query = f"""
        INSERT INTO {cst.INSTRUCT_TEXT_TASKS_TABLE}
        ({cst.TASK_ID}, {cst.FIELD_SYSTEM}, {cst.FIELD_INSTRUCTION},
        {cst.FIELD_INPUT}, {cst.FIELD_OUTPUT}, {cst.FORMAT},
        {cst.NO_INPUT_FORMAT}, {cst.FILE_FORMAT})
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
    """
    await connection.execute(
        query,
        task_record[cst.TASK_ID],
        task.field_system,
        task.field_instruction,
        task.field_input,
        task.field_output,
        task.format,
        task.no_input_format,
        task.file_format,
    )


async def _insert_chat_task(connection: Connection, task: ChatRawTask, task_record: dict) -> None:
    query = f"""
        INSERT INTO {cst.CHAT_TASKS_TABLE}
        ({cst.TASK_ID}, {cst.CHAT_TEMPLATE}, {cst.CHAT_COLUMN},
        {cst.CHAT_ROLE_FIELD}, {cst.CHAT_CONTENT_FIELD}, {cst.CHAT_USER_REFERENCE},
        {cst.CHAT_ASSISTANT_REFERENCE}, {cst.FILE_FORMAT})
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
    """
    await connection.execute(
        query,
        task_record[cst.TASK_ID],
        task.chat_template,
        task.chat_column,
        task.chat_role_field,
        task.chat_content_field,
        task.chat_user_reference,
        task.chat_assistant_reference,
        task.file_format,
    )


async def _insert_image_task(connection: Connection, task: ImageRawTask, task_record: dict) -> None:
    query = f"""
        INSERT INTO {cst.IMAGE_TASKS_TABLE}
        ({cst.TASK_ID}, {cst.MODEL_TYPE})
        VALUES ($1, $2)
    """
    await connection.execute(query, task_record[cst.TASK_ID], task.model_type.value)

    if task.image_text_pairs:
        query_pairs = f"""
            INSERT INTO {cst.IMAGE_TEXT_PAIRS_TABLE}
            ({cst.TASK_ID}, {cst.IMAGE_URL}, {cst.TEXT_URL})
            VALUES ($1, $2, $3)
        """
        for pair in task.image_text_pairs:
            await connection.execute(query_pairs, task_record[cst.TASK_ID], pair.image_url, pair.text_url)


async def _insert_dpo_task(connection: Connection, task: DpoRawTask, task_record: dict) -> None:
    query = f"""
        INSERT INTO {cst.DPO_TASKS_TABLE}
        ({cst.TASK_ID}, {cst.FIELD_PROMPT}, {cst.FIELD_SYSTEM}, {cst.FIELD_CHOSEN}, {cst.FIELD_REJECTED},
        {cst.PROMPT_FORMAT}, {cst.CHOSEN_FORMAT}, {cst.REJECTED_FORMAT}, {cst.FILE_FORMAT})
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
    """
    await connection.execute(
        query,
        task_record[cst.TASK_ID],
        task.field_prompt,
        task.field_system,
        task.field_chosen,
        task.field_rejected,
        task.prompt_format,
        task.chosen_format,
        task.rejected_format,
        task.file_format,
    )


async def _insert_grpo_task(connection: Connection, task: GrpoRawTask, task_record: dict) -> None:
    query_grpo = f"""
        INSERT INTO {cst.GRPO_TASKS_TABLE}
        ({cst.TASK_ID}, {cst.FIELD_PROMPT}, {cst.FILE_FORMAT}, {cst.FIELD_EXTRA_COLUMN})
        VALUES ($1, $2, $3, $4)
    """
    await connection.execute(
        query_grpo,
        task_record[cst.TASK_ID],
        task.field_prompt,
        task.file_format,
        task.extra_column,
    )

    for reward_function in task.reward_functions:
        query_reward_functions = f"""
            WITH ins AS (
                INSERT INTO {cst.REWARD_FUNCTIONS_TABLE}
                ({cst.REWARD_FUNC}, {cst.FUNC_HASH}, {cst.IS_GENERIC})
                VALUES ($1, $2, $3)
                ON CONFLICT ({cst.FUNC_HASH}) DO NOTHING
                RETURNING {cst.REWARD_ID}
            )
            SELECT {cst.REWARD_ID} FROM ins
            UNION ALL
            SELECT {cst.REWARD_ID} FROM {cst.REWARD_FUNCTIONS_TABLE} WHERE {cst.FUNC_HASH} = $2
            LIMIT 1
        """
        reward_id = await connection.fetchval(
            query_reward_functions,
            reward_function.reward_func,
            reward_function.func_hash,
            reward_function.is_generic,
        )

        query_grpo_task_functions = f"""
            INSERT INTO {cst.GRPO_TASK_FUNCTIONS_TABLE}
            ({cst.TASK_ID}, {cst.REWARD_ID}, {cst.REWARD_WEIGHT})
            VALUES ($1, $2, $3)
        """
        await connection.execute(
            query_grpo_task_functions,
            task_record[cst.TASK_ID],
            reward_id,
            reward_function.reward_weight,
        )


async def _insert_env_task(connection: Connection, task: EnvRawTask, task_record: dict) -> None:
    query_env = f"""
        INSERT INTO {cst.ENV_TASKS_TABLE}
        ({cst.TASK_ID}, {cst.ENVIRONMENT_NAMES}, {cst.ENVIRONMENT_WEIGHTS}, {cst.EVAL_SEED})
        VALUES ($1, $2, $3, $4)
    """
    env_names = [e.value for e in task.environment_names] if task.environment_names else []
    env_weights = [w.model_dump() for w in task.environment_weights] if task.environment_weights else []
    await connection.execute(
        query_env,
        task_record[cst.TASK_ID],
        env_names,
        env_weights,
        task.eval_seed,
    )
