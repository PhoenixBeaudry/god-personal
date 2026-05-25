import json
from typing import Any

from asyncpg import Connection
from fastapi import Depends
from fastapi import HTTPException
from loguru import logger  # noqa

from core.models.utility_models import HIGHER_IS_BETTER_TASK_TYPES
from core.models.utility_models import ImageTextPair
from core.models.utility_models import RewardFunction
from core.models.utility_models import TaskType
from core.models.utility_models import normalize_task_type
from core.models.utility_models import scores_higher_is_better
from validator.db import constants as cst
from validator.db.sql import tasks as tasks_sql
from validator.db.sql.normalization import normalise_float
from validator.shared.config import Config
from validator.shared.dependencies import get_config
from validator.shared.models import AnyTypeTask
from validator.shared.models import AnyTypeTaskWithHotkeyDetails
from validator.shared.models import ChatTask
from validator.shared.models import ChatTaskWithHotkeyDetails
from validator.shared.models import DpoTask
from validator.shared.models import DpoTaskWithHotkeyDetails
from validator.shared.models import EnvTask
from validator.shared.models import EnvTaskWithHotkeyDetails
from validator.shared.models import GrpoTask
from validator.shared.models import GrpoTaskWithHotkeyDetails
from validator.shared.models import HotkeyDetails
from validator.shared.models import ImageTask
from validator.shared.models import ImageTaskWithHotkeyDetails
from validator.shared.models import InstructTextTask
from validator.shared.models import InstructTextTaskWithHotkeyDetails
from validator.tasks.details import hide_sensitive_data_till_finished


_TASK_MODEL_BY_TYPE = {
    TaskType.INSTRUCTTEXTTASK: InstructTextTask,
    TaskType.CHATTASK: ChatTask,
    TaskType.IMAGETASK: ImageTask,
    TaskType.DPOTASK: DpoTask,
    TaskType.GRPOTASK: GrpoTask,
    TaskType.ENVIRONMENTTASK: EnvTask,
}
_TASK_WITH_HOTKEY_DETAILS_MODEL_BY_TYPE = {
    TaskType.INSTRUCTTEXTTASK: InstructTextTaskWithHotkeyDetails,
    TaskType.CHATTASK: ChatTaskWithHotkeyDetails,
    TaskType.IMAGETASK: ImageTaskWithHotkeyDetails,
    TaskType.DPOTASK: DpoTaskWithHotkeyDetails,
    TaskType.GRPOTASK: GrpoTaskWithHotkeyDetails,
    TaskType.ENVIRONMENTTASK: EnvTaskWithHotkeyDetails,
}
_AUDIT_TASK_TABLE_BY_TYPE = {
    TaskType.INSTRUCTTEXTTASK: cst.INSTRUCT_TEXT_TASKS_TABLE,
    TaskType.CHATTASK: cst.CHAT_TASKS_TABLE,
    TaskType.IMAGETASK: cst.IMAGE_TASKS_TABLE,
    TaskType.DPOTASK: cst.DPO_TASKS_TABLE,
    TaskType.GRPOTASK: cst.GRPO_TASKS_TABLE,
    TaskType.ENVIRONMENTTASK: cst.ENV_TASKS_TABLE,
}
_HIGHER_IS_BETTER_SQL_VALUES = ", ".join(f"'{task_type.value}'" for task_type in HIGHER_IS_BETTER_TASK_TYPES)


def _task_model_for_type(task_type: TaskType | str):
    return _TASK_MODEL_BY_TYPE.get(normalize_task_type(task_type))


def _with_hotkey_details_model_for_type(task_type: TaskType | str):
    return _TASK_WITH_HOTKEY_DETAILS_MODEL_BY_TYPE.get(normalize_task_type(task_type))


def _task_fields_for_model(task_model, task_data: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in task_data.items() if key in task_model.model_fields}


def _parse_image_text_pairs(raw_pairs) -> list[ImageTextPair]:
    if not raw_pairs:
        return []
    if isinstance(raw_pairs, str):
        try:
            raw_pairs = json.loads(raw_pairs)
        except json.JSONDecodeError:
            return []
    if not isinstance(raw_pairs, list):
        return []

    image_text_pairs = []
    for pair in raw_pairs:
        try:
            pair_data = pair if isinstance(pair, dict) else json.loads(pair)
            image_text_pairs.append(ImageTextPair(**pair_data))
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    return image_text_pairs


def _parse_reward_functions(raw_reward_functions) -> list[RewardFunction]:
    if not raw_reward_functions:
        return []

    reward_functions = []
    for reward_function in raw_reward_functions:
        try:
            reward_data = reward_function if isinstance(reward_function, dict) else json.loads(reward_function)
            reward_functions.append(RewardFunction(**reward_data))
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    return reward_functions


def _build_task_for_audit(task_type: TaskType | str, task_data: dict[str, Any]) -> AnyTypeTask | None:
    task_model = _task_model_for_type(task_type)
    if task_model is None:
        return None
    return task_model(**_task_fields_for_model(task_model, task_data))


def _build_task_with_hotkey_details(
    task_type: TaskType | str,
    task_data: dict[str, Any],
    hotkey_details: list[HotkeyDetails],
) -> AnyTypeTaskWithHotkeyDetails | None:
    normalized_task_type = normalize_task_type(task_type)
    task = _build_task_for_audit(normalized_task_type, task_data)
    if task is None:
        return None
    task = hide_sensitive_data_till_finished(task)
    details_model = _with_hotkey_details_model_for_type(normalized_task_type)
    if details_model is None:
        return None
    return details_model(**task.model_dump(), hotkey_details=hotkey_details)


def _group_task_ids_by_type(tasks_by_id: dict[str, dict]) -> dict[TaskType, list[str]]:
    grouped_task_ids = {task_type: [] for task_type in _AUDIT_TASK_TABLE_BY_TYPE}
    for task_id, task_data in tasks_by_id.items():
        try:
            grouped_task_ids[normalize_task_type(task_data.get(cst.TASK_TYPE))].append(task_id)
        except ValueError:
            logger.warning(f"Unknown task type {task_data.get(cst.TASK_TYPE)} for task_id {task_id}")
    return grouped_task_ids


async def get_recent_tasks(
    hotkeys: list[str] | None = None,
    limit: int = 100,
    page: int = 1,
    config: Config = Depends(get_config),
    include_tournament_tasks=False,
) -> list[AnyTypeTask]:
    tournament_tasks_clause = (
        "" if include_tournament_tasks else f"WHERE {cst.TASK_ID} NOT IN (SELECT {cst.TASK_ID} FROM {cst.TOURNAMENT_TASKS_TABLE})"
    )
    tournament_tasks_clause_hotkeys = (
        "" if include_tournament_tasks else f"AND {cst.TASK_ID} NOT IN (SELECT {cst.TASK_ID} FROM {cst.TOURNAMENT_TASKS_TABLE})"
    )

    # Always exclude benchmark tasks from auditing
    benchmark_tasks_clause = f"""
        AND {cst.TASK_ID} NOT IN (
            SELECT {cst.TASK_ID} FROM {cst.BENCHMARK_ROOT_TASKS_TABLE}
            UNION
            SELECT {cst.COPY_TASK_ID} FROM {cst.BENCHMARK_TASK_COPIES_TABLE}
        )
    """

    async with await config.psql_db.connection() as connection:
        connection: Connection
        base_query = f"""
        WITH task_ids AS (
            {
            f'''
                SELECT DISTINCT s.{cst.TASK_ID}
                FROM {cst.SUBMISSIONS_TABLE} s
                WHERE s.{cst.HOTKEY} = ANY($1)
                {tournament_tasks_clause_hotkeys}
                {benchmark_tasks_clause}
                ORDER BY s.{cst.CREATED_ON} DESC
                LIMIT $2 OFFSET $3
                '''
            if hotkeys is not None
            else f'''
                SELECT {cst.TASK_ID}
                FROM {cst.TASKS_TABLE}
                {tournament_tasks_clause}
                {benchmark_tasks_clause}
                ORDER BY {cst.CREATED_AT} DESC
                LIMIT $1 OFFSET $2
                '''
        }
        ),
        image_pairs AS (
            SELECT
                itp.{cst.TASK_ID},
                ARRAY_AGG(json_build_object(
                    'image_url', itp.{cst.IMAGE_URL},
                    'text_url', itp.{cst.TEXT_URL}
                ) ORDER BY itp.{cst.ID}) as image_text_pairs
            FROM task_ids
            JOIN {cst.IMAGE_TEXT_PAIRS_TABLE} itp ON task_ids.{cst.TASK_ID} = itp.{cst.TASK_ID}
            GROUP BY itp.{cst.TASK_ID}
        ),
        reward_functions AS (
            SELECT
                gtf.{cst.TASK_ID},
                ARRAY_AGG(json_build_object(
                    'reward_func', rf.{cst.REWARD_FUNC},
                    'func_hash', rf.{cst.FUNC_HASH},
                    'is_generic', rf.{cst.IS_GENERIC},
                    'reward_weight', gtf.{cst.REWARD_WEIGHT}
                )::text) as reward_functions
            FROM task_ids
            JOIN {cst.GRPO_TASK_FUNCTIONS_TABLE} gtf ON task_ids.{cst.TASK_ID} = gtf.{cst.TASK_ID}
            JOIN {cst.REWARD_FUNCTIONS_TABLE} rf ON rf.{cst.REWARD_ID} = gtf.{cst.REWARD_ID}
            GROUP BY gtf.{cst.TASK_ID}
        )
        -- Main query joining all necessary tables
        SELECT
            t.*,
            itt.field_system as itt_field_system,
            itt.field_instruction,
            itt.field_input,
            itt.field_output,
            itt.format as itt_format,
            itt.no_input_format,
            itt.file_format as itt_file_format,
            it.model_type,
            ip.image_text_pairs,
            dt.field_prompt as dpo_field_prompt,
            dt.field_chosen,
            dt.field_rejected,
            dt.prompt_format,
            dt.chosen_format,
            dt.rejected_format,
            dt.file_format as dpo_file_format,
            gt.field_prompt as grpo_field_prompt,
            gt.file_format as grpo_file_format,
            rf.reward_functions,
            ct.chat_template,
            ct.chat_column,
            ct.chat_role_field,
            ct.chat_content_field,
            ct.chat_user_reference,
            ct.chat_assistant_reference,
            ct.file_format as chat_file_format,
            et.{cst.ENVIRONMENT_NAMES} as env_{cst.ENVIRONMENT_NAMES},
            et.{cst.EVAL_SEED} as env_{cst.EVAL_SEED}
        FROM task_ids
        JOIN {cst.TASKS_TABLE} t ON t.{cst.TASK_ID} = task_ids.{cst.TASK_ID}
        LEFT JOIN {cst.INSTRUCT_TEXT_TASKS_TABLE} itt ON t.{cst.TASK_ID} = itt.{cst.TASK_ID}
        LEFT JOIN {cst.IMAGE_TASKS_TABLE} it ON t.{cst.TASK_ID} = it.{cst.TASK_ID}
        LEFT JOIN image_pairs ip ON t.{cst.TASK_ID} = ip.{cst.TASK_ID}
        LEFT JOIN {cst.DPO_TASKS_TABLE} dt ON t.{cst.TASK_ID} = dt.{cst.TASK_ID}
        LEFT JOIN {cst.GRPO_TASKS_TABLE} gt ON t.{cst.TASK_ID} = gt.{cst.TASK_ID}
        LEFT JOIN {cst.CHAT_TASKS_TABLE} ct ON t.{cst.TASK_ID} = ct.{cst.TASK_ID}
        LEFT JOIN {cst.ENV_TASKS_TABLE} et ON t.{cst.TASK_ID} = et.{cst.TASK_ID}
        LEFT JOIN reward_functions rf ON t.{cst.TASK_ID} = rf.{cst.TASK_ID}
        """

        if hotkeys is not None:
            rows = await connection.fetch(base_query, hotkeys, limit, (page - 1) * limit)
        else:
            rows = await connection.fetch(base_query, limit, (page - 1) * limit)

        tasks_processed = []
        for row in rows:
            task_data = dict(row)
            task_type = task_data[cst.TASK_TYPE]
            try:
                normalized_task_type = normalize_task_type(task_type)
            except ValueError:
                logger.warning(f"Unknown task type: {task_type}, skipping task {task_data.get('task_id')}")
                continue

            if normalized_task_type == TaskType.INSTRUCTTEXTTASK:
                task_data["field_system"] = task_data.pop("itt_field_system")
                task_data["format"] = task_data.pop("itt_format")
                task_data["file_format"] = task_data.pop("itt_file_format")
            elif normalized_task_type == TaskType.IMAGETASK:
                task_data["image_text_pairs"] = _parse_image_text_pairs(task_data.pop("image_text_pairs", None))
            elif normalized_task_type == TaskType.DPOTASK:
                task_data["field_prompt"] = task_data.pop("dpo_field_prompt")
                task_data["file_format"] = task_data.pop("dpo_file_format")
            elif normalized_task_type == TaskType.ENVIRONMENTTASK:
                task_data[cst.ENVIRONMENT_NAMES] = task_data.pop(f"env_{cst.ENVIRONMENT_NAMES}", [])
                task_data[cst.EVAL_SEED] = task_data.pop(f"env_{cst.EVAL_SEED}", None)
            elif normalized_task_type == TaskType.GRPOTASK:
                task_data["field_prompt"] = task_data.pop("grpo_field_prompt")
                task_data["file_format"] = task_data.pop("grpo_file_format")
                task_data["reward_functions"] = _parse_reward_functions(task_data.get("reward_functions"))
            elif normalized_task_type == TaskType.CHATTASK:
                task_data["file_format"] = task_data.pop("chat_file_format")

            task = _build_task_for_audit(normalized_task_type, task_data)
            if task is None:
                logger.warning(f"Unknown task type: {task_type}, skipping task {task_data.get('task_id')}")
                continue

            task = hide_sensitive_data_till_finished(task)
            tasks_processed.append(task)

        return tasks_processed


async def _process_task_batch(
    connection, hotkey: str, task_ids: list[str], include_tournament_tasks=False
) -> list[AnyTypeTaskWithHotkeyDetails]:
    """
    Helper function to process a batch of task IDs.
    """
    tournament_tasks_clause = (
        "" if include_tournament_tasks else f"AND {cst.TASK_ID} NOT IN (SELECT {cst.TASK_ID} FROM {cst.TOURNAMENT_TASKS_TABLE})"
    )

    # Always exclude benchmark tasks from auditing
    benchmark_tasks_clause = f"""
        AND {cst.TASK_ID} NOT IN (
            SELECT {cst.TASK_ID} FROM {cst.BENCHMARK_ROOT_TASKS_TABLE}
            UNION
            SELECT {cst.COPY_TASK_ID} FROM {cst.BENCHMARK_TASK_COPIES_TABLE}
        )
    """

    tasks_with_details = []

    tasks_by_id = {}
    if task_ids:
        task_placeholders = ", ".join("$%d::uuid" % (i + 1) for i in range(len(task_ids)))
        tasks_query = f"""
            SELECT
                t.*
            FROM
                {cst.TASKS_TABLE} t
            WHERE
                t.{cst.TASK_ID} IN ({task_placeholders})
                {tournament_tasks_clause}
                {benchmark_tasks_clause}
        """

        tasks_rows = await connection.fetch(tasks_query, *task_ids)

        tasks_by_id = {str(row[cst.TASK_ID]): dict(row) for row in tasks_rows}
    else:
        return []

    # Step 3: Get all hotkey-specific details for these tasks in a single query
    details_rows = []
    if task_ids:
        details_placeholders = ", ".join("$%d::uuid" % (i + 2) for i in range(len(task_ids)))
        details_query = f"""
            SELECT
                t.{cst.TASK_ID}::text AS task_id,
                s.{cst.SUBMISSION_ID} AS submission_id,
                tn.{cst.QUALITY_SCORE} AS quality_score,
                tn.{cst.TEST_LOSS} AS test_loss,
                tn.{cst.SYNTH_LOSS} AS synth_loss,
                tn.{cst.SCORE_REASON} AS score_reason,
                RANK() OVER (
                    PARTITION BY t.{cst.TASK_ID}
                    ORDER BY CASE
                        WHEN t.{cst.TASK_TYPE} IN ({_HIGHER_IS_BETTER_SQL_VALUES}) THEN -tn.{cst.TEST_LOSS}
                        ELSE tn.{cst.TEST_LOSS}
                    END ASC NULLS LAST
                ) AS rank,
                s.{cst.REPO} AS repo,
                o.{cst.OFFER_RESPONSE} AS offer_response,
                t.{cst.TASK_TYPE} AS task_type
            FROM
                {cst.TASKS_TABLE} t
            LEFT JOIN
                {cst.TASK_NODES_TABLE} tn ON t.{cst.TASK_ID} = tn.{cst.TASK_ID} AND tn.{cst.HOTKEY} = $1
            LEFT JOIN
                {cst.SUBMISSIONS_TABLE} s ON t.{cst.TASK_ID} = s.{cst.TASK_ID} AND s.{cst.HOTKEY} = $1
            LEFT JOIN
                {cst.OFFER_RESPONSES_TABLE} o ON t.{cst.TASK_ID} = o.{cst.TASK_ID} AND o.{cst.HOTKEY} = $1
            WHERE
                t.{cst.TASK_ID} IN ({details_placeholders})
        """

        details_rows = await connection.fetch(details_query, hotkey, *task_ids)

    # Step 4: Group details by task_id
    details_by_task_id = {}
    for row in details_rows:
        task_id = row["task_id"]
        if task_id not in details_by_task_id:
            details_by_task_id[task_id] = []

        detail = dict(row)

        if detail.get("offer_response"):
            try:
                detail["offer_response"] = json.loads(detail["offer_response"])
            except (json.JSONDecodeError, TypeError):
                detail["offer_response"] = None

        for field in ["quality_score", "test_loss", "synth_loss"]:
            if detail.get(field) is not None:
                detail[field] = normalise_float(detail[field])

        details_by_task_id[task_id].append(detail)

    # Step 5: Get type-specific data for each task type
    grouped_task_ids = _group_task_ids_by_type(tasks_by_id)
    instruct_text_task_ids = grouped_task_ids[TaskType.INSTRUCTTEXTTASK]
    image_task_ids = grouped_task_ids[TaskType.IMAGETASK]
    dpo_task_ids = grouped_task_ids[TaskType.DPOTASK]
    grpo_task_ids = grouped_task_ids[TaskType.GRPOTASK]
    chat_task_ids = grouped_task_ids[TaskType.CHATTASK]
    env_task_ids = grouped_task_ids[TaskType.ENVIRONMENTTASK]

    # Get all InstructTextTask specific data in one query
    instruct_text_task_data = {}
    if instruct_text_task_ids:
        placeholders = ", ".join("$%d::uuid" % (i + 1) for i in range(len(instruct_text_task_ids)))
        query = f"""
            SELECT * FROM {cst.INSTRUCT_TEXT_TASKS_TABLE}
            WHERE {cst.TASK_ID} IN ({placeholders})
        """
        rows = await connection.fetch(query, *instruct_text_task_ids)
        instruct_text_task_data = {str(row[cst.TASK_ID]): dict(row) for row in rows}

    # Get all ChatTask specific data in one query
    chat_task_data = {}
    if chat_task_ids:
        placeholders = ", ".join("$%d::uuid" % (i + 1) for i in range(len(chat_task_ids)))
        query = f"""
            SELECT * FROM {cst.CHAT_TASKS_TABLE}
            WHERE {cst.TASK_ID} IN ({placeholders})
        """
        rows = await connection.fetch(query, *chat_task_ids)
        chat_task_data = {str(row[cst.TASK_ID]): dict(row) for row in rows}

    # Get all ImageTask specific data in one query
    image_task_data = {}
    if image_task_ids:
        placeholders = ", ".join("$%d::uuid" % (i + 1) for i in range(len(image_task_ids)))
        query = f"""
            SELECT * FROM {cst.IMAGE_TASKS_TABLE}
            WHERE {cst.TASK_ID} IN ({placeholders})
        """
        rows = await connection.fetch(query, *image_task_ids)
        image_task_data = {str(row[cst.TASK_ID]): dict(row) for row in rows}

    # Get all DpoTask specific data in one query
    dpo_task_data = {}
    if dpo_task_ids:
        placeholders = ", ".join("$%d::uuid" % (i + 1) for i in range(len(dpo_task_ids)))
        query = f"""
            SELECT * FROM {cst.DPO_TASKS_TABLE}
            WHERE {cst.TASK_ID} IN ({placeholders})
        """
        rows = await connection.fetch(query, *dpo_task_ids)
        dpo_task_data = {str(row[cst.TASK_ID]): dict(row) for row in rows}

    # Get all EnvTask specific data in one query
    env_task_data = {}
    if env_task_ids:
        placeholders = ", ".join("$%d::uuid" % (i + 1) for i in range(len(env_task_ids)))
        query = f"""
            SELECT {cst.TASK_ID}, {cst.ENVIRONMENT_NAMES}, {cst.EVAL_SEED} FROM {cst.ENV_TASKS_TABLE}
            WHERE {cst.TASK_ID} IN ({placeholders})
        """
        rows = await connection.fetch(query, *env_task_ids)
        env_task_data = {str(row[cst.TASK_ID]): dict(row) for row in rows}

    # Get all GrpoTask specific data in one query
    grpo_task_data = {}
    if grpo_task_ids:
        placeholders = ", ".join("$%d::uuid" % (i + 1) for i in range(len(grpo_task_ids)))
        query = f"""
            SELECT * FROM {cst.GRPO_TASKS_TABLE}
            WHERE {cst.TASK_ID} IN ({placeholders})
        """
        rows = await connection.fetch(query, *grpo_task_ids)
        grpo_task_data = {str(row[cst.TASK_ID]): dict(row) for row in rows}

        # Fetch reward functions for each GRPO task
        for task_id in grpo_task_ids:
            reward_functions_query = f"""
                SELECT rf.{cst.REWARD_FUNC}, rf.{cst.FUNC_HASH}, rf.{cst.IS_GENERIC}, gtf.{cst.REWARD_WEIGHT}
                FROM {cst.REWARD_FUNCTIONS_TABLE} rf
                JOIN {cst.GRPO_TASK_FUNCTIONS_TABLE} gtf ON rf.{cst.REWARD_ID} = gtf.{cst.REWARD_ID}
                WHERE gtf.{cst.TASK_ID} = $1
            """
            reward_rows = await connection.fetch(reward_functions_query, task_id)
            reward_functions = [
                RewardFunction(
                    reward_func=row[cst.REWARD_FUNC],
                    func_hash=row[cst.FUNC_HASH],
                    is_generic=row[cst.IS_GENERIC],
                    reward_weight=row[cst.REWARD_WEIGHT],
                )
                for row in reward_rows
            ]

            if task_id in grpo_task_data:
                grpo_task_data[task_id]["reward_functions"] = reward_functions

    task_specific_data_by_type = {
        TaskType.INSTRUCTTEXTTASK: instruct_text_task_data,
        TaskType.CHATTASK: chat_task_data,
        TaskType.IMAGETASK: image_task_data,
        TaskType.DPOTASK: dpo_task_data,
        TaskType.GRPOTASK: grpo_task_data,
        TaskType.ENVIRONMENTTASK: env_task_data,
    }

    # Step 6: Assemble final results
    for task_id in task_ids:
        if task_id not in tasks_by_id:
            continue

        task_data = tasks_by_id[task_id].copy()
        task_type = task_data.get(cst.TASK_TYPE)
        try:
            normalized_task_type = normalize_task_type(task_type)
        except ValueError:
            logger.warning(f"Unknown task type {task_type} for task_id {task_id}")
            continue

        specific_task_data = task_specific_data_by_type.get(normalized_task_type, {})
        if task_id in specific_task_data:
            task_data.update(specific_task_data[task_id])

        hotkey_details = []
        if task_id in details_by_task_id:
            for detail in details_by_task_id[task_id]:
                hotkey_details.append(
                    HotkeyDetails(
                        hotkey=hotkey,
                        submission_id=detail.get("submission_id"),
                        quality_score=detail.get("quality_score"),
                        test_loss=detail.get("test_loss"),
                        synth_loss=detail.get("synth_loss"),
                        score_reason=detail.get("score_reason"),
                        rank=detail.get("rank"),
                        repo=detail.get("repo"),
                        offer_response=detail.get("offer_response"),
                    )
                )

        task_with_details = _build_task_with_hotkey_details(normalized_task_type, task_data, hotkey_details)
        if task_with_details:
            tasks_with_details.append(task_with_details)

    return tasks_with_details


async def get_recent_tasks_for_hotkey(
    hotkey: str, limit: int = 100, page: int = 1, config: Config = Depends(get_config), include_tournament_tasks=False
) -> list[AnyTypeTaskWithHotkeyDetails]:
    """
    Retrieves recent tasks for a specific hotkey with detailed information.
    """
    MAX_BATCH_SIZE = 500
    tournament_tasks_clause = (
        "" if include_tournament_tasks else f"AND {cst.TASK_ID} NOT IN (SELECT {cst.TASK_ID} FROM {cst.TOURNAMENT_TASKS_TABLE})"
    )

    # Always exclude benchmark tasks from auditing
    benchmark_tasks_clause = f"""
        AND {cst.TASK_ID} NOT IN (
            SELECT {cst.TASK_ID} FROM {cst.BENCHMARK_ROOT_TASKS_TABLE}
            UNION
            SELECT {cst.COPY_TASK_ID} FROM {cst.BENCHMARK_TASK_COPIES_TABLE}
        )
    """

    async with await config.psql_db.connection() as connection:
        task_ids_query = f"""
            SELECT
                s.{cst.TASK_ID}::text AS task_id
            FROM
                {cst.SUBMISSIONS_TABLE} s
            WHERE
                s.{cst.HOTKEY} = $1
                {tournament_tasks_clause}
                {benchmark_tasks_clause}
            ORDER BY
                s.{cst.CREATED_ON} DESC
            LIMIT $2 OFFSET $3
        """
        offset = (page - 1) * limit
        task_ids_rows = await connection.fetch(task_ids_query, hotkey, limit, offset)

        if not task_ids_rows:
            return []

        task_ids = [row["task_id"] for row in task_ids_rows]

        if len(task_ids) > MAX_BATCH_SIZE:
            all_results = []
            for i in range(0, len(task_ids), MAX_BATCH_SIZE):
                batch_ids = task_ids[i : i + MAX_BATCH_SIZE]
                batch_results = await _process_task_batch(connection, hotkey, batch_ids, include_tournament_tasks)
                all_results.extend(batch_results)
            return all_results

        return await _process_task_batch(connection, hotkey, task_ids, include_tournament_tasks)


async def get_task_with_hotkey_details(task_id: str, config: Config = Depends(get_config)) -> AnyTypeTaskWithHotkeyDetails:
    # First get all the task details like normal
    task_raw = await tasks_sql.get_task_by_id(task_id, config.psql_db)
    if task_raw is None:
        raise HTTPException(status_code=404, detail="Task not found")

    logger.info("Got a task!!")

    task = hide_sensitive_data_till_finished(task_raw)

    higher_is_better = scores_higher_is_better(task_raw.task_type)
    rank_order = f"tn.{cst.TEST_LOSS} DESC NULLS LAST" if higher_is_better else f"tn.{cst.TEST_LOSS} ASC NULLS LAST"

    query = f"""
        SELECT
            tn.{cst.HOTKEY},
            s.{cst.SUBMISSION_ID},
            tn.{cst.QUALITY_SCORE},
            tn.{cst.TEST_LOSS},
            tn.{cst.SYNTH_LOSS},
            tn.{cst.SCORE_REASON},
            RANK() OVER (ORDER BY {rank_order}) as rank,
            s.{cst.REPO},
            o.{cst.OFFER_RESPONSE}
        FROM {cst.TASK_NODES_TABLE} tn
        LEFT JOIN {cst.SUBMISSIONS_TABLE} s
            ON tn.{cst.TASK_ID} = s.{cst.TASK_ID}
            AND tn.{cst.HOTKEY} = s.{cst.HOTKEY}
        LEFT JOIN {cst.OFFER_RESPONSES_TABLE} o
            ON tn.{cst.TASK_ID} = o.{cst.TASK_ID}
            AND tn.{cst.HOTKEY} = o.{cst.HOTKEY}
        WHERE tn.{cst.TASK_ID} = $1
    """
    async with await config.psql_db.connection() as connection:
        connection: Connection
        results = await connection.fetch(query, task_id)

    logger.info(f"Got {len(results)} results for task {task_id}")

    hotkey_details = []
    for result in results:
        result_dict = dict(result)
        if result_dict[cst.OFFER_RESPONSE] is not None:
            result_dict[cst.OFFER_RESPONSE] = json.loads(result_dict[cst.OFFER_RESPONSE])

        float_fields = [cst.QUALITY_SCORE, cst.TEST_LOSS, cst.SYNTH_LOSS]
        for field in float_fields:
            result_dict[field] = normalise_float(result_dict[field])

        hotkey_details.append(HotkeyDetails(**result_dict))

    details_model = _with_hotkey_details_model_for_type(task.task_type)
    if details_model is None:
        raise HTTPException(status_code=500, detail=f"Unsupported task type: {task.task_type}")
    return details_model(**task.model_dump(), hotkey_details=hotkey_details)


async def store_latest_scores_url(url: str, config: Config = Depends(get_config)) -> None:
    async with await config.psql_db.connection() as connection:
        connection: Connection

        # First expire all existing URLs
        expire_query = f"""
            UPDATE {cst.LATEST_SCORES_URL_TABLE}
            SET expired_at = NOW()
            WHERE expired_at IS NULL
        """
        await connection.execute(expire_query)

        # Then insert the new URL
        insert_query = f"""
            INSERT INTO {cst.LATEST_SCORES_URL_TABLE} (url)
            VALUES ($1)
        """
        await connection.execute(insert_query, url)


async def get_latest_scores_url(config: Config = Depends(get_config)) -> str | None:
    async with await config.psql_db.connection() as connection:
        connection: Connection

        query = f"""
            SELECT url FROM {cst.LATEST_SCORES_URL_TABLE} WHERE expired_at IS NULL ORDER BY created_at DESC LIMIT 1
        """
        return await connection.fetchval(query)
