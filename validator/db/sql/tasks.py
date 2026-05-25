from datetime import datetime
from typing import Literal
from uuid import UUID

from asyncpg.connection import Connection
from fiber.chain.models import Node

import validator.db.constants as cst
from core.constants import NETUID
from core.logging import get_logger
from core.models.network_models import DetailedNetworkStats
from core.models.network_models import NetworkStats
from core.models.utility_models import ImageTextPair
from core.models.utility_models import TaskStatus
from core.models.utility_models import TaskType
from core.models.utility_models import normalize_task_type
from validator.db.database import PSQLDB
from validator.db.sql.task_inserts import insert_base_task
from validator.db.sql.task_inserts import insert_task_specific_data
from validator.db.sql.task_rows import TASK_SPECIFIC_UPDATE_TABLE_BY_TYPE
from validator.db.sql.task_rows import build_task_from_data
from validator.db.sql.task_rows import get_table_fields
from validator.db.sql.task_rows import update_task_specific_fields
from validator.infrastructure.minio_client import async_minio_client
from validator.shared.models import AnyTypeRawTask
from validator.shared.models import AnyTypeTask
from validator.shared.models import InstructTextTask
from validator.shared.models import RewardFunction


logger = get_logger(__name__)


async def add_task(task: AnyTypeRawTask, psql_db: PSQLDB) -> AnyTypeRawTask:
    """Add a new task"""
    async with await psql_db.connection() as connection:
        async with connection.transaction():
            task_record = await insert_base_task(connection, task)
            await insert_task_specific_data(connection, task, task_record)
            task.task_id = task_record[cst.TASK_ID]
            return task


async def get_nodes_assigned_to_task(task_id: str, psql_db: PSQLDB) -> list[Node]:
    """Get all nodes assigned to a task for the current NETUID"""
    async with await psql_db.connection() as connection:
        connection: Connection
        rows = await connection.fetch(
            f"""
            SELECT nodes.* FROM {cst.NODES_TABLE} nodes
            JOIN {cst.TASK_NODES_TABLE} ON nodes.hotkey = task_nodes.hotkey
            WHERE task_nodes.task_id = $1
            AND nodes.netuid = $2
            AND task_nodes.netuid = $2
            """,
            task_id,
            NETUID,
        )
        # Filter out 'trust' field as it was removed from Node model
        return [Node(**{k: v for k, v in dict(row).items() if k != cst.TRUST}) for row in rows]


async def get_tasks_with_status(
    status: TaskStatus,
    psql_db: PSQLDB,
    include_not_ready_tasks: bool = False,
    tournament_filter: Literal["all", "only", "exclude"] = "all",
    benchmark_filter: Literal["include", "exclude", "only"] = "exclude",
    backend: str | None = None,
) -> list[AnyTypeRawTask]:
    if (benchmark_filter == "include" or benchmark_filter == "only") and tournament_filter == "only":
        raise ValueError(f"Cannot include benchmark tasks and only tournament tasks: {benchmark_filter} and {tournament_filter}")

    delay_timestamp_clause = (
        "" if include_not_ready_tasks else f"AND ({cst.NEXT_DELAY_AT} IS NULL OR {cst.NEXT_DELAY_AT} <= NOW())"
    )
    if tournament_filter == "exclude":
        tournament_tasks_clause = f"AND {cst.TASK_ID} NOT IN (SELECT {cst.TASK_ID}::uuid FROM {cst.TOURNAMENT_TASKS_TABLE})"
    elif tournament_filter == "only":
        tournament_tasks_clause = f"AND {cst.TASK_ID} IN (SELECT {cst.TASK_ID}::uuid FROM {cst.TOURNAMENT_TASKS_TABLE})"
    elif tournament_filter == "all":
        tournament_tasks_clause = ""

    if benchmark_filter == "include":
        benchmark_tasks_clause = ""
    elif benchmark_filter == "only":
        benchmark_tasks_clause = f"""
            AND {cst.TASK_ID} IN (
                SELECT {cst.TASK_ID} FROM {cst.BENCHMARK_ROOT_TASKS_TABLE}
                UNION
                SELECT {cst.COPY_TASK_ID} FROM {cst.BENCHMARK_TASK_COPIES_TABLE}
            )
        """
    else:
        benchmark_tasks_clause = f"""
            AND {cst.TASK_ID} NOT IN (
                SELECT {cst.TASK_ID} FROM {cst.BENCHMARK_ROOT_TASKS_TABLE}
                UNION
                SELECT {cst.COPY_TASK_ID} FROM {cst.BENCHMARK_TASK_COPIES_TABLE}
            )
        """

    if backend:
            backend_clause = f"AND ({cst.BACKEND} = $2 OR {cst.BACKEND} IS NULL)"
            query_params = [status.value, backend]
    else:
        backend_clause = ""
        query_params = [status.value]
        

    async with await psql_db.connection() as connection:
        connection: Connection
        
        base_query = f"""
            SELECT * FROM {cst.TASKS_TABLE}
            WHERE {cst.STATUS} = $1
            {delay_timestamp_clause}
            {tournament_tasks_clause}
            {benchmark_tasks_clause}
            {backend_clause}
        """
        base_rows = await connection.fetch(base_query, *query_params)

        tasks = []
        for row in base_rows:
            task_type = row[cst.TASK_TYPE]
            if task_type == TaskType.INSTRUCTTEXTTASK.value:
                specific_query = f"""
                    SELECT t.*, tt.field_system,
                           tt.field_instruction, tt.field_input, tt.field_output,
                           tt.format, tt.no_input_format, tt.file_format
                    FROM {cst.TASKS_TABLE} t
                    LEFT JOIN {cst.INSTRUCT_TEXT_TASKS_TABLE} tt ON t.{cst.TASK_ID} = tt.{cst.TASK_ID}
                    WHERE t.{cst.TASK_ID} = $1
                """
            elif task_type == TaskType.IMAGETASK.value:
                specific_query = f"""
                    SELECT t.*, it.model_type
                    FROM {cst.TASKS_TABLE} t
                    LEFT JOIN {cst.IMAGE_TASKS_TABLE} it ON t.{cst.TASK_ID} = it.{cst.TASK_ID}
                    WHERE t.{cst.TASK_ID} = $1
                """
            elif task_type == TaskType.DPOTASK.value:
                specific_query = f"""
                    SELECT t.*, dt.field_prompt, dt.field_system, dt.field_chosen, dt.field_rejected,
                           dt.prompt_format, dt.chosen_format, dt.rejected_format, dt.file_format
                    FROM {cst.TASKS_TABLE} t
                    LEFT JOIN {cst.DPO_TASKS_TABLE} dt ON t.{cst.TASK_ID} = dt.{cst.TASK_ID}
                    WHERE t.{cst.TASK_ID} = $1
                """
            elif task_type == TaskType.GRPOTASK.value:
                specific_query = f"""
                    SELECT t.*, gt.field_prompt, gt.file_format, gt.extra_column
                    FROM {cst.TASKS_TABLE} t
                    LEFT JOIN {cst.GRPO_TASKS_TABLE} gt ON t.{cst.TASK_ID} = gt.{cst.TASK_ID}
                    WHERE t.{cst.TASK_ID} = $1
                """
            elif task_type == TaskType.ENVIRONMENTTASK.value:
                specific_query = f"""
                    SELECT t.*, et.environment_names, et.environment_weights
                    FROM {cst.TASKS_TABLE} t
                    LEFT JOIN {cst.ENV_TASKS_TABLE} et ON t.{cst.TASK_ID} = et.{cst.TASK_ID}
                    WHERE t.{cst.TASK_ID} = $1
                """
            elif task_type == TaskType.CHATTASK.value:
                specific_query = f"""
                    SELECT
                        t.*,
                        gt.file_format,
                        gt.chat_template,
                        gt.chat_column,
                        gt.chat_role_field,
                        gt.chat_content_field,
                        gt.chat_user_reference,
                        gt.chat_assistant_reference
                    FROM {cst.TASKS_TABLE} t
                    LEFT JOIN {cst.CHAT_TASKS_TABLE} gt
                        ON t.{cst.TASK_ID} = gt.{cst.TASK_ID}
                    WHERE t.{cst.TASK_ID} = $1
                """
            else:
                logger.warning(f"Unknown task type {task_type} for task_id {row[cst.TASK_ID]}")
                continue

            specific_row = await connection.fetchrow(specific_query, row[cst.TASK_ID])
            if specific_row:
                task = await build_task_from_data(
                    task_type,
                    specific_row,
                    row[cst.TASK_ID],
                    psql_db,
                    connection,
                    image_text_pairs_loader=get_image_text_pairs,
                    reward_functions_loader=get_reward_functions,
                )
                if task:
                    tasks.append(task)

        logger.info(f"Retrieved {len(tasks)} tasks with status {status.value}")
        return tasks


async def assign_node_to_task(task_id: str, node: Node, psql_db: PSQLDB) -> None:
    """Assign a node to a task"""
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            INSERT INTO {cst.TASK_NODES_TABLE}
            ({cst.TASK_ID}, {cst.HOTKEY}, {cst.NETUID})
            VALUES ($1, $2, $3)
        """
        await connection.execute(query, task_id, node.hotkey, NETUID)


async def set_expected_repo_name(task_id: str, node: Node, psql_db: PSQLDB, expected_repo_name: str) -> None:
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            UPDATE {cst.TASK_NODES_TABLE}
            SET {cst.EXPECTED_REPO_NAME} = $1
            WHERE {cst.TASK_ID} = $2
            AND {cst.HOTKEY} = $3
            AND {cst.NETUID} = $4
        """
        await connection.execute(query, expected_repo_name, task_id, node.hotkey, NETUID)


async def set_starting_model_repo(task_id: str, hotkey: str, starting_model_repo: str, psql_db: PSQLDB) -> None:
    """Set the per-miner starting model for model continuation between rounds."""
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            UPDATE {cst.TASK_NODES_TABLE}
            SET {cst.STARTING_MODEL_REPO} = $1
            WHERE {cst.TASK_ID} = $2
            AND {cst.HOTKEY} = $3
            AND {cst.NETUID} = $4
        """
        await connection.execute(query, starting_model_repo, task_id, hotkey, NETUID)


async def get_starting_model_repo(task_id: str, hotkey: str, psql_db: PSQLDB) -> str | None:
    """Get the per-miner starting model for this task, if set."""
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            SELECT {cst.STARTING_MODEL_REPO}
            FROM {cst.TASK_NODES_TABLE}
            WHERE {cst.TASK_ID} = $1
            AND {cst.HOTKEY} = $2
            AND {cst.NETUID} = $3
        """
        row = await connection.fetchrow(query, task_id, hotkey, NETUID)
        return row[cst.STARTING_MODEL_REPO] if row and row[cst.STARTING_MODEL_REPO] else None


async def update_task(updated_task: AnyTypeRawTask, psql_db: PSQLDB) -> AnyTypeRawTask:
    existing_task = await get_task(updated_task.task_id, psql_db)

    if not existing_task:
        raise ValueError(f"Task {updated_task.task_id} not found in the database?")

    existing_task_dict = existing_task.model_dump()
    updates = {}
    for field, value in updated_task.dict(exclude_unset=True, exclude={cst.ASSIGNED_MINERS, cst.UPDATED_AT}).items():
        if existing_task_dict.get(field, None) != value:
            updates[field] = value

    async with await psql_db.connection() as connection:
        connection: Connection
        async with connection.transaction():
            base_task_fields = await get_table_fields(cst.TASKS_TABLE, connection)
            base_updates = {k: v for k, v in updates.items() if k in base_task_fields}
            if cst.AUGMENTATION_CONFIG in base_updates and base_updates[cst.AUGMENTATION_CONFIG] is not None:
                val = base_updates[cst.AUGMENTATION_CONFIG]
                base_updates[cst.AUGMENTATION_CONFIG] = val if isinstance(val, dict) else val.model_dump()
            if cst.BASELINE_STATS in base_updates and base_updates[cst.BASELINE_STATS] is not None:
                val = base_updates[cst.BASELINE_STATS]
                base_updates[cst.BASELINE_STATS] = val if isinstance(val, dict) else val.model_dump()
            if base_updates:
                set_clause = ", ".join([f"{column} = ${i + 2}" for i, column in enumerate(base_updates.keys())])
                values = list(base_updates.values())
                query = f"""
                    UPDATE {cst.TASKS_TABLE}
                    SET {set_clause}, {cst.UPDATED_AT} = CURRENT_TIMESTAMP
                    WHERE {cst.TASK_ID} = $1
                """
                await connection.execute(query, updated_task.task_id, *values)
            else:
                query = f"""
                    UPDATE {cst.TASKS_TABLE}
                    SET {cst.UPDATED_AT} = CURRENT_TIMESTAMP
                    WHERE {cst.TASK_ID} = $1
                """
                await connection.execute(query, updated_task.task_id)

            task_type = normalize_task_type(updated_task.task_type)
            specific_table = TASK_SPECIFIC_UPDATE_TABLE_BY_TYPE.get(task_type)
            if specific_table is not None:
                await update_task_specific_fields(connection, updated_task.task_id, specific_table, updates)

            if task_type == TaskType.IMAGETASK:
                if "image_text_pairs" in updates:
                    await delete_image_text_pairs(updated_task.task_id, psql_db)
                    pairs = [ImageTextPair(**pair) for pair in updates["image_text_pairs"]]
                    await add_image_text_pairs(updated_task.task_id, pairs, psql_db)

            elif task_type == TaskType.GRPOTASK:
                if "reward_functions" in updates:
                    await delete_reward_functions(updated_task.task_id, psql_db)
                    reward_functions = [RewardFunction(**reward_function) for reward_function in updates["reward_functions"]]
                    await add_reward_functions(updated_task.task_id, reward_functions, psql_db)

            if updated_task.assigned_miners is not None:
                await connection.execute(
                    f"DELETE FROM {cst.TASK_NODES_TABLE} WHERE {cst.TASK_ID} = $1 AND {cst.NETUID} = $2",
                    updated_task.task_id,
                    NETUID,
                )
                if updated_task.assigned_miners:
                    query = f"""
                        INSERT INTO {cst.TASK_NODES_TABLE} ({cst.TASK_ID}, {cst.HOTKEY}, {cst.NETUID})
                        SELECT $1, nodes.{cst.HOTKEY}, $3
                        FROM {cst.NODES_TABLE} nodes
                        WHERE nodes.{cst.HOTKEY} = ANY($2)
                        AND nodes.{cst.NETUID} = $3
                    """
                    await connection.execute(query, updated_task.task_id, updated_task.assigned_miners, NETUID)

    return await get_task(updated_task.task_id, psql_db)


async def get_current_task_stats(psql_db: PSQLDB, include_tournament_tasks=False) -> NetworkStats:
    async with await psql_db.connection() as connection:
        tournament_tasks_clause = (
            ""
            if include_tournament_tasks
            else f"WHERE {cst.TASK_ID} NOT IN (SELECT {cst.TASK_ID}::uuid FROM {cst.TOURNAMENT_TASKS_TABLE})"
        )

        # Always exclude benchmark tasks from stats
        benchmark_tasks_clause = f"""
            AND {cst.TASK_ID} NOT IN (
                SELECT {cst.TASK_ID} FROM {cst.BENCHMARK_ROOT_TASKS_TABLE}
                UNION
                SELECT {cst.COPY_TASK_ID} FROM {cst.BENCHMARK_TASK_COPIES_TABLE}
            )
        """

        query = f"""
            SELECT
                COUNT(*) FILTER (WHERE {cst.STATUS} = $1) as number_of_jobs_training,
                COUNT(*) FILTER (WHERE {cst.STATUS} = $2) as number_of_jobs_preevaluation,
                COUNT(*) FILTER (WHERE {cst.STATUS} = $3) as number_of_jobs_evaluating,
                COUNT(*) FILTER (WHERE {cst.STATUS} = $4) as number_of_jobs_success,
                MIN(termination_at) FILTER (WHERE {cst.STATUS} = $1) as next_training_end
            FROM {cst.TASKS_TABLE}
            {tournament_tasks_clause}
            {benchmark_tasks_clause}
        """
        row = await connection.fetchrow(
            query,
            TaskStatus.TRAINING.value,
            TaskStatus.PREEVALUATION.value,
            TaskStatus.EVALUATING.value,
            TaskStatus.SUCCESS.value,
        )

        return NetworkStats(
            number_of_jobs_training=row["number_of_jobs_training"],
            number_of_jobs_preevaluation=row["number_of_jobs_preevaluation"],
            number_of_jobs_evaluating=row["number_of_jobs_evaluating"],
            number_of_jobs_success=row["number_of_jobs_success"],
            next_training_end=row["next_training_end"],
        )


async def get_detailed_task_stats(psql_db: PSQLDB, include_tournament_tasks=False) -> DetailedNetworkStats:
    tournament_tasks_clause = (
        ""
        if include_tournament_tasks
        else f"WHERE {cst.TASK_ID} NOT IN (SELECT {cst.TASK_ID}::uuid FROM {cst.TOURNAMENT_TASKS_TABLE})"
    )

    # Always exclude benchmark tasks from stats
    benchmark_tasks_clause = f"""
        AND {cst.TASK_ID} NOT IN (
            SELECT {cst.TASK_ID} FROM {cst.BENCHMARK_ROOT_TASKS_TABLE}
            UNION
            SELECT {cst.COPY_TASK_ID} FROM {cst.BENCHMARK_TASK_COPIES_TABLE}
        )
    """

    async with await psql_db.connection() as connection:
        query = f"""
            SELECT
                COUNT(*) FILTER (WHERE {cst.STATUS} = $1) as number_of_jobs_training,
                COUNT(*) FILTER (WHERE {cst.STATUS} = $2) as number_of_jobs_preevaluation,
                COUNT(*) FILTER (WHERE {cst.STATUS} = $3) as number_of_jobs_evaluating,
                COUNT(*) FILTER (WHERE {cst.STATUS} = $4) as number_of_jobs_success,
                MIN(termination_at) FILTER (WHERE {cst.STATUS} = $1) as next_training_end
            FROM {cst.TASKS_TABLE}
            {tournament_tasks_clause}
            {benchmark_tasks_clause}
        """
        row = await connection.fetchrow(
            query,
            TaskStatus.TRAINING.value,
            TaskStatus.PREEVALUATION.value,
            TaskStatus.EVALUATING.value,
            TaskStatus.SUCCESS.value,
        )

        type_query = f"""
            SELECT
                {cst.TASK_TYPE},
                COUNT(*) FILTER (WHERE {cst.STATUS} = $1) as training_count,
                COUNT(*) FILTER (WHERE {cst.STATUS} = $2) as preevaluation_count,
                COUNT(*) FILTER (WHERE {cst.STATUS} = $3) as evaluating_count,
                COUNT(*) FILTER (WHERE {cst.STATUS} = $4) as success_count
            FROM {cst.TASKS_TABLE}
            GROUP BY {cst.TASK_TYPE}
        """
        type_rows = await connection.fetch(
            type_query,
            TaskStatus.TRAINING.value,
            TaskStatus.PREEVALUATION.value,
            TaskStatus.EVALUATING.value,
            TaskStatus.SUCCESS.value,
        )

        stats = DetailedNetworkStats(
            number_of_jobs_training=row["number_of_jobs_training"],
            number_of_jobs_preevaluation=row["number_of_jobs_preevaluation"],
            number_of_jobs_evaluating=row["number_of_jobs_evaluating"],
            number_of_jobs_success=row["number_of_jobs_success"],
            next_training_end=row["next_training_end"],
        )

        type_mapping = {
            TaskType.INSTRUCTTEXTTASK.value: "instruct",
            TaskType.DPOTASK.value: "dpo",
            TaskType.GRPOTASK.value: "grpo",
            TaskType.ENVIRONMENTTASK.value: "env",
            TaskType.CHATTASK.value: "chat",
            TaskType.IMAGETASK.value: "image",
        }

        for row in type_rows:
            prefix = type_mapping.get(row[cst.TASK_TYPE])
            if prefix:
                for status, count in {
                    "training": row["training_count"],
                    "preevaluation": row["preevaluation_count"],
                    "evaluating": row["evaluating_count"],
                    "success": row["success_count"],
                }.items():
                    setattr(stats, f"{prefix}_{status}", count)

        return stats



async def delete_task(task_id: UUID, psql_db: PSQLDB) -> None:
    """Delete a task and its associated node assignments"""
    async with await psql_db.connection() as connection:
        async with connection.transaction():
            # First delete task_nodes entries for this netuid
            await connection.execute(
                f"""
                DELETE FROM {cst.TASK_NODES_TABLE}
                WHERE {cst.TASK_ID} = $1 AND {cst.NETUID} = $2
                """,
                task_id,
                NETUID,
            )

            # Then delete the task if it has no more node assignments
            await connection.execute(
                f"""
                DELETE FROM {cst.TASKS_TABLE}
                WHERE {cst.TASK_ID} = $1
                AND NOT EXISTS (
                    SELECT 1 FROM {cst.TASK_NODES_TABLE}
                    WHERE {cst.TASK_ID} = $1
                    AND {cst.NETUID} = $2
                )
                """,
                task_id,
                NETUID,
            )


async def get_miners_for_task(task_id: UUID, psql_db: PSQLDB) -> list[Node]:
    """Retrieve all miners assigned to a specific task."""
    async with await psql_db.connection() as connection:
        query = f"""
            SELECT nodes.* FROM {cst.NODES_TABLE} nodes
            JOIN {cst.TASK_NODES_TABLE} task_nodes
            ON nodes.hotkey = task_nodes.hotkey AND nodes.netuid = task_nodes.netuid
            WHERE task_nodes.task_id = $1
        """
        rows = await connection.fetch(query, task_id)
        # Filter out 'trust' field as it was removed from Node model
        return [Node(**{k: v for k, v in dict(row).items() if k != cst.TRUST}) for row in rows]


async def get_task(task_id: UUID, psql_db: PSQLDB, connection: Connection | None = None) -> AnyTypeTask:
    async def _get_task_inner(conn: Connection) -> AnyTypeTask:
        base_query = f"""
            SELECT * FROM {cst.TASKS_TABLE} WHERE {cst.TASK_ID} = $1
        """
        base_row = await conn.fetchrow(base_query, task_id)

        if not base_row:
            return None

        task_type = base_row[cst.TASK_TYPE]

        if task_type == TaskType.INSTRUCTTEXTTASK.value:
            specific_query = f"""
                SELECT t.*, tt.field_system,
                       tt.field_instruction, tt.field_input, tt.field_output,
                       tt.format, tt.no_input_format
                FROM {cst.TASKS_TABLE} t
                LEFT JOIN {cst.INSTRUCT_TEXT_TASKS_TABLE} tt ON t.{cst.TASK_ID} = tt.{cst.TASK_ID}
                WHERE t.{cst.TASK_ID} = $1
            """
        elif task_type == TaskType.IMAGETASK.value:
            specific_query = f"""
                SELECT t.*, it.model_type
                FROM {cst.TASKS_TABLE} t
                LEFT JOIN {cst.IMAGE_TASKS_TABLE} it ON t.{cst.TASK_ID} = it.{cst.TASK_ID}
                WHERE t.{cst.TASK_ID} = $1
            """
        elif task_type == TaskType.DPOTASK.value:
            specific_query = f"""
                SELECT t.*, dt.field_prompt, dt.field_system, dt.field_chosen, dt.field_rejected,
                       dt.prompt_format, dt.chosen_format, dt.rejected_format, dt.file_format
                FROM {cst.TASKS_TABLE} t
                LEFT JOIN {cst.DPO_TASKS_TABLE} dt ON t.{cst.TASK_ID} = dt.{cst.TASK_ID}
                WHERE t.{cst.TASK_ID} = $1
            """
        elif task_type == TaskType.GRPOTASK.value:
            specific_query = f"""
                SELECT t.*, gt.field_prompt, gt.file_format, gt.extra_column
                FROM {cst.TASKS_TABLE} t
                LEFT JOIN {cst.GRPO_TASKS_TABLE} gt ON t.{cst.TASK_ID} = gt.{cst.TASK_ID}
                WHERE t.{cst.TASK_ID} = $1
            """
        elif task_type == TaskType.ENVIRONMENTTASK.value:
            specific_query = f"""
                SELECT t.*, et.environment_names, et.environment_weights
                FROM {cst.TASKS_TABLE} t
                LEFT JOIN {cst.ENV_TASKS_TABLE} et ON t.{cst.TASK_ID} = et.{cst.TASK_ID}
                WHERE t.{cst.TASK_ID} = $1
            """
        elif task_type == TaskType.CHATTASK.value:
            specific_query = f"""
                SELECT
                    t.*,
                    gt.file_format,
                    gt.chat_template,
                    gt.chat_column,
                    gt.chat_role_field,
                    gt.chat_content_field,
                    gt.chat_user_reference,
                    gt.chat_assistant_reference
                FROM {cst.TASKS_TABLE} t
                LEFT JOIN {cst.CHAT_TASKS_TABLE} gt
                    ON t.{cst.TASK_ID} = gt.{cst.TASK_ID}
                WHERE t.{cst.TASK_ID} = $1
            """
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

        full_row = await conn.fetchrow(specific_query, task_id)

        if not full_row:
            return None

        return await build_task_from_data(
            task_type,
            full_row,
            task_id,
            psql_db,
            conn,
            image_text_pairs_loader=get_image_text_pairs,
            reward_functions_loader=get_reward_functions,
        )

    if connection is not None:
        return await _get_task_inner(connection)

    async with await psql_db.connection() as connection:
        return await _get_task_inner(connection)


async def get_task_by_id(task_id: UUID, psql_db: PSQLDB) -> AnyTypeTask:
    """Get a task by ID along with its winning submissions and task-specific details"""
    async with await psql_db.connection() as connection:
        connection: Connection

        base_query = f"""
            SELECT * FROM {cst.TASKS_TABLE} WHERE {cst.TASK_ID} = $1
        """
        base_row = await connection.fetchrow(base_query, task_id)

        if not base_row:
            return None

        task_type = base_row[cst.TASK_TYPE]

        victorious_repo_cte = f"""
            WITH victorious_repo AS (
                SELECT submissions.task_id, submissions.repo
                FROM {cst.SUBMISSIONS_TABLE} submissions
                JOIN {cst.TASK_NODES_TABLE} task_nodes
                ON submissions.task_id = task_nodes.task_id
                AND submissions.hotkey = task_nodes.hotkey
                AND submissions.netuid = task_nodes.netuid
                WHERE submissions.task_id = $1
                AND task_nodes.quality_score IS NOT NULL
                ORDER BY task_nodes.quality_score DESC
                LIMIT 1
            )
        """

        if task_type == TaskType.INSTRUCTTEXTTASK.value:
            specific_query = f"""
                {victorious_repo_cte}
                SELECT
                    tasks.*,
                    tt.field_system, tt.field_instruction, tt.field_input, tt.field_output,
                    tt.format, tt.no_input_format, tt.system_format, tt.file_format,
                    COALESCE(tasks.training_repo_backup, victorious_repo.repo) as trained_model_repository
                FROM {cst.TASKS_TABLE} tasks
                LEFT JOIN {cst.INSTRUCT_TEXT_TASKS_TABLE} tt ON tasks.{cst.TASK_ID} = tt.{cst.TASK_ID}
                LEFT JOIN victorious_repo ON tasks.task_id = victorious_repo.task_id
                WHERE tasks.{cst.TASK_ID} = $1
            """
        elif task_type == TaskType.CHATTASK.value:
            specific_query = f"""
                {victorious_repo_cte}
                SELECT
                    tasks.*,
                    tt.file_format,
                    tt.chat_template,
                    tt.chat_column,
                    tt.chat_role_field,
                    tt.chat_content_field,
                    tt.chat_user_reference,
                    tt.chat_assistant_reference,
                    COALESCE(tasks.training_repo_backup, victorious_repo.repo) as trained_model_repository
                FROM {cst.TASKS_TABLE} tasks
                LEFT JOIN {cst.CHAT_TASKS_TABLE} tt
                    ON tasks.{cst.TASK_ID} = tt.{cst.TASK_ID}
                LEFT JOIN victorious_repo ON tasks.task_id = victorious_repo.task_id
                WHERE tasks.{cst.TASK_ID} = $1
            """
        elif task_type == TaskType.IMAGETASK.value:
            specific_query = f"""
                {victorious_repo_cte}
                SELECT
                    tasks.*,
                    it.model_type,
                    victorious_repo.repo as trained_model_repository
                FROM {cst.TASKS_TABLE} tasks
                LEFT JOIN {cst.IMAGE_TASKS_TABLE} it ON tasks.{cst.TASK_ID} = it.{cst.TASK_ID}
                LEFT JOIN victorious_repo ON tasks.task_id = victorious_repo.task_id
                WHERE tasks.{cst.TASK_ID} = $1
            """
        elif task_type == TaskType.DPOTASK.value:
            specific_query = f"""
                {victorious_repo_cte}
                SELECT
                    tasks.*,
                    dt.field_prompt, dt.field_system, dt.field_chosen, dt.field_rejected,
                    dt.prompt_format, dt.chosen_format, dt.rejected_format, dt.file_format,
                    COALESCE(tasks.training_repo_backup, victorious_repo.repo) as trained_model_repository
                FROM {cst.TASKS_TABLE} tasks
                LEFT JOIN {cst.DPO_TASKS_TABLE} dt ON tasks.{cst.TASK_ID} = dt.{cst.TASK_ID}
                LEFT JOIN victorious_repo ON tasks.task_id = victorious_repo.task_id
                WHERE tasks.{cst.TASK_ID} = $1
            """
        elif task_type == TaskType.GRPOTASK.value:
            specific_query = f"""
                {victorious_repo_cte}
                SELECT
                    tasks.*,
                    gt.field_prompt, gt.file_format, gt.extra_column,
                    COALESCE(tasks.training_repo_backup, victorious_repo.repo) as trained_model_repository
                FROM {cst.TASKS_TABLE} tasks
                LEFT JOIN {cst.GRPO_TASKS_TABLE} gt ON tasks.{cst.TASK_ID} = gt.{cst.TASK_ID}
                LEFT JOIN victorious_repo ON tasks.task_id = victorious_repo.task_id
                WHERE tasks.{cst.TASK_ID} = $1
            """
        elif task_type == TaskType.ENVIRONMENTTASK.value:
            specific_query = f"""
                {victorious_repo_cte}
                SELECT
                    tasks.*,
                    et.environment_names, et.environment_weights, et.eval_seed,
                    COALESCE(tasks.training_repo_backup, victorious_repo.repo) as trained_model_repository
                FROM {cst.TASKS_TABLE} tasks
                LEFT JOIN {cst.ENV_TASKS_TABLE} et ON tasks.{cst.TASK_ID} = et.{cst.TASK_ID}
                LEFT JOIN victorious_repo ON tasks.task_id = victorious_repo.task_id
                WHERE tasks.{cst.TASK_ID} = $1
            """
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

        row = await connection.fetchrow(specific_query, task_id)
        if not row:
            return None

        return await build_task_from_data(
            task_type,
            row,
            task_id,
            psql_db,
            connection,
            public=True,
            image_text_pairs_loader=get_image_text_pairs,
            reward_functions_loader=get_reward_functions,
        )


async def get_tasks_by_ids(task_ids: list[UUID], psql_db: PSQLDB, connection: Connection | None = None) -> list[AnyTypeTask]:
    """Get multiple tasks by their IDs efficiently in batch.

    Args:
        task_ids: List of task IDs to fetch
        psql_db: Database connection
        connection: Optional existing connection to reuse

    Returns:
        List of task objects in the exact same order as input task_ids
    """
    if not task_ids:
        return []

    async def _get_tasks_by_ids_inner(conn: Connection) -> list[AnyTypeTask]:
        # Get base task data for all IDs

        base_query = f"""
            SELECT * FROM {cst.TASKS_TABLE}
            WHERE {cst.TASK_ID} = ANY($1)
            ORDER BY array_position($1, {cst.TASK_ID})
        """
        base_rows = await conn.fetch(base_query, task_ids)

        if not base_rows:
            return []

        # Group tasks by type for efficient batch loading
        tasks_by_type = {}
        for row in base_rows:
            task_type = row[cst.TASK_TYPE]
            if task_type not in tasks_by_type:
                tasks_by_type[task_type] = []
            tasks_by_type[task_type].append(row)

        # Load specific data for each task type
        all_tasks = []

        for task_type, type_rows in tasks_by_type.items():
            type_task_ids = [row[cst.TASK_ID] for row in type_rows]
            tasks = await _load_tasks_by_type(conn, task_type, type_task_ids, type_rows, psql_db)
            all_tasks.extend(tasks)

        # Create a mapping for quick lookup
        tasks_dict = {task.task_id: task for task in all_tasks}

        # Return tasks in the same order as input task_ids
        result = []
        for task_id in task_ids:
            task = tasks_dict.get(task_id)
            if task:
                result.append(task)
            else:
                logger.warning(f"Task {task_id} not found in batch load")

        return result

    if connection is not None:
        return await _get_tasks_by_ids_inner(connection)

    async with await psql_db.connection() as connection:
        return await _get_tasks_by_ids_inner(connection)


async def _load_tasks_by_type(
    conn: Connection, task_type: str, type_task_ids: list[UUID], type_rows: list, psql_db: PSQLDB
) -> list[AnyTypeTask]:
    """Load tasks of a specific type with their type-specific data"""
    specific_query = _get_specific_query_for_task_type(task_type)
    if not specific_query:
        logger.warning(f"Unknown task type {task_type}, skipping tasks")
        return []

    specific_rows = await conn.fetch(specific_query, type_task_ids)
    specific_rows_dict = {row[cst.TASK_ID]: row for row in specific_rows}

    tasks = []
    for row in type_rows:
        task_id = row[cst.TASK_ID]
        specific_row = specific_rows_dict.get(task_id)

        if specific_row:
            task = await _create_task_from_data(task_type, specific_row, task_id, psql_db, conn)
            if task:
                tasks.append(task)
        else:
            logger.warning(f"Specific data not found for task {task_id} of type {task_type}")

    return tasks


def _get_specific_query_for_task_type(task_type: str) -> str | None:
    """Get the specific query for a given task type"""
    if task_type == TaskType.INSTRUCTTEXTTASK.value:
        return f"""
            SELECT t.*, tt.field_system,
                   tt.field_instruction, tt.field_input, tt.field_output,
                   tt.format, tt.no_input_format
            FROM {cst.TASKS_TABLE} t
            LEFT JOIN {cst.INSTRUCT_TEXT_TASKS_TABLE} tt ON t.{cst.TASK_ID} = tt.{cst.TASK_ID}
            WHERE t.{cst.TASK_ID} = ANY($1)
        """
    elif task_type == TaskType.CHATTASK.value:
        return f"""
            SELECT t.*, ct.chat_template,
                   ct.chat_column, ct.chat_role_field, ct.chat_content_field,
                   ct.chat_user_reference, ct.chat_assistant_reference
            FROM {cst.TASKS_TABLE} t
            LEFT JOIN {cst.CHAT_TASKS_TABLE} ct ON t.{cst.TASK_ID} = ct.{cst.TASK_ID}
            WHERE t.{cst.TASK_ID} = ANY($1)
        """
    elif task_type == TaskType.IMAGETASK.value:
        return f"""
            SELECT t.*, it.model_type
            FROM {cst.TASKS_TABLE} t
            LEFT JOIN {cst.IMAGE_TASKS_TABLE} it ON t.{cst.TASK_ID} = it.{cst.TASK_ID}
            WHERE t.{cst.TASK_ID} = ANY($1)
        """
    elif task_type == TaskType.DPOTASK.value:
        return f"""
            SELECT t.*, dt.field_prompt, dt.field_system, dt.field_chosen, dt.field_rejected,
                   dt.prompt_format, dt.chosen_format, dt.rejected_format, dt.file_format
            FROM {cst.TASKS_TABLE} t
            LEFT JOIN {cst.DPO_TASKS_TABLE} dt ON t.{cst.TASK_ID} = dt.{cst.TASK_ID}
            WHERE t.{cst.TASK_ID} = ANY($1)
        """
    elif task_type == TaskType.GRPOTASK.value:
        return f"""
            SELECT t.*, gt.field_prompt, gt.file_format
            FROM {cst.TASKS_TABLE} t
            LEFT JOIN {cst.GRPO_TASKS_TABLE} gt ON t.{cst.TASK_ID} = gt.{cst.TASK_ID}
            WHERE t.{cst.TASK_ID} = ANY($1)
        """
    elif task_type == TaskType.ENVIRONMENTTASK.value:
        return f"""
            SELECT t.*, et.environment_names, et.environment_weights
            FROM {cst.TASKS_TABLE} t
            LEFT JOIN {cst.ENV_TASKS_TABLE} et ON t.{cst.TASK_ID} = et.{cst.TASK_ID}
            WHERE t.{cst.TASK_ID} = ANY($1)
        """
    return None


async def _create_task_from_data(
    task_type: str, task_data: dict, task_id: UUID, psql_db: PSQLDB, conn: Connection
) -> AnyTypeTask | None:
    """Create a task object from the given data based on task type"""
    return await build_task_from_data(
        task_type,
        task_data,
        task_id,
        psql_db,
        conn,
        image_text_pairs_loader=get_image_text_pairs,
        reward_functions_loader=get_reward_functions,
    )


async def get_tasks_by_account_id(psql_db: PSQLDB, account_id: UUID, limit: int = 100, offset: int = 0) -> list[AnyTypeTask]:
    async with await psql_db.connection() as connection:
        connection: Connection
        base_query = f"""
            WITH victorious_repo AS (
                SELECT
                    submissions.{cst.TASK_ID},
                    submissions.{cst.REPO},
                    ROW_NUMBER() OVER (
                        PARTITION BY submissions.{cst.TASK_ID}
                        ORDER BY task_nodes.{cst.QUALITY_SCORE} DESC
                    ) AS rn
                FROM {cst.SUBMISSIONS_TABLE} submissions
                JOIN {cst.TASK_NODES_TABLE} task_nodes
                    ON submissions.{cst.TASK_ID} = task_nodes.{cst.TASK_ID}
                   AND submissions.{cst.HOTKEY} = task_nodes.{cst.HOTKEY}
                   AND submissions.{cst.NETUID} = task_nodes.{cst.NETUID}
                WHERE task_nodes.{cst.QUALITY_SCORE} IS NOT NULL
            )
            SELECT
                tasks.*,
                COALESCE(tasks.training_repo_backup, victorious_repo.{cst.REPO}) AS trained_model_repository
            FROM {cst.TASKS_TABLE} tasks
            LEFT JOIN victorious_repo
                ON tasks.{cst.TASK_ID} = victorious_repo.{cst.TASK_ID}
               AND victorious_repo.rn = 1
            WHERE tasks.{cst.ACCOUNT_ID} = $1
            AND tasks.{cst.TASK_ID} NOT IN (
                SELECT {cst.TASK_ID} FROM {cst.BENCHMARK_ROOT_TASKS_TABLE}
                UNION
                SELECT {cst.COPY_TASK_ID} FROM {cst.BENCHMARK_TASK_COPIES_TABLE}
            )
            ORDER BY tasks.{cst.CREATED_AT} DESC
            LIMIT $2 OFFSET $3
        """

        rows = await connection.fetch(base_query, account_id, limit, offset)
        tasks = []

        for row in rows:
            task_data = dict(row)
            task_type = task_data[cst.TASK_TYPE]

            if task_type == TaskType.INSTRUCTTEXTTASK.value:
                instruct_text_query = f"""
                    SELECT field_system, field_instruction, field_input, field_output,
                           format, no_input_format
                    FROM {cst.INSTRUCT_TEXT_TASKS_TABLE}
                    WHERE {cst.TASK_ID} = $1
                """
                instruct_text_row = await connection.fetchrow(instruct_text_query, task_data[cst.TASK_ID])
                if instruct_text_row:
                    task_data.update(dict(instruct_text_row))

            elif task_type == TaskType.CHATTASK.value:
                chat_query = f"""
                    SELECT chat_template, chat_column, chat_role_field, chat_content_field,
                           chat_user_reference, chat_assistant_reference
                    FROM {cst.CHAT_TASKS_TABLE}
                    WHERE {cst.TASK_ID} = $1
                """
                chat_row = await connection.fetchrow(chat_query, task_data[cst.TASK_ID])
                if chat_row:
                    task_data.update(dict(chat_row))

            elif task_type == TaskType.IMAGETASK.value:
                image_query = f"""
                    SELECT {cst.MODEL_TYPE}
                    FROM {cst.IMAGE_TASKS_TABLE}
                    WHERE {cst.TASK_ID} = $1
                """
                image_row = await connection.fetchrow(image_query, task_data[cst.TASK_ID])
                if image_row:
                    task_data.update(dict(image_row))

            elif task_type == TaskType.DPOTASK.value:
                dpo_query = f"""
                    SELECT field_prompt, field_system, field_chosen, field_rejected,
                           prompt_format, chosen_format, rejected_format, file_format
                    FROM {cst.DPO_TASKS_TABLE}
                    WHERE {cst.TASK_ID} = $1
                """
                dpo_row = await connection.fetchrow(dpo_query, task_data[cst.TASK_ID])
                if dpo_row:
                    task_data.update(dict(dpo_row))
            elif task_type == TaskType.GRPOTASK.value:
                grpo_query = f"""
                    SELECT field_prompt, file_format
                    FROM {cst.GRPO_TASKS_TABLE}
                    WHERE {cst.TASK_ID} = $1
                """
                grpo_row = await connection.fetchrow(grpo_query, task_data[cst.TASK_ID])
                if grpo_row:
                    task_data.update(dict(grpo_row))
            elif task_type == TaskType.ENVIRONMENTTASK.value:
                env_query = f"""
                    SELECT environment_names, environment_weights
                    FROM {cst.ENV_TASKS_TABLE}
                    WHERE {cst.TASK_ID} = $1
                """
                env_row = await connection.fetchrow(env_query, task_data[cst.TASK_ID])
                if env_row:
                    task_data.update(dict(env_row))
            else:
                logger.warning(f"Unknown task type {task_type} for task_id {task_data[cst.TASK_ID]}")
                continue

            task = await build_task_from_data(
                task_type,
                task_data,
                task_data[cst.TASK_ID],
                psql_db,
                connection,
                public=True,
                image_text_pairs_loader=get_image_text_pairs,
                reward_functions_loader=get_reward_functions,
            )
            if task:
                tasks.append(task)
        return tasks


async def get_completed_organic_tasks(
    psql_db: PSQLDB,
    hours: int | None = None,
    task_type: TaskType | None = None,
    search_model_name: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[AnyTypeTask]:
    """Get completed organic tasks with optional filters

    Args:
        psql_db: Database connection
        hours: Optional number of hours to look back
        task_type: Optional task type filter
        search_model_name: Optional search term to filter models by name
        limit: Number of tasks per page
        offset: Offset for pagination
    """
    async with await psql_db.connection() as connection:
        connection: Connection

        where_clauses = [f"tasks.{cst.STATUS} = $1", f"tasks.{cst.IS_ORGANIC} = true"]
        params = [TaskStatus.SUCCESS.value]
        param_count = 1

        if hours is not None:
            param_count += 1
            where_clauses.append(f"tasks.{cst.TERMINATION_AT} >= NOW() - ${param_count} * INTERVAL '1 hour'")
            params.append(hours)

        if task_type is not None:
            param_count += 1
            where_clauses.append(f"tasks.{cst.TASK_TYPE} = ${param_count}")
            params.append(task_type.value)

        if search_model_name is not None:
            search_terms = search_model_name.lower().split()
            for term in search_terms:
                param_count += 1
                where_clauses.append(f"tasks.result_model_name_lower LIKE ${param_count}")
                params.append(f"%{term}%")

        # Always exclude benchmark tasks from organic task queries
        where_clauses.append(f"""
            tasks.{cst.TASK_ID} NOT IN (
                SELECT {cst.TASK_ID} FROM {cst.BENCHMARK_ROOT_TASKS_TABLE}
                UNION
                SELECT {cst.COPY_TASK_ID} FROM {cst.BENCHMARK_TASK_COPIES_TABLE}
            )
        """)

        where_clause = " AND ".join(where_clauses)

        query = f"""
            WITH victorious_repo AS (
                SELECT
                    submissions.{cst.TASK_ID},
                    submissions.{cst.REPO},
                    ROW_NUMBER() OVER (
                        PARTITION BY submissions.{cst.TASK_ID}
                        ORDER BY task_nodes.{cst.QUALITY_SCORE} DESC
                    ) AS rn
                FROM {cst.SUBMISSIONS_TABLE} submissions
                JOIN {cst.TASK_NODES_TABLE} task_nodes
                    ON submissions.{cst.TASK_ID} = task_nodes.{cst.TASK_ID}
                    AND submissions.{cst.HOTKEY} = task_nodes.{cst.HOTKEY}
                    AND submissions.{cst.NETUID} = task_nodes.{cst.NETUID}
                WHERE task_nodes.{cst.QUALITY_SCORE} IS NOT NULL
                ORDER BY task_nodes.{cst.QUALITY_SCORE} DESC
            )
            SELECT
                tasks.{cst.TASK_ID},
                COALESCE(tasks.training_repo_backup, victorious_repo.{cst.REPO}) as trained_model_repository
            FROM {cst.TASKS_TABLE} tasks
            LEFT JOIN victorious_repo
                ON tasks.{cst.TASK_ID} = victorious_repo.{cst.TASK_ID}
                AND victorious_repo.rn = 1
            WHERE {where_clause}
            ORDER BY tasks.{cst.TERMINATION_AT} DESC
            LIMIT ${param_count + 1} OFFSET ${param_count + 2}
        """

        params.extend([limit, offset])
        task_ids = await connection.fetch(query, *params)

    tasks_list = []
    for task_row in task_ids:
        task = await get_task_by_id(task_row[cst.TASK_ID], psql_db)
        tasks_list.append(task)

    return tasks_list


async def get_expected_repo_name(task_id: UUID, hotkey: str, psql_db: PSQLDB) -> str | None:
    async with await psql_db.connection() as connection:
        query = f"""
            SELECT {cst.EXPECTED_REPO_NAME}
            FROM {cst.TASK_NODES_TABLE}
            WHERE {cst.TASK_ID} = $1 AND {cst.HOTKEY} = $2 AND {cst.NETUID} = $3
        """
        return await connection.fetchval(query, task_id, hotkey, NETUID)


async def add_task_evaluation_pairs(task_id: UUID, psql_db: PSQLDB, include_failed_training: bool = False) -> None:
    """Seed missing evaluation rows for a task from current assigned task_nodes.

    Existing rows are preserved (including deployment ids and terminal statuses).
    When include_failed_training is True, eval rows are seeded for all miners
    regardless of training status (needed for PvP where failed miners auto-lose).
    """
    async with await psql_db.connection() as connection:
        async with connection.transaction():
            if include_failed_training:
                training_filter = f"""
                    tn.{cst.TASK_ID} NOT IN (SELECT {cst.TASK_ID} FROM {cst.TOURNAMENT_TASKS_TABLE})
                    OR EXISTS (
                        SELECT 1 FROM {cst.TOURNAMENT_TASK_HOTKEY_TRAININGS_TABLE} ttht
                        WHERE ttht.{cst.TASK_ID} = tn.{cst.TASK_ID}
                          AND ttht.{cst.HOTKEY} = tn.{cst.HOTKEY}
                          AND ttht.{cst.TRAINING_STATUS} IN ('success', 'failure')
                    )
                """
            else:
                training_filter = f"""
                    tn.{cst.TASK_ID} NOT IN (SELECT {cst.TASK_ID} FROM {cst.TOURNAMENT_TASKS_TABLE})
                    OR EXISTS (
                        SELECT 1 FROM {cst.TOURNAMENT_TASK_HOTKEY_TRAININGS_TABLE} ttht
                        WHERE ttht.{cst.TASK_ID} = tn.{cst.TASK_ID}
                          AND ttht.{cst.HOTKEY} = tn.{cst.HOTKEY}
                          AND ttht.{cst.TRAINING_STATUS} = 'success'
                    )
                """
            query = f"""
                INSERT INTO {cst.EVALUATIONS_TABLE}
                ({cst.TASK_ID}, {cst.HOTKEY}, {cst.NETUID}, {cst.EVALUATION_STATUS}, {cst.CREATED_AT}, {cst.UPDATED_AT})
                SELECT tn.{cst.TASK_ID}, tn.{cst.HOTKEY}, tn.{cst.NETUID}, 'pending', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
                FROM {cst.TASK_NODES_TABLE} tn
                WHERE tn.{cst.TASK_ID} = $1 AND tn.{cst.NETUID} = $2
                AND ({training_filter})
                ON CONFLICT ({cst.TASK_ID}, {cst.HOTKEY}, {cst.NETUID}) DO NOTHING
            """
            await connection.execute(query, task_id, NETUID)


async def get_task_evaluation_rows(task_id: UUID, psql_db: PSQLDB) -> list[dict]:
    async with await psql_db.connection() as connection:
        rows = await connection.fetch(
            f"""
            SELECT
                e.{cst.TASK_ID},
                e.{cst.HOTKEY},
                e.{cst.NETUID},
                tn.{cst.EXPECTED_REPO_NAME},
                e.{cst.EVALUATION_STATUS},
                e.{cst.DEPLOYMENT_ID},
                e.{cst.DEPLOYMENT_ENV_ID}
            FROM {cst.EVALUATIONS_TABLE} e
            LEFT JOIN {cst.TASK_NODES_TABLE} tn
                ON e.{cst.TASK_ID} = tn.{cst.TASK_ID}
                AND e.{cst.HOTKEY} = tn.{cst.HOTKEY}
                AND e.{cst.NETUID} = tn.{cst.NETUID}
            WHERE e.{cst.TASK_ID} = $1 AND e.{cst.NETUID} = $2
            """,
            task_id,
            NETUID,
        )
        return [dict(row) for row in rows]


async def get_task_evaluations_by_status(task_id: UUID, status: str, psql_db: PSQLDB) -> list[dict]:
    async with await psql_db.connection() as connection:
        rows = await connection.fetch(
            f"""
            SELECT
                e.{cst.TASK_ID},
                e.{cst.HOTKEY},
                e.{cst.NETUID},
                tn.{cst.EXPECTED_REPO_NAME},
                e.{cst.EVALUATION_STATUS},
                e.{cst.DEPLOYMENT_ID},
                e.{cst.DEPLOYMENT_ENV_ID}
            FROM {cst.EVALUATIONS_TABLE} e
            LEFT JOIN {cst.TASK_NODES_TABLE} tn
                ON e.{cst.TASK_ID} = tn.{cst.TASK_ID}
                AND e.{cst.HOTKEY} = tn.{cst.HOTKEY}
                AND e.{cst.NETUID} = tn.{cst.NETUID}
            WHERE e.{cst.TASK_ID} = $1 AND e.{cst.EVALUATION_STATUS} = $2 AND e.{cst.NETUID} = $3
            """,
            task_id,
            status,
            NETUID,
        )
        return [dict(row) for row in rows]


async def count_task_evaluations_by_status(status: str, psql_db: PSQLDB) -> int:
    async with await psql_db.connection() as connection:
        return await connection.fetchval(
            f"""
            SELECT COUNT(*)
            FROM {cst.EVALUATIONS_TABLE}
            WHERE {cst.EVALUATION_STATUS} = $1 AND {cst.NETUID} = $2
            """,
            status,
            NETUID,
        )


async def get_task_ids_with_evaluation_statuses(
    statuses: list[str],
    psql_db: PSQLDB,
    task_statuses: list[str] | None = None,
    tournament_only: bool = False,
) -> list[UUID]:
    """Get task ids that currently have evaluation rows in the given statuses."""
    if not statuses:
        return []

    task_status_clause = ""
    query_params: list[object] = [statuses, NETUID]

    if task_statuses:
        task_status_clause = f"AND t.{cst.STATUS} = ANY($3)"
        query_params.append(task_statuses)

    tournament_clause = ""
    if tournament_only:
        tournament_clause = f"AND e.{cst.TASK_ID} IN (SELECT {cst.TASK_ID} FROM {cst.TOURNAMENT_TASKS_TABLE})"

    async with await psql_db.connection() as connection:
        rows = await connection.fetch(
            f"""
            SELECT DISTINCT e.{cst.TASK_ID}
            FROM {cst.EVALUATIONS_TABLE} e
            JOIN {cst.TASKS_TABLE} t
                ON e.{cst.TASK_ID} = t.{cst.TASK_ID}
            WHERE e.{cst.EVALUATION_STATUS} = ANY($1)
              AND e.{cst.NETUID} = $2
              {task_status_clause}
              {tournament_clause}
            """,
            *query_params,
        )
    return [row[cst.TASK_ID] for row in rows]


async def update_task_evaluations_status(task_id: UUID, hotkeys: list[str], status: str, psql_db: PSQLDB) -> None:
    if not hotkeys:
        return
    async with await psql_db.connection() as connection:
        await connection.execute(
            f"""
            UPDATE {cst.EVALUATIONS_TABLE}
            SET {cst.EVALUATION_STATUS} = $4, {cst.UPDATED_AT} = CURRENT_TIMESTAMP
            WHERE {cst.TASK_ID} = $1 AND {cst.HOTKEY} = ANY($2) AND {cst.NETUID} = $3
            """,
            task_id,
            hotkeys,
            NETUID,
            status,
        )


async def reset_task_evaluations_to_pending(task_id: UUID, psql_db: PSQLDB) -> None:
    """Reset in-progress evaluation rows for a task back to pending."""
    async with await psql_db.connection() as connection:
        await connection.execute(
            f"""
            UPDATE {cst.EVALUATIONS_TABLE}
            SET {cst.EVALUATION_STATUS} = 'pending', {cst.UPDATED_AT} = CURRENT_TIMESTAMP
            WHERE {cst.TASK_ID} = $1 AND {cst.EVALUATION_STATUS} = 'evaluating' AND {cst.NETUID} = $2
            """,
            task_id,
            NETUID,
        )


async def reset_all_task_evaluations_to_pending(task_id: UUID, psql_db: PSQLDB) -> None:
    """Reset every evaluation row for a task back to pending."""
    async with await psql_db.connection() as connection:
        await connection.execute(
            f"""
            UPDATE {cst.EVALUATIONS_TABLE}
            SET {cst.EVALUATION_STATUS} = 'pending', {cst.UPDATED_AT} = CURRENT_TIMESTAMP
            WHERE {cst.TASK_ID} = $1 AND {cst.NETUID} = $2
            """,
            task_id,
            NETUID,
        )


async def set_evaluation_deployment_ids(
    task_id: UUID,
    hotkey: str,
    deployment_id: str | None,
    deployment_env_id: str | None,
    psql_db: PSQLDB,
) -> None:
    """Store deployment IDs for an evaluation row. Overwrites on retry."""
    async with await psql_db.connection() as connection:
        await connection.execute(
            f"""
            UPDATE {cst.EVALUATIONS_TABLE}
            SET {cst.DEPLOYMENT_ID} = $4, {cst.DEPLOYMENT_ENV_ID} = $5, {cst.UPDATED_AT} = CURRENT_TIMESTAMP
            WHERE {cst.TASK_ID} = $1 AND {cst.HOTKEY} = $2 AND {cst.NETUID} = $3
            """,
            task_id,
            hotkey,
            NETUID,
            deployment_id,
            deployment_env_id,
        )


async def get_deployment_ids_from_evaluating_tasks(psql_db: PSQLDB) -> set[str]:
    """
    Get all deployment IDs from evaluation rows with status 'pending' or 'evaluating'.
    Used on validator restart to preserve deployments that may be resumed.
    Returns deployment_id and deployment_env_id values.
    """
    async with await psql_db.connection() as connection:
        rows = await connection.fetch(
            f"""
            SELECT {cst.DEPLOYMENT_ID}, {cst.DEPLOYMENT_ENV_ID}
            FROM {cst.EVALUATIONS_TABLE}
            WHERE {cst.EVALUATION_STATUS} = ANY($1) AND {cst.NETUID} = $2
            """,
            ["pending", "evaluating"],
            NETUID,
        )
    ids: set[str] = set()
    for row in rows:
        for col in (cst.DEPLOYMENT_ID, cst.DEPLOYMENT_ENV_ID):
            val = row.get(col)
            if val and isinstance(val, str):
                ids.add(val)
    return ids


async def add_image_text_pairs(task_id: UUID, pairs: list[ImageTextPair], psql_db: PSQLDB) -> None:
    query = f"""
        INSERT INTO {cst.IMAGE_TEXT_PAIRS_TABLE} ({cst.TASK_ID}, {cst.IMAGE_URL}, {cst.TEXT_URL})
        VALUES ($1, $2, $3)
    """

    async with await psql_db.connection() as conn:
        async with conn.transaction():
            for pair in pairs:
                await conn.execute(query, task_id, pair.image_url, pair.text_url)


async def get_image_text_pairs(task_id: UUID, psql_db: PSQLDB, connection: Connection | None = None) -> list[ImageTextPair]:
    async def _get_image_text_pairs(conn: Connection) -> list[ImageTextPair]:
        query = f"""
            SELECT {cst.IMAGE_URL}, {cst.TEXT_URL}
            FROM {cst.IMAGE_TEXT_PAIRS_TABLE}
            WHERE {cst.TASK_ID} = $1
            ORDER BY {cst.ID}
        """
        rows = await conn.fetch(query, task_id)
        return [ImageTextPair(image_url=row["image_url"], text_url=row["text_url"]) for row in rows]

    if connection is not None:
        return await _get_image_text_pairs(connection)

    async with await psql_db.connection() as connection:
        return await _get_image_text_pairs(connection)


async def delete_image_text_pairs(task_id: UUID, psql_db: PSQLDB) -> None:
    query = f"""
        DELETE FROM {cst.IMAGE_TEXT_PAIRS_TABLE}
        WHERE {cst.TASK_ID} = $1
    """

    async with await psql_db.connection() as connection:
        connection: Connection
        await connection.execute(query, task_id)


async def delete_reward_functions(task_id: UUID, psql_db: PSQLDB) -> None:
    query = f"""
        DELETE FROM {cst.GRPO_TASK_FUNCTIONS_TABLE}
        WHERE {cst.TASK_ID} = $1
    """

    async with await psql_db.connection() as connection:
        connection: Connection
        await connection.execute(query, task_id)


async def add_reward_functions(task_id: UUID, reward_functions: list[RewardFunction], psql_db: PSQLDB) -> None:
    reward_functions_query = f"""
        WITH ins AS (
            INSERT INTO {cst.REWARD_FUNCTIONS_TABLE} ({cst.REWARD_FUNC}, {cst.FUNC_HASH}, {cst.IS_GENERIC})
            VALUES ($1, $2, $3)
            ON CONFLICT ({cst.FUNC_HASH}) DO NOTHING
            RETURNING {cst.REWARD_ID}
        )
        SELECT {cst.REWARD_ID} FROM ins
        UNION ALL
        SELECT {cst.REWARD_ID} FROM {cst.REWARD_FUNCTIONS_TABLE} WHERE {cst.FUNC_HASH} = $2
        LIMIT 1
    """
    grpo_task_functions_query = f"""
        INSERT INTO {cst.GRPO_TASK_FUNCTIONS_TABLE} ({cst.TASK_ID}, {cst.REWARD_ID}, {cst.REWARD_WEIGHT})
        VALUES ($1, $2, $3)
        ON CONFLICT ({cst.TASK_ID}, {cst.REWARD_ID}) DO NOTHING
    """

    async with await psql_db.connection() as connection:
        connection: Connection
        for reward_function in reward_functions:
            reward_id = await connection.fetchval(
                reward_functions_query, reward_function.reward_func, reward_function.func_hash, reward_function.is_generic
            )
            await connection.execute(grpo_task_functions_query, task_id, reward_id, reward_function.reward_weight)


async def get_reward_functions(task_id: UUID, psql_db: PSQLDB, connection: Connection | None = None) -> list[RewardFunction]:
    async def _get_reward_functions(conn: Connection) -> list[RewardFunction]:
        query = f"""
            SELECT
                rf.{cst.REWARD_ID},
                rf.{cst.REWARD_FUNC},
                rf.{cst.FUNC_HASH},
                rf.{cst.IS_GENERIC},
                rf.{cst.IS_MANUAL},
                gtf.{cst.REWARD_WEIGHT}
            FROM {cst.REWARD_FUNCTIONS_TABLE} rf
            JOIN {cst.GRPO_TASK_FUNCTIONS_TABLE} gtf ON rf.{cst.REWARD_ID} = gtf.{cst.REWARD_ID}
            WHERE gtf.{cst.TASK_ID} = $1
        """
        rows = await conn.fetch(query, task_id)
        return [
            RewardFunction(
                reward_id=str(row[cst.REWARD_ID]),
                reward_func=row[cst.REWARD_FUNC],
                func_hash=row[cst.FUNC_HASH],
                is_generic=row[cst.IS_GENERIC],
                is_manual=row[cst.IS_MANUAL],
                reward_weight=row[cst.REWARD_WEIGHT],
            )
            for row in rows
        ]

    if connection is not None:
        return await _get_reward_functions(connection)

    async with await psql_db.connection() as connection:
        return await _get_reward_functions(connection)


async def get_model_cache_stats(psql_db: PSQLDB, tau_days: float = 10, max_lookup_days: float = 30) -> dict[str, dict]:
    """Get cache statistics for models with time-weighted frequency calculation.

    Args:
        psql_db: Database connection
        tau_days: Time constant (τ) for exponential decay in days
        max_lookup_days: Maximum number of days to look back for usage data

    Returns:
        Dictionary mapping model_id to stats containing:
        - time_weighted_freq: Time-weighted frequency of model usage
        - size_params: Number of parameters in the model
        - cache_score: Product of frequency and size
    """
    async with await psql_db.connection() as connection:
        connection: Connection
        query = """
            WITH daily_counts AS (
                SELECT
                    model_id,
                    DATE_TRUNC('day', created_at) as usage_date,
                    COUNT(*) as daily_uses,
                    MAX(model_params_count) as params_count
                FROM tasks
                WHERE created_at > NOW() - $2 * INTERVAL '1 day'
                GROUP BY model_id, DATE_TRUNC('day', created_at)
            ),
            model_usage AS (
                SELECT
                    model_id,
                    SUM(
                        daily_uses * exp(
                            -EXTRACT(EPOCH FROM (NOW() - usage_date)) /
                            EXTRACT(EPOCH FROM ($1 * INTERVAL '1 day'))
                        )
                    ) as time_weighted_freq,
                    MAX(params_count) as size_params
                FROM daily_counts
                GROUP BY model_id
            )
            SELECT
                model_id,
                time_weighted_freq,
                size_params,
                time_weighted_freq * size_params as cache_score
            FROM model_usage
            ORDER BY cache_score DESC
        """
        rows = await connection.fetch(query, tau_days, max_lookup_days)

        return {
            str(row["model_id"]): {
                "time_weighted_freq": float(row["time_weighted_freq"] or 0),
                "size_params": int(row["size_params"] or 0),
                "cache_score": float(row["cache_score"] or 0),
            }
            for row in rows
        }


async def get_successful_matching_tasks(
    model_repo: str,
    ds_repo: str,
    field_instruction: str,
    field_input: str,
    field_output: str,
    psql_db: PSQLDB,
    include_tournament_tasks=False,
) -> list[InstructTextTask]:
    """Get most recent successful task with matching model_id and dataset within last 7 days"""
    tournament_tasks_clause = (
        ""
        if include_tournament_tasks
        else f"AND t.{cst.TASK_ID} NOT IN (SELECT {cst.TASK_ID}::uuid FROM {cst.TOURNAMENT_TASKS_TABLE})"
    )

    # Always exclude benchmark tasks
    benchmark_tasks_clause = f"""
        AND t.{cst.TASK_ID} NOT IN (
            SELECT {cst.TASK_ID} FROM {cst.BENCHMARK_ROOT_TASKS_TABLE}
            UNION
            SELECT {cst.COPY_TASK_ID} FROM {cst.BENCHMARK_TASK_COPIES_TABLE}
        )
    """

    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            WITH victorious_repo AS (
                SELECT
                    submissions.{cst.TASK_ID},
                    submissions.{cst.REPO},
                    ROW_NUMBER() OVER (
                        PARTITION BY submissions.{cst.TASK_ID}
                        ORDER BY task_nodes.{cst.QUALITY_SCORE} DESC
                    ) AS rn
                FROM {cst.SUBMISSIONS_TABLE} submissions
                JOIN {cst.TASK_NODES_TABLE} task_nodes
                    ON submissions.{cst.TASK_ID} = task_nodes.{cst.TASK_ID}
                    AND submissions.{cst.HOTKEY} = task_nodes.{cst.HOTKEY}
                WHERE task_nodes.{cst.QUALITY_SCORE} IS NOT NULL
            )
            SELECT t.*, tt.field_system,
                   tt.field_instruction, tt.field_input, tt.field_output,
                   tt.format, tt.no_input_format,
                   COALESCE(t.training_repo_backup, vr.{cst.REPO}) as trained_model_repository
            FROM {cst.TASKS_TABLE} t
            LEFT JOIN {cst.INSTRUCT_TEXT_TASKS_TABLE} tt ON t.{cst.TASK_ID} = tt.{cst.TASK_ID}
            LEFT JOIN victorious_repo vr ON t.{cst.TASK_ID} = vr.{cst.TASK_ID} AND vr.rn = 1
            WHERE t.{cst.MODEL_ID} = $1
            AND t.{cst.DS} = $2
            AND tt.{cst.FIELD_INSTRUCTION} = $3
            AND tt.{cst.FIELD_INPUT} = $4
            AND tt.{cst.FIELD_OUTPUT} = $5
            AND t.{cst.STATUS} = $6
            AND t.{cst.TASK_TYPE} = $7
            AND t.{cst.CREATED_AT} >= NOW() - INTERVAL '7 days'
            {tournament_tasks_clause}
            {benchmark_tasks_clause}
            ORDER BY t.{cst.CREATED_AT} DESC
            LIMIT 100
        """

        rows = await connection.fetch(
            query,
            model_repo,
            ds_repo,
            field_instruction,
            field_input,
            field_output,
            TaskStatus.SUCCESS.value,
            TaskType.INSTRUCTTEXTTASK.value,
        )

        tasks = []
        for row in rows:
            task_data = dict(row)
            task = InstructTextTask(**task_data)
            tasks.append(task)
        return tasks


async def copy_task_for_benchmark(
    original_task: AnyTypeRawTask, psql_db: PSQLDB, new_status: TaskStatus = TaskStatus.PREPARING_DATA
) -> AnyTypeRawTask:
    """
    Create a copy of a task for benchmarking purposes.

    This function creates a deep copy of a task with a new task_id but identical content.
    Synthetic data is set to the test data.
    New status is set to the provided status (default is LOOKING_FOR_NODES)
    Presigned URLs are regenerated to ensure they remain valid.

    Args:
        original_task: The task to copy
        psql_db: Database connection

    Returns:
        The copied task with a new task_id
    """
    try:
        copied_task = original_task.model_copy(deep=True)
        copied_task.task_id = None
        copied_task.status = new_status
        copied_task.created_at = datetime.now()
        copied_task.updated_at = datetime.now()
        copied_task.started_at = None
        copied_task.termination_at = None
        copied_task.completed_at = None

        if copied_task.test_data:
            logger.info("Regenerating presigned URL for test_data")
            copied_task.test_data = await async_minio_client.get_new_presigned_url(copied_task.test_data)

        if copied_task.training_data:
            logger.info("Regenerating presigned URL for training_data")
            copied_task.training_data = await async_minio_client.get_new_presigned_url(copied_task.training_data)

        copied_task = await add_task(copied_task, psql_db)

        logger.info(f"Created benchmark task copy {copied_task.task_id} from original {original_task.task_id}")
        return copied_task

    except Exception as e:
        logger.error(f"Error copying task {original_task.task_id}: {str(e)}", exc_info=True)
        raise


async def get_env_task_eval_seed(task_id: UUID, psql_db: PSQLDB) -> int | None:
    """
    Get the eval_seed for an environment task.
    This is kept separate from normal task fetches to keep eval_seed confidential.
    """
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            SELECT {cst.EVAL_SEED}
            FROM {cst.ENV_TASKS_TABLE}
            WHERE {cst.TASK_ID} = $1
        """
        row = await connection.fetchrow(query, task_id)
        if row:
            return row[cst.EVAL_SEED]
        return None
