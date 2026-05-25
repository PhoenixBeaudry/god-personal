from dotenv import load_dotenv


load_dotenv(".vali.env", override=True)

import asyncio

from validator.infrastructure.cache import cleanup_temp_files
from validator.lifecycle.tasks import process_completed_tasks
from validator.lifecycle.tasks import process_pending_tasks
from validator.shared.config import load_config
from validator.shared.connections import try_db_connections
from validator.shared.refresh_nodes import refresh_nodes_periodically


async def run_validator_lifecycle() -> None:
    config = load_config()
    await try_db_connections(config)

    cleanup_temp_files()

    await asyncio.gather(
        process_pending_tasks(config),
        refresh_nodes_periodically(config),
        process_completed_tasks(config)
    )


if __name__ == "__main__":
    asyncio.run(run_validator_lifecycle())
