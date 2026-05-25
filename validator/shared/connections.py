
import httpx
from tenacity import retry
from tenacity import retry_if_exception_type
from tenacity import stop_after_attempt
from tenacity import wait_exponential

from core.logging import get_logger
from validator.shared.config import Config


logger = get_logger(__name__)

retry_http_with_backoff = retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.RequestError)),
    reraise=True,
)

retry_with_backoff = retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True,
)

retry_http_fast = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.2, min=0.1, max=1),
    retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.RequestError)),
    reraise=True,
)

async def try_db_connections(config: Config) -> None:
    logger.info("Attempting to connect to PostgreSQL...")
    await config.psql_db.connect()
    await config.psql_db.pool.execute("SELECT 1=1 as one")
    logger.info("PostgreSQL connected successfully")

    logger.info("Attempting to connect to Redis")
    await config.redis_db.ping()
    logger.info("Redis connected successfully")
