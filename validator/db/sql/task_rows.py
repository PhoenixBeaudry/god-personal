
from asyncpg.connection import Connection

from core.logging import get_logger


logger = get_logger(__name__)

async def get_table_fields(table_name: str, connection: Connection) -> set[str]:
    """Get all column names for a given table"""
    query = """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = $1
    """
    rows = await connection.fetch(query, table_name)
    return {row["column_name"] for row in rows}
