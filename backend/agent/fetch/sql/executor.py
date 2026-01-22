"""SQL query executor - executes SQL queries against DuckDB."""

import logging
from typing import Any

import duckdb

logger = logging.getLogger(__name__)


def execute_sql(
    sql: str,
    conn: duckdb.DuckDBPyConnection,
) -> tuple[list[dict[str, Any]], str | None]:
    """Execute SQL query against DuckDB.

    Args:
        sql: SQL query string
        conn: DuckDB connection

    Returns:
        Tuple of (rows, error_msg)
    """
    try:
        result = conn.execute(sql)
        cols = [d[0] for d in result.description]
        rows = [dict(zip(cols, r, strict=True)) for r in result.fetchall()]
        return rows, None

    except Exception as e:
        logger.error(f"SQL execution failed: {e}")
        return [], str(e)


__all__ = ["execute_sql"]
