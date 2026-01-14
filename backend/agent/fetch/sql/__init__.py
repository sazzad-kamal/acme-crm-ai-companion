"""
SQL utilities for the fetch node.

Exports:
    get_connection: Get DuckDB connection
    execute_sql_plan: Execute SQL plan
    SQLPlan: SQL plan model
"""

from backend.agent.fetch.sql.connection import (
    CSV_TABLES,
    close_connection,
    get_connection,
    get_csv_base_path,
    reset_connection,
)
from backend.agent.fetch.sql.executor import (
    SQLExecutionError,
    SQLExecutionStats,
    SQLValidationError,
    execute_sql_plan,
    resolve_placeholders,
    validate_sql,
)

__all__ = [
    "CSV_TABLES",
    "close_connection",
    "get_connection",
    "get_csv_base_path",
    "reset_connection",
    "SQLExecutionError",
    "SQLExecutionStats",
    "SQLValidationError",
    "execute_sql_plan",
    "resolve_placeholders",
    "validate_sql",
]
