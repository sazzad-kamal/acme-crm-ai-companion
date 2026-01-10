"""
CRM Data Store package.

Provides minimal DuckDB connection management with CSV loading.
"""

from backend.agent.datastore.connection import (
    CSV_TABLES,
    close_connection,
    get_connection,
    get_csv_base_path,
    reset_connection,
)

__all__ = [
    "get_connection",
    "reset_connection",
    "close_connection",
    "get_csv_base_path",
    "CSV_TABLES",
]
