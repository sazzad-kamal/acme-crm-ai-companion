"""
Minimal DuckDB connection manager with CSV loading.

Replaces the complex CRMDataStore class with direct SQL execution.
Creates views matching the schema in prompt.txt (excludes notes columns).
"""

import logging
import threading
from pathlib import Path

import duckdb

logger = logging.getLogger(__name__)

# CSV tables to load
_CSV_TABLES = ["companies", "contacts", "activities", "history", "opportunities"]

# Schema columns for each table (matches prompt.txt, excludes notes)
_TABLE_COLUMNS = {
    "companies": [
        "company_id", "name", "status", "plan", "account_owner",
        "industry", "segment", "region", "renewal_date", "health_flags"
    ],
    "contacts": [
        "contact_id", "company_id", "first_name", "last_name", "email",
        "phone", "job_title", "role", "lifecycle_stage"
    ],
    "opportunities": [
        "opportunity_id", "company_id", "primary_contact_id", "name", "stage",
        "type", "value", "owner", "expected_close_date", "days_in_stage"
    ],
    "activities": [
        "activity_id", "company_id", "contact_id", "opportunity_id", "type",
        "subject", "due_datetime", "owner", "priority", "status"
    ],
    "history": [
        "history_id", "company_id", "contact_id", "opportunity_id", "type",
        "subject", "source", "occurred_at", "owner"
    ],
}


def _get_csv_base_path() -> Path:
    """Get the base path for CSV files (data/crm/ or data/csv/)."""
    backend_root = Path(__file__).parent.parent.parent.parent
    preferred = backend_root / "data" / "crm"
    if preferred.exists() and preferred.is_dir():
        return preferred
    return backend_root / "data" / "csv"


def _load_csvs(conn: duckdb.DuckDBPyConnection, csv_path: Path) -> None:
    """Load all CSV files into DuckDB and create views with schema columns only."""
    for table in _CSV_TABLES:
        csv_file = csv_path / f"{table}.csv"
        if csv_file.exists():
            # Load full CSV into raw table
            conn.execute(
                f"CREATE TABLE IF NOT EXISTS {table}_raw AS "
                f"SELECT * FROM read_csv_auto('{csv_file.as_posix()}')"
            )
            # Create view with only schema columns (excludes notes)
            columns = ", ".join(_TABLE_COLUMNS[table])
            conn.execute(
                f"CREATE VIEW IF NOT EXISTS {table} AS "
                f"SELECT {columns} FROM {table}_raw"
            )
            logger.debug(f"Loaded table '{table}' from {csv_file}")
        else:
            logger.warning(f"CSV file not found: {csv_file}")


# Thread-local connection storage
_thread_local = threading.local()


def get_connection(csv_path: Path | None = None) -> duckdb.DuckDBPyConnection:
    """
    Get a thread-local DuckDB connection with CSV tables loaded.

    The connection is created once per thread and reused.
    """
    if not hasattr(_thread_local, "conn") or _thread_local.conn is None:
        _thread_local.conn = duckdb.connect(":memory:")
        _load_csvs(_thread_local.conn, csv_path or _get_csv_base_path())
        logger.debug("Created new DuckDB connection with CSV tables")
    conn: duckdb.DuckDBPyConnection = _thread_local.conn
    return conn


def reset_connection() -> None:
    """Reset the thread-local connection (for testing)."""
    if hasattr(_thread_local, "conn") and _thread_local.conn is not None:
        _thread_local.conn.close()
        _thread_local.conn = None


__all__ = ["get_connection", "reset_connection"]
