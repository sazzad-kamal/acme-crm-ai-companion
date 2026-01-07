"""
CRM Data Store base functionality.

Provides base class with DuckDB connection, table loading, and query helpers.
"""

from __future__ import annotations

import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import duckdb

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection


# =============================================================================
# Protocol for Mixin Type Checking
# =============================================================================


class DataStoreMixinProtocol(Protocol):
    """Protocol defining what mixins can access from the base class."""

    @property
    def conn(self) -> DuckDBPyConnection: ...
    def _ensure_table(self, table_name: str) -> bool: ...
    def _fetch_one_dict(self, query: str, params: list[Any] | None = None) -> dict[str, Any] | None: ...
    def _fetch_all_dicts(self, query: str, params: list[Any] | None = None) -> list[dict[str, Any]]: ...
    def _get_date_cutoff(self, days: int) -> str: ...
    def _build_company_cache(self) -> None: ...
    _company_names_cache: dict[str, str] | None
    _company_ids_cache: set[str] | None


# =============================================================================
# Configuration
# =============================================================================

CSV_TABLES = {
    "companies": "companies.csv",
    "contacts": "contacts.csv",
    "activities": "activities.csv",
    "history": "history.csv",
    "opportunities": "opportunities.csv",
    "groups": "groups.csv",
    "group_members": "group_members.csv",
    "attachments": "attachments.csv",
}

REQUIRED_TABLES = {"companies", "contacts", "activities", "history", "opportunities"}


def get_csv_base_path() -> Path:
    """
    Get the base path for CSV files with fallback logic.

    Priority:
    1. data/crm/ (if exists)
    2. data/csv/ (fallback)
    3. Raise error if neither exists
    """
    backend_root = Path(__file__).parent.parent.parent

    preferred = backend_root / "data" / "crm"
    if preferred.exists() and preferred.is_dir():
        return preferred

    # Fallback to csv directory (always exists in project)
    return backend_root / "data" / "csv"


# =============================================================================
# Base Data Store
# =============================================================================


class CRMDataStoreBase:
    """
    DuckDB-based CRM data store base with lazy loading.

    Provides connection management, table loading, and query helpers.
    Extended by domain-specific mixins.
    """

    def __init__(self, csv_path: Path | None = None) -> None:
        self._csv_path = csv_path
        self._conn: duckdb.DuckDBPyConnection | None = None
        self._loaded_tables: set[str] = set()
        self._company_names_cache: dict[str, str] | None = None
        self._company_ids_cache: set[str] | None = None

    def __enter__(self) -> CRMDataStoreBase:
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any) -> None:
        self.close()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
        self._loaded_tables.clear()
        self._company_names_cache = None
        self._company_ids_cache = None

    @property
    def csv_path(self) -> Path:
        if self._csv_path is None:
            self._csv_path = get_csv_base_path()
        return self._csv_path

    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        if self._conn is None:
            self._conn = duckdb.connect(":memory:")
        return self._conn

    def _ensure_table(self, table_name: str) -> bool:
        if table_name in self._loaded_tables:
            return True

        filename = CSV_TABLES[table_name]  # All callers use valid table names
        csv_file = self.csv_path / filename

        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} AS
            SELECT * FROM read_csv_auto('{csv_file.as_posix()}')
        """)
        self._loaded_tables.add(table_name)
        return True

    def _fetch_one_dict(self, query: str, params: list[Any] | None = None) -> dict[str, Any] | None:
        result = self.conn.execute(query, params or []).fetchone()
        if not result:
            return None
        return dict(zip([d[0] for d in self.conn.description], result, strict=True))

    def _fetch_all_dicts(self, query: str, params: list[Any] | None = None) -> list[dict[str, Any]]:
        result = self.conn.execute(query, params or []).fetchall()
        if not result:
            return []
        columns = [d[0] for d in self.conn.description]
        return [dict(zip(columns, row, strict=True)) for row in result]

    def _ensure_core_tables(self) -> None:
        for table in REQUIRED_TABLES:
            self._ensure_table(table)

    def _build_company_cache(self) -> None:
        if self._company_names_cache is not None:
            return

        self._ensure_table("companies")
        result = self.conn.execute("SELECT company_id, name FROM companies").fetchall()
        self._company_names_cache = {name.lower(): cid for cid, name in result}
        self._company_ids_cache = {cid for cid, _ in result}

    def _get_date_cutoff(self, days: int) -> str:
        cutoff = datetime.now() - timedelta(days=days)
        return cutoff.isoformat()


# =============================================================================
# Thread-local datastore instance
# =============================================================================

_thread_local = threading.local()


def _get_datastore_instance(datastore_class: type) -> Any:
    if not hasattr(_thread_local, "datastore"):
        _thread_local.datastore = datastore_class()
    return _thread_local.datastore
