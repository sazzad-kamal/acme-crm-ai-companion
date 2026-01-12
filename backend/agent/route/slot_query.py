"""
Slot-based query system for reliable SQL generation.

Instead of LLM generating raw SQL, it outputs structured slots:
- table: which table to query
- filters: list of {field, op, value} conditions
- columns: SELECT columns (optional)
- order_by: ORDER BY clause (optional)

We then build valid SQL programmatically using pypika.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from pydantic import BaseModel, Field
from pypika import Order, Query, Table
from pypika.functions import Lower

logger = logging.getLogger(__name__)

# =============================================================================
# Table Definitions
# =============================================================================

TableName = Literal[
    "opportunities",
    "contacts",
    "activities",
    "companies",
    "history",
]

# SQL columns per table (excludes RAG fields: notes, description)
# Columns match the actual CSV schema in backend/data/csv/
TABLE_COLUMNS: dict[TableName, list[str]] = {
    "opportunities": ["opportunity_id", "company_id", "name", "stage", "value", "owner", "expected_close_date", "type"],
    "contacts": ["contact_id", "company_id", "first_name", "last_name", "email", "phone", "job_title", "role"],
    "activities": ["activity_id", "company_id", "contact_id", "opportunity_id", "type", "subject", "due_datetime"],
    "companies": ["company_id", "name", "status", "plan", "account_owner", "health_flags", "renewal_date"],
    "history": ["history_id", "company_id", "contact_id", "type", "date"],
}


# =============================================================================
# Pydantic Models
# =============================================================================


class Filter(BaseModel):
    """A single filter condition."""

    field: str = Field(description="Column name to filter on")
    op: Literal["eq", "neq", "gt", "lt", "gte", "lte", "in", "not_in", "like"] = Field(
        description="Operator: eq, neq, gt, lt, gte, lte, in, not_in, like"
    )
    value: Any = Field(description="Value to compare against")


class SlotQuery(BaseModel):
    """A single slot-based query."""

    table: TableName = Field(description="Which table to query")
    filters: list[Filter] = Field(
        default_factory=list,
        description="List of filter conditions",
    )
    columns: list[str] | None = Field(
        default=None,
        description="Columns to select. None means all columns",
    )
    order_by: str | None = Field(
        default=None,
        description="ORDER BY clause (e.g., 'value DESC')",
    )


class SlotPlan(BaseModel):
    """LLM output containing slot-based queries."""

    queries: list[SlotQuery] = Field(
        default_factory=list,
        description="List of slot queries to execute",
    )
    needs_rag: bool = Field(
        default=False,
        description="Whether RAG context is needed",
    )


# =============================================================================
# SQL Builder with pypika
# =============================================================================


def _build_criterion(table: Table, f: Filter) -> Any:
    """Build a pypika criterion from a Filter."""
    col = table[f.field]
    val = f.value

    if f.op == "eq":
        # Case-insensitive partial match for string fields like 'name'
        if f.field in ("name", "health_flags") and isinstance(val, str):
            return Lower(col).like(f"%{val.lower()}%")
        return col == val
    elif f.op == "neq":
        return col != val
    elif f.op == "gt":
        return col > val
    elif f.op == "lt":
        return col < val
    elif f.op == "gte":
        return col >= val
    elif f.op == "lte":
        return col <= val
    elif f.op == "in":
        return col.isin(val) if isinstance(val, list) else col == val
    elif f.op == "not_in":
        return col.notin(val) if isinstance(val, list) else col != val
    elif f.op == "like":
        pattern = f"%{val}%" if not val.startswith("%") else val
        return Lower(col).like(pattern.lower())
    else:
        raise ValueError(f"Unknown operator: {f.op}")


def _parse_order_by(order_by: str) -> tuple[str, Order]:
    """Parse 'field DESC' into (field, Order)."""
    parts = order_by.strip().split()
    field = parts[0]
    direction = Order.desc if len(parts) > 1 and parts[1].upper() == "DESC" else Order.asc
    return field, direction


def _build_query(slot: SlotQuery) -> Query:
    """Build a pypika Query from a SlotQuery."""
    table = Table(slot.table)

    # SELECT columns
    cols = slot.columns or TABLE_COLUMNS.get(slot.table, ["*"])
    query = Query.from_(table).select(*[table[c] for c in cols])

    # WHERE filters
    for f in slot.filters:
        query = query.where(_build_criterion(table, f))

    # ORDER BY
    if slot.order_by:
        field, direction = _parse_order_by(slot.order_by)
        query = query.orderby(table[field], order=direction)

    return query


def _get_company_name_filter(filters: list[Filter]) -> tuple[str | None, list[Filter]]:
    """Extract company_name filter from list, return (company_name, remaining_filters)."""
    company_name = None
    remaining = []
    for f in filters:
        if f.field == "company_name":
            company_name = f.value
        else:
            remaining.append(f)
    return company_name, remaining


def _build_query_with_company_join(slot: SlotQuery) -> Query:
    """Build a pypika Query with JOIN to companies table."""
    table = Table(slot.table)
    companies = Table("companies")

    # Extract company_name filter
    company_name, remaining_filters = _get_company_name_filter(slot.filters)

    # SELECT columns - include company name
    if slot.columns:
        cols = [table[c] for c in slot.columns]
    else:
        cols = [table[c] for c in TABLE_COLUMNS.get(slot.table, [])]
        cols.append(companies.name.as_("company_name"))

    # Build query with JOIN
    query = (
        Query.from_(table)
        .join(companies)
        .on(table.company_id == companies.company_id)
        .select(*cols)
    )

    # Add company name filter (case-insensitive partial match)
    if company_name:
        query = query.where(Lower(companies.name).like(f"%{company_name.lower()}%"))

    # Add remaining filters
    for f in remaining_filters:
        query = query.where(_build_criterion(table, f))

    # ORDER BY
    if slot.order_by:
        field, direction = _parse_order_by(slot.order_by)
        query = query.orderby(table[field], order=direction)

    return query


def slot_to_sql(slot: SlotQuery) -> str:
    """
    Convert a SlotQuery to SQL using pypika.

    Handles company name joins automatically when filtering by company_name
    on tables that have company_id.
    """
    # Check if company_name filter is present and not on companies table
    has_company_name = any(f.field == "company_name" for f in slot.filters)
    if has_company_name and slot.table != "companies":
        query = _build_query_with_company_join(slot)
    else:
        query = _build_query(slot)

    sql: str = query.get_sql()
    logger.debug("Built SQL for '%s': %s", slot.table, sql)
    return sql


__all__ = ["Filter", "SlotQuery", "SlotPlan", "slot_to_sql", "TABLE_COLUMNS", "TableName"]
