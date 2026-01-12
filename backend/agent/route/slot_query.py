"""
Slot-based query system for reliable SQL generation.

Instead of LLM generating raw SQL, it outputs structured slots:
- table: which table to query
- filters: list of {field, op, value} conditions
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
# Pydantic Models
# =============================================================================
class Filter(BaseModel):
    """A single filter condition."""

    field: str = Field(description="Column name to filter on")
    op: Literal["=", "!=", ">", "<", ">=", "<=", "IN", "NOT IN", "LIKE"] = Field(
        description="SQL operator"
    )
    value: Any = Field(description="Value to compare against")

class SlotQuery(BaseModel):
    """A single slot-based query."""

    table: str = Field(description="Which table to query")
    filters: list[Filter] = Field(
        default_factory=list,
        description="List of filter conditions",
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
    col, val = table[f.field], f.value

    match f.op:
        case "=": return col == val
        case "!=": return col != val
        case ">": return col > val
        case "<": return col < val
        case ">=": return col >= val
        case "<=": return col <= val
        case "IN": return col.isin(val) if isinstance(val, list) else col == val
        case "NOT IN": return col.notin(val) if isinstance(val, list) else col != val
        case "LIKE": return Lower(col).like(f"%{val.lower()}%")
        case _: raise ValueError(f"Unknown operator: {f.op}")


def _build_query(slot: SlotQuery) -> Query:
    """Build a pypika Query from a SlotQuery."""
    table = Table(slot.table)
    query = Query.from_(table).select("*")

    for f in slot.filters:
        query = query.where(_build_criterion(table, f))

    if slot.order_by:
        field, _, dir = slot.order_by.partition(" ")
        query = query.orderby(table[field], order=Order.desc if dir.upper() == "DESC" else Order.asc)

    return query


def slot_to_sql(slot: SlotQuery) -> str:
    """Convert a SlotQuery to SQL using pypika."""
    query = _build_query(slot)
    sql: str = query.get_sql()
    logger.debug("Built SQL for '%s': %s", slot.table, sql)
    return sql


__all__ = ["Filter", "SlotQuery", "SlotPlan", "slot_to_sql"]
