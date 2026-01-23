"""LLM judge for SQL semantic equivalence."""

from __future__ import annotations

import logging
from enum import Enum

from pydantic import BaseModel, Field

from backend.core.llm import create_openai_chain

logger = logging.getLogger(__name__)


class ErrorType(str, Enum):
    """Categories of SQL differences."""

    LIKE_VS_EXACT = "like_vs_exact"  # LIKE vs exact match on entity names
    CASE_SENSITIVITY = "case_sensitivity"  # LOWER()/UPPER() vs exact case
    EXTRA_JOIN = "extra_join"  # Added unnecessary JOIN
    MISSING_JOIN = "missing_join"  # Missing a required JOIN from expected
    JOIN_TYPE = "join_type"  # LEFT vs INNER JOIN differences
    JOIN_CARDINALITY = "join_cardinality"  # Row multiplication from joins
    EXTRA_FILTER = "extra_filter"  # Added OR/extra WHERE conditions
    MISSING_FILTER = "missing_filter"  # Missing required filter
    OPERATOR_PRECEDENCE = "operator_precedence"  # AND/OR precedence issues
    GROUPING = "grouping"  # Aggregation differences (often from LEFT vs INNER JOIN)
    ORDER_BY = "order_by"  # ORDER BY differences
    COLUMN_DIFF = "column_diff"  # Extra columns (usually ignorable)
    ALIAS_DIFF = "alias_diff"  # Different table/column aliases (purely syntactic)
    OTHER = "other"  # Other differences


class JudgeError(BaseModel):
    """A single categorized error from the judge."""

    type: ErrorType = Field(description="Category of the difference")
    description: str = Field(description="Detailed explanation of the issue")


class JudgeResult(BaseModel):
    """Structured output from the SQL judge."""

    passed: bool = Field(description="Whether the generated SQL is semantically equivalent to expected SQL")
    errors: list[JudgeError] = Field(default_factory=list, description="Categorized issues found")


_SYSTEM_PROMPT = """Compare two SQL queries. Pass if generated query returns the correct data.

Check (must match):
- Tables and joins
- Filters (same rows selected)
- Aggregations

Ignore (OK to differ):
- Extra columns (superset is fine)
- Column order, filter order
- LIKE vs exact match (if same result)
- Case-insensitive vs exact match (if same result)
- ORDER BY, LIMIT differences
- Aliases, syntax style

If incorrect, categorize each issue with one of these types:
- like_vs_exact: LIKE vs exact match differences on entity names
- case_sensitivity: LOWER()/UPPER() vs exact case comparison
- extra_join: Added unnecessary JOIN not in expected query
- missing_join: Missing a required JOIN that expected query has
- join_type: LEFT JOIN vs INNER JOIN differences
- join_cardinality: Row multiplication from one-to-many joins
- extra_filter: Added OR conditions or extra WHERE filters
- missing_filter: Missing a required filter from expected query
- operator_precedence: AND/OR precedence or missing parentheses
- grouping: Aggregation differences (often from LEFT vs INNER JOIN showing 0 vs hiding rows)
- order_by: ORDER BY clause added or different
- column_diff: Extra columns or different SELECT shape
- alias_diff: Different table or column aliases (c vs co, etc.) - purely syntactic
- other: Any other difference"""

_HUMAN_PROMPT = """## Generated SQL
{generated_sql}

## Expected SQL
{expected_sql}"""


def judge_sql_equivalence(
    generated_sql: str,
    expected_sql: str,
) -> tuple[bool, list[JudgeError]]:
    """LLM judge for SQL semantic equivalence."""
    chain = create_openai_chain(
        system_prompt=_SYSTEM_PROMPT,
        human_prompt=_HUMAN_PROMPT,
        structured_output=JudgeResult,
        streaming=False,
    )

    try:
        result: JudgeResult = chain.invoke({
            "generated_sql": generated_sql,
            "expected_sql": expected_sql,
        })

        logger.debug(f"SQL Judge: passed={result.passed}, errors={result.errors}")
        return result.passed, result.errors

    except Exception as e:
        logger.warning(f"SQL Judge error: {e}")
        return False, [JudgeError(type=ErrorType.OTHER, description=f"Judge API error: {e}")]


__all__ = ["judge_sql_equivalence", "JudgeResult", "JudgeError", "ErrorType"]
