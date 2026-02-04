"""LLM judge for Act! API fetch quality - validates question → API call mapping."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from backend.core.llm import create_openai_chain
from backend.eval.act.schema import ACT_API_SCHEMA

# SLO thresholds for fetch quality
SLO_FETCH_CORRECTNESS = 0.85
SLO_FETCH_COMPLETENESS = 0.80


class FetchJudgeResult(BaseModel):
    """Result from the fetch judge."""

    correctness: float = Field(
        description="0-1: Does the API call correctly fetch data needed to answer the question?"
    )
    completeness: float = Field(
        description="0-1: Does it fetch all necessary fields and related entities?"
    )
    appropriate_endpoint: bool = Field(
        description="Is the correct Act! API endpoint being used?"
    )
    appropriate_filters: bool = Field(
        description="Are the query parameters (filters, ordering, limits) sensible?"
    )
    explanation: str = Field(description="Brief reasoning about the fetch strategy")


_SYSTEM_PROMPT = """You are a senior Act! CRM database architect evaluating API fetch strategies.

Your role: Validate that the API call correctly fetches data needed to answer the user's question.

## Act! CRM Web API Schema (discovered from live API)
{schema}

## Known API Behaviors:
- `/api/activities` times out on large databases (230K+ contacts) - use `/api/calendar` instead
- `/api/opportunities` has `contacts[]` array with linked contact objects
- `estimatedCloseDate` (not `closeDate`) for deal close dates
- `weightedTotal` and `productTotal` for opportunity values (not `estimatedValue`)
- History entries with "Field changed" regarding are audit logs, not meaningful activities

## Evaluation Criteria:
1. **Correctness**: Does the API call fetch the right data to answer the question?
2. **Completeness**: Are all necessary related entities fetched (e.g., contacts for opportunities)?
3. **Appropriate Endpoint**: Is the correct endpoint used (e.g., /api/calendar not /api/activities)?
4. **Appropriate Filters**: Are query params ($top, $orderby, $filter) sensible for the question?

Score 0.0-1.0 for correctness and completeness. Be strict - garbage in = garbage out."""

_HUMAN_PROMPT = """## User Question
{question}

## API Call Made
Endpoint: {endpoint}
Parameters: {params}

## Data Returned
{data_summary}

Evaluate: Is this the right API call for this question?"""


def _summarize_data(data: Any) -> str:
    """Create a summary of returned data for the judge."""
    if not data:
        return "No data returned"

    if isinstance(data, dict):
        # Nested structure like {"opportunities": [...], "contacts": [...]}
        parts = []
        for key, value in data.items():
            if isinstance(value, list):
                parts.append(f"{key}: {len(value)} items")
                if value:
                    # Show sample field names
                    sample = value[0] if isinstance(value[0], dict) else {}
                    if sample:
                        fields = list(sample.keys())[:10]
                        parts.append(f"  Fields: {', '.join(fields)}")
            else:
                parts.append(f"{key}: {type(value).__name__}")
        return "\n".join(parts)

    if isinstance(data, list):
        count = len(data)
        if count == 0:
            return "Empty list returned"
        sample = data[0] if isinstance(data[0], dict) else {}
        fields = list(sample.keys())[:15] if sample else []
        return f"{count} items returned\nFields: {', '.join(fields)}"

    return f"Single item: {type(data).__name__}"


def judge_fetch(
    question: str,
    endpoint: str,
    params: dict[str, Any],
    data: Any,
) -> tuple[bool, float, float, str]:
    """
    Judge whether the API fetch correctly addresses the question.

    Args:
        question: The user's question
        endpoint: The API endpoint called (e.g., "/api/opportunities")
        params: Query parameters used
        data: The data returned from the API

    Returns:
        (passed, correctness, completeness, explanation)

    Raises:
        Exception: If the LLM chain fails (caller should handle).
    """
    chain = create_openai_chain(
        system_prompt=_SYSTEM_PROMPT.format(schema=ACT_API_SCHEMA),
        human_prompt=_HUMAN_PROMPT,
        max_tokens=512,
        structured_output=FetchJudgeResult,
        streaming=False,
    )

    result: FetchJudgeResult = chain.invoke({
        "question": question,
        "endpoint": endpoint,
        "params": json.dumps(params, default=str),
        "data_summary": _summarize_data(data),
    })

    passed = (
        result.correctness >= SLO_FETCH_CORRECTNESS
        and result.completeness >= SLO_FETCH_COMPLETENESS
        and result.appropriate_endpoint
    )

    return (
        passed,
        result.correctness,
        result.completeness,
        result.explanation,
    )


__all__ = [
    "FetchJudgeResult",
    "SLO_FETCH_COMPLETENESS",
    "SLO_FETCH_CORRECTNESS",
    "judge_fetch",
]
