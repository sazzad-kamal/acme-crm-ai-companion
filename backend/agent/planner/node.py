"""Planner node - orchestrates complex multi-step queries."""

import json
import logging
import os
import re
from typing import Any, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from backend.agent.compare.node import compare_node
from backend.agent.fetch.node import fetch_node
from backend.agent.state import AgentState
from backend.agent.trend.node import trend_node

logger = logging.getLogger(__name__)

_MOCK_LLM = os.getenv("MOCK_LLM", "false").lower() == "1"


# System prompt for query decomposition
_DECOMPOSITION_PROMPT = """You are a query planner for a CRM data system. Your job is to break down complex questions into simpler sub-queries.

Available query types:
- "fetch": Simple data retrieval (e.g., "Get all deals", "Show contacts for Acme")
- "compare": Comparison queries (e.g., "Compare Q1 vs Q2 revenue")
- "trend": Time-series analysis (e.g., "Show revenue trend by month")

Given a complex question, return a JSON array of sub-queries to execute. Each sub-query should have:
- "type": One of "fetch", "compare", "trend"
- "query": The simplified sub-query text
- "depends_on": Optional index of a previous query this depends on (for sequential execution)

Examples:

Question: "Show all deals and compare Q1 vs Q2 revenue"
[
  {"type": "fetch", "query": "Show all deals"},
  {"type": "compare", "query": "Compare Q1 vs Q2 revenue"}
]

Question: "What is our total pipeline value and how has it grown over time?"
[
  {"type": "fetch", "query": "What is our total pipeline value"},
  {"type": "trend", "query": "Show pipeline value growth over time"}
]

Question: "Show me Acme's deals and compare their performance to last year"
[
  {"type": "fetch", "query": "Show me Acme's deals"},
  {"type": "compare", "query": "Compare Acme's deals this year vs last year"}
]

Return ONLY valid JSON array. No explanations.
"""


def _decompose_query(question: str) -> list[dict[str, Any]]:
    """Use LLM to decompose a complex query into sub-queries."""
    if _MOCK_LLM:
        # Return simple fetch for tests
        return [{"type": "fetch", "query": question}]

    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        messages = [
            SystemMessage(content=_DECOMPOSITION_PROMPT),
            HumanMessage(content=f"Question: {question}"),
        ]

        response = llm.invoke(messages)
        content = response.content

        # Extract JSON from response
        if isinstance(content, str):
            # Try to find JSON array in response
            json_match = re.search(r"\[.*\]", content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())

            # If no array found, treat as simple fetch
            return [{"type": "fetch", "query": question}]

        return [{"type": "fetch", "query": question}]

    except Exception as e:
        logger.error(f"[Planner] Query decomposition failed: {e}")
        return [{"type": "fetch", "query": question}]


def _execute_subquery(
    subquery: dict[str, Any],
    base_state: AgentState,
    previous_results: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute a single sub-query and return results."""
    query_type = subquery.get("type", "fetch")
    query_text = subquery.get("query", "")

    # Create state for sub-query
    sub_state = cast(
        AgentState,
        {
            "question": query_text,
            "messages": base_state.get("messages", []),
        },
    )

    # Add context from previous results if available
    if previous_results:
        # Could enhance the query with context from previous results
        pass

    # Route to appropriate agent
    if query_type == "compare":
        result = compare_node(sub_state)
    elif query_type == "trend":
        result = trend_node(sub_state)
    else:
        result = fetch_node(sub_state)

    return {
        "type": query_type,
        "query": query_text,
        "result": result.get("sql_results", {}),
        "error": result.get("error"),
    }


def _aggregate_results(subquery_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate results from multiple sub-queries."""
    aggregated: dict[str, Any] = {
        "subqueries": [],
        "data": {},
        "comparisons": [],
        "trends": [],
    }

    for i, result in enumerate(subquery_results):
        query_info = {
            "index": i,
            "type": result["type"],
            "query": result["query"],
            "has_data": bool(result.get("result", {}).get("data")),
            "error": result.get("error"),
        }
        aggregated["subqueries"].append(query_info)

        # Aggregate data by type
        sql_results = result.get("result", {})

        if result["type"] == "fetch" and sql_results.get("data"):
            aggregated["data"][f"fetch_{i}"] = sql_results["data"]

        elif result["type"] == "compare" and sql_results.get("comparison"):
            aggregated["comparisons"].append(sql_results["comparison"])

        elif result["type"] == "trend" and sql_results.get("trend_analysis"):
            aggregated["trends"].append({
                "data": sql_results.get("data", []),
                "analysis": sql_results["trend_analysis"],
            })

    return aggregated


def planner_node(state: AgentState) -> AgentState:
    """Planner node that orchestrates complex multi-step queries."""
    question = state["question"]
    logger.info(f"[Planner] Processing: {question[:50]}...")

    result: dict[str, Any] = {
        "sql_results": {},
    }

    # Step 1: Decompose the query into sub-queries
    subqueries = _decompose_query(question)
    logger.info(f"[Planner] Decomposed into {len(subqueries)} sub-queries")

    # Step 2: Execute sub-queries (respecting dependencies)
    subquery_results: list[dict[str, Any]] = []
    previous_results: dict[str, Any] | None = None

    for i, subquery in enumerate(subqueries):
        depends_on = subquery.get("depends_on")

        # Get previous results if this query depends on another
        if depends_on is not None and depends_on < len(subquery_results):
            previous_results = subquery_results[depends_on].get("result")

        logger.info(f"[Planner] Executing subquery {i + 1}: {subquery.get('type')} - {subquery.get('query', '')[:40]}...")
        subquery_result = _execute_subquery(subquery, state, previous_results)
        subquery_results.append(subquery_result)

    # Step 3: Aggregate results
    aggregated = _aggregate_results(subquery_results)

    # Build final result
    sql_results: dict[str, Any] = {
        "_debug": {
            "subquery_count": len(subqueries),
            "subqueries": [sq.get("query", "") for sq in subqueries],
        },
        "aggregated": aggregated,
    }

    # Set data for downstream nodes (use first available data)
    if aggregated.get("data"):
        first_data_key = list(aggregated["data"].keys())[0]
        sql_results["data"] = aggregated["data"][first_data_key]

    result["sql_results"] = sql_results
    logger.info(f"[Planner] Complete: {len(subquery_results)} results aggregated")

    return cast(AgentState, result)


__all__ = ["planner_node"]
