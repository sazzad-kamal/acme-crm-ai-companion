"""Tests for planner node."""

import os
import pytest

# Set mock mode for tests
os.environ["MOCK_LLM"] = "1"

from backend.agent.planner.node import (
    _decompose_query,
    _aggregate_results,
)


class TestDecomposeQuery:
    """Tests for query decomposition."""

    def test_mock_mode_returns_simple_fetch(self):
        """In mock mode, should return simple fetch query."""
        result = _decompose_query("Show all deals and compare Q1 vs Q2")

        assert len(result) == 1
        assert result[0]["type"] == "fetch"

    def test_returns_list_of_subqueries(self):
        """Should return a list of subquery dictionaries."""
        result = _decompose_query("Show all deals")

        assert isinstance(result, list)
        assert all(isinstance(q, dict) for q in result)
        assert all("type" in q and "query" in q for q in result)


class TestAggregateResults:
    """Tests for result aggregation."""

    def test_aggregates_fetch_results(self):
        """Should aggregate fetch results into data dict."""
        subquery_results = [
            {
                "type": "fetch",
                "query": "Show deals",
                "result": {"data": [{"id": 1}, {"id": 2}]},
            },
        ]

        aggregated = _aggregate_results(subquery_results)

        assert "data" in aggregated
        assert "fetch_0" in aggregated["data"]
        assert len(aggregated["data"]["fetch_0"]) == 2

    def test_aggregates_comparison_results(self):
        """Should aggregate comparison results into comparisons list."""
        subquery_results = [
            {
                "type": "compare",
                "query": "Compare Q1 vs Q2",
                "result": {
                    "comparison": {
                        "entity_a": "Q1",
                        "entity_b": "Q2",
                        "metrics": {"revenue": {"difference": 1000}},
                    }
                },
            },
        ]

        aggregated = _aggregate_results(subquery_results)

        assert "comparisons" in aggregated
        assert len(aggregated["comparisons"]) == 1
        assert aggregated["comparisons"][0]["entity_a"] == "Q1"

    def test_aggregates_trend_results(self):
        """Should aggregate trend results into trends list."""
        subquery_results = [
            {
                "type": "trend",
                "query": "Show revenue trend",
                "result": {
                    "data": [{"month": 1, "revenue": 100}],
                    "trend_analysis": {"direction": "increasing"},
                },
            },
        ]

        aggregated = _aggregate_results(subquery_results)

        assert "trends" in aggregated
        assert len(aggregated["trends"]) == 1
        assert aggregated["trends"][0]["analysis"]["direction"] == "increasing"

    def test_tracks_subquery_metadata(self):
        """Should track subquery info in subqueries list."""
        subquery_results = [
            {"type": "fetch", "query": "Get deals", "result": {"data": []}},
            {"type": "compare", "query": "Compare A vs B", "result": {}},
        ]

        aggregated = _aggregate_results(subquery_results)

        assert len(aggregated["subqueries"]) == 2
        assert aggregated["subqueries"][0]["type"] == "fetch"
        assert aggregated["subqueries"][1]["type"] == "compare"

    def test_handles_errors_in_results(self):
        """Should track errors in subquery metadata."""
        subquery_results = [
            {
                "type": "fetch",
                "query": "Get invalid data",
                "result": {},
                "error": "SQL execution failed",
            },
        ]

        aggregated = _aggregate_results(subquery_results)

        assert aggregated["subqueries"][0]["error"] == "SQL execution failed"
