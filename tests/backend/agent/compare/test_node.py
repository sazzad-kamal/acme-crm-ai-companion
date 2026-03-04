"""Tests for compare node."""

import pytest

from backend.agent.compare.node import (
    _extract_comparison_entities,
    _calculate_comparison,
    _is_time_period,
)


class TestExtractComparisonEntities:
    """Tests for entity extraction from comparison queries."""

    def test_extracts_vs_comparison(self):
        """Should extract entities from 'X vs Y' format."""
        a, b = _extract_comparison_entities("Compare Q1 vs Q2 revenue")
        assert a == "q1"
        assert b == "q2 revenue"

    def test_extracts_versus_comparison(self):
        """Should extract entities from 'X versus Y' format."""
        a, b = _extract_comparison_entities("Q1 versus Q2")
        assert a == "q1"
        assert b == "q2"

    def test_extracts_difference_between(self):
        """Should extract from 'difference between X and Y' format."""
        a, b = _extract_comparison_entities("What's the difference between Acme and TechCorp")
        assert a == "acme"
        assert b == "techcorp"

    def test_extracts_compared_to(self):
        """Should extract from 'X compared to Y' format."""
        a, b = _extract_comparison_entities("Revenue compared to last year")
        assert a == "revenue"
        assert b == "last year"

    def test_returns_none_for_non_comparison(self):
        """Should return None for non-comparison queries."""
        a, b = _extract_comparison_entities("Show me all deals")
        assert a is None
        assert b is None


class TestIsTimePeriod:
    """Tests for time period detection."""

    def test_detects_quarters(self):
        """Should detect quarter references."""
        assert _is_time_period("Q1") is True
        assert _is_time_period("q2 revenue") is True

    def test_detects_year(self):
        """Should detect year references."""
        assert _is_time_period("2024") is True
        assert _is_time_period("last year") is True

    def test_detects_months(self):
        """Should detect month references."""
        assert _is_time_period("January") is True
        assert _is_time_period("by month") is True

    def test_non_time_returns_false(self):
        """Should return False for non-time entities."""
        assert _is_time_period("Acme Corp") is False
        assert _is_time_period("revenue") is False


class TestCalculateComparison:
    """Tests for comparison calculation."""

    def test_calculates_numeric_differences(self):
        """Should calculate differences for numeric columns."""
        data_a = [{"revenue": 100}, {"revenue": 200}]
        data_b = [{"revenue": 150}, {"revenue": 250}]

        result = _calculate_comparison(data_a, data_b, "Q1", "Q2")

        assert result["entity_a"] == "Q1"
        assert result["entity_b"] == "Q2"
        assert "revenue" in result["metrics"]
        assert result["metrics"]["revenue"]["Q1"] == 300  # sum of a
        assert result["metrics"]["revenue"]["Q2"] == 400  # sum of b
        assert result["metrics"]["revenue"]["difference"] == 100

    def test_calculates_percent_change(self):
        """Should calculate percentage change."""
        data_a = [{"value": 100}]
        data_b = [{"value": 150}]

        result = _calculate_comparison(data_a, data_b, "before", "after")

        assert result["metrics"]["value"]["percent_change"] == 50.0

    def test_handles_empty_data(self):
        """Should handle empty data sets."""
        result = _calculate_comparison([], [], "A", "B")

        assert result["entity_a"] == "A"
        assert result["entity_b"] == "B"
        assert result["metrics"]["_row_count"]["A"] == 0
        assert result["metrics"]["_row_count"]["B"] == 0

    def test_includes_row_counts(self):
        """Should include row count comparison."""
        data_a = [{"id": 1}, {"id": 2}]
        data_b = [{"id": 1}, {"id": 2}, {"id": 3}]

        result = _calculate_comparison(data_a, data_b, "A", "B")

        assert result["metrics"]["_row_count"]["A"] == 2
        assert result["metrics"]["_row_count"]["B"] == 3
        assert result["metrics"]["_row_count"]["difference"] == 1
