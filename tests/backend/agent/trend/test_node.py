"""Tests for trend node."""

import pytest

from backend.agent.trend.node import (
    _detect_granularity,
    _enhance_question_for_trend,
    _calculate_trend_metrics,
)


class TestDetectGranularity:
    """Tests for granularity detection."""

    def test_detects_monthly(self):
        """Should detect monthly granularity."""
        assert _detect_granularity("Show revenue by month") == "monthly"
        assert _detect_granularity("Monthly growth rate") == "monthly"
        assert _detect_granularity("Month over month comparison") == "monthly"

    def test_detects_quarterly(self):
        """Should detect quarterly granularity."""
        assert _detect_granularity("Revenue by quarter") == "quarterly"
        assert _detect_granularity("Quarterly trends") == "quarterly"

    def test_detects_yearly(self):
        """Should detect yearly granularity."""
        assert _detect_granularity("Year over year growth") == "yearly"
        assert _detect_granularity("Annual revenue trend") == "yearly"

    def test_detects_weekly(self):
        """Should detect weekly granularity."""
        assert _detect_granularity("Weekly sales data") == "weekly"
        assert _detect_granularity("By week breakdown") == "weekly"

    def test_defaults_to_monthly(self):
        """Should default to monthly for ambiguous queries."""
        assert _detect_granularity("Show revenue trend") == "monthly"
        assert _detect_granularity("How has pipeline changed") == "monthly"


class TestEnhanceQuestionForTrend:
    """Tests for question enhancement."""

    def test_adds_grouping_instruction(self):
        """Should add grouping when not present."""
        enhanced = _enhance_question_for_trend("Show revenue trend", "monthly")
        assert "group" in enhanced.lower()
        assert "monthly" in enhanced.lower()

    def test_preserves_existing_grouping(self):
        """Should not modify questions with existing group by."""
        original = "Show revenue grouped by month"
        enhanced = _enhance_question_for_trend(original, "monthly")
        assert enhanced == original

    def test_adds_order_instruction(self):
        """Should add order by date."""
        enhanced = _enhance_question_for_trend("Revenue trend", "monthly")
        assert "order" in enhanced.lower() or "ascending" in enhanced.lower()


class TestCalculateTrendMetrics:
    """Tests for trend metric calculation."""

    def test_calculates_direction_increasing(self):
        """Should detect increasing trend."""
        data = [
            {"revenue": 100},
            {"revenue": 150},
            {"revenue": 200},
        ]
        metrics = _calculate_trend_metrics(data)

        assert metrics["columns"]["revenue"]["direction"] == "increasing"
        assert metrics["columns"]["revenue"]["first_value"] == 100
        assert metrics["columns"]["revenue"]["last_value"] == 200

    def test_calculates_direction_decreasing(self):
        """Should detect decreasing trend."""
        data = [
            {"value": 200},
            {"value": 150},
            {"value": 100},
        ]
        metrics = _calculate_trend_metrics(data)

        assert metrics["columns"]["value"]["direction"] == "decreasing"

    def test_calculates_percent_change(self):
        """Should calculate total percent change."""
        data = [
            {"value": 100},
            {"value": 200},
        ]
        metrics = _calculate_trend_metrics(data)

        assert metrics["columns"]["value"]["percent_change"] == 100.0

    def test_calculates_period_changes(self):
        """Should calculate period-over-period changes."""
        data = [
            {"value": 100},
            {"value": 150},  # +50%
            {"value": 225},  # +50%
        ]
        metrics = _calculate_trend_metrics(data)

        assert metrics["columns"]["value"]["period_changes"] == [50.0, 50.0]

    def test_handles_insufficient_data(self):
        """Should handle data with fewer than 2 points."""
        data = [{"value": 100}]
        metrics = _calculate_trend_metrics(data)

        assert "error" in metrics

    def test_calculates_min_max(self):
        """Should calculate min and max values."""
        data = [
            {"value": 150},
            {"value": 100},
            {"value": 200},
        ]
        metrics = _calculate_trend_metrics(data)

        assert metrics["columns"]["value"]["min_value"] == 100
        assert metrics["columns"]["value"]["max_value"] == 200

    def test_calculates_average(self):
        """Should calculate average."""
        data = [
            {"value": 100},
            {"value": 200},
            {"value": 300},
        ]
        metrics = _calculate_trend_metrics(data)

        assert metrics["columns"]["value"]["average"] == 200.0
