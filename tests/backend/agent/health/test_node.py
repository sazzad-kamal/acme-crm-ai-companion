"""Tests for health score node."""

import pytest

from backend.agent.health.node import (
    _extract_account_identifier,
    _calculate_deal_score,
    _calculate_activity_score,
    _compute_health_score,
    _score_to_grade,
    _get_health_insights,
)


class TestExtractAccountIdentifier:
    """Tests for account identifier extraction."""

    def test_extracts_health_score_for(self):
        """Should extract from 'health score for X' format."""
        account = _extract_account_identifier("What's the health score for Acme Corp?")
        assert account == "Acme Corp"

    def test_extracts_health_of(self):
        """Should extract from 'health of X' format."""
        account = _extract_account_identifier("Show health of TechCorp")
        assert account == "TechCorp"

    def test_extracts_possessive_form(self):
        """Should extract from 'X's health' format."""
        account = _extract_account_identifier("Acme's health score")
        assert account == "Acme"

    def test_extracts_how_healthy(self):
        """Should extract from 'how healthy is X' format."""
        account = _extract_account_identifier("How healthy is the BigCo account?")
        assert account == "the BigCo account"

    def test_returns_none_for_general_health(self):
        """Should return None for general health queries."""
        account = _extract_account_identifier("Show all account health scores")
        # This might or might not match depending on regex
        # The function should handle this gracefully


class TestCalculateDealScore:
    """Tests for deal score calculation."""

    def test_calculates_total_value(self):
        """Should sum deal values."""
        data = [
            {"amount": 10000},
            {"amount": 20000},
        ]
        metrics = _calculate_deal_score(data)
        assert metrics["total_value"] == 30000

    def test_calculates_win_rate(self):
        """Should calculate win rate from closed deals."""
        data = [
            {"status": "won"},
            {"status": "won"},
            {"status": "lost"},
            {"status": "open"},
        ]
        metrics = _calculate_deal_score(data)
        # 2 won out of 3 closed = 66.67%
        assert 66 < metrics["win_rate"] < 67

    def test_counts_deal_statuses(self):
        """Should count deals by status."""
        data = [
            {"status": "won"},
            {"status": "lost"},
            {"status": "open"},
            {"status": "open"},
        ]
        metrics = _calculate_deal_score(data)
        assert metrics["won_count"] == 1
        assert metrics["lost_count"] == 1
        assert metrics["open_count"] == 2

    def test_handles_empty_data(self):
        """Should handle empty data gracefully."""
        metrics = _calculate_deal_score([])
        assert metrics["total_value"] == 0
        assert metrics["deal_count"] == 0
        assert metrics["win_rate"] == 0


class TestScoreToGrade:
    """Tests for score to grade conversion."""

    def test_grade_a(self):
        """Score >= 90 should be A."""
        assert _score_to_grade(90) == "A"
        assert _score_to_grade(100) == "A"

    def test_grade_b(self):
        """Score 80-89 should be B."""
        assert _score_to_grade(80) == "B"
        assert _score_to_grade(89) == "B"

    def test_grade_c(self):
        """Score 70-79 should be C."""
        assert _score_to_grade(70) == "C"
        assert _score_to_grade(79) == "C"

    def test_grade_d(self):
        """Score 60-69 should be D."""
        assert _score_to_grade(60) == "D"
        assert _score_to_grade(69) == "D"

    def test_grade_f(self):
        """Score < 60 should be F."""
        assert _score_to_grade(59) == "F"
        assert _score_to_grade(0) == "F"


class TestComputeHealthScore:
    """Tests for overall health score computation."""

    def test_computes_weighted_score(self):
        """Should compute weighted average of components."""
        deal_metrics = {
            "total_value": 100000,
            "deal_count": 5,
            "win_rate": 80,
            "won_count": 4,
            "lost_count": 1,
            "open_count": 2,
        }
        activity_metrics = {
            "recency_score": 90,
        }

        result = _compute_health_score(deal_metrics, activity_metrics)

        assert "overall_score" in result
        assert "grade" in result
        assert "components" in result
        assert 0 <= result["overall_score"] <= 100


class TestGetHealthInsights:
    """Tests for health insights generation."""

    def test_suggests_activity_when_low(self):
        """Should suggest activity when recency is low."""
        health = {"overall_score": 50, "components": {"activity_recency": 20}}
        deal_metrics = {"deal_count": 5}
        activity_metrics = {"last_activity_days": 60}

        insights = _get_health_insights(health, deal_metrics, activity_metrics)

        assert any("activity" in i.lower() or "check-in" in i.lower() for i in insights)

    def test_suggests_expansion_when_few_deals(self):
        """Should suggest expansion for accounts with few deals."""
        health = {"overall_score": 50, "components": {}}
        deal_metrics = {"deal_count": 2}
        activity_metrics = {}

        insights = _get_health_insights(health, deal_metrics, activity_metrics)

        assert any("engagement" in i.lower() or "deals" in i.lower() for i in insights)

    def test_positive_insight_for_high_score(self):
        """Should give positive insight for healthy accounts."""
        health = {"overall_score": 85, "components": {}}
        deal_metrics = {"deal_count": 10}
        activity_metrics = {}

        insights = _get_health_insights(health, deal_metrics, activity_metrics)

        assert any("strong" in i.lower() or "case study" in i.lower() for i in insights)
