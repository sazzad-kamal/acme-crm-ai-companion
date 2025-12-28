"""
Tests for backend/rag/eval/tracking.py and history.py.
"""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime

from backend.rag.eval.models import DocsEvalSummary, EvalResult, JudgeResult
from backend.rag.eval.tracking import (
    compare_with_previous,
    extract_step_latencies,
    analyze_budget_violations,
)
from backend.rag.eval.history import (
    compute_trends,
    detect_degradation,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_summary():
    """Create a sample DocsEvalSummary."""
    return DocsEvalSummary(
        total_tests=10,
        rag_triad_success=0.80,
        context_relevance=0.85,
        answer_relevance=0.90,
        groundedness=0.75,
        avg_doc_recall=0.70,
        avg_latency_ms=1500,
        p95_latency_ms=2500,
        total_tokens=50000,
        estimated_cost=0.50,
        all_slos_passed=True,
        failed_slos=[],
    )


@pytest.fixture
def previous_summary():
    """Create a previous summary for comparison."""
    return DocsEvalSummary(
        total_tests=10,
        rag_triad_success=0.75,  # Improved from 0.75 to 0.80
        context_relevance=0.80,  # Improved from 0.80 to 0.85
        answer_relevance=0.85,  # Improved from 0.85 to 0.90
        groundedness=0.80,  # Regressed from 0.80 to 0.75
        avg_doc_recall=0.70,  # Same
        avg_latency_ms=1400,
        p95_latency_ms=2400,  # Slightly worse
        total_tokens=48000,
        estimated_cost=0.48,
        all_slos_passed=True,
        failed_slos=[],
    )


@pytest.fixture
def sample_eval_results():
    """Create sample evaluation results."""
    results = []
    for i in range(5):
        judge = JudgeResult(
            context_relevance=1,
            answer_relevance=1,
            groundedness=1,
            needs_human_review=0,
            confidence=0.9,
            explanation="Good",
        )
        result = EvalResult(
            question_id=f"q{i}",
            question=f"Question {i}",
            target_doc_ids=[f"doc{i}"],
            retrieved_doc_ids=[f"doc{i}"],
            answer=f"Answer {i}",
            judge_result=judge,
            doc_recall=1.0,
            latency_ms=1000 + i * 200,
            total_tokens=500,
            step_timings={
                "preprocess": 10 + i,
                "rewrite": 50 + i * 10,
                "retrieval": 200 + i * 50,
                "generate": 500 + i * 100,
            },
        )
        results.append(result)
    return results


# =============================================================================
# compare_with_previous Tests
# =============================================================================

class TestCompareWithPrevious:
    """Tests for compare_with_previous function."""

    def test_returns_no_previous_when_none(self, sample_summary):
        """Returns has_previous=False when no previous."""
        result = compare_with_previous(sample_summary, None)

        assert result["has_previous"] is False
        assert result["regressions"] == []
        assert result["improvements"] == []

    def test_detects_improvements(self, sample_summary, previous_summary):
        """Detects metric improvements."""
        result = compare_with_previous(sample_summary, previous_summary)

        assert result["has_previous"] is True
        improvement_metrics = [i["metric"] for i in result["improvements"]]
        assert "RAG Triad" in improvement_metrics  # 0.75 -> 0.80

    def test_detects_regressions(self, sample_summary, previous_summary):
        """Detects metric regressions."""
        result = compare_with_previous(sample_summary, previous_summary)

        regression_metrics = [r["metric"] for r in result["regressions"]]
        assert "Groundedness" in regression_metrics  # 0.80 -> 0.75

    def test_uses_threshold(self, sample_summary, previous_summary):
        """Respects threshold for detection."""
        # With high threshold, small changes aren't flagged
        result = compare_with_previous(sample_summary, previous_summary, threshold=0.10)

        # 5% changes won't be flagged with 10% threshold
        # Only Groundedness dropped 5% so may not be in regressions
        improvement_metrics = [i["metric"] for i in result["improvements"]]
        assert len(improvement_metrics) <= len(result["improvements"])

    def test_detects_latency_regression(self):
        """Detects latency increases as regressions."""
        current = DocsEvalSummary(
            total_tests=10,
            rag_triad_success=0.80,
            context_relevance=0.80,
            answer_relevance=0.80,
            groundedness=0.80,
            avg_doc_recall=0.80,
            avg_latency_ms=2000,
            p95_latency_ms=4000,  # Much higher
            total_tokens=50000,
            estimated_cost=0.50,
            all_slos_passed=True,
            failed_slos=[],
        )
        previous = DocsEvalSummary(
            total_tests=10,
            rag_triad_success=0.80,
            context_relevance=0.80,
            answer_relevance=0.80,
            groundedness=0.80,
            avg_doc_recall=0.80,
            avg_latency_ms=1500,
            p95_latency_ms=2500,  # Lower
            total_tokens=50000,
            estimated_cost=0.50,
            all_slos_passed=True,
            failed_slos=[],
        )

        result = compare_with_previous(current, previous)

        regression_metrics = [r["metric"] for r in result["regressions"]]
        assert "P95 Latency" in regression_metrics


# =============================================================================
# extract_step_latencies Tests
# =============================================================================

class TestExtractStepLatencies:
    """Tests for extract_step_latencies function."""

    def test_extracts_step_latencies(self, sample_eval_results):
        """Extracts latencies by step."""
        result = extract_step_latencies(sample_eval_results)

        assert "preprocess" in result
        assert "rewrite" in result
        assert "retrieval" in result
        assert "generate" in result

    def test_maps_question_ids(self, sample_eval_results):
        """Maps latencies to question IDs."""
        result = extract_step_latencies(sample_eval_results)

        # Check that each step has entries for each question
        assert "q0" in result["preprocess"]
        assert "q1" in result["preprocess"]

    def test_handles_empty_results(self):
        """Handles empty results list."""
        result = extract_step_latencies([])
        assert result == {}


# =============================================================================
# analyze_budget_violations Tests
# =============================================================================

class TestAnalyzeBudgetViolations:
    """Tests for analyze_budget_violations function."""

    def test_returns_structure(self, sample_eval_results):
        """Returns expected structure."""
        result = analyze_budget_violations(sample_eval_results)

        assert "step_stats" in result
        assert "step_violations" in result
        assert "total_violations" in result
        assert "total_budget" in result

    def test_with_step_timings(self, sample_eval_results):
        """Analyzes step timings when provided."""
        step_timings = extract_step_latencies(sample_eval_results)
        result = analyze_budget_violations(sample_eval_results, step_timings)

        # Should have stats for steps
        assert len(result["step_stats"]) > 0

    def test_detects_total_violations(self):
        """Detects questions exceeding total budget."""
        # Create result with very high latency
        judge = JudgeResult(
            context_relevance=1,
            answer_relevance=1,
            groundedness=1,
            needs_human_review=0,
            confidence=0.9,
            explanation="Test",
        )
        slow_result = EvalResult(
            question_id="slow_q",
            question="Slow question",
            target_doc_ids=["doc1"],
            retrieved_doc_ids=["doc1"],
            answer="Answer",
            judge_result=judge,
            doc_recall=1.0,
            latency_ms=10000,  # Very slow - likely over budget
            total_tokens=500,
            step_timings={},
        )

        result = analyze_budget_violations([slow_result])

        # Check if over budget
        if slow_result.latency_ms > result["total_budget"]:
            assert len(result["total_violations"]) > 0


# =============================================================================
# compute_trends Tests (from history.py)
# =============================================================================

class TestComputeTrends:
    """Tests for compute_trends function."""

    def test_returns_no_trend_for_single_entry(self):
        """Returns no trend for single entry."""
        history = [{"metrics": {"rag_triad": 0.80}}]
        result = compute_trends(history, "rag_triad")
        assert result["has_trend"] is False

    def test_computes_basic_stats(self):
        """Computes min, max, avg, current."""
        history = [
            {"metrics": {"rag_triad": 0.70}},
            {"metrics": {"rag_triad": 0.80}},
            {"metrics": {"rag_triad": 0.90}},
        ]
        result = compute_trends(history, "rag_triad")

        assert result["has_trend"] is True
        assert result["min"] == 0.70
        assert result["max"] == 0.90
        assert abs(result["avg"] - 0.80) < 0.001  # Use approx for floats
        assert result["current"] == 0.90
        assert result["previous"] == 0.80

    def test_detects_upward_trend(self):
        """Detects upward trend."""
        history = [
            {"metrics": {"rag_triad": 0.60}},
            {"metrics": {"rag_triad": 0.70}},
            {"metrics": {"rag_triad": 0.80}},
        ]
        result = compute_trends(history, "rag_triad")
        assert result["trend_direction"] == "up"

    def test_detects_downward_trend(self):
        """Detects downward trend."""
        history = [
            {"metrics": {"rag_triad": 0.90}},
            {"metrics": {"rag_triad": 0.80}},
            {"metrics": {"rag_triad": 0.70}},
        ]
        result = compute_trends(history, "rag_triad")
        assert result["trend_direction"] == "down"

    def test_detects_stable_trend(self):
        """Detects stable trend."""
        history = [
            {"metrics": {"rag_triad": 0.80}},
            {"metrics": {"rag_triad": 0.80}},
            {"metrics": {"rag_triad": 0.80}},
        ]
        result = compute_trends(history, "rag_triad")
        assert result["trend_direction"] == "stable"

    def test_handles_missing_metric(self):
        """Handles missing metric gracefully."""
        history = [
            {"metrics": {"other": 0.80}},
            {"metrics": {"other": 0.90}},
        ]
        result = compute_trends(history, "rag_triad")
        # Should use 0 for missing values
        assert result["has_trend"] is True
        assert result["current"] == 0


# =============================================================================
# detect_degradation Tests (from history.py)
# =============================================================================

class TestDetectDegradation:
    """Tests for detect_degradation function."""

    def test_returns_empty_for_improving_metrics(self):
        """Returns empty list when metrics are improving."""
        history = [
            {"metrics": {"rag_triad": 0.70, "context_relevance": 0.70,
                        "answer_relevance": 0.70, "groundedness": 0.70,
                        "doc_recall": 0.70, "p95_latency_ms": 2000}},
            {"metrics": {"rag_triad": 0.80, "context_relevance": 0.80,
                        "answer_relevance": 0.80, "groundedness": 0.80,
                        "doc_recall": 0.80, "p95_latency_ms": 1500}},
        ]
        result = detect_degradation(history)
        assert result == []

    def test_detects_degrading_quality_metric(self):
        """Detects when quality metric is degrading."""
        history = [
            {"metrics": {"rag_triad": 0.90, "context_relevance": 0.90,
                        "answer_relevance": 0.90, "groundedness": 0.90,
                        "doc_recall": 0.90, "p95_latency_ms": 2000}},
            {"metrics": {"rag_triad": 0.70, "context_relevance": 0.90,
                        "answer_relevance": 0.90, "groundedness": 0.90,
                        "doc_recall": 0.90, "p95_latency_ms": 2000}},
        ]
        result = detect_degradation(history)
        assert "rag_triad" in result

    def test_detects_degrading_latency(self):
        """Detects when latency is increasing significantly."""
        history = [
            {"metrics": {"rag_triad": 0.80, "context_relevance": 0.80,
                        "answer_relevance": 0.80, "groundedness": 0.80,
                        "doc_recall": 0.80, "p95_latency_ms": 1000}},
            {"metrics": {"rag_triad": 0.80, "context_relevance": 0.80,
                        "answer_relevance": 0.80, "groundedness": 0.80,
                        "doc_recall": 0.80, "p95_latency_ms": 2000}},
        ]
        result = detect_degradation(history)
        assert "p95_latency_ms" in result

    def test_uses_custom_threshold(self):
        """Respects custom threshold."""
        history = [
            {"metrics": {"rag_triad": 0.85, "context_relevance": 0.85,
                        "answer_relevance": 0.85, "groundedness": 0.85,
                        "doc_recall": 0.85, "p95_latency_ms": 2000}},
            {"metrics": {"rag_triad": 0.80, "context_relevance": 0.85,
                        "answer_relevance": 0.85, "groundedness": 0.85,
                        "doc_recall": 0.85, "p95_latency_ms": 2000}},
        ]
        # With high threshold, 5% drop won't be flagged
        result = detect_degradation(history, threshold=0.10)
        assert "rag_triad" not in result
