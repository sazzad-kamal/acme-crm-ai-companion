"""
Tests for shared evaluation utilities.
"""

import json
import pytest
from pathlib import Path

from backend.common.eval_base import (
    compare_to_baseline,
    save_baseline,
    print_baseline_comparison,
    compute_p95,
    compute_pass_rate,
    format_check_mark,
    format_percentage,
    format_latency,
    format_delta,
    REGRESSION_THRESHOLD,
)


# =============================================================================
# Tests: Baseline Comparison
# =============================================================================

class TestCompareToBaseline:
    """Tests for compare_to_baseline function."""

    def test_no_baseline_file_returns_no_regression(self, tmp_path: Path):
        """When baseline file doesn't exist, return no regression."""
        baseline_path = tmp_path / "nonexistent.json"

        is_regression, baseline_score = compare_to_baseline(0.8, baseline_path)

        assert is_regression is False
        assert baseline_score is None

    def test_detects_regression_when_score_drops_beyond_threshold(self, tmp_path: Path):
        """Detect regression when current score drops more than threshold."""
        baseline_path = tmp_path / "baseline.json"
        baseline_path.write_text(json.dumps({"summary": {"overall_score": 0.9}}))

        # 0.9 - 0.05 threshold = 0.85, so 0.80 is a regression
        is_regression, baseline_score = compare_to_baseline(0.80, baseline_path)

        assert is_regression is True
        assert baseline_score == 0.9

    def test_no_regression_when_score_within_threshold(self, tmp_path: Path):
        """No regression when score drops less than threshold."""
        baseline_path = tmp_path / "baseline.json"
        baseline_path.write_text(json.dumps({"summary": {"overall_score": 0.9}}))

        # 0.9 - 0.05 threshold = 0.85, so 0.86 is not a regression
        is_regression, baseline_score = compare_to_baseline(0.86, baseline_path)

        assert is_regression is False
        assert baseline_score == 0.9

    def test_no_regression_when_score_improves(self, tmp_path: Path):
        """No regression when score improves."""
        baseline_path = tmp_path / "baseline.json"
        baseline_path.write_text(json.dumps({"summary": {"overall_score": 0.8}}))

        is_regression, baseline_score = compare_to_baseline(0.95, baseline_path)

        assert is_regression is False
        assert baseline_score == 0.8

    def test_uses_custom_score_key(self, tmp_path: Path):
        """Use custom score key when specified."""
        baseline_path = tmp_path / "baseline.json"
        baseline_path.write_text(json.dumps({
            "summary": {"rag_triad_success": 0.85, "overall_score": 0.9}
        }))

        is_regression, baseline_score = compare_to_baseline(
            0.75, baseline_path, score_key="rag_triad_success"
        )

        assert is_regression is True
        assert baseline_score == 0.85

    def test_handles_flat_json_structure(self, tmp_path: Path):
        """Handle baseline JSON without nested summary."""
        baseline_path = tmp_path / "baseline.json"
        baseline_path.write_text(json.dumps({"overall_score": 0.9}))

        is_regression, baseline_score = compare_to_baseline(0.95, baseline_path)

        assert is_regression is False
        assert baseline_score == 0.9

    def test_handles_malformed_json(self, tmp_path: Path):
        """Handle malformed JSON gracefully."""
        baseline_path = tmp_path / "baseline.json"
        baseline_path.write_text("not valid json {{{")

        is_regression, baseline_score = compare_to_baseline(0.8, baseline_path)

        assert is_regression is False
        assert baseline_score is None


class TestSaveBaseline:
    """Tests for save_baseline function."""

    def test_saves_baseline_to_file(self, tmp_path: Path):
        """Save baseline creates JSON file with summary."""
        baseline_path = tmp_path / "baseline.json"
        summary = {"overall_score": 0.9, "latency_p95": 150}

        save_baseline(summary, baseline_path)

        assert baseline_path.exists()
        saved = json.loads(baseline_path.read_text())
        assert saved["summary"]["overall_score"] == 0.9
        assert saved["summary"]["latency_p95"] == 150

    def test_creates_parent_directories(self, tmp_path: Path):
        """Create parent directories if they don't exist."""
        baseline_path = tmp_path / "nested" / "dir" / "baseline.json"
        summary = {"overall_score": 0.9}

        save_baseline(summary, baseline_path)

        assert baseline_path.exists()

    def test_overwrites_existing_baseline(self, tmp_path: Path):
        """Overwrite existing baseline file."""
        baseline_path = tmp_path / "baseline.json"
        baseline_path.write_text(json.dumps({"summary": {"overall_score": 0.5}}))

        save_baseline({"overall_score": 0.9}, baseline_path)

        saved = json.loads(baseline_path.read_text())
        assert saved["summary"]["overall_score"] == 0.9


# =============================================================================
# Tests: Metrics Computation
# =============================================================================

class TestComputeP95:
    """Tests for compute_p95 function."""

    def test_empty_list_returns_zero(self):
        """Empty list returns 0.0."""
        assert compute_p95([]) == 0.0

    def test_single_value(self):
        """Single value returns that value."""
        assert compute_p95([100.0]) == 100.0

    def test_computes_p95_correctly(self):
        """Compute P95 for a list of values."""
        values = list(range(1, 101))  # 1-100
        p95 = compute_p95(values)
        # Index 95 (0-based) in sorted 1-100 = 96
        assert p95 == 96

    def test_handles_unsorted_list(self):
        """Handle unsorted input."""
        values = [100, 1, 50, 25, 75]
        p95 = compute_p95(values)
        assert p95 == 100  # 95th percentile of 5 values is the last one


class TestComputePassRate:
    """Tests for compute_pass_rate function."""

    def test_empty_list_returns_zero(self):
        """Empty list returns 0.0."""
        assert compute_pass_rate([]) == 0.0

    def test_all_pass(self):
        """All values pass."""
        assert compute_pass_rate([1, 1, 1, 1]) == 1.0

    def test_all_fail(self):
        """All values fail."""
        assert compute_pass_rate([0, 0, 0, 0]) == 0.0

    def test_mixed_results(self):
        """Mixed pass/fail results."""
        assert compute_pass_rate([1, 0, 1, 0]) == 0.5

    def test_boolean_values(self):
        """Handle boolean values."""
        assert compute_pass_rate([True, True, False]) == pytest.approx(2/3)

    def test_custom_threshold(self):
        """Use custom threshold."""
        assert compute_pass_rate([1, 2, 3, 4], threshold=3) == 0.5


# =============================================================================
# Tests: Formatting Helpers
# =============================================================================

class TestFormatCheckMark:
    """Tests for format_check_mark function."""

    def test_true_returns_green_check(self):
        """True returns green check mark."""
        result = format_check_mark(True)
        assert "[green]" in result
        assert "✓" in result

    def test_false_returns_red_cross(self):
        """False returns red cross mark."""
        result = format_check_mark(False)
        assert "[red]" in result
        assert "✗" in result


class TestFormatPercentage:
    """Tests for format_percentage function."""

    def test_high_value_is_green(self):
        """High value (>= 90%) is green."""
        result = format_percentage(0.95)
        assert "[green]" in result
        assert "95.0%" in result

    def test_medium_value_is_yellow(self):
        """Medium value (70-90%) is yellow."""
        result = format_percentage(0.75)
        assert "[yellow]" in result
        assert "75.0%" in result

    def test_low_value_is_red(self):
        """Low value (< 70%) is red."""
        result = format_percentage(0.50)
        assert "[red]" in result
        assert "50.0%" in result

    def test_custom_thresholds(self):
        """Use custom thresholds."""
        result = format_percentage(0.85, thresholds=(0.95, 0.80))
        assert "[yellow]" in result


class TestFormatLatency:
    """Tests for format_latency function."""

    def test_formats_milliseconds(self):
        """Format latency in milliseconds."""
        result = format_latency(150.5)
        assert result == "150ms"

    def test_with_slo_under(self):
        """Under SLO is green."""
        result = format_latency(100, slo_ms=200)
        assert "[green]" in result
        assert "100ms" in result

    def test_with_slo_over(self):
        """Over SLO is red."""
        result = format_latency(300, slo_ms=200)
        assert "[red]" in result
        assert "300ms" in result


class TestFormatDelta:
    """Tests for format_delta function."""

    def test_positive_delta_default_green(self):
        """Positive delta is green by default."""
        result = format_delta(0.05)
        assert "[green]" in result
        assert "+5.0%" in result

    def test_negative_delta_default_red(self):
        """Negative delta is red by default."""
        result = format_delta(-0.05)
        assert "[red]" in result
        assert "-5.0%" in result

    def test_inverted_colors(self):
        """Inverted colors when is_positive_good=False."""
        result = format_delta(0.05, is_positive_good=False)
        assert "[red]" in result


# =============================================================================
# Tests: Constants
# =============================================================================

class TestConstants:
    """Tests for module constants."""

    def test_regression_threshold_is_five_percent(self):
        """Regression threshold is 5%."""
        assert REGRESSION_THRESHOLD == 0.05
