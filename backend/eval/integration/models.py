"""Data models for integration (conversation) evaluation."""

from __future__ import annotations

from pydantic import BaseModel, Field

from backend.eval.answer.text.models import (
    SLO_TEXT_ANSWER_CORRECTNESS,
    SLO_TEXT_ANSWER_RELEVANCY,
    SLO_TEXT_FAITHFULNESS,
)
from backend.eval.shared.models import BaseEvalResults

# SLO thresholds for integration evaluation
SLO_CONVO_STEP_PASS_RATE = 0.95  # 95% of questions should pass

# Latency SLOs (milliseconds)
SLO_LATENCY_P50_MS = 3000  # 3 seconds
SLO_LATENCY_P95_MS = 8000  # 8 seconds


class ConvoStepResult(BaseModel):
    """Result of a single question in a conversation."""

    question: str
    answer: str
    # RAGAS metrics (0.0-1.0) - answer quality
    relevance_score: float = 0.0  # RAGAS answer_relevancy
    faithfulness_score: float = 0.0  # RAGAS faithfulness
    answer_correctness_score: float = 0.0  # RAGAS answer_correctness
    errors: list[str] = Field(default_factory=list)
    # RAGAS reliability tracking
    ragas_metrics_total: int = 0  # Number of metrics evaluated (usually 2)
    ragas_metrics_failed: int = 0  # Number of metrics that returned NaN
    # Action quality (0.0-1.0) - from action judge
    expected_action: bool | None = None  # None if not in fixture
    suggested_action: str | None = None
    action_relevance: float = 0.0
    action_actionability: float = 0.0
    action_appropriateness: float = 0.0
    action_passed: bool = True  # True if no action or action judged pass
    # Per-question metrics (latency, cost)
    latency_ms: float = 0.0  # End-to-end latency for this question

    @property
    def action_missing(self) -> bool:
        """Action was expected but not provided."""
        return self.expected_action is True and self.suggested_action is None

    @property
    def action_spurious(self) -> bool:
        """Action was not expected but was provided."""
        return self.expected_action is False and self.suggested_action is not None

    @property
    def passed(self) -> bool:
        """Question passes if no errors, meets answer quality, and action quality."""
        if self.errors:
            return False
        if not self.action_passed:
            return False
        # Only check RAGAS scores when actually evaluated
        if self.ragas_metrics_total > 0:
            return (
                self.relevance_score >= SLO_TEXT_ANSWER_RELEVANCY
                and self.faithfulness_score >= SLO_TEXT_FAITHFULNESS
                and self.answer_correctness_score >= SLO_TEXT_ANSWER_CORRECTNESS
            )
        return True


class ConvoEvalResults(BaseEvalResults):
    """Aggregated results from all question tests."""

    cases: list[ConvoStepResult] = Field(default_factory=list)
    # RAGAS metrics (0.0-1.0) - answer quality
    avg_relevance: float = 0.0  # RAGAS answer_relevancy
    avg_faithfulness: float = 0.0  # RAGAS faithfulness
    avg_answer_correctness: float = 0.0  # RAGAS answer_correctness
    # RAGAS reliability tracking
    ragas_metrics_total: int = 0  # Total individual metrics evaluated (questions x 2)
    ragas_metrics_failed: int = 0  # Individual metrics that returned NaN
    # Action quality
    avg_action_relevance: float = 0.0
    avg_action_actionability: float = 0.0
    avg_action_appropriateness: float = 0.0
    actions_judged: int = 0
    actions_passed: int = 0
    actions_missing: int = 0
    actions_spurious: int = 0
    # Latency metrics (milliseconds)
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    latency_avg_ms: float = 0.0
    # Eval metadata
    eval_version: str = ""
    eval_checksum: str = ""

    @property
    def ragas_success_rate(self) -> float:
        """RAGAS metrics success rate."""
        if self.ragas_metrics_total == 0:
            return 1.0
        return (self.ragas_metrics_total - self.ragas_metrics_failed) / self.ragas_metrics_total

    @property
    def latency_p50_passed(self) -> bool:
        """Check if p50 latency meets SLO."""
        return self.latency_p50_ms <= SLO_LATENCY_P50_MS

    @property
    def latency_p95_passed(self) -> bool:
        """Check if p95 latency meets SLO."""
        return self.latency_p95_ms <= SLO_LATENCY_P95_MS

    def _percentile(self, latencies: list[float], p: int) -> float:
        """Calculate percentile from sorted latencies."""
        if not latencies:
            return 0.0
        idx = int(len(latencies) * p / 100)
        idx = min(idx, len(latencies) - 1)
        return latencies[idx]

    def compute_aggregates(self) -> None:
        """Compute aggregate metrics from individual question results."""
        if not self.cases:
            return

        self.passed = sum(1 for c in self.cases if c.passed)

        self.ragas_metrics_total = sum(c.ragas_metrics_total for c in self.cases)
        self.ragas_metrics_failed = sum(c.ragas_metrics_failed for c in self.cases)

        # RAGAS averages (only for cases that were actually evaluated)
        evaluated = [c for c in self.cases if c.ragas_metrics_total > 0]
        if evaluated:
            self.avg_relevance = sum(c.relevance_score for c in evaluated) / len(evaluated)
            self.avg_faithfulness = sum(c.faithfulness_score for c in evaluated) / len(evaluated)
            self.avg_answer_correctness = sum(c.answer_correctness_score for c in evaluated) / len(evaluated)

        # Action aggregates (only for steps where the judge actually ran)
        judged = [c for c in self.cases if c.suggested_action and not c.action_spurious]
        self.actions_judged = len(judged)
        if judged:
            self.actions_passed = sum(1 for c in judged if c.action_passed)
            self.avg_action_relevance = sum(c.action_relevance for c in judged) / len(judged)
            self.avg_action_actionability = sum(c.action_actionability for c in judged) / len(judged)
            self.avg_action_appropriateness = sum(c.action_appropriateness for c in judged) / len(judged)

        self.actions_missing = sum(1 for c in self.cases if c.action_missing)
        self.actions_spurious = sum(1 for c in self.cases if c.action_spurious)

        # Latency aggregates
        latencies = sorted(c.latency_ms for c in self.cases if c.latency_ms > 0)
        if latencies:
            self.latency_avg_ms = sum(latencies) / len(latencies)
            self.latency_p50_ms = self._percentile(latencies, 50)
            self.latency_p95_ms = self._percentile(latencies, 95)
            self.latency_p99_ms = self._percentile(latencies, 99)
