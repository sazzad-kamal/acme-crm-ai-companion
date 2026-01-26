"""Data models for action quality evaluation."""

from __future__ import annotations

from pydantic import BaseModel, Field, computed_field

# SLO thresholds for action quality
SLO_ACTION_PASS_RATE = 0.80


class ActionCaseResult(BaseModel):
    """Result for a single action evaluation case."""

    question: str
    answer: str
    suggested_action: str | None
    expected_action: bool = False  # Whether an action was expected for this question
    relevance: float = 0.0
    actionability: float = 0.0
    appropriateness: float = 0.0
    action_passed: bool = False  # From judge, or set by runner based on outcome
    errors: list[str] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def passed(self) -> bool:
        """Pass if no errors and action_passed is True."""
        if self.errors:
            return False
        return self.action_passed


class ActionEvalResults(BaseModel):
    """Aggregated action evaluation results."""

    total: int = 0
    passed: int = 0
    cases: list[ActionCaseResult] = Field(default_factory=list)
    total_with_actions: int = 0  # Only count cases with actions
    avg_relevance: float = 0.0
    avg_actionability: float = 0.0
    avg_appropriateness: float = 0.0

    # Breakdown counts
    action_expected_passed: int = 0  # Action expected, produced, judged pass
    action_expected_failed: int = 0  # Action expected, produced, judged fail
    action_missing: int = 0  # Action expected but not produced
    spurious_action: int = 0  # Action not expected but produced
    correct_silence: int = 0  # Action not expected and not produced

    @property
    def failed(self) -> int:
        """Number of failed cases."""
        return self.total - self.passed

    @property
    def pass_rate(self) -> float:
        """Overall pass rate."""
        return self.passed / self.total if self.total > 0 else 0.0

    @property
    def action_pass_rate(self) -> float:
        """Pass rate for cases with actions only."""
        if self.total_with_actions == 0:
            return 0.0
        action_cases = [c for c in self.cases if c.suggested_action]
        passed_actions = sum(1 for c in action_cases if c.action_passed)
        return passed_actions / self.total_with_actions

    def compute_aggregates(self) -> None:
        """Compute aggregate metrics from individual case results."""
        if not self.cases:
            return

        self.passed = sum(1 for c in self.cases if c.passed)
        self.total_with_actions = sum(1 for c in self.cases if c.suggested_action)

        # Breakdown counts (exclude error cases)
        ok = [c for c in self.cases if not c.errors]
        self.action_expected_passed = sum(
            1 for c in ok if c.expected_action and c.suggested_action and c.action_passed
        )
        self.action_expected_failed = sum(
            1 for c in ok if c.expected_action and c.suggested_action and not c.action_passed
        )
        self.action_missing = sum(
            1 for c in ok if c.expected_action and not c.suggested_action
        )
        self.spurious_action = sum(
            1 for c in ok if not c.expected_action and c.suggested_action
        )
        self.correct_silence = sum(
            1 for c in ok if not c.expected_action and not c.suggested_action
        )

        # Action metrics (only for judged cases: expected + produced)
        judged = [
            c for c in self.cases
            if c.suggested_action and c.expected_action and not c.errors
        ]
        if judged:
            self.avg_relevance = sum(c.relevance for c in judged) / len(judged)
            self.avg_actionability = sum(c.actionability for c in judged) / len(judged)
            self.avg_appropriateness = sum(c.appropriateness for c in judged) / len(judged)


__all__ = ["ActionCaseResult", "ActionEvalResults", "SLO_ACTION_PASS_RATE"]
