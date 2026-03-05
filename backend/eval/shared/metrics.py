"""Cost and latency metrics tracking for evaluation.

Provides utilities for tracking per-question and aggregate metrics
including token costs and response latencies.
"""

import time
from dataclasses import dataclass, field
from typing import Any

# Pricing per 1M tokens (from CLAUDE.md)
OPENAI_PRICING = {
    "gpt-5": {"input": 1.25, "output": 10.0},
    "gpt-5.2": {"input": 1.75, "output": 14.0},
    "gpt-5.2-pro": {"input": 21.0, "output": 168.0},
}

# Default model used for cost estimation
DEFAULT_MODEL = "gpt-5.2"


@dataclass
class QuestionMetrics:
    """Metrics for a single question evaluation."""

    question: str
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    llm_calls: int = 0
    sql_queries: int = 0
    retries: int = 0

    @property
    def cost_usd(self) -> float:
        """Estimated cost in USD based on default model pricing."""
        pricing = OPENAI_PRICING.get(DEFAULT_MODEL, OPENAI_PRICING["gpt-5.2"])
        input_cost = (self.input_tokens / 1_000_000) * pricing["input"]
        output_cost = (self.output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost


@dataclass
class EvalMetrics:
    """Aggregate metrics for an evaluation run."""

    questions: list[QuestionMetrics] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0

    @property
    def total_latency_ms(self) -> float:
        """Total latency across all questions."""
        return sum(q.latency_ms for q in self.questions)

    @property
    def avg_latency_ms(self) -> float:
        """Average latency per question."""
        if not self.questions:
            return 0.0
        return self.total_latency_ms / len(self.questions)

    @property
    def p50_latency_ms(self) -> float:
        """50th percentile latency."""
        return self._percentile(50)

    @property
    def p95_latency_ms(self) -> float:
        """95th percentile latency."""
        return self._percentile(95)

    @property
    def p99_latency_ms(self) -> float:
        """99th percentile latency."""
        return self._percentile(99)

    def _percentile(self, p: int) -> float:
        """Calculate percentile latency."""
        if not self.questions:
            return 0.0
        latencies = sorted(q.latency_ms for q in self.questions)
        idx = int(len(latencies) * p / 100)
        idx = min(idx, len(latencies) - 1)
        return latencies[idx]

    @property
    def total_cost_usd(self) -> float:
        """Total estimated cost in USD."""
        return sum(q.cost_usd for q in self.questions)

    @property
    def total_input_tokens(self) -> int:
        """Total input tokens."""
        return sum(q.input_tokens for q in self.questions)

    @property
    def total_output_tokens(self) -> int:
        """Total output tokens."""
        return sum(q.output_tokens for q in self.questions)

    @property
    def total_llm_calls(self) -> int:
        """Total LLM API calls."""
        return sum(q.llm_calls for q in self.questions)

    @property
    def duration_seconds(self) -> float:
        """Total evaluation duration in seconds."""
        if self.end_time == 0.0:
            return time.time() - self.start_time
        return self.end_time - self.start_time

    def finish(self) -> None:
        """Mark evaluation as complete."""
        self.end_time = time.time()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_questions": len(self.questions),
            "duration_seconds": round(self.duration_seconds, 2),
            "latency": {
                "total_ms": round(self.total_latency_ms, 2),
                "avg_ms": round(self.avg_latency_ms, 2),
                "p50_ms": round(self.p50_latency_ms, 2),
                "p95_ms": round(self.p95_latency_ms, 2),
                "p99_ms": round(self.p99_latency_ms, 2),
            },
            "tokens": {
                "input": self.total_input_tokens,
                "output": self.total_output_tokens,
                "total": self.total_input_tokens + self.total_output_tokens,
            },
            "cost_usd": round(self.total_cost_usd, 4),
            "llm_calls": self.total_llm_calls,
        }


class MetricsTimer:
    """Context manager for timing operations."""

    def __init__(self) -> None:
        self.start_time: float = 0.0
        self.end_time: float = 0.0

    def __enter__(self) -> "MetricsTimer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        self.end_time = time.perf_counter()

    @property
    def elapsed_ms(self) -> float:
        """Elapsed time in milliseconds."""
        return (self.end_time - self.start_time) * 1000


__all__ = [
    "QuestionMetrics",
    "EvalMetrics",
    "MetricsTimer",
    "OPENAI_PRICING",
    "DEFAULT_MODEL",
]
