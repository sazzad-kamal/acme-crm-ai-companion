"""Data models for RAG comparison results."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ConfigCaseResult:
    """Result for a single question evaluated against one retrieval config."""

    question: str
    reference_answer: str
    answer: str
    contexts: list[str]
    sources: list[str]
    latency_seconds: float
    answer_correctness: float = 0.0
    answer_relevancy: float = 0.0
    faithfulness: float = 0.0
    nan_metrics: list[str] = field(default_factory=list)

    @property
    def composite_score(self) -> float:
        """Weighted composite: 0.4 * relevancy + 0.4 * faithfulness + 0.2 * correctness."""
        return 0.4 * self.answer_relevancy + 0.4 * self.faithfulness + 0.2 * self.answer_correctness


@dataclass
class ConfigResults:
    """Aggregated results for one retrieval configuration."""

    config_name: str
    cases: list[ConfigCaseResult] = field(default_factory=list)

    @property
    def avg_correctness(self) -> float:
        return sum(c.answer_correctness for c in self.cases) / len(self.cases) if self.cases else 0.0

    @property
    def avg_relevancy(self) -> float:
        return sum(c.answer_relevancy for c in self.cases) / len(self.cases) if self.cases else 0.0

    @property
    def avg_faithfulness(self) -> float:
        return sum(c.faithfulness for c in self.cases) / len(self.cases) if self.cases else 0.0

    @property
    def avg_composite(self) -> float:
        return sum(c.composite_score for c in self.cases) / len(self.cases) if self.cases else 0.0

    @property
    def avg_latency(self) -> float:
        return sum(c.latency_seconds for c in self.cases) / len(self.cases) if self.cases else 0.0

    @property
    def p90_latency(self) -> float:
        if not self.cases:
            return 0.0
        latencies = sorted(c.latency_seconds for c in self.cases)
        idx = int(len(latencies) * 0.9)
        return latencies[min(idx, len(latencies) - 1)]

    @property
    def total_nan_count(self) -> int:
        return sum(len(c.nan_metrics) for c in self.cases)

    def to_dict(self) -> dict:
        return {
            "config_name": self.config_name,
            "num_questions": len(self.cases),
            "avg_correctness": round(self.avg_correctness, 4),
            "avg_relevancy": round(self.avg_relevancy, 4),
            "avg_faithfulness": round(self.avg_faithfulness, 4),
            "avg_composite": round(self.avg_composite, 4),
            "avg_latency_seconds": round(self.avg_latency, 2),
            "p90_latency_seconds": round(self.p90_latency, 2),
            "nan_count": self.total_nan_count,
        }


@dataclass
class ComparisonResults:
    """Full comparison across all retrieval configurations."""

    configs: list[ConfigResults] = field(default_factory=list)

    @property
    def winner(self) -> ConfigResults | None:
        if not self.configs:
            return None
        return max(self.configs, key=lambda c: c.avg_composite)

    @property
    def production_config(self) -> ConfigResults | None:
        for c in self.configs:
            if c.config_name == "vector_top5":
                return c
        return None

    def to_dict(self) -> dict:
        winner = self.winner
        return {
            "configs": [c.to_dict() for c in self.configs],
            "winner": winner.config_name if winner else None,
            "winner_composite": round(winner.avg_composite, 4) if winner else None,
        }
