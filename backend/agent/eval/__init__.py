"""
Agent evaluation suite.

Provides comprehensive LLM quality evaluation:
- Routing accuracy (mode selection, company extraction)
- Tool selection accuracy
- Answer quality (relevance, groundedness)

Note: Tool correctness is tested via pytest (backend/agent/tests/).

Usage:
    python -m backend.agent.eval.e2e_eval
    python -m backend.agent.eval.e2e_eval --parallel -w 4
"""

from backend.agent.eval.models import (
    AgentEvalSummary,
    E2EEvalResult,
    E2EEvalSummary,
)

__all__ = [
    "E2EEvalResult",
    "E2EEvalSummary",
    "AgentEvalSummary",
]
