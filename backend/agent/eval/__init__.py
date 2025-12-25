"""
Agent evaluation suite.

Provides comprehensive evaluation for:
- Tool correctness (do tools return accurate data?)
- Router accuracy (does it pick the right mode?)
- End-to-end orchestration (full pipeline quality)
"""

from backend.agent.eval.models import (
    ToolEvalResult,
    RouterEvalResult,
    E2EEvalResult,
    AgentEvalSummary,
)

__all__ = [
    "ToolEvalResult",
    "RouterEvalResult",
    "E2EEvalResult",
    "AgentEvalSummary",
]
