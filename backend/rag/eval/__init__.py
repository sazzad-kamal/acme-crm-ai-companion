# backend.rag.eval - Evaluation Harnesses
"""
RAG evaluation tools and metrics.

Modules:
- models: Evaluation data models (JudgeResult, EvalResult)
- judge: LLM-as-judge evaluation functions
- docs_eval: Documentation RAG evaluation harness
- account_eval: Account-scoped RAG evaluation harness
"""

from backend.rag.eval.models import JudgeResult, EvalResult, AccountEvalResult
from backend.rag.eval.judge import judge_response, judge_account_response

__all__ = [
    "JudgeResult",
    "EvalResult",
    "AccountEvalResult",
    "judge_response",
    "judge_account_response",
]
