"""
BACKWARD COMPATIBILITY SHIM - Use backend.rag.eval.docs_eval instead.

This module redirects to the new location at backend.rag.eval
All imports from this module will continue to work.

DEPRECATED: Import from backend.rag.eval instead:
    from backend.rag.eval import JudgeResult, EvalResult
    from backend.rag.eval.docs_eval import run_evaluation
"""

import warnings

# Issue deprecation warning
warnings.warn(
    "Importing from backend.rag.eval is deprecated. "
    "Use 'from backend.rag.eval.docs_eval import run_evaluation' instead.",
    DeprecationWarning,
    stacklevel=2,
)

# =============================================================================
# Redirect all imports to new location
# =============================================================================

from backend.rag.eval.models import JudgeResult, EvalResult
from backend.rag.eval.judge import judge_response, compute_doc_recall
from backend.rag.eval.docs_eval import (
    load_eval_questions,
    evaluate_question,
    run_evaluation,
    print_summary,
    main,
    EVAL_QUESTIONS_PATH,
)
from backend.rag.prompts import (
    EVAL_JUDGE_SYSTEM as JUDGE_SYSTEM,
    EVAL_JUDGE_PROMPT as JUDGE_PROMPT_TEMPLATE,
)

__all__ = [
    "JudgeResult",
    "EvalResult",
    "judge_response",
    "compute_doc_recall",
    "load_eval_questions",
    "evaluate_question",
    "run_evaluation",
    "print_summary",
    "main",
    "EVAL_QUESTIONS_PATH",
    "JUDGE_SYSTEM",
    "JUDGE_PROMPT_TEMPLATE",
]
