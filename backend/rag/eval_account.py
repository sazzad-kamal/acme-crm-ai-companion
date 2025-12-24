"""
BACKWARD COMPATIBILITY SHIM - Use backend.rag.eval.account_eval instead.

This module redirects to the new location at backend.rag.eval
All imports from this module will continue to work.

DEPRECATED: Import from backend.rag.eval instead:
    from backend.rag.eval import AccountEvalResult
    from backend.rag.eval.account_eval import run_evaluation
"""

import warnings

# Issue deprecation warning
warnings.warn(
    "Importing from backend.rag.eval_account is deprecated. "
    "Use 'from backend.rag.eval.account_eval import run_evaluation' instead.",
    DeprecationWarning,
    stacklevel=2,
)

# =============================================================================
# Redirect all imports to new location
# =============================================================================

from backend.rag.eval.models import JudgeResult, AccountEvalResult
from backend.rag.eval.judge import judge_account_response, check_privacy_leakage
from backend.rag.eval.account_eval import (
    generate_eval_questions,
    ensure_private_collection_exists,
    evaluate_question,
    run_evaluation,
    print_summary,
    main,
    OUTPUT_PATH,
    NUM_QUESTIONS_PER_COMPANY,
    NUM_COMPANIES,
    RANDOM_SEED,
)
from backend.rag.prompts import ACCOUNT_EVAL_JUDGE_SYSTEM as JUDGE_SYSTEM

__all__ = [
    "JudgeResult",
    "AccountEvalResult",
    "judge_account_response",
    "check_privacy_leakage",
    "generate_eval_questions",
    "ensure_private_collection_exists",
    "evaluate_question",
    "run_evaluation",
    "print_summary",
    "main",
    "OUTPUT_PATH",
    "NUM_QUESTIONS_PER_COMPANY",
    "NUM_COMPANIES",
    "RANDOM_SEED",
    "JUDGE_SYSTEM",
]
