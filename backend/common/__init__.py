# backend.common - Shared Utilities
"""
Shared utilities used by multiple backend modules.

Modules:
- llm_client: LLM API interactions (shared between agent and eval)
- validation: Input validation and sanitization
- eval_base: Shared evaluation utilities (console, formatting)
"""

from backend.common.llm_client import call_llm, call_llm_safe, call_llm_with_metrics
from backend.common.validation import (
    validate_question,
    validate_company_id,
    validate_mode,
    validate_chat_request,
    sanitize_string,
    is_safe_input,
)

__all__ = [
    # LLM
    "call_llm",
    "call_llm_safe",
    "call_llm_with_metrics",
    # Validation
    "validate_question",
    "validate_company_id",
    "validate_mode",
    "validate_chat_request",
    "sanitize_string",
    "is_safe_input",
]
