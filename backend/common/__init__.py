# backend.common - Shared Utilities
"""
Shared utilities used by multiple backend modules.

Modules:
- llm_client: LLM API interactions (shared between agent and rag)
- performance: Timing, parallel execution, resource management
- validation: Input validation and sanitization
- eval_base: Shared evaluation utilities (console, formatting)

Other modules live where they're actually used:
- models: backend.rag.models (DocumentChunk, ScoredChunk)
- formatters: backend.agent.formatters
- context_builder, prompts, company_resolver: backend.rag.pipeline.*
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
from backend.common.performance import (
    TimeoutConfig,
    get_timeout_config,
    timed,
    timed_decorator,
    run_parallel,
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
    # Performance
    "TimeoutConfig",
    "get_timeout_config",
    "timed",
    "timed_decorator",
    "run_parallel",
]
