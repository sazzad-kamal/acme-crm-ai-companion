"""
BACKWARD COMPATIBILITY SHIM - Use backend.rag.pipeline instead.

This module redirects to the new location at backend.rag.pipeline.docs
All imports from this module will continue to work.

DEPRECATED: Import from backend.rag.pipeline instead:
    from backend.rag.pipeline import answer_question
"""

import warnings

# Issue deprecation warning
warnings.warn(
    "Importing from backend.rag.pipeline is deprecated. "
    "Use 'from backend.rag.pipeline import answer_question' instead.",
    DeprecationWarning,
    stacklevel=2,
)

# =============================================================================
# Redirect all imports to new location
# =============================================================================

from backend.rag.pipeline.docs import (
    answer_question,
    rewrite_query,
    generate_hyde_answer,
    generate_answer,
)
from backend.rag.pipeline.base import (
    PipelineProgress,
    apply_lexical_gate,
    apply_per_doc_cap,
    build_context,
)
from backend.rag.prompts import (
    DOCS_QUERY_REWRITE_SYSTEM as QUERY_REWRITE_SYSTEM,
    DOCS_HYDE_SYSTEM as HYDE_SYSTEM,
    DOCS_ANSWER_SYSTEM as ANSWER_SYSTEM,
)

__all__ = [
    "answer_question",
    "rewrite_query",
    "generate_hyde_answer",
    "generate_answer",
    "PipelineProgress",
    "apply_lexical_gate",
    "apply_per_doc_cap",
    "build_context",
    "QUERY_REWRITE_SYSTEM",
    "HYDE_SYSTEM",
    "ANSWER_SYSTEM",
]
