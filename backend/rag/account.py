"""
BACKWARD COMPATIBILITY SHIM - Use backend.rag.pipeline.account instead.

This module redirects to the new location at backend.rag.pipeline.account
All imports from this module will continue to work.

DEPRECATED: Import from backend.rag.pipeline instead:
    from backend.rag.pipeline import answer_account_question
"""

import warnings

# Issue deprecation warning
warnings.warn(
    "Importing from backend.rag.account is deprecated. "
    "Use 'from backend.rag.pipeline import answer_account_question' instead.",
    DeprecationWarning,
    stacklevel=2,
)

# =============================================================================
# Redirect all imports to new location
# =============================================================================

from backend.rag.pipeline.account import (
    answer_account_question,
    load_companies_df,
    resolve_company_id,
    rewrite_query,
    generate_hyde,
    generate_answer,
    get_private_backend,
    get_docs_backend,
)
from backend.rag.prompts import (
    ACCOUNT_QUERY_REWRITE_SYSTEM as QUERY_REWRITE_SYSTEM,
    ACCOUNT_HYDE_SYSTEM as HYDE_SYSTEM,
    ACCOUNT_ANSWER_SYSTEM as ANSWER_SYSTEM,
)

__all__ = [
    "answer_account_question",
    "load_companies_df",
    "resolve_company_id",
    "rewrite_query",
    "generate_hyde",
    "generate_answer",
    "get_private_backend",
    "get_docs_backend",
    "QUERY_REWRITE_SYSTEM",
    "HYDE_SYSTEM",
    "ANSWER_SYSTEM",
]
