"""
BACKWARD COMPATIBILITY SHIM - Use backend.rag.retrieval.private instead.

This module redirects to the new location at backend.rag.retrieval.private
All imports from this module will continue to work.

DEPRECATED: Import from backend.rag.retrieval instead:
    from backend.rag.retrieval import PrivateRetrievalBackend, create_private_backend
"""

import warnings

# Issue deprecation warning
warnings.warn(
    "Importing from backend.rag.private is deprecated. "
    "Use 'from backend.rag.retrieval import PrivateRetrievalBackend' instead.",
    DeprecationWarning,
    stacklevel=2,
)

# =============================================================================
# Redirect all imports to new location
# =============================================================================

from backend.rag.retrieval.private import (
    PrivateRetrievalBackend,
    create_private_backend,
)
from backend.rag.config import get_config

# =============================================================================
# Backward Compatibility Exports
# =============================================================================

PRIVATE_COLLECTION_NAME = get_config().private_collection_name

__all__ = [
    "PrivateRetrievalBackend",
    "create_private_backend",
    "PRIVATE_COLLECTION_NAME",
]
