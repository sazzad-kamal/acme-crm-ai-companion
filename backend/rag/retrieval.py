"""
BACKWARD COMPATIBILITY SHIM - Use backend.rag.retrieval instead.

This module redirects to the new location at backend.rag.retrieval.base
All imports from this module will continue to work.

DEPRECATED: Import from backend.rag.retrieval instead:
    from backend.rag.retrieval import RetrievalBackend, create_backend
"""

import warnings

# Issue deprecation warning
warnings.warn(
    "Importing from backend.rag.retrieval is deprecated. "
    "Use 'from backend.rag.retrieval import RetrievalBackend' instead.",
    DeprecationWarning,
    stacklevel=2,
)

# =============================================================================
# Redirect all imports to new location
# =============================================================================

from backend.rag.retrieval.base import (
    RetrievalBackend,
    create_backend,
)
from backend.rag.retrieval.embedding import (
    get_cached_embedding as _get_cached_embedding,
    cache_embedding as _cache_embedding,
    clear_embedding_cache,
)
from backend.rag.config import get_config

# =============================================================================
# Backward Compatibility Exports
# =============================================================================

# Export constants for backward compatibility (use config instead)
QDRANT_PATH = get_config().qdrant_path
COLLECTION_NAME = get_config().docs_collection_name
EMBEDDING_MODEL = get_config().embedding_model
RERANKER_MODEL = get_config().reranker_model
EMBEDDING_DIM = get_config().embedding_dim

__all__ = [
    "RetrievalBackend",
    "create_backend",
    "clear_embedding_cache",
    "QDRANT_PATH",
    "COLLECTION_NAME",
    "EMBEDDING_MODEL",
    "RERANKER_MODEL",
    "EMBEDDING_DIM",
]
