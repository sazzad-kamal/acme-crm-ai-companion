"""
Document chunk models for the RAG pipeline.

NOTE: These models have been moved to backend.common.models.
This module re-exports them for backward compatibility.
"""

# Re-export from canonical location
from backend.common.models import DocumentChunk, ScoredChunk

__all__ = ["DocumentChunk", "ScoredChunk"]
