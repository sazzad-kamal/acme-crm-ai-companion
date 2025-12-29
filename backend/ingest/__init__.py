"""
Document ingestion package for LlamaIndex RAG.

Provides functions to ingest documentation and private CRM text into Qdrant.
"""

from backend.ingest.ingest_docs import ingest_docs
from backend.ingest.ingest_private import ingest_private_texts

__all__ = ["ingest_docs", "ingest_private_texts"]
