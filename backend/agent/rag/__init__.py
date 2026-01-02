"""
RAG (Retrieval Augmented Generation) module.

Provides document and account search capabilities via Qdrant vector store.

Public API:
- tool_docs_rag: Search CRM documentation
- tool_account_rag: Search account-specific texts
- ingest_docs: Ingest markdown docs into Qdrant
- ingest_private_texts: Ingest private texts into Qdrant
- get_qdrant_client: Get shared Qdrant client
- close_qdrant_client: Close the shared client
"""

from backend.agent.rag.config import (
    QDRANT_PATH,
    DOCS_DIR,
    JSONL_PATH,
    DOCS_COLLECTION,
    PRIVATE_COLLECTION,
    EMBEDDING_MODEL,
)
from backend.agent.rag.client import (
    get_qdrant_client,
    close_qdrant_client,
    get_docs_index,
)
from backend.agent.rag.tools import (
    tool_docs_rag,
    tool_account_rag,
)
from backend.agent.rag.ingest import (
    ingest_docs,
    ingest_private_texts,
)

__all__ = [
    # Config
    "QDRANT_PATH",
    "DOCS_DIR",
    "JSONL_PATH",
    "DOCS_COLLECTION",
    "PRIVATE_COLLECTION",
    "EMBEDDING_MODEL",
    # Client
    "get_qdrant_client",
    "close_qdrant_client",
    "get_docs_index",
    # Tools
    "tool_docs_rag",
    "tool_account_rag",
    # Ingest
    "ingest_docs",
    "ingest_private_texts",
]
