"""
Qdrant client singleton and index management.

Provides thread-safe access to Qdrant client and LlamaIndex indexes.
"""

import logging
import threading

from qdrant_client import QdrantClient

from backend.agent.rag.config import DOCS_COLLECTION, EMBEDDING_MODEL, QDRANT_PATH

logger = logging.getLogger(__name__)


# =============================================================================
# Singleton Qdrant Client
# =============================================================================

_qdrant_client: QdrantClient | None = None
_qdrant_lock = threading.Lock()


def get_qdrant_client() -> QdrantClient:
    """Get shared Qdrant client (singleton)."""
    global _qdrant_client

    if _qdrant_client is not None:
        return _qdrant_client

    with _qdrant_lock:
        if _qdrant_client is not None:
            return _qdrant_client

        QDRANT_PATH.mkdir(parents=True, exist_ok=True)
        _qdrant_client = QdrantClient(path=str(QDRANT_PATH))
        return _qdrant_client


def close_qdrant_client() -> None:
    """Close the shared Qdrant client (for cleanup)."""
    global _qdrant_client
    with _qdrant_lock:
        if _qdrant_client is not None:
            _qdrant_client.close()
            _qdrant_client = None


# =============================================================================
# LlamaIndex Index Singletons
# =============================================================================

_docs_index = None
_index_lock = threading.Lock()
_embed_model = None


def _get_embed_model():
    """Get the embedding model (lazy load)."""
    global _embed_model
    if _embed_model is None:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        _embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
    return _embed_model


def get_docs_index():
    """Get or create the docs vector index."""
    global _docs_index

    if _docs_index is not None:
        return _docs_index

    with _index_lock:
        if _docs_index is not None:
            return _docs_index

        from llama_index.core import Settings, VectorStoreIndex
        from llama_index.vector_stores.qdrant import QdrantVectorStore

        Settings.embed_model = _get_embed_model()

        client = get_qdrant_client()
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=DOCS_COLLECTION,
        )
        _docs_index = VectorStoreIndex.from_vector_store(vector_store)
        return _docs_index


__all__ = [
    "get_qdrant_client",
    "close_qdrant_client",
    "get_docs_index",
]
