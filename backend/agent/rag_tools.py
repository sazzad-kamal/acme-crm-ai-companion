"""
LlamaIndex-based RAG tools for the agent layer.

Provides two simple vector search tools:
- tool_docs_rag: Search product documentation
- tool_account_rag: Search company-scoped CRM text

Uses Qdrant for vector storage with basic similarity search.
"""

import logging
import threading
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from backend.agent.schemas import Source

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

_BACKEND_ROOT = Path(__file__).parent.parent
QDRANT_PATH = _BACKEND_ROOT / "data" / "qdrant"
DOCS_COLLECTION = "acme_crm_docs"
PRIVATE_COLLECTION = "acme_crm_private"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

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
_private_index = None
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

        from llama_index.core import VectorStoreIndex, Settings
        from llama_index.vector_stores.qdrant import QdrantVectorStore

        Settings.embed_model = _get_embed_model()

        client = get_qdrant_client()
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=DOCS_COLLECTION,
        )
        _docs_index = VectorStoreIndex.from_vector_store(vector_store)
        return _docs_index


def get_private_index():
    """Get or create the private text vector index."""
    global _private_index

    if _private_index is not None:
        return _private_index

    with _index_lock:
        if _private_index is not None:
            return _private_index

        from llama_index.core import VectorStoreIndex, Settings
        from llama_index.vector_stores.qdrant import QdrantVectorStore

        Settings.embed_model = _get_embed_model()

        client = get_qdrant_client()
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=PRIVATE_COLLECTION,
        )
        _private_index = VectorStoreIndex.from_vector_store(vector_store)
        return _private_index


# =============================================================================
# RAG Tools
# =============================================================================

def tool_docs_rag(question: str, top_k: int = 5) -> tuple[str, list[Source]]:
    """
    Search product documentation and return relevant context.

    Args:
        question: User's question
        top_k: Number of chunks to retrieve

    Returns:
        Tuple of (context_text, sources)
    """
    try:
        index = get_docs_index()
        retriever = index.as_retriever(similarity_top_k=top_k)
        nodes = retriever.retrieve(question)

        context_parts = []
        sources = []
        seen_docs = set()

        for node in nodes:
            context_parts.append(node.text)
            doc_id = node.metadata.get("doc_id", node.metadata.get("file_name", "unknown"))
            if doc_id not in seen_docs:
                seen_docs.add(doc_id)
                label = doc_id.replace("_", " ").replace(".md", "").title()
                sources.append(Source(type="doc", id=doc_id, label=label))

        context = "\n\n---\n\n".join(context_parts)
        logger.info(f"Docs RAG: retrieved {len(nodes)} chunks from {len(sources)} docs")
        return context, sources

    except Exception as e:
        logger.warning(f"Docs RAG failed: {e}")
        return "", []


def tool_account_rag(
    question: str,
    company_id: str,
    top_k: int = 5,
) -> tuple[str, list[Source]]:
    """
    Search company-scoped CRM text (notes, attachments).

    Args:
        question: User's question
        company_id: Company ID for filtering
        top_k: Number of chunks to retrieve

    Returns:
        Tuple of (context_text, sources)
    """
    try:
        from llama_index.core import VectorStoreIndex, Settings
        from llama_index.vector_stores.qdrant import QdrantVectorStore

        Settings.embed_model = _get_embed_model()

        client = get_qdrant_client()

        # Create vector store with Qdrant filter for company_id
        qdrant_filter = Filter(
            must=[FieldCondition(key="company_id", match=MatchValue(value=company_id))]
        )

        vector_store = QdrantVectorStore(
            client=client,
            collection_name=PRIVATE_COLLECTION,
        )

        index = VectorStoreIndex.from_vector_store(vector_store)
        retriever = index.as_retriever(
            similarity_top_k=top_k,
            vector_store_kwargs={"qdrant_filters": qdrant_filter},
        )
        nodes = retriever.retrieve(question)

        context_parts = []
        sources = []

        for node in nodes:
            context_parts.append(node.text)
            source_type = node.metadata.get("type", "note")
            source_id = node.metadata.get("source_id", node.metadata.get("doc_id", "unknown"))
            label = node.metadata.get("title", source_type.replace("_", " ").title())
            sources.append(Source(type=source_type, id=source_id, label=label))

        context = "\n\n---\n\n".join(context_parts)
        logger.info(f"Account RAG: retrieved {len(nodes)} chunks for company {company_id}")
        return context, sources

    except Exception as e:
        logger.warning(f"Account RAG failed: {e}")
        return "", []


# =============================================================================
# Collection Existence Check
# =============================================================================

def collections_exist() -> tuple[bool, bool]:
    """
    Check if RAG collections exist in Qdrant.

    Returns:
        Tuple of (docs_exists, private_exists)
    """
    try:
        client = get_qdrant_client()
        docs_exists = client.collection_exists(DOCS_COLLECTION)
        private_exists = client.collection_exists(PRIVATE_COLLECTION)
        return docs_exists, private_exists
    except Exception as e:
        logger.warning(f"Error checking collections: {e}")
        return False, False


__all__ = [
    "tool_docs_rag",
    "tool_account_rag",
    "collections_exist",
    "get_qdrant_client",
    "close_qdrant_client",
    "QDRANT_PATH",
    "DOCS_COLLECTION",
    "PRIVATE_COLLECTION",
    "EMBEDDING_MODEL",
]
