"""
RAG search tools for the agent layer.

Provides:
- tool_entity_rag: Search entity-scoped CRM text
"""

import logging
import threading

from qdrant_client.models import FieldCondition, Filter, MatchValue

from backend.agent.fetch.rag.client import get_qdrant_client
from backend.agent.fetch.rag.config import (
    EMBEDDING_MODEL,
    HYBRID_SEARCH_ENABLED,
    PRIVATE_COLLECTION,
    RERANKER_ENABLED,
    RERANKER_TOP_K,
    RETRIEVAL_TOP_K,
    SPARSE_EMBEDDING_MODEL,
    SPARSE_TOP_K,
)

logger = logging.getLogger(__name__)

# Thread-safe lazy initialization
_embed_model = None
_vector_index = None
_init_lock = threading.Lock()
_settings_initialized = False


def _ensure_initialized():
    """Initialize embedding model, settings, and vector index (thread-safe)."""
    global _embed_model, _vector_index, _settings_initialized
    if _vector_index is not None:
        return

    with _init_lock:
        if _vector_index is not None:
            return

        from llama_index.core import Settings, VectorStoreIndex
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        from llama_index.vector_stores.qdrant import QdrantVectorStore

        # Initialize embedding model once
        _embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
        Settings.embed_model = _embed_model
        _settings_initialized = True

        # Build vector store config
        client = get_qdrant_client()
        vector_store_kwargs = {
            "client": client,
            "collection_name": PRIVATE_COLLECTION,
        }
        if HYBRID_SEARCH_ENABLED:
            vector_store_kwargs["enable_hybrid"] = True
            vector_store_kwargs["fastembed_sparse_model"] = SPARSE_EMBEDDING_MODEL

        vector_store = QdrantVectorStore(**vector_store_kwargs)  # type: ignore[arg-type]
        _vector_index = VectorStoreIndex.from_vector_store(vector_store)

        logger.debug("Initialized RAG vector index")


def tool_entity_rag(
    question: str,
    filters: dict[str, str],
) -> tuple[str, list[dict]]:
    """
    Search entity-scoped CRM text (notes, attachments).

    Uses over-retrieval + reranking for better precision.
    Filters by ANY provided entity ID (OR logic) to get all related documents.

    Args:
        question: User's question
        filters: Dict of entity IDs to filter by (company_id, contact_id, opportunity_id, activity_id)

    Returns:
        Tuple of (context_text, source_metadata)
    """
    try:
        _ensure_initialized()

        # Build OR filter - match documents with ANY of the provided entity IDs
        should_conditions = []
        for key, value in filters.items():
            if value:
                should_conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )

        qdrant_filter = Filter(should=should_conditions) if should_conditions else None  # type: ignore[arg-type]

        # Configure retriever
        retriever_kwargs = {
            "similarity_top_k": RETRIEVAL_TOP_K,
            "vector_store_kwargs": {"qdrant_filters": qdrant_filter},
        }
        if HYBRID_SEARCH_ENABLED:
            retriever_kwargs["sparse_top_k"] = SPARSE_TOP_K
            retriever_kwargs["vector_store_query_mode"] = "hybrid"

        retriever = _vector_index.as_retriever(**retriever_kwargs)  # type: ignore[union-attr]
        nodes = retriever.retrieve(question)

        # Rerank if enabled and we have more nodes than needed
        if RERANKER_ENABLED and len(nodes) > RERANKER_TOP_K:
            from backend.agent.fetch.rag.reranker import rerank_nodes

            nodes = rerank_nodes(nodes, question)
            logger.info(f"Entity RAG: reranked to {len(nodes)} chunks with filters={filters}")
        else:
            logger.info(f"Entity RAG: retrieved {len(nodes)} chunks with filters={filters}")

        context_parts = []
        sources = []

        for node in nodes:
            context_parts.append(node.text)
            source_type = node.metadata.get("type", "note")
            source_id = node.metadata.get("source_id", node.metadata.get("doc_id", "unknown"))
            label = node.metadata.get("title", source_type.replace("_", " ").title())
            sources.append({"type": source_type, "id": source_id, "label": label})

        context = "\n\n---\n\n".join(context_parts)
        return context, sources

    except Exception as e:
        logger.warning(f"Entity RAG failed: {e}")
        return "", []


__all__ = [
    "tool_entity_rag",
]
