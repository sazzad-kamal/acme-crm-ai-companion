"""Reranker module for improving retrieval precision."""

from __future__ import annotations

import logging
import threading

from llama_index.core.schema import NodeWithScore

logger = logging.getLogger(__name__)

# Thread-safe lazy initialization for reranker
_reranker = None
_reranker_lock = threading.Lock()


def _get_reranker():
    """Lazy-load reranker postprocessor (thread-safe singleton)."""
    global _reranker
    if _reranker is None:
        with _reranker_lock:
            # Double-check after acquiring lock
            if _reranker is None:
                from llama_index.core.postprocessor import SentenceTransformerRerank

                from backend.agent.rag.config import RERANKER_MODEL, RERANKER_TOP_K

                logger.info(f"Loading reranker model: {RERANKER_MODEL}")
                _reranker = SentenceTransformerRerank(
                    model=RERANKER_MODEL,
                    top_n=RERANKER_TOP_K,
                )
    return _reranker


def rerank_nodes(
    nodes: list[NodeWithScore],
    query: str,
    top_k: int | None = None,
) -> list[NodeWithScore]:
    """
    Rerank retrieved nodes using cross-encoder.

    Args:
        nodes: Retrieved nodes from vector search
        query: Original question
        top_k: Number of nodes to return (uses config default if None)

    Returns:
        Top-k nodes sorted by reranker score
    """
    if not nodes:
        return nodes

    from backend.agent.rag.config import RERANKER_TOP_K

    effective_top_k = top_k if top_k is not None else RERANKER_TOP_K

    if len(nodes) <= effective_top_k:
        return nodes  # No need to rerank if already under limit

    reranker = _get_reranker()

    # LlamaIndex's postprocessor handles everything
    from llama_index.core.schema import QueryBundle

    query_bundle = QueryBundle(query_str=query)

    reranked: list[NodeWithScore] = reranker.postprocess_nodes(nodes, query_bundle)

    logger.debug(f"Reranked {len(nodes)} nodes -> {len(reranked)}")
    return reranked


__all__ = [
    "rerank_nodes",
]
