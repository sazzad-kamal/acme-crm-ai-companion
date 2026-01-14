"""Reranker module for improving retrieval precision."""

from __future__ import annotations

import logging
import threading

from llama_index.core.schema import NodeWithScore, QueryBundle

from backend.agent.fetch.rag.config import RERANKER_MODEL, RERANKER_TOP_K

logger = logging.getLogger(__name__)

_reranker = None
_reranker_lock = threading.Lock()


def _get_reranker():
    """Lazy-load reranker (thread-safe singleton)."""
    global _reranker
    if _reranker is None:
        with _reranker_lock:
            if _reranker is None:
                from llama_index.core.postprocessor import SentenceTransformerRerank

                logger.info(f"Loading reranker model: {RERANKER_MODEL}")
                _reranker = SentenceTransformerRerank(model=RERANKER_MODEL, top_n=RERANKER_TOP_K)
    return _reranker


def rerank_nodes(nodes: list[NodeWithScore], query: str) -> list[NodeWithScore]:
    """Rerank nodes using cross-encoder, returning top RERANKER_TOP_K results."""
    if not nodes or len(nodes) <= RERANKER_TOP_K:
        return nodes

    reranked: list[NodeWithScore] = _get_reranker().postprocess_nodes(nodes, QueryBundle(query_str=query))
    logger.debug(f"Reranked {len(nodes)} nodes -> {len(reranked)}")
    return reranked


__all__ = [
    "rerank_nodes",
]
