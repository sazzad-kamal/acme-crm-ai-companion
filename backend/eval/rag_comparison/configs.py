"""Retrieval configuration definitions and query engine factory.

Defines 6 retrieval strategies and a factory function that builds a
LlamaIndex QueryEngine for each configuration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import ResponseMode, get_response_synthesizer
from llama_index.core.retrievers import QueryFusionRetriever

logger = logging.getLogger(__name__)


@dataclass
class RetrievalConfig:
    """A single retrieval strategy configuration."""

    name: str
    retriever_type: Literal["vector", "bm25", "hybrid"]
    top_k: int = 5
    reranker: bool = False
    rerank_top_n: int = 5

    def __str__(self) -> str:
        parts = [f"{self.retriever_type} top_k={self.top_k}"]
        if self.reranker:
            parts.append(f"rerank→{self.rerank_top_n}")
        return f"{self.name} ({', '.join(parts)})"


# The 6 configurations from the comparison plan
DEFAULT_CONFIGS: list[RetrievalConfig] = [
    RetrievalConfig(name="vector_top5", retriever_type="vector", top_k=5),
    RetrievalConfig(name="vector_top10", retriever_type="vector", top_k=10),
    RetrievalConfig(name="bm25_top5", retriever_type="bm25", top_k=5),
    RetrievalConfig(name="hybrid_top5", retriever_type="hybrid", top_k=5),
    RetrievalConfig(name="vector_top10_rerank5", retriever_type="vector", top_k=10, reranker=True, rerank_top_n=5),
    RetrievalConfig(name="hybrid_top10_rerank5", retriever_type="hybrid", top_k=10, reranker=True, rerank_top_n=5),
]


def _get_all_nodes(index: VectorStoreIndex) -> list:
    """Extract all nodes from the index docstore."""
    docstore = index.storage_context.docstore
    return list(docstore.docs.values())


def _build_vector_retriever(index: VectorStoreIndex, top_k: int):
    """Build a standard vector similarity retriever."""
    return index.as_retriever(similarity_top_k=top_k)


def _build_bm25_retriever(index: VectorStoreIndex, top_k: int):
    """Build a BM25 keyword retriever from indexed nodes."""
    from llama_index.retrievers.bm25 import BM25Retriever

    nodes = _get_all_nodes(index)
    return BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=top_k)


def _build_hybrid_retriever(index: VectorStoreIndex, top_k: int):
    """Build a hybrid retriever combining vector + BM25 via reciprocal rank fusion."""
    vector_retriever = _build_vector_retriever(index, top_k)
    bm25_retriever = _build_bm25_retriever(index, top_k)

    return QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        similarity_top_k=top_k,
        num_queries=1,  # Don't generate sub-queries, just fuse
        mode="reciprocal_rerank",
    )


def _build_reranker(top_n: int):
    """Build a SentenceTransformer cross-encoder reranker."""
    from llama_index.core.postprocessor import SentenceTransformerRerank

    return SentenceTransformerRerank(
        model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_n=top_n,
    )


def build_query_engine(config: RetrievalConfig, index: VectorStoreIndex) -> RetrieverQueryEngine:
    """Build a query engine for the given retrieval configuration.

    Args:
        config: Retrieval strategy configuration
        index: The LlamaIndex vector store index

    Returns:
        A configured RetrieverQueryEngine
    """
    logger.info(f"[RAG Compare] Building query engine: {config}")

    # Build retriever based on type
    if config.retriever_type == "vector":
        retriever = _build_vector_retriever(index, config.top_k)
    elif config.retriever_type == "bm25":
        retriever = _build_bm25_retriever(index, config.top_k)
    elif config.retriever_type == "hybrid":
        retriever = _build_hybrid_retriever(index, config.top_k)
    else:
        raise ValueError(f"Unknown retriever type: {config.retriever_type}")

    # Build node postprocessors (reranker)
    node_postprocessors = []
    if config.reranker:
        node_postprocessors.append(_build_reranker(config.rerank_top_n))

    # Build response synthesizer
    synthesizer = get_response_synthesizer(response_mode=ResponseMode.COMPACT)

    return RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=synthesizer,
        node_postprocessors=node_postprocessors,
    )


def get_configs_by_name(names: list[str] | None = None) -> list[RetrievalConfig]:
    """Get configs filtered by name. Returns all if names is None."""
    if names is None:
        return DEFAULT_CONFIGS
    name_set = set(names)
    configs = [c for c in DEFAULT_CONFIGS if c.name in name_set]
    unknown = name_set - {c.name for c in configs}
    if unknown:
        raise ValueError(f"Unknown config names: {unknown}. Available: {[c.name for c in DEFAULT_CONFIGS]}")
    return configs


__all__ = ["RetrievalConfig", "DEFAULT_CONFIGS", "build_query_engine", "get_configs_by_name"]
