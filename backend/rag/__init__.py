# backend.rag - RAG Pipeline for Acme CRM
"""
Retrieval-Augmented Generation (RAG) pipeline for CRM documentation.

Architecture Overview:
======================
This package follows a clean, modular architecture optimized for:
- Clear separation of concerns
- Easy testing and maintenance
- Extensibility for new features

Subpackages:
------------
- config: Centralized configuration (RAGConfig)
- models: Document and chunk models (DocumentChunk, ScoredChunk)
- utils: Utilities (tokenization, chunking, text processing)
- prompts: All LLM prompts centralized for easy maintenance
- ingest: Document ingestion scripts
    - docs: Markdown documentation ingestion
    - private_text: Private CRM text ingestion
    - text_builder: JSONL builder for private texts
- retrieval: Retrieval backends
    - base: RetrievalBackend (hybrid Qdrant + BM25)
    - private: PrivateRetrievalBackend (account-scoped)
    - embedding: Embedding cache utilities
- pipeline: RAG pipelines
    - base: Shared utilities (progress, context building)
    - docs: Documentation RAG (answer_question)
    - account: Account-scoped RAG (answer_account_question)
- eval: Evaluation harnesses
    - models: Evaluation data models
    - judge: LLM-as-judge functions
    - docs_eval: Documentation RAG evaluation
    - account_eval: Account RAG evaluation
- audit: Audit logging

Quick Start:
------------
    # Documentation RAG
    from backend.rag import answer_question, create_backend
    
    backend = create_backend()
    result = answer_question("What is an Opportunity?", backend)
    print(result["answer"])
    
    # Account-scoped RAG
    from backend.rag import answer_account_question
    
    result = answer_account_question(
        "What's the status?",
        company_name="Acme Manufacturing"
    )
    print(result["answer"])
"""

# =============================================================================
# Core Exports
# =============================================================================

# Configuration
from backend.rag.config import get_config, RAGConfig

# Models
from backend.rag.models import DocumentChunk, ScoredChunk

# Utilities
from backend.rag.utils import (
    estimate_tokens,
    preprocess_query,
    recursive_split,
    extract_citations,
    simple_tokenize,
)

# =============================================================================
# Retrieval Exports
# =============================================================================

from backend.rag.retrieval import (
    RetrievalBackend,
    create_backend,
    PrivateRetrievalBackend,
    create_private_backend,
)

# =============================================================================
# Pipeline Exports
# =============================================================================

from backend.rag.pipeline import (
    answer_question,
    answer_account_question,
    PipelineProgress,
    build_context,
)

# =============================================================================
# Audit Exports
# =============================================================================

from backend.rag.audit import AuditEntry, log_audit_entry


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Configuration
    "get_config",
    "RAGConfig",
    # Models
    "DocumentChunk",
    "ScoredChunk",
    # Utilities
    "estimate_tokens",
    "preprocess_query",
    "recursive_split",
    "extract_citations",
    "simple_tokenize",
    # Retrieval
    "RetrievalBackend",
    "create_backend",
    "PrivateRetrievalBackend",
    "create_private_backend",
    # Pipelines
    "answer_question",
    "answer_account_question",
    "PipelineProgress",
    "build_context",
    # Audit
    "AuditEntry",
    "log_audit_entry",
]
