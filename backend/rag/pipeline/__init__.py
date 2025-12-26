# backend.rag.pipeline - RAG Pipelines
"""
RAG pipeline implementations.

Modules:
- base: Progress tracking (PipelineProgress)
- docs: Documentation RAG pipeline (answer_question)
- account: Account-scoped RAG pipeline (answer_account_question)
- gating: Chunk filtering and gating functions
- context_builder: Context building utilities
- prompts: LLM prompt templates
- company_resolver: Company resolution utilities
"""

from backend.rag.pipeline.base import PipelineProgress
from backend.rag.pipeline.docs import answer_question
from backend.rag.pipeline.account import answer_account_question
from backend.rag.pipeline.gating import (
    apply_lexical_gate,
    apply_per_doc_cap,
    apply_per_type_cap,
)
from backend.rag.pipeline.context_builder import build_context

__all__ = [
    "PipelineProgress",
    "build_context",
    "answer_question",
    "answer_account_question",
    "apply_lexical_gate",
    "apply_per_doc_cap",
    "apply_per_type_cap",
]
