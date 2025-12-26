# backend.common - Shared Utilities
"""
Shared utilities used by multiple backend modules.

Only llm_client is here because it's shared between agent and rag modules.
Other modules live where they're actually used:
- models: backend.rag.models (DocumentChunk, ScoredChunk)
- formatters: backend.agent.formatters  
- context_builder, prompts, company_resolver: backend.rag.pipeline.*
"""

from backend.common.llm_client import call_llm, call_llm_safe, call_llm_with_metrics

__all__ = [
    "call_llm",
    "call_llm_safe",
    "call_llm_with_metrics",
]
