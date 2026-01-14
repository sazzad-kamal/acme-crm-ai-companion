"""
RAG utilities for the fetch node.

Exports:
    tool_entity_rag: Search entity-scoped CRM text
    ingest_private_texts: Ingest CRM private text into Qdrant
"""

from backend.agent.fetch.rag.tools import tool_entity_rag

__all__ = [
    "tool_entity_rag",
]
