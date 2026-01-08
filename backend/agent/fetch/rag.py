"""
Fetch node RAG wrappers.

Thin wrappers around RAG tools for error handling.
"""

import logging

from backend.agent.core.state import Source

logger = logging.getLogger(__name__)


def call_account_rag(question: str, company_id: str) -> tuple[str, list[Source]]:
    """Call the account RAG tool with error handling.

    Args:
        question: The user's question
        company_id: The company ID to search

    Returns:
        Tuple of (context string, list of sources)
    """
    try:
        from backend.agent.rag.tools import tool_account_rag
        return tool_account_rag(question, company_id)
    except Exception as e:
        logger.warning(f"Account RAG failed: {e}")
        return "", []


__all__ = [
    "call_account_rag",
]
