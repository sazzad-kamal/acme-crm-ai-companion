"""
Shared LLM infrastructure.

Used by both agent/ and eval/ for ChatOpenAI instantiation.
"""

from backend.llm.client import (
    call_llm,
    create_chain,
    get_chat_model,
)

__all__ = [
    "get_chat_model",
    "create_chain",
    "call_llm",
]
