"""
Core agent types, configuration, state, and LLM infrastructure.

This module provides foundational types, configuration,
state definitions, and LLM client for the agent system.
"""

from backend.agent.core.config import (
    AgentConfig,
    get_config,
    is_mock_mode,
)
from backend.agent.core.llm import (
    call_llm,
    create_chain,
    get_chat_model,
    load_prompt,
)
from backend.agent.core.state import AgentState, Message, format_conversation_for_prompt

__all__ = [
    # Config
    "AgentConfig",
    "get_config",
    "is_mock_mode",
    # LLM
    "call_llm",
    "create_chain",
    "get_chat_model",
    "load_prompt",
    # State
    "AgentState",
    "Message",
    "format_conversation_for_prompt",
]
