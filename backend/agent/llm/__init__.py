"""
LLM interaction module.

Contains shared helper functions and the base system prompt.

Node-specific code has moved to vertical slices:
- route/: Router logic and prompts
- answer/: Answer templates
- followup/: Follow-up templates
"""

from backend.agent.llm.helpers import (
    call_docs_rag,
    call_account_rag,
    generate_follow_up_suggestions,
    call_answer_chain,
    call_not_found_chain,
    FollowUpSuggestions,
)
from backend.agent.llm.prompts import AGENT_SYSTEM_PROMPT

__all__ = [
    # Helpers
    "call_docs_rag",
    "call_account_rag",
    "generate_follow_up_suggestions",
    "call_answer_chain",
    "call_not_found_chain",
    "FollowUpSuggestions",
    # Shared prompt
    "AGENT_SYSTEM_PROMPT",
]
