"""
Answer node LLM functions.

Chain creation and invocation for answer generation.
"""

import json
import logging
from functools import lru_cache
from typing import Any

from backend.core.llm import LONG_RESPONSE_MAX_TOKENS, create_openai_chain

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a helpful CRM assistant for Acme CRM Suite.
Your job is to answer questions using ONLY the provided CRM data context.

GROUNDING RULES:
- Use EXACT numbers and dates from context - never say "several", "some", "multiple", "recent"
- When asked "how many", extract the explicit count from the data
- If specific data isn't in the context, just say it's not available - don't over-explain

EXAMPLES:
Good: "Beta Tech has 3 open opportunities totaling $245,000"
Good: "Last activity: call on December 15, 2024 with John Smith"
Good: "Renewal amount is not available in the current data."
Bad: "They have several opportunities" (vague)
Bad: "Amount: I don't have that information; amounts are tracked in..." (over-explaining)

RESPONSE STYLE:
- Lead with the key answer in 1 sentence
- Use bullet points for supporting details
- Be conversational and natural, not robotic
- Keep it SHORT - no padding or filler

FORMATTING:
- Currency: $1,250,000
- Dates: March 31, 2026
- If no data found, acknowledge briefly and offer to help differently"""

_HUMAN_PROMPT = """Answer the user's question using ONLY the provided data below.

User's question: {question}

{conversation_history_section}

=== CRM DATA (SQL Query Results) ===
{sql_results}

{account_context_section}

Please provide a helpful, grounded response following the rules in your system prompt.
If the data is empty or doesn't contain the answer, acknowledge this briefly."""


@lru_cache
def _get_answer_chain() -> Any:
    """Get or create the answer chain (cached singleton).

    Returns the LCEL chain directly so LangGraph's astream_events
    can capture on_chat_model_stream events for token streaming.
    """
    chain = create_openai_chain(
        system_prompt=_SYSTEM_PROMPT,
        human_prompt=_HUMAN_PROMPT,
        max_tokens=LONG_RESPONSE_MAX_TOKENS,
    )
    logger.debug("Created answer chain")
    return chain


def call_answer_chain(
    question: str,
    sql_results: dict[str, Any] | None = None,
    rag_context: str = "",
    conversation_history: str = "",
) -> str:
    """Call the answer chain and return the answer string."""
    result: str = _get_answer_chain().invoke({
        "question": question,
        "conversation_history_section": f"=== RECENT CONVERSATION ===\n{conversation_history}\n" if conversation_history else "",
        "sql_results": json.dumps(sql_results, indent=2, default=str) if sql_results else "(No data retrieved)",
        "account_context_section": f"=== ACCOUNT CONTEXT (RAG) ===\n{rag_context}\n" if rag_context else "",
    })
    return result


__all__ = ["call_answer_chain"]
