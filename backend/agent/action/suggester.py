"""Action suggestion LLM chain functions."""

import logging
from functools import cache
from typing import Any

from pydantic import BaseModel, Field

from backend.core.llm import SHORT_RESPONSE_MAX_TOKENS, create_openai_chain

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a CRM assistant deciding whether to suggest a next action.

Given a user's question and the assistant's answer, decide whether an actionable next step is appropriate.

SUGGEST an action when:
- The data reveals a follow-up opportunity (at-risk deal, upcoming renewal, stalled pipeline)
- A specific person, company, or deal could benefit from outreach
- There's a clear CRM workflow step (schedule call, send email, create task, update stage, etc.)

DO NOT suggest an action when:
- The question is purely informational (listing data, counts, amounts)
- No clear next step emerges from the data
- The answer indicates data is not available

If suggesting, be specific: reference entities from the answer, specify the action type."""

_HUMAN_PROMPT = """Question: {question}

Answer: {answer}"""


class ActionSuggestion(BaseModel):
    """Structured output for action suggestion decision."""

    should_suggest: bool = Field(description="Whether an action is appropriate for this question/answer")
    action: str = Field(default="", description="The specific action to suggest, if appropriate")


@cache
def _get_action_chain() -> Any:
    """Get cached action chain."""
    chain = create_openai_chain(
        system_prompt=_SYSTEM_PROMPT,
        human_prompt=_HUMAN_PROMPT,
        max_tokens=SHORT_RESPONSE_MAX_TOKENS,
        structured_output=ActionSuggestion,
        streaming=False,
    )
    logger.debug("Created action chain")
    return chain


def call_action_chain(question: str, answer: str) -> str | None:
    """Decide whether to suggest an action. Returns action string or None."""
    result: ActionSuggestion = _get_action_chain().invoke({
        "question": question,
        "answer": answer,
    })
    if result.should_suggest and result.action.strip():
        return result.action.strip()
    return None


__all__ = ["call_action_chain"]
