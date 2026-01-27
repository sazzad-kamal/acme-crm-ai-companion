"""Action suggestion LLM chain functions."""

import logging
from functools import cache
from typing import Any, Literal

from pydantic import BaseModel, Field

from backend.core.llm import create_openai_chain

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a CRM assistant that classifies questions and optionally suggests next actions.

Classify based on the question and answer:
- lookup: question asks for a single fact and the answer is a plain value with no actionable context
- aggregation: question asks for a summary or comparison and the answer is a list or count with no actionable context
- contextual: question or answer involves rich context that naturally leads to a next step,
  or the question references a time-sensitive or status-critical CRM concept
  mandating a follow-up action (renewal, health status, pipeline stage, closing date, due date, etc)

When suggesting, be specific: reference entities from the answer and specify the action type."""

_HUMAN_PROMPT = """Question: {question}

Answer: {answer}"""

_BLOCKED_TYPES = frozenset({"lookup", "aggregation"})


class ActionSuggestion(BaseModel):
    """Structured output for action suggestion decision."""

    question_type: Literal["lookup", "aggregation", "contextual"] = Field(
        description="Classification of the question type"
    )
    action: str = Field(default="", description="The suggested next action")


@cache
def _get_action_chain() -> Any:
    """Get cached action chain."""
    chain = create_openai_chain(
        system_prompt=_SYSTEM_PROMPT,
        human_prompt=_HUMAN_PROMPT,
        max_tokens=300,
        structured_output=ActionSuggestion,
        streaming=False,
    )
    logger.debug("Created action chain")
    return chain


def call_action_chain(question: str, answer: str) -> str | None:
    """Suggest an action. Returns action string or None."""
    result: ActionSuggestion = _get_action_chain().invoke({
        "question": question,
        "answer": answer,
    })
    action = result.action.strip()

    if result.question_type in _BLOCKED_TYPES:
        logger.debug("Action filtered (type=%s)", result.question_type)
        return None

    if action:
        return action

    logger.debug("Action empty (type=%s)", result.question_type)
    return None


__all__ = ["call_action_chain"]
