"""Action suggestion LLM chain functions."""

import logging
from functools import cache
from typing import Any

from backend.core.llm import create_openai_chain

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a CRM assistant that suggests concrete next actions based on a question and answer.

If the question and answer involve rich context that naturally leads to next steps
(e.g. renewals, health status, pipeline stage, closing dates, due dates, follow-ups),
suggest a short numbered action plan (2-4 steps). Reference entities from the answer.
Each step should be specific and actionable (who, what, when).

Format:
1. First step
2. Second step
3. Third step

If no action is appropriate (simple lookups, counts, or aggregations), respond with exactly: NONE"""

_HUMAN_PROMPT = """Question: {question}

Answer: {answer}"""

_NONE_MARKER = "NONE"


@cache
def _get_action_chain() -> Any:
    """Get cached action chain."""
    chain = create_openai_chain(
        system_prompt=_SYSTEM_PROMPT,
        human_prompt=_HUMAN_PROMPT,
        max_tokens=500,
        streaming=True,
    )
    logger.debug("Created action chain")
    return chain


def call_action_chain(question: str, answer: str) -> str | None:
    """Suggest an action. Returns action string or None."""
    result: str = _get_action_chain().invoke({
        "question": question,
        "answer": answer,
    })
    action = result.strip()

    if not action or action.upper() == _NONE_MARKER:
        logger.debug("No action suggested")
        return None

    return action


__all__ = ["call_action_chain"]
