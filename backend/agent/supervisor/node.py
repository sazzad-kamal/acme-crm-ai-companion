"""Supervisor node - routes questions based on intent classification."""

import logging
from typing import cast

from backend.agent.state import AgentState, format_conversation_for_prompt
from backend.agent.supervisor.classifier import Intent, classify_intent

logger = logging.getLogger(__name__)


def supervisor_node(state: AgentState) -> AgentState:
    """Classify question intent and route accordingly.

    This node acts as a router/supervisor that decides how to handle
    the user's question:
    - DATA_QUERY: Route to Fetch node for SQL execution
    - CLARIFY: Route directly to Answer (asks for clarification)
    - HELP: Route directly to Answer (responds without SQL)
    """
    question = state["question"]
    history = format_conversation_for_prompt(state.get("messages", []))

    logger.info(f"[Supervisor] Classifying: {question[:50]}...")

    intent = classify_intent(question, history)

    logger.info(f"[Supervisor] Intent: {intent.value}")

    return cast(AgentState, {
        "intent": intent.value,
        "loop_count": 0,  # Initialize loop counter
    })


__all__ = ["supervisor_node"]
