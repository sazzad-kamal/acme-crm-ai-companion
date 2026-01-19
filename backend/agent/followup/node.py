"""Follow-up suggestion node for agent workflow."""

import logging

from backend.agent.followup.suggester import generate_follow_up_suggestions
from backend.agent.state import AgentState, format_conversation_for_prompt

logger = logging.getLogger(__name__)


def followup_node(state: AgentState) -> AgentState:
    """Generate follow-up suggestions based on current state."""
    logger.info("[Followup] Generating suggestions...")

    sql_results = state.get("sql_results", {})
    company_info = sql_results.get("company_info") or sql_results.get("companies") or []
    company_name = company_info[0].get("name") if company_info else None
    available_data = {k: len(v) if isinstance(v, list) else 1 for k, v in sql_results.items() if v}

    try:
        suggestions = generate_follow_up_suggestions(
            question=state["question"],
            company_name=company_name,
            conversation_history=format_conversation_for_prompt(state.get("messages", [])),
            available_data=available_data,
        )

        # Filter empty suggestions
        if suggestions:
            suggestions = [s for s in suggestions if s and s.strip()]

        logger.info(f"[Followup] Generated {len(suggestions)} suggestions")

        return {"follow_up_suggestions": suggestions}

    except Exception as e:
        logger.warning(f"[Followup] Failed: {e}")
        return {"follow_up_suggestions": []}


__all__ = ["followup_node"]
