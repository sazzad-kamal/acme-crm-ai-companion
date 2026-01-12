"""
LangGraph routing node for agent workflow.

Uses slot-based query planning to generate SQL queries.
"""

import logging

from backend.agent.core.state import AgentState, format_history_for_prompt
from backend.agent.route.query_planner import SlotPlan, get_slot_plan

logger = logging.getLogger(__name__)

# Maps starter question patterns to owner IDs
# Sales Rep = jsmith, CSM = amartin, Manager = None (sees all)
_STARTER_OWNER_MAP = {
    "show my open deals": "jsmith",
    "my open deals": "jsmith",
    "show my deals": "jsmith",
    "show my at-risk renewals": "amartin",
    "at-risk renewals": "amartin",
    "my at-risk renewals": "amartin",
    "show team pipeline": None,
    "team pipeline": None,
    "show all deals": None,
}


def _detect_owner_from_starter(question: str) -> str | None:
    """Detect owner ID from starter question patterns."""
    q = question.lower().strip().rstrip("?")
    for pattern, owner in _STARTER_OWNER_MAP.items():
        if pattern in q:
            return owner
    return None


def route_node(state: AgentState) -> AgentState:
    """
    Route node that generates SQL query plan from user question.

    Sets query_plan in state for downstream nodes.
    """
    question = state["question"]
    logger.info(f"[Route] Processing: {question[:50]}...")

    owner = _detect_owner_from_starter(question)

    try:
        slot_plan = get_slot_plan(
            question=question,
            conversation_history=format_history_for_prompt(state.get("messages", [])),
            owner=owner,
        )

        logger.info(f"[Route] Result: {len(slot_plan.queries)} queries, needs_rag={slot_plan.needs_rag}")

        return {
            "slot_plan": slot_plan,
            "owner": owner,
            "needs_rag": slot_plan.needs_rag,
        }

    except Exception as e:
        logger.error(f"[Route] Failed: {e}")

        return {
            "slot_plan": SlotPlan(queries=[], needs_rag=False),
            "owner": owner,
            "needs_rag": False,
            "error": f"Query planning failed: {e}",
        }
