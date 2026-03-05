"""Action suggestion node for agent workflow."""

import logging

from backend.agent.action.suggester import call_action_chain
from backend.agent.state import AgentState
from backend.agent.validate.contract import create_action_validator

logger = logging.getLogger(__name__)

# Lazy-initialized contract validator
_action_validator = None


def _get_action_validator():
    """Get or create the action validator (lazy init to avoid circular imports)."""
    global _action_validator
    if _action_validator is None:
        _action_validator = create_action_validator()
    return _action_validator


def action_node(state: AgentState) -> AgentState:
    """Suggest an actionable next step based on the answer."""
    if state.get("error"):
        return {"suggested_action": None}

    logger.info("[Action] Evaluating action suggestion...")

    try:
        raw_action = call_action_chain(
            question=state["question"],
            answer=state["answer"],
        )

        # Apply contract validation: validate → repair → fallback
        validator = _get_action_validator()
        contract_result = validator.enforce(raw_action)

        if contract_result.was_repaired:
            logger.info(f"[Action] Contract: repaired {len(contract_result.errors)} errors")
        elif contract_result.used_fallback:
            logger.info("[Action] Contract: used fallback (NONE)")

        action = contract_result.output

        # Handle NONE case
        if action and action.strip().upper() == "NONE":
            action = None

        if action:
            logger.info("[Action] Suggested action")
        else:
            logger.info("[Action] No action suggested")

        return {"suggested_action": action}

    except Exception as e:
        logger.warning(f"[Action] Failed: {e}")
        return {"suggested_action": None}


__all__ = ["action_node"]
