"""Reusable contract validation layer.

Provides a generalized validate → repair → fallback pattern for all LLM outputs.
This ensures every response meets its contract before being returned.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from backend.agent.validate.action import ActionValidationResult
    from backend.agent.validate.answer import AnswerValidationResult
    from backend.agent.validate.followup import FollowupValidationResult

logger = logging.getLogger(__name__)

# Type variables for generic validation
T = TypeVar("T")  # Output type (str, list[str], etc.)
R = TypeVar("R")  # Validation result type


@dataclass
class ContractResult(Generic[T]):
    """Result of contract validation with repair/fallback."""

    output: T
    is_valid: bool
    was_repaired: bool
    used_fallback: bool
    errors: list[str]
    repair_errors: list[str]


class ContractValidator(Generic[T, R]):
    """Generic validate → repair → fallback contract enforcer.

    Usage:
        validator = ContractValidator(
            name="answer",
            validate_fn=validate_answer,
            repair_fn=repair_answer,
            fallback_fn=lambda: "I apologize, I couldn't generate a valid response.",
            get_errors=lambda r: r.errors,
            is_valid=lambda r: r.is_valid,
        )
        result = validator.enforce(raw_output)
    """

    def __init__(
        self,
        name: str,
        validate_fn: Callable[[T], R],
        repair_fn: Callable[[T, list[str]], T],
        fallback_fn: Callable[[], T],
        get_errors: Callable[[R], list[str]],
        is_valid: Callable[[R], bool],
        max_repair_attempts: int = 1,
    ):
        """Initialize the contract validator.

        Args:
            name: Name for logging (e.g., "answer", "action", "followup")
            validate_fn: Function to validate output, returns validation result
            repair_fn: Function to repair output given errors
            fallback_fn: Function to generate safe fallback output
            get_errors: Function to extract errors from validation result
            is_valid: Function to check if validation result is valid
            max_repair_attempts: Max repair attempts before fallback (default: 1)
        """
        self.name = name
        self.validate_fn = validate_fn
        self.repair_fn = repair_fn
        self.fallback_fn = fallback_fn
        self.get_errors = get_errors
        self.is_valid = is_valid
        self.max_repair_attempts = max_repair_attempts

    def enforce(self, output: T) -> ContractResult[T]:
        """Enforce the contract: validate → repair → fallback.

        Args:
            output: The raw LLM output to validate

        Returns:
            ContractResult with the final valid output
        """
        # Step 1: Validate
        validation_result = self.validate_fn(output)

        if self.is_valid(validation_result):
            logger.info(f"[Contract:{self.name}] Valid on first pass")
            return ContractResult(
                output=output,
                is_valid=True,
                was_repaired=False,
                used_fallback=False,
                errors=[],
                repair_errors=[],
            )

        initial_errors = self.get_errors(validation_result)
        logger.info(f"[Contract:{self.name}] Validation failed: {initial_errors}")

        # Step 2: Attempt repair
        current_output = output
        repair_errors: list[str] = []

        for attempt in range(self.max_repair_attempts):
            logger.info(f"[Contract:{self.name}] Repair attempt {attempt + 1}/{self.max_repair_attempts}")

            try:
                repaired = self.repair_fn(current_output, initial_errors)
                repair_validation = self.validate_fn(repaired)

                if self.is_valid(repair_validation):
                    logger.info(f"[Contract:{self.name}] Repair succeeded on attempt {attempt + 1}")
                    return ContractResult(
                        output=repaired,
                        is_valid=True,
                        was_repaired=True,
                        used_fallback=False,
                        errors=initial_errors,
                        repair_errors=[],
                    )

                # Repair didn't fully fix it
                repair_errors = self.get_errors(repair_validation)
                current_output = repaired
                initial_errors = repair_errors
                logger.info(f"[Contract:{self.name}] Repair incomplete, remaining errors: {repair_errors}")

            except Exception as e:
                logger.warning(f"[Contract:{self.name}] Repair failed with exception: {e}")
                repair_errors.append(str(e))

        # Step 3: Fallback
        logger.info(f"[Contract:{self.name}] Using fallback after {self.max_repair_attempts} repair attempts")

        try:
            fallback = self.fallback_fn()
            return ContractResult(
                output=fallback,
                is_valid=True,
                was_repaired=False,
                used_fallback=True,
                errors=initial_errors,
                repair_errors=repair_errors,
            )
        except Exception as e:
            # Even fallback failed - return original with error flag
            logger.error(f"[Contract:{self.name}] Fallback also failed: {e}")
            return ContractResult(
                output=output,
                is_valid=False,
                was_repaired=False,
                used_fallback=False,
                errors=initial_errors,
                repair_errors=repair_errors + [str(e)],
            )


# Pre-configured validators for common outputs


def create_answer_validator(fallback_text: str | None = None) -> ContractValidator[str, "AnswerValidationResult"]:
    """Create a contract validator for answer responses.

    Args:
        fallback_text: Custom fallback text (optional)

    Returns:
        Configured ContractValidator for answers
    """
    from backend.agent.validate.answer import validate_answer
    from backend.agent.validate.repair import repair_answer

    default_fallback = (
        fallback_text
        or "I apologize, but I couldn't generate a properly formatted response. "
        "Please try rephrasing your question."
    )

    return ContractValidator(
        name="answer",
        validate_fn=validate_answer,
        repair_fn=repair_answer,
        fallback_fn=lambda: default_fallback,
        get_errors=lambda r: r.errors,
        is_valid=lambda r: r.is_valid,
    )


def create_action_validator(fallback_text: str | None = None) -> ContractValidator[str | None, "ActionValidationResult"]:
    """Create a contract validator for action responses.

    Args:
        fallback_text: Custom fallback text (optional, default is "NONE")

    Returns:
        Configured ContractValidator for actions
    """
    from backend.agent.validate.action import validate_action
    from backend.agent.validate.repair import repair_action

    # For actions, NONE is a valid fallback
    default_fallback = fallback_text or "NONE"

    def safe_repair(output: str | None, errors: list[str]) -> str:
        return repair_action(output or "", errors)

    return ContractValidator(
        name="action",
        validate_fn=validate_action,
        repair_fn=safe_repair,
        fallback_fn=lambda: default_fallback,
        get_errors=lambda r: r.errors,
        is_valid=lambda r: r.is_valid,
    )


def create_followup_validator(
    fallback_questions: list[str] | None = None,
) -> ContractValidator[list[str], "FollowupValidationResult"]:
    """Create a contract validator for followup suggestions.

    Args:
        fallback_questions: Custom fallback questions (optional)

    Returns:
        Configured ContractValidator for followups
    """
    from backend.agent.validate.followup import validate_followup
    from backend.agent.validate.repair import repair_followup

    default_fallback = fallback_questions or [
        "What else can I help with?",
        "Any other questions?",
        "Need more details?",
    ]

    return ContractValidator(
        name="followup",
        validate_fn=validate_followup,
        repair_fn=repair_followup,
        fallback_fn=lambda: default_fallback.copy(),
        get_errors=lambda r: r.errors,
        is_valid=lambda r: r.is_valid,
    )


__all__ = [
    "ContractValidator",
    "ContractResult",
    "create_answer_validator",
    "create_action_validator",
    "create_followup_validator",
]
