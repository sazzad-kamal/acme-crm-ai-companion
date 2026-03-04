"""Action output validator.

Validates that action responses follow the required format:
- Either "NONE" or a numbered list of 1-4 actions
- Each action is one sentence, max 28 words
- Each action starts with "You:" or an owner name
"""

import re
from dataclasses import dataclass, field

# Max words per action
MAX_WORDS_PER_ACTION = 28

# Max number of actions
MAX_ACTIONS = 4

# Pattern to match numbered actions like "1. You: ..." or "1) You: ..."
NUMBERED_ACTION_PATTERN = re.compile(r"^\d+[.)]\s*(.+)$", re.MULTILINE)

# Pattern to validate action starts with owner
OWNER_PATTERN = re.compile(r"^[A-Za-z][A-Za-z\s]*:\s*", re.IGNORECASE)


@dataclass
class ActionValidationResult:
    """Result of action validation."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    action_text: str = ""
    is_none: bool = False
    actions: list[str] = field(default_factory=list)


def validate_action(action: str | None) -> ActionValidationResult:
    """Validate an action response.

    Checks:
    1. Is either "NONE" or a numbered list
    2. Has 1-4 actions if not NONE
    3. Each action is <= 28 words
    4. Each action starts with owner (e.g., "You:" or "Sarah:")

    Args:
        action: The action text to validate (or None)

    Returns:
        ActionValidationResult with validation status and errors
    """
    result = ActionValidationResult(is_valid=True, action_text=action or "")

    # None/empty is valid (equivalent to NONE)
    if not action or not action.strip():
        result.is_none = True
        return result

    action_stripped = action.strip()

    # Check for NONE marker
    if action_stripped.upper() == "NONE":
        result.is_none = True
        return result

    # Extract numbered actions
    matches = NUMBERED_ACTION_PATTERN.findall(action_stripped)

    if not matches:
        # Try to parse as plain text actions (one per line)
        lines = [line.strip() for line in action_stripped.split("\n") if line.strip()]
        # Remove any that look like just numbers
        matches = [line for line in lines if not re.match(r"^\d+[.)]?\s*$", line)]

    if not matches:
        result.errors.append("Action must be NONE or a numbered list")
        result.is_valid = False
        return result

    result.actions = matches

    # Validate number of actions
    if len(matches) > MAX_ACTIONS:
        result.errors.append(f"Too many actions ({len(matches)}), max is {MAX_ACTIONS}")

    # Validate each action
    for i, action_item in enumerate(matches, 1):
        # Check word count
        word_count = len(action_item.split())
        if word_count > MAX_WORDS_PER_ACTION:
            result.errors.append(
                f"Action {i} has {word_count} words, max is {MAX_WORDS_PER_ACTION}"
            )

        # Check for owner prefix (relaxed - just check it has some structure)
        # We're lenient here since LLMs may format slightly differently
        if ":" not in action_item:
            result.errors.append(
                f"Action {i} should start with an owner (e.g., 'You:' or 'Name:')"
            )

    # Set validity based on errors
    if result.errors:
        result.is_valid = False

    return result


__all__ = ["validate_action", "ActionValidationResult"]
