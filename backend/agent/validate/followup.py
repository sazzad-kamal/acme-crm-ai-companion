"""Followup output validator.

Validates that followup suggestions follow the required format:
- Exactly 3 questions
- Each question under 10 words
- No "and" combining multiple asks
"""

from dataclasses import dataclass, field

# Required number of follow-up questions
REQUIRED_QUESTIONS = 3

# Max words per question
MAX_WORDS_PER_QUESTION = 10


@dataclass
class FollowupValidationResult:
    """Result of followup validation."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)


def validate_followup(questions: list[str]) -> FollowupValidationResult:
    """Validate followup suggestions.

    Checks:
    1. Exactly 3 questions
    2. Each question is <= 10 words
    3. No question contains "and" (combining asks)

    Args:
        questions: List of followup questions

    Returns:
        FollowupValidationResult with validation status and errors
    """
    result = FollowupValidationResult(is_valid=True, questions=questions)

    if not questions:
        result.errors.append("No followup questions provided")
        result.is_valid = False
        return result

    # Check count
    if len(questions) != REQUIRED_QUESTIONS:
        result.errors.append(
            f"Expected {REQUIRED_QUESTIONS} questions, got {len(questions)}"
        )

    # Validate each question
    for i, question in enumerate(questions, 1):
        if not question or not question.strip():
            result.errors.append(f"Question {i} is empty")
            continue

        # Check word count
        word_count = len(question.split())
        if word_count > MAX_WORDS_PER_QUESTION:
            result.errors.append(
                f"Question {i} has {word_count} words, max is {MAX_WORDS_PER_QUESTION}"
            )

        # Check for "and" that might combine multiple asks
        # Be lenient - only flag obvious cases like " and " with spaces
        question_lower = question.lower()
        if " and " in question_lower:
            # Check if it's combining asks (has question-like words on both sides)
            parts = question_lower.split(" and ")
            if len(parts) >= 2:
                # Only flag if both parts look like questions
                question_words = {"what", "who", "when", "where", "why", "how", "which", "is", "are", "do", "does", "can"}
                first_word = parts[0].split()[0] if parts[0].split() else ""
                if first_word in question_words:
                    result.errors.append(
                        f"Question {i} may combine multiple asks with 'and'"
                    )

    # Set validity based on errors
    if result.errors:
        result.is_valid = False

    return result


__all__ = ["validate_followup", "FollowupValidationResult"]
