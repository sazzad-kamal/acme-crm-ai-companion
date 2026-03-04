"""Answer output validator.

Validates that answer responses follow the required format:
- Must have Answer section with evidence tags [E1], [E2], etc.
- Must have Evidence section defining each tag
- Each evidence tag in Answer must be defined in Evidence
"""

import re
from dataclasses import dataclass, field

# Required sections in the answer
REQUIRED_SECTIONS = ["Answer:", "Evidence:"]

# Pattern to match evidence tags like [E1], [E2], etc. (inline usage)
EVIDENCE_TAG_PATTERN = re.compile(r"\[E(\d+)\]")

# Pattern to match evidence definitions like "- E1:" or "E1:" (in Evidence section)
EVIDENCE_DEF_PATTERN = re.compile(r"[-•*]?\s*E(\d+)\s*:")


@dataclass
class AnswerValidationResult:
    """Result of answer validation."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    answer_text: str = ""
    evidence_tags_used: set[str] = field(default_factory=set)
    evidence_tags_defined: set[str] = field(default_factory=set)


def validate_answer(answer: str) -> AnswerValidationResult:
    """Validate an answer response.

    Checks:
    1. Has required sections (Answer:, Evidence:)
    2. Evidence tags in Answer are defined in Evidence section
    3. Answer is not empty

    Args:
        answer: The answer text to validate

    Returns:
        AnswerValidationResult with validation status and errors
    """
    result = AnswerValidationResult(is_valid=True, answer_text=answer)

    if not answer or not answer.strip():
        result.is_valid = False
        result.errors.append("Answer is empty")
        return result

    answer_lower = answer.lower()

    # Check for required sections
    for section in REQUIRED_SECTIONS:
        if section.lower() not in answer_lower:
            result.errors.append(f"Missing required section: {section}")

    # Extract Answer section content
    answer_section = _extract_section(answer, "Answer:")
    if not answer_section:
        result.errors.append("Answer section is empty")

    # Find evidence tags used in Answer section
    if answer_section:
        result.evidence_tags_used = set(EVIDENCE_TAG_PATTERN.findall(answer_section))

    # Extract Evidence section and find defined tags
    evidence_section = _extract_section(answer, "Evidence:")
    if evidence_section:
        # Look for both [E1] format and "- E1:" definition format
        tags_bracket = set(EVIDENCE_TAG_PATTERN.findall(evidence_section))
        tags_definition = set(EVIDENCE_DEF_PATTERN.findall(evidence_section))
        result.evidence_tags_defined = tags_bracket | tags_definition

    # Check that all used tags are defined
    undefined_tags = result.evidence_tags_used - result.evidence_tags_defined
    if undefined_tags:
        tags_str = ", ".join(f"[E{t}]" for t in sorted(undefined_tags))
        result.errors.append(f"Evidence tags used but not defined: {tags_str}")

    # If there are evidence tags used but no Evidence section
    if result.evidence_tags_used and not evidence_section:
        result.errors.append("Evidence tags used in Answer but Evidence section is missing/empty")

    # Set validity based on errors
    if result.errors:
        result.is_valid = False

    return result


def _extract_section(text: str, section_name: str) -> str:
    """Extract content of a named section from the text.

    Args:
        text: Full text
        section_name: Section header to find (e.g., "Answer:")

    Returns:
        Section content or empty string if not found
    """
    # Known section markers
    section_markers = [
        "Answer:",
        "Evidence:",
        "Data not available:",
        "Clarifying question:",
    ]

    text_lower = text.lower()
    section_lower = section_name.lower()

    start_idx = text_lower.find(section_lower)
    if start_idx == -1:
        return ""

    # Move past the section header
    start_idx += len(section_name)

    # Find the next section header
    end_idx = len(text)
    for marker in section_markers:
        marker_lower = marker.lower()
        if marker_lower == section_lower:
            continue
        next_section_idx = text_lower.find(marker_lower, start_idx)
        if next_section_idx != -1 and next_section_idx < end_idx:
            end_idx = next_section_idx

    return text[start_idx:end_idx].strip()


__all__ = ["validate_answer", "AnswerValidationResult"]
