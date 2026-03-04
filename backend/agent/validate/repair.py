"""Repair chain for fixing invalid outputs.

When a validator fails, this chain attempts to fix the output
by prompting the LLM with the validation errors.
"""

import logging
from typing import Any

from backend.core.llm import create_openai_chain

logger = logging.getLogger(__name__)

_REPAIR_SYSTEM_PROMPT = """You are a response repair assistant. Your job is to fix an LLM response that failed validation.

You will receive:
1. The ORIGINAL RESPONSE that failed validation
2. The VALIDATION ERRORS that need to be fixed

Your task:
- Fix ONLY the issues listed in VALIDATION ERRORS
- Keep as much of the original content as possible
- Return the corrected response

Be concise and preserve the original intent."""

_REPAIR_HUMAN_PROMPT = """=== ORIGINAL RESPONSE ===
{original}

=== VALIDATION ERRORS ===
{errors}

=== INSTRUCTIONS ===
{instructions}

Fix the response to address all validation errors. Return only the corrected response, nothing else."""


def _get_repair_chain() -> Any:
    """Get the repair chain."""
    return create_openai_chain(
        system_prompt=_REPAIR_SYSTEM_PROMPT,
        human_prompt=_REPAIR_HUMAN_PROMPT,
        max_tokens=2000,
    )


def repair_answer(original: str, errors: list[str]) -> str:
    """Repair an invalid answer response.

    Args:
        original: The original answer that failed validation
        errors: List of validation errors

    Returns:
        Repaired answer text
    """
    instructions = """
Fix the answer to include:
1. An "Answer:" section with evidence tags like [E1], [E2] for claims
2. An "Evidence:" section that defines each evidence tag used
3. Optionally "Data not available:" and "Clarifying question:" sections

Example format:
Answer: The deal is in Negotiation stage [E1] with a value of $50,000 [E2].

Evidence:
- E1: opportunities table, row 1, stage="Negotiation"
- E2: opportunities table, row 1, value=50000

Data not available: None

Clarifying question: None
"""
    try:
        chain = _get_repair_chain()
        result: str = chain.invoke({
            "original": original,
            "errors": "\n".join(f"- {e}" for e in errors),
            "instructions": instructions,
        })
        logger.info(f"[Repair] Answer repaired, {len(errors)} errors fixed")
        return result.strip()
    except Exception as e:
        logger.error(f"[Repair] Answer repair failed: {e}")
        return original


def repair_action(original: str, errors: list[str]) -> str:
    """Repair an invalid action response.

    Args:
        original: The original action that failed validation
        errors: List of validation errors

    Returns:
        Repaired action text (or "NONE")
    """
    instructions = """
Fix the action to be either:
1. "NONE" if no actions are appropriate
2. A numbered list of 1-4 actions where:
   - Each action starts with an owner (e.g., "You:" or "Sarah:")
   - Each action is max 28 words
   - Each action is one sentence

Example format:
1. You: Schedule a follow-up call with the procurement team to discuss pricing.
2. Sarah: Send the updated proposal by Friday.
"""
    try:
        chain = _get_repair_chain()
        result: str = chain.invoke({
            "original": original,
            "errors": "\n".join(f"- {e}" for e in errors),
            "instructions": instructions,
        })
        logger.info(f"[Repair] Action repaired, {len(errors)} errors fixed")
        return result.strip()
    except Exception as e:
        logger.error(f"[Repair] Action repair failed: {e}")
        return original


def repair_followup(original: list[str], errors: list[str]) -> list[str]:
    """Repair invalid followup suggestions.

    Args:
        original: The original followup questions that failed validation
        errors: List of validation errors

    Returns:
        Repaired list of followup questions
    """
    instructions = """
Fix the followup questions to:
1. Have exactly 3 questions
2. Each question must be 10 words or less
3. Each question asks ONE thing (no "and" combining asks)
4. Questions should be punchy and direct

Example format:
1. What's the deal timeline?
2. Who are the key stakeholders?
3. Which competitors are involved?

Return exactly 3 questions, numbered 1-3.
"""
    try:
        chain = _get_repair_chain()
        original_text = "\n".join(f"{i+1}. {q}" for i, q in enumerate(original))
        result: str = chain.invoke({
            "original": original_text,
            "errors": "\n".join(f"- {e}" for e in errors),
            "instructions": instructions,
        })

        # Parse the result back into a list
        lines = result.strip().split("\n")
        questions = []
        for line in lines:
            # Remove numbering and clean up
            cleaned = line.strip()
            if cleaned and cleaned[0].isdigit():
                # Remove "1. " or "1) " prefix
                cleaned = cleaned.lstrip("0123456789").lstrip(".)")
                cleaned = cleaned.strip()
            if cleaned:
                questions.append(cleaned)

        # Ensure we have exactly 3
        if len(questions) < 3:
            questions.extend(original[len(questions):3])
        elif len(questions) > 3:
            questions = questions[:3]

        logger.info(f"[Repair] Followup repaired, {len(errors)} errors fixed")
        return questions

    except Exception as e:
        logger.error(f"[Repair] Followup repair failed: {e}")
        return original


__all__ = ["repair_answer", "repair_action", "repair_followup"]
