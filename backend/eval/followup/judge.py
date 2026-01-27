"""LLM judge for followup suggestion quality."""

from __future__ import annotations

from pydantic import BaseModel, Field

from backend.core.llm import create_openai_chain
from backend.eval.followup.models import SLO_FOLLOWUP_DIVERSITY, SLO_FOLLOWUP_RELEVANCE


class FollowupJudgeResult(BaseModel):
    """Result from the followup judge."""

    relevance: float = Field(description="0-1: Are suggestions relevant to the original question?")
    diversity: float = Field(description="0-1: Do suggestions cover different angles/topics?")
    explanation: str = Field(description="Brief reasoning")


_SYSTEM_PROMPT = """Evaluate follow-up question suggestions for a CRM assistant.

Context: The assistant suggests follow-up questions after answering a CRM query.

Score each dimension 0.0 to 1.0:
1. Relevance: Are the suggestions relevant to the question AND answer? Do they reference specific entities, numbers, or dates from the answer?
2. Diversity: Do suggestions cover different angles or directions?

Consider:
- Follow-ups should be natural next questions a user might ask given the answer
- Good follow-ups reference specifics from the answer (names, dates, amounts, stages)
- At least one suggestion should offer a different direction"""

_HUMAN_PROMPT = """Original Question: {question}

{answer_section}Generated Follow-up Suggestions:
{suggestions}"""


def judge_followup_suggestions(
    question: str,
    suggestions: list[str],
    answer: str = "",
) -> tuple[bool, float, float, str]:
    """
    Judge followup suggestion quality.

    Returns:
        (passed, relevance, diversity, explanation)

    Raises:
        Exception: If the LLM chain fails (caller should handle).
    """
    chain = create_openai_chain(
        system_prompt=_SYSTEM_PROMPT,
        human_prompt=_HUMAN_PROMPT,
        max_tokens=512,
        structured_output=FollowupJudgeResult,
        streaming=False,
    )
    formatted_suggestions = "\n".join(f"- {s}" for s in suggestions)
    answer_section = f"Answer: {answer}\n\n" if answer else ""
    result: FollowupJudgeResult = chain.invoke({
        "question": question,
        "answer_section": answer_section,
        "suggestions": formatted_suggestions,
    })
    passed = (
        result.relevance >= SLO_FOLLOWUP_RELEVANCE
        and result.diversity >= SLO_FOLLOWUP_DIVERSITY
    )
    return (
        passed,
        result.relevance,
        result.diversity,
        result.explanation,
    )


__all__ = ["FollowupJudgeResult", "judge_followup_suggestions"]
