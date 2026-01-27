"""LLM judge for followup suggestion quality."""

from __future__ import annotations

from pydantic import BaseModel, Field

from backend.core.llm import create_openai_chain
from backend.eval.followup.models import (
    SLO_FOLLOWUP_ANSWER_GROUNDING,
    SLO_FOLLOWUP_DIVERSITY,
    SLO_FOLLOWUP_QUESTION_RELEVANCE,
)


class FollowupJudgeResult(BaseModel):
    """Result from the followup judge."""

    question_relevance: float = Field(description="0-1: Are suggestions related to the original question topic?")
    answer_grounding: float = Field(description="0-1: Do suggestions mention specific names from the answer (e.g. 'Enterprise Upgrade', 'Acme Corp')?")
    diversity: float = Field(description="0-1: Do suggestions cover different angles/topics?")
    explanation: str = Field(description="Brief reasoning")


_SYSTEM_PROMPT = """Evaluate follow-up question suggestions for a CRM assistant."""

_HUMAN_PROMPT = """Question: {question}
Answer: {answer}

Suggestions:
{suggestions}"""


def judge_followup_suggestions(
    question: str,
    suggestions: list[str],
    answer: str = "",
) -> tuple[bool, float, float, float, str]:
    """
    Judge followup suggestion quality.

    Returns:
        (passed, question_relevance, answer_grounding, diversity, explanation)

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
    result: FollowupJudgeResult = chain.invoke({
        "question": question,
        "answer": answer,
        "suggestions": formatted_suggestions,
    })
    passed = (
        result.question_relevance >= SLO_FOLLOWUP_QUESTION_RELEVANCE
        and result.answer_grounding >= SLO_FOLLOWUP_ANSWER_GROUNDING
        and result.diversity >= SLO_FOLLOWUP_DIVERSITY
    )
    return (
        passed,
        result.question_relevance,
        result.answer_grounding,
        result.diversity,
        result.explanation,
    )


__all__ = ["FollowupJudgeResult", "judge_followup_suggestions"]
