"""Shared data models for answer evaluation."""

from __future__ import annotations

from pydantic import BaseModel


class Question(BaseModel):
    """A question from the evaluation set."""

    text: str
    difficulty: int = 1
    expected_sql: str
    expected_answer: str = ""  # For RAGAS answer_correctness (empty = skip metric)
    expected_action: bool = False  # Whether this question should produce a suggested action


__all__ = ["Question"]
